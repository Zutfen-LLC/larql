//! One-shot calibration of the FFN activation gate.
//!
//! Mirrors Metal's [`calibration`] approach (`calibrate()` /
//! `DEFAULT_FLOP_THRESHOLD` / `MIN_FLOP_FLOOR`): instead of a single fixed
//! `ACTIVATION_NATIVE_MIN_ELEMS` constant that must be conservative for every
//! backend, we time the device-resident FFN chain against the host-only
//! reference at a representative decode shape on the *first* device-resident
//! FFN call, then cache the resulting threshold on the runtime. The hostonly
//! path stays the fallback; only the routing decision changes.
//!
//! The calibration is correctness-neutral: both paths produce numerically
//! equivalent FFN output (parity-tested elsewhere) — we only pick which one
//! runs.
//!
//! Mirrors larql-compute-metal/src/calibration.rs (the model this adapts).

use std::time::Instant;

use crate::backend::{CudaRuntime, RuntimeError};

/// Conservative default before calibration runs (or when the device chain
/// loses the calibration benchmark). This is the historical
/// `ACTIVATION_NATIVE_MIN_ELEMS` value — keeping it as the default preserves
/// the old routing decision as a safe fallback.
pub const DEFAULT_ACTIVATION_MIN_ELEMS: usize = 8192;

/// Absolute floor: never route to the device chain below this. Tiny shapes
/// pay only launch/upload overhead with negligible compute, so the host path
/// is always faster. Mirrors Metal's `MIN_FLOP_FLOOR`.
pub const MIN_ACTIVATION_FLOOR: usize = 1024;

/// Representative decode shape used for the one-shot timing run. A single
/// token (`seq=1`) through a `[inter, hidden]` Q4_K FFN — the exact shape
/// the device chain was built to accelerate. `inter` is chosen at the old
/// default gate (8192): if the device chain wins at the *threshold itself*,
/// then any larger `inter` clears the gate; if it loses here, the device
/// chain isn't beneficial and the gate stays at the conservative default.
const CALIB_INTER: usize = DEFAULT_ACTIVATION_MIN_ELEMS;
/// Llama-3 8B-class hidden size (256-multiple for the Q4_K matvec).
const CALIB_HIDDEN: usize = 4096;

/// Pure decision function: given the median device-chain and host-only
/// timings (microseconds), return the calibrated gate threshold.
///
/// - If the device chain wins (`device_median < host_median`), route
///   aggressively down to `MIN_ACTIVATION_FLOOR`.
/// - Otherwise, keep the conservative `DEFAULT_ACTIVATION_MIN_ELEMS`.
///
/// Extracted from [`calibrate_activation_threshold`] so the routing decision
/// can be unit-tested without a GPU (the benchmark itself needs a runtime).
pub fn decide_gate(device_median: u64, host_median: u64) -> usize {
    if device_median < host_median {
        MIN_ACTIVATION_FLOOR
    } else {
        DEFAULT_ACTIVATION_MIN_ELEMS
    }
}

/// Time the device-resident FFN chain vs the host-only reference at the
/// representative shape and return the gate threshold.
///
/// Returns a value inside `[MIN_ACTIVATION_FLOOR, DEFAULT_ACTIVATION_MIN_ELEMS]`.
/// On any runtime error during the timing run, calibration falls back to the
/// conservative `DEFAULT_ACTIVATION_MIN_ELEMS` (the documented
/// host-only-fallback contract for every native dispatch in this backend).
pub(crate) fn calibrate_activation_threshold(runtime: &CudaRuntime) -> usize {
    // Build a deterministic Q4_K FFN at the representative shape via the
    // shared substrate quantizer (same path the test fixtures use) so the
    // chain and the host reference see identical weights.
    let (gate_q4k, up_q4k, down_q4k) = match synth_q4k_ffn(CALIB_INTER, CALIB_HIDDEN) {
        Ok(t) => t,
        Err(_) => return DEFAULT_ACTIVATION_MIN_ELEMS,
    };

    let h_in: Vec<f32> = vec![0.5f32; CALIB_HIDDEN];

    // Warm the runtime caches (kernels, buffer pool) on both paths so the
    // timed runs reflect steady-state decode, not first-touch cold-start.
    if warm_device(runtime, &gate_q4k, &up_q4k, &down_q4k, &h_in, CALIB_INTER, CALIB_HIDDEN)
        .is_err()
        || warm_host(runtime, &gate_q4k, &up_q4k, &down_q4k, &h_in, CALIB_INTER, CALIB_HIDDEN)
            .is_err()
    {
        return DEFAULT_ACTIVATION_MIN_ELEMS;
    }

    const N: usize = 7;
    let mut device_times: Vec<u64> = Vec::with_capacity(N);
    let mut host_times: Vec<u64> = Vec::with_capacity(N);
    for _ in 0..N {
        let t0 = Instant::now();
        if time_device_chain(
            runtime,
            &gate_q4k,
            &up_q4k,
            &down_q4k,
            &h_in,
            CALIB_INTER,
            CALIB_HIDDEN,
        )
        .is_err()
        {
            return DEFAULT_ACTIVATION_MIN_ELEMS;
        }
        device_times.push(t0.elapsed().as_micros() as u64);

        let t0 = Instant::now();
        if time_host_chain(
            runtime,
            &gate_q4k,
            &up_q4k,
            &down_q4k,
            &h_in,
            CALIB_INTER,
            CALIB_HIDDEN,
        )
        .is_err()
        {
            return DEFAULT_ACTIVATION_MIN_ELEMS;
        }
        host_times.push(t0.elapsed().as_micros() as u64);
    }

    device_times.sort_unstable();
    host_times.sort_unstable();
    decide_gate(device_times[N / 2], host_times[N / 2])
}

/// Time the device-resident FFN chain: upload → gate matvec → up matvec →
/// GEGLU-SiLU activation → down matvec → single dtoh readback. Mirrors
/// `host_ffn_block_device`'s device portion (without the host norm/residual,
/// which are identical on both paths and don't affect the routing decision).
fn time_device_chain(
    runtime: &CudaRuntime,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    h_in: &[f32],
    inter: usize,
    hidden: usize,
) -> Result<(), RuntimeError> {
    use crate::pipeline::matvec_dev_by_fmt;
    use larql_compute::QuantFormat;
    let x_dev = runtime.upload_f32(h_in)?;
    let gate_dev = matvec_dev_by_fmt(runtime, QuantFormat::Q4_K, gate, &x_dev, inter, hidden)?;
    let up_dev = matvec_dev_by_fmt(runtime, QuantFormat::Q4_K, up, &x_dev, inter, hidden)?;
    let act_dev = runtime.launch_geglu_silu_dev(&gate_dev, &up_dev, inter)?;
    let down_dev = matvec_dev_by_fmt(runtime, QuantFormat::Q4_K, down, &act_dev, hidden, inter)?;
    let _out = runtime.sync_dtoh_f32(&down_dev)?;
    Ok(())
}

/// Warm-up wrapper (identical to [`time_device_chain`]; named separately for
/// call-site clarity).
fn warm_device(
    runtime: &CudaRuntime,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    h_in: &[f32],
    inter: usize,
    hidden: usize,
) -> Result<(), RuntimeError> {
    time_device_chain(runtime, gate, up, down, h_in, inter, hidden)
}

/// Time the host-routed FFN chain: three quant matvecs + GEGLU-SiLU, each a
/// separate htod/launch/dtoh round-trip (the per-projection overhead the
/// device chain collapses). This uses the runtime's own matvec launchers so
/// the comparison isolates the *chain collapse* benefit (single upload +
/// chained device kernels + single readback) rather than raw host CPU
/// throughput — i.e. it measures exactly what `host_ffn_block_hostonly`
/// costs vs `host_ffn_block_device`.
fn time_host_chain(
    runtime: &CudaRuntime,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    h_in: &[f32],
    inter: usize,
    hidden: usize,
) -> Result<(), RuntimeError> {
    let gate_vec = runtime.launch_q4k_matvec(gate, h_in, inter, hidden)?;
    let up_vec = runtime.launch_q4k_matvec(up, h_in, inter, hidden)?;
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        // SiLU(gate) * up — the CPU reference (apply_activation_gated).
        act[i] = (gate_vec[i] / (1.0 + (-gate_vec[i]).exp())) * up_vec[i];
    }
    let _down_vec = runtime.launch_q4k_matvec(down, &act, hidden, inter)?;
    Ok(())
}

/// Warm-up wrapper for [`time_host_chain`].
fn warm_host(
    runtime: &CudaRuntime,
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    h_in: &[f32],
    inter: usize,
    hidden: usize,
) -> Result<(), RuntimeError> {
    time_host_chain(runtime, gate, up, down, h_in, inter, hidden)
}

/// Triple of Q4_K-quantized FFN weight rows: `(gate, up, down)` each a
/// `[rows, hidden]`-major byte blob. Used by [`synth_q4k_ffn`].
type Q4kFfnTriple = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Build deterministic Q4_K gate/up/down weight rows for the calibration
/// shape. Returns `(gate, up, down)` each a Q4_K byte blob sized
/// `[inter, hidden]` / `[inter, hidden]` / `[hidden, inter]` respectively
/// (rows-major, matching the matvec `[num_rows, hidden]` contract).
fn synth_q4k_ffn(inter: usize, hidden: usize) -> Result<Q4kFfnTriple, RuntimeError> {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;
    // LCG ramp mirroring the test-fixture builder (deterministic, no RNG dep).
    let lcg = |seed: u32, n: usize| -> Vec<f32> {
        let mut s = seed.wrapping_mul(2654435761);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 8) & 0xffff) as f32 / 65535.0
            })
            .collect()
    };
    let gate = quantize_q4_k(&lcg(1, inter * hidden));
    let up = quantize_q4_k(&lcg(2, inter * hidden));
    let down = quantize_q4_k(&lcg(3, hidden * inter));
    Ok((gate, up, down))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `decide_gate` routes aggressively (down to the floor) when the device
    /// chain wins, and conservatively (default) when it doesn't. This is the
    /// core routing decision, unit-testable without a GPU.
    #[test]
    fn decide_gate_routes_to_floor_when_device_wins() {
        assert_eq!(decide_gate(10, 20), MIN_ACTIVATION_FLOOR);
        assert_eq!(decide_gate(0, 1), MIN_ACTIVATION_FLOOR);
    }

    #[test]
    fn decide_gate_keeps_default_when_host_wins() {
        assert_eq!(decide_gate(20, 10), DEFAULT_ACTIVATION_MIN_ELEMS);
        assert_eq!(decide_gate(1, 0), DEFAULT_ACTIVATION_MIN_ELEMS);
    }

    /// A tie (device median == host median) keeps the conservative default —
    /// we only route to the device chain on a strict win, so the host path
    /// (the parity oracle) stays the tie-breaker.
    #[test]
    fn decide_gate_tie_keeps_default() {
        assert_eq!(decide_gate(10, 10), DEFAULT_ACTIVATION_MIN_ELEMS);
    }

    /// The legal envelope invariant: every return value of the calibration
    /// path (default on error, floor/default via `decide_gate`) sits inside
    /// `[MIN_ACTIVATION_FLOOR, DEFAULT_ACTIVATION_MIN_ELEMS]`.
    #[test]
    fn envelope_is_well_formed() {
        assert!(MIN_ACTIVATION_FLOOR < DEFAULT_ACTIVATION_MIN_ELEMS);
        assert!(MIN_ACTIVATION_FLOOR >= 1);
    }
}
