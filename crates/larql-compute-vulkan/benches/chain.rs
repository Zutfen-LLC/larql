//! FFN-chain benchmark: the Vulkan twin of the CUDA GPU-1003 pipeline bench.
//!
//! Times the device-resident FFN side chain (rms_norm -> gate/up matmul ->
//! GEGLU -> down matmul -> residual) against a CPU baseline, at decode
//! (seq=1) and prefill (seq=8) shapes. A CPU baseline column runs in the
//! same Criterion group so HTML reports give a direct comparison.
//!
//! Run: `cargo bench -p larql-compute-vulkan --bench chain`
//!
//! CPU-only hosts: compiles cleanly, skips Vulkan benches, runs CPU baseline
//! only. The on-device benefit is only observable on a Vulkan compute device.

use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::cpu::ops::q4_common::quantize_q4_k;
use larql_compute::{CpuBackend, ComputeBackend, QuantMatVec};
use larql_compute_vulkan::VulkanBackend;

const HIDDEN: usize = 2560;
const INTER: usize = 10240;
const SEQ_LEN: usize = 8;

fn lcg_f32(seed: u32, rows: usize, cols: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..rows * cols)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) & 0xffff) as f32 / 65535.0
        })
        .collect()
}

/// Leak the weights to `'static` so the bench closure can borrow them.
struct ChainWeights {
    gate: &'static [u8],
    up: &'static [u8],
    down: &'static [u8],
    norm_w: &'static [f32],
}

fn build_chain_weights() -> ChainWeights {
    let gate_f = lcg_f32(6, INTER, HIDDEN);
    let up_f = lcg_f32(7, INTER, HIDDEN);
    let down_f = lcg_f32(5, HIDDEN, INTER);
    let gate: &'static [u8] = Box::leak(quantize_q4_k(&gate_f).into_boxed_slice());
    let up: &'static [u8] = Box::leak(quantize_q4_k(&up_f).into_boxed_slice());
    let down: &'static [u8] = Box::leak(quantize_q4_k(&down_f).into_boxed_slice());
    let norm_w: &'static [f32] = Box::leak(vec![1.0f32; HIDDEN].into_boxed_slice());
    ChainWeights { gate, up, down, norm_w }
}

/// CPU reference for the FFN chain (flat-input variant).
#[allow(clippy::too_many_arguments)]
fn cpu_ffn_chain(
    x: &[f32],
    norm_w: &[f32],
    gate: &[u8],
    up: &[u8],
    down: &[u8],
    hidden: usize,
    inter: usize,
    seq: usize,
) -> Vec<f32> {
    let cpu = CpuBackend;
    let mut h_norm = vec![0.0f32; seq * hidden];
    for s in 0..seq {
        let base = s * hidden;
        let sqsum: f64 = x[base..base + hidden].iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = (sqsum / hidden as f64 + 1e-6).sqrt() as f32;
        for j in 0..hidden {
            h_norm[base + j] = x[base + j] / rms * norm_w[j];
        }
    }
    let gate_out = cpu.q4k_matmul(gate, &h_norm, inter, hidden, seq).unwrap();
    let up_out = cpu.q4k_matmul(up, &h_norm, inter, hidden, seq).unwrap();
    let act = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate_out, &up_out);
    let down_out = cpu.q4k_matmul(down, &act, hidden, inter, seq).unwrap();
    let mut out = vec![0.0f32; seq * hidden];
    for i in 0..seq * hidden {
        out[i] = x[i] + down_out[i];
    }
    out
}

fn bench_decode_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain/decode");
    group.sample_size(15);
    group.throughput(Throughput::Elements(HIDDEN as u64));

    let w = build_chain_weights();
    let x: Vec<f32> = lcg_f32(99, HIDDEN, 1);

    // CPU baseline.
    let cpu_id = format!("cpu/token_h{}", HIDDEN);
    group.bench_with_input(BenchmarkId::from_parameter(&cpu_id), &x, |b, x| {
        b.iter(|| {
            cpu_ffn_chain(x, w.norm_w, w.gate, w.up, w.down, HIDDEN, INTER, 1);
        });
    });

    // Vulkan device-resident FFN chain.
    let vulkan = VulkanBackend::new().expect("vulkan backend");
    if vulkan.native_runtime_available() {
        let vulkan_id = format!("vulkan/token_h{}", HIDDEN);
        group.bench_with_input(BenchmarkId::from_parameter(&vulkan_id), &x, |b, x| {
            b.iter(|| {
                vulkan.ffn_chain(x, w.norm_w, w.gate, w.up, w.down, HIDDEN, INTER, 1, 0.0, 1e-6);
            });
        });
    } else {
        eprintln!("[chain] no Vulkan runtime — device bench skipped (CPU-only host)");
    }

    group.finish();
}

fn bench_prefill_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain/prefill");
    group.sample_size(15);
    group.throughput(Throughput::Elements((SEQ_LEN * HIDDEN) as u64));

    let w = build_chain_weights();
    let x: Vec<f32> = lcg_f32(42, SEQ_LEN, HIDDEN);

    let cpu_id = format!("cpu/seq{}_h{}", SEQ_LEN, HIDDEN);
    group.bench_with_input(BenchmarkId::from_parameter(&cpu_id), &x, |b, x| {
        b.iter(|| {
            cpu_ffn_chain(x, w.norm_w, w.gate, w.up, w.down, HIDDEN, INTER, SEQ_LEN);
        });
    });

    let vulkan = VulkanBackend::new().expect("vulkan backend");
    if vulkan.native_runtime_available() {
        let vulkan_id = format!("vulkan/seq{}_h{}", SEQ_LEN, HIDDEN);
        group.bench_with_input(BenchmarkId::from_parameter(&vulkan_id), &x, |b, x| {
            b.iter(|| {
                vulkan.ffn_chain(x, w.norm_w, w.gate, w.up, w.down, HIDDEN, INTER, SEQ_LEN, 0.0, 1e-6);
            });
        });
    }

    group.finish();
}

/// One-shot metrics report (Vulkan vs CPU), mirroring the GPU-1003 report.
fn metrics_report() {
    let vulkan = match VulkanBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[chain] vulkan backend init failed: {e}");
            return;
        }
    };
    if !vulkan.native_runtime_available() {
        eprintln!("[chain] no Vulkan runtime — metrics skipped (CPU-only host)");
        return;
    }
    let w = build_chain_weights();
    let decode_x = lcg_f32(99, HIDDEN, 1);
    let prefill_x = lcg_f32(42, SEQ_LEN, HIDDEN);

    // Decode.
    let t0 = Instant::now();
    let _ = vulkan.ffn_chain(&decode_x, w.norm_w, w.gate, w.up, w.down, HIDDEN, INTER, 1, 0.0, 1e-6);
    let vulkan_decode_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let _ = cpu_ffn_chain(&decode_x, w.norm_w, w.gate, w.up, w.down, HIDDEN, INTER, 1);
    let cpu_decode_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Prefill.
    let t2 = Instant::now();
    let _ = vulkan.ffn_chain(&prefill_x, w.norm_w, w.gate, w.up, w.down, HIDDEN, INTER, SEQ_LEN, 0.0, 1e-6);
    let vulkan_prefill_ms = t2.elapsed().as_secs_f64() * 1000.0;

    let t3 = Instant::now();
    let _ = cpu_ffn_chain(&prefill_x, w.norm_w, w.gate, w.up, w.down, HIDDEN, INTER, SEQ_LEN);
    let cpu_prefill_ms = t3.elapsed().as_secs_f64() * 1000.0;

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  GPU-4003 Vulkan FFN Chain Metrics");
    println!("═══════════════════════════════════════════════════════");
    println!("  decode (seq=1):  vulkan={vulkan_decode_ms:.3}ms  cpu={cpu_decode_ms:.3}ms");
    println!("  prefill (seq={SEQ_LEN}): vulkan={vulkan_prefill_ms:.3}ms  cpu={cpu_prefill_ms:.3}ms");
    println!("  device: {}", vulkan.device_info());
    println!("═══════════════════════════════════════════════════════");
}

fn run_metrics_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(metrics_report);
}

fn bench_with_metrics(c: &mut Criterion) {
    run_metrics_once();
    bench_decode_chain(c);
    bench_prefill_chain(c);
}

criterion_group!(benches, bench_with_metrics);
criterion_main!(benches);
