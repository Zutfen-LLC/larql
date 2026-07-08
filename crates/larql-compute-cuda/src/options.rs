//! Backend construction options and runtime-tunable native-path thresholds.
//!
//! [`BackendOptions`] is the static backend-construction config (device ordinal
//! and CPU-delegate policy). [`NativeThresholds`] holds the seven CUDA
//! native-path crossover gates: each decides whether a given op is worth a
//! device round-trip (htod + launch + sync + dtoh) or should stay on the host
//! reference.
//!
//! ## Env-tunable native-path gates
//!
//! The seven gates were chosen without hardware. Each is overridable via an
//! env var with an unambiguous `LARQL_CUDA_*` prefix so Phase-A hardware tuning
//! is measurement, not recompile cycles. The built-in constants stay as the
//! defaults; invalid/empty/zero/nonsensical values fall back to the default
//! rather than panicking. The values are resolved **once** at first use
//! ([`native_thresholds`] caches via a `OnceLock`) so the decode/prefill hot
//! paths never call `getenv` per token/layer.
//!
//! | Env var | Default | What it gates |
//! |---|---|---|
//! | `LARQL_CUDA_NORM_NATIVE_MIN_ELEMS` | `8192` | RMS-norm device dispatch |
//! | `LARQL_CUDA_ACTIVATION_NATIVE_MIN_ELEMS` | `8192` | GEGLU/activation dispatch |
//! | `LARQL_CUDA_RESIDUAL_NATIVE_MIN_ELEMS` | `8192` | Residual-add dispatch |
//! | `LARQL_CUDA_ROPE_NATIVE_MIN_ELEMS` | `8192` | RoPE dispatch |
//! | `LARQL_CUDA_DECODE_ATTN_NATIVE_MIN_WORK` | `8192` | Decode-attention work |
//! | `LARQL_CUDA_PREFILL_ATTN_NATIVE_MIN_WORK` | `8192` | Prefill-attention work |
//! | `LARQL_CUDA_GEMV_FLOP_THRESHOLD` | `500_000_000` | Dense f32/f16 GEMV FLOPs |

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct BackendOptions {
    pub allow_cpu_delegate: bool,
    pub device_ordinal: usize,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            allow_cpu_delegate: true,
            device_ordinal: 0,
        }
    }
}

// ── Native-path threshold defaults (the constants the audit found) ──────────
//
// Single source of truth for the seven crossover values. The pipeline gates
// (`pipeline.rs`) and the GEMV flop gate (`trait_impl.rs`) read these via
// [`native_thresholds`]; the values here are also the env-var fallbacks.

/// Minimum element count for a native norm dispatch to be worth the device
/// round-trip (host `rms_norm_eps` is faster below this). See the
/// `native_norm_worthwhile` gate in `pipeline.rs`.
pub const DEFAULT_NORM_NATIVE_MIN_ELEMS: usize = 8192;
/// Minimum element count for a native activation dispatch. Mirrors the norm
/// gate. See `native_activation_worthwhile`.
pub const DEFAULT_ACTIVATION_NATIVE_MIN_ELEMS: usize = 8192;
/// Minimum element count for a native residual add. Mirrors the norm gate.
/// See `native_residual_worthwhile`.
pub const DEFAULT_RESIDUAL_NATIVE_MIN_ELEMS: usize = 8192;
/// Minimum element count for a native RoPE dispatch. Mirrors the norm gate.
/// See `native_rope_worthwhile`.
pub const DEFAULT_ROPE_NATIVE_MIN_ELEMS: usize = 8192;
/// Minimum attention work (`num_q × total_len × head_dim`) for a native
/// decode-attention dispatch. See `native_decode_attention_worthwhile`.
pub const DEFAULT_DECODE_ATTN_NATIVE_MIN_WORK: usize = 8192;
/// Minimum attention work (`seq × num_q × seq × head_dim`) for a native
/// prefill-attention dispatch. See `native_prefill_attention_worthwhile`.
pub const DEFAULT_PREFILL_ATTN_NATIVE_MIN_WORK: usize = 8192;
/// Below this FLOP count (`2*n*k`) a dense f32/f16 GEMV stays on the
/// zero-copy CPU path; the htod + sync + dtoh round-trip costs more than the
/// CPU loop below it. Mirrors Metal's `DEFAULT_FLOP_THRESHOLD`. See the GEMV
/// gate in `trait_impl.rs`.
pub const DEFAULT_GEMV_FLOP_THRESHOLD: usize = 500_000_000;

const ENV_NORM_NATIVE_MIN_ELEMS: &str = "LARQL_CUDA_NORM_NATIVE_MIN_ELEMS";
const ENV_ACTIVATION_NATIVE_MIN_ELEMS: &str = "LARQL_CUDA_ACTIVATION_NATIVE_MIN_ELEMS";
const ENV_RESIDUAL_NATIVE_MIN_ELEMS: &str = "LARQL_CUDA_RESIDUAL_NATIVE_MIN_ELEMS";
const ENV_ROPE_NATIVE_MIN_ELEMS: &str = "LARQL_CUDA_ROPE_NATIVE_MIN_ELEMS";
const ENV_DECODE_ATTN_NATIVE_MIN_WORK: &str = "LARQL_CUDA_DECODE_ATTN_NATIVE_MIN_WORK";
const ENV_PREFILL_ATTN_NATIVE_MIN_WORK: &str = "LARQL_CUDA_PREFILL_ATTN_NATIVE_MIN_WORK";
const ENV_GEMV_FLOP_THRESHOLD: &str = "LARQL_CUDA_GEMV_FLOP_THRESHOLD";

/// Env var (presence-as-truth) that surfaces weight-cache diagnostics through
/// the existing `device_info()` diagnostics surface. Off by default so the
/// backend stays quiet.
pub const ENV_GPU_DIAG: &str = "LARQL_GPU_DIAG";

/// The seven env-tunable native-path crossover gates. Resolved once from the
/// process env at first use (see [`native_thresholds`]); fields are the values
/// the gate predicates compare against.
#[derive(Debug, Clone, Copy)]
pub struct NativeThresholds {
    pub norm_native_min_elems: usize,
    pub activation_native_min_elems: usize,
    pub residual_native_min_elems: usize,
    pub rope_native_min_elems: usize,
    pub decode_attn_native_min_work: usize,
    pub prefill_attn_native_min_work: usize,
    pub gemv_flop_threshold: usize,
}

/// Parse a usize threshold from a raw env value, falling back to `default` on
/// any invalid input. Invalid means: unset (`None`), empty/whitespace, a
/// non-numeric string, numeric overflow, **or zero** — a zero gate would make
/// every op "worth it" (silently disabling the CPU fallback), which is
/// nonsensical as a crossover value, so it is treated as "use the default"
/// rather than honoured. Pure (no env read) so it is unit-tested directly.
pub(crate) fn parse_threshold(raw: Option<&str>, default: usize) -> usize {
    match raw.map(str::trim) {
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n > 0 => n,
            // empty, zero, non-numeric, or overflow → default
            _ => default,
        },
        None => default,
    }
}

impl NativeThresholds {
    /// Build the threshold set from the raw env values. Pure (takes the raw
    /// strings, not the process env) so the per-field mapping is unit-testable
    /// without racy `set_var` calls.
    fn from_raw(values: NativeThresholdEnv<'_>) -> Self {
        Self {
            norm_native_min_elems: parse_threshold(values.norm, DEFAULT_NORM_NATIVE_MIN_ELEMS),
            activation_native_min_elems: parse_threshold(
                values.activation,
                DEFAULT_ACTIVATION_NATIVE_MIN_ELEMS,
            ),
            residual_native_min_elems: parse_threshold(
                values.residual,
                DEFAULT_RESIDUAL_NATIVE_MIN_ELEMS,
            ),
            rope_native_min_elems: parse_threshold(values.rope, DEFAULT_ROPE_NATIVE_MIN_ELEMS),
            decode_attn_native_min_work: parse_threshold(
                values.decode_attn,
                DEFAULT_DECODE_ATTN_NATIVE_MIN_WORK,
            ),
            prefill_attn_native_min_work: parse_threshold(
                values.prefill_attn,
                DEFAULT_PREFILL_ATTN_NATIVE_MIN_WORK,
            ),
            gemv_flop_threshold: parse_threshold(values.gemv_flop, DEFAULT_GEMV_FLOP_THRESHOLD),
        }
    }

    /// Read the seven env vars from the process environment once and build the
    /// threshold set. This is the single env-reading entry point; it is cached
    /// by [`native_thresholds`] so the hot paths never `getenv`.
    fn from_process_env() -> Self {
        Self::from_raw(NativeThresholdEnv {
            norm: std::env::var(ENV_NORM_NATIVE_MIN_ELEMS).ok().as_deref(),
            activation: std::env::var(ENV_ACTIVATION_NATIVE_MIN_ELEMS)
                .ok()
                .as_deref(),
            residual: std::env::var(ENV_RESIDUAL_NATIVE_MIN_ELEMS).ok().as_deref(),
            rope: std::env::var(ENV_ROPE_NATIVE_MIN_ELEMS).ok().as_deref(),
            decode_attn: std::env::var(ENV_DECODE_ATTN_NATIVE_MIN_WORK)
                .ok()
                .as_deref(),
            prefill_attn: std::env::var(ENV_PREFILL_ATTN_NATIVE_MIN_WORK)
                .ok()
                .as_deref(),
            gemv_flop: std::env::var(ENV_GEMV_FLOP_THRESHOLD).ok().as_deref(),
        })
    }
}

/// Raw env-string carrier used by [`NativeThresholds::from_raw`] (test seam).
struct NativeThresholdEnv<'a> {
    norm: Option<&'a str>,
    activation: Option<&'a str>,
    residual: Option<&'a str>,
    rope: Option<&'a str>,
    decode_attn: Option<&'a str>,
    prefill_attn: Option<&'a str>,
    gemv_flop: Option<&'a str>,
}

/// Process-wide native-path thresholds, built from env on first use and
/// cached. The seven gate predicates in the pipeline and the GEMV flop gate
/// read through here, so `getenv` is paid exactly once per process and the
/// values are stable across the decode/prefill hot path.
pub fn native_thresholds() -> &'static NativeThresholds {
    static THRESHOLDS: OnceLock<NativeThresholds> = OnceLock::new();
    THRESHOLDS.get_or_init(NativeThresholds::from_process_env)
}

/// True when weight-cache diagnostics should be surfaced (presence-as-truth).
/// Read through here so the diagnostic surface does one `var_os` per
/// `device_info()` call rather than spelling the literal everywhere.
pub fn gpu_diag_enabled() -> bool {
    std::env::var_os(ENV_GPU_DIAG).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_threshold_honours_valid_positive_values() {
        assert_eq!(parse_threshold(Some("1"), 8192), 1);
        assert_eq!(parse_threshold(Some("4096"), 8192), 4096);
        assert_eq!(
            parse_threshold(Some("  12345  "), 8192),
            12345,
            "trims whitespace"
        );
    }

    #[test]
    fn parse_threshold_falls_back_on_unset() {
        assert_eq!(parse_threshold(None, 8192), 8192);
        assert_eq!(parse_threshold(None, 500_000_000), 500_000_000);
    }

    #[test]
    fn parse_threshold_falls_back_on_empty() {
        assert_eq!(parse_threshold(Some(""), 8192), 8192);
        assert_eq!(parse_threshold(Some("   "), 8192), 8192);
    }

    #[test]
    fn parse_threshold_falls_back_on_zero() {
        // Zero is nonsensical as a crossover (would disable the CPU fallback
        // for every op); default instead of honouring it.
        assert_eq!(parse_threshold(Some("0"), 8192), 8192);
        assert_eq!(parse_threshold(Some("000"), 8192), 8192);
    }

    #[test]
    fn parse_threshold_falls_back_on_garbage() {
        assert_eq!(parse_threshold(Some("not-a-number"), 8192), 8192);
        assert_eq!(parse_threshold(Some("-1"), 8192), 8192);
        assert_eq!(parse_threshold(Some("1.5"), 8192), 8192);
        assert_eq!(parse_threshold(Some("8_192"), 8192), 8192);
    }

    #[test]
    fn parse_threshold_falls_back_on_overflow() {
        // usize overflow on parse → fallback, not panic.
        let huge = format!("{}000", usize::MAX);
        assert_eq!(parse_threshold(Some(&huge), 8192), 8192);
    }

    #[test]
    fn from_raw_uses_defaults_when_all_unset() {
        let t = NativeThresholds::from_raw(NativeThresholdEnv {
            norm: None,
            activation: None,
            residual: None,
            rope: None,
            decode_attn: None,
            prefill_attn: None,
            gemv_flop: None,
        });
        assert_eq!(t.norm_native_min_elems, DEFAULT_NORM_NATIVE_MIN_ELEMS);
        assert_eq!(
            t.activation_native_min_elems,
            DEFAULT_ACTIVATION_NATIVE_MIN_ELEMS
        );
        assert_eq!(
            t.residual_native_min_elems,
            DEFAULT_RESIDUAL_NATIVE_MIN_ELEMS
        );
        assert_eq!(t.rope_native_min_elems, DEFAULT_ROPE_NATIVE_MIN_ELEMS);
        assert_eq!(
            t.decode_attn_native_min_work,
            DEFAULT_DECODE_ATTN_NATIVE_MIN_WORK
        );
        assert_eq!(
            t.prefill_attn_native_min_work,
            DEFAULT_PREFILL_ATTN_NATIVE_MIN_WORK
        );
        assert_eq!(t.gemv_flop_threshold, DEFAULT_GEMV_FLOP_THRESHOLD);
    }

    #[test]
    fn from_raw_applies_each_override_independently() {
        let t = NativeThresholds::from_raw(NativeThresholdEnv {
            norm: Some("1024"),
            activation: Some("2048"),
            residual: Some("100"),
            rope: Some("9999"),
            decode_attn: Some("4"),
            prefill_attn: Some("7"),
            // zero → default (proves the invalid-value fallback applies per
            // field without disabling the others' overrides)
            gemv_flop: Some("0"),
        });
        assert_eq!(t.norm_native_min_elems, 1024);
        assert_eq!(t.activation_native_min_elems, 2048);
        assert_eq!(t.residual_native_min_elems, 100);
        assert_eq!(t.rope_native_min_elems, 9999);
        assert_eq!(t.decode_attn_native_min_work, 4);
        assert_eq!(t.prefill_attn_native_min_work, 7);
        assert_eq!(t.gemv_flop_threshold, DEFAULT_GEMV_FLOP_THRESHOLD);
    }

    #[test]
    fn native_thresholds_defaults_match_audit_constants() {
        // The cached registry (resolved from the real process env at first
        // use) must equal the documented defaults on a host that doesn't set
        // the override vars — pinning the no-hardware behaviour.
        let t = native_thresholds();
        assert_eq!(t.norm_native_min_elems, 8192);
        assert_eq!(t.activation_native_min_elems, 8192);
        assert_eq!(t.residual_native_min_elems, 8192);
        assert_eq!(t.rope_native_min_elems, 8192);
        assert_eq!(t.decode_attn_native_min_work, 8192);
        assert_eq!(t.prefill_attn_native_min_work, 8192);
        assert_eq!(t.gemv_flop_threshold, 500_000_000);
    }

    #[test]
    fn env_var_names_use_unambiguous_cuda_prefix() {
        // The audit requires an unambiguous `LARQL_CUDA_` prefix so these
        // can't be confused with the CPU decode fast-path `LARQL_Q4K_*` flags.
        for name in [
            ENV_NORM_NATIVE_MIN_ELEMS,
            ENV_ACTIVATION_NATIVE_MIN_ELEMS,
            ENV_RESIDUAL_NATIVE_MIN_ELEMS,
            ENV_ROPE_NATIVE_MIN_ELEMS,
            ENV_DECODE_ATTN_NATIVE_MIN_WORK,
            ENV_PREFILL_ATTN_NATIVE_MIN_WORK,
            ENV_GEMV_FLOP_THRESHOLD,
        ] {
            assert!(
                name.starts_with("LARQL_CUDA_"),
                "threshold env var must use the LARQL_CUDA_ prefix: {name}"
            );
        }
    }
}
