//! Pipeline-level benchmark: prefill + decode + transfer-volume metrics.
//!
//! Records the five pipeline metrics the later transfer-reduction work needs
//! as a baseline:
//!   - `prefill_ms`         — time for `prefill_kquant` at seq_len positions
//!   - `decode_ms/token`    — per-token `decode_token` latency (steady state)
//!   - first-token latency  — prefill + first decode (TTFT)
//!   - `htod_bytes/token`   — host→device bytes per decode token
//!   - `dtoh_bytes/token`   — device→host bytes per decode token
//!   - `weight_cache_hit_rate` — cached weight reuse (higher = fewer uploads)
//!
//! A CPU baseline column runs in the same Criterion group so HTML reports
//! give a direct comparison.
//!
//! Run: `cargo bench -p larql-compute-cuda --bench pipeline`
//! Save baseline: `cargo bench -p larql-compute-cuda --bench pipeline -- --save-baseline gpu1003`
//!
//! CPU-only hosts: compiles cleanly, skips CUDA benches, runs CPU baseline only.

use std::time::Instant;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use larql_compute::{CpuBackend, DecodeBackend, QuantFormat, QuantMatVec};
use larql_compute_cuda::CudaBackend;

const HIDDEN: usize = 2560;
const INTER: usize = 10240;
const NUM_Q: usize = 8;
const NUM_KV: usize = 2;
const HEAD_DIM: usize = 256;
const SEQ_LEN: usize = 8;
const DECODE_TOKENS: usize = 16;

fn lcg_f32(seed: u32, rows: usize, cols: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761);
    (0..rows * cols)
        .map(|_| {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 8) & 0xffff) as f32 / 65535.0
        })
        .collect()
}

/// Build a synthetic single-layer Q4_K pipeline layer, leaking weights to
/// `'static` so the returned owned layer can borrow them.
fn build_pipeline_layer() -> larql_compute::FullPipelineLayer<'static> {
    use larql_compute::{Activation, FfnType, NormType, QuantWeight};

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;
    assert!(HIDDEN.is_multiple_of(256) && q_dim.is_multiple_of(256));

    let wq_f32 = lcg_f32(1, q_dim, HIDDEN);
    let wk_f32 = lcg_f32(2, kv_dim, HIDDEN);
    let wv_f32 = lcg_f32(3, kv_dim, HIDDEN);
    let wo_f32 = lcg_f32(4, HIDDEN, q_dim);

    let wq: &'static [u8] = Box::leak(quantize_q4_k(&wq_f32).into_boxed_slice());
    let wk: &'static [u8] = Box::leak(quantize_q4_k(&wk_f32).into_boxed_slice());
    let wv: &'static [u8] = Box::leak(quantize_q4_k(&wv_f32).into_boxed_slice());
    let wo: &'static [u8] = Box::leak(quantize_q4_k(&wo_f32).into_boxed_slice());

    // FFN gate/up: Q4_K; down: Q6_K (inter → hidden).
    let gate_f32 = lcg_f32(6, INTER, HIDDEN);
    let up_f32 = lcg_f32(7, INTER, HIDDEN);
    let down_f32 = lcg_f32(5, HIDDEN, INTER);
    let gate_q4k: &'static [u8] = Box::leak(quantize_q4_k(&gate_f32).into_boxed_slice());
    let up_q4k: &'static [u8] = Box::leak(quantize_q4_k(&up_f32).into_boxed_slice());
    let down_q6k: &'static [u8] = Box::leak(quantize_q6_k(&down_f32).into_boxed_slice());

    let norm_w: &'static [f32] = Box::leak(vec![1.0f32; HIDDEN].into_boxed_slice());
    let qk_norm_w: &'static [f32] = Box::leak(vec![1.0f32; HEAD_DIM].into_boxed_slice());

    let qw = |d: &'static [u8]| QuantWeight { data: d, scales: None, format: QuantFormat::Q4_K };
    let qw_q6k = |d: &'static [u8]| QuantWeight { data: d, scales: None, format: QuantFormat::Q6_K };

    larql_compute::FullPipelineLayer {
        wq: qw(wq),
        wk: qw(wk),
        wv: qw(wv),
        wo: qw(wo),
        gate: qw(gate_q4k),
        up: qw(up_q4k),
        down: qw_q6k(down_q6k),
        input_norm: norm_w,
        post_attn_norm: norm_w,
        pre_ffn_norm: Some(norm_w),
        post_ffn_norm: Some(norm_w),
        input_norm_bias: None,
        post_attn_norm_bias: None,
        norm_offset: 0.0,
        qk_norm_offset: 0.0,
        eps: 1e-6,
        has_post_norms: true,
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        activation: Activation::Silu,
        attn_scale: (1.0 / (HEAD_DIM as f64).sqrt()) as f32,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10000.0,
        rope_position_divisor: 1.0,
        rope_llama3_scaling: None,
        rotary_dim: HEAD_DIM,
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        q_norm_weight: Some(qk_norm_w),
        k_norm_weight: Some(qk_norm_w),
        ffn_up_bias: None,
        ffn_down_bias: None,
        moe: None,
        ffn_is_remote: false,
        moe_combined_output_norm: false,
        moe_outer_post_norm: None,
        ple_input_gate: None,
        ple_projection: None,
        ple_post_norm: None,
        kv_shared_source: None,
        residual_multiplier: 1.0,
    }
}

fn synth_hidden(seed: u32, len: usize) -> Vec<f32> {
    lcg_f32(seed, len, 1)
}

/// Snapshot + print the transfer/cache metrics for one CUDA run.
fn print_pipeline_report(backend: &CudaBackend, label: &str, decode_tokens: usize) {
    let transfer = match backend.transfer_stats() {
        Some(t) => t,
        None => {
            eprintln!("[pipeline] no transfer stats (scaffold path)");
            return;
        }
    };
    let cache = match backend.weight_cache_stats() {
        Some(c) => c,
        None => {
            eprintln!("[pipeline] no cache stats (scaffold path)");
            return;
        }
    };

    let htod_per_token = transfer.htod_bytes / decode_tokens.max(1) as u64;
    let dtoh_per_token = transfer.dtoh_bytes / decode_tokens.max(1) as u64;
    let total_lookups =
        cache.bytes_hits + cache.bytes_misses + cache.float_hits + cache.float_misses;
    let hit_rate = if total_lookups > 0 {
        (cache.bytes_hits + cache.float_hits) as f64 / total_lookups as f64
    } else {
        0.0
    };

    println!();
    println!("═══ GPU-1003 Pipeline Report: {label} ═══");
    println!("  htod_bytes/token:        {htod_per_token}");
    println!("  dtoh_bytes/token:        {dtoh_per_token}");
    println!("  weight_cache_hit_rate:   {hit_rate:.4} ({total_lookups} lookups)");
    println!(
        "  cache: bytes={}h/{}m floats={}h/{}m uploaded={}B",
        cache.bytes_hits, cache.bytes_misses, cache.float_hits, cache.float_misses, cache.bytes_uploaded,
    );
    println!("═════════════════════════════════════════");
}

// ── Criterion benches ──────────────────────────────────────────────────

fn bench_prefill(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/prefill");
    group.sample_size(15);
    group.throughput(Throughput::Elements((SEQ_LEN * HIDDEN) as u64));

    let layer = build_pipeline_layer();
    let layers = vec![layer];
    let x: Vec<f32> = synth_hidden(42, SEQ_LEN * HIDDEN);

    // CPU baseline: one q4k_matvec projection as a proxy.
    let cpu = CpuBackend;
    let cpu_id = format!("cpu/seq{}_h{}", SEQ_LEN, HIDDEN);
    group.bench_with_input(
        BenchmarkId::from_parameter(&cpu_id),
        &x,
        |b, x| {
            b.iter(|| {
                cpu.q4k_matvec(layers[0].wq.data, x, NUM_Q * HEAD_DIM, HIDDEN);
            });
        },
    );

    // CUDA prefill_kquant.
    let cuda = CudaBackend::new().expect("cuda backend");
    if cuda.native_runtime_available() {
        let kv_shapes = vec![(NUM_KV, HEAD_DIM)];
        cuda.preallocate_kv_cache_per_layer(&kv_shapes, SEQ_LEN + DECODE_TOKENS + 4);
        cuda.reset_kv_cache();

        let cuda_id = format!("cuda/seq{}_h{}", SEQ_LEN, HIDDEN);
        group.bench_with_input(
            BenchmarkId::from_parameter(&cuda_id),
            &x,
            |b, x| {
                b.iter(|| {
                    cuda.reset_kv_cache();
                    cuda.prefill_kquant(&layers, x, HIDDEN, INTER, SEQ_LEN, true, 0.0);
                });
            },
        );
    }

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/decode");
    group.sample_size(15);
    group.throughput(Throughput::Elements(HIDDEN as u64));

    let layer = build_pipeline_layer();
    let layers = vec![layer];
    let prefill_x: Vec<f32> = synth_hidden(42, SEQ_LEN * HIDDEN);
    let decode_x: Vec<f32> = synth_hidden(99, HIDDEN);

    // CPU baseline.
    let cpu = CpuBackend;
    let cpu_id = format!("cpu/token_h{}", HIDDEN);
    group.bench_with_input(
        BenchmarkId::from_parameter(&cpu_id),
        &decode_x,
        |b, x| {
            b.iter(|| {
                cpu.q4k_matvec(layers[0].wq.data, x, NUM_Q * HEAD_DIM, HIDDEN);
            });
        },
    );

    // CUDA decode_token (steady state).
    let cuda = CudaBackend::new().expect("cuda backend");
    if cuda.native_runtime_available() {
        let kv_shapes = vec![(NUM_KV, HEAD_DIM)];
        cuda.preallocate_kv_cache_per_layer(&kv_shapes, SEQ_LEN + DECODE_TOKENS + 4);
        cuda.reset_kv_cache();
        let _ = cuda.prefill_kquant(&layers, &prefill_x, HIDDEN, INTER, SEQ_LEN, true, 0.0);

        let cuda_id = format!("cuda/token_h{}", HIDDEN);
        group.bench_with_input(
            BenchmarkId::from_parameter(&cuda_id),
            &decode_x,
            |b, x| {
                b.iter(|| {
                    cuda.decode_token(&layers, x, HIDDEN, INTER);
                });
            },
        );
    }

    group.finish();
}

/// First-token latency (TTFT): prefill + first decode in one measurement.
fn bench_first_token(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline/first_token");
    group.sample_size(10);
    group.throughput(Throughput::Elements((SEQ_LEN * HIDDEN) as u64));

    let layer = build_pipeline_layer();
    let layers = vec![layer];
    let prefill_x: Vec<f32> = synth_hidden(42, SEQ_LEN * HIDDEN);
    let decode_x: Vec<f32> = synth_hidden(99, HIDDEN);

    let cuda = CudaBackend::new().expect("cuda backend");
    if cuda.native_runtime_available() {
        let kv_shapes = vec![(NUM_KV, HEAD_DIM)];
        cuda.preallocate_kv_cache_per_layer(&kv_shapes, SEQ_LEN + DECODE_TOKENS + 4);

        let cuda_id = format!("cuda/ttft_seq{}", SEQ_LEN);
        group.bench_with_input(
            BenchmarkId::from_parameter(&cuda_id),
            &(prefill_x, decode_x),
            |b, (px, dx)| {
                b.iter(|| {
                    cuda.reset_kv_cache();
                    cuda.prefill_kquant(&layers, px, HIDDEN, INTER, SEQ_LEN, true, 0.0);
                    cuda.decode_token(&layers, dx, HIDDEN, INTER);
                });
            },
        );
    }

    group.finish();
}

/// One-shot metrics report (transfer bytes, cache hit rate). Runs at bench
/// startup so the volume metrics appear alongside Criterion's timing report.
fn metrics_report() {
    let cuda = CudaBackend::new().expect("cuda backend");
    if !cuda.native_runtime_available() {
        eprintln!("[pipeline] no CUDA runtime — transfer/cache metrics skipped (CPU-only host)");
        return;
    }

    let layer = build_pipeline_layer();
    let layers = vec![layer];
    let prefill_x: Vec<f32> = synth_hidden(42, SEQ_LEN * HIDDEN);
    let decode_x: Vec<f32> = synth_hidden(99, HIDDEN);

    let kv_shapes = vec![(NUM_KV, HEAD_DIM)];
    cuda.preallocate_kv_cache_per_layer(&kv_shapes, SEQ_LEN + DECODE_TOKENS + 4);
    cuda.reset_kv_cache();

    // Prefill, then decode DECODE_TOKENS tokens.
    let t0 = Instant::now();
    let _ = cuda.prefill_kquant(&layers, &prefill_x, HIDDEN, INTER, SEQ_LEN, true, 0.0);
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let mut prev = decode_x.clone();
    for _ in 0..DECODE_TOKENS {
        if let Some(out) = cuda.decode_token(&layers, &prev, HIDDEN, INTER) {
            prev = out;
        }
    }
    let decode_total_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let decode_ms_per_token = decode_total_ms / DECODE_TOKENS as f64;

    // First-token latency (fresh prefill + first decode).
    cuda.reset_kv_cache();
    let t2 = Instant::now();
    let _ = cuda.prefill_kquant(&layers, &prefill_x, HIDDEN, INTER, SEQ_LEN, true, 0.0);
    let _ = cuda.decode_token(&layers, &decode_x, HIDDEN, INTER);
    let ttft_ms = t2.elapsed().as_secs_f64() * 1000.0;

    println!();
    println!("═══════════════════════════════════════════════════════");
    println!("  GPU-1003 Pipeline Metrics (CUDA, {DECODE_TOKENS} decode tokens)");
    println!("═══════════════════════════════════════════════════════");
    println!("  prefill_ms:          {prefill_ms:.3}  (seq_len={SEQ_LEN})");
    println!("  decode_ms/token:     {decode_ms_per_token:.3}");
    println!("  first-token latency: {ttft_ms:.3} ms (TTFT)");
    print_pipeline_report(&cuda, "post-decode", DECODE_TOKENS);
    println!("═══════════════════════════════════════════════════════");
}

// Run the metrics report once at bench process startup (before Criterion's
// measurement loop). Criterion doesn't support pre-group hooks natively,
// so we use a one-shot `Once` guard invoked from the criterion_group setup.
fn run_metrics_once() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(metrics_report);
}

fn bench_with_metrics(c: &mut Criterion) {
    run_metrics_once();
    bench_prefill(c);
    bench_decode(c);
    bench_first_token(c);
}

criterion_group!(benches, bench_with_metrics);
criterion_main!(benches);
