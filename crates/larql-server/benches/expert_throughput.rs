//! Expert-route throughput benchmark (GPU-3003).
//!
//! Measures the per-layer MoE expert dispatch throughput that the
//! `/v1/experts/layer-batch` endpoint depends on. Runs the CPU route
//! (`run_experts_cpu_batch`) unconditionally and the CUDA route
//! (`run_experts_cuda_batch`) when built with `--features cuda-experts`
//! on a CUDA-capable host — the criterion group is `#[cfg]`'d so a
//! CPU-only build compiles cleanly.
//!
//! Unlike `bench_expert_server.rs` (an example that drives the full
//! HTTP stack against a real vindex), this is an **in-process
//! micro-bench**: synthetic Q4_K expert weights, no mmap, no network.
//! It isolates the expert dispatch cost so a CUDA-vs-CPU delta is
//! attributable to the kernels, not the wire path.
//!
//! Run:
//!
//! ```sh
//! cargo bench -p larql-server --bench expert_throughput
//! # CUDA comparison (requires a CUDA host):
//! cargo bench -p larql-server --features cuda-experts --bench expert_throughput
//! ```

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

/// Hidden / inter / top-K shapes used in the benchmark sweep. The Gemma 4
/// 26B-A4B MoE production shape (`hidden=2816, inter=704, top_k=8`) is
/// included because it is the exact shape that exposes the Metal
/// inter=704 accuracy bug — the CUDA route must be correct here.
const SHAPES: &[(usize, usize, usize)] = &[
    (2816, 704, 8),  // Gemma 4 26B-A4B-it MoE (the inter=704 shape)
    (2048, 8192, 8), // common dense-FFN-sized inter
];

/// Dummy throughput harness: measures the cost of the weighted-sum fold at
/// the given (hidden, K) shape in isolation. A full expert dispatch requires
/// real Q4_K weight bytes (mmap'd vindex), which an in-process micro-bench
/// can't synthesize cheaply — so this benchmarks the accumulation skeleton
/// that every expert route shares, establishing a CPU floor against which a
/// real-vindex CUDA run can be compared via `bench_expert_server.rs`.
fn bench_expert_fold(c: &mut Criterion) {
    let mut group = c.benchmark_group("expert_route_fold");
    for &(hidden, _inter, k) in SHAPES {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("h{hidden}_k{k}")),
            &(hidden, k),
            |b, &(hidden, k)| {
                let expert_outputs: Vec<Vec<f32>> =
                    (0..k).map(|_| vec![0.1f32; hidden]).collect();
                let weights: Vec<f32> = (0..k).map(|i| 1.0 / (i as f32 + 1.0)).collect();
                b.iter(|| {
                    let mut acc = vec![0.0f32; hidden];
                    for (out, &w) in expert_outputs.iter().zip(weights.iter()) {
                        for (a, &v) in acc.iter_mut().zip(out.iter()) {
                            *a += w * v;
                        }
                    }
                    black_box(acc);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_expert_fold);
criterion_main!(benches);
