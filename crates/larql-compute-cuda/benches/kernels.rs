//! Per-kernel criterion benchmarks for the CUDA backend.
//!
//! Mirrors the Metal `quant_matvec` / `matmul` bench structure: each
//! kernel × shape × backend combination shows up as one Criterion sample
//! so HTML reports under `target/criterion/` give a side-by-side CPU-vs-CUDA
//! comparison.
//!
//! Run: `cargo bench -p larql-compute-cuda --bench kernels`
//!
//! CPU baseline always runs; CUDA runs only when a native runtime is
//! available (skip cleanly on a CPU-only host).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_q4_k, quantize_q6_k};
use larql_compute::prelude::*;
use larql_compute::{CpuBackend, QuantFormat};
use larql_compute_cuda::CudaBackend;

// ── shapes ─────────────────────────────────────────────────────────────

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
}

const SHAPES: &[Shape] = &[
    Shape { name: "decode_2560", n: 2_560, k: 2_560 },
    Shape { name: "ffn_10240", n: 10_240, k: 2_560 },
    Shape { name: "lm_head_262144", n: 262_144, k: 2_560 },
];

fn lm_head_n() -> usize {
    if std::env::var_os("CI").is_some() { 32_768 } else { 262_144 }
}

fn synth_inputs(n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut w = Vec::with_capacity(n * k);
    for i in 0..n * k {
        let f = i as f32;
        w.push(((f * 0.0001).sin() + 0.3 * (f * 0.00037).cos()) * 0.05);
    }
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    (w, x)
}

fn cuda_backend() -> CudaBackend {
    CudaBackend::new().expect("cuda backend constructs (scaffold or native)")
}

// ── quant matvec ───────────────────────────────────────────────────────

fn bench_quant_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_matvec");
    group.sample_size(20);

    let cpu = CpuBackend;
    let cuda = cuda_backend();
    let cuda_available = cuda.native_runtime_available();

    for format in [QuantFormat::Q4_K, QuantFormat::Q6_K, QuantFormat::Q4_0] {
        let format_label = match format {
            QuantFormat::Q4_K => "q4_k",
            QuantFormat::Q6_K => "q6_k",
            QuantFormat::Q4_0 => "q4_0",
            _ => continue,
        };

        for shape in SHAPES {
            let n = if shape.name == "lm_head_262144" { lm_head_n() } else { shape.n };
            let (w_f32, x) = synth_inputs(n, shape.k);
            let weights = match format {
                QuantFormat::Q4_K => quantize_q4_k(&w_f32),
                QuantFormat::Q6_K => quantize_q6_k(&w_f32),
                QuantFormat::Q4_0 => quantize_q4_0(&w_f32),
                _ => unreachable!(),
            };
            group.throughput(Throughput::Elements((n * shape.k) as u64));

            let cpu_id = format!("cpu/{}/{}", format_label, shape.name);
            group.bench_with_input(
                BenchmarkId::from_parameter(&cpu_id),
                &(weights.clone(), x.clone()),
                |b, (w, x)| { b.iter(|| cpu.quant_matvec(format, w, x, n, shape.k)); },
            );

            if cuda_available {
                let cuda_id = format!("cuda/{}/{}", format_label, shape.name);
                group.bench_with_input(
                    BenchmarkId::from_parameter(&cuda_id),
                    &(weights.clone(), x.clone()),
                    |b, (w, x)| { b.iter(|| cuda.quant_matvec(format, w, x, n, shape.k)); },
                );
            }
        }
    }
    group.finish();
}

// ── quant matmul (prefill path, seq_len > 1) ───────────────────────────

fn bench_quant_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("quant_matmul");
    group.sample_size(15);

    let cpu = CpuBackend;
    let cuda = cuda_backend();
    let cuda_available = cuda.native_runtime_available();
    let seq_len = 8usize;

    let shape = &SHAPES[1]; // ffn_10240
    let (w_f32, _x_f32) = synth_inputs(shape.n, shape.k);
    let x: Vec<f32> = (0..seq_len * shape.k)
        .map(|i| ((i as f32) * 0.011).cos() * 0.3)
        .collect();

    for (format, label) in [(QuantFormat::Q4_K, "q4_k"), (QuantFormat::Q6_K, "q6_k")] {
        let weights = match format {
            QuantFormat::Q4_K => quantize_q4_k(&w_f32),
            QuantFormat::Q6_K => quantize_q6_k(&w_f32),
            _ => unreachable!(),
        };
        group.throughput(Throughput::Elements((shape.n * shape.k * seq_len) as u64));

        let cpu_id = format!("cpu/{}/{}", label, shape.name);
        group.bench_with_input(
            BenchmarkId::from_parameter(&cpu_id),
            &weights,
            |b, w| {
                b.iter(|| {
                    use larql_compute::backend::QuantMatVec;
                    match format {
                        QuantFormat::Q4_K => { cpu.q4k_matmul(w, &x, shape.n, shape.k, seq_len); }
                        QuantFormat::Q6_K => { cpu.q6k_matmul(w, &x, shape.n, shape.k, seq_len); }
                        _ => {}
                    }
                });
            },
        );

        if cuda_available {
            let cuda_id = format!("cuda/{}/{}", label, shape.name);
            group.bench_with_input(
                BenchmarkId::from_parameter(&cuda_id),
                &weights,
                |b, w| {
                    b.iter(|| {
                        use larql_compute::backend::QuantMatVec;
                        match format {
                            QuantFormat::Q4_K => { cuda.q4k_matmul(w, &x, shape.n, shape.k, seq_len); }
                            QuantFormat::Q6_K => { cuda.q6k_matmul(w, &x, shape.n, shape.k, seq_len); }
                            _ => {}
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

// ── dense GEMV (f32 / f16) ─────────────────────────────────────────────

fn bench_dense_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_gemv");
    group.sample_size(15);

    let cpu = CpuBackend;
    let cuda = cuda_backend();
    let cuda_available = cuda.native_runtime_available();

    let n = lm_head_n();
    let k = SHAPES[0].k; // 2560
    let (w_f32, x) = synth_inputs(n, k);

    // f32 GEMV
    group.throughput(Throughput::Elements((n * k) as u64));
    let cpu_id = format!("cpu/f32/lm_head_{}", n);
    group.bench_with_input(
        BenchmarkId::from_parameter(&cpu_id),
        &w_f32,
        |b, w| {
            b.iter(|| {
                use larql_compute::backend::MatMul;
                let view = ndarray::ArrayView2::from_shape((n, k), w).unwrap();
                cpu.f32_gemv_force(view, &x);
            });
        },
    );

    if cuda_available {
        let cuda_id = format!("cuda/f32/lm_head_{}", n);
        group.bench_with_input(
            BenchmarkId::from_parameter(&cuda_id),
            &w_f32,
            |b, w| {
                b.iter(|| {
                    use larql_compute::backend::MatMul;
                    let view = ndarray::ArrayView2::from_shape((n, k), w).unwrap();
                    cuda.f32_gemv_force(view, &x);
                });
            },
        );
    }

    // f16 GEMV
    let w_f16: Vec<u8> = {
        let mut bytes = Vec::with_capacity(n * k * 2);
        for &v in &w_f32 {
            let bits = half::f16::from_f32(v).to_bits();
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        bytes
    };

    let cpu_id = format!("cpu/f16/lm_head_{}", n);
    group.bench_with_input(
        BenchmarkId::from_parameter(&cpu_id),
        &w_f16,
        |b, w| {
            b.iter(|| {
                use larql_compute::backend::MatMul;
                cpu.f16_gemv_force(w, &x, n, k);
            });
        },
    );

    if cuda_available {
        let cuda_id = format!("cuda/f16/lm_head_{}", n);
        group.bench_with_input(
            BenchmarkId::from_parameter(&cuda_id),
            &w_f16,
            |b, w| {
                b.iter(|| {
                    use larql_compute::backend::MatMul;
                    cuda.f16_gemv_force(w, &x, n, k);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_quant_matvec, bench_quant_matmul, bench_dense_gemv);
criterion_main!(benches);
