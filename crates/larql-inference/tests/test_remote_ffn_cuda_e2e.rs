//! GPU-2004 — end-to-end local-CUDA-attention + remote-CPU-FFN validation.
//!
//! Proves a CUDA client can run attention locally and dispatch the FFN to a
//! remote CPU shard, producing token-identical output to a CPU-only reference.
//! This is the NVIDIA twin of the existing Metal/CPU remote-MoE integration
//! tests — no such run existed for CUDA before GPU-2004.
//!
//! ## Topology
//!
//! ```text
//!   CUDA client (this process)              CPU server (subprocess)
//!   ──────────────────────────              ──────────────────────
//!   CudaBackend                              larql-server --ffn-only
//!     attention (Q4_K, local)                  FFN (Q4_K, local)
//!     norms / RoPE (local)
//!     FFN → HTTP round trip ──────────────►  /v1/walk-ffn
//!     ◄──────────────── residual ──────────  per-layer dequant + matmul
//! ```
//!
//! Dense Q4_K models (Llama-class) use `generate_with_remote_ffn` +
//! `LayerShardedBackend`, NOT `generate_with_remote_moe`. The MoE path
//! requires router weights (`build_router` returns `None` for dense archs),
//! so every layer's FFN callback would contribute zeros. The dense
//! remote-FFN function marks each layer `ffn_is_remote = true` and routes
//! the full FFN through `LayerShardedBackend::forward`, which is what the
//! CudaBackend `decode_token_with_moe` override (GPU-2001) feeds into.
//!
//! ## NOT FOR CI
//!
//! Requires:
//!   - `cargo test --features cuda` (the `cuda` feature compiles the
//!     CudaBackend; without it this file is cfg'd out entirely).
//!   - A CUDA-capable GPU (`nvidia-smi` visible) — gate via `LARQL_REQUIRE_CUDA=1`.
//!   - A release build of `larql-server` at `target/release/larql-server`
//!     (the server runs the CPU FFN; a debug build works but is ~20x slower).
//!   - A Q4_K vindex on disk. Override with `LARQL_TEST_VINDEX=<path>`.
//!
//! Optionally provide an already-running server with
//! `LARQL_REMOTE_FFN_URL=http://127.0.0.1:PORT`; otherwise the test spawns
//! one automatically on a free port.
//!
//! Run manually:
//!
//! ```sh
//! cargo build --release -p larql-server
//! cargo test --features cuda -p larql-inference \
//!   --test test_remote_ffn_cuda_e2e -- --ignored --nocapture
//! ```
//!
//! Skip semantics: returns cleanly (prints a skip note) when CUDA hardware,
//! the release binary, or a Q4_K vindex is absent — same convention as
//! `test_generate_q4k_cpu` and `multimodal_e2e`.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use larql_inference::{
    compute_backend, generate_with_remote_ffn, layer_graph::generate::eos::EosConfig,
    vindex::generate_kquant_cpu, ComputeBackendKind, LayerShardedBackend,
};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

// ─── Prerequisite discovery ─────────────────────────────────────────────

/// Workspace root (parent of the crate-level `target/`).
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Path to the release `larql-server` binary.
fn server_binary() -> PathBuf {
    workspace_root().join("target/release/larql-server")
}

/// Search known locations for a Q4_K vindex. Mirrors `test_generate_q4k_cpu`.
fn find_q4k_vindex() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LARQL_TEST_VINDEX") {
        let path = PathBuf::from(p);
        if path.is_dir() {
            return Some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let candidates = [
        PathBuf::from("output/llama2-7b-q4k.vindex"),
        PathBuf::from("output/mistral-7b-v0.1-q4k.vindex"),
        PathBuf::from("output/gemma3-4b-q4k-v2.vindex"),
        PathBuf::from(&home).join(".cache/larql/local/llama2-7b-q4k.vindex"),
        PathBuf::from(&home).join(".cache/larql/local/mistral-7b-v0.1-q4k.vindex"),
    ];
    for candidate in &candidates {
        if candidate.is_dir() {
            if let Ok(cfg) = load_vindex_config(candidate) {
                if cfg.quant == QuantFormat::Q4K {
                    return Some(candidate.clone());
                }
            }
        }
    }
    None
}

/// Return `false` early if CUDA is not usable. We check `LARQL_REQUIRE_CUDA`
/// (explicit opt-in) and then try to construct the backend — if the driver
/// is missing `compute_backend(Cuda)` returns an error.
fn cuda_available() -> bool {
    if std::env::var("LARQL_REQUIRE_CUDA").as_deref() != Ok("1") {
        return false;
    }
    compute_backend(ComputeBackendKind::Cuda).is_ok()
}

/// Combined gate: print a skip reason and return `false` when any
/// prerequisite is missing.
fn prerequisites_or_skip() -> Option<PathBuf> {
    if !cuda_available() {
        eprintln!(
            "SKIP: CUDA not available. Set LARQL_REQUIRE_CUDA=1 on a CUDA host \
             with `--features cuda`."
        );
        return None;
    }
    let vindex = find_q4k_vindex()?;
    let bin = server_binary();
    if !bin.exists() && std::env::var("LARQL_REMOTE_FFN_URL").is_err() {
        eprintln!(
            "SKIP: release server binary not found at {} — \
             run `cargo build --release -p larql-server` first, or set \
             LARQL_REMOTE_FFN_URL to use an already-running server.",
            bin.display()
        );
        return None;
    }
    Some(vindex)
}

// ─── Server lifecycle ───────────────────────────────────────────────────

/// Spawn a `larql-server --ffn-only` subprocess on the given port.
/// Returns the child handle; the caller is responsible for killing it.
fn spawn_ffn_server(vindex: &Path, port: u16) -> std::io::Result<Child> {
    let bin = server_binary();
    let log_file = std::fs::File::create(format!("/tmp/larql-ffn-server-{port}.log"))
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null());
    Command::new(&bin)
        .arg(vindex)
        .arg("--ffn-only")
        .arg("--port")
        .arg(port.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--no-memcheck")
        .stdout(log_file)
        .stderr(Stdio::inherit())
        .spawn()
}

/// Wait until `/v1/stats` responds (max ~120s for large model loads).
fn wait_for_server(url: &str) -> bool {
    let deadline = Instant::now() + Duration::from_secs(120);
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap_or_else(|_| reqwest::blocking::Client::new());
    while Instant::now() < deadline {
        if client
            .get(format!("{url}/v1/stats"))
            .send()
            .is_ok_and(|r| r.status().is_success())
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    false
}

/// Pick a free TCP port by binding to :0.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8190)
}

// ─── Test ───────────────────────────────────────────────────────────────

#[test]
#[ignore = "GPU-2004: requires CUDA GPU + release larql-server + Q4_K vindex. \
            Run with LARQL_REQUIRE_CUDA=1 --features cuda --ignored."]
fn cuda_attention_remote_cpu_ffn_matches_reference() {
    let Some(vindex_path) = prerequisites_or_skip() else {
        return;
    };
    eprintln!("vindex: {}", vindex_path.display());

    // ── 1. Start (or connect to) the CPU FFN server ────────────────────
    let (ffn_url, _server_guard) = match std::env::var("LARQL_REMOTE_FFN_URL") {
        Ok(url) => (url, None),
        Err(_) => {
            let port = free_port();
            let url = format!("http://127.0.0.1:{port}");
            eprintln!("spawning larql-server --ffn-only on {url} …");
            let child = spawn_ffn_server(&vindex_path, port).expect("spawn server");
            let guard = ServerGuard(child);
            if !wait_for_server(&url) {
                panic!("FFN server did not become healthy at {url} within 120s");
            }
            eprintln!("FFN server ready at {url}");
            (url, Some(guard))
        }
    };

    // ── 2. Load the client vindex (attention Q4_K locally) ─────────────
    let mut cb = SilentLoadCallbacks;
    let weights = load_model_weights_kquant(&vindex_path, &mut cb).expect("load weights");
    let tokenizer = load_vindex_tokenizer(&vindex_path).expect("load tokenizer");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");
    index
        .load_interleaved_kquant(&vindex_path)
        .expect("load FFN Q4K");
    let _ = index.load_lm_head_kquant(&vindex_path);

    let prompt = "The capital of France is";
    // Encode once using a borrow of weights; the arch reference is stable.
    let prompt_ids =
        larql_inference::encode_prompt(&tokenizer, &*weights.arch, prompt).expect("tokenize");
    eprintln!("prompt: {prompt:?} -> {} tokens", prompt_ids.len());

    let max_tokens = 5;

    // ── 3. CPU-only reference ──────────────────────────────────────────
    // generate_kquant_cpu takes &mut ModelWeights (it swaps attn/FFN tensors
    // in/out per layer), and ModelWeights is not Clone (it owns Mmaps for
    // multi-GB packed-byte tensors). Reload from disk for the reference pass
    // so the later CUDA pass starts from a pristine copy.
    let t0 = Instant::now();
    let mut ref_weights =
        load_model_weights_kquant(&vindex_path, &mut SilentLoadCallbacks).expect("reload weights");
    let reference: Vec<String> = generate_kquant_cpu(
        &mut ref_weights,
        &tokenizer,
        &prompt_ids,
        max_tokens,
        &index,
    )
    .into_iter()
    .map(|(t, _)| t)
    .collect();
    eprintln!(
        "CPU reference: {reference:?} ({:.2}s)",
        t0.elapsed().as_secs_f64()
    );

    // ── 4. CUDA client + remote FFN ────────────────────────────────────
    let backend = compute_backend(ComputeBackendKind::Cuda).expect("CUDA backend");
    eprintln!("client backend: {}", backend.name());
    let remote = LayerShardedBackend::connect(&ffn_url, Duration::from_secs(30))
        .expect("connect to FFN server");
    remote.reset_wire_counters();

    let eos = EosConfig::from_vindex_dir(&vindex_path);
    let t0 = Instant::now();
    let result = generate_with_remote_ffn(
        &weights,
        &tokenizer,
        prompt_ids.clone(),
        max_tokens,
        &index,
        &*backend,
        &remote,
        &eos,
    )
    .expect("remote-FFN generate");
    let elapsed = t0.elapsed();
    let remote_tokens: Vec<String> = result.tokens;
    eprintln!(
        "CUDA+remote-FFN: {remote_tokens:?} ({:.2}s)",
        elapsed.as_secs_f64()
    );

    // ── 5. Parity assertion ────────────────────────────────────────────
    assert_eq!(
        reference, remote_tokens,
        "CUDA-local-attention + remote-CPU-FFN diverged from CPU-only reference"
    );
    eprintln!(
        "OK token-identical to CPU reference ({} tokens)",
        remote_tokens.len()
    );

    // ── 6. Record transport metrics ────────────────────────────────────
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    let n_tokens = remote_tokens.len().max(1);
    let calls_per_token = num_layers; // one HTTP call per layer per decode step
    let payload_bytes = (remote.wire_bytes_sent() + remote.wire_bytes_recv()) as usize;
    let payload_bytes_per_token = payload_bytes / n_tokens;
    eprintln!("── remote-FFN transport metrics ──");
    eprintln!("  remote_calls_per_token:   {calls_per_token}  (num_layers={num_layers})");
    eprintln!("  payload_bytes_per_token:  {payload_bytes_per_token}");
    eprintln!(
        "  (expected ~{} B/token = {} layers x {} hidden x f32 x 2 directions)",
        num_layers * hidden * 4 * 2,
        num_layers,
        hidden,
    );

    // Sanity: payload should be non-zero — a zero means no FFN calls fired.
    assert!(
        payload_bytes > 0,
        "zero wire bytes — FFN was not dispatched remotely (decode_token_with_moe override?)"
    );
}

/// RAII guard that kills the server subprocess on drop.
struct ServerGuard(Child);
impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
