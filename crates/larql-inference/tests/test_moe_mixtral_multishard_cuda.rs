//! GPU-3005 — first validated large-MoE CUDA + multi-shard run.
//!
//! Mixtral is the only large MoE that extracts cleanly to Q4_K under the
//! PerExpert extraction scheme with standard attention: its expert sub-blocks
//! are plain up/down MLPs gated by a single router projection per layer
//! (`num_experts=8`, `num_experts_per_token=2`). No end-to-end CUDA-client +
//! multi-shard MoE run was validated before this test.
//!
//! ## What this test proves
//!
//! ```text
//!   CUDA client                         2+ CPU expert servers
//!   ───────────                         ──────────────────────
//!   CudaBackend                           larql-server --experts a-b
//!     attention (Q4_K, local)               expert FFN (Q4_K, local)
//!     norms / RoPE (local)
//!     MoE router (local, moe_remote/mod.rs)
//!     expert FFN → POST /v1/expert/batch
//!         ──────────────────────────►     grouped by shard range
//!         ◄──────────── outputs ─────────  weighted sum assembled locally
//! ```
//!
//! 1. A single CPU-only `generate_kquant_cpu` reference run.
//! 2. For each shard count in {1, 2, 4}: spawn N expert servers with
//!    `--experts` ranges, connect a CUDA client via `--moe-shards`, run
//!    `generate_with_remote_moe`, and assert token-for-token parity with
//!    the reference.
//! 3. The client-side MoE router (`moe_remote::router`) is exercised on
//!    every layer of every token — parity proves it selects the same
//!    experts the local path would, since the expert weights are
//!    byte-identical (same vindex, sliced into expert-server shards).
//! 4. tok/s and `remote_calls/token` are measured across the sweep and
//!    the scaling curve is recorded in `docs/moe-sharding.md`.
//!
//! Reuses GPU-2004 harness conventions (skip semantics, server lifecycle,
//! vindex discovery, RAII guards).
//!
//! ## NOT FOR CI
//!
//! Requires (same prereqs as GPU-2004 plus an MoE vindex):
//!   - `cargo test --features cuda`.
//!   - A CUDA GPU — opt in with `LARQL_REQUIRE_CUDA=1`.
//!   - `target/release/larql-server`.
//!   - A Q4_K **MoE** vindex (Mixtral). `find_moe_q4k_vindex` searches the
//!     standard locations; override with `LARQL_TEST_VINDEX=<path>`. The
//!     vindex must carry `router_weights.bin` + per-layer expert weights
//!     (`layers/`), i.e. an `--level all --quant q4k` extract.
//!
//! ```sh
//! cargo build --release -p larql-server
//! LARQL_MOE_BYTES=1 LARQL_REQUIRE_CUDA=1 cargo test --features cuda \
//!   -p larql-inference --test test_moe_mixtral_multishard_cuda \
//!   -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use larql_inference::{
    compute_backend, encode_prompt,
    layer_graph::generate::eos::EosConfig,
    layer_graph::grid::{generate_with_remote_moe, GridGenerateResult},
    vindex::generate_kquant_cpu, ComputeBackendKind, RemoteMoeBackend, ShardConfig,
};
use larql_vindex::{
    load_model_weights_kquant, load_vindex_config, load_vindex_tokenizer, QuantFormat,
    SilentLoadCallbacks, VectorIndex,
};

// ─── Prerequisite discovery (mirrors GPU-2004) ──────────────────────────

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn server_binary() -> PathBuf {
    workspace_root().join("target/release/larql-server")
}

/// Search known locations for a Q4_K **MoE** vindex. An MoE vindex must
/// have `router_weights.bin`; dense vindexes (Llama/Mistral-7B) do not,
/// so they are rejected here even if they happen to be Q4_K.
fn find_moe_q4k_vindex() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LARQL_TEST_VINDEX") {
        let path = PathBuf::from(p);
        if path.is_dir() && is_moe_q4k(&path) {
            return Some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let candidates = [
        PathBuf::from("output/mixtral-8x7b-q4k.vindex"),
        PathBuf::from("output/mixtral-8x7b-instruct-q4k.vindex"),
        PathBuf::from(&home).join(".cache/larql/local/mixtral-8x7b-q4k.vindex"),
        PathBuf::from(&home).join(".cache/larql/local/mixtral-8x7b-instruct-q4k.vindex"),
    ];
    for candidate in &candidates {
        if candidate.is_dir() && is_moe_q4k(candidate) {
            return Some(candidate.clone());
        }
    }
    None
}

/// A directory qualifies as a MoE Q4_K vindex when (a) its `index.json`
/// declares Q4K, (b) it carries a router weights file, and (c) the model
/// architecture reports `num_experts > 1`.
fn is_moe_q4k(path: &Path) -> bool {
    let Ok(cfg) = load_vindex_config(path) else {
        return false;
    };
    if cfg.quant != QuantFormat::Q4K {
        return false;
    }
    if !path.join("router_weights.bin").exists() {
        return false;
    }
    // Architecture check: a weights reload lets us read num_experts without
    // inventing a config parser here. Cheap relative to a model load.
    let mut cb = SilentLoadCallbacks;
    load_model_weights_kquant(path, &mut cb)
        .ok()
        .map(|w| (*w.arch).num_experts() > 1)
        .unwrap_or(false)
}

fn cuda_available() -> bool {
    if std::env::var("LARQL_REQUIRE_CUDA").as_deref() != Ok("1") {
        return false;
    }
    compute_backend(ComputeBackendKind::Cuda).is_ok()
}

fn prerequisites_or_skip() -> Option<PathBuf> {
    if !cuda_available() {
        eprintln!(
            "SKIP: CUDA not available. Set LARQL_REQUIRE_CUDA=1 on a CUDA host \
             with `--features cuda`."
        );
        return None;
    }
    let bin = server_binary();
    if !bin.exists() {
        eprintln!(
            "SKIP: release server binary not found at {} — \
             run `cargo build --release -p larql-server` first.",
            bin.display()
        );
        return None;
    }
    let vindex = find_moe_q4k_vindex().or_else(|| {
        eprintln!(
            "SKIP: no Q4_K MoE vindex found. Extract Mixtral with \
             `larql build mixtral-8x7b --level all --quant q4k`, then set \
             LARQL_TEST_VINDEX=<path>."
        );
        None
    })?;
    Some(vindex)
}

// ─── Server lifecycle (mirrors GPU-2004) ────────────────────────────────

fn spawn_expert_server(vindex: &Path, port: u16, experts: (usize, usize)) -> std::io::Result<Child> {
    let bin = server_binary();
    let log_file = std::fs::File::create(format!("/tmp/larql-expert-server-{port}.log"))
        .map(Stdio::from)
        .unwrap_or_else(|_| Stdio::null());
    let range = format!("{}-{}", experts.0, experts.1);
    eprintln!("  spawning expert server on :{port}  experts {range}");
    Command::new(&bin)
        .arg(vindex)
        .arg("--ffn-only")
        .arg("--experts")
        .arg(range)
        .arg("--port")
        .arg(port.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--no-memcheck")
        .stdout(log_file)
        .stderr(Stdio::inherit())
        .spawn()
}

fn wait_for_server(url: &str) -> bool {
    let deadline = Instant::now() + Duration::from_secs(180);
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

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8190)
}

/// RAII guard that kills every spawned subprocess on drop.
struct ShardCluster(Vec<Child>);
impl Drop for ShardCluster {
    fn drop(&mut self) {
        for child in self.0.iter_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

// ─── Shard topology ─────────────────────────────────────────────────────

/// Partition `n_experts` into `n_shards` contiguous ranges. The last shard
/// absorbs the remainder.
fn expert_ranges(n_experts: usize, n_shards: usize) -> Vec<(usize, usize)> {
    let per = n_experts / n_shards;
    let rem = n_experts % n_shards;
    let mut ranges = Vec::with_capacity(n_shards);
    let mut start = 0;
    for i in 0..n_shards {
        let size = per + if i < rem { 1 } else { 0 };
        let end = start + size - 1;
        ranges.push((start, end));
        start = end + 1;
    }
    ranges
}

// ─── Test ───────────────────────────────────────────────────────────────

#[test]
#[ignore = "GPU-3005: requires CUDA GPU + release larql-server + Q4_K MoE vindex. \
            Run with LARQL_REQUIRE_CUDA=1 LARQL_MOE_BYTES=1 --features cuda --ignored."]
fn mixtral_multishard_cuda_matches_cpu_reference() {
    let Some(vindex_path) = prerequisites_or_skip() else {
        return;
    };
    eprintln!("vindex: {}", vindex_path.display());

    let prompt = "The capital of France is";
    let max_tokens = 8;

    // ── 0. Load once to learn num_experts + build the reference prompt ──
    let mut cb = SilentLoadCallbacks;
    let weights =
        load_model_weights_kquant(&vindex_path, &mut cb).expect("load weights for probe");
    let arch = &*weights.arch;
    let num_experts = arch.num_experts();
    let top_k = arch.num_experts_per_token();
    eprintln!("arch: num_experts={num_experts}  top_k={top_k}");
    assert!(
        num_experts > 1,
        "vindex is not MoE (num_experts={num_experts}) — wrong vindex for GPU-3005"
    );
    let tokenizer = load_vindex_tokenizer(&vindex_path).expect("load tokenizer");
    let prompt_ids =
        encode_prompt(&tokenizer, arch, prompt).expect("tokenize");
    eprintln!("prompt: {prompt:?} -> {} tokens", prompt_ids.len());

    // ── 1. CPU-only reference (single process, in-process experts) ──────
    let mut ref_weights =
        load_model_weights_kquant(&vindex_path, &mut SilentLoadCallbacks).expect("reload weights");
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut SilentLoadCallbacks).expect("load index");
    index.load_attn_kquant(&vindex_path).expect("load attn Q4K");
    index
        .load_interleaved_kquant(&vindex_path)
        .expect("load FFN Q4K");
    let _ = index.load_lm_head_kquant(&vindex_path);

    let t0 = Instant::now();
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

    // ── 2. CUDA backend (reused across all shard counts) ────────────────
    let backend = compute_backend(ComputeBackendKind::Cuda).expect("CUDA backend");
    eprintln!("client backend: {}", backend.name());

    let eos = EosConfig::from_vindex_dir(&vindex_path);

    // Scaling sweep: tok/s and remote_calls/token across 1, 2, 4 shards.
    let mut scaling: Vec<(usize, f64, f64)> = Vec::new(); // (shards, tok/s, calls/token)

    for &n_shards in &[1usize, 2, 4] {
        if n_shards > num_experts {
            eprintln!("-- skipping {n_shards} shards (exceeds num_experts={num_experts})");
            continue;
        }
        eprintln!("━━━ {n_shards} shard(s) ━━━");

        let ranges = expert_ranges(num_experts, n_shards);
        let mut cluster = ShardCluster(Vec::new());
        let mut configs = Vec::with_capacity(n_shards);

        for &range in &ranges {
            let port = free_port();
            let url = format!("http://127.0.0.1:{port}");
            let child = spawn_expert_server(&vindex_path, port, range).expect("spawn server");
            cluster.0.push(child);
            if !wait_for_server(&url) {
                panic!("expert server on :{port} (experts {}-{}) unhealthy after 180s", range.0, range.1);
            }
            configs.push(ShardConfig::new(range.0, range.1, url));
        }

        // Connect a fresh MoE backend for this topology.
        let remote = RemoteMoeBackend::connect(configs).expect("connect MoE backend");

        // Reload weights so the CUDA pass starts pristine (generate mutates
        // per-layer attn/FFN tensors — see GPU-2004 for the same pattern).
        let w = load_model_weights_kquant(&vindex_path, &mut SilentLoadCallbacks).expect("reload");
        let mut idx =
            VectorIndex::load_vindex(&vindex_path, &mut SilentLoadCallbacks).expect("reload index");
        idx.load_attn_kquant(&vindex_path).expect("load attn Q4K");
        idx.load_interleaved_kquant(&vindex_path).expect("load FFN Q4K");
        let _ = idx.load_lm_head_kquant(&vindex_path);

        let calls_before = remote.remote_calls_total();
        let bytes_before = remote.wire_bytes_total();

        let t0 = Instant::now();
        let GridGenerateResult { tokens, .. } = generate_with_remote_moe(
            &w,
            &tokenizer,
            prompt_ids.clone(),
            max_tokens,
            &idx,
            &remote,
            &*backend,
            &eos,
        )
        .expect("remote-MoE generate");
        let elapsed = t0.elapsed();

        let calls = remote.remote_calls_total() - calls_before;
        let bytes = remote.wire_bytes_total() - bytes_before;

        // ── Parity vs the single-process CPU reference ──────────────────
        assert_eq!(
            tokens, reference,
            "{n_shards}-shard MoE run diverged from CPU reference"
        );

        let n_tok = tokens.len().max(1);
        let tok_s = n_tok as f64 / elapsed.as_secs_f64();
        let calls_per_tok = calls as f64 / n_tok as f64;
        eprintln!(
            "  {n_shards} shards: {tokens:?}  ({elapsed:.2?})  tok/s={tok_s:.2}  calls/tok={calls_per_tok:.2}  bytes={bytes}"
        );
        scaling.push((n_shards, tok_s, calls_per_tok));

        // Sanity: the remote path must have actually dispatched. A zero
        // means the MoE router selected no experts (wrong vindex) or the
        // backend silently fell through to local experts.
        assert!(calls > 0, "zero remote calls — MoE not dispatched remotely");

        // `cluster` drops here → all expert servers killed before next sweep.
        drop(remote);
    }

    // ── 3. Scaling summary → stdout (captured into docs/moe-sharding.md) ─
    eprintln!("━━━ scaling curve (tok/s vs shard count) ━━━");
    eprintln!("  shards |  tok/s  | calls/token");
    for (n, ts, cpt) in &scaling {
        eprintln!("  {n:^6} | {ts:^6.2}  | {cpt:.2}");
    }
    // The headline success criterion: at least the 1- and 2-shard runs
    // produced tokens identical to the CPU reference (asserted above), and
    // we measured tok/s + calls/token for each. Record in docs/moe-sharding.md.
}

// Prevent dead-code warnings for helpers that are only referenced inside
// the #[ignore] test body on builds without the cuda feature.
#[allow(dead_code)]
fn _touch() {
    let _ = workspace_root;
    let _ = OnceLock::<()>::new();
}
