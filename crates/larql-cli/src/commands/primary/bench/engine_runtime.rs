//! I/O-bound runtime for the KV-engine bench. Wraps the engine's
//! `prefill` / `decode_step` API. Excluded from the per-file coverage gate
//! because each call hits real weights / Metal pipeline; pure helpers live
//! in `engine.rs`.

use std::time::Instant;

use larql_kv::EngineKind;

use super::args::BenchArgs;
use super::engine::{
    argmax_token, format_engine_label, format_kv_memory_note, summarize_engine_result,
};
use super::row::BenchRow;

/// Run the KV-engine bench path for a single engine kind, with or
/// without a quantised `index`.
///
/// One unified path for both the dense / CPU bench and the Q4K bench.
/// When `index` is `Some`, the engine routes through `prefill_quant` /
/// `decode_step_quant` (and the executor variants under `--via-executor`),
/// which dispatch on whatever quant format the vindex carries —
/// `Q4_K`, `Q6_K`, future formats — without the bench needing to
/// know which one. When `index` is `None`, the dense `prefill` /
/// `decode_step` path runs.
///
/// FFN selection follows the same dual mode. With `--ffn-policy`, the
/// validated policy builds a [`larql_inference::ffn_policy::BoundFfnRouter`]
/// against `&weights` (and `index` when present) and the engine
/// dispatches FFN through it. Without `--ffn-policy`:
///
/// - **Quantised path (`index = Some`)**: default to [`NullFfn`].
///   Legacy engines route FFN internally from the vindex bytes;
///   migrated engines on `*_via_executor` honour whatever FFN the
///   caller supplied (so `NullFfn` would silently skip FFN if a
///   migrated engine didn't substitute its own — they all do).
/// - **Dense path (`index = None`)**: default to [`WeightFfn`] over
///   `&weights` — local dense FFN from the model's `tensors`.
///
/// The harness aborts on any [`EngineError`] (nothing to fall back
/// to in a bench), but surfaces the typed message so production logs
/// can distinguish a dispatch bug (`InvariantViolation`) from a
/// kernel / data failure (`BackendFailure`).
pub(super) fn run_engine(
    weights: &mut larql_inference::ModelWeights,
    index: Option<&larql_vindex::VectorIndex>,
    token_ids: &[u32],
    kv_ref_bytes: usize,
    kind: EngineKind,
    backend: Box<dyn larql_inference::EngineBackend>,
    ffn_policy: Option<&larql_inference::ffn_policy::ValidatedFfnLayerPolicy>,
    args: &BenchArgs,
) -> Result<BenchRow, Box<dyn std::error::Error>> {
    use larql_inference::ffn::{FfnBackend, NullFfn, WeightFfn};
    use larql_inference::forward::hidden_to_raw_logits;

    let is_quant = index.is_some();

    let mut engine = kind.build_with_profiling(backend, args.profile);
    let info = engine.info();
    let label = format_engine_label(&info.name, &info.backend, &info.config, is_quant);

    if args.verbose {
        eprintln!(
            "[bench] {}{}",
            if is_quant { "Q4K engine: " } else { "" },
            info.summary()
        );
    }

    // Compute backend used for `pick_next` on the quant path (lm_head
    // top-k against the vindex) and as the `&dyn ComputeBackend` argument
    // for `prefill_quant`. The Metal factory is required when the user
    // asked for `--backends metal` and the index is Q4K — Engines that
    // probe `backend.supports_quant(Q4_K)` would otherwise see a
    // CpuBackend that advertises support but silently falls back to
    // the slow CPU path on `decode_token`.
    let requested =
        crate::commands::backend::backend_kinds_from_args(&args.backends, args.cpu, args.metal)
            .map_err(|e| format!("--backends: {e}"))?;
    let primary_kind = crate::commands::backend::primary_backend_kind(
        &requested,
        larql_inference::ComputeBackendKind::Cpu,
    );
    let compute_backend: Box<dyn larql_inference::ComputeBackend> =
        larql_inference::compute_backend(primary_kind)
            .map_err(|e| format!("backend selection: {e}"))?;
    let be = compute_backend.as_ref();

    let executor = if args.via_executor {
        Some(larql_inference::layer_executor::LocalWalkExecutor::new(be))
    } else {
        None
    };

    // pick-next: quant path uses vindex lm_head top-k for speed;
    // dense path uses raw f32 logits argmax.
    let pick_next =
        |hidden: &ndarray::Array2<f32>, weights: &larql_inference::ModelWeights| -> u32 {
            if let Some(idx) = index {
                use larql_inference::layer_graph::generate::lm_head_topk;
                let h_1d = ndarray::Array1::from_iter(hidden.iter().copied());
                lm_head_topk(idx, weights, &h_1d, 1, be)
                    .first()
                    .map(|(t, _)| *t)
                    .unwrap_or_else(|| argmax_token(&hidden_to_raw_logits(weights, hidden)))
            } else {
                argmax_token(&hidden_to_raw_logits(weights, hidden))
            }
        };

    // FFN selection + the prefill / decode loop are scoped per quant
    // vs dense branch so the `WeightFfn { weights }` (dense path's
    // immutable `&weights` borrow) doesn't conflict with the quant
    // path's `&mut weights` (needed for lazy dequant inside
    // `prefill_quant`). The router holds `&weights` too, so it's
    // also scoped to its branch.
    let max_steps = args.warmup + args.tokens;
    let mut decode_ms_all: Vec<f64> = Vec::with_capacity(max_steps);
    // `--ffn-policy` on the quant path: the router needs `&weights`
    // for Walk{k} FFN-tensor lookups, but `prefill_quant` needs
    // `&mut weights` for lazy dequant. Borrows conflict at the call
    // site, so the quant path can't honor `--ffn-policy` today. Log
    // the limitation (matches the pre-merge Q4K behaviour). The
    // non-quant path honors it normally.
    if is_quant && ffn_policy.is_some() {
        eprintln!(
            "[bench] --ffn-policy provided but the quant path does not yet \
             honor it (engine's internal Q4K FFN routing is used instead). \
             Use the dense bench path (no vindex) to exercise the policy."
        );
    }

    let (mut hidden, prefill_ms, mut last_token) = if let Some(idx) = index {
        // Quantised path. Always NullFfn (legacy engines route FFN
        // internally from the vindex; migrated engines substitute
        // their own under *_via_executor).
        let null_ffn = NullFfn;
        let ffn: &dyn FfnBackend = &null_ffn;
        let t_pre = Instant::now();
        let hidden = match executor.as_ref() {
            Some(exec) => engine
                .prefill_quant_via_executor(weights, exec, ffn, idx, token_ids)
                .map_err(|e| format!("engine prefill (quant + executor) failed: {e}"))?,
            None => engine
                .prefill_quant(weights, ffn, idx, token_ids, be)
                .map_err(|e| format!("engine prefill (quant) failed: {e}"))?,
        };
        let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
        let last_token = pick_next(&hidden, weights);
        (hidden, prefill_ms, last_token)
    } else {
        // Dense path.
        let weight_ffn = WeightFfn { weights };
        let router = match ffn_policy {
            Some(p) => Some(
                p.build_router(weights, None)
                    .map_err(|e| format!("--ffn-policy build: {e}"))?,
            ),
            None => None,
        };
        let ffn: &dyn FfnBackend = match &router {
            Some(r) => r,
            None => &weight_ffn,
        };
        let t_pre = Instant::now();
        let hidden = engine
            .prefill(weights, ffn, token_ids)
            .map_err(|e| format!("engine prefill failed: {e}"))?;
        let prefill_ms = t_pre.elapsed().as_secs_f64() * 1000.0;
        let last_token = pick_next(&hidden, weights);
        (hidden, prefill_ms, last_token)
    };

    // Decode loop. Re-build the FFN inside the appropriate branch to
    // avoid the same borrow conflict as the prefill path.
    if let Some(idx) = index {
        let null_ffn = NullFfn;
        let ffn: &dyn FfnBackend = &null_ffn;
        for _ in 0..max_steps {
            let t = Instant::now();
            hidden = match executor.as_ref() {
                Some(exec) => engine
                    .decode_step_quant_via_executor(weights, exec, ffn, idx, last_token)
                    .map_err(|e| format!("engine decode_step (quant + executor) failed: {e}"))?,
                None => engine
                    .decode_step_quant(weights, ffn, idx, last_token, be)
                    .map_err(|e| format!("engine decode_step (quant) failed: {e}"))?,
            };
            decode_ms_all.push(t.elapsed().as_secs_f64() * 1000.0);
            last_token = pick_next(&hidden, weights);
        }
    } else {
        let weight_ffn = WeightFfn { weights };
        let router = match ffn_policy {
            Some(p) => Some(
                p.build_router(weights, None)
                    .map_err(|e| format!("--ffn-policy build: {e}"))?,
            ),
            None => None,
        };
        let ffn: &dyn FfnBackend = match &router {
            Some(r) => r,
            None => &weight_ffn,
        };
        for _ in 0..max_steps {
            let t = Instant::now();
            hidden = engine
                .decode_step(weights, ffn, last_token)
                .map_err(|e| format!("engine decode_step failed: {e}"))?;
            decode_ms_all.push(t.elapsed().as_secs_f64() * 1000.0);
            last_token = pick_next(&hidden, weights);
        }
    }

    // Drop hidden explicitly so the borrow checker / dead-code lint
    // doesn't flag the last loop iteration's assignment as unread —
    // the bench reports timing + memory, not the final hidden state.
    let _ = hidden;

    let summary = summarize_engine_result(&decode_ms_all, args.warmup);
    let note = format_kv_memory_note(engine.memory_bytes(), engine.cold_bytes(), kv_ref_bytes);

    if args.verbose {
        eprintln!(
            "[bench] {} post-decode: {}",
            info.name,
            engine.info().description
        );
    }
    if args.profile {
        if let Some(s) = engine.stage_summary() {
            s.print();
        }
    }

    Ok(BenchRow {
        backend: label,
        prefill_ms,
        avg_decode_ms: summary.avg_decode_ms,
        p50_ms: summary.p50_ms,
        p99_ms: summary.p99_ms,
        tok_per_s: summary.tok_per_s,
        stages: None,
        profile: None,
        ffn_rtt_ms: None,
        attn_ms: None,
        wire_bytes_per_tok: None,
        shard_efficiency: None,
        n_steps: summary.n_steps,
        note,
    })
}
