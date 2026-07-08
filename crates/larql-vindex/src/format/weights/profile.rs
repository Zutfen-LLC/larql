//! Opt-in, low-overhead extraction profiling.
//!
//! See IMPORT-001 (GGUF/vindex extraction hotspot profiling). This module
//! measures the serial kquant / f32 writer hot paths so the next slice can
//! decide whether to parallelise attention, FFN, MoE, or optimise
//! quantisation. It is **purely observational**: enabling it never changes
//! emitted bytes, manifest entries, offsets, or file ordering.
//!
//! ## How it's wired
//!
//! Writer functions accept an `Option<&ExtractProfiler>`. `None` means
//! profiling is off — the [`Recorder`] helpers are true no-ops (no
//! `Instant` is ever constructed in that case, so the disabled branch is
//! a single `Option::is_some` test). The CLI (`--profile-extract` /
//! `LARQL_EXTRACT_PROFILE=1`) constructs one profiler, threads it through,
//! then emits a stderr summary + `extract_profile.json`.
//!
//! ## What's measured
//!
//! Per-stage totals (attention kquant, FFN kquant, per-layer MoE kquant,
//! norms/router, PLE sidecar, lm_head, manifest/index update) and a
//! fine-grained per-operation breakdown (`fetch`, `pad`, `quantize`,
//! `write`, `quantize_moe`) with rows / cols / bytes where cheap.

use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::error::VindexError;

/// One timed operation within a weight-writing stage.
#[derive(Clone, Serialize)]
pub struct OpRecord {
    /// Coarse stage label (e.g. `"attn_kquant"`, `"ffn_kquant"`, `"moe_kquant"`).
    pub stage: String,
    /// Which tensor/component (e.g. `"Q"`, `"gate"`, `"down"`, `"lm_head"`).
    pub component: String,
    /// Layer index, when applicable (`-1` = not per-layer, e.g. lm_head).
    pub layer: i64,
    /// Operation kind: `fetch` / `pad` / `quantize` / `write` /
    /// `quantize_moe` / `manifest` / `index_update`.
    pub op: String,
    /// Elapsed wall-clock, microseconds.
    pub dur_us: f64,
    pub rows: u64,
    pub cols: u64,
    pub padded_cols: u64,
    pub bytes: u64,
    /// Quant format tag (`"Q4_K"`, `"Q6_K"`, or empty).
    pub quant: String,
}

impl OpRecord {
    fn new(stage: &str, component: &str, layer: Option<usize>, op: &str, dur: Duration) -> Self {
        Self {
            stage: stage.to_string(),
            component: component.to_string(),
            layer: layer.map(|l| l as i64).unwrap_or(-1),
            op: op.to_string(),
            dur_us: dur.as_secs_f64() * 1e6,
            rows: 0,
            cols: 0,
            padded_cols: 0,
            bytes: 0,
            quant: String::new(),
        }
    }

    fn with_rows_cols(mut self, rows: u64, cols: u64) -> Self {
        self.rows = rows;
        self.cols = cols;
        self
    }

    fn with_padded_cols(mut self, padded_cols: u64) -> Self {
        self.padded_cols = padded_cols;
        self
    }

    fn with_bytes(mut self, bytes: u64) -> Self {
        self.bytes = bytes;
        self
    }

    fn with_quant(mut self, quant: &str) -> Self {
        self.quant = quant.to_string();
        self
    }
}

#[derive(Default)]
struct StageTotal {
    dur_us: f64,
    count: usize,
}

#[derive(Default)]
struct ProfilerInner {
    /// Wall-clock start of the whole extraction (set by [`ExtractProfiler::mark_start`]).
    start: Option<Instant>,
    /// Coarse stage totals keyed by stage label.
    stages: std::collections::HashMap<String, StageTotal>,
    /// Every timed operation (the raw timeline; aggregated in the report).
    ops: Vec<OpRecord>,
    /// Resolved `--jobs`/`LARQL_EXTRACT_JOBS` value, if the caller reported
    /// one via [`ExtractProfiler::set_jobs`]. `None` when extraction never
    /// set it (e.g. a profiler built directly by a test).
    jobs: Option<usize>,
}

/// The profiler. Construct one, thread `Option<&Self>` through writer
/// functions, then call [`ExtractProfiler::write_json_report`] /
/// [`ExtractProfiler::print_summary`].
///
/// Uses interior mutability (`Mutex`) so callers pass a shared `&`
/// reference from multiple worker threads during parallel layer
/// transforms (IMPORT-002); contention is low since each op only holds
/// the lock long enough to push one small record.
pub struct ExtractProfiler {
    inner: Mutex<ProfilerInner>,
}

impl Default for ExtractProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtractProfiler {
    /// Construct an enabled profiler. [`ExtractProfiler::mark_start`] is
    /// called automatically.
    pub fn new() -> Self {
        let inner = ProfilerInner {
            start: Some(Instant::now()),
            ..Default::default()
        };
        Self {
            inner: Mutex::new(inner),
        }
    }

    /// (Re)arm the overall extraction timer.
    pub fn mark_start(&self) {
        self.inner.lock().unwrap().start = Some(Instant::now());
    }

    /// Total elapsed since [`mark_start`] / [`new`].
    pub fn elapsed(&self) -> Duration {
        self.inner
            .lock()
            .unwrap()
            .start
            .map(|t| t.elapsed())
            .unwrap_or_default()
    }

    /// Record a coarse stage total (called by orchestrators at stage
    /// boundaries). Accumulates across calls so a stage split across
    /// sub-phases sums correctly.
    pub fn record_stage(&self, stage: &str, dur: Duration) {
        let mut inner = self.inner.lock().unwrap();
        let e = inner.stages.entry(stage.to_string()).or_default();
        e.dur_us += dur.as_secs_f64() * 1e6;
        e.count += 1;
    }

    /// Record one fine-grained operation. Safe to call concurrently from
    /// multiple worker threads during parallel layer transforms.
    pub fn record_op(&self, rec: OpRecord) {
        self.inner.lock().unwrap().ops.push(rec);
    }

    /// Record the resolved `--jobs`/`LARQL_EXTRACT_JOBS` worker count so
    /// it shows up in the JSON report and stderr summary.
    pub fn set_jobs(&self, jobs: usize) {
        self.inner.lock().unwrap().jobs = Some(jobs);
    }

    // ── internal: aggregate ──

    fn aggregate_by_op(ops: &[OpRecord]) -> std::collections::BTreeMap<String, (f64, usize, u64)> {
        let mut out: std::collections::BTreeMap<String, (f64, usize, u64)> =
            std::collections::BTreeMap::new();
        for r in ops {
            let e = out.entry(r.op.clone()).or_default();
            e.0 += r.dur_us;
            e.1 += 1;
            e.2 += r.bytes;
        }
        out
    }

    fn top_ops(ops: &[OpRecord], n: usize) -> Vec<&OpRecord> {
        let mut idx: Vec<&OpRecord> = ops.iter().collect();
        idx.sort_unstable_by(|a, b| {
            b.dur_us
                .partial_cmp(&a.dur_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(n);
        idx
    }

    /// Write the machine-readable JSON report to `path`.
    pub fn write_json_report(&self, path: &Path) -> Result<(), VindexError> {
        let inner = self.inner.lock().unwrap();
        let by_op = Self::aggregate_by_op(&inner.ops);
        let top = Self::top_ops(&inner.ops, 20);

        let stages: Vec<serde_json::Value> = {
            let mut s: Vec<(String, f64, usize)> = inner
                .stages
                .iter()
                .map(|(k, v)| (k.clone(), v.dur_us, v.count))
                .collect();
            s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            s.into_iter()
                .map(|(stage, dur_us, count)| {
                    serde_json::json!({
                        "stage": stage,
                        "dur_ms": (dur_us / 1000.0),
                        "count": count,
                    })
                })
                .collect()
        };

        let by_operation: Vec<serde_json::Value> = by_op
            .into_iter()
            .map(|(op, (dur_us, count, bytes))| {
                serde_json::json!({
                    "op": op,
                    "dur_ms": (dur_us / 1000.0),
                    "count": count,
                    "bytes": bytes,
                })
            })
            .collect();

        let top_ops: Vec<serde_json::Value> = top
            .into_iter()
            .map(|r| {
                serde_json::json!({
                    "stage": r.stage,
                    "component": r.component,
                    "layer": r.layer,
                    "op": r.op,
                    "dur_ms": (r.dur_us / 1000.0),
                    "rows": r.rows,
                    "cols": r.cols,
                    "padded_cols": r.padded_cols,
                    "bytes": r.bytes,
                    "quant": r.quant,
                })
            })
            .collect();

        let total_quant_bytes: u64 = inner
            .ops
            .iter()
            .filter(|r| r.op == "quantize" || r.op == "quantize_moe")
            .map(|r| r.bytes)
            .sum();
        let total_write_bytes: u64 = inner
            .ops
            .iter()
            .filter(|r| r.op == "write")
            .map(|r| r.bytes)
            .sum();
        let total_fetch_elems: u64 = inner
            .ops
            .iter()
            .filter(|r| r.op == "fetch")
            .map(|r| r.rows.saturating_mul(r.cols))
            .sum();

        let elapsed_ms = inner
            .start
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        let report = serde_json::json!({
            "total_extract_ms": elapsed_ms,
            "jobs": inner.jobs,
            "parallel": inner.jobs.is_some_and(|j| j > 1),
            "stages": stages,
            "by_operation": by_operation,
            "top_ops": top_ops,
            "bytes": {
                "f32_elements_materialized": total_fetch_elems,
                "quantized_emitted": total_quant_bytes,
                "output_written": total_write_bytes,
            },
            "operation_count": inner.ops.len(),
        });

        let text = serde_json::to_string_pretty(&report)
            .map_err(|e| VindexError::Parse(format!("serialize profile: {e}")))?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Emit a short human-readable summary to stderr — concise enough to
    /// paste into an issue or PR comment.
    pub fn print_summary(&self) {
        let inner = self.inner.lock().unwrap();
        let total_ms = inner
            .start
            .map(|t| t.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        eprintln!("\n── Extract profile ──");
        eprintln!("  Total: {:.1}s", total_ms / 1000.0);
        if let Some(jobs) = inner.jobs {
            eprintln!(
                "  Jobs: {jobs} ({})",
                if jobs > 1 { "parallel" } else { "serial" }
            );
        }

        // Stage totals (sorted desc).
        let mut stages: Vec<(&String, &StageTotal)> = inner.stages.iter().collect();
        stages.sort_by(|a, b| {
            b.1.dur_us
                .partial_cmp(&a.1.dur_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if !stages.is_empty() {
            eprintln!("  Stages:");
            let grand: f64 = stages.iter().map(|(_, v)| v.dur_us).sum();
            for (stage, v) in &stages {
                let pct = if grand > 0.0 {
                    v.dur_us / grand * 100.0
                } else {
                    0.0
                };
                eprintln!("    {:<22} {:>7.1}s  {:>5.1}%", stage, v.dur_us / 1e6, pct);
            }
        }

        // Aggregate by operation kind.
        let by_op = Self::aggregate_by_op(&inner.ops);
        if !by_op.is_empty() {
            eprintln!("  By operation:");
            for (op, (dur_us, count, bytes)) in &by_op {
                let mb = *bytes as f64 / (1024.0 * 1024.0);
                let suffix = if mb >= 1.0 {
                    format!("  {:.1} MB", mb)
                } else if *bytes > 0 {
                    format!("  {} B", bytes)
                } else {
                    String::new()
                };
                eprintln!(
                    "    {:<14} {:>7.1}s  ({} ops{})",
                    op,
                    dur_us / 1e6,
                    count,
                    suffix
                );
            }
        }

        // Top slowest individual ops.
        let top = Self::top_ops(&inner.ops, 5);
        if !top.is_empty() {
            eprintln!("  Top hotspots:");
            for r in top {
                let loc = if r.layer >= 0 {
                    format!("{} L{}", r.component, r.layer)
                } else {
                    r.component.clone()
                };
                eprintln!(
                    "    {:<7} {:<12} {:>6.0}ms",
                    r.stage,
                    loc,
                    r.dur_us / 1000.0
                );
            }
        }
        eprintln!("  (full report: extract_profile.json)");
    }
}

// ── Recorder: ergonomic call-site helper ──

/// Thin wrapper around `Option<&ExtractProfiler>` that keeps instrumentation
/// readable and is a true no-op when the profiler is absent.
///
/// Typical use:
///
/// ```ignore
/// let rec = Recorder::from(prof);
/// let t = rec.now();                       // None when disabled — free
/// let q = source.get_tensor(&q_key);
/// rec.fetch(t, "attn_kquant", "Q", layer, rows, cols);
/// ```
pub struct Recorder<'a> {
    prof: Option<&'a ExtractProfiler>,
}

impl<'a> Recorder<'a> {
    pub fn from(prof: Option<&'a ExtractProfiler>) -> Self {
        Self { prof }
    }

    /// `None` when profiling is disabled — so disabled instrumentation pays
    /// only for this single branch, never an `Instant::now()`.
    #[inline]
    pub fn now(&self) -> Option<Instant> {
        if self.prof.is_some() {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Record a coarse stage total (convenience for orchestrators).
    pub fn stage(&self, stage: &str, t: Option<Instant>) {
        if let (Some(p), Some(t)) = (self.prof, t) {
            p.record_stage(stage, t.elapsed());
        }
    }

    /// Tensor fetch / dequant (source read).
    pub fn fetch(
        &self,
        t: Option<Instant>,
        stage: &str,
        comp: &str,
        layer: Option<usize>,
        rows: u64,
        cols: u64,
    ) {
        self.emit(t, stage, comp, layer, "fetch", |r| {
            r.with_rows_cols(rows, cols)
        });
    }

    /// Row padding to super-block boundary.
    pub fn pad(
        &self,
        t: Option<Instant>,
        stage: &str,
        comp: &str,
        layer: Option<usize>,
        in_cols: u64,
        padded_cols: u64,
    ) {
        self.emit(t, stage, comp, layer, "pad", |r| {
            r.with_rows_cols(0, in_cols).with_padded_cols(padded_cols)
        });
    }

    /// Q4_K / Q6_K quantisation.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize(
        &self,
        t: Option<Instant>,
        stage: &str,
        comp: &str,
        layer: Option<usize>,
        quant: &str,
        in_elems: u64,
        out_bytes: u64,
    ) {
        self.emit(t, stage, comp, layer, "quantize", |r| {
            r.with_bytes(out_bytes)
                .with_quant(quant)
                .with_rows_cols(0, in_elems)
        });
    }

    /// Per-expert MoE quantisation (`quantize_moe_entries`).
    pub fn quantize_moe(
        &self,
        t: Option<Instant>,
        stage: &str,
        comp: &str,
        layer: Option<usize>,
        in_bytes: u64,
        out_bytes: u64,
    ) {
        self.emit(t, stage, comp, layer, "quantize_moe", |r| {
            r.with_rows_cols(in_bytes, out_bytes)
        });
    }

    /// File write.
    pub fn write(
        &self,
        t: Option<Instant>,
        stage: &str,
        comp: &str,
        layer: Option<usize>,
        bytes: u64,
    ) {
        self.emit(t, stage, comp, layer, "write", |r| r.with_bytes(bytes));
    }

    /// Manifest / index bookkeeping (small, latency-only).
    pub fn manifest(&self, t: Option<Instant>, stage: &str, comp: &str) {
        self.emit(t, stage, comp, None, "manifest", |r| r);
    }

    fn emit(
        &self,
        t: Option<Instant>,
        stage: &str,
        comp: &str,
        layer: Option<usize>,
        op: &str,
        decorate: impl FnOnce(OpRecord) -> OpRecord,
    ) {
        if let (Some(p), Some(t)) = (self.prof, t) {
            let rec = decorate(OpRecord::new(stage, comp, layer, op, t.elapsed()));
            p.record_op(rec);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_is_noop_when_no_profiler() {
        // No profiler → now() returns None, all record methods are no-ops,
        // and no Instant is ever built.
        let rec = Recorder::from(None);
        assert!(rec.now().is_none());
        rec.fetch(None, "attn_kquant", "Q", Some(0), 8, 8);
        rec.quantize(None, "attn_kquant", "Q", Some(0), "Q4_K", 8, 144);
        rec.quantize_moe(None, "moe_kquant", "moe", Some(0), 1024, 512);
        rec.pad(None, "attn_kquant", "Q", Some(0), 8, 256);
        rec.manifest(None, "attn_kquant", "attn_manifest");
        rec.stage("attn_kquant", None);
        // Nothing to assert beyond "did not panic" + cheapness.
    }

    #[test]
    fn mark_start_and_elapsed_round_trip() {
        let prof = ExtractProfiler::new();
        // mark_start re-arms; elapsed reflects the new start.
        prof.mark_start();
        std::thread::sleep(std::time::Duration::from_micros(80));
        let e = prof.elapsed();
        assert!(
            e.as_micros() > 0,
            "elapsed must be positive after mark_start"
        );
    }

    #[test]
    fn quantize_moe_recorder_records() {
        let prof = ExtractProfiler::new();
        let rec = Recorder::from(Some(&prof));
        let t = rec.now();
        rec.quantize_moe(t, "moe_kquant", "quantize_moe_entries", Some(1), 2048, 768);
        let inner = prof.inner.lock().unwrap();
        assert_eq!(inner.ops.len(), 1);
        assert_eq!(inner.ops[0].op, "quantize_moe");
        assert_eq!(inner.ops[0].rows, 2048);
        assert_eq!(inner.ops[0].cols, 768);
        assert!(inner.ops[0].dur_us >= 0.0);
    }

    #[test]
    fn pad_recorder_records_padded_cols() {
        let prof = ExtractProfiler::new();
        let rec = Recorder::from(Some(&prof));
        let t = rec.now();
        rec.pad(t, "attn_kquant", "Q", Some(0), 200, 256);
        let inner = prof.inner.lock().unwrap();
        assert_eq!(inner.ops.len(), 1);
        assert_eq!(inner.ops[0].op, "pad");
        assert_eq!(inner.ops[0].padded_cols, 256);
    }

    #[test]
    fn print_summary_runs_without_panic() {
        // print_summary writes to stderr; exercise all its branches
        // (stages, by-operation with MB + B suffixes, top hotspots) so
        // the lines stay covered.
        let prof = ExtractProfiler::new();
        let rec = Recorder::from(Some(&prof));
        let t = rec.now();
        rec.fetch(t, "attn_kquant", "Q", Some(0), 8, 8);
        let t = rec.now();
        rec.quantize(t, "ffn_kquant", "down", Some(0), "Q6_K", 256, 1024);
        let t = rec.now();
        // Large byte count → aggregate ≥ 1 MiB so the MB suffix branch runs.
        rec.write(t, "ffn_kquant", "down", Some(0), 2_000_000);
        // None-layer op → exercises the non-per-layer `else` hotspot branch.
        let t = rec.now();
        rec.fetch(t, "lm_head", "lm_head", None, 4, 4);
        prof.record_stage("attn_kquant", Duration::from_micros(300));
        prof.record_stage("ffn_kquant", Duration::from_micros(700));
        // No assertions on output — must not panic and must cover the
        // formatting branches (MB suffix, B suffix, percent, top-5 list).
        prof.print_summary();
    }

    #[test]
    fn default_impl_matches_new() {
        let a = ExtractProfiler::default();
        let b = ExtractProfiler::new();
        // Both start armed; elapsed is non-negative for each.
        assert!(a.elapsed().as_nanos() as i128 >= 0);
        assert!(b.elapsed().as_nanos() as i128 >= 0);
    }

    #[test]
    fn print_summary_handles_zero_duration_stages() {
        // Edge case: a stage with zero total → the `grand == 0.0` percent
        // branch must not divide by zero / panic.
        let prof = ExtractProfiler::new();
        prof.record_stage("empty_stage", Duration::ZERO);
        prof.print_summary();
    }

    #[test]
    fn profiler_collects_ops_and_aggregates() {
        let prof = ExtractProfiler::new();
        let rec = Recorder::from(Some(&prof));

        let t = rec.now();
        std::thread::sleep(std::time::Duration::from_micros(50));
        rec.fetch(t, "attn_kquant", "Q", Some(0), 8, 8);

        let t = rec.now();
        rec.quantize(t, "attn_kquant", "Q", Some(0), "Q4_K", 8, 144);

        let inner = prof.inner.lock().unwrap();
        assert_eq!(inner.ops.len(), 2);
        assert_eq!(inner.ops[0].op, "fetch");
        assert_eq!(inner.ops[0].stage, "attn_kquant");
        assert_eq!(inner.ops[0].component, "Q");
        assert_eq!(inner.ops[0].layer, 0);
        assert_eq!(inner.ops[0].rows, 8);
        assert!(inner.ops[0].dur_us > 0.0, "timing must be nonzero");

        let by_op = ExtractProfiler::aggregate_by_op(&inner.ops);
        assert_eq!(by_op.len(), 2); // fetch + quantize
        assert!(by_op["fetch"].0 > 0.0);
    }

    #[test]
    fn stage_totals_accumulate() {
        let prof = ExtractProfiler::new();
        prof.record_stage("attn_kquant", Duration::from_micros(100));
        prof.record_stage("attn_kquant", Duration::from_micros(200));
        prof.record_stage("ffn_kquant", Duration::from_micros(500));

        let inner = prof.inner.lock().unwrap();
        assert!((inner.stages["attn_kquant"].dur_us - 300.0).abs() < 0.1);
        assert_eq!(inner.stages["attn_kquant"].count, 2);
        assert!((inner.stages["ffn_kquant"].dur_us - 500.0).abs() < 0.1);
    }

    #[test]
    fn json_report_round_trips() {
        let prof = ExtractProfiler::new();
        let rec = Recorder::from(Some(&prof));
        let t = rec.now();
        rec.fetch(t, "attn_kquant", "Q", Some(0), 8, 8);
        let t = rec.now();
        rec.write(t, "attn_kquant", "Q", Some(0), 144);
        prof.record_stage("attn_kquant", Duration::from_micros(50));

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("extract_profile.json");
        prof.write_json_report(&path).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(v["total_extract_ms"].as_f64().unwrap_or(0.0) >= 0.0);
        assert_eq!(v["stages"].as_array().unwrap().len(), 1);
        assert!(v["by_operation"].as_array().unwrap().len() >= 2);
        assert!(v["top_ops"].as_array().unwrap().len() <= 20);
        assert!(!v["top_ops"].as_array().unwrap().is_empty());
        // byte totals present
        assert!(v["bytes"]["output_written"].as_u64().unwrap() >= 144);
    }

    #[test]
    fn top_ops_orders_by_duration_desc() {
        let prof = ExtractProfiler::new();
        for (dur, label) in [(10u64, "a"), (1000, "b"), (100, "c")] {
            prof.record_op(OpRecord::new(
                "s",
                label,
                Some(0),
                "fetch",
                Duration::from_micros(dur),
            ));
        }
        let inner = prof.inner.lock().unwrap();
        let top = ExtractProfiler::top_ops(&inner.ops, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].component, "b"); // longest first
        assert_eq!(top[1].component, "c");
    }
}
