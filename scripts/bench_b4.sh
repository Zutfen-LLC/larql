#!/usr/bin/env bash
# LARQL-GPU-B4-CORRECTION D: reproducible B4 benchmark.
#
# Two explicit phases (the committed B4 report conflated these by setting
# LARQL_GPU_PROFILE=1 for the performance runs):
#
#   perf       UNINSTRUMENTED (LARQL_GPU_PROFILE / LARQL_GPU_DIAG unset).
#              4 modes × 5 reps × (5 warmup + 79 measured) decode steps.
#              This is the wall-clock source of truth.
#   structural INSTRUMENTED (LARQL_GPU_PROFILE=1). 1 rep per mode. Counters
#              only — never the wall-clock source of truth.
#
# Modes: BaselineA (graphs=0 b4=0), B4A (0,1), BaselineB (1,0), B4B (1,1).
#
# Output: every per-repetition raw value is preserved (raw JSONL + raw stdout),
# and medians/MAD are computed from those raw records by b4_aggregate.py — no
# hand-entered performance claims.
set -euo pipefail

VINDEX="${LARQL_BENCH_VINDEX:-/home/zutfen/models/qwen2.5-3b-q4k.vindex}"
LARQL="${LARQL_BENCH_BIN:-/home/zutfen/code/larql/target/release/larql}"
PROMPT="${LARQL_BENCH_PROMPT:-Write a step by step guide on how to bake a cake from scratch.}"
TOKENS="${LARQL_BENCH_TOKENS:-79}"   # measured decode steps (warmup discarded by the engine)
WARMUP="${LARQL_BENCH_WARMUP:-5}"
REPS="${LARQL_BENCH_REPS:-5}"
OUTDIR="${LARQL_BENCH_OUTDIR:-/tmp/b4_bench}"
AGG="${LARQL_BENCH_AGG:-/home/zutfen/code/larql/scripts/b4_aggregate.py}"

: "${LD_LIBRARY_PATH:=/home/zutfen/openblas-local/usr/lib:/home/zutfen/cuda-lib}"
: "${CUDA_HOME:=/home/zutfen/cuda-headers/cuda_cudart-linux-x86_64-12.4.127-archive}"
export LD_LIBRARY_PATH CUDA_HOME

mkdir -p "$OUTDIR"
RAW_PERF="$OUTDIR/perf_raw.jsonl"
RAW_STRUCT="$OUTDIR/structural_raw.jsonl"
PERF_LOG="$OUTDIR/perf_stdout.log"
STRUCT_LOG="$OUTDIR/structural_stdout.log"
: > "$RAW_PERF"; : > "$RAW_STRUCT"; : > "$PERF_LOG"; : > "$STRUCT_LOG"

# ── environment provenance ─────────────────────────────────────────────
sha="$(git -C /home/zutfen/code/larql rev-parse HEAD 2>/dev/null || echo unknown)"
gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
nvrtc="$(ls /home/zutfen/cuda-lib/libnvrtc-builtins.so.* 2>/dev/null | head -1 || echo unknown)"
echo "# LARQL-GPU-B4-CORRECTION benchmark ($(date -Iseconds))" | tee "$OUTDIR/header.txt"
{
  echo "# sha=$sha"
  echo "# bin=$LARQL"
  echo "# vindex=$VINDEX"
  echo "# gpu=$gpu"
  echo "# nvrtc=$nvrtc"
  echo "# prompt=\"$PROMPT\""
  echo "# tokens=$TOKENS warmup=$WARMUP reps=$REPS"
  echo "# effective LARQL_* env at script start:"
  env | grep '^LARQL_' | sort | sed 's/^/#   /' || true
} >> "$OUTDIR/header.txt"
cat "$OUTDIR/header.txt"

# Extract one data row from bench output → JSON record on stdout.
# Row layout: "  larql-cuda  <pre>ms  <mean>ms  <p50>ms  <tok/s>  <steps>  <note...>"
emit_record() {
  local mode="$1" graphs="$2" b4="$3" instrumented="$4" rep="$5" out="$6"
  python3 - "$mode" "$graphs" "$b4" "$instrumented" "$rep" "$out" <<'PY'
import json, re, sys
mode, graphs, b4, instrumented, rep, path = sys.argv[1:7]
instrumented = instrumented == "1"
text = open(path).read()
m = re.search(r'(?m)^\s*larql-cuda\s+([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)ms\s+([\d.]+)\s+(\d+)(.*)$', text)
if not m:
    print(json.dumps({"mode": mode, "graphs": int(graphs), "b4": int(b4),
                       "instrumented": instrumented, "rep": int(rep), "failed": True,
                       "reason": "no larql-cuda data row"}))
    sys.exit(0)
note = m.group(6).strip()
rec = {"mode": mode, "graphs": int(graphs), "b4": int(b4),
       "instrumented": instrumented, "rep": int(rep),
       "prefill_ms": float(m.group(1)), "mean_ms": float(m.group(2)),
       "p50_ms": float(m.group(3)), "tok_s": float(m.group(4)),
       "n_steps": int(m.group(5)), "early_stop": "early stop" in note, "note": note}
print(json.dumps(rec))
PY
}

run_one() {
  local mode="$1" graphs="$2" b4="$3" instrumented="$4" rep="$5" rawfile="$6" logfile="$7"
  local -a extra=()
  [ "$graphs" = "1" ] && extra+=(LARQL_CUDA_GRAPHS=1)
  [ "$b4" = "1" ] && extra+=(LARQL_CUDA_DEVICE_GREEDY=1)
  if [ "$instrumented" = "1" ]; then
    extra+=(LARQL_GPU_PROFILE=1)
  else
    # Force the perf phase to be genuinely uninstrumented even if the
    # surrounding shell exports LARQL_GPU_PROFILE.
    extra+=()
  fi
  local out
  # timeout: generous; one long-lived process per rep (B4 §13).
  set +e
  out=$(env "${extra[@]}" timeout 600 "$LARQL" bench "$VINDEX" \
        --backends cuda --tokens "$TOKENS" --warmup "$WARMUP" --prompt "$PROMPT" 2>&1)
  local rc=$?
  set -e
  printf '%s\n' "$out" >> "$logfile"
  if [ $rc -ne 0 ]; then
    # 124 = timeout; record the failure rather than aborting the whole sweep.
    python3 -c "import json,sys; print(json.dumps({'mode':'$mode','graphs':$graphs,'b4':$b4,'instrumented':$( [ "$instrumented" = "1" ] && echo True || echo False ),'rep':$rep,'failed':True,'rc':$rc}))" >> "$rawfile"
    echo "  [FAIL] $mode rep$rep graphs=$graphs b4=$b4 instr=$instrumented rc=$rc"
    return
  fi
  printf '%s\n' "$out" > "$OUTDIR/${mode}_$( [ "$instrumented" = "1" ] && echo struct || echo perf )_rep${rep}.out"
  emit_record "$mode" "$graphs" "$b4" "$instrumented" "$rep" "$OUTDIR/${mode}_$( [ "$instrumented" = "1" ] && echo struct || echo perf )_rep${rep}.out" >> "$rawfile"
  local mean p50
  mean=$(printf '%s\n' "$out" | awk '/larql-cuda/{print $3}' | sed 's/ms//')
  p50=$(printf '%s\n' "$out" | awk '/larql-cuda/{print $4}' | sed 's/ms//')
  echo "  [ok]   $mode rep$rep graphs=$graphs b4=$b4 instr=$instrumented mean=${mean}ms p50=${p50}ms"
}

# ── Phase 1: performance (uninstrumented) ──────────────────────────────
echo "=== Phase 1: performance (UNINSTRUMENTED, source of truth) ==="
perf_phase() {
  run_one "BaselineA" 0 0 0 1 "$RAW_PERF" "$PERF_LOG"
  run_one "B4A"       0 1 0 1 "$RAW_PERF" "$PERF_LOG"
  run_one "BaselineB" 1 0 0 1 "$RAW_PERF" "$PERF_LOG"
  run_one "B4B"       1 1 0 1 "$RAW_PERF" "$PERF_LOG"
}
for rep in $(seq 1 "$REPS"); do
  echo "# perf rep $rep/$REPS"
  perf_phase
done

echo
echo "=== Phase 1 aggregation (medians + MAD from raw records) ==="
python3 "$AGG" < "$RAW_PERF" | tee "$OUTDIR/perf_aggregated.json"

# ── Phase 2: structural (instrumented, counters only) ──────────────────
echo
echo "=== Phase 2: structural (INSTRUMENTED, counters only — NOT wall-clock truth) ==="
for spec in "BaselineA:0:0" "B4A:0:1" "BaselineB:1:0" "B4B:1:1"; do
  mode="${spec%%:*}"; rest="${spec#*:}"; graphs="${rest%%:*}"; b4="${rest##*:}"
  echo "# structural $mode graphs=$graphs b4=$b4"
  run_one "$mode" "$graphs" "$b4" 1 1 "$RAW_STRUCT" "$STRUCT_LOG"
done

echo
echo "=== Phase 2 aggregation (structural counters; instrumented=true) ==="
python3 "$AGG" < "$RAW_STRUCT" | tee "$OUTDIR/structural_aggregated.json"

echo
echo "# done. artifacts in $OUTDIR"
echo "# raw perf JSONL:   $RAW_PERF"
echo "# raw struct JSONL: $RAW_STRUCT"
