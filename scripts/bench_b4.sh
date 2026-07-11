#!/usr/bin/env bash
# LARQL-GPU-B4 benchmark: 4 modes × 5 reps, 79 measured decode steps.
# Production-default Qwen2.5-3B Q4_K_M vindex, RTX 3060.
set -uo pipefail

VINDEX="${LARQL_BENCH_VINDEX:-/home/zutfen/models/qwen2.5-3b-q4k.vindex}"
LARQL="${LARQL_BENCH_BIN:-/home/zutfen/code/larql/target/release/larql}"
PROMPT="Write a step by step guide on how to bake a cake from scratch."
TOKENS=79        # 5 warmup (discarded) + 79 measured? bench --tokens is measured;
                 # the D6 methodology used --warmup 3 --tokens 79. Packet §13 says
                 # 5 warmup + 79 measured. We use --warmup 5 --tokens 79.
WARMUP=5
REPS=5
OUT="${LARQL_BENCH_OUT:-/tmp/b4_bench.txt}"

: "${LD_LIBRARY_PATH:=/home/zutfen/openblas-local/usr/lib:/home/zutfen/cuda-lib}"
: "${CUDA_HOME:=/home/zutfen/cuda-headers/cuda_cudart-linux-x86_64-12.4.127-archive}"
export LD_LIBRARY_PATH CUDA_HOME

run_mode() {
  local label="$1" graphs="$2" b4="$3"
  local extra=()
  [ "$graphs" = "1" ] && extra+=(LARQL_CUDA_GRAPHS=1)
  [ "$b4" = "1" ] && extra+=(LARQL_CUDA_DEVICE_GREEDY=1)
  for rep in $(seq 1 "$REPS"); do
    # One long-lived process per run (§13). Capture p50 + tok/s + stage lines.
    local out
    out=$(env "${extra[@]}" LARQL_GPU_PROFILE=1 timeout 300 "$LARQL" bench "$VINDEX" \
        --backends cuda --tokens "$TOKENS" --warmup "$WARMUP" --prompt "$PROMPT" 2>&1)
    local p50 mean toks
    p50=$(echo "$out" | awk '/larql-cuda/{print $5}')
    mean=$(echo "$out" | awk '/larql-cuda/{print $4}')
    toks=$(echo "$out" | awk '/larql-cuda/{print $6}')
    local lmhead finalnorm gpufwd
    lmhead=$(echo "$out" | awk '/lm_head/{print $2}')
    finalnorm=$(echo "$out" | awk '/final_norm/{print $2}')
    gpufwd=$(echo "$out" | awk '/GPU fwd/{print $2}')
    printf '%s rep%d graphs=%s b4=%s mean=%s p50=%s tok/s=%s gpu_fwd=%s final_norm=%s lm_head=%s\n' \
      "$label" "$rep" "$graphs" "$b4" "$mean" "$p50" "$toks" "$gpufwd" "$finalnorm" "$lmhead" | tee -a "$OUT"
  done
}

: > "$OUT"
echo "# LARQL-GPU-B4 benchmark ($(date -Iseconds))" | tee -a "$OUT"
echo "# vindex=$VINDEX prompt=\"$PROMPT\" tokens=$TOKENS warmup=$WARMUP reps=$REPS" | tee -a "$OUT"
run_mode "BaselineA" 0 0
run_mode "B4A"       0 1
run_mode "BaselineB" 1 0
run_mode "B4B"       1 1
echo "# done" | tee -a "$OUT"
