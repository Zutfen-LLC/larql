#!/usr/bin/env python3
"""Aggregate LARQL-GPU-PROFILE-001 bench JSONs into a summary + machine-readable JSON.

Reads ~/profiling-results/*.json (emitted by run-bench-matrix.sh) and prints a
median/min/max/MAD summary table per case, then writes the consolidated
machine-readable file bench/baselines/cuda-post-residency-profile-2026-07-10.json.
"""
import json, glob, os, statistics, sys

RESULTS_DIR = os.path.expanduser("~/profiling-results")
OUT_JSON = os.path.join(
    os.path.dirname(__file__), "cuda-post-residency-profile-2026-07-10.json"
)


def mad(data):
    m = statistics.median(data)
    return statistics.median([abs(x - m) for x in data])


def load_case(pattern):
    """Return list of per-rep result rows for a case pattern."""
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, pattern))):
        d = json.load(open(f))
        for r in d["results"]:
            rows.append(r)
    return rows


def summarize(rows):
    if not rows:
        return None
    decodes = [r["ms_per_tok"]["mean"] for r in rows]
    return {
        "n_reps": len(rows),
        "decode_ms_median": statistics.median(decodes),
        "decode_ms_min": min(decodes),
        "decode_ms_max": max(decodes),
        "decode_ms_mad": mad(decodes),
        "tok_per_s": statistics.median([r["tok_per_s"] for r in rows]),
        "prefill_ms": rows[0]["prefill_ms"],
        "stages": rows[0].get("stages"),
        "profile": rows[0].get("profile"),
    }


CASES = [
    ("decode-short-context", "q4k", "*decode-short-context_q4k_pass1*"),
    ("decode-medium-context", "q4k", "*decode-medium-context_q4k_pass1*"),
    ("decode-long-context", "q4k", "*decode-long-context_q4k_pass1*"),
    ("decode-short-context", "q4k-uniform", "*decode-short-context_q4k-uniform_pass1*"),
    ("decode-medium-context", "q4k-uniform", "*decode-medium-context_q4k-uniform_pass1*"),
    ("decode-long-context", "q4k-uniform", "*decode-long-context_q4k-uniform_pass1*"),
    # Pass 2 (instrumented, 1 rep)
    ("decode-short-context", "q4k-pass2", "*decode-short-context_q4k_pass2*"),
    ("decode-medium-context", "q4k-pass2", "*decode-medium-context_q4k_pass2*"),
    ("decode-long-context", "q4k-pass2", "*decode-long-context_q4k_pass2*"),
    ("decode-short-context", "q4k-uniform-pass2", "*decode-short-context_q4k-uniform_pass2*"),
    ("decode-long-context", "q4k-uniform-pass2", "*decode-long-context_q4k-uniform_pass2*"),
]

summary = {}
for name, variant, pat in CASES:
    rows = load_case(pat)
    s = summarize(rows)
    if s is None:
        print(f"  (no data: {name}/{variant})", file=sys.stderr)
        continue
    key = f"{name}__{variant}"
    summary[key] = s
    stages = s["stages"] or {}
    prof = s["profile"] or {}
    print(
        f"{name:24} {variant:18}  decode={s['decode_ms_median']:6.2f}ms "
        f"(±{s['decode_ms_mad']:.2f})  {s['tok_per_s']:4.1f}tok/s  "
        f"prefill={s['prefill_ms']:7.1f}ms  "
        f"gpu={stages.get('gpu_fwd_ms',0):.1f} lm={stages.get('lm_head_ms',0):.1f}"
    )
    if prof:
        print(
            f"{'':24} {'':18}  launches={prof.get('launches_per_tok',0):.0f}/tok "
            f"htod={prof.get('htod_mib_per_tok',0):.3f}MiB "
            f"dtoh={prof.get('dtoh_mib_per_tok',0):.3f}MiB "
            f"syncs={prof.get('syncs_per_tok',0):.0f}/tok "
            f"mirror={prof.get('mirror_ms_per_tok',0):.3f}ms "
            f"rdback={prof.get('hidden_readback_ms_per_tok',0):.3f}ms"
        )

# Write machine-readable output
output = {
    "profile_id": "larql-gpu-profile-001-post-residency-decode-bottleneck",
    "date": "2026-07-10",
    "hardware": "NVIDIA GeForce RTX 3060 (12 GB, sm_86)",
    "caveat": "RTX 3060, NOT the packet-required RTX 3090. Bottleneck ranking "
    "transfers (same sm_86); absolute tok/s and launch-overhead fraction differ.",
    "cases": summary,
}
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nwrote {OUT_JSON}", file=sys.stderr)
