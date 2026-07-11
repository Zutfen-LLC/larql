#!/usr/bin/env python3
"""B4-CORRECTION D: pure benchmark aggregation for LARQL-GPU-B4.

The committed benchmark script (`bench_b4.sh`) emits one machine-readable
JSON record per (mode, repetition). This module aggregates those raw records
into per-mode median / MAD / min / max WITHOUT hand-entered values, and labels
each result as instrumented or uninstrumented.

The math is intentionally tiny and pure so it is unit-testable on its own
(``python3 scripts/b4_aggregate.py --selftest``). The shell script pipes raw
JSONL through ``aggregate_stdin``.

B4-CORRECTION D methodology:
  * performance source-of-truth runs are UNINSTRUMENTED (no LARQL_GPU_PROFILE)
  * a separate INSTRUMENTED run supplies structural counters only
  * medians + MAD are computed from the raw per-repetition values
"""
from __future__ import annotations

import json
import math
import sys
from typing import Iterable, Sequence


def median(xs: Sequence[float]) -> float:
    """Median of a non-empty sequence (mean of the two middle values for an
    even-length sequence). Pure; raises ``ValueError`` on empty input."""
    if not xs:
        raise ValueError("median of empty sequence")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (float(s[mid - 1]) + float(s[mid])) / 2.0


def mad(xs: Sequence[float]) -> float:
    """Median absolute deviation about the median. Pure; ``ValueError`` on
    empty input. The B4 noise metric (MAD ≤ ~0.07 ms in prior runs)."""
    if not xs:
        raise ValueError("mad of empty sequence")
    m = median(xs)
    return median([abs(x - m) for x in xs])


def aggregate_values(xs: Sequence[float]) -> dict:
    """Aggregate a non-empty list of per-repetition values into the record
    shape committed to the bench JSON. Pure."""
    if not xs:
        raise ValueError("aggregate_values of empty sequence")
    xs_list = sorted(float(x) for x in xs)
    med = median(xs_list)
    return {
        "n": len(xs_list),
        "raw": xs_list,
        "min": xs_list[0],
        "max": xs_list[-1],
        "mean": sum(xs_list) / len(xs_list),
        "median": med,
        "mad": mad(xs_list),
    }


def aggregate_runs(records: Iterable[dict]) -> dict:
    """Group records by ``(mode, instrumented)`` and aggregate the ``p50_ms``
    and ``mean_ms`` fields. Records missing those fields or carrying a
    ``"failed": true`` flag are EXCLUDED from the value aggregation but counted
    in ``missing``/``failed`` so the report can show dropped reps honestly.
    Pure (no I/O)."""
    groups: dict[tuple[str, bool], dict[str, list[float]]] = {}
    failed = 0
    missing = 0
    for r in records:
        if r.get("failed"):
            failed += 1
            continue
        key = (r.get("mode", "?"), bool(r.get("instrumented", False)))
        bucket = groups.setdefault(key, {"p50_ms": [], "mean_ms": []})
        p50 = r.get("p50_ms")
        mean = r.get("mean_ms")
        if p50 is None or mean is None:
            missing += 1
            continue
        bucket["p50_ms"].append(float(p50))
        bucket["mean_ms"].append(float(mean))
    out = {}
    for (mode, instrumented), bucket in sorted(groups.items()):
        out[mode] = {
            "instrumented": instrumented,
            "p50_ms": aggregate_values(bucket["p50_ms"]),
            "mean_ms": aggregate_values(bucket["mean_ms"]),
        }
    return {"modes": out, "failed": failed, "missing": missing}


def aggregate_stdin() -> dict:
    """Read JSONL records from stdin, aggregate, return the result dict."""
    records = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            print(f"# b4_aggregate: skipping unparseable line: {exc}", file=sys.stderr)
            missing_count = 0
    return aggregate_runs(records)


def _selftest() -> int:
    """Pure assertions for median / MAD / failed-missing handling /
    instrumented-vs-uninstrumented labeling. Returns nonzero on failure."""
    failures = 0

    def check(name, got, want):
        nonlocal failures
        if got != want:
            print(f"  FAIL {name}: got {got!r} want {want!r}")
            failures += 1
        else:
            print(f"  ok   {name}: {got!r}")

    # median: odd / even / single
    check("median odd", median([3.0, 1.0, 2.0]), 2.0)
    check("median even", median([4.0, 1.0, 3.0, 2.0]), 2.5)
    check("median single", median([7.0]), 7.0)
    # MAD: deviations 1, 0, 1 → median 1
    check("mad basic", mad([2.0, 3.0, 4.0]), 1.0)
    check("mad single", mad([5.0]), 0.0)
    try:
        median([])
        print("  FAIL median([]) did not raise")
        failures += 1
    except ValueError:
        print("  ok   median([]) raised")

    # aggregate_values shape
    agg = aggregate_values([108.0, 109.0, 107.0])
    check("agg median", agg["median"], 108.0)
    check("agg min", agg["min"], 107.0)
    check("agg max", agg["max"], 109.0)
    check("agg mean", round(agg["mean"], 4), round(108.0, 4))
    check("agg mad", agg["mad"], 1.0)
    check("agg n", agg["n"], 3)

    # failed / missing handling + instrumented labeling
    records = [
        {"mode": "B4A", "instrumented": False, "p50_ms": 100.0, "mean_ms": 100.5},
        {"mode": "B4A", "instrumented": False, "p50_ms": 102.0, "mean_ms": 102.5},
        {"mode": "B4A", "instrumented": False, "failed": True},
        {"mode": "B4A", "instrumented": False, "p50_ms": None},
        {"mode": "BaselineA", "instrumented": True, "p50_ms": 200.0, "mean_ms": 200.0},
    ]
    res = aggregate_runs(records)
    check("failed count", res["failed"], 1)
    check("missing count", res["missing"], 1)
    check("B4A instrumented", res["modes"]["B4A"]["instrumented"], False)
    check("B4A p50 median", res["modes"]["B4A"]["p50_ms"]["median"], 101.0)
    check(
        "BaselineA instrumented",
        res["modes"]["BaselineA"]["instrumented"],
        True,
    )

    if failures:
        print(f"SELFTEST FAILED: {failures} check(s)")
        return 1
    print("SELFTEST OK")
    return 0


def main(argv: Sequence[str]) -> int:
    if "--selftest" in argv:
        return _selftest()
    result = aggregate_stdin()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
