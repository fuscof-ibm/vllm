#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize bench_sweep.sh results. Tabulates main vs PR across concurrencies.

Usage:
    ./bench_summarize.py bench_results/main bench_results/pr40172
"""

import json
import statistics
import sys
from pathlib import Path

METRICS = [
    ("output_throughput", "out tok/s", 2),
    ("mean_tpot_ms", "mean TPOT (ms)", 2),
    ("median_itl_ms", "med ITL (ms)", 2),
    ("p99_itl_ms", "p99 ITL (ms)", 2),
    ("median_ttft_ms", "med TTFT (ms)", 2),
]


def load_runs(dirpath: Path) -> dict[int, list[dict]]:
    runs: dict[int, list[dict]] = {}
    for p in sorted(dirpath.glob("c*_run*.json")):
        with p.open() as f:
            d = json.load(f)
        conc = int(d.get("max_concurrency") or 0)
        runs.setdefault(conc, []).append(d)
    return runs


def agg(runs: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in runs if key in r and r[key] is not None]
    if not vals:
        return float("nan"), 0.0
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def fmt(v: float, prec: int) -> str:
    return "nan" if v != v else f"{v:.{prec}f}"


def main(main_dir: str, pr_dir: str) -> None:
    main_runs = load_runs(Path(main_dir))
    pr_runs = load_runs(Path(pr_dir))
    concs = sorted(set(main_runs) | set(pr_runs))

    for key, label, prec in METRICS:
        print(f"\n## {label}")
        print(f"{'conc':>5} | {'main':>18} | {'PR':>18} | {'delta':>8}")
        print("-" * 60)
        for c in concs:
            m, ms = agg(main_runs.get(c, []), key)
            p, ps = agg(pr_runs.get(c, []), key)
            if m and p and m == m and p == p:
                delta = (p - m) / m * 100.0
                # Throughput: higher is better; latencies: lower is better.
                sign = "+" if delta >= 0 else ""
                delta_s = f"{sign}{delta:.1f}%"
            else:
                delta_s = "n/a"
            print(
                f"{c:>5} | {fmt(m, prec):>9}±{fmt(ms, prec):<7} | "
                f"{fmt(p, prec):>9}±{fmt(ps, prec):<7} | {delta_s:>8}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: bench_summarize.py <main_dir> <pr_dir>")
    main(sys.argv[1], sys.argv[2])
