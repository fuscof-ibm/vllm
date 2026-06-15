#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize bench_sweep.sh results. Tabulates baseline vs comparison across
concurrencies.

Usage:
    ./bench_summarize.py bench_results/main bench_results/pr40172
    ./bench_summarize.py base/ comp/ --labels "APC=off:APC=on"
"""

import argparse
import json
import statistics
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


def run(main_dir: str, pr_dir: str, main_label: str, pr_label: str) -> None:
    main_runs = load_runs(Path(main_dir))
    pr_runs = load_runs(Path(pr_dir))
    concs = sorted(set(main_runs) | set(pr_runs))

    # Value cells are "  9.99±  9.99" — 17 chars. Pad header to match,
    # but widen if a custom label is longer.
    col_w = max(17, len(main_label), len(pr_label))

    for key, label, prec in METRICS:
        print(f"\n## {label}")
        print(
            f"{'conc':>5} | {main_label:>{col_w}} | {pr_label:>{col_w}} | {'delta':>8}"
        )
        print("-" * (5 + 3 + col_w + 3 + col_w + 3 + 8))
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
            m_cell = f"{fmt(m, prec):>9}±{fmt(ms, prec):<7}"
            p_cell = f"{fmt(p, prec):>9}±{fmt(ps, prec):<7}"
            print(f"{c:>5} | {m_cell:>{col_w}} | {p_cell:>{col_w}} | {delta_s:>8}")


def parse_labels(s: str) -> tuple[str, str]:
    parts = s.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            "expected 'BASELINE_LABEL:COMPARISON_LABEL' (colon-separated)"
        )
    return parts[0], parts[1]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("main_dir", help="baseline results dir")
    p.add_argument("pr_dir", help="comparison results dir")
    p.add_argument(
        "--labels",
        type=parse_labels,
        default=("main", "PR"),
        help="colon-separated column labels, e.g. 'APC=off:APC=on'",
    )
    args = p.parse_args()
    run(args.main_dir, args.pr_dir, args.labels[0], args.labels[1])
