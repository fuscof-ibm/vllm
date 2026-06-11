#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare eager / nocompile / full runs from bench_compile_sweep.sh.

Reads the c<N>.json files written by `vllm bench serve` under each config
subdir and prints a 3-way table per metric, with speedup ratios:

  full / nocompile  : torch.compile contribution (compile vs CUDA-graphs-only)
  nocompile / eager : CUDA-graph contribution
  full / eager      : combined contribution

Throughput metrics: ratio > 1.0 is better. Latency metrics: ratio < 1.0 is
better — the "→" arrow points to the winning side either way.

Usage:
    ./bench_compile_compare.py                                  # auto-detect
    ./bench_compile_compare.py bench_results/compile_sweep/granite-4.0-h-small
    ./bench_compile_compare.py <eager_dir> <nocompile_dir> <full_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# (json key, display label, decimals, higher_is_better)
METRICS = [
    ("output_throughput", "out tok/s", 2, True),
    ("request_throughput", "req/s", 3, True),
    ("mean_tpot_ms", "mean TPOT ms", 2, False),
    ("median_itl_ms", "med ITL ms", 2, False),
    ("p99_itl_ms", "p99 ITL ms", 2, False),
    ("median_ttft_ms", "med TTFT ms", 2, False),
    ("mean_e2el_ms", "mean E2E ms", 1, False),
]

CONFIGS = ("eager", "nocompile", "full")


def load_runs(dirpath: Path) -> dict[int, dict]:
    """Map concurrency -> result dict for each c<N>.json in dir."""
    runs: dict[int, dict] = {}
    if not dirpath.exists():
        return runs
    for p in sorted(dirpath.glob("c*.json")):
        # skip warmup or anything that isn't c<int>.json
        stem = p.stem
        if not stem.startswith("c") or not stem[1:].isdigit():
            continue
        try:
            with p.open() as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        conc = int(d.get("max_concurrency") or stem[1:])
        runs[conc] = d
    return runs


def fmt(v, prec: int) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:.{prec}f}"


def ratio(num, den, higher_is_better: bool) -> str:
    """Return ratio string with arrow indicating which side wins."""
    if num is None or den is None or den == 0:
        return "  n/a "
    r = num / den
    # For higher-is-better metrics, num>den means num wins.
    # For lower-is-better, num<den means num wins.
    better_num = r > 1.0 if higher_is_better else r < 1.0
    arrow = "→" if better_num else "←"
    # Always print as multiplicative factor of the larger side.
    factor = max(r, 1.0 / r) if r > 0 else float("nan")
    return f"{arrow}{factor:.2f}×"


def header(title: str, higher_is_better: bool) -> None:
    direction = "higher is better" if higher_is_better else "lower is better"
    print(f"\n## {title}  [{direction}]")
    cols = ("conc", *CONFIGS, "noc/eag", "full/noc", "full/eag")
    widths = (5, 11, 11, 11, 9, 9, 9)
    print(" | ".join(c.rjust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))


def main(argv: list[str]) -> int:
    dirs: dict[str, Path]
    if len(argv) == 1:
        # Auto-detect a single sweep root under bench_results/compile_sweep/.
        root = Path("bench_results/compile_sweep")
        candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
        if len(candidates) != 1:
            print(
                "Could not auto-detect sweep dir. Pass either:\n"
                "  one root: bench_results/compile_sweep/<model_tag>\n"
                "  three dirs: <eager> <nocompile> <full>",
                file=sys.stderr,
            )
            return 2
        sweep_root = candidates[0]
        dirs = {cfg: sweep_root / cfg for cfg in CONFIGS}
    elif len(argv) == 2:
        sweep_root = Path(argv[1])
        dirs = {cfg: sweep_root / cfg for cfg in CONFIGS}
    elif len(argv) == 4:
        dirs = dict(zip(CONFIGS, map(Path, argv[1:4])))
    else:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    runs = {cfg: load_runs(d) for cfg, d in dirs.items()}
    for cfg, d in dirs.items():
        n = len(runs[cfg])
        marker = "" if n else "  (empty!)"
        print(f"  {cfg:9s} {d}  -> {n} concurrency point(s){marker}")

    concs = sorted(set().union(*(set(r) for r in runs.values())))
    if not concs:
        print(
            "ERR: no c<N>.json results found under any of the three dirs.",
            file=sys.stderr,
        )
        return 1

    for key, label, prec, higher_is_better in METRICS:
        # Skip metric if no run anywhere reports it.
        if not any(key in r for cfg in CONFIGS for r in runs[cfg].values()):
            continue
        header(label, higher_is_better)
        for c in concs:
            vals = {cfg: runs[cfg].get(c, {}).get(key) for cfg in CONFIGS}
            row = [str(c).rjust(5)]
            row += [fmt(vals[cfg], prec).rjust(11) for cfg in CONFIGS]
            # noc/eag, full/noc, full/eag — always num/den ordering;
            # ratio() flips the visual direction with an arrow.
            row.append(
                ratio(vals["nocompile"], vals["eager"], higher_is_better).rjust(9)
            )
            row.append(
                ratio(vals["full"], vals["nocompile"], higher_is_better).rjust(9)
            )
            row.append(ratio(vals["full"], vals["eager"], higher_is_better).rjust(9))
            print(" | ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
