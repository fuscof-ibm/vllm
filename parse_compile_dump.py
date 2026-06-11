#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize what torch.compile did to a vLLM model.

Reads the server log + debug-dump directory produced by:
    -O.debug_dump_path=<dir> -O.compile_cache_save_format=unpacked
    VLLM_LOGGING_LEVEL=DEBUG

and prints:
  - per-partition compile times and cache load times
  - per-pass fusion match counts
  - total compile time and number of partitions

Usage:
    ./parse_compile_dump.py <server_log> <dump_dir>
"""

from __future__ import annotations

import contextlib
import sys
from collections import defaultdict
from pathlib import Path

import regex as re

# vLLM logs are like: "INFO  09-30 ... [backends.py:393] Compiling a graph ..."
# We tolerate any prefix and just key on the message text.
RE_COMPILE = re.compile(r"Compiling a graph for compile range (\S+) takes ([\d.]+) s")
RE_LOAD = re.compile(
    r"Directly load the compiled graph\(s\) for compile range (\S+) "
    r"from the cache, took ([\d.]+) s"
)
RE_PASS_REPLACED = re.compile(
    r"\[([A-Za-z_][A-Za-z0-9_]*)\.py:\d+\] Replaced (\d+) patterns"
)
RE_PASS_TIME = re.compile(r"([A-Z][A-Za-z]+Pass) completed in ([\d.]+) ms")
RE_MATCH_TABLE = re.compile(r"fusion pass matches: (\{[^}]*\})")
RE_PROFILING = re.compile(r"Initial profiling/warmup run took ([\d.]+) s")
RE_TOTAL = re.compile(r"torch\.compile takes ([\d.]+) s")
RE_NUM_GRAPHS = re.compile(r"num_graphs[= ]+(\d+)")


def parse_log(path: Path) -> dict:
    out = {
        "compiles": [],  # (range, seconds)
        "cache_loads": [],  # (range, seconds)
        "pass_times_ms": defaultdict(list),  # PassName -> [ms, ...]
        "pass_replaced": [],  # (file_stem, count)
        "match_table": None,  # last seen dict
        "profiling_s": None,
        "total_compile_s": None,
    }
    text = path.read_text(errors="replace")
    for m in RE_COMPILE.finditer(text):
        out["compiles"].append((m.group(1), float(m.group(2))))
    for m in RE_LOAD.finditer(text):
        out["cache_loads"].append((m.group(1), float(m.group(2))))
    for m in RE_PASS_REPLACED.finditer(text):
        out["pass_replaced"].append((m.group(1), int(m.group(2))))
    for m in RE_PASS_TIME.finditer(text):
        out["pass_times_ms"][m.group(1)].append(float(m.group(2)))
    for m in RE_MATCH_TABLE.finditer(text):
        # last one wins — that's the final summary
        with contextlib.suppress(Exception):
            out["match_table"] = eval(m.group(1), {"__builtins__": {}}, {})
    if m := RE_PROFILING.search(text):
        out["profiling_s"] = float(m.group(1))
    if m := RE_TOTAL.search(text):
        out["total_compile_s"] = float(m.group(1))
    return out


def scan_dump(path: Path) -> dict:
    if not path.exists():
        return {"patterns_files": {}, "fx_files": 0, "output_code_files": 0}
    pattern_files: dict[str, int] = defaultdict(int)
    for p in path.rglob("patterns.*.py"):
        # patterns.<PassName>.<i>.py
        parts = p.name.split(".")
        if len(parts) >= 3:
            pattern_files[parts[1]] += 1
    fx_files = sum(1 for _ in path.rglob("*.fx*"))
    output_code = sum(1 for _ in path.rglob("output_code*.py"))
    return {
        "patterns_files": dict(pattern_files),
        "fx_files": fx_files,
        "output_code_files": output_code,
    }


def fmt_table(rows: list[tuple], headers: tuple) -> str:
    if not rows:
        return "  (none)"
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]
    lines = []
    for i, row in enumerate([headers] + rows):
        lines.append("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, widths)))
        if i == 0:
            lines.append("  " + "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    log_path = Path(argv[1])
    dump_path = Path(argv[2])
    if not log_path.exists():
        print(f"ERR: log not found: {log_path}", file=sys.stderr)
        return 1

    log = parse_log(log_path)
    dump = scan_dump(dump_path)

    print(f"=== compile summary: {log_path}")
    if log["total_compile_s"] is not None:
        print(f"  total torch.compile wall time : {log['total_compile_s']:.2f} s")
    if log["profiling_s"] is not None:
        print(f"  profiling/warmup run          : {log['profiling_s']:.2f} s")

    n_partitions_per_range: dict[str, int] = defaultdict(int)
    for rng, _ in log["compiles"]:
        n_partitions_per_range[rng] += 1
    for rng, _ in log["cache_loads"]:
        n_partitions_per_range[rng] += 1
    if n_partitions_per_range:
        first = next(iter(n_partitions_per_range))
        print(
            f"  partitions per shape range    : {n_partitions_per_range[first]} "
            f"(across {len(n_partitions_per_range)} range(s))"
        )

    print()
    print("=== compile time by shape range")
    by_range: dict[str, float] = defaultdict(float)
    for rng, sec in log["compiles"]:
        by_range[rng] += sec
    rows = sorted(
        ((rng, f"{s:.2f}") for rng, s in by_range.items()), key=lambda r: -float(r[1])
    )
    print(fmt_table(rows, ("range", "compile_s")))

    if log["cache_loads"]:
        print()
        print("=== cache loads (already-compiled graphs)")
        rows = [(rng, f"{s:.3f}") for rng, s in log["cache_loads"][:20]]
        print(fmt_table(rows, ("range", "load_s")))

    print()
    print("=== fusion-pass matches (final match_table)")
    table = log["match_table"] or {}
    if not table and log["pass_replaced"]:
        # fall back to the per-pass "Replaced N" lines
        agg: dict[str, int] = defaultdict(int)
        for stem, n in log["pass_replaced"]:
            agg[stem] += n
        table = dict(agg)
    rows = sorted(((k, v) for k, v in table.items()), key=lambda r: -r[1])
    print(fmt_table(rows, ("pass", "matches")))

    if log["pass_times_ms"]:
        print()
        print("=== per-pass timing (sum across all invocations)")
        rows = []
        for name, samples in log["pass_times_ms"].items():
            rows.append(
                (name, len(samples), f"{sum(samples):.1f}", f"{max(samples):.1f}")
            )
        rows.sort(key=lambda r: -float(r[2]))
        print(fmt_table(rows, ("pass", "n", "total_ms", "max_ms")))

    print()
    print(f"=== dump dir: {dump_path}")
    print(f"  output_code.py files (compiled subgraphs): {dump['output_code_files']}")
    print(f"  fx graph dumps: {dump['fx_files']}")
    if dump["patterns_files"]:
        rows = sorted(dump["patterns_files"].items(), key=lambda r: -r[1])
        print("  pattern dumps per pass:")
        print(fmt_table(rows, ("pass", "files")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
