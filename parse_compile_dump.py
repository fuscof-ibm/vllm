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
# Extract `'pass_config': {...}` from the engine config log line.
RE_PASS_CONFIG = re.compile(r"'pass_config':\s*\{([^}]*)\}")
RE_PC_FLAG = re.compile(r"'(fuse_\w+|enable_\w+)':\s*(True|False)")
# Extract splitting_ops list (helps explain partition counts).
RE_SPLITTING_OPS = re.compile(r"'splitting_ops':\s*\[([^\]]*)\]")


# Knowledge base for the optimization-hint section.
# Each entry: pass class name -> dict with:
#   what:    one-line description of what the fusion does
#   pattern: source file containing the FX pattern (for inspection / templating)
#   manual:  status / pointer for the hand-fused-kernel path (PR #43355-style),
#            or None if the pass is already an Inductor codegen win without a
#            standalone CUDA kernel candidate.
#   gate:    short note on what enables it (helps diagnose 0-match cases)
PASS_INFO: dict[str, dict[str, str | None]] = {
    "RMSNormQuantFusionPass": {
        "what": "fuse RMSNorm + per-tensor fp8 quant into one kernel",
        "pattern": "vllm/compilation/passes/fusion/rms_quant_fusion.py",
        "manual": "hand-fused op exists (torch.ops._C.rms_norm_static_fp8_quant); "
        "call site rewire is the model-side step",
        "gate": "pass_config.fuse_norm_quant + a quantized model",
        "flag": "fuse_norm_quant",
    },
    "ActivationQuantFusionPass": {
        "what": "fuse SiLU+Mul (or GELU) + per-tensor fp8 quant",
        "pattern": "vllm/compilation/passes/fusion/act_quant_fusion.py",
        "manual": "hand-fused op exists (torch.ops._C.silu_and_mul_quant); "
        "rewire MLP forward to call it directly",
        "gate": "pass_config.fuse_act_quant + quant'd MLP",
        "flag": "fuse_act_quant",
    },
    "AllReduceFusionPass": {
        "what": "fuse allreduce + RMSNorm (+ optional quant) for TP",
        "pattern": "vllm/compilation/passes/fusion/allreduce_rms_fusion.py",
        "manual": "needs FlashInfer fused allreduce; only fires on TP>1, "
        "Hopper/Blackwell, FlashInfer present",
        "gate": "pass_config.fuse_allreduce_rms + TP>1",
        "flag": "fuse_allreduce_rms",
    },
    "AttnQuantFusionPass": {
        "what": "fuse attention output + fp8 quant",
        "pattern": "vllm/compilation/passes/fusion/attn_quant_fusion.py",
        "manual": "candidate for a custom op that returns pre-quantized output",
        "gate": "pass_config.fuse_attn_quant + quantized model",
        "flag": "fuse_attn_quant",
    },
    "MLAAttnQuantFusionPass": {
        "what": "MLA-attention-specific output + quant fusion",
        "pattern": "vllm/compilation/passes/fusion/mla_attn_quant_fusion.py",
        "manual": "DeepSeek-V2/V3 MLA only — not relevant for hybrid Granite/Qwen",
        "gate": "MLA backend + pass_config.fuse_attn_quant",
        "flag": "fuse_attn_quant",
    },
    "MLARoPEKVCacheCatFusionPass": {
        "what": "fuse RoPE + KV cache cat for MLA",
        "pattern": "vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py",
        "manual": "MLA-only; non-MLA equivalent is RopeKVCacheFusion (PR #43355)",
        "gate": "pass_config.fuse_rope_kvcache_cat_mla + MLA backend",
        "flag": "fuse_rope_kvcache_cat_mla",
    },
    "RopeKVCacheFusionPass": {
        "what": "fuse rotary embedding + reshape_and_cache_flash",
        "pattern": "vllm/compilation/passes/fusion/rope_kvcache_fusion.py",
        "manual": "PR #43355 adds the CUDA kernel and FlashAttn opt-in; "
        "Step 2 of #43224 is to call torch.ops.vllm.fused_rope_kvcache "
        "directly from model attention forward and delete the FX pass",
        "gate": "pass_config.fuse_rope_kvcache + custom rotary_embedding op + "
        "splitting_ops=[] (or use_inductor_graph_partition=True)",
        "flag": "fuse_rope_kvcache",
    },
    "QKNormRoPEFusionPass": {
        "what": "fuse Q/K RMSNorm + RoPE",
        "pattern": "vllm/compilation/passes/fusion/qk_norm_rope_fusion.py",
        "manual": "candidate for a single Q/K-norm+RoPE custom op; "
        "only fires when the model applies QK-norm",
        "gate": "model with QK-norm (e.g. Qwen3) + custom rotary_embedding",
        "flag": None,
    },
    "SequenceParallelismPass": {
        "what": "shard residual+RMSNorm across TP ranks (sequence parallel)",
        "pattern": "vllm/compilation/passes/fusion/sequence_parallelism.py",
        "manual": "structural, not a kernel — would require rewriting the "
        "TP comm pattern in the model. Lower priority than kernel fusions.",
        "gate": "pass_config.enable_sp + dense (non-MoE) + TP>1",
        "flag": "enable_sp",
    },
    "AsyncTPPass": {
        "what": "overlap GEMM with TP communication (async TP)",
        "pattern": "vllm/compilation/passes/fusion/collective_fusion.py",
        "manual": "structural; needs torch's async-TP support and is "
        "hardware-specific. Not a candidate for a single hand-fused kernel.",
        "gate": "pass_config.fuse_gemm_comms + dense + TP>1",
        "flag": "fuse_gemm_comms",
    },
    "RocmAiterRMSNormQuantFusionPass": {
        "what": "ROCm/AITER variant of RMSNorm+quant fusion",
        "pattern": "vllm/compilation/passes/fusion/rocm_aiter_fusion.py",
        "manual": "ROCm only — already routes to a hand-tuned AITER kernel",
        "gate": "is_rocm() + AITER enabled",
    },
    "RocmAiterAllReduceFusionPass": {
        "what": "ROCm/AITER variant of allreduce+RMSNorm fusion",
        "pattern": "vllm/compilation/passes/fusion/allreduce_rms_fusion.py",
        "manual": "ROCm only — already a hand-tuned AITER kernel",
        "gate": "is_rocm() + AITER + TP>1",
    },
    "RocmAiterSiluMulFp8GroupQuantFusionPass": {
        "what": "ROCm/AITER fused SiLU+Mul+per-group fp8 quant",
        "pattern": "vllm/compilation/passes/fusion/rocm_aiter_fusion.py",
        "manual": "ROCm only — already hand-tuned",
        "gate": "is_rocm() + AITER + per-group fp8",
    },
    "RocmAiterTritonAddRMSNormPadFusionPass": {
        "what": "ROCm/AITER add+RMSNorm with hidden-size padding for gpt-oss",
        "pattern": "vllm/compilation/passes/fusion/rocm_aiter_fusion.py",
        "manual": "ROCm + hidden_size==2880 only",
        "gate": "AITER + gpt-oss-shaped models",
    },
    "MLADualRMSNormFusionPass": {
        "what": "MLA-specific dual RMSNorm fusion (Q/K-norm)",
        "pattern": "vllm/compilation/passes/fusion/rocm_aiter_fusion.py",
        "manual": "MLA + AITER — ROCm-only hand-tuned",
        "gate": "AITER fused_qk_rmsnorm available + MLA",
    },
}


def hint_priority(name: str, matches: int) -> tuple[str, int]:
    """Return (priority_label, sort_key) for a pass given its match count.

    Heuristic: more matches → more leverage. ROCm-specific passes are deprioritized
    on a CUDA dump (they only fire on AMD anyway).
    """
    if matches == 0:
        return ("DID NOT FIRE", -1)
    rocm_only = name.startswith("RocmAiter") or name == "MLADualRMSNormFusionPass"
    if rocm_only:
        return ("ROCM-ONLY", matches)
    if matches >= 32:
        return ("HIGH", matches + 10_000)
    if matches >= 8:
        return ("MEDIUM", matches + 1_000)
    return ("LOW", matches)


def emit_hints(
    table: dict[str, int],
    pass_config: dict[str, bool],
    splitting_ops: list[str],
    pass_config_seen: bool = False,
) -> None:
    """Print actionable hints based on which fusion passes fired."""
    print()
    print("=== optimization hints (manual hand-fused-kernel candidates)")

    # Surface the pass_config gates the engine actually used, if we found them.
    if pass_config:
        on = sorted(k for k, v in pass_config.items() if v)
        off = sorted(k for k, v in pass_config.items() if not v)
        print(f"  pass_config gates ON  : {on or '(none)'}")
        print(f"  pass_config gates OFF : {off or '(none)'}")
    elif pass_config_seen:
        print("  pass_config: {} (all defaults — no fusion gates overridden)")

    # Splitting ops give a sense of how many partitions exist and which kernels
    # the graph is broken at. Useful when the user is reasoning about manual
    # fusion *across* an attention/SSM boundary (which the compiler can't do
    # without inductor_graph_partition).
    if splitting_ops:
        print(
            f"  graph split at {len(splitting_ops)} op(s): "
            f"{', '.join(splitting_ops[:6])}" + (" …" if len(splitting_ops) > 6 else "")
        )

    if not table:
        print()
        print("  no fusion-pass matches were recorded.")
        # Try to explain why.
        defaults_only = pass_config_seen and not pass_config
        all_off = pass_config and not any(pass_config.values())
        if defaults_only or all_off:
            why = (
                "pass_config is empty → all gates at their defaults"
                if defaults_only
                else "ALL pass_config flags are False"
            )
            print(f"  → {why}. With a non-quantized,")
            print("    TP=1 model, no kernel fusions can fire. To exercise them:")
            print("      • use an FP8/INT8 quantized variant (enables the *quant")
            print("        fusions), or")
            print("      • run with TP>1 + FlashInfer (enables AllReduceFusion +")
            print("        SequenceParallelism), or")
            print("      • set pass_config.fuse_rope_kvcache=True and pass")
            print("        splitting_ops=[] (PR #43355 path) on a model with the")
            print("        custom rotary_embedding op.")
        elif not pass_config_seen:
            print("  (could not parse pass_config from log — was the run started")
            print("   with VLLM_LOGGING_LEVEL=DEBUG and the engine config printed?)")
        else:
            print("  → some gates are on but no patterns matched. The model may")
            print("    not contain the patterns those passes look for; inspect")
            print("    the per-pass pattern files listed in PASS_INFO.")
        # Even with no matches, the IR cleanup passes ran — point at them.
        print()
        print("  what compile DID do on this run (from per-pass timing):")
        print("    Inductor codegen + CUDA-graph capture only. The compile vs.")
        print("    nocompile delta in your bench results is purely Inductor's")
        print("    work (fused elementwise/contiguity ops it generates), not")
        print("    any of the cataloged fusion passes. Manual optimization for")
        print("    this model should target Mamba/SSM kernels and the existing")
        print("    custom ops, not the FX-pass catalog.")
        return

    # Annotate every observed pass with priority + info; sort high to low.
    annotated = []
    for name, n in table.items():
        info = PASS_INFO.get(name, {})
        prio, sort_key = hint_priority(name, n)
        annotated.append((sort_key, prio, name, n, info))
    annotated.sort(key=lambda x: -x[0])

    for _, prio, name, n, info in annotated:
        print(f"\n  [{prio}]  {name}  matches={n}")
        if info:
            if what := info.get("what"):
                print(f"    what    : {what}")
            if manual := info.get("manual"):
                print(f"    manual  : {manual}")
            if pattern := info.get("pattern"):
                print(f"    pattern : {pattern}")
            if gate := info.get("gate"):
                print(f"    gate    : {gate}")
        else:
            print("    (no entry in PASS_INFO — add one to extend hints)")

    # Footer: which entries in PASS_INFO did not appear, with explanation
    # using the pass_config we parsed from the log when possible.
    unseen = sorted(set(PASS_INFO) - set(table))
    if unseen:
        print()
        print("  passes that did NOT fire on this run:")
        for name in unseen:
            info = PASS_INFO[name]
            flag = info.get("flag")
            if flag and pass_config and flag in pass_config:
                state = "ON" if pass_config[flag] else "OFF"
                reason = f"gate {flag}={state} → " + (
                    "model didn't match patterns"
                    if pass_config[flag]
                    else "fusion disabled in this run"
                )
            else:
                reason = f"gate: {info.get('gate', '?')}"
            print(f"    - {name:36s} {reason}")
        print("    → 0-match passes whose gate is ON are the highest-leverage")
        print("      'missed opportunity' candidates. 0-match with gate OFF")
        print("      means you'd need to change the run config (quant, TP,")
        print("      pass_config flags) before that fusion is available.")

    print()
    print("  template for manual fusion (per PR #43355 / issue #43224):")
    print("    1) write the CUDA kernel in csrc/<fusion>_kernels.cu")
    print("    2) bind it as torch.ops.vllm.<fused_op>")
    print("    3) call the fused op directly from model code, replacing the")
    print("       unfused two-call sequence")
    print("    4) drop the FX pass once every model has been rewired")


def parse_log(path: Path) -> dict:
    out = {
        "compiles": [],  # (range, seconds)
        "cache_loads": [],  # (range, seconds)
        "pass_times_ms": defaultdict(list),  # PassName -> [ms, ...]
        "pass_replaced": [],  # (file_stem, count)
        "match_table": None,  # last seen dict
        "profiling_s": None,
        "total_compile_s": None,
        "pass_config": {},  # flag -> bool
        "pass_config_seen": False,  # True if 'pass_config': {...} was logged
        "splitting_ops": [],  # list of attention/SSM op names
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
    if m := RE_PASS_CONFIG.search(text):
        out["pass_config_seen"] = True
        for fm in RE_PC_FLAG.finditer(m.group(1)):
            out["pass_config"][fm.group(1)] = fm.group(2) == "True"
    if m := RE_SPLITTING_OPS.search(text):
        out["splitting_ops"] = [
            s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()
        ]
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

    emit_hints(
        table,
        log["pass_config"],
        log["splitting_ops"],
        log["pass_config_seen"],
    )

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
