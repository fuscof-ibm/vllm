#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot APC sweep results: per-model 4-panel dashboards plus a relative
throughput summary across models."""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("bench_results/apc_sweep")
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

# (display name, dir slug)
MODELS = [
    ("Nemotron-3-Super-120B-A12B-NVFP4",
     "NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"),
    ("Qwen3.5-9B", "Qwen3.5-9B"),
    ("Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B"),  # dir says 3.6 but it's 3.5
]

# (config label, dir prefix, color, marker)
CONFIGS = [
    ("c68c55d4 APC=off", "main_apc_off", "tab:blue", "o"),
    ("c68c55d4 APC=on", "main_apc_on", "tab:orange", "s"),
    ("b730c4635 APC=on", "b730c4635_apc_on", "tab:green", "^"),
]

# (json key, panel title, y-label, higher_is_better)
METRICS = [
    ("output_throughput", "Output throughput", "tokens/s", True),
    ("mean_tpot_ms", "Mean TPOT", "ms", False),
    ("median_itl_ms", "Median ITL", "ms", False),
    ("p99_itl_ms", "P99 ITL", "ms", False),
]


def load(dirpath: Path) -> dict[int, list[dict]]:
    runs: dict[int, list[dict]] = {}
    if not dirpath.exists():
        return runs
    for p in sorted(dirpath.glob("c*_run*.json")):
        with p.open() as f:
            d = json.load(f)
        conc = int(d.get("max_concurrency") or 0)
        runs.setdefault(conc, []).append(d)
    return runs


def agg(runs: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in runs if r.get(key) is not None]
    if not vals:
        return float("nan"), 0.0
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def collect(model_slug: str) -> dict[str, dict[int, tuple[float, float]]]:
    """Return {config_label: {conc: (mean, std)}} for every metric key
    via nested call (we call this per-metric)."""
    raise NotImplementedError  # placeholder, see series() below


def series(model_slug: str, key: str
           ) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """{config_label: (concs, means, stds)} for a given metric key."""
    out = {}
    for label, slug, _, _ in CONFIGS:
        runs = load(ROOT / f"{slug}_{model_slug}")
        if not runs:
            continue
        concs = sorted(runs)
        means, stds = [], []
        for c in concs:
            m, s = agg(runs[c], key)
            means.append(m)
            stds.append(s)
        out[label] = (concs, means, stds)
    return out


def plot_model_dashboard(display: str, slug: str) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"APC sweep — {display}", fontsize=14, fontweight="bold")

    color_map = {label: c for label, _, c, _ in CONFIGS}
    marker_map = {label: m for label, _, _, m in CONFIGS}

    for ax, (key, title, ylab, _) in zip(axes.flat, METRICS):
        data = series(slug, key)
        for label in [c[0] for c in CONFIGS]:
            if label not in data:
                continue
            concs, means, stds = data[label]
            ax.errorbar(
                concs, means, yerr=stds,
                label=label, marker=marker_map[label],
                color=color_map[label], capsize=3, linewidth=1.6,
            )
        ax.set_title(title)
        ax.set_xlabel("concurrency")
        ax.set_ylabel(ylab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "4", "8", "16", "32"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / f"dashboard_{slug}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_throughput_relative() -> Path:
    """For each model, plot throughput of APC-on configs as % of APC=off."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        "Output throughput relative to APC=off (higher is better)",
        fontsize=13, fontweight="bold")

    for ax, (display, slug) in zip(axes, MODELS):
        data = series(slug, "output_throughput")
        if "c68c55d4 APC=off" not in data:
            continue
        base_concs, base_means, _ = data["c68c55d4 APC=off"]
        base = dict(zip(base_concs, base_means))
        for label, _, color, marker in CONFIGS:
            if label not in data or label == "c68c55d4 APC=off":
                continue
            concs, means, _ = data[label]
            rel = [m / base[c] * 100.0 if c in base and base[c] else
                   float("nan") for c, m in zip(concs, means)]
            ax.plot(concs, rel, label=label, marker=marker,
                    color=color, linewidth=1.8)
        ax.axhline(100, color="tab:blue", linestyle="--",
                   linewidth=1, alpha=0.7, label="APC=off baseline")
        ax.set_title(display, fontsize=10)
        ax.set_xlabel("concurrency")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "4", "8", "16", "32"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("throughput (% of APC=off)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "throughput_relative.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_tpot_relative() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle(
        "Mean TPOT relative to APC=off (lower is better)",
        fontsize=13, fontweight="bold")

    for ax, (display, slug) in zip(axes, MODELS):
        data = series(slug, "mean_tpot_ms")
        if "c68c55d4 APC=off" not in data:
            continue
        base_concs, base_means, _ = data["c68c55d4 APC=off"]
        base = dict(zip(base_concs, base_means))
        for label, _, color, marker in CONFIGS:
            if label not in data or label == "c68c55d4 APC=off":
                continue
            concs, means, _ = data[label]
            rel = [m / base[c] * 100.0 if c in base and base[c] else
                   float("nan") for c, m in zip(concs, means)]
            ax.plot(concs, rel, label=label, marker=marker,
                    color=color, linewidth=1.8)
        ax.axhline(100, color="tab:blue", linestyle="--",
                   linewidth=1, alpha=0.7, label="APC=off baseline")
        ax.set_title(display, fontsize=10)
        ax.set_xlabel("concurrency")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "4", "8", "16", "32"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("mean TPOT (% of APC=off)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "tpot_relative.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_throughput_delta_bars(display: str, slug: str) -> Path:
    """Per-model grouped bar chart: c68c55d4 APC=off vs b730c4635
    APC=on absolute output throughput per concurrency. Annotation on
    the APC=on bar shows the residual gap to APC=off."""
    data = series(slug, "output_throughput")
    needed = ["c68c55d4 APC=off", "b730c4635 APC=on"]
    if any(label not in data for label in needed):
        return Path()

    color_map = {label: c for label, _, c, _ in CONFIGS}
    off_concs, off_means, _ = data["c68c55d4 APC=off"]
    off = dict(zip(off_concs, off_means))
    pr_on_concs, pr_on_means, _ = data["b730c4635 APC=on"]
    pr_on = dict(zip(pr_on_concs, pr_on_means))
    concs = sorted(set(off) & set(pr_on))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = list(range(len(concs)))
    width = 0.38

    off_vals = [off[c] for c in concs]
    pr_vals = [pr_on[c] for c in concs]

    ax.bar([xi - width / 2 for xi in x], off_vals, width,
           label="c68c55d4 APC=off", color=color_map["c68c55d4 APC=off"],
           edgecolor="black", linewidth=0.6)
    pr_bars = ax.bar([xi + width / 2 for xi in x], pr_vals, width,
                     label="b730c4635 APC=on",
                     color=color_map["b730c4635 APC=on"],
                     edgecolor="black", linewidth=0.6)

    for bar, c in zip(pr_bars, concs):
        d = (pr_on[c] - off[c]) / off[c] * 100.0
        color = "darkred" if d < 0 else "darkgreen"
        ax.annotate(f"{d:+.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2,
                        bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concs])
    ax.set_xlabel("concurrency")
    ax.set_ylabel("output throughput (tokens/s)")
    ax.set_title(f"APC residual overhead — {display}\n"
                 "Output throughput: APC=off (c68c55d4) "
                 "vs APC=on (b730c4635)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    top = max(off_vals + pr_vals) * 1.18
    ax.set_ylim(0, top)

    fig.tight_layout()
    out = OUT / f"throughput_delta_{slug}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    saved = []
    for display, slug in MODELS:
        saved.append(plot_model_dashboard(display, slug))
    saved.append(plot_throughput_relative())
    saved.append(plot_tpot_relative())
    for display, slug in MODELS:
        saved.append(plot_throughput_delta_bars(display, slug))
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
