<!-- markdownlint-disable -->

## Purpose

`_copy_mamba_state_block` copies temporal and conv states with 1-byte loads and stores, leaving most of the HBM bandwidth on the floor. Temporal states are 20–30× the size of the conv states depending on the model, so the copy is dominated by the temporal path.

This PR switches the temporal-state copy to `uint64` loads/stores (8× wider transactions per instruction). To make every issued address safe, `MambaSpecDecodeGPUContext` now asserts at setup that the temporal-state base pointer and per-block stride are both 8-byte aligned; if either fails, we fall back to the existing 1-byte path rather than issuing misaligned wide accesses.

Net effect: the kernel is now HBM-bandwidth-bound instead of instruction-issue-bound, and the per-decode-step latency contributed by this copy drops by ~5–6× in the small-batch regime that matters most for serving.

## Test Plan

### Microbenchmark

Developed a microbenchmark harness `benchmarks/bench_copy_mamba_state_block.py`,
which isolates `_copy_mamba_state_block` behind a minimal Triton wrapper launched at
the production grid (num_reqs, 48) using Qwen/Qwen3.5-9B's real state layout
(24 linear layers × conv+temporal, tp=1, num_spec=2),
with each request assigned distinct src/dst block ids so L2 can't mask DRAM traffic.

Each request corresponds to 25.88 MiB of copies: 24 temporal and 24 SD conv.

We sweep the num_req to `{1, 4, 8, 16, 32, 64, 128}` using two GPU architectures:

- H100 (HBM3, ~3.35 TB/s peak)
- GB200 (HBM3e, ~8 TB/s peak)

### End-to-end serving

`vllm bench serve` against `vllm serve` with MTP speculative decoding (which exercises the temporal-state copy on every accepted draft step):

```bash
vllm serve \
    --model Qwen/Qwen3.5-9B \
    -tp 1 -pp 1 -dp 1 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --enable-prefix-caching
```

Benchmark workload:

```
    vllm bench serve \
        --dataset-name random \
        --seed 42 \
        --temperature 0 \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate inf \
        --max-concurrency "$conc" \
        --ignore-eos \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "50,90,99"
}
```

Workload: `--dataset-name random`, `INPUT_LEN=500`, `OUTPUT_LEN=5330`, `NUM_PROMPTS=500`.

The workload is chosen to exercise the `_copy_mamba_state_block` which happens when a block is crossed.
For Qwen3.5.9B the block size is 533. The output length is chosen to be 10x the block size.

Hardware: NVIDIA GB200. Concurrency sweep `{32, 64, 128}`. At each concurrency we ran the benchmark **4 times per branch and discarded the 1st run** to strip warmup/JIT effects; reported numbers aggregate the remaining 3 runs, with std devs computed as sample std dev (n−1).

Branches compared:

- `main` — `4e5ca89cfe98121642d76b40e32a006f4d0fbf3b`
- `postprocess_vectorize` — `4575b6f2321d69c9b41ce286f4d520ea52bd3857`

## Test Result

### Microbenchmark — kernel throughput

**H100 (HBM3, ~3.35 TB/s peak):**

| reqs | MAIN (GB/s) | PR (GB/s) | Gain | MAIN % peak | PR % peak |
|-----:|------------:|----------:|-----:|------------:|----------:|
| 1    | 110.3       | 642.2     | **5.82×** | 3.3%   | 19.2% |
| 4    | 400.6       | 1847.4    | **4.61×** | 12.0%  | 55.1% |
| 8    | 737.1       | 2430.9    | **3.30×** | 22.0%  | 72.6% |
| 16   | 1296.2      | 2741.0    | **2.11×** | 38.7%  | **81.8%** |
| 32   | 1925.4      | 2717.6    | **1.41×** | 57.5%  | **81.1%** |
| 64   | 1898.1      | 2784.1    | **1.47×** | 56.7%  | **83.1%** |
| 128  | 2112.5      | 2824.1    | **1.34×** | 63.1%  | **84.3%** |

PR saturates HBM from `reqs≥16` (~81–84% of peak). MAIN never exceeds 63% of peak — launch/serialization-bound.

**GB200 (HBM3e, ~8 TB/s peak):**

| reqs | MAIN (GB/s) | PR (GB/s) | Gain | MAIN % peak | PR % peak |
|-----:|------------:|----------:|-----:|------------:|----------:|
| 1    | 148.5       | 850.2     | **5.72×** | 1.9%   | 10.6% |
| 4    | 291.9       | 1671.9    | **5.73×** | 3.6%   | 20.9% |
| 8    | 570.1       | 3070.8    | **5.39×** | 7.1%   | 38.4% |
| 16   | 1055.9      | 4714.9    | **4.47×** | 13.2%  | 58.9% |
| 32   | 1760.0      | 5818.1    | **3.31×** | 22.0%  | 72.7% |
| 64   | 2098.4      | 5496.3    | **2.62×** | 26.2%  | 68.7% |
| 128  | 2583.6      | 6022.2    | **2.33×** | 32.3%  | **75.3%** |

PR reaches ~75% of peak at `reqs=128` and is still climbing. MAIN caps at 32% of peak — same overhead ceiling as H100.

### Microbenchmark — kernel latency

**H100 (avg kernel latency, µs):**

| reqs | MAIN | PR | Speedup | Absolute saving |
|-----:|-----:|---:|--------:|----------------:|
| 1    | 491.97  | 84.50   | **5.82×** | −407.5 µs |
| 4    | 541.85  | 117.49  | **4.61×** | −424.4 µs |
| 8    | 588.92  | 178.58  | **3.30×** | −410.3 µs |
| 16   | 669.82  | 316.76  | **2.11×** | −353.1 µs |
| 32   | 901.88  | 638.97  | **1.41×** | −262.9 µs |
| 64   | 1829.67 | 1247.38 | **1.47×** | −582.3 µs |
| 128  | 3288.01 | 2459.43 | **1.34×** | −828.6 µs |

Total wall across the sweep: **2.24 s → 1.52 s (−32%)**. A near-constant ~400 µs saving at low `reqs` (fixed overhead removed), growing again at `reqs≥64` as the workload becomes bandwidth-heavy.

**GB200 (avg kernel latency, µs):**

| reqs | MAIN | PR | Speedup | Absolute saving |
|-----:|-----:|---:|--------:|----------------:|
| 1    | 365.31  | 63.82   | **5.72×** | −301.5 µs |
| 4    | 743.50  | 129.82  | **5.73×** | −613.7 µs |
| 8    | 761.45  | 141.37  | **5.39×** | −620.1 µs |
| 16   | 822.26  | 184.14  | **4.47×** | −638.1 µs |
| 32   | 986.63  | 298.45  | **3.31×** | −688.2 µs |
| 64   | 1654.98 | 631.86  | **2.62×** | −1023.1 µs |
| 128  | 2688.44 | 1153.36 | **2.33×** | −1535.1 µs |

Absolute saving grows monotonically with `reqs` (301 → 1535 µs) because HBM3e headroom lets the vectorized path keep scaling.

**Cross-platform takeaways:**

- Low-`reqs` uplift is **~5.7–5.8× on both GPUs** — the decode-step regime where mamba state-copy latency dominates per-token time.
- PR speedup shrinks with batch size on H100 because it hits HBM; on GB200 it shrinks more slowly because HBM3e still has slack at `reqs=128`.
- Decode regime (`reqs=1–8`): sub-200 µs on PR vs 0.36–0.76 ms on MAIN.

### End-to-end serving (GB200)

Steady-state:

| Metric | Branch | conc=32 | conc=64 | conc=128 |
|---|---|---|---|---|
| Output tok/s | main | 7278.1 ± 15.0 | 11350.0 ± 32.2 | 15127.7 ± 111.6 |
| Output tok/s | PR   | 7333.1 ± 15.4 | 11686.7 ± 14.3 | 15351.5 ± 44.1 |
| Δ throughput | | **+0.75 %** | **+2.97 %** | **+1.48 %** |
| Median TPOT (ms) | main | 4.267 ± 0.012 | 5.453 ± 0.032 | 8.100 ± 0.079 |
| Median TPOT (ms) | PR   | 4.230 ± 0.000 | 5.233 ± 0.035 | 7.993 ± 0.035 |
| Δ median TPOT    |      | **−0.87 %** | **−4.03 %** | **−1.32 %** |
| Median ITL (ms)  | main | 11.687 ± 0.006 | 14.810 ± 0.079 | 22.900 ± 0.062 |
| Median ITL (ms)  | PR   | 11.580 ± 0.017 | 14.467 ± 0.021 | 22.683 ± 0.049 |
| Δ median ITL     |      | **−0.92 %** | **−2.32 %** | **−0.95 %** |
| Acceptance rate (%) | main | 98.63 ± 0.05 | 98.26 ± 0.35 | 98.46 ± 0.31 |
| Acceptance rate (%) | PR   | 98.66 ± 0.08 | 98.59 ± 0.10 | 98.68 ± 0.06 |

P99 tails:

| P99 metric | Branch | conc=32 | conc=64 | conc=128 |
|---|---|---|---|---|
| TPOT (ms) | main | 4.98 ± 0.03 | 6.70 ± 0.08 | 9.59 ± 0.17 |
| TPOT (ms) | PR   | 4.94 ± 0.07 | 6.45 ± 0.11 | 9.40 ± 0.05 |
| Δ P99 TPOT |     | **−0.80 %** | **−3.73 %** | **−1.98 %** |
| ITL (ms)  | main | 77.15 ± 0.16 | 80.57 ± 0.17 | 88.10 ± 1.23 |
| ITL (ms)  | PR   | 76.97 ± 0.21 | 79.53 ± 0.52 | 87.75 ± 0.77 |
| Δ P99 ITL  |     | **−0.23 %** | **−1.29 %** | **−0.40 %** |
| E2EL (ms) | main | 26728 ± 227 | 35903 ± 400 | 51639 ± 844 |
| E2EL (ms) | PR   | 26436 ± 328 | 34561 ± 640 | 50599 ± 197 |
| TTFT (ms) | main | 434 ± 16 | 956 ± 391 | 1721 ± 518 |
| TTFT (ms) | PR   | 847 ± 754 | 1046 ± 553 | 1432 ± 23 |

**Analysis.**

- Throughput and per-token latency improve at every concurrency, with deltas well outside 1σ:
    - `conc=32`: +0.75 % throughput, ~0.9 % lower median TPOT
    - `conc=64`: +2.97 % throughput, ~4.0 % lower median TPOT
    - `conc=128`: +1.48 % throughput, ~1.3 % lower median TPOT
- Gains peak at `conc=64`. At `conc=32` the step is still GPU-idle-bound so a faster kernel moves the wall clock only slightly; at `conc=128` the GPU is closer to saturation so the copy is a smaller share of the step. Mid-concurrency is where this kernel is on the critical path.
- **Speculative decoding is unaffected.** Acceptance rates and per-position acceptance are indistinguishable between branches (all ~98.5–98.7 %). The speedup is pure kernel time, not a change in draft/verify behavior.
- P99 TPOT / E2EL tails improve proportionally to the medians (−0.8 % / −3.7 % / −2.0 % for P99 TPOT at conc 32 / 64 / 128; −1.1 % / −3.7 % / −2.0 % for P99 E2EL). At `conc=64` and `conc=128` the P99 improvements exceed 1σ.
- P99 ITL barely moves. These spikes (77–88 ms) are dominated by rare CUDA-graph / eviction events rather than this copy, so vectorization has no purchase on them — as expected.
- P99 TTFT is inconclusive: TTFT is prefill-dominated and shouldn't be affected by this change in either direction. Run-to-run variance on both branches is large (σ from ~15 to ~750 ms); the apparent regression at `conc=32` on the PR branch (847 ± 754 ms vs 434 ± 16 ms) is driven by a single outlier run and is not signal.

### Summary

`_copy_mamba_state_block` goes from ~3–32 % of HBM peak (launch-bound) to ~75–84 % of HBM peak (bandwidth-bound) with a **~5.7–5.8× decode-regime speedup** on both H100 and GB200. End-to-end on GB200 with MTP speculative decoding this shows up as **+0.75 % / +2.97 % / +1.48 %** output throughput and correspondingly lower median and P99 TPOT/E2EL at concurrencies 32 / 64 / 128. Metrics outside the change's blast radius (ITL spikes, TTFT tails, MTP acceptance) are unaffected as expected.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
