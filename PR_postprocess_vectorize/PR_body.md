## Purpos.

PR [#40172](https://github.com/vllm-project/vllm/pull/40172) introduced the [postprocess_mamba_fused_kernel](https://github.com/vllm-project/vllm/blob/0d12618e98ff2d21d36081e0e9b4eb23573b6d38/vllm/v1/worker/mamba_utils.py#L133) in MRV1 to copy states in hybrid models when prefix caching is enabled in align mode (and under MTP). Every accepted draft step that crosses a block boundary triggers a full sweep of `(num_reqs × total_states)` block copies — for Qwen/Qwen3.5-9B that's 24 linear-attention layers × (conv + temporal) = 48 state copies per accepted step, ~2 MiB temporal + ~80 KiB conv per block.

A later PR [#42406](https://github.com/vllm-project/vllm/pull/42406) refactored the kernel to reuse the copy mechanism, now delegated to the [_copy_mamba_state_block](https://github.com/vllm-project/vllm/blob/0d12618e98ff2d21d36081e0e9b4eb23573b6d38/vllm/v1/worker/mamba_utils.py#L27) to support align mode prefix caching in MRV2.

The current copy body issues **1-byte loads and stores**. Temporal states are 20–30× the size of conv states and are contiguous, so the temporal path dominates and leaves most HBM bandwidth on the floor: the kernel tops out at ~62 % of HBM3 peak on H100 and ~33 % of HBM3e peak on GB200.

This PR switches the temporal-state copy to **`uint64` loads/stores** (8× wider transactions per instruction). To make every issued address safe by construction, `MambaSpecDecodeGPUContext` now asserts at setup that the temporal-state base pointer and per-block stride are both 8-byte aligned; bf16/fp16/fp32 temporal states with contiguous inner dims satisfy this trivially, so the assert is a cheap guardrail rather than a fallback path. The conv path is untouched.

Net effect: the kernel becomes HBM-bandwidth-bound — ~82–84 % of peak on H100 from `reqs≥16` and up to ~74 % on GB200 from `reqs≥32`. For smaller number of requests (`reqs=1–8`) is up to ~5.9–6.5× on both H100 and GB200.

End-to-end on GB200 with MTP this shows up as **+0.75 % / +2.97 % / +1.48 %** output throughput and correspondingly lower median and P99 TPOT at concurrencies 32 / 64 / 128, with MTP acceptance unchanged.


## Test Plan

### Microbenchmark

Developed a microbenchmark harness [bench_copy_mamba_state_block.py](https://github.com/fuscof-ibm/vllm/blob/postprocess_benchmark/bench_copy_mamba_state_block.py),
which isolates `_copy_mamba_state_block` behind a minimal Triton wrapper launched at
the production grid (num_reqs, 48) using Qwen/Qwen3.5-9B's real state layout
(24 linear layers × conv+temporal, tp=1, num_spec=2),
with each request assigned distinct src/dst block ids so L2 can't mask DRAM traffic.

Each request corresponds to 49.88 MiB of copies: 24 temporal (float32) and 24 SD conv (bfloat16).

We sweep the num_req to `{1, 4, 8, 16, 32, 64, 128}` using two GPU architectures:

- H100 (HBM3, ~3.35 TB/s peak)
- GB200 (HBM3e, ~8 TB/s peak)

### End-to-end serving

`vllm bench serve` against `vllm serve` with MTP and prefix caching enabled (which exercises the state copies):

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
```

Workload: `--dataset-name random`, `INPUT_LEN=500`, `OUTPUT_LEN=5330`, `NUM_PROMPTS=500`.

The workload is chosen to exercise the `_copy_mamba_state_block` which happens when a block is crossed.
For Qwen3.5-9B the block size is 533. The output length is chosen to be 10x the block size.

Hardware: NVIDIA GB200. Concurrency sweep `{32, 64, 128}`. At each concurrency we ran the benchmark **4 times per branch and discarded the 1st run** to strip warmup/JIT effects; reported numbers aggregate the remaining 3 runs, with std devs computed as sample std dev (n−1).

Branches compared:

- `main` — [`4e5ca89c`](https://github.com/vllm-project/vllm/commit/4e5ca89cfe98121642d76b40e32a006f4d0fbf3b)
- `postprocess_vectorize` — [`9256ed051`](https://github.com/fuscof-ibm/vllm/commit/9256ed051e0034ca261718cdf8e0663711f9d7d2)

## Test Result

### Microbenchmark — kernel throughput

**H100 (HBM3, ~3.35 TB/s peak):**

| reqs | MAIN (GB/s) | PR (GB/s) | Gain | MAIN % peak | PR % peak |
|-----:|------------:|----------:|-----:|------------:|----------:|
| 1    | 108.6       | 641.0     | **5.90×** | 3.2%   | 19.1% |
| 4    | 388.9       | 1846.5    | **4.75×** | 11.6%  | 55.1% |
| 8    | 712.3       | 2472.6    | **3.47×** | 21.3%  | 73.8% |
| 16   | 1260.7      | 2799.2    | **2.22×** | 37.6%  | **83.6%** |
| 32   | 1901.1      | 2739.3    | **1.44×** | 56.7%  | **81.8%** |
| 64   | 1852.3      | 2800.8    | **1.51×** | 55.3%  | **83.6%** |
| 128  | 2086.3      | 2822.9    | **1.35×** | 62.3%  | **84.3%** |

PR saturates HBM from `reqs≥16` (~82–84% of peak). MAIN caps at 62% of peak — launch/serialization-bound.

**GB200 (HBM3e, ~8 TB/s peak):**

| reqs | MAIN (GB/s) | PR (GB/s) | Gain | MAIN % peak | PR % peak |
|-----:|------------:|----------:|-----:|------------:|----------:|
| 1    | 73.8        | 479.4     | **6.50×** | 0.9%   | 6.0%  |
| 4    | 280.1       | 1640.6    | **5.86×** | 3.5%   | 20.5% |
| 8    | 547.9       | 3050.0    | **5.57×** | 6.8%   | 38.1% |
| 16   | 1019.2      | 4737.9    | **4.65×** | 12.7%  | 59.2% |
| 32   | 1730.0      | 5900.6    | **3.41×** | 21.6%  | **73.8%** |
| 64   | 2065.9      | 5165.9    | **2.50×** | 25.8%  | 64.6% |
| 128  | 2606.3      | 5788.0    | **2.22×** | 32.6%  | 72.4% |

PR reaches ~74% of peak at `reqs=32` and still has headroom. MAIN caps at 33% of peak — same overhead ceiling as H100.

### Microbenchmark — kernel latency

**H100 (avg kernel latency, µs):**

| reqs | MAIN | PR | Speedup | Absolute saving |
|-----:|-----:|---:|--------:|----------------:|
| 1    | 963.25  | 163.18  | **5.90×** | −800.1 µs |
| 4    | 1075.78 | 226.58  | **4.75×** | −849.2 µs |
| 8    | 1174.75 | 338.41  | **3.47×** | −836.3 µs |
| 16   | 1327.50 | 597.85  | **2.22×** | −729.7 µs |
| 32   | 1760.57 | 1221.86 | **1.44×** | −538.7 µs |
| 64   | 3613.88 | 2390.06 | **1.51×** | −1223.8 µs |
| 128  | 6417.25 | 4742.66 | **1.35×** | −1674.6 µs |

Total wall across the sweep: **4.38 s → 3.00 s (−32%)**. A near-constant ~800 µs saving at low `reqs` (fixed overhead removed), growing to ~1.7 ms at `reqs=128` as the workload becomes bandwidth-heavy.

**GB200 (avg kernel latency, µs):**

| reqs | MAIN | PR | Speedup | Absolute saving |
|-----:|-----:|---:|--------:|----------------:|
| 1    | 1417.62 | 218.20  | **6.50×** | −1199.4 µs |
| 4    | 1493.72 | 255.02  | **5.86×** | −1238.7 µs |
| 8    | 1527.27 | 274.35  | **5.57×** | −1252.9 µs |
| 16   | 1642.00 | 353.22  | **4.65×** | −1288.8 µs |
| 32   | 1934.67 | 567.24  | **3.41×** | −1367.4 µs |
| 64   | 3240.21 | 1295.81 | **2.50×** | −1944.4 µs |
| 128  | 5136.81 | 2313.11 | **2.22×** | −2823.7 µs |

Total wall across the sweep: **4.50 s → 1.64 s (−63.6%)**. Absolute saving grows monotonically with `reqs` (1.20 → 2.82 ms) because HBM3e headroom lets the vectorized path keep scaling.

**Cross-platform takeaways:**

- Low-`reqs` uplift is **~5.9–6.5× on both GPUs**.
- PR saturates H100 HBM at `reqs≥16` (~84% peak); on GB200 it reaches ~74% at `reqs=32`.
- Decode regime (`reqs=1–8`): sub-350 µs on PR vs 0.96–1.53 ms on MAIN.

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
<!--
| E2EL (ms) | main | 26728 ± 227 | 35903 ± 400 | 51639 ± 844 |
| E2EL (ms) | PR   | 26436 ± 328 | 34561 ± 640 | 50599 ± 197 |
| TTFT (ms) | main | 434 ± 16 | 956 ± 391 | 1721 ± 518 |
| TTFT (ms) | PR   | 847 ± 754 | 1046 ± 553 | 1432 ± 23 |
-->

**Analysis.**

- Throughput and per-token latency improve at every concurrency, with deltas well outside 1σ:
    - `conc=32`: +0.75 % throughput, ~0.9 % lower median TPOT
    - `conc=64`: +2.97 % throughput, ~4.0 % lower median TPOT
    - `conc=128`: +1.48 % throughput, ~1.3 % lower median TPOT
- Gains peak at `conc=64`. At `conc=32` the step is still GPU-idle-bound so a faster kernel moves the wall clock only slightly; at `conc=128` the GPU is closer to saturation so the copy is a smaller share of the step. Mid-concurrency is where this kernel is on the critical path.
- **Speculative decoding is unaffected.** Acceptance rates and per-position acceptance are indistinguishable between branches (all ~98.5–98.7 %). The speedup is pure kernel time, not a change in draft/verify behavior.

<!--
- P99 TPOT / E2EL tails improve proportionally to the medians (−0.8 % / −3.7 % / −2.0 % for P99 TPOT at conc 32 / 64 / 128; −1.1 % / −3.7 % / −2.0 % for P99 E2EL). At `conc=64` and `conc=128` the P99 improvements exceed 1σ.
- P99 ITL barely moves. These spikes (77–88 ms) are dominated by rare CUDA-graph / eviction events rather than this copy, so vectorization has no purchase on them — as expected.
- P99 TTFT is inconclusive: TTFT is prefill-dominated and shouldn't be affected by this change in either direction. Run-to-run variance on both branches is large (σ from ~15 to ~750 ms); the apparent regression at `conc=32` on the PR branch (847 ± 754 ms vs 434 ± 16 ms) is driven by a single outlier run and is not signal.
-->

### Summary

`_copy_mamba_state_block` goes from ~3–33 % of HBM peak (launch-bound) to ~74–84 % of HBM peak (bandwidth-bound) with a **~5.9–6.5× decode-regime speedup** on both H100 and GB200. End-to-end on GB200 with MTP speculative decoding this shows up as **+0.75 % / +2.97 % / +1.48 %** output throughput and correspondingly lower median and P99 TPOT at concurrencies 32 / 64 / 128. Metrics outside the change's blast radius (ITL spikes, TTFT tails, MTP acceptance) are unaffected as expected.

### AI assistance disclosure

This PR was prepared with AI assistance (Claude Code). The submitter designed the change, ran every benchmark reported here on their own hardware, and reviewed every changed line before commit;
AI assistance was used for microbenchmark scaffolding.

**Duplicate-work check.** Before opening this PR, the following searches returned no open PR targeting the `_copy_mamba_state_block` temporal-copy path:

```bash
gh pr list --repo vllm-project/vllm --state open --search "_copy_mamba_state_block"
gh pr list --repo vllm-project/vllm --state open --search "postprocess_mamba_fused_kernel"
gh pr list --repo vllm-project/vllm --state open --search "42406 in:body"
```

PR [#40172](https://github.com/vllm-project/vllm/pull/40172) introduced the kernel and PR [#42406](https://github.com/vllm-project/vllm/pull/42406) refactored dispatch; this PR is the first change to widen the temporal load/store transactions.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.

</details>
