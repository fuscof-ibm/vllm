# End-to-end Performance: `postprocess_vectorize` vs `main`

## Setup

**Model / serving config:**

```bash
vllm serve \
    --model Qwen/Qwen3.5-9B \
    -tp 1 -pp 1 -dp 1 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --enable-prefix-caching
```

**Workload:**

- `vllm bench serve` with `--dataset-name random`
- `INPUT_LEN=500`, `OUTPUT_LEN=5330`, `NUM_PROMPTS=500`
- `--request-rate inf`, `--ignore-eos`, `--temperature 0`, `--seed 42`
- Percentile metrics: `ttft,tpot,itl,e2el` at `50,90,99`

**Hardware:** NVIDIA GB200.

**Branches compared:**

- `main` — commit `4e5ca89cfe98121642d76b40e32a006f4d0fbf3b`
- `postprocess_vectorize` — commit `4575b6f2321d69c9b41ce286f4d520ea52bd3857`

**Concurrency sweep:** `{32, 64, 128}`.

**Run protocol:** at each concurrency we ran the benchmark **4 times** per branch
and **discarded the 1st run** to remove warmup / JIT effects.
All numbers below are aggregated over the remaining 3 runs. Standard
deviations reported are sample std dev (n−1).

## Steady-state results

| Metric | Branch | conc=32 | conc=64 | conc=128 |
|---|---|---|---|---|
| Output tok/s | main | 7278.1 ± 15.0 | 11350.0 ± 32.2 | 15127.7 ± 111.6 |
| Output tok/s | vectorize | 7333.1 ± 15.4 | 11686.7 ± 14.3 | 15351.5 ± 44.1 |
| Δ throughput | | **+0.75 %** | **+2.97 %** | **+1.48 %** |
| Median TPOT (ms) | main | 4.267 ± 0.012 | 5.453 ± 0.032 | 8.100 ± 0.079 |
| Median TPOT (ms) | vectorize | 4.230 ± 0.000 | 5.233 ± 0.035 | 7.993 ± 0.035 |
| Median ITL (ms) | main | 11.687 ± 0.006 | 14.810 ± 0.079 | 22.900 ± 0.062 |
| Median ITL (ms) | vectorize | 11.580 ± 0.017 | 14.467 ± 0.021 | 22.683 ± 0.049 |
| Acceptance rate (%) | main | 98.63 ± 0.05 | 98.26 ± 0.35 | 98.46 ± 0.31 |
| Acceptance rate (%) | vectorize | 98.66 ± 0.08 | 98.59 ± 0.10 | 98.68 ± 0.06 |

## P99 tails

| P99 metric | Branch | conc=32 | conc=64 | conc=128 |
|---|---|---|---|---|
| TPOT (ms) | main | 4.98 ± 0.03 | 6.70 ± 0.08 | 9.59 ± 0.17 |
| TPOT (ms) | vectorize | 4.94 ± 0.07 | 6.45 ± 0.11 | 9.40 ± 0.05 |
| ITL (ms) | main | 77.15 ± 0.16 | 80.57 ± 0.17 | 88.10 ± 1.23 |
| ITL (ms) | vectorize | 76.97 ± 0.21 | 79.53 ± 0.52 | 87.75 ± 0.77 |
| E2EL (ms) | main | 26728 ± 227 | 35903 ± 400 | 51639 ± 844 |
| E2EL (ms) | vectorize | 26436 ± 328 | 34561 ± 640 | 50599 ± 197 |
| TTFT (ms) | main | 434 ± 16 | 956 ± 391 | 1721 ± 518 |
| TTFT (ms) | vectorize | 847 ± 754 | 1046 ± 553 | 1432 ± 23 |

## Analysis

**Throughput and steady-state latency.** The `postprocess_vectorize` branch wins
at every concurrency, with the deltas well outside 1σ:

- conc=32: +0.75 % throughput, ~0.9 % lower median TPOT
- conc=64: +2.97 % throughput, ~4.0 % lower median TPOT
- conc=128: +1.48 % throughput, ~1.3 % lower median TPOT

Gains peak at **conc=64**. This is the regime where postprocess CPU work is a
meaningful fraction of per-step time: batch size is large enough that Python
per-token overhead matters, but the GPU is not yet saturated. At conc=32 the
step is still GPU-bound so vectorizing CPU-side work moves the needle only
slightly. At conc=128 the GPU saturates, and again postprocess is a smaller
share of the step.

**Speculative decoding is unaffected.** Acceptance rates and per-position
acceptance are indistinguishable between branches (all ~98.5–98.7 %). The
speedup is pure CPU-side work reduction, not a change in draft/verify
behavior.

**P99 tails (TPOT, E2EL).** Steady-state tails improve proportionally to the
medians:

- P99 TPOT: −0.8 % / −3.7 % / −2.0 % at conc 32 / 64 / 128
- P99 E2EL: −1.1 % / −3.7 % / −2.0 %

At conc=64 and conc=128 the P99 improvements exceed 1σ. At conc=32 the P99
TPOT delta (~0.04 ms) is inside the noise band.

**P99 ITL barely moves.** These spikes (77–88 ms) are dominated by rare
CUDA-graph / eviction events rather than postprocess CPU work, so
vectorization has no purchase on them. Deltas are within or comparable to σ.

**P99 TTFT is inconclusive.** Even after dropping the 1st run, TTFT tails
show large run-to-run variance on both branches (σ from ~15 to ~750 ms). TTFT
is prefill-dominated and shouldn't be affected by postprocess vectorization
in either direction. The apparent regression at conc=32 for the vectorize
branch (847 ± 754 ms vs 434 ± 16 ms) is driven by a single outlier run and
should not be treated as signal.

## Conclusion

`postprocess_vectorize` delivers a real, reproducible improvement to
steady-state throughput and per-token latency (median and P99 TPOT / E2EL)
with the largest gains at mid-range concurrency. Speculative decoding
behavior is unchanged. Metrics outside the change's blast radius (ITL
spikes, TTFT tails) are unaffected as expected.
