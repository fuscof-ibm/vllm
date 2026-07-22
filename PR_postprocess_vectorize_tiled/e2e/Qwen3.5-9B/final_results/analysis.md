# E2E Performance Analysis: `postprocess_vectorize_tiled` vs `main`

**Model:** Qwen3.5-9B (with speculative decoding, 2-token draft)
**Workload:** 500 requests, 250k input tokens, 2.665M generated tokens
**Setup:** 3 runs per branch per concurrency level (16 / 32 / 64)

- `main` @ `9459fc647105f10f754697b3bf136d194564d603`
- `postprocess_vectorize_tiled` (PR) @ `03521ef846a9e2e1c3988e21761d1d13be84686e`

Values below are `mean ± stddev` across 3 runs (sample stddev, n=3).
Δ rows compare PR to main; for latency metrics negative is better, for throughput positive is better.

| Metric | Branch | conc=16 | conc=32 | conc=64 |
|---|---|---|---|---|
| Output tok/s | main | 6219.19 ± 10.28 | 9744.67 ± 5.37 | 13848.81 ± 37.01 |
| Output tok/s | PR   | 6331.49 ± 9.81  | 9864.17 ± 10.30 | 14004.27 ± 20.58 |
| Δ throughput |      | **+1.81 %** | **+1.23 %** | **+1.12 %** |
| Median TPOT (ms) | main | 2.407 ± 0.006 | 3.027 ± 0.006 | 4.113 ± 0.025 |
| Median TPOT (ms) | PR   | 2.373 ± 0.006 | 2.987 ± 0.006 | 4.063 ± 0.032 |
| Δ median TPOT    |      | **−1.39 %** | **−1.32 %** | **−1.22 %** |
| Median ITL (ms)  | main | 7.003 ± 0.006 | 8.623 ± 0.012 | 11.773 ± 0.032 |
| Median ITL (ms)  | PR   | 6.890 ± 0.010 | 8.517 ± 0.015 | 11.693 ± 0.021 |
| Δ median ITL     |      | **−1.62 %** | **−1.24 %** | **−0.68 %** |
| Mean E2EL (ms)   | main | 13224.01 ± 15.90 | 16889.25 ± 7.79 | 23644.57 ± 32.34 |
| Mean E2EL (ms)   | PR   | 13010.79 ± 8.04  | 16680.60 ± 25.10 | 23416.33 ± 40.85 |
| Δ mean E2EL      |      | **−1.61 %** | **−1.24 %** | **−0.97 %** |
| Acceptance rate (%) | main | 98.68 ± 0.05 | 98.61 ± 0.03 | 98.60 ± 0.09 |
| Acceptance rate (%) | PR   | 98.67 ± 0.03 | 98.51 ± 0.08 | 98.62 ± 0.05 |

## Summary

- **Consistent ~1.1–1.8% throughput improvement** across all concurrency levels, well above the ~0.1–0.3% run-to-run noise (visible from the stddevs).
- **Latency reductions track throughput gains**: TPOT, ITL, and E2EL all improve by ~0.7–1.6%, monotonically decreasing as concurrency rises.
- **Gain shrinks as concurrency grows**: at conc=16 the postprocess path is a larger fraction of the per-step cost, so tiling helps more; at conc=64 GEMM/attention dominates and the relative win compresses.
- **Speculative decoding is unaffected**: acceptance rate stays ~98.5–98.7% and acceptance length stays 2.97 on both branches — this is purely a postprocess optimization with no semantic effect on drafts.
- **No regressions** on any metric at any concurrency level.
