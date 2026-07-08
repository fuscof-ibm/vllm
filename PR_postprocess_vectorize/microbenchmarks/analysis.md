# Microbenchmark Analysis: `_copy_mamba_state_block`

MAIN vs `postprocess_vectorize` PR. Benchmark: `benchmark/bench_copy_mamba_state_block.py`
(Qwen/Qwen3.5-9B, tp=1, num_spec=2, 48 states/req, 25.88 MiB/req, warmup=20, timed=200).

## Memory Bandwidth

### H100 (HBM3, ~3.35 TB/s peak)

| reqs | MAIN (GB/s) | PR (GB/s) | Gain | MAIN % peak | PR % peak |
|-----:|------------:|----------:|-----:|------------:|----------:|
| 1    | 110.3       | 642.2     | **5.82×** | 3.3%   | 19.2% |
| 4    | 400.6       | 1847.4    | **4.61×** | 12.0%  | 55.1% |
| 8    | 737.1       | 2430.9    | **3.30×** | 22.0%  | 72.6% |
| 16   | 1296.2      | 2741.0    | **2.11×** | 38.7%  | **81.8%** |
| 32   | 1925.4      | 2717.6    | **1.41×** | 57.5%  | **81.1%** |
| 64   | 1898.1      | 2784.1    | **1.47×** | 56.7%  | **83.1%** |
| 128  | 2112.5      | 2824.1    | **1.34×** | 63.1%  | **84.3%** |

PR saturates HBM from reqs≥16 (~81–84% of peak). MAIN never exceeds 63% — launch/serialization-bound.

### GB200 (HBM3e, ~8 TB/s peak)

| reqs | MAIN (GB/s) | PR (GB/s) | Gain | MAIN % peak | PR % peak |
|-----:|------------:|----------:|-----:|------------:|----------:|
| 1    | 148.5       | 850.2     | **5.72×** | 1.9%   | 10.6% |
| 4    | 291.9       | 1671.9    | **5.73×** | 3.6%   | 20.9% |
| 8    | 570.1       | 3070.8    | **5.39×** | 7.1%   | 38.4% |
| 16   | 1055.9      | 4714.9    | **4.47×** | 13.2%  | 58.9% |
| 32   | 1760.0      | 5818.1    | **3.31×** | 22.0%  | 72.7% |
| 64   | 2098.4      | 5496.3    | **2.62×** | 26.2%  | 68.7% |
| 128  | 2583.6      | 6022.2    | **2.33×** | 32.3%  | **75.3%** |

PR reaches ~75% of peak at reqs=128 and is still climbing. MAIN caps at 32% of peak — same overhead ceiling as H100.

## Kernel Timings

### H100 (avg kernel latency, µs)

| reqs | MAIN | PR | Speedup | Absolute saving |
|-----:|-----:|---:|--------:|----------------:|
| 1    | 491.97  | 84.50   | **5.82×** | −407.5 µs |
| 4    | 541.85  | 117.49  | **4.61×** | −424.4 µs |
| 8    | 588.92  | 178.58  | **3.30×** | −410.3 µs |
| 16   | 669.82  | 316.76  | **2.11×** | −353.1 µs |
| 32   | 901.88  | 638.97  | **1.41×** | −262.9 µs |
| 64   | 1829.67 | 1247.38 | **1.47×** | −582.3 µs |
| 128  | 3288.01 | 2459.43 | **1.34×** | −828.6 µs |

Total wall: **2.24 s → 1.52 s (−32%)**. Near-constant ~400 µs saving at low reqs (fixed overhead removed), growing again at reqs≥64 as workload becomes bandwidth-heavy.

### GB200 (avg kernel latency, µs)

| reqs | MAIN | PR | Speedup | Absolute saving |
|-----:|-----:|---:|--------:|----------------:|
| 1    | 365.31  | 63.82   | **5.72×** | −301.5 µs |
| 4    | 743.50  | 129.82  | **5.73×** | −613.7 µs |
| 8    | 761.45  | 141.37  | **5.39×** | −620.1 µs |
| 16   | 822.26  | 184.14  | **4.47×** | −638.1 µs |
| 32   | 986.63  | 298.45  | **3.31×** | −688.2 µs |
| 64   | 1654.98 | 631.86  | **2.62×** | −1023.1 µs |
| 128  | 2688.44 | 1153.36 | **2.33×** | −1535.1 µs |

Absolute saving grows monotonically with reqs (301 → 1535 µs) because HBM3e headroom lets the vectorized path keep scaling.

## Cross-platform Notes

- **Low-reqs uplift is ~5.7–5.8× on both GPUs** — the decode-step regime where mamba state copy latency matters most.
- **PR speedup shrinks with batch size on H100** because it hits HBM; on GB200 it shrinks more slowly because HBM3e still has slack.
- **Decode regime (reqs=1–8):** sub-200 µs on PR vs 0.36–0.76 ms on MAIN — the primary latency win for serving.
