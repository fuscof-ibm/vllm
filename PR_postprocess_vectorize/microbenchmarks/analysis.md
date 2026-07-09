# Microbenchmark Analysis: `_copy_mamba_state_block`

MAIN vs `postprocess_vectorize` PR. Benchmark: `benchmark/bench_copy_mamba_state_block.py`
(Qwen/Qwen3.5-9B, tp=1, num_spec=2, 48 states/req, 49.88 MiB/req, warmup=20, timed=200,
temporal dtype=float32).

## Memory Bandwidth

### H100 (HBM3, ~3.35 TB/s peak)

| reqs | MAIN (GB/s) | PR (GB/s) | Gain | MAIN % peak | PR % peak |
|-----:|------------:|----------:|-----:|------------:|----------:|
| 1    | 108.6       | 641.0     | **5.90×** | 3.2%   | 19.1% |
| 4    | 388.9       | 1846.5    | **4.75×** | 11.6%  | 55.1% |
| 8    | 712.3       | 2472.6    | **3.47×** | 21.3%  | 73.8% |
| 16   | 1260.7      | 2799.2    | **2.22×** | 37.6%  | **83.6%** |
| 32   | 1901.1      | 2739.3    | **1.44×** | 56.7%  | **81.8%** |
| 64   | 1852.3      | 2800.8    | **1.51×** | 55.3%  | **83.6%** |
| 128  | 2086.3      | 2822.9    | **1.35×** | 62.3%  | **84.3%** |

PR saturates HBM from reqs≥16 (~82–84% of peak). MAIN caps at 62% — launch/serialization bound.

### GB200 (HBM3e, ~8 TB/s peak)

| reqs | MAIN (GB/s) | PR (GB/s) | Gain | MAIN % peak | PR % peak |
|-----:|------------:|----------:|-----:|------------:|----------:|
| 1    | 73.8        | 479.4     | **6.50×** | 0.9%   | 6.0%  |
| 4    | 280.1       | 1640.6    | **5.86×** | 3.5%   | 20.5% |
| 8    | 547.9       | 3050.0    | **5.57×** | 6.8%   | 38.1% |
| 16   | 1019.2      | 4737.9    | **4.65×** | 12.7%  | 59.2% |
| 32   | 1730.0      | 5900.6    | **3.41×** | 21.6%  | **73.8%** |
| 64   | 2065.9      | 5165.9    | **2.50×** | 25.8%  | 64.6% |
| 128  | 2606.3      | 5788.0    | **2.22×** | 32.6%  | 72.4% |

PR reaches ~74% of peak at reqs=32 with headroom remaining. MAIN caps at 33% — same overhead ceiling as H100.

## Kernel Timings

### H100 (avg kernel latency, µs)

| reqs | MAIN | PR | Speedup | Absolute saving |
|-----:|-----:|---:|--------:|----------------:|
| 1    | 963.25  | 163.18  | **5.90×** | −800.1 µs |
| 4    | 1075.78 | 226.58  | **4.75×** | −849.2 µs |
| 8    | 1174.75 | 338.41  | **3.47×** | −836.3 µs |
| 16   | 1327.50 | 597.85  | **2.22×** | −729.7 µs |
| 32   | 1760.57 | 1221.86 | **1.44×** | −538.7 µs |
| 64   | 3613.88 | 2390.06 | **1.51×** | −1223.8 µs |
| 128  | 6417.25 | 4742.66 | **1.35×** | −1674.6 µs |

Total wall: **4.38 s → 3.00 s (−32%)**. Near-constant ~800 µs saving at low reqs (fixed overhead removed), growing to ~1.7 ms at reqs=128 as workload becomes bandwidth-heavy.

### GB200 (avg kernel latency, µs)

| reqs | MAIN | PR | Speedup | Absolute saving |
|-----:|-----:|---:|--------:|----------------:|
| 1    | 1417.62 | 218.20  | **6.50×** | −1199.4 µs |
| 4    | 1493.72 | 255.02  | **5.86×** | −1238.7 µs |
| 8    | 1527.27 | 274.35  | **5.57×** | −1252.9 µs |
| 16   | 1642.00 | 353.22  | **4.65×** | −1288.8 µs |
| 32   | 1934.67 | 567.24  | **3.41×** | −1367.4 µs |
| 64   | 3240.21 | 1295.81 | **2.50×** | −1944.4 µs |
| 128  | 5136.81 | 2313.11 | **2.22×** | −2823.7 µs |

Total wall: **4.50 s → 1.64 s (−63.6%)**. Absolute saving grows monotonically with reqs (1.20 → 2.82 ms) because HBM3e headroom lets the vectorized path keep scaling.

## Cross-platform Notes

- **Low-reqs uplift is 5.9–6.5× on both GPUs** — the decode-step regime where mamba state copy latency matters most.
- **PR saturates H100 HBM at reqs≥16** (~84% of peak); on GB200 it reaches ~74% at reqs=32 and still has slack.
- **Decode regime (reqs=1–8):** sub-350 µs on PR vs 0.96–1.53 ms on MAIN — the primary latency win for serving.
