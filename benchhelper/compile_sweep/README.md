# compile_sweep

Tooling to measure how much `torch.compile` actually helps a given vLLM
model, and to point at concrete things to optimize by hand.

The sweep boots three vLLM servers in series with the same workload and
measures their throughput / latency, then a parser reads the `full` run's
debug dump and tells you which fusion passes fired (or why they didn't).

## What it runs

| cfg         | flags                                              | what it isolates                |
|-------------|----------------------------------------------------|---------------------------------|
| `eager`     | `-O0`                                              | no compile, no CUDA graphs      |
| `nocompile` | `-cc '{"mode":0,"cudagraph_mode":"FULL"}'`         | CUDA graphs only, no compile    |
| `full`      | `-cc '{"debug_dump_path":...}'` (default `-O2`)    | compile + `FULL_AND_PIECEWISE`  |

So:

- `nocompile / eager` &nbsp;= CUDA-graph contribution
- `full / nocompile` &nbsp;&nbsp;= `torch.compile` contribution
- `full / eager` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= combined

The `full` run also writes an FX/Inductor debug dump that
`parse_compile_dump.py` reads.

## Files

- `bench_compile_sweep.sh` — orchestrates the three servers + client
  concurrency sweep. Captures server logs and the FX dump.
- `bench_compile_compare.py` — reads the `c<N>.json` files written by
  `vllm bench serve` and prints a 3-way table per metric, with speedup
  ratios.
- `parse_compile_dump.py` — parses `server_full.log` + the dump dir
  and prints compile times, fusion-pass match counts, and
  manual-optimization hints (which pass_config gates were ON/OFF,
  which fusion passes are candidates for hand-fused kernels).

## Quick start

```bash
# defaults: granite-4.0-h-small, 1 GPU, port 8000, concurrencies 1/4/16
./benchhelper/compile_sweep/bench_compile_sweep.sh

# different model + 4-way TP
MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    GPU="0,1,2,3" TP=4 \
    ./benchhelper/compile_sweep/bench_compile_sweep.sh

# heavier workload, custom concurrency list
INPUT_LEN=512 OUTPUT_LEN=1024 NUM_PROMPTS=400 \
    CONCURRENCIES="1 8 32 64" \
    ./benchhelper/compile_sweep/bench_compile_sweep.sh
```

Then look at the results:

```bash
# 3-way comparison table
./benchhelper/compile_sweep/bench_compile_compare.py \
    bench_results/compile_sweep/<model_tag>

# what compile actually did, plus optimization hints
./benchhelper/compile_sweep/parse_compile_dump.py \
    bench_results/compile_sweep/<model_tag>/logs/server_full.log \
    bench_results/compile_sweep/<model_tag>/compile_dump
```

## Environment variables

| var            | default                              | meaning                                                  |
|----------------|--------------------------------------|----------------------------------------------------------|
| `MODEL`        | `ibm-granite/granite-4.0-h-small`    | HF repo or local path                                    |
| `GPU`          | `0`                                  | comma-separated GPU ids for `CUDA_VISIBLE_DEVICES`       |
| `TP`           | `1`                                  | tensor-parallel size; must equal the number of GPUs      |
| `PORT`         | `8000`                               | OpenAI server port                                       |
| `INPUT_LEN`    | `256`                                | random-input length (tokens)                             |
| `OUTPUT_LEN`   | `512`                                | random-output length (tokens)                            |
| `NUM_PROMPTS`  | `200`                                | prompts per concurrency point                            |
| `CONCURRENCIES`| `1 4 16`                             | space-separated concurrency points                       |
| `MODEL_TAG`    | `basename $MODEL`                    | output subdir under `bench_results/compile_sweep/`       |

## Output layout

```sh
bench_results/compile_sweep/<model_tag>/
  logs/
    server_eager.log
    server_nocompile.log
    server_full.log         # consumed by parse_compile_dump.py
  compile_dump/             # FX / output_code dumps from the `full` run
  eager/      c1.json  c4.json  c16.json    # consumed by bench_compile_compare.py
  nocompile/  c1.json  c4.json  c16.json
  full/       c1.json  c4.json  c16.json
```

## Example output: bench_compile_compare.py

Real numbers from `granite-4.0-h-small`, bf16, single H100, `TP=1`,
`INPUT_LEN=256`, `OUTPUT_LEN=512`, `NUM_PROMPTS=200`:

```sh
## out tok/s  [higher is better]
 conc |       eager |   nocompile |        full |   noc/eag |  full/noc |  full/eag
------+-------------+-------------+-------------+-----------+-----------+----------
    1 |       27.42 |      105.44 |      110.73 |    →3.85× |    →1.05× |    →4.04×
    4 |      105.52 |      261.29 |      268.86 |    →2.48× |    →1.03× |    →2.55×
   16 |      396.13 |      673.32 |      682.14 |    →1.70× |    →1.01× |    →1.72×

## mean TPOT ms  [lower is better]
 conc |       eager |   nocompile |        full |   noc/eag |  full/noc |  full/eag
------+-------------+-------------+-------------+-----------+-----------+----------
    1 |       36.33 |        9.36 |        8.95 |    →3.88× |    →1.05× |    →4.06×
    4 |       37.71 |       15.09 |       14.70 |    →2.50× |    →1.03× |    →2.57×
   16 |       38.38 |       22.61 |       22.28 |    →1.70× |    →1.01× |    →1.72×

## med TTFT ms  [lower is better]
 conc |       eager |   nocompile |        full |   noc/eag |  full/noc |  full/eag
------+-------------+-------------+-------------+-----------+-----------+----------
    1 |      104.35 |       72.54 |       49.42 |    →1.44× |    →1.47× |    →2.11×
    4 |      134.50 |      123.01 |       89.33 |    →1.09× |    →1.38× |    →1.51×
   16 |      246.52 |      221.19 |      203.62 |    →1.11× |    →1.09× |    →1.21×
```

The arrow points at the winning side regardless of whether the metric is
higher-is-better or lower-is-better.

How to read this:

- `noc/eag` = **CUDA graphs** are doing almost all the work on this model
  (3.85× at c=1, shrinking to 1.70× at c=16 as the GPU saturates).
- `full/noc` = `torch.compile` adds **1–5%** in steady-state TPOT/throughput
  but a noticeable **1.4×** TTFT win — Inductor's compiled prefill is
  faster than the eager / CUDA-graphed path even when no FX fusion fires.
- `full/eag` is the headline number you'd quote, but the split shows that
  on this config most of it is "captured into a CUDA graph", not "fused
  into a better kernel".

## Example output: parse_compile_dump.py

Same Granite 4 run:

```sh
=== compile summary: bench_results/compile_sweep/granite-4.0-h-small/logs/server_full.log
  profiling/warmup run          : 46.58 s

=== fusion-pass matches (final match_table)
  (none)

=== optimization hints (manual hand-fused-kernel candidates)
  pass_config: {} (all defaults — no fusion gates overridden)
  graph split at 16 op(s): vllm::unified_attention_with_output, vllm::unified_mla_attention_with_output, vllm::mamba_mixer2, vllm::mamba_mixer, vllm::short_conv, vllm::linear_attention …

  no fusion-pass matches were recorded.
  → pass_config is empty → all gates at their defaults. With a non-quantized,
    TP=1 model, no kernel fusions can fire. To exercise them:
      • use an FP8/INT8 quantized variant (enables the *quant
        fusions), or
      • run with TP>1 + FlashInfer (enables AllReduceFusion +
        SequenceParallelism), or
      • set pass_config.fuse_rope_kvcache=True and pass
        splitting_ops=[] (PR #43355 path) on a model with the
        custom rotary_embedding op.

  what compile DID do on this run (from per-pass timing):
    Inductor codegen + CUDA-graph capture only. The compile vs.
    nocompile delta in your bench results is purely Inductor's
    work (fused elementwise/contiguity ops it generates), not
    any of the cataloged fusion passes. Manual optimization for
    this model should target Mamba/SSM kernels and the existing
    custom ops, not the FX-pass catalog.

=== per-pass timing (sum across all invocations)
  pass                                n   total_ms  max_ms
  ----------------------------------  --  --------  ------
  VllmIRInplaceFunctionalizationPass  41  262.3     12.3
  VllmIRLoweringPass                  5   92.4      21.4
  PostCleanupPass                     10  55.5      8.4
  FixFunctionalizationPass            5   41.8      10.6
  ...
```

This run is the "uninteresting" case: bf16 + TP=1 → all fusion gates
are at their defaults (False) → no FX fusion can fire, and the only
passes that ran are the IR-cleanup ones. The hint section calls that
out explicitly and tells you which knob you'd flip to make a *quant
or*AllReduce / *SequenceParallel fusion reachable.

When a fusion *does* fire, each one gets a `[HIGH|MEDIUM|LOW]` priority,
a one-line description of what it fuses, the FX pattern file you can
read, and the manual-kernel status (e.g. PR #43355 for
`RopeKVCacheFusionPass`). The footer is the four-step template for
adding a hand-fused op (kernel → torch.ops binding → call-site rewire
→ drop the FX pass once every model is rewired, per issue #43224).

## Notes

- `bench_compile_sweep.sh` requires `.venv/bin/python` (uv venv, per
  `AGENTS.md`); it errors out early otherwise.
- The `full` run's `debug_dump_path` is what makes
  `parse_compile_dump.py` useful — without it the dump dir is empty.
- TP>1 must match `GPU` cardinality (e.g. `GPU="0,1" TP=2`); vLLM
  refuses to start otherwise.
- Workload is decode-heavy by default (`OUTPUT_LEN=512`,
  `INPUT_LEN=256`). Bump `INPUT_LEN` to stress prefill / TTFT.
