# APC sweep helpers

Three small scripts to compare server-side configurations on a decode-heavy
workload:

- [serve.sh](serve.sh) — starts `vllm serve` for `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
  on TP=4 with Qwen3-Next MTP (2 spec tokens) and `--enable-prefix-caching`.
- [bench_sweep.sh](bench_sweep.sh) — client-side only. Sweeps concurrency
  `1, 4, 8, 16, 32` × 3 runs against an already-running server, using
  `random` prompts (`input=256`, `output=2048`, `num_prompts=500`,
  `--ignore-eos`). Results land in `bench_results/<label>/`.
- [bench_summarize.py](bench_summarize.py) — 2-way diff (`main_dir` vs
  `pr_dir`) over throughput / TPOT / ITL / TTFT.

## Goal

Compare three configurations of the same decode-heavy run:

| Label            | Commit      | APC | What it shows                              |
| ---------------- | ----------- | --- | ------------------------------------------ |
| `main-apc-off`   | `c68c55d4`  | off | Pure baseline.                             |
| `main-apc-on`    | `c68c55d4`  | on  | APC overhead on baseline (no prefix hits). |
| `pr40172-apc-on` | `b730c4635` | on  | Kernel gain stacked on APC-on.             |

`c68c55d4` = RVV VLEN=256 commit (treated here as `main` baseline).
`b730c4635` = [#40172](https://github.com/vllm-project/vllm/pull/40172) fused
Mamba postprocess kernel.

## Caveat: `random` does not exercise APC

[bench_sweep.sh](bench_sweep.sh) uses `--dataset-name random` with no shared
prefix portion (no `--random-prefix-len`). APC has nothing to hit, so the
`main-apc-on` vs `main-apc-off` delta measures **APC bookkeeping overhead**,
not APC benefit. This is intentional for now — switch to a prefix-sharing
dataset if you need to measure APC gain.

## Procedure

For each of the three configurations, in a separate shell on the server box:

```bash
# 1. Check out the right commit.
git checkout <commit>          # c68c55d4 or b730c4635

# 2. Edit serve.sh: keep or remove --enable-prefix-caching as required.

# 3. Start the server.
./serve.sh
```

Then in a client shell, after the server is ready:

```bash
cd benchhelper/apc
./bench_sweep.sh main-apc-off    # or main-apc-on / pr40172-apc-on
```

Results: `bench_results/main-apc-off/`, `bench_results/main-apc-on/`,
`bench_results/pr40172-apc-on/`.

## Summarize

Current summarizer is 2-way. Run it twice for the 3-way comparison:

```bash
# APC overhead on baseline:
./bench_summarize.py bench_results/main-apc-off bench_results/main-apc-on

# Kernel gain at APC-on:
./bench_summarize.py bench_results/main-apc-on bench_results/pr40172-apc-on
```

### Custom column labels

By default the table columns are headed `main` / `PR`. Override them with the
optional `--labels` flag, which takes a **colon-separated** pair where the
first half labels the baseline (first dir) and the second half labels the
comparison (second dir):

```bash
./bench_summarize.py \
  bench_results/main-apc-off \
  bench_results/main-apc-on \
  --labels "c68c55d4 APC=off:c68c55d4 APC=on"
```

The flag is optional — omitting it keeps the `main` / `PR` defaults. The
column width auto-expands to fit longer labels.

> Caveat: `:` is the separator, so labels themselves cannot contain a colon
> (no URLs, no `HH:MM` timestamps).
