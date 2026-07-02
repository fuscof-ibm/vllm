# Skipping the mamba postprocess kernel — investigation notes

This document walks through the "skip the mamba postprocess when
provably a no-op" optimization: the original upstream proposal, how it
was adapted to current vLLM `main`, a follow-up refactor, and the H100
benchmark data that surfaced from the exercise.

Timeline in this branch:

| Commit | Author-date | Summary |
|---|---|---|
| `4e5ca89cfe98121642d76b40e32a006f4d0fbf3b` | 2026-07-01 | Baseline `main` at investigation time |
| `b199bb3b6c558166953619a340221b55d84805b7` | 2026-07-01 | Port of PR 42574's idea to current `main` |
| `eb67c9e80ff212603372c1b273e7c63494a9d6c9` | 2026-07-02 | Fusion of the CPU predicate into the staging loop |

---

## 1. The original PR — `vllm-project/vllm#42574`

Upstream context (Nemotron-Super-120B-A12B-NVFP4, MTP=3, GB300 single
GPU, aiperf 480-req, synthetic acceptance-length 3): every decode
iteration in `_update_states_after_model_execute` did a **blocking**
`.cpu().numpy()` on `num_accepted_tokens.gpu[:num_reqs]` and then fed
the result into a Python `postprocess_mamba` call:

```python
for i, num_tokens in enumerate(
    self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()
):
    self.input_batch.num_accepted_tokens_cpu[i] = num_tokens
# ... postprocess_mamba(...)
```

The `.cpu()` stalls the EngineCore CPU until the GPU has finished the
target forward + sampler + sum kernel. The insight was that
`postprocess_mamba` only does work when a request crosses a mamba block
boundary this iteration:

```
aligned_new_computed_tokens >= num_tokens_running_state
```

Because `num_accepted_tokens` is bounded by `n_draft + 1` (i.e.
`num_speculative_tokens + 1`), we can **decide on the CPU** whether any
request could possibly cross a boundary this step. For the typical
Nemotron config (`mamba_block_size = 4336`, `num_speculative_tokens =
3`), the worst-case advance is 4 tokens/step and the boundary is
provably uncrossable in ~98% of decode steps.

When we can prove the postprocess would be a no-op, PR 42574:

1. Issues an **async, non-blocking** device-to-host copy of
   `num_accepted_tokens` into the existing pinned buffer.
2. Records `num_accepted_tokens_event`.
3. Skips the `postprocess_mamba` call entirely.
4. Lets the existing `event.synchronize()` in `_prepare_inputs`
   (which fires after the *next* iteration's draft forwards) absorb
   the deferred wait. By that time the GPU has long since finished the
   copy, so the synchronize is essentially free.

When the skip condition cannot be proven, the code falls back to the
original blocking `.cpu()` + `postprocess_mamba` path.

Reported numbers on Nemotron-Super-120B-A12B-NVFP4 (MTP=3, GB300):

|  | baseline | patched | Δ |
|---|---|---|---|
| Overall TPS | 65,945 | 77,411 | **+17.4%** |
| Decode TPS | 2,153 | 2,495 | +15.9% |
| Inter-token latency | 7.43 ms | 6.41 ms | −13.7% |
| Avg request latency | 8,700 ms | 7,412 ms | −14.8% |
| Bench wall time | 265 s | 226 s | −14.8% |

nsys audit confirmed the mechanism was **sync-deferral, not
GPU-work reduction**: total GPU kernel time was unchanged
(47.46 s → 47.47 s over the profile window), but slow
`cudaMemcpyAsync` calls (>1 ms) dropped from 1,081 to 80 and wait time
migrated from `cudaMemcpyAsync` into `cudaEventSynchronize`.

---

## 2. Applying it on current `main` — commit `b199bb3b6`

The direct patch from PR 42574 does not apply cleanly on
current `main` (`4e5ca89cf`) because the code it targeted is gone.
PR #40172 replaced the Python `.cpu().numpy()` + `postprocess_mamba`
path with a **fully-fused Triton kernel**
(`mamba_utils.postprocess_mamba_align_gpu`) that performs state copies
and the per-request accepted-token update entirely on the GPU. The
per-step blocking sync no longer exists on `main`.

However, the underlying observation from PR 42574 still applies:
`postprocess_mamba_fused_kernel` is a no-op across every thread of its
grid whenever no request can cross a boundary this step. In that
regime we're paying:

- The D→D init copy of `num_accepted_tokens_out` inside
  `run_fused_postprocess`.
- A `postprocess_mamba_fused_kernel` launch of grid
  `(num_reqs, num_layers × num_state_types)`.
- A D→H async copy of the kernel's output.

The `postprocess_skip` branch adapts the idea to the fused-kernel path:

- Introduces `can_skip_mamba_postprocess` in `vllm/v1/worker/mamba_utils.py`
  — a CPU-only predicate that walks all requests and returns True iff
  no request can cross a boundary this step, mirroring the kernel's
  own `needs_copy = aligned_new_computed >= num_tokens_running_state`
  check.
- In `_update_states_after_model_execute` (`gpu_model_runner.py`), when
  the predicate returns True, skips `postprocess_mamba_align_gpu`
  entirely and only performs a non-blocking D→H copy of
  `num_accepted_tokens.gpu` into the CPU tensor, then records the
  event.

The `mamba_block_size` used by the predicate is read from
`bufs.postprocess_align.block_size`, matching the value the kernel
uses. Everything else — staging in `_prepare_inputs`, event handling,
downstream synchronize — is unchanged.

**Semantics** relative to the pre-existing code: the predicate reads
`num_scheduled`, `num_computed_tokens`, and
`scheduled_spec_decode_tokens`. All three are stable between
`_prepare_inputs` and `_update_states_after_model_execute` (the
existing invariant documented on `stage_postprocess_metadata_to_gpu`),
so the pre-forward and post-forward predicate calls return the same
bool for every request.

---

## 3. Fusing predicate + staging — commit `eb67c9e8`

The `b199bb3b6` design pays the predicate as a **separate per-request
loop** in `_update_states_after_model_execute`. Meanwhile,
`stage_postprocess_inputs_to_gpu` in `_prepare_inputs` runs two more
per-request loops filling the four staging buffers, and issues four
H→D copies — *unconditionally*, including on steps where we already
know the kernel will be skipped.

The fusion collapses this into a single per-request pass:

- `stage_postprocess_inputs_to_gpu` is rewritten as one loop over
  `req_ids[:num_reqs]` that fills all four numpy views
  (`mamba_state_idx` / `num_scheduled` / `num_computed` / `num_draft`)
  **and** updates `can_skip` inline using the same three dict lookups
  the loop already needs (`num_scheduled[req_id]`,
  `requests[req_id].num_computed_tokens`,
  `scheduled_spec_decode_tokens.get(req_id, ())`).
- When the predicate proves the kernel would be a no-op for every
  request, the four `.copy_to_gpu()` calls are **skipped** — those
  buffers are only consumed by the fused postprocess kernel, which we
  know won't run this step.
- The decision is cached on
  `MambaSpecDecodeGPUContext.skip_next_postprocess`. In
  `_update_states_after_model_execute`, the runner reads that flag and
  either issues the async D→H copy of `num_accepted_tokens` (and
  records the event), or falls back to `postprocess_mamba_align_gpu`
  as before.

Three helpers are deleted as orphans:

- `can_skip_mamba_postprocess` (predicate is now inline in the staging
  loop).
- `stage_mamba_state_idx_to_gpu` and
  `stage_postprocess_metadata_to_gpu` (both collapsed into the single
  fused pass).

**Semantic equivalence to `b199bb3b6`:** the predicate consumes the
same three inputs (`num_scheduled`, `num_computed_tokens`,
`scheduled_spec_decode_tokens`) and the same `num_reqs` and `req_ids`.
Because `num_computed_tokens` is only written by `_update_states` (top
of `execute_model`) and `deferred_state_corrections_fn` (start of
`_prepare_inputs`, before staging) — with no writer between staging
and `_update_states_after_model_execute` — the pre-forward and
post-forward predicates return the same bool on every step. The staged
buffers are only read by the kernel we're gating, so dropping the four
`.copy_to_gpu()` calls on skip steps is invisible to correctness.

Per-step CPU-side work, before vs after fusion:

| step type | `b199bb3b6` (split) | `eb67c9e8` (fused) |
|---|---|---|
| **Skip step** | 2 Python loops + 1 predicate loop + 4 H→D copies | 1 fused loop, 0 H→D copies |
| **Non-skip step** | 2 Python loops + 1 predicate loop + 4 H→D copies + kernel | 1 fused loop + 4 H→D copies + kernel |

---

## 4. Benchmark results — H100 / Qwen3.5-9B / MTP=2

Workload
([`PR_optimize_postprocess/skip/H100/results_postprocess_skip_Qwen3.5-9B.md`](H100/results_postprocess_skip_Qwen3.5-9B.md)):

- Model: `Qwen/Qwen3.5-9B`
- Speculative config: `{"method":"qwen3_next_mtp","num_speculative_tokens":2}`
- Random dataset, input 256, output 2048, 500 prompts
- Concurrency 32, three runs per commit
- Hardware: single H100, `-tp 1 -pp 1 -dp 1`

Following the convention of discarding the first (warm-up) run per
commit and averaging runs 2 and 3.

### Per-run summary (runs 2 and 3)

| Commit | Run | Total tok/s | Output tok/s | Median TPOT (ms) | Median ITL (ms) |
|---|---|---|---|---|---|
| `4e5ca89cf` (main) | 2 | 4546.10 | 4040.98 | 7.23 | 13.79 |
| `4e5ca89cf` (main) | 3 | 4664.49 | 4146.21 | 7.17 | 13.79 |
| `b199bb3b6` (split skip) | 2 | 4420.57 | 3929.40 | 7.42 | 13.98 |
| `b199bb3b6` (split skip) | 3 | 4483.26 | 3985.12 | 7.38 | 13.96 |
| `eb67c9e8` (fused) | 2 | 4535.72 | 4031.75 | 7.45 | 13.96 |
| `eb67c9e8` (fused) | 3 | 4463.64 | 3967.68 | 7.47 | 13.95 |

### Means (runs 2 and 3)

| Metric | `4e5ca89cf` main | `b199bb3b6` split | `eb67c9e8` fused |
|---|---|---|---|
| Total tok/s | **4605.30** | 4451.92 | 4499.68 |
| Output tok/s | 4093.60 | 3957.26 | 3999.72 |
| Median TPOT (ms) | 7.20 | 7.40 | 7.46 |
| Median ITL (ms) | 13.79 | 13.97 | 13.96 |

### Δ vs `main` (`4e5ca89cf`)

| Metric | `b199bb3b6` split | `eb67c9e8` fused |
|---|---|---|
| Total tok/s | **−3.33 %** | **−2.29 %** |
| Median TPOT | +2.78 % | +3.61 % |
| Median ITL | +1.31 % | +1.23 % |

---

## 5. Findings

**PR 42574's win does not carry over to H100 / Qwen3.5-9B / MTP=2.**
Both skip implementations regress against `main` on this workload,
though the fusion recovers roughly 1 percentage point of throughput
versus the split design.

Reasoning:

1. **The kernel-launch saving is small on this hardware.** On H100 the
   fused postprocess kernel is very cheap when it early-outs: the grid
   is `num_reqs × num_layers × num_state_types` (≈ 32 × small), each
   thread checks `needs_copy` and returns. A per-step launch is on the
   order of 10-20 µs. PR 42574's win came from a *different* mechanism
   — deferring a **blocking** `.cpu()`. There is no per-step blocking
   sync to defer on current `main`.

2. **The four H→D staging copies are tiny.** At `num_reqs=32` each
   `.copy_to_gpu(num_reqs)` transfers 32 × 4 B = 128 B; four copies is
   512 B/step. PCIe transfer time is negligible; the cost is the ~1-3 µs
   of per-call driver overhead, so ~4-12 µs of CPU-side savings per
   skip step.

3. **CPU-side loop cost dominates.** At num_reqs=32 the per-step
   Python overhead of one extra loop (`can_skip_mamba_postprocess` in
   the split version) is a handful of µs. That is comparable to *or
   larger than* the launch + copy savings — hence the regression in
   `b199bb3b6`.

4. **Fusion helps but is not enough.** Merging the predicate into the
   existing staging loop removes an entire per-step Python loop and
   drops four H→D copies on skip steps. The measured gain over the
   split version (~+1 pp total throughput) matches the expected
   direction. It does not, however, close the gap to `main` on this
   workload.

5. **The optimization is workload-shaped.** PR 42574's regime
   (Nemotron-Super-120B, MTP=3, ~480 requests, GB300, block_size=4336)
   is very different from this one (Qwen3.5-9B, MTP=2, 32 concurrent
   requests, H100). Larger models make per-step launch overhead a
   smaller fraction of decode; higher MTP + smaller block_size ratios
   change the skip rate. The measured regression on H100/9B/MTP=2 is
   consistent with the CPU predicate cost outweighing the launch
   saving at this scale.

---

## 6. Suggested next steps

Before iterating further on the skip design, confirm the premise on
this workload:

1. **Measure skip rate.** Add two counters
   (`ctx.n_skip += int(can_skip); ctx.n_total += 1`) and log every N
   seconds. If skip rate is under ~90%, the optimization cannot win —
   we are paying the extra branch every step for a very small share of
   savings.

2. **Verify `mamba_block_size`.** Print
   `bufs.postprocess_align.block_size` at startup for the Qwen3-Next
   configuration used here. Combined with the mean `num_accepted +
   n_draft` per step, this predicts the theoretical maximum skip rate.

3. **If the counters say the premise holds**, the next-order
   optimization is to **amortize the predicate across many steps** by
   tracking a batch-level scalar `steps_until_full_check` computed
   from the smallest boundary distance across all requests. On steps
   where the batch composition is stable and the counter is positive,
   the fast path is a single integer decrement — near-zero CPU cost.
   The full O(num_reqs) loop only runs on counter expiration or when
   the batch changes.

4. **If the counters say the premise fails on H100/9B/MTP=2** but
   holds on larger regimes, gate the optimization behind a heuristic
   (e.g. on `num_speculative_tokens ≥ 3` and `mamba_block_size` above
   some threshold) so it only fires where PR 42574's numbers reproduce.
