## Summary

`kernels-mamba-test` sits right at the 40-min Buildkite budget. Build 82371 was
cancelled at 40:13 with 976/1041 tests finished; `test_mamba_ssm_ssd.py` alone
contributes ~1378s (~23 min) — 25 of its 94 tests exceed 30s each. This PR
removes two parametrization entries in that file whose coverage is fully
subsumed by neighbouring cases or existing values, buying **~347s** (~5.8 min,
measured on build 82371) of headroom without losing kernel coverage.

## Why target the top-K slowest tests

The runtime is not uniformly distributed. From build 82371's timing analysis
(methodology and awk scripts in `trim_testing.md`):

| File | Total | Count | Avg |
|---|---:|---:|---:|
| `test_mamba_ssm_ssd.py` | **1377.6 s** | 94 | 14.66 s |
| `test_replayssm_prefill_decode_equivalence_mamba2.py` | 235.7 s (partial) | 12 | 19.65 s |
| `test_mamba_ssm.py` | 214.3 s | 304 | 0.70 s |
| `test_memcpy_u64_tiled.py` | 162.0 s | 180 | 0.90 s |
| `test_precopy_mamba_align.py` | 134.1 s | 147 | 0.91 s |
| `test_causal_conv1d.py` | 123.4 s | 164 | 0.75 s |

Two files (`test_mamba_ssm_ssd.py` and `test_replayssm_prefill_decode_equivalence_mamba2.py`)
account for ~27 min of the 40-min budget. Within `test_mamba_ssm_ssd.py`,
25 of 94 tests exceed 30 s each and together consume **~1237 s (~21 min)**.
The observed top-10 individual tests each cost **60–77 s**.

The implication for a trimming strategy:

- **Broad-brush pruning is low-leverage.** Removing a full parametrization
  entry from `test_mamba_ssm.py` (0.70 s avg) saves under a second per test;
  removing one from `test_mamba_ssm_ssd.py`'s heavy long tail saves 30–70 s
  per test. A single top-10 test is worth ~100× a fast test.
- **Coverage risk is decoupled from wallclock cost.** A slow test isn't
  more valuable than a fast one — it just takes longer. Removing coverage-
  redundant slow tests is strictly better than removing coverage-redundant
  fast tests.
- **The top-K is where the budget lives.** Removing all "obviously
  redundant" fast tests across the suite would recover ~10 s; removing two
  redundant *slow* parametrizations from the top-30 recovers **~347 s
  measured on build 82371** — enough to move the job from over-budget
  to under.

This PR therefore targets only entries in `test_mamba_ssm_ssd.py`'s slow
tail: the removed `cont_batch case[4]` contains three top-10 slowest
individual tests (70.3 s, 54.2 s, 52.4 s), and the removed `d_head=8`
group contributes four tests to the top-30 at 33–46 s each. Every second
recovered is high-leverage.

## What is removed and why

### 1. `test_mamba_chunk_scan_cont_batch`: drop `seq_len_chunk_size_cases[4]`

`case[4] = (64, 256, 1, [(5,), (1,), (1,), (1,)])` — 1 example, 4 batches of
1-5 tokens under `chunk_size=256`.

Strict subset of `case[5] = (64, 256, 2, [(5, 30), (1, 2), (1, 2), (1, 2)])`
on every coverage axis:

| Axis | case[4] | case[5] |
|---|:-:|:-:|
| Sub-chunk sequences (all tokens < chunk_size) | ✓ | ✓ |
| Mid-physical-chunk sequence boundaries | · | ✓ |
| Multi-example continuous batching | · | ✓ (2 examples) |
| State passing across batches | ✓ | ✓ |

Case[5] catches everything case[4] catches, plus mid-chunk sequence boundaries
and multi-example batching.

**Measured savings (build 82371):** 182.58 s across 6 tests. Three of
those hit the observed top-10 slowest (70.3 s, 54.2 s, 52.4 s — all
`n_heads=4`); the other three (`n_heads=8`) run at ≤3.6 s each.

### 2. `test_mamba_chunk_scan_single_example`: drop `d_head=8`

`d_head` currently sweeps `[5, 8, 32, 128]`:

- `5` — non-power-of-2, unique code path (unaligned d_head)
- `8` — small power-of-2
- `32` — medium power-of-2
- `128` — large power-of-2

`d_head=8` is the weakest independent value: `5` uniquely covers the unaligned
case, and `32`/`128` cover the power-of-2 regime at two scales. `d_head=8`
sits between them in shape without exercising a distinct autotune tile size
that `32` doesn't already touch.

Empirically confirmed against the underlying kernels in
`vllm/model_executor/layers/mamba/ops/`:

- No `hdim == 8` / `head_dim == 8` branches in `ssd_chunk_scan.py`,
  `ssd_chunk_state.py`, or `ssd_bmm.py`.
- Autotune `BLOCK_SIZE_M/N` configs in those kernels start at ≥ 32; head-dim
  is padded to the block size and masked, so `hdim=5`, `hdim=8`, `hdim=32`
  all execute the same masked path. `hdim=5` already exercises the
  "hdim < BLOCK_SIZE" regime that `hdim=8` was covering.

This removal captures four tests in the observed top-30 slowest
(`d_head=8, n_heads=4` variants at 33-46s each).

**Measured savings (build 82371):** 164.68 s across 12 tests. Four
`n_heads=4` variants dominate at 33–46 s each (all in the observed
top-30); the other 8 (`n_heads ∈ {16, 32}`) run at ~1 s each.

## What is NOT removed

Candidates that looked prunable on first inspection but earn their keep:

- **`test_mamba_chunk_scan_cont_batch_prefill_chunking` (entire test).** After
  #24683 (chunk-aligned mamba2), `compute_varlen_chunk_metadata` guarantees
  no logical chunk crosses a physical boundary by construction, so the
  specific #23279 corner case is no longer reachable. But the
  split-and-resume correctness invariant this test guards —
  `kernel(concat(halves)) == concat(kernel(half1), kernel(half2, state=state_from_half1))`
  — is still a real, kernel-local property; each of its configs tests a
  distinct chunk-size × seqlen regime.
- **`test_mamba_chunk_scan_cont_batch` case[2] and case[6].** Case[2] tests
  sub-chunk sequences under a *small* `chunk_size=8` regime (different
  physical-chunk arithmetic than case[5]'s `chunk_size=256`); case[6]
  uniquely tests init-state exponential decay across many physical chunks
  with long sequences.
- **`test_mamba_chunk_scan_single_example` `n_heads=16`.** Middle
  tile-size regime — plausibly redundant, but the per-test savings are small
  enough that the coverage risk isn't worth it.

## Expected savings

### Measured on build 82371

Per-test wallclock extracted from `ci-82371-kernels-mamba-test.log` (awk
recipe in `trim_testing.md`). All 18 removed tests ran to completion in
that build, so this is an exact measurement, not a projection:

**`test_mamba_chunk_scan_cont_batch` — `cases4` group (6 tests, 182.6s):**

| Test ID (suffix) | Wallclock |
|---|---:|
| `cases4-5-4-itype0` | 70.28 s |
| `cases4-32-4-itype0` | 54.23 s |
| `cases4-16-4-itype0` | 52.42 s |
| `cases4-5-8-itype0` | 3.61 s |
| `cases4-32-8-itype0` | 1.05 s |
| `cases4-16-8-itype0` | 0.99 s |
| **Subtotal** | **182.58 s** |

**`test_mamba_chunk_scan_single_example` — `d_head=8` group (12 tests, 164.7s):**

| Test ID (suffix) | Wallclock |
|---|---:|
| `seq_len_chunk_size0-8-4-itype0` | 45.62 s |
| `seq_len_chunk_size1-8-4-itype0` | 43.48 s |
| `seq_len_chunk_size0-8-4-itype1` | 34.69 s |
| `seq_len_chunk_size1-8-4-itype1` | 33.20 s |
| `seq_len_chunk_size1-8-16-itype0` | 1.04 s |
| `seq_len_chunk_size0-8-32-itype1` | 1.02 s |
| `seq_len_chunk_size1-8-32-itype0` | 1.01 s |
| `seq_len_chunk_size0-8-16-itype1` | 0.97 s |
| `seq_len_chunk_size1-8-32-itype1` | 0.96 s |
| `seq_len_chunk_size1-8-16-itype1` | 0.94 s |
| `seq_len_chunk_size0-8-16-itype0` | 0.88 s |
| `seq_len_chunk_size0-8-32-itype0` | 0.87 s |
| **Subtotal** | **164.68 s** |

**Grand total: 347.26 s (~5.8 min) recovered from `test_mamba_ssm_ssd.py`.**

### Observation

The cost within each removed group is *strongly concentrated on
`n_heads=4`*: 4 of the 6 heavy tests (>30 s) live there, and every
`n_heads ∈ {8, 16, 32}` variant runs in ~1 s. Same reasoning holds for
`d_head=8`: only the four `n_heads=4` variants are top-30 slowest; the
other 8 are all near the noise floor.

This means the removals are *high-precision*: they cut ~7 tests worth
of budget-relevant wallclock (70+54+52+45+43+35+33 = **~332 s** from
7 heavy tests alone; the remaining 11 removed tests add only ~15 s).

### Projected wallclock

| | Before | After |
|---|---:|---:|
| Measured `test_mamba_ssm_ssd.py` total | 1377.6 s | ~1030 s |
| Projected full-suite (build 82371, extrapolated) | ~42:43 | **~36:56** |
| Headroom vs 40-min budget | −2:43 (over budget) | **+3:04** |

Note: the earlier estimate of ~445 s savings was optimistic — the actual
number is 347 s. Still enough to move the job from over-budget to under,
but the margin is tighter (~3 min headroom, not ~5 min).

## Duplicate work check

Per AGENTS.md §1, checked for open PRs and prior work on the same area
before drafting.

**Open-PR search:**

```
gh pr list --repo vllm-project/vllm --state open --search "test_mamba_ssm_ssd"
gh pr list --repo vllm-project/vllm --state open --search "kernels-mamba-test timeout"
gh pr list --repo vllm-project/vllm --state open --search "CI Perf mamba"
gh pr list --repo vllm-project/vllm --state open --search "mamba parametrize"
```

No open PR trims `tests/kernels/mamba/test_mamba_ssm_ssd.py` or targets the
`kernels-mamba-test` timeout.

**Prior in-tree work:**

- **#26538** (Fardin Hoque, Amazon) — `[CI Perf] Prune Tests in kernel/mamba`
  (merged as `577c72a227`). Pruned the same file's parametrization once
  before, removing overlapping combinations from `test_mamba_chunk_scan_cont_batch`
  and simplifying other kernel tests. This PR continues the same
  work on a *disjoint* set of remaining redundancies — the entries
  removed here were not touched by #26538, and the new build-82371 timing
  data (top-K analysis in `trim_testing.md`) identifies them as the
  highest-leverage remaining targets.

**Not busywork:** this is not a one-off mechanical cleanup — it's a
targeted response to a CI job that has already hit its timeout, backed
by per-test wallclock measurements and per-removal coverage-axis
analysis. The change is substantive under AGENTS.md's "low-value
busywork" bar because it (a) prevents CI cancellations on future PRs
touching `kernels/mamba` code, and (b) each removal is justified against
a specific redundancy, not chosen for size reduction alone.

## Test plan

- [ ] `pytest -v tests/kernels/mamba/test_mamba_ssm_ssd.py` — all remaining
  tests pass on H200.
- [ ] Full `pytest -v -s kernels/mamba` — fits comfortably under the 40-min CI
  budget (target: ≤37 min).
- [x] Test-count deltas verified with `pytest --collect-only`
  (see below).

### Verified test-count deltas (`pytest --collect-only`)

Ran `pytest --collect-only tests/kernels/mamba/test_mamba_ssm_ssd.py` on
`main` and on this branch:

| Test | main | branch | Δ |
|---|---:|---:|---:|
| `test_mamba_chunk_scan_single_example` | 48 | 36 | **−12** |
| `test_mamba_chunk_scan_cont_batch` | 42 | 36 | **−6** |
| `test_mamba_chunk_scan_cont_batch_prefill_chunking` | 4 | 4 | 0 |
| **Total** | **94** | **76** | **−18** |

The 18 removed test IDs (diff of the two collected sets):

**`test_mamba_chunk_scan_single_example` — `d_head=8` group (12 tests)**

```
[seq_len_chunk_size0-8-{4,16,32}-itype{0,1}]   (6 tests)
[seq_len_chunk_size1-8-{4,16,32}-itype{0,1}]   (6 tests)
```

**`test_mamba_chunk_scan_cont_batch` — `cases4` group (6 tests)**

Pre-trim `cases4 = (64, 256, 1, [(5,), (1,), (1,), (1,)])`. Post-trim
collection re-indexes (pre `cases5` → post `cases4`, pre `cases6` → post
`cases5`) — those are renames, not removals. The 6 actually-removed IDs:

```
[seq_len_chunk_size_cases4-{5,16,32}-{4,8}-itype0]
```

Structural predictions (`2×3×2 = 12` for `d_head=8` and `3×2×1 = 6` for
`cases4`) match the collection diff exactly.

## Model evaluation

Not applicable — this PR touches only test parametrization; no model, kernel,
or serving code changes.

## AI assistance

This PR was drafted with Claude Code assistance. The coverage-axis analysis
and the "strict subset" argument for case[4] were reviewed against the actual
behaviour of `compute_varlen_chunk_metadata` and the test bodies before
proposing removals. The `d_head=8` removal was cross-checked by grepping the
underlying Triton kernels for hardcoded head-dim branches — none exist, and
autotune tiles start at 32, so `hdim=5` already covers the small-hdim masked
path.
