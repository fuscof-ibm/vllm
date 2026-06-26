# `postprocess_mamba_fused_kernel`: 2D vs 3D grid analysis

Investigation of a proposed change to the fused mamba postprocess kernel
([vllm/v1/worker/mamba_utils.py](../vllm/v1/worker/mamba_utils.py)) that
promotes the byte-chunk index to a third grid axis. Benchmarks on Qwen3.5-9B
at H100 / concurrency 32 show a consistent **~10% throughput regression**
versus the existing 2D grid. This document records why the 2D grid is
already well-matched to that workload and why the 3D variant pays
significantly more overhead than it gains.

---

## Context

The kernel runs once per scheduler step in the spec-decode + hybrid +
align-mode path. For each request and each (layer, state-type) pair it:

1. Decides whether a per-block state copy is needed (depends on
   `num_accepted_tokens`, `num_scheduled`, `num_computed`, `num_draft`,
   `block_size`).
2. If yes, copies the appropriate slice of the Mamba state from a source
   physical block to a destination physical block.

There are two state types per Mamba layer:

- **Conv state** — small sliding-window buffer (`conv_kernel_size + num_spec - 1`
  positions of a ~few-thousand-wide vector).
- **Temporal SSM state** — the recurrent matrix carried across timesteps;
  for GDN/Mamba2-style models this is `(num_v_heads, head_v_dim, head_k_dim)`
  per block, typically MB-scale.

The kernel is launched on every decode step regardless of whether any
request actually needs a copy. Decision logic and copy are fused into one
launch to avoid an extra CPU↔GPU sync.

---

## Part 1 — The 2D grid in detail

### Launch shape

```python
total_states = num_layers * num_state_types
grid = (num_reqs, total_states)
postprocess_mamba_fused_kernel[grid](...)
```

Each Triton program is uniquely identified by:

```python
req_idx   = tl.program_id(0)   # 0 .. num_reqs - 1
state_idx = tl.program_id(1)   # 0 .. total_states - 1
```

`state_idx` is a flattened index over `(mamba_layer_idx, state_type)`. All
per-state metadata (`base_addr`, `block_stride`, `elem_size`,
`inner_size`, `conv_width`, `dim_row_count`, `dim_row_stride`,
`group_idx`) is stored in flat tensors indexed by `state_idx`.

#### How the flattening works

Mamba layers and full-attention layers live in separate KV-cache groups.
The kernel only sees layers in the mamba group(s):
`kv_cache_config.kv_cache_groups[mamba_group_id].layer_names`. For
Qwen3.5-9B that's the 24 linear-attention layers; the 8 full-attention
layers are absent from `state_idx` entirely.

The flattening is built once at engine init in
`MambaSpecDecodeGPUContext.initialize_from_forward_context`
([vllm/v1/worker/mamba_utils.py:457](../vllm/v1/worker/mamba_utils.py#L457)):

```python
idx = 0
for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
    layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
    for layer_name in layer_names:
        attention = forward_context[layer_name]
        kv_caches: list[torch.Tensor] = attention.kv_cache
        for state_type_idx, state in enumerate(kv_caches):
            self.state_base_addrs[idx]    = state.data_ptr()
            self.state_block_strides[idx] = ...
            # ... more per-state metadata ...
            self.state_group_indices[idx] = group_local_idx
            idx += 1
```

So the iteration order is **group → layer → state_type**. With a single
mamba group (the common case), this reduces to **mamba_layer →
state_type**.

#### State-type ordering

`enumerate(kv_caches)` follows the order in which the attention layer
populated its `kv_cache` list. For GDN /  Mamba-style models that list is
`[conv_state, temporal_ssm_state]` — confirmed by
`MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func` returning
`(get_conv_copy_spec, get_temporal_copy_spec)` in
[vllm/model_executor/layers/mamba/mamba_utils.py:352](../vllm/model_executor/layers/mamba/mamba_utils.py#L352)
and by `kv_cache[0]` being unpacked as `conv_state` in
[vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1311](../vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1311).

So for a single-group Mamba model:

```
state_idx = 2 * L + s

where:
  L ∈ [0, num_mamba_layers)        — index among mamba layers only,
                                     NOT the transformer layer index
  s ∈ {0, 1}                       — 0 = conv state, 1 = temporal SSM state
```

#### The crucial subtlety: `L` is *not* the transformer layer index

For Qwen3.5-9B the 32 transformer layers are arranged as
`[linear, linear, linear, full] × 8`. Mapping transformer-layer index to
mamba-layer index:

| Transformer layer | Type | Mamba index `L` | `state_idx` for (conv, temporal) |
|---|---|---|---|
| 0 | linear  | 0  | (0, 1) |
| 1 | linear  | 1  | (2, 3) |
| 2 | linear  | 2  | (4, 5) |
| 3 | full    | —  | — (not in grid) |
| 4 | linear  | 3  | (6, 7) |
| 5 | linear  | 4  | (8, 9) |
| 6 | linear  | 5  | (10, 11) |
| 7 | full    | —  | — (not in grid) |
| ... | ... | ... | ... |
| 30 | linear | 23 | (46, 47) |
| 31 | full   | —  | — (not in grid) |

`state_idx` ranges `[0, 48)`, dense over the 24 mamba layers, with no
holes for the full-attention layers.

#### Per-state-type sizes are identical across mamba layers

A useful property for reasoning about the kernel: all conv states across
all mamba layers share the same shape, same dtype, same `copy_size` —
they were built from the same model config. Likewise for temporal states.
So the even-indexed `state_idx` entries are all "80 KB conv copies" and
the odd-indexed ones are all "2 MB temporal copies" (for Qwen3.5-9B at
num_spec=2). Per-layer divergence in the metadata tensors (`base_addr`,
`group_idx`) is only in *which physical buffer* to copy from / to, not
in copy size.

This is why `max_chunks` in the 3D grid is one global scalar rather than
a per-state vector — every mamba layer's temporal state has the same
upper-bound chunk count, and the global `max_chunks` is simply the larger
of `(conv_chunks, temporal_chunks)`.

#### Multi-group case

When the model has multiple mamba groups (uncommon, but supported), the
loop visits group 0's layers first, then group 1's, etc. Within each
group the same `mamba_layer → state_type` order applies.
`state_group_indices[state_idx]` records which group this state belongs
to, so the kernel can look up the correct block-table pointer at
[vllm/v1/worker/mamba_utils.py:110-116](../vllm/v1/worker/mamba_utils.py#L110-L116):

```python
group_idx          = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
group_base_addr    = tl.load(block_table_ptrs_ptr + group_idx)
block_table_typed  = group_base_addr.to(tl.pointer_type(tl.int32))
block_table_base   = block_table_typed + req_idx * block_table_stride_req
```

Different groups have independently allocated physical blocks, so a
state in group 1 cannot use group 0's block table.

### What one program does

A single program's lifecycle, in order:

1. **Load per-request scalars** (5 × int32):
   `num_accepted_tokens`, `mamba_state_idx (src_block_idx)`,
   `num_scheduled_tokens`, `num_computed_tokens`, `num_draft_tokens`.
2. **Compute the decision**:
   ```
   num_tokens_running_state = num_computed + num_scheduled - num_draft
   new_num_computed         = num_tokens_running_state + num_accepted - 1
   aligned_new_computed     = (new_num_computed // block_size) * block_size
   needs_copy               = aligned_new_computed >= num_tokens_running_state
   ```
   If `not needs_copy`, return immediately.
3. **Compute copy parameters**:
   ```
   accept_token_bias = aligned_new_computed - num_tokens_running_state
   dest_block_idx    = aligned_new_computed // block_size - 1
   ```
4. **Load per-state metadata** (≈6 scalars).
5. **Resolve the block table**: load this state's group index, then the
   group's block-table base pointer, then the source and destination block
   IDs for this request. Block IDs are widened to int64 here to prevent
   the per-block stride from overflowing int32 on large caches.
6. **Branch on state type** (`conv_width > 0` for conv states):
   - **Temporal**: `src_addr = base + actual_src_block_id * block_stride`,
     `copy_size = inner_size * elem_size`.
   - **SD conv**: `src_addr = base + src_block_id * block_stride + bias_offset`,
     `copy_size = (conv_width - bias) * inner_size * elem_size`.
   - **DS conv**: row-by-row copy with `dim_rows` and per-row bias slicing.
7. **Run the serial inner loop**:
   ```python
   offsets = tl.arange(0, COPY_BLOCK_SIZE)   # COPY_BLOCK_SIZE = 1024
   for i in range(0, copy_size, COPY_BLOCK_SIZE):
       mask     = (i + offsets) < copy_size
       curr_src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))
       curr_dst = (dst_addr + i + offsets).to(tl.pointer_type(tl.uint8))
       data     = tl.load(curr_src, mask=mask)
       tl.store(curr_dst, data, mask=mask)
   ```

The number of inner-loop iterations is `cdiv(copy_size, 1024)`.

### Two levels of parallelism

This is the property that matters most for understanding the design.

**Level 1 — programs across SMs (the "grid").** The CUDA driver hands all
`num_reqs × total_states` programs to the SM scheduler, which distributes
them round-robin onto SMs. Each program is one CTA; with ~16 CTAs resident
per H100 SM at this resource profile, total concurrent capacity is
roughly `132 × 16 ≈ 2,100 CTAs`. If the launch fits inside that capacity,
all programs run in a single "wave" and the kernel completes in roughly
one program's wall time. If it overshoots, the scheduler must wait for
programs to retire before launching the next wave — adding latency
proportional to the overshoot.

**Level 2 — SIMT lanes inside a program (the `tl.arange`).** The line
`offsets = tl.arange(0, 1024)` declares 1024 byte-lanes that are processed
in true SIMT parallelism by the program's threads (default 4 warps =
128 threads per CTA, so 8 byte-lanes per thread per iteration, coalesced
into a single 1024-byte HBM transaction).

So a single program is doing
`cdiv(copy_size, 1024) × 1024 bytes ≈ copy_size` of HBM-bound traffic, in
one wave of CTAs alongside ~2,000 other programs.

### Cost structure

- **Per-program fixed cost** (prelude): ~12 scalar loads, a few arithmetic
  ops, conditional branches. Order of nanoseconds-to-microseconds depending
  on cache state.
- **Per-iteration variable cost** (inside the loop): one coalesced
  1024-byte load + one coalesced 1024-byte store. HBM-bound.
- **Metadata is loaded once per program** and held in registers for the
  duration of the loop — there is no redundant L1/L2 traffic across loop
  iterations.

Critical implication: **the prelude is amortized over `copy_size / 1024`
iterations**. As `copy_size` grows, the kernel approaches the HBM
bandwidth peak per program.

---

## Part 2 — Concrete numbers for Qwen3.5-9B

From the published [Qwen3.5-9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json):

```json
"num_hidden_layers":        32,
"full_attention_interval":  4,
"linear_conv_kernel_dim":   4,
"linear_key_head_dim":      128,
"linear_value_head_dim":    128,
"linear_num_key_heads":     16,
"linear_num_value_heads":   32,
"mamba_ssm_dtype":          "float32",
"dtype":                    "bfloat16"
```

`layer_types` follows a `[linear, linear, linear, full]` × 8 pattern.

### Layer accounting

- 32 total transformer layers
- `full_attention_interval = 4` → 1 of every 4 is full attention
- **24 linear-attention (Mamba/GDN) layers**, 8 full-attention layers
- `num_state_types = 2` (conv + temporal SSM)

So:

```
total_states = 24 × 2 = 48
```

### State shapes (per block, TP=1, num_spec=2 since acceptance_length ≈ 1.85)

From `MambaStateShapeCalculator.gated_delta_net_state_shape` in
[vllm/model_executor/layers/mamba/mamba_utils.py:213](../vllm/model_executor/layers/mamba/mamba_utils.py#L213):

**Conv state** (default SD layout — `state_len` outer, `dim` inner):

```
conv_dim   = key_head_dim × num_key_heads × 2 + value_head_dim × num_value_heads
           = 128 × 16 × 2 + 128 × 32
           = 4096 + 4096
           = 8192
state_len  = conv_kernel - 1 + num_spec = 4 - 1 + 2 = 5
shape      = (5, 8192)
elems      = 40,960
bytes (bf16) = 40,960 × 2 = 81,920 B ≈ 80 KB
```

**Temporal SSM state**:

```
shape      = (num_v_heads, head_v_dim, head_k_dim) = (32, 128, 128)
elems      = 524,288
bytes (fp32) = 524,288 × 4 = 2,097,152 B = 2 MB
```

The temporal state dominates by 25×.

### 2D grid program count and per-program work

At benchmark concurrency `num_reqs = 32`:

```
2D grid       = (32, 48)        → 1,536 programs
H100 capacity ≈ 132 SMs × 16 CTAs ≈ 2,100 concurrent CTAs
```

**1,536 programs fit comfortably in one wave** — roughly 12 programs land
on each SM, well within the 16-CTA limit.

Inner-loop iteration counts per program:

| State type | `copy_size` | iterations of 1024-byte chunks |
|---|---|---|
| Conv (bf16) | 80 KB | 80 |
| Temporal SSM (fp32) | 2 MB | **2,048** |

So a temporal-state program does 2,048 iterations of `load 1024B → store
1024B`. At H100 HBM3 peak (~3 TB/s), 2 MB of one-way traffic takes ~0.7 µs;
with read + write the program spends ~1.4 µs purely on HBM, plus loop
overhead. Programs run concurrently, so the kernel's wall-clock is bounded
by the slowest program (~2–3 µs), not the sum.

### Occupancy estimate

Under the model that the 2D grid fits in one wave:

```
SMs in use per wave   = min(132, ceil(1536 / 16)) = 96 SMs busy
Programs / SM         = 1536 / 96 ≈ 16 (saturates the slot limit)
Wall-clock per launch ≈ max-program-time ≈ 2–3 µs
```

In practice this is a healthy regime: the kernel is HBM-bound (the right
limiter for a memcpy), prelude overhead is fully amortized (~12 scalar
loads vs. 2,048 iterations of useful work), and there is no second-wave
scheduling delay.

### When the 2D grid *does* break down

The single-wave assumption fails as `num_reqs × total_states` drops below
the SM count. For Qwen3.5-9B:

| `num_reqs` | programs | SM utilization | regime |
|---|---|---|---|
| 1 | 48 | 36% of one SM-row | severely undersaturated |
| 4 | 192 | ~1.5 SM-rows | undersaturated |
| 16 | 768 | ~half wave | mild headroom |
| 32 | 1,536 | full wave | **target regime** |
| 64 | 3,072 | 1.5 waves | mild overshoot |
| 256 | 12,288 | ~6 waves | scheduler-bound |

The 3D grid was motivated by the `num_reqs ≤ 4` cases — the rows above
the target regime, where SMs are idle. We will see in Part 4 why the
remediation does not pay off at the benchmark's actual operating point.

---

## Part 3 — The 3D grid: what it changes

### Launch shape

```python
grid = (num_reqs, total_states, max_chunks)
```

with a new third axis:

```python
chunk_idx = tl.program_id(2)
```

`max_chunks` is precomputed once at engine init as the upper bound, across
all states, of `cdiv(state_copy_size, COPY_BLOCK_SIZE)`. Each program is
now responsible for **one `COPY_BLOCK_SIZE`-byte slice** of the
`(req, state)` copy:

```python
chunk_offset = chunk_idx * COPY_BLOCK_SIZE
if chunk_offset >= copy_size:
    return                                    # this state needs fewer chunks
mask = (chunk_offset + offsets) < copy_size
# one load + one store, no inner loop
```

The DS-conv path linearises `(d, i_in_row)` into `chunk_idx` similarly via
`chunks_per_row = cdiv(per_row_bytes, COPY_BLOCK_SIZE)` and integer divmod.

### Intent

For undersaturated launches (small `num_reqs`), spreading the byte copy
of one `(req, state)` across many programs lets idle SMs participate in
that copy. Bigger copies fan out into more programs; smaller copies have
early-exit programs that cost only the prelude.

### What this costs

`max_chunks` is the **upper bound across all states**. Every (`req`,
`state`) launch carries that worst-case program count, regardless of the
state's actual copy size or whether `needs_copy` is even true for that
request.

---

## Part 4 — Why the 3D grid loses at `num_reqs = 32`

### Program-count explosion for Qwen3.5-9B

With `COPY_BLOCK_SIZE = 1024`:

```
chunks for conv copy     = cdiv(81,920, 1024)    = 80
chunks for temporal copy = cdiv(2,097,152, 1024) = 2,048
max_chunks               = max(80, 2,048)        = 2,048
```

```
3D grid       = (32, 48, 2048) → 3,145,728 programs
vs 2D launch                     →     1,536 programs
explosion factor                 → 2,048×
```

H100 capacity is ~2,100 concurrent CTAs. **The 3D launch requires ~1,500
waves**, vs the 2D's single wave.

### Where the work actually lands

| Population | Count | What it does |
|---|---|---|
| Real temporal chunks (`needs_copy=True`) | up to 32 × 24 × 2,048 = 1,572,864 | 1 KB of real HBM copy each |
| Real conv chunks (`needs_copy=True`) | up to 32 × 24 × 80 = 61,440 | 1 KB of real HBM copy each |
| Conv early-exits | up to 32 × 24 × 1,968 = 1,511,424 | prelude + early return |
| `needs_copy=False` early-exits | the rest | prelude + early return |

Even in the optimistic case where every request needs a copy, the kernel
launches **~1.5 M useless conv early-exits** alongside the real work.
Realistic decode steps have `needs_copy=False` for many `(req, state)`
pairs, multiplying the early-exit count further.

### Why early-exits aren't free

An "early-exit program" still pays:

1. **Dispatch onto an SM** — register allocation, shared memory reservation,
   PC initialization. Each early-exit CTA occupies one of the SM's 16
   resident slots until it retires, blocking other programs from filling
   that slot.
2. **Prelude execution** — loads the 5 per-request scalars, evaluates the
   `needs_copy` decision, loads state metadata, computes chunk
   coordinates, performs the `chunk_offset >= copy_size` check, then
   returns. Order of microseconds per CTA, magnified across millions
   of programs.
3. **Redundant metadata loads** — in the 2D variant, the prelude ran
   *once* per `(req, state)`. In the 3D variant it runs `max_chunks` times
   per `(req, state)`. The values are L1/L2-cached, but cache bandwidth
   and warp scheduler slots are still consumed.

### Per-program work shrinks toward the noise floor

In the 2D grid, a temporal-state program does ~2 MB of HBM-bound copy and
amortises a ~12-load prelude over 2,048 inner iterations. That ratio
**(work / setup)** is excellent — the kernel approaches HBM peak per program.

In the 3D grid (with `COPY_BLOCK_SIZE = 1024`), a temporal-state program
does **1 KB** of copy, with the same prelude. The work/setup ratio
collapses by 2,048×. Each program now spends most of its life in the
prelude, not the copy.

### Measured impact

From [results_3dgrid_qwen_h100.md](results_3dgrid_qwen_h100.md), median of
runs 2 and 3 per commit (concurrency 32):

| Metric | 2D baseline | 3D grid | Delta |
|---|---|---|---|
| Total throughput (tok/s) | 4,445 | 4,006 | **−9.9%** |
| Median TPOT (ms) | 7.50 | 8.27 | **+10.3%** |
| Median ITL (ms) | 14.04 | 15.77 | **+12.3%** |
| Median E2EL (ms) | 15,410 | 17,000 | **+10.3%** |
| Acceptance length | 1.83 | 1.86 | unchanged |

Acceptance length is unchanged, confirming the change is correct — the
regression is pure overhead, not a behavioural difference.

### The size argument was directionally right, but for a different operating point

The 3D grid genuinely helps the cases the 2D grid is designed *not* to
optimise for: low batch (`num_reqs ≤ 4`), where the 2D launch leaves most
SMs idle. The mistake in the original analysis was extrapolating that
benefit to the production operating point. At `num_reqs = 32` the 2D grid
already saturates the GPU, so adding more programs cannot speed anything
up — it can only add overhead.

---

## Part 5 — Implications for a fix

The diagnosis is clear: `COPY_BLOCK_SIZE = 1024` is the dominant problem,
not the 3D grid concept per se. Each program does too little work for the
fixed prelude cost it carries.

### Grid sizes vs `COPY_BLOCK_SIZE` (Qwen3.5-9B, `num_reqs = 32`)

| `COPY_BLOCK_SIZE` | `max_chunks` | 3D grid total | Programs / SM (H100) | Waves |
|---|---|---|---|---|
| 1 KB (current) | 2,048 | 3,145,728 | ~23,830 | ~1,500 |
| 4 KB | 512 | 786,432 | ~5,957 | ~375 |
| 16 KB | 128 | 196,608 | ~1,490 | ~94 |
| **64 KB** | **32** | **49,152** | **~372** | **~24** |
| 128 KB | 16 | 24,576 | ~186 | ~12 |
| 256 KB | 8 | 12,288 | ~93 | ~6 |

`COPY_BLOCK_SIZE = 64 KB` puts each program at 64 KB of real HBM work
(meaningful) and lands the grid at ~24 waves on H100. The conv copy then
needs only 2 chunks (80 KB / 64 KB = ⌈1.25⌉), so the conv path stays
cheap with minimal early-exit waste.

### Three options, in order of preference

1. **Keep 2D, raise `COPY_BLOCK_SIZE` to 16–64 KB.** Simplest. Same grid,
   same launch math, just bigger memcpy windows inside each program.
   Likely improves both small-batch and large-batch cases at the same
   time, because larger HBM transactions reach peak bandwidth more easily.
   No regression risk in the production regime.

2. **Keep 3D, but gate it.** Use the 3D grid only when
   `num_reqs × total_states < SM_count × k` (for some `k ≈ 1.5`). For
   Qwen3.5-9B that gates 3D to `num_reqs ≤ 4`. Above that, fall back to
   2D. Requires a host-side branch and dual kernel paths. More complex
   for marginal benefit.

3. **Drop the 3D grid.** Salvage only the module-level constant for
   `COPY_BLOCK_SIZE` (which was a hygiene improvement regardless) and
   any test-coverage additions. Re-evaluate parallelism after option 1.

A `triton.autotune` over `{COPY_BLOCK_SIZE: [16K, 32K, 64K],
num_warps: [4, 8]}` on the 2D kernel would let the compiler pick the
right point per shape without committing to a single value.

---

## Appendix: how to reproduce the numbers

### Confirm `max_chunks` for your specific run

Add a print before the kernel launch in
[vllm/v1/worker/mamba_utils.py:584](../vllm/v1/worker/mamba_utils.py#L584):

```python
total_states = self.num_layers * self.num_state_types
print(f"[postprocess] num_layers={self.num_layers} "
      f"num_state_types={self.num_state_types} "
      f"max_chunks={self.max_chunks} "
      f"grid={(num_reqs, total_states, self.max_chunks)}")
```

### Microbenchmark

[bench_postprocess.py](../bench_postprocess.py) sweeps `(num_reqs,
inner_size)` and reports per-launch microseconds via CUDA events. Each
config is wrapped in an NVTX range so nsys can label each shape:

```bash
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
  --force-overwrite=true -o postprocess_3d \
  .venv/bin/python bench_postprocess.py

nsys stats --report cuda_gpu_kern_sum postprocess_3d.nsys-rep \
  | grep -i postprocess_mamba
```

### Serving benchmark (the regression measurement)

Compare 2D (`f237e16b`) vs 3D (`187ab11b`) on Qwen3.5-9B at concurrency
32, 500 prompts. See [results_3dgrid_qwen_h100.md](results_3dgrid_qwen_h100.md)
for raw output. Ignore the first run of each commit (warmup / compilation
artefacts).
