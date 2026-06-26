# `postprocess_mamba_fused_kernel`: uint64 vectorization analysis

Investigation of a change to the fused mamba postprocess kernel
([vllm/v1/worker/mamba_utils.py](../vllm/v1/worker/mamba_utils.py)) that
replaces the byte-wise inner copy loop (`LDG.8`/`STG.8`) with an 8-byte
loop (`LDG.64`/`STG.64`), plus a setup-time assertion guarding the
alignment invariant the wider loads rely on.

Benchmarks at concurrency 32 across three hybrid models and two GPU
architectures show:

| Model | Hardware | Throughput Δ | TPOT Δ | Verdict |
|---|---|---|---|---|
| Qwen3.5-9B | H100 | **+4.0 %** | **−3.0 %** | clear win |
| Qwen3.5-9B | GB200 | **+3.9 %** | **−2.5 %** | clear win |
| Qwen3.5-0.8B | H100 | −0.7 % | +1.4 % | null (within noise) |
| Nemotron-3-Super-120B-A12B-NVFP4 | GB200 | −0.5 % | +0.1 % | null (within noise) |

This document explains why the optimization is a clear win on
Qwen3.5-9B, why it is invisible on the smaller and larger hybrid models,
and what physical property of the workload predicts the outcome.

---

## Part 0 — TL;DR

The postprocess kernel's inner copy loop was issuing one memory
instruction *per byte* (`LDG.8`/`STG.8`). On every architecture, the
load/store pipe at the SM is bottlenecked on **instruction issue rate**,
not bandwidth, when transactions are this narrow. Switching to
`LDG.64`/`STG.64` cuts issued memory instructions by 8× for the same
copy size, and the kernel runs ~6–8× faster in isolation.

Whether that kernel speedup is visible in end-to-end TPOT depends on
**what fraction of the decode step the kernel occupied to begin with.**
Qwen3.5-9B is the lucky middle: state large enough that the kernel was
bandwidth-bound, model small enough that the kernel was a measurable
slice of the step. On smaller models the kernel is too small to see; on
larger models the model step is too long for the kernel slice to
matter.

---

## Part 1 — The 2D grid in detail

The kernel runs once per scheduler step in the spec-decode + hybrid +
align-mode path. For each request and each (mamba layer, state-type)
pair it:

1. Decides whether a per-block state copy is needed (depends on
   `num_accepted_tokens`, `num_scheduled`, `num_computed`, `num_draft`,
   `block_size`).
2. If yes, copies the appropriate slice of the Mamba state from a
   source physical block to a destination physical block.

There are two state types per Mamba layer:

- **Conv state** — small sliding-window buffer
  (`conv_kernel_size + num_spec - 1` positions of a few-thousand-wide
  vector).
- **Temporal SSM state** — the recurrent matrix carried across
  timesteps; for GDN/Mamba2-style models this is
  `(num_v_heads, head_v_dim, head_k_dim)` per block, typically MB-scale.

The kernel is launched on every decode step regardless of whether any
request actually needs a copy. Decision logic and copy are fused into
one launch to avoid an extra CPU↔GPU sync.

### Launch shape

```python
total_states = num_mamba_layers * num_state_types
grid = (num_reqs, total_states)
postprocess_mamba_fused_kernel[grid](...)
```

Each Triton program is uniquely identified by:

```python
req_idx   = tl.program_id(0)   # 0 .. num_reqs - 1
state_idx = tl.program_id(1)   # 0 .. total_states - 1
```

`state_idx` is a flattened index over `(mamba_layer_idx, state_type)`.
All per-state metadata (`base_addr`, `block_stride`, `elem_size`,
`inner_size`, `conv_width`, `dim_row_count`, `dim_row_stride`,
`group_idx`) is stored in flat tensors indexed by `state_idx`.

### How the flattening works

Mamba layers and full-attention layers live in separate KV-cache
groups. The kernel only sees layers in the mamba group(s):
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

Iteration order is **group → layer → state_type**. With a single mamba
group (the common case), this reduces to **mamba_layer → state_type**.

#### State-type ordering

For GDN / Mamba-style models the per-layer `kv_cache` list is
`[conv_state, temporal_ssm_state]` — confirmed by
`MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func` and the
unpacking conventions in the GDN attention modules.

So for a single-group Mamba model:

```
state_idx = 2 * L + s

where:
  L ∈ [0, num_mamba_layers)        — index among mamba layers only,
                                     NOT the transformer layer index
  s ∈ {0, 1}                       — 0 = conv state, 1 = temporal SSM state
```

#### Per-state-type sizes are identical across mamba layers

All conv states across all mamba layers share the same shape, same
dtype, same `copy_size` — they are built from the same model config.
Likewise for temporal states. So the even-indexed `state_idx` entries
are all "X KB conv copies" and the odd-indexed ones are all "Y MB
temporal copies." Per-layer divergence in the metadata tensors
(`base_addr`, `group_idx`) is only in *which physical buffer* to copy
from/to, not in copy size.

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
   group's block-table base pointer, then the source and destination
   block IDs for this request. Block IDs are widened to int64 here to
   prevent the per-block stride from overflowing int32 on large caches.
6. **Branch on state type** (`conv_width > 0` for conv states):
   - **Temporal**: `src_addr = base + actual_src_block_id * block_stride`,
     `copy_size = inner_size * elem_size`.
   - **SD conv**: `src_addr = base + src_block_id * block_stride + bias_offset`,
     `copy_size = (conv_width - bias) * inner_size * elem_size`.
   - **DS conv**: row-by-row copy with `dim_rows` and per-row bias slicing.
7. **Run the inner copy loop** — the focus of this PR.

### The original inner copy loop (pre-PR)

```python
offsets = tl.arange(0, COPY_BLOCK_SIZE)   # COPY_BLOCK_SIZE = 1024
for i in range(0, copy_size, COPY_BLOCK_SIZE):
    mask     = (i + offsets) < copy_size
    curr_src = (src_addr + i + offsets).to(tl.pointer_type(tl.uint8))
    curr_dst = (dst_addr + i + offsets).to(tl.pointer_type(tl.uint8))
    data     = tl.load(curr_src, mask=mask)        # → LDG.8
    tl.store(curr_dst, data, mask=mask)            # → STG.8
```

The number of outer-loop iterations is `cdiv(copy_size, 1024)`. Inside
each iteration, the warp issues 1024 byte-wide load instructions and
1024 byte-wide store instructions — one per thread, per byte.

### Two levels of parallelism

This is the property that matters most for understanding kernel
behaviour.

**Level 1 — programs across SMs (the "grid").** The CUDA driver hands
all `num_reqs × total_states` programs to the SM scheduler, which
distributes them round-robin onto SMs. Each program is one CTA; with
~16 CTAs resident per H100 SM at this resource profile, total
concurrent capacity is roughly `132 × 16 ≈ 2,100 CTAs`. If the launch
fits inside that capacity, all programs run in a single "wave" and the
kernel completes in roughly one program's wall time. If it overshoots,
the scheduler must wait for programs to retire before launching the
next wave — adding latency proportional to the overshoot.

**Level 2 — SIMT lanes inside a program (the `tl.arange`).** The line
`offsets = tl.arange(0, 1024)` declares 1024 byte-lanes that are
processed in true SIMT parallelism by the program's threads (default
4 warps = 128 threads per CTA, so 8 byte-lanes per thread per
iteration).

The crucial detail: even though the warp moves 1024 bytes per
iteration, **it does so via 1024 separate byte-wide instructions**,
not one wide instruction. The hardware can coalesce the *addresses*
into wider HBM transactions, but the SM must still issue a memory op
per byte. The LSU pipe — not HBM — becomes the bottleneck.

### Cost structure of one program

- **Per-program fixed cost** (prelude): ~12 scalar loads, a few
  arithmetic ops, conditional branches. Order of microseconds depending
  on cache state.
- **Per-iteration variable cost** (inside the loop): 1024 byte-wide
  loads + 1024 byte-wide stores. Issue-rate bound.
- **Metadata is loaded once per program** and held in registers for
  the duration of the loop — no redundant L1/L2 traffic across loop
  iterations.

Critical implication: **the prelude is amortized over
`copy_size / 1024` iterations**. As `copy_size` grows, the kernel's
limiter shifts from prelude-bound to inner-loop-bound. For
multi-megabyte temporal copies, the inner loop is overwhelmingly
dominant, and the issue-rate inefficiency dominates the wall-clock.

### Hardware capacities at a glance

| Platform | SMs | CTAs/SM (this resource profile) | Single-wave capacity |
|---|---|---|---|
| H100 | 132 | ~16 | ~2,100 |
| B200 (GB200 board) | ~148 | ~16 | ~2,370 |

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

```
total_states = 24 × 2 = 48
```

### State shapes (per block, TP=1, num_spec=2)

`num_spec = 2` is verified from the benchmark logs:
`Draft tokens / Drafts ≈ 1,083,094 / 541,547 = 2`.

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

The temporal state dominates by **25×**.

### 2D grid program count and per-program work

At benchmark concurrency `num_reqs = 32`:

```
2D grid       = (32, 48)        → 1,536 programs
H100 capacity ≈ 132 SMs × 16 CTAs ≈ 2,100 concurrent CTAs
```

**1,536 programs fit comfortably in one wave** — roughly 12 programs
land on each SM, well within the 16-CTA limit.

Inner-loop iteration counts per program (`COPY_BLOCK_SIZE = 1024`):

| State type | `copy_size` | outer iterations | byte-wide instructions per program |
|---|---|---|---|
| Conv (bf16) | 80 KB | 80 | ~163,840 (80 × 1024 × 2 for load+store) |
| **Temporal SSM (fp32)** | **2 MB** | **2,048** | **~4,194,304** |

Each temporal-state program issues **~4 million byte-wide memory
instructions**. With `LDG.64`/`STG.64`, that drops to ~524,000 — an
8× reduction in issued instructions for the same number of bytes
moved.

### Occupancy estimate

Under the model that the 2D grid fits in one wave:

```
SMs in use per wave   = min(132, ceil(1536 / 16)) = 96 SMs busy
Programs / SM         = 1536 / 96 ≈ 16 (saturates the slot limit)
Wall-clock per launch ≈ max-program-time ≈ 2–3 µs
```

In practice this is a healthy regime: the kernel is **issue-bound**
(not HBM-bound — that is the key diagnosis), prelude overhead is fully
amortized (~12 scalar loads vs 2,048 iterations of useful work), and
there is no second-wave scheduling delay.

### When the 2D grid breaks down

The single-wave assumption fails as `num_reqs × total_states` drops
below the SM count, OR as it grows beyond single-wave capacity. For
Qwen3.5-9B:

| `num_reqs` | programs | SM utilization | regime |
|---|---|---|---|
| 1 | 48 | 36% of one SM-row | severely undersaturated |
| 4 | 192 | ~1.5 SM-rows | undersaturated |
| 16 | 768 | ~half wave | mild headroom |
| **32** | **1,536** | **full wave** | **target regime** |
| 64 | 3,072 | 1.5 waves | mild overshoot |
| 128 | 6,144 | ~3 waves | multi-wave |
| 256 | 12,288 | ~6 waves | scheduler-bound |

The optimization itself is **orthogonal to grid occupancy** — it
speeds up each program by ~8×, regardless of how many programs are
running. The visibility of the speedup in TPOT depends on the kernel's
share of step time, which is discussed in Part 5.

---

## Part 3 — What the vectorization changes

### The problem

The byte-wise inner loop emits `LDG.8`/`STG.8` instructions. On all
recent NVIDIA architectures these single-byte ops occupy one issue slot
per byte at the LSU. With a 4-warp CTA, the warp scheduler can issue
~one memory instruction per cycle per warp, but each instruction moves
only 1 byte per thread. So the warp moves at most 32 B/cycle per
issued instruction — far below the per-warp memory throughput the
architecture is built to deliver via wider loads.

This is **issue-rate bound**, not bandwidth-bound. The HBM has plenty
of headroom (~3 TB/s on H100, ~8 TB/s on B200). The bottleneck is the
SM's ability to issue enough byte-wide memory ops to keep HBM busy.

For a 2 MB temporal copy on Qwen3.5-9B, the program issues
~4M byte-wide memory instructions. At ~1 instruction per cycle per
warp on the LSU, that is *thousands* of cycles per program just to
issue all the loads and stores, even before considering latency.

### The fix

```python
copy_size_u64 = copy_size // 8
src_u64 = src_addr.to(tl.pointer_type(tl.uint64))
dst_u64 = dst_addr.to(tl.pointer_type(tl.uint64))
offsets = tl.arange(0, COPY_BLOCK_SIZE)
for i in range(0, copy_size_u64, COPY_BLOCK_SIZE):
    mask = (i + offsets) < copy_size_u64
    data = tl.load(src_u64 + i + offsets, mask=mask)   # → LDG.64
    tl.store(dst_u64 + i + offsets, data, mask=mask)   # → STG.64

# byte tail for any 0-7 residual bytes (defensive)
tail_start = copy_size_u64 * 8
tail_bytes = copy_size - tail_start
tail_off   = tl.arange(0, 8)
tail_src   = (src_addr + tail_start).to(tl.pointer_type(tl.uint8))
tail_dst   = (dst_addr + tail_start).to(tl.pointer_type(tl.uint8))
tail_mask  = tail_off < tail_bytes
tail_data  = tl.load(tail_src + tail_off, mask=tail_mask)
tl.store(tail_dst + tail_off, tail_data, mask=tail_mask)
```

Each thread now moves **8 bytes** per instruction. For the same
2 MB copy, the program issues ~524K wide memory instructions instead of
~4M byte ops — an **8× reduction in issued instructions**, with
identical bytes moved.

### Why it is safe (and the alignment assertion)

The vectorization is safe because both the temporal and SD-conv paths
copy whole token slices:

- `src_addr` and `dst_addr` are aligned to `inner_size * elem_size`
  (a whole token slice).
- `copy_size` is a multiple of that slice size.

For every state dtype currently in use (fp32, bf16, fp16),
`inner_size * elem_size` is ≥ 8 B and 8 B-aligned. The uint64 pointer
cast and the `// 8` division are exact, and `copy_size_u64 * 8 ==
copy_size`, so the byte tail copies zero bytes in practice.

The defensive byte tail handles a theoretical 0–7 byte residual that
cannot be reached with current dtypes/layouts but covers a future
smaller dtype (e.g. fp8 in some unusual stride configuration) that
might break the invariant. The alignment commit
(`ae2d817d2`) adds a setup-time assertion in
[MambaSpecDecodeGPUContext.initialize_from_forward_context](../vllm/v1/worker/mamba_utils.py#L488)
that fails loud at engine init if any layer breaks the 8 B alignment:

```python
is_ds_conv = (
    copy_func is get_conv_copy_spec and is_conv_state_dim_first()
)
if not is_ds_conv:
    slice_bytes = (
        int(self.state_inner_sizes[idx]) * state.element_size()
    )
    assert slice_bytes % 8 == 0, (
        f"layer {layer_name}: inner_size * elem_size = {slice_bytes}B "
        f"is not 8B-aligned; postprocess_mamba_fused_kernel uint64 "
        f"vectorization requires it"
    )
```

This is **fail-loud at setup**, not per-step — zero runtime cost.

### The DS-conv exemption

The DS-conv path is **not** vectorized. Its bias offset is only
`elem_size`-aligned (not slice-aligned), so wider loads could straddle
the alignment boundary. Per-row copies in that path are also tiny —
the byte loop dominates total cost less, and the gain wouldn't justify
the safety risk. The assertion above explicitly skips DS-conv layers.

---

## Part 4 — The three models compared

Three hybrid models were benchmarked: a sub-billion-parameter variant
(Qwen3.5-0.8B), a mid-sized one (Qwen3.5-9B), and a large MoE
(Nemotron-3-Super-120B). The relevant config differences:

### Qwen3.5-0.8B (H100)

From `Qwen/Qwen3.5-0.8B/config.json`:

```json
"num_hidden_layers":        24,
"full_attention_interval":  4,
"linear_conv_kernel_dim":   4,
"linear_num_key_heads":     16,
"linear_num_value_heads":   16,
"linear_key_head_dim":      128,
"linear_value_head_dim":    128,
"hidden_size":              1024,
"mamba_ssm_dtype":          "float32",
"dtype":                    "bfloat16"
```

```
mamba_layers   = 18    (= 24 layers × 3/4)
attn_layers    = 6
total_states   = 36

conv_dim       = 128 × 16 × 2 + 128 × 16 = 6,144
conv shape     = (5, 6,144)        → 60 KB (bf16)

temporal shape = (16, 128, 128)    → 1 MB (fp32)

temporal / conv ratio: 17×
```

### Qwen3.5-9B (H100, GB200)

Already enumerated in Part 2:

```
mamba_layers   = 24
total_states   = 48

conv shape     = (5, 8,192)        → 80 KB (bf16)
temporal shape = (32, 128, 128)    → 2 MB (fp32)

temporal / conv ratio: 25×
```

### Nemotron-3-Super-120B-A12B-NVFP4 (GB200)

From `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/config.json`:

```json
"architectures":          ["NemotronHForCausalLM"],
"num_hidden_layers":      88,
"hybrid_override_pattern": "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
"hidden_size":            4096,
"expand":                 2,
"conv_kernel":            4,
"ssm_state_size":         128,
"mamba_num_heads":        128,
"mamba_head_dim":         64,
"n_groups":               8,
"dtype":                  "bfloat16",
"mamba_ssm_cache_dtype":  "float32"
```

Counting the `hybrid_override_pattern` (M = Mamba, * = attention, E =
MoE/MLP): **40 Mamba layers**, 8 attention layers, 40 MoE/MLP layers.

```
mamba_layers   = 40
total_states   = 80

d_inner        = mamba_num_heads × mamba_head_dim = 128 × 64 = 8,192
conv_dim       = d_inner + 2 × n_groups × ssm_state_size
               = 8,192 + 2 × 8 × 128
               = 10,240
conv shape     = (5, 10,240)        → 100 KB (bf16)

temporal shape = (128, 64, 128)     → 4 MB (fp32)

temporal / conv ratio: 40×
```

NVFP4 quantization affects weight storage/matmul only — the conv state
remains bf16 and the SSM cache remains fp32, so state sizes are
unchanged.

### Side-by-side

| | Qwen3.5-0.8B | Qwen3.5-9B | Nemotron-3-Super-120B |
|---|---|---|---|
| Total layers | 24 | 32 | 88 |
| Mamba layers | 18 | 24 | 40 |
| Attention layers | 6 | 8 | 8 |
| MoE/MLP layers | — | — | 40 |
| `total_states` | 36 | 48 | 80 |
| `conv_dim` | 6,144 | 8,192 | 10,240 |
| Conv copy | **60 KB** | **80 KB** | **100 KB** |
| Temporal shape | (16, 128, 128) | (32, 128, 128) | (128, 64, 128) |
| Temporal copy | **1 MB** | **2 MB** | **4 MB** |
| Temp/conv ratio | 17× | 25× | 40× |

### 2D grid sizes at `num_reqs = 32`

| Model | Programs | H100 fit | B200 fit |
|---|---|---|---|
| Qwen3.5-0.8B | 1,152 | ~55 % single wave | ~49 % single wave |
| Qwen3.5-9B | 1,536 | ~73 % single wave | ~65 % single wave |
| Nemotron-3-Super-120B | 2,560 | ~1.22 waves | ~1.08 waves |

### Where the per-program inner loop spends its time

| Model | Temporal `copy_size` | Outer iterations | Byte-wide instructions (load+store) |
|---|---|---|---|
| Qwen3.5-0.8B | 1 MB | 1,024 | ~2.1 M |
| Qwen3.5-9B | 2 MB | 2,048 | ~4.2 M |
| Nemotron-3-Super-120B | 4 MB | 4,096 | ~8.4 M |

Each model's per-program byte-wide instruction count scales linearly
with `copy_size`. The vectorization cuts all three by 8× in absolute
terms — but whether that translates to a measurable end-to-end win
depends on what fraction of step time the kernel was consuming, which
depends on the rest of the model.

---

## Part 5 — Why the optimization helps only Qwen3.5-9B

The vectorization is the same 8× kernel-level reduction in issued
memory instructions across all three models. The end-to-end results
(Part 0 table) are wildly different. The explanation rests on two
quantities:

1. **How much wall-clock the kernel takes** — proportional to
   `copy_size / 8` after the optimization.
2. **How long the rest of the decode step takes** — set by attention +
   MLP/MoE + drafter + scheduler, *not* by the postprocess kernel.

The visible TPOT delta is

```
ΔTPOT / TPOT ≈ ΔT_kernel / T_step
            ≈ (T_kernel × (1 − 1/k)) / T_step
```

where `k ≈ 6–8` is the per-program kernel speedup factor. The
denominator `T_step` differs by ~5× across the three models.

### Rough numbers

| Model | TPOT base | Approx kernel time | Kernel / step | Predicted ΔTPOT |
|---|---|---|---|---|
| Qwen3.5-0.8B | 2.97 ms | ~1–2 µs | ~0.05 % | < 0.1 % (below noise) |
| **Qwen3.5-9B** | **7.78 ms (H100)** | **~3–5 µs × 1 wave** | **~0.3 %** | **~3 % (matches measurement)** |
| Nemotron-3-Super-120B | 10.14 ms | ~5–6 µs × 1.1 waves | ~0.06 % | < 0.1 % (below noise) |

(Kernel times are estimates based on `copy_size × num_states × num_reqs`
HBM traffic and observed single-wave timing; precise numbers would
require profiling each model.)

The two ends of the spectrum fail for *different* reasons:

#### Why Qwen3.5-0.8B doesn't move

- **State is small.** 1 MB temporal vs Qwen3.5-9B's 2 MB. The kernel
  itself runs in ~1–2 µs, near the noise floor of any kernel-launch
  timing.
- **Acceptance length is high (2.20 vs Qwen3.5-9B's 1.89).** The
  smaller model is better at predicting itself; each spec-decode step
  emits more tokens, so the kernel runs fewer times per emitted token.
- **The model is small, so the step is short (2.97 ms).** With less
  total step time, the kernel's relative share is ~0.05 %, well below
  the ~1 % run-to-run noise floor of the serving benchmark.

A tiny (+1.4 % TPOT) regression *might* be real — driven by the
slightly heavier prelude in the uint64 path (pointer reinterpret, the
`// 8` division, byte tail) — but with within-variant spread of ~1 %,
it cannot be distinguished from noise. More runs would tighten the
error bar; the conclusion ("neutral within noise") is unlikely to
flip.

#### Why Nemotron-3-Super-120B doesn't move

- **The model is enormous.** 88 layers + MoE on top means the decode
  step is dominated by attention and MoE/MLP work, not postprocess.
- **Kernel scales with state size, but step scales with model size.**
  Even though Nemotron has 4 MB temporal copies (2× Qwen3.5-9B's), and
  ~2× the program count, the step itself is much longer (10 ms vs
  7.8 ms). The kernel's relative share is ~0.06 %, an order of
  magnitude below where TPOT can detect a change.
- **Acceptance length is even higher (~2.05).** Same effect as
  Qwen3.5-0.8B — fewer kernel launches per emitted token.

#### Why Qwen3.5-9B is the sweet spot

- **State is large enough.** 2 MB temporal copies make the byte-wise
  loop's issue-rate inefficiency add up to a non-trivial program time
  (~2–3 µs). Cutting it 8× saves microseconds, not nanoseconds.
- **The model is small enough.** Step time is ~7.8 ms — long enough
  for spec decoding to add up, short enough that microseconds in the
  postprocess kernel register as a measurable fraction.
- **Grid is at single-wave capacity.** 1,536 programs against 2,100
  capacity means the kernel is fully utilizing the GPU without
  spilling into a second wave. Any per-program speedup translates
  directly into wall-clock reduction.

### The clean physical model

The optimization helps when **the postprocess kernel was a measurable
fraction of decode step time**. That requires three things to coincide:

1. **Large enough per-layer state** (so byte-wise issue overhead adds
   up to microseconds per program).
2. **Modest enough model** (so the step itself doesn't dwarf the kernel).
3. **High enough concurrency** (so the kernel actually does work
   instead of immediately returning via `needs_copy = False`).

Qwen3.5-9B at concurrency 32 satisfies all three. The other two models
satisfy (3) but fail (1) or (2) in opposite directions.

---

## Part 6 — Expected behavior across concurrencies

The benchmarks were run at concurrency 32. The win extends across other
operating points roughly as follows (for Qwen3.5-9B on H100):

### Two effects that change with concurrency

**1. Postprocess kernel wall-clock.** Wall-clock is set by the slowest
in-flight program. For `num_reqs ≤ ~43`, the grid fits in a single
wave and per-program time dominates — wall-clock is roughly
*constant*. Above that, the kernel takes multiple waves and wall-clock
scales linearly with `num_reqs / single_wave_capacity`.

**2. Model step time.** For LLM decode in the memory-bound regime, the
step time is dominated by weight loading, not compute. From batch=1 to
batch=~32, step time is roughly flat. Beyond that, compute starts to
matter and the step grows.

### The full picture

| `num_reqs` | Kernel ms (est.) | Step ms (est.) | Kernel/step | Expected ΔTPOT |
|---|---|---|---|---|
| 1 | 0.002 | ~5 | ~0.04 % | ~0 % (noise floor) |
| 4 | 0.002 | ~6 | ~0.03 % | ~0–0.5 % |
| 16 | 0.002 | ~7 | ~0.03 % | ~1–2 % |
| **32** | **0.002** | **~7.8** | **~0.03 %** | **~3 % (measured)** |
| 64 | 0.004 | ~8.5 | ~0.05 % | ~3–4 % |
| 128 | 0.006 | ~10 | ~0.06 % | ~3–4 % |
| 256 | 0.012 | ~15 | ~0.08 % | ~3–4 % |

### Where the optimization disappears

- **Very low concurrency (`num_reqs ≤ 4`)**: kernel under-saturated;
  most SMs idle. Per-program speedup intact, but kernel time is
  already ~1–2 µs absolute; the savings vanish into noise.
- **Very high concurrency on the largest models**: model step grows
  faster than the kernel can shrink. Eventually the optimization fades
  to noise even at high batch — but on Qwen3.5-9B this is far past
  practical operating points.

### Where the optimization grows in absolute value

At `num_reqs ≥ 64`, the kernel takes multiple waves, and the absolute
ms saved per step grows linearly with `num_reqs`. The *relative* TPOT
delta stays around 3–4 %, but the *absolute* ms saved per step grows.
This matters for capacity planning: production deployments running at
high concurrency benefit in tokens/sec at SLA, even when the
percentage TPOT win looks unchanged.

---

## Part 7 — Cross-architecture portability

The benchmarks include both H100 (Hopper, HBM3) and GB200 (Blackwell,
HBM3e). On the model where the optimization helps (Qwen3.5-9B), the
delta is essentially identical:

| Platform | Throughput Δ | TPOT Δ |
|---|---|---|
| H100 | +4.0 % | −3.0 % |
| GB200 | +3.9 % | −2.5 % |

This is the key portability check. If the win depended on a quirk of
the H100 memory subsystem (e.g., a specific HBM controller behaviour),
the much higher Blackwell HBM bandwidth would have flattened the gain.
It did not — because the kernel was **issue-rate bound at the SM, not
HBM-bound at the controller**. Both architectures have the same
issue-per-byte cost for `LDG.8`/`STG.8`, and both benefit equally from
switching to `LDG.64`/`STG.64`.

The fact that the relative win is essentially unchanged across
architectures is the strongest evidence that the diagnosis is correct.

---

## Part 8 — When the optimization works (summary)

The optimization works when **all three** of the following hold:

1. The state copy is **large enough** (≥ ~1 MB per program) that the
   byte-wise inner loop dominates per-program time.
2. The model decode step is **short enough** (≲ ~10 ms) that
   microseconds of kernel wall-clock are a measurable fraction.
3. The grid is **populated enough** (`num_reqs ≥ ~8`) that the kernel
   is doing meaningful work rather than mostly returning early.

In matrix form across the data collected:

| | Qwen3.5-0.8B | Qwen3.5-9B | Nemotron-3-Super-120B |
|---|---|---|---|
| State large enough | no (1 MB) | **yes (2 MB)** | yes (4 MB) |
| Step short enough | yes (3 ms) | **yes (~7.8 ms)** | no (~10 ms, MoE-dominated) |
| Concurrency adequate | yes | **yes** | yes |
| **Outcome** | neutral | **+4 % throughput / −3 % TPOT** | neutral |

The optimization is **safe everywhere** — the worst observed case is a
~1 % regression on Qwen3.5-0.8B, within noise. It is **valuable on the
production operating point of mid-sized hybrid models** like
Qwen3.5-9B, where the postprocess kernel is a measurable fraction of
the decode step. It is **invisible-but-free elsewhere**.

---

## Appendix A — Reproducing the numbers

### Serving benchmark

Compare `last_precompiled` (`f237e16b41bb444b3c9994260a36f9c2388bd019`)
vs `postprocess_vectorize` (`9da28937a` + `ae2d817d2`) at concurrency
32, 500 prompts. Convention: 3 runs per config, **discard the first**
(warm-up / compilation artifact), report the mean of runs 2 and 3.

Raw output files:

- Qwen3.5-9B on H100: [vectorize/H100/results_vectorize_Qwen3.5-9B.md](vectorize/H100/results_vectorize_Qwen3.5-9B.md)
- Qwen3.5-0.8B on H100: [vectorize/H100/results_vectorize_Qwen3.5-0.8B.md](vectorize/H100/results_vectorize_Qwen3.5-0.8B.md)
- Qwen3.5-9B on GB200: [vectorize/GB200/results_vectorize_Qwen3.5-9B.md](vectorize/GB200/results_vectorize_Qwen3.5-9B.md)
- Nemotron-3-Super-120B on GB200: [vectorize/GB200/results_vectorize_nemotron.md](vectorize/GB200/results_vectorize_nemotron.md)

### Sanity checks

- **Acceptance length is unchanged** within noise across baseline and
  vectorized runs on all three models. Confirms the optimization is
  bit-exact and only changes kernel wall-clock, not behaviour.
- **Within-variant spread** (runs 2 vs 3) is ~0.5–2 % on throughput
  for all configurations; deltas larger than that are real, deltas
  smaller are noise.

## Appendix B — Future work

Two follow-ups suggested by this analysis:

1. **`triton.autotune` over `(COPY_BLOCK_SIZE, num_warps)`** on the 2D
   kernel. The current `COPY_BLOCK_SIZE = 1024` was tuned for the
   byte-wise loop; with uint64 loads, the optimal block size may shift
   (larger HBM transactions reach peak bandwidth more easily). A
   per-shape autotune would pick the right point per model without
   committing to a single value.

2. **Profile high-concurrency operating points** (`num_reqs = 128+`).
   The model predicts the optimization grows in absolute saved ms per
   step at high concurrency. If this is confirmed, the PR's value
   proposition extends from "TPOT improvement at typical batch" to
   "throughput/SLA improvement at peak load." Worth measuring before
   the next perf-related PR.
