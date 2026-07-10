# install

uv venv --python 3.12 .venv-cpu
source .venv-cpu/bin/activate
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu \
  uv pip install --editable .

# To see

vim csrc/cpu/spec_decode_utils.cpp

./vllm/utils/cpu_triton_utils.py

# How the CPU backend works overall

## Platform hook: `check_and_update_config`

Declared abstract in [vllm/platforms/interface.py:561-571](../vllm/platforms/interface.py#L561-L571).
Called once from `VllmConfig.__post_init__` at
[vllm/config/vllm.py:1367](../vllm/config/vllm.py#L1367), after CLI parsing and
before workers/runners are constructed. Each platform overrides it to (1)
reject unsupported feature combinations by raising, or (2) normalize defaults.

`CpuPlatform.check_and_update_config` in
[vllm/platforms/cpu.py:111-313](../vllm/platforms/cpu.py#L111-L313):

- Force-disables cascade attention.
- Sets CPU-friendly default `block_size=128`.
- Force-disables async scheduling (this is why `mamba_cache_mode == "align"`
  even gets exercised on CPU).
- Rewrites executor backend `uni` → `mp`.
- Sets `worker_cls` to `CPUWorker`.
- Warn-and-disable DBO.
- Force-disables MLA-associated features.

Other hard refusals live in
[`get_attn_backend_cls`](../vllm/platforms/cpu.py#L75-L87): MLA, sparse
attention, and (elsewhere) FP8 KV cache with KV transfer.

## Triton on CPU

`HAS_TRITON` in [vllm/triton_utils/importing.py:15](../vllm/triton_utils/importing.py#L15)
is true only when:

1. `find_spec("triton")` succeeds (not the XPU variant).
2. Exactly one active Triton backend driver is registered.
3. On `vllm-cpu` wheels: the "cpu" backend is present in `triton.backends`.

Call sites pattern: `if HAS_TRITON: <triton path> else: <fallback>`. Examples:
[topk_topp_sampler.py:13-15](../vllm/v1/sample/ops/topk_topp_sampler.py#L13-L15),
[punica_gpu.py:16-19](../vllm/lora/punica_wrapper/punica_gpu.py#L16-L19),
[punica_xpu.py:18-21](../vllm/lora/punica_wrapper/punica_xpu.py#L18-L21),
[gumbel.py:5-14](../vllm/v1/worker/gpu/sample/gumbel.py#L5-L14),
[qkv_padded_fp8_quant.py:102](../vllm/kernels/triton/qkv_padded_fp8_quant.py#L102).

## `cpu_triton_utils` shim

Runs only when `HAS_TRITON` is false. Drop-in replacement for Triton kernel
*call sites*:

1. Each patched symbol (e.g. `batch_memcpy_kernel`) is a
   [`_FuncWrapper`](../vllm/utils/cpu_triton_utils.py#L14-L19) whose
   `__getitem__` returns the underlying Python function — this reproduces
   Triton's `kernel[grid](...)` launch syntax so generic call sites don't
   change.
2. [`CPUModelRunner._postprocess_triton`](../vllm/v1/worker/cpu_model_runner.py#L62-L119)
   rebinds module-level names (e.g.
   `vllm.v1.worker.mamba_utils.batch_memcpy_kernel = cpu_tl.batch_memcpy_kernel`)
   after import.
3. Most `_impl` functions forward to a C++ op via
   `torch.ops._C.<name>_impl`, with int32↔int64 dtype coercions since C++
   expects int64 pointers. See e.g.
   [_compute_slot_mapping_kernel_impl](../vllm/utils/cpu_triton_utils.py#L23-L48).
4. [_batch_memcpy_impl](../vllm/utils/cpu_triton_utils.py#L476-L482) is the
   odd one out — pure Python `ctypes.memmove` loop, no C++ op behind it.

## Gap: hybrid + spec-decode + `mamba_cache_mode == "align"`

The three fused mamba Triton kernels
[`postprocess_mamba_fused_kernel`](../vllm/v1/worker/mamba_utils.py#L132),
[`preprocess_mamba_align_fused_kernel`](../vllm/v1/worker/mamba_utils.py#L263),
[`precopy_mamba_align_fused_kernel`](../vllm/v1/worker/mamba_utils.py#L308)
have **no C++ fallback** — neither a `_FuncWrapper` shim nor a
`torch.ops._C.*_impl` counterpart. `_postprocess_triton` only patches
`batch_memcpy_kernel` at
[cpu_model_runner.py:119](../vllm/v1/worker/cpu_model_runner.py#L119).

These kernels are reached when:

- `speculative_config is not None`
- `model_config.is_hybrid`
- `cache_config.mamba_cache_mode == "align"`

Gate lives at
[gpu_model_runner.py:1035-1037](../vllm/v1/worker/gpu_model_runner.py#L1035-L1037)
(`with_postprocess_align`) and the launch at
[gpu_model_runner.py:1534-1550](../vllm/v1/worker/gpu_model_runner.py#L1534-L1550)
(`postprocess_mamba_align_gpu`). V2 also uses these kernels via
[vllm/v1/worker/gpu/model_states/mamba_hybrid.py](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py).

Nothing in `CPUModelRunner.__init__` blocks this combo, so on a `HAS_TRITON=False`
build it crashes at forward time when the fused kernel is launched on CPU
tensors.

### `mamba_cache_mode` values

From [config/cache.py:134-142](../vllm/config/cache.py#L134-L142):

- `"none"`: prefix caching disabled.
- `"all"`: default when prefix caching is enabled — uses
  `postprocess_mamba_all` (pure Python) at
  [gpu_model_runner.py:1561-1570](../vllm/v1/worker/gpu_model_runner.py#L1561-L1570).
- `"align"`: opt-in — uses fused Triton kernels.

### Fix option A: fail-closed

Raise `NotImplementedError` in `check_and_update_config` when the combo is
requested without Triton. Mirrors the MLA/sparse pattern in
[cpu.py:84-86](../vllm/platforms/cpu.py#L84-L86).

```python
from vllm.triton_utils import HAS_TRITON

if (
    not HAS_TRITON
    and model_config is not None
    and model_config.is_hybrid
    and vllm_config.speculative_config is not None
    and cache_config.mamba_cache_mode == "align"
):
    raise NotImplementedError(
        "Speculative decoding on a hybrid model with "
        "mamba_cache_mode='align' requires Triton on CPU (install "
        "triton-cpu). Alternatives: disable speculative decoding, or "
        "set mamba_cache_mode to a non-align mode."
    )
```

### Fix option B: soft-disable (preferred)

Downgrade `"align"` → `"all"` and warn. Mirrors the DBO pattern at
[cpu.py:151-153](../vllm/platforms/cpu.py#L151-L153). Prefix caching stays
enabled; only the align-specific fused-Triton path is avoided. `"all"` is
documented as the default when prefix caching is enabled, so this is the
least-surprising override.

```python
if (
    not HAS_TRITON
    and model_config is not None
    and model_config.is_hybrid
    and vllm_config.speculative_config is not None
    and cache_config.mamba_cache_mode == "align"
):
    logger.warning_once(
        "mamba_cache_mode='align' with speculative decoding on a hybrid "
        "model uses fused Triton kernels that have no CPU fallback. "
        "Falling back to mamba_cache_mode='all' (prefix caching remains "
        "enabled). Install triton-cpu to use align mode on CPU."
    )
    cache_config.mamba_cache_mode = "all"
```

### Prefer "all"-fallback over "disable prefix caching entirely"

- **Blast radius**: switching to `"all"` is mamba-internal; disabling prefix
  caching regresses all requests, mamba or not.
- **User intent**: prefix caching was explicitly enabled — keep the
  user-visible feature, downgrade only the incompatible optimization.
- **Matches docstring intent**: `"all"` is documented as the default when
  prefix caching is enabled, so switching to it is not a surprising override.

### Verification results

**Is `mamba_cache_mode` populated when `check_and_update_config` runs?** Yes.
Ordering inside `VllmConfig.__post_init__`:

- [config/vllm.py:863](../vllm/config/vllm.py#L863) —
  `self.try_verify_and_update_config()` runs first and calls
  [`MambaModelConfig.verify_and_update_config`](../vllm/model_executor/models/config.py#L539-L594),
  which finalizes `mamba_cache_mode`: `"none"` → `"all"` (if
  `model_config.supports_mamba_prefix_caching`) or `"align"` (otherwise), when
  prefix caching is on.
- [config/vllm.py:1367](../vllm/config/vllm.py#L1367) —
  `current_platform.check_and_update_config(self)` runs afterward.

So by the time the CPU hook sees `cache_config.mamba_cache_mode`, it's the
final value. The Option B override lands after the auto-selection.

**Does V2 respect the same `"align"` gate?** Yes for the three align kernels,
no for one unrelated scatter. Concretely:

- Align-specific gating in V2 at
  [mamba_hybrid.py:86](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L86)
  (`self._align_mode = ... == "align"`), with guarded early-returns at
  [mamba_hybrid.py:179](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L179)
  (`preprocess_state`) and
  [mamba_hybrid.py:316-329](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L316-L329)
  (`postprocess_state`). Overriding align→all short-circuits all three align
  kernels on V2 too.
- **Caveat**:
  [`_scatter_num_accepted_kernel`](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L332-L343),
  fired at
  [mamba_hybrid.py:303](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L303),
  is *not* gated on `_align_mode` — it runs in `postprocess_state` whenever
  `num_sampled` is a tensor (real spec-decode). And
  [vllm/v1/worker/cpu/model_runner.py](../vllm/v1/worker/cpu/model_runner.py)
  (the V2 CPU runner) is a 17-line class that only overrides
  `warming_up_model` — no `_postprocess_triton` monkey-patch pass at all. V2 +
  CPU + spec-decode + hybrid without triton-cpu would hit this scatter kernel
  regardless of `mamba_cache_mode`. The align→all downgrade is **necessary but
  not sufficient for V2** — a full V2-CPU fix is out of scope here.

**Bonus finding**:
[`MambaModelConfig`](../vllm/model_executor/models/config.py#L571-L574)
asserts `scheduler_config.enable_chunked_prefill` when it picks `"align"`.
Chunked prefill is on by default on CPU, so this assert does not help us avoid
the combo automatically.

### Bottom line

Option B (`"align"` → `"all"` with a warning) is safe and effective on **V1
(default) CPU**, which is what actually ships. The V2 CPU path has an
independent Triton-fallback gap that this fix doesn't address; if V2-CPU
becomes a supported combination it needs its own `_postprocess_triton`
equivalent (or a `_scatter_num_accepted_kernel` fallback in
`cpu_triton_utils`).

## V2 CPU: full fix plan

Two independent gaps that Option B alone does not close.

### Gap 1: V2 CPU runner does no monkey-patching at all

[vllm/v1/worker/cpu/model_runner.py](../vllm/v1/worker/cpu/model_runner.py) is a
17-line stub. It inherits the V2 `GPUModelRunner` but never runs the
`_postprocess_triton` pass that V1's `CPUModelRunner.__init__` runs at
[cpu_model_runner.py:62-119](../vllm/v1/worker/cpu_model_runner.py#L62-L119).
Every kernel V1 patches — `_compute_slot_mapping_kernel`, `eagle_*`,
`rejection_*_sample_kernel`, `expand_kernel`, `sample_recovered_tokens_kernel`,
`batch_memcpy_kernel` — is used by V2's sample / spec-decode / block-table
paths too, so V2-CPU without triton-cpu will crash on any of those long before
hitting mamba.

**Fix**: extract `_postprocess_triton` out of V1 `CPUModelRunner` into a
module-level helper (e.g. `vllm.utils.cpu_triton_utils.apply_cpu_fallbacks()`)
and call it from both runners:

```python
# cpu_model_runner.py (V1) and cpu/model_runner.py (V2)
class CPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        # ... existing init ...
        cpu_tl.apply_cpu_fallbacks()
```

The current V1 method already has the correct `HAS_TRITON` early-return, so
extraction is mechanical.

### Gap 2: `_scatter_num_accepted_kernel` has no fallback

Defined only in V2 at
[mamba_hybrid.py:332-343](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L332-L343),
fired unconditionally at
[mamba_hybrid.py:303](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L303)
whenever `num_sampled` is a tensor. Not gated on align, so even with the
Option B downgrade this still fires.

The kernel body is trivial — scatter-with-sentinel-filter — so a pure PyTorch
shim suffices:

```python
# cpu_triton_utils.py
def _scatter_num_accepted_impl(
    idx_mapping, num_sampled, num_accepted, **kwargs
):
    valid = idx_mapping >= 0
    if not valid.any():
        return
    targets = idx_mapping[valid].long()
    values = torch.clamp_min(num_sampled[valid], 1).to(num_accepted.dtype)
    num_accepted.scatter_(0, targets, values)

scatter_num_accepted_kernel = _FuncWrapper(_scatter_num_accepted_impl)
```

Then inside `apply_cpu_fallbacks()`:

```python
import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh
mh._scatter_num_accepted_kernel = cpu_tl.scatter_num_accepted_kernel
```

The leading-underscore name is currently a module-private symbol referenced
via `_scatter_num_accepted_kernel[(n,)](...)` at its call site, so the
monkey-patch replaces the name at that spot.

### Same align→all downgrade also applies

Option B in `check_and_update_config` is platform-wide — it already covers V2
automatically, since V2's `_align_mode` flag reads
`cache_config.mamba_cache_mode` at
[mamba_hybrid.py:86](../vllm/v1/worker/gpu/model_states/mamba_hybrid.py#L86).
No V2-specific change needed there.

### Suggested PR split

1. **PR 1 (already staged locally)**: align→all downgrade in
   `CpuPlatform.check_and_update_config`. Ships the V1-CPU fix on its own.
2. **PR 2**: extract `_postprocess_triton` to
   `cpu_triton_utils.apply_cpu_fallbacks()`; call from V2 CPU runner. Fixes
   V2-CPU spec-decode broadly.
3. **PR 3**: add `_scatter_num_accepted_kernel` fallback + monkey-patch in the
   shared helper. Closes the last V2-CPU mamba gap.

Splitting keeps each change small and reviewable and gives clean rollback
boundaries if any follow-up exposes something not covered by V1's existing
patches (e.g. V2-specific kernel signatures that don't match V1's C++ ops).
