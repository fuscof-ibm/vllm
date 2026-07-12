# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark for the `_copy_mamba_state_block` device function.

Isolates the per-(request, state) mamba state copy body from the surrounding
postprocess decision logic. State layouts for supported models (all tp=1):

- Qwen/Qwen3.5-9B (gated delta net, num_spec=2): 24 linear layers x 2 state
  types = 48 states,
  - conv state: state_len=5, conv_dim=8192 (bf16, 80 KiB/block)
  - temporal: (num_v_heads=32, head_v_dim=128, head_k_dim=128) fp32
    (2 MiB/block; mamba_ssm_cache_dtype from HF config).
- Qwen/Qwen3.5-0.8B (gated delta net, num_spec=2): 18 linear layers x 2 state
  types = 36 states,
  - conv state: state_len=5, conv_dim=6144 (bf16, 60 KiB/block)
  - temporal: (num_v_heads=16, head_v_dim=128, head_k_dim=128) fp32
    (1 MiB/block; mamba_ssm_dtype=float32 in HF config).
- nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (mamba2, num_spec=1): 40
  mamba layers x 2 = 80 states,
  - conv state: state_len=4, conv_dim=10240 (bf16, 80 KiB/block)
  - temporal: (mamba_num_heads=128, mamba_head_dim=64, ssm_state_size=128)
    fp32 (4 MiB/block; mamba_ssm_cache_dtype="float32").

Grid mirrors the fused-kernel launch shape (num_reqs, total_states).

Usage:
    # nsys (whole-run trace)
    nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \
      --force-overwrite=true -o bench_copy_mamba \
      .venv/bin/python bench_copy_mamba_state_block.py

    nsys stats --report cuda_gpu_kern_sum bench_copy_mamba.nsys-rep \
      | grep -i bench_copy_mamba_state_block

    # ncu (kernel metrics; use --nvtx-include to slice per-config).
    # The trailing "/" is Nsight's push/pop match marker, not part of the name;
    # the range names below intentionally contain no "/" so they parse.
    ncu --set full --nvtx --nvtx-include "reqs=64_conv=SD/" \
      -o bench_copy_mamba .venv/bin/python bench_copy_mamba_state_block.py

    # Wall-clock only:
    .venv/bin/python bench_copy_mamba_state_block.py
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.cuda.nvtx as nvtx

from vllm.triton_utils import tl, triton
from vllm.v1.worker.mamba_utils import _copy_mamba_state_block

DEVICE = "cuda"


@dataclass(frozen=True)
class ModelStateConfig:
    """State layout parameters for one hybrid-mamba model at tp=1.

    All values are derived from the model's HF config plus a chosen num_spec.
    Conv/temporal shapes here are per-block (num_blocks is added at runtime).
    """

    name: str
    num_state_layers: int
    conv_dim: int
    conv_state_len: int
    temporal_shape: tuple[int, int, int]
    conv_dtype: torch.dtype
    temporal_dtype: torch.dtype
    num_spec: int
    layer_kind: str  # "gated_delta_net", "mamba2", ...
    notes: str = ""

    @property
    def total_states(self) -> int:
        return self.num_state_layers * 2  # conv + temporal


# Qwen/Qwen3.5-9B: gated delta net, 24 linear + 8 full-attn out of 32.
# See huggingface.co/Qwen/Qwen3.5-9B/config.json and
# MambaStateShapeCalculator.gated_delta_net_state_shape.
#   conv_dim = head_k*num_k_heads*2 + head_v*num_v_heads
#            = 128*16*2 + 128*32 = 8192
#   temporal = (num_v_heads, head_v_dim, head_k_dim) = (32, 128, 128)
_QWEN3_5_9B = ModelStateConfig(
    name="Qwen/Qwen3.5-9B",
    num_state_layers=24,
    conv_dim=8192,
    conv_state_len=4 - 1 + 2,  # conv_kernel - 1 + num_spec
    temporal_shape=(32, 128, 128),
    conv_dtype=torch.bfloat16,
    temporal_dtype=torch.float32,
    num_spec=2,
    layer_kind="gated_delta_net",
    notes="24 linear + 8 full_attention (interval 4) out of 32",
)

# nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4: mamba2, 40 M + 8 * + 40 E
# out of 88 layers (hybrid_override_pattern in HF config).
# See huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/config.json
# and MambaStateShapeCalculator.mamba2_state_shape.
#   intermediate = mamba_num_heads * mamba_head_dim = 128*64 = 8192
#   conv_dim = intermediate + 2*n_groups*ssm_state_size
#            = 8192 + 2*8*128 = 10240
#   temporal = (mamba_num_heads, mamba_head_dim, ssm_state_size)
#            = (128, 64, 128)
# num_spec=1 matches num_nextn_predict_layers in the HF config.
_NEMOTRON3_SUPER_120B = ModelStateConfig(
    name="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    num_state_layers=40,
    conv_dim=10240,
    conv_state_len=4 - 1 + 1,  # conv_kernel - 1 + num_spec
    temporal_shape=(128, 64, 128),
    conv_dtype=torch.bfloat16,
    temporal_dtype=torch.float32,
    num_spec=1,
    layer_kind="mamba2",
    notes="40 mamba + 8 attention + 40 moe out of 88",
)

# Qwen/Qwen3.5-0.8B: gated delta net, 18 linear + 6 full-attn out of 24.
# See huggingface.co/Qwen/Qwen3.5-0.8B/config.json and
# MambaStateShapeCalculator.gated_delta_net_state_shape.
#   conv_dim = head_k*num_k_heads*2 + head_v*num_v_heads
#            = 128*16*2 + 128*16 = 6144
#   temporal = (num_v_heads, head_v_dim, head_k_dim) = (16, 128, 128)
_QWEN3_5_0_8B = ModelStateConfig(
    name="Qwen/Qwen3.5-0.8B",
    num_state_layers=18,
    conv_dim=6144,
    conv_state_len=4 - 1 + 2,  # conv_kernel - 1 + num_spec
    temporal_shape=(16, 128, 128),
    conv_dtype=torch.bfloat16,
    temporal_dtype=torch.float32,
    num_spec=2,
    layer_kind="gated_delta_net",
    notes="18 linear + 6 full_attention (interval 4) out of 24",
)

MODEL_CONFIGS: dict[str, ModelStateConfig] = {
    _QWEN3_5_9B.name: _QWEN3_5_9B,
    _QWEN3_5_0_8B.name: _QWEN3_5_0_8B,
    _NEMOTRON3_SUPER_120B.name: _NEMOTRON3_SUPER_120B,
}

# Each request gets its own src/dst block ids inside every state tensor so
# concurrent copies don't collide and L2 doesn't mask DRAM bandwidth; the
# state tensors are sized to 2*num_reqs blocks in bench_one.
SRC_COL = 0
DST_COL = 1
TOKEN_BIAS = 0

COPY_BLOCK_SIZE = 1024
DEFAULT_ITERS_WARMUP = 20
DEFAULT_ITERS_TIMED = 200


@triton.jit
def bench_copy_mamba_state_block(
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    num_reqs,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
):
    """Minimal wrapper: run the copy body for every (batch, state) program.

    Grid: (num_reqs, total_states).
    """
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    if batch_idx >= num_reqs:
        return
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        state_dim_row_count_ptr,
        state_dim_row_stride_ptr,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
    )


def build_state_tensors(
    model: ModelStateConfig, conv_state_dim_first: bool, num_blocks: int
):
    """Allocate one conv and one temporal state tensor per state-carrying layer.

    Returns interleaved [conv0, temp0, conv1, temp1, ...] to match the flat
    metadata layout produced by MambaSpecDecodeGPUContext.
    """
    conv_shape_per_block = (
        (model.conv_dim, model.conv_state_len)
        if conv_state_dim_first
        else (model.conv_state_len, model.conv_dim)
    )

    state_tensors: list[torch.Tensor] = []
    is_conv_list: list[bool] = []
    for _ in range(model.num_state_layers):
        conv = torch.empty(
            (num_blocks, *conv_shape_per_block),
            dtype=model.conv_dtype,
            device=DEVICE,
        )
        temporal = torch.empty(
            (num_blocks, *model.temporal_shape),
            dtype=model.temporal_dtype,
            device=DEVICE,
        )
        state_tensors.append(conv)
        is_conv_list.append(True)
        state_tensors.append(temporal)
        is_conv_list.append(False)
    return state_tensors, is_conv_list


def build_metadata(state_tensors, is_conv_list, conv_state_dim_first: bool):
    """Pack per-state metadata arrays consumed by `_copy_mamba_state_block`."""
    n = len(state_tensors)
    state_base_addrs = torch.empty(n, dtype=torch.int64, device=DEVICE)
    state_block_strides = torch.empty(n, dtype=torch.int64, device=DEVICE)
    state_elem_sizes = torch.empty(n, dtype=torch.int32, device=DEVICE)
    state_inner_sizes = torch.empty(n, dtype=torch.int64, device=DEVICE)
    state_conv_widths = torch.empty(n, dtype=torch.int32, device=DEVICE)
    state_group_indices = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    state_dim_row_count = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    state_dim_row_stride = torch.zeros(n, dtype=torch.int64, device=DEVICE)

    for i, (t, is_conv) in enumerate(zip(state_tensors, is_conv_list)):
        state_base_addrs[i] = t.data_ptr()
        state_block_strides[i] = t.stride(0) * t.element_size()
        state_elem_sizes[i] = t.element_size()
        if is_conv:
            if conv_state_dim_first:
                # DS layout: (dim, state_len) per block
                state_conv_widths[i] = t.size(2)
                state_inner_sizes[i] = 1
                state_dim_row_count[i] = t.size(1)
                state_dim_row_stride[i] = t.stride(1) * t.element_size()
            else:
                # SD layout: (state_len, dim) per block
                state_conv_widths[i] = t.size(1)
                state_inner_sizes[i] = t.stride(1)
        else:
            state_conv_widths[i] = 0
            state_inner_sizes[i] = t[0].numel()

    return (
        state_base_addrs,
        state_block_strides,
        state_elem_sizes,
        state_inner_sizes,
        state_conv_widths,
        state_group_indices,
        state_dim_row_count,
        state_dim_row_stride,
    )


def bytes_moved_per_launch(state_tensors, is_conv_list, conv_state_dim_first: bool):
    """Bytes copied (read+write) across all (num_reqs, state) programs, per
    request. Multiply by num_reqs to get per-launch total."""
    per_req = 0
    for t, is_conv in zip(state_tensors, is_conv_list):
        elem = t.element_size()
        if is_conv:
            if conv_state_dim_first:
                per_row = (t.size(2) - TOKEN_BIAS) * elem
                copy_bytes = t.size(1) * per_row
            else:
                copy_bytes = (t.size(1) - TOKEN_BIAS) * t.stride(1) * elem
        else:
            copy_bytes = t[0].numel() * elem
        per_req += copy_bytes
    return per_req


def bench_one(
    model: ModelStateConfig,
    num_reqs: int,
    conv_state_dim_first: bool,
    iters_warmup: int,
    iters_timed: int,
) -> float:
    # Give every request distinct src/dst block ids so concurrent copies don't
    # share cache lines. State tensors hold at least 2*num_reqs blocks.
    num_blocks = 2 * num_reqs
    state_tensors, is_conv_list = build_state_tensors(
        model, conv_state_dim_first, num_blocks
    )
    (
        state_base_addrs,
        state_block_strides,
        state_elem_sizes,
        state_inner_sizes,
        state_conv_widths,
        state_group_indices,
        state_dim_row_count,
        state_dim_row_stride,
    ) = build_metadata(state_tensors, is_conv_list, conv_state_dim_first)

    # Single mamba group. Per-request rows point to distinct src/dst blocks:
    # req i reads block 2*i and writes block 2*i+1. Two columns are enough
    # since token_bias=0 (no extra columns read by the temporal path).
    block_table = torch.zeros(num_reqs, 2, dtype=torch.int32, device=DEVICE)
    row_ids = torch.arange(num_reqs, dtype=torch.int32, device=DEVICE)
    block_table[:, SRC_COL] = 2 * row_ids
    block_table[:, DST_COL] = 2 * row_ids + 1
    block_table_ptrs = torch.tensor(
        [block_table.data_ptr()], dtype=torch.int64, device=DEVICE
    )
    block_table_stride_req = block_table.stride(0)

    grid = (num_reqs, model.total_states)
    args = (
        SRC_COL,
        DST_COL,
        TOKEN_BIAS,
        block_table_ptrs,
        block_table_stride_req,
        state_base_addrs,
        state_block_strides,
        state_elem_sizes,
        state_inner_sizes,
        state_conv_widths,
        state_group_indices,
        state_dim_row_count,
        state_dim_row_stride,
        num_reqs,
    )

    def launch():
        bench_copy_mamba_state_block[grid](
            *args,
            COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
            CONV_STATE_DIM_FIRST=conv_state_dim_first,
        )

    for _ in range(iters_warmup):
        launch()
    torch.cuda.synchronize()

    layout = "DS" if conv_state_dim_first else "SD"
    # Underscore separator: '/' is Nsight's NVTX level marker and breaks
    # --nvtx-include filters when it appears inside a range name.
    label = f"reqs={num_reqs}_conv={layout}"
    nvtx.range_push(label)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters_timed):
        launch()
    end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()

    avg_us = start.elapsed_time(end) * 1000.0 / iters_timed
    bytes_per_req = bytes_moved_per_launch(
        state_tensors, is_conv_list, conv_state_dim_first
    )
    total_bytes = bytes_per_req * num_reqs
    # Read + write: memory traffic is 2x copy size.
    gbps = (2 * total_bytes) / (avg_us * 1e-6) / 1e9
    print(
        f"{label:24s} grid={grid!s:16s} "
        f"copy/req={bytes_per_req / 1024 / 1024:6.2f} MiB  "
        f"avg={avg_us:8.2f} us  BW~{gbps:6.1f} GB/s"
    )
    return avg_us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS.keys()),
        default=_QWEN3_5_9B.name,
        help="Model whose state layout drives the benchmark.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names for --model and exit.",
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64, 128],
        help="num_reqs values to sweep (default matches serving benchmark)",
    )
    parser.add_argument(
        "--conv-layout",
        choices=["SD", "DS", "both"],
        default="SD",
        help="Conv state layout to benchmark. Real Qwen3.5-9B is SD.",
    )
    parser.add_argument(
        "--iters-warmup",
        type=int,
        default=DEFAULT_ITERS_WARMUP,
        help="Warmup launches before the timed loop.",
    )
    parser.add_argument(
        "--iters-timed",
        type=int,
        default=DEFAULT_ITERS_TIMED,
        help=(
            "Timed launches inside the NVTX range. Drop to 1 when profiling "
            "with `ncu --launch-count N` so each NVTX range yields one launch."
        ),
    )
    args = parser.parse_args()

    if args.list_models:
        for name in sorted(MODEL_CONFIGS.keys()):
            print(name)
        return

    model = MODEL_CONFIGS[args.model]
    conv_elem = torch.tensor([], dtype=model.conv_dtype).element_size()
    temp_elem = torch.tensor([], dtype=model.temporal_dtype).element_size()
    conv_bytes_per_block = model.conv_dim * model.conv_state_len * conv_elem
    th, td1, td2 = model.temporal_shape
    temp_inner = th * td1 * td2
    temp_bytes_per_block = temp_inner * temp_elem

    assert torch.cuda.is_available(), "CUDA required"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"model: {model.name} ({model.layer_kind}) tp=1 num_spec={model.num_spec}\n"
        f"state_layers={model.num_state_layers} state_types=2 "
        f"total_states={model.total_states}  # {model.notes}\n"
        f"conv: dim={model.conv_dim} state_len={model.conv_state_len} "
        f"dtype={model.conv_dtype} "
        f"(={conv_bytes_per_block / 1024:.1f} KiB/block)\n"
        f"temporal: shape={model.temporal_shape} inner={temp_inner} "
        f"dtype={model.temporal_dtype} "
        f"(={temp_bytes_per_block / 1024 / 1024:.2f} MiB/block)\n"
        f"iters warmup={args.iters_warmup} timed={args.iters_timed}"
    )
    print()

    layouts = (
        [False]
        if args.conv_layout == "SD"
        else ([True] if args.conv_layout == "DS" else [False, True])
    )

    torch.cuda.cudart().cudaProfilerStart()
    t0 = time.perf_counter()
    for conv_ds in layouts:
        for nr in args.concurrencies:
            bench_one(model, nr, conv_ds, args.iters_warmup, args.iters_timed)
    torch.cuda.cudart().cudaProfilerStop()
    print(f"\ntotal wall: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
