# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Benchmark for batch_memcpy_kernel with different BLOCK_SIZE values.

Measures throughput (GB/s) of the Triton byte-copy kernel across various
copy sizes and BLOCK_SIZE settings, with a torch.copy_ baseline.

Usage:
    python benchmarks/kernels/benchmark_batch_memcpy.py
    python benchmarks/kernels/benchmark_batch_memcpy.py --batch-size 64
    python benchmarks/kernels/benchmark_batch_memcpy.py --save-path ./memcpy_bench.png
"""

import torch

from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.mamba_utils import batch_memcpy_kernel

BLOCK_SIZES = [128, 256, 512, 1024, 2048]
COPY_SIZES = [1024, 4096, 16384, 65536, 262144]


def _make_buffers(batch_size: int, copy_size: int, device: str = "cuda"):
    srcs = [
        torch.randint(0, 256, (copy_size,), dtype=torch.uint8, device=device)
        for _ in range(batch_size)
    ]
    dsts = [
        torch.zeros(copy_size, dtype=torch.uint8, device=device)
        for _ in range(batch_size)
    ]

    src_ptrs = torch.tensor(
        [s.data_ptr() for s in srcs], dtype=torch.int64, device=device
    )
    dst_ptrs = torch.tensor(
        [d.data_ptr() for d in dsts], dtype=torch.int64, device=device
    )
    sizes = torch.full((batch_size,), copy_size, dtype=torch.int64, device=device)
    return srcs, dsts, src_ptrs, dst_ptrs, sizes


def get_benchmark(batch_size: int):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["copy_size"],
            x_vals=COPY_SIZES,
            line_arg="provider",
            line_vals=[f"block_{bs}" for bs in BLOCK_SIZES] + ["torch"],
            line_names=[f"BLOCK_SIZE={bs}" for bs in BLOCK_SIZES] + ["torch.copy_"],
            styles=[
                ("blue", "-"),
                ("green", "-"),
                ("red", "-"),
                ("orange", "-"),
                ("purple", "-"),
                ("black", "--"),
            ],
            ylabel="GB/s",
            plot_name=f"batch-memcpy-throughput-batch{batch_size}",
            args={"batch_size": batch_size},
        )
    )
    def benchmark(copy_size, provider, batch_size):
        srcs, dsts, src_ptrs, dst_ptrs, sizes = _make_buffers(batch_size, copy_size)

        if provider == "torch":

            def fn():
                for s, d in zip(srcs, dsts):
                    d.copy_(s)
        else:
            block_size = int(provider.split("_")[1])
            grid = (batch_size,)

            def fn():
                batch_memcpy_kernel[grid](
                    src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=block_size
                )

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

        total_bytes = batch_size * copy_size * 2  # read + write
        gb_per_s = total_bytes / (ms * 1e-3) / 1e9
        gb_per_s_max = total_bytes / (min_ms * 1e-3) / 1e9
        gb_per_s_min = total_bytes / (max_ms * 1e-3) / 1e9
        return gb_per_s, gb_per_s_max, gb_per_s_min

    return benchmark


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark batch_memcpy_kernel block sizes."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of concurrent copies (default: 32)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the plot (default: show only)",
    )
    args = parser.parse_args()

    bench = get_benchmark(args.batch_size)
    bench.run(save_path=args.save_path or "", print_data=True)
