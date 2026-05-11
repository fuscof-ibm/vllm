# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform
from vllm.utils.mem_constants import GiB_bytes

from ..utils import create_new_process_for_each_test

# Tiny hybrid (mamba + attention) model, already used by
# tests/models/language/generation/test_hybrid.py. Chosen so the test
# stays fast and does not pull large weights.
HYBRID_MODEL = "hmellor/tiny-random-BambaForCausalLM"

_FORK_MODE = "fork" if not current_platform.is_rocm() else "spawn"


@create_new_process_for_each_test(_FORK_MODE)
def test_hybrid_sleep_wake_roundtrip():
    """
    Hybrid (mamba) sleep/wake correctness.

    On hybrid models the mamba postprocess path caches per-layer state
    data_ptrs and per-group block-table data_ptrs inside
    MambaGPUContext. Those underlying tensors live in the CuMem
    "kv_cache" pool, so their addresses can be invalidated across a
    sleep -> wake_up cycle. If post_kv_cache_wake_up does not
    invalidate the cached addresses, the next decode step either
    crashes on an illegal memory access or silently reads garbage and
    the generated text diverges.

    This test exercises both the full wake path
    (``llm.wake_up()``) and the staged wake path
    (``llm.wake_up(tags=["kv_cache"])``) that is commonly used to
    overlap weight and kv-cache reload.
    """
    free, total = torch.cuda.mem_get_info()
    used_bytes_baseline = total - free

    llm = LLM(
        HYBRID_MODEL,
        enable_sleep_mode=True,
        max_model_len=512,
        max_num_seqs=4,
    )
    prompt = "How are you?"
    params = SamplingParams(temperature=0, max_tokens=16)

    # First forward populates MambaGPUContext metadata.
    output1 = llm.generate(prompt, params)
    text1 = output1[0].outputs[0].text

    # Drop the kv_cache pool. Mamba state tensors and the input_batch
    # block-table tensors both live here.
    llm.sleep(level=1)
    free_after, total = torch.cuda.mem_get_info()
    used_after = total - free_after - used_bytes_baseline
    # Sanity: kv cache memory was actually released.
    assert used_after < 7 * GiB_bytes

    # Full wake triggers post_kv_cache_wake_up which must invalidate
    # MambaGPUContext.is_initialized so the next forward re-captures
    # the remapped data_ptrs.
    llm.wake_up()
    output2 = llm.generate(prompt, params)
    assert text1 == output2[0].outputs[0].text

    # Staged wake path: weights first, then kv_cache. Only the second
    # call re-enters post_kv_cache_wake_up.
    llm.sleep(level=1)
    llm.wake_up(tags=["weights"])
    llm.wake_up(tags=["kv_cache"])
    output3 = llm.generate(prompt, params)
    assert text1 == output3[0].outputs[0].text
