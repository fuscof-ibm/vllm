#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

vllm serve   \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8    \
    --tensor-parallel-size 4        \
    --max-model-len 262144          \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'  \
    --enable-prefix-caching

