#!/bin/bash

for bs in 16; do
  #python benchmarks/replayssm/e2e_decode_speedup.py \
  #  --model-id ibm-granite/granite-4.0-h-tiny \
  #  --mamba-backend flashinfer \
  #  --batch-size $bs --num-steps 500 --no-disable-flashinfer-autotune

  VLLM_USE_V2_MODEL_RUNNER=0 python benchmarks/replayssm/e2e_decode_speedup.py \
    --model-id ibm-granite/granite-4.0-h-tiny \
    --mamba-backend triton \
    --batch-size $bs --num-steps 500 --no-disable-flashinfer-autotune
done
