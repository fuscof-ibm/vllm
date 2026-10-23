#!/bin/bash

for bs in 256; do
  python benchmarks/replayssm/e2e_decode_speedup.py \
    --model-id ibm-granite/granite-4.0-h-small \
    --mamba-backend flashinfer \
    --batch-size $bs --num-steps 500 --no-disable-flashinfer-autotune
done
