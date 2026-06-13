#!/usr/bin/env bash
# Client-side sweep to highlight PR #40172 (fused Mamba postprocess kernel).
# Assumes the server is already running on localhost:8000.
#
# Usage:
#   ./bench_sweep.sh main            # run against the `main` server, tag results
#   ./bench_sweep.sh pr40172         # run against the PR server, tag results
#   LABEL=foo ./bench_sweep.sh foo   # custom tag
set -euo pipefail

LABEL="${1:-${LABEL:-run}}"
OUTDIR="./bench_results/${LABEL}"
mkdir -p "$OUTDIR"

# Decode-heavy random dataset: short prompt, long output.
# This is where the per-step CPU bubble matters most.
#
MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
INPUT_LEN=256
OUTPUT_LEN=2048
NUM_PROMPTS=500

# Warmup (discarded) then sweep concurrency.
CONCURRENCIES=(1 4 8 16 32)

run_one() {
    local conc="$1"
    local tag="$2"
    local outfile="${OUTDIR}/c${conc}_${tag}.json"
    echo "=== [$LABEL] concurrency=${conc} tag=${tag} ==="
    vllm bench serve \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate inf \
        --max-concurrency "$conc" \
        --ignore-eos \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "50,90,99" \
        --save-result \
        --result-dir "$OUTDIR" \
        --result-filename "$(basename "$outfile")" \
        --label "${LABEL}-c${conc}-${tag}"
}

# One warmup at low concurrency to populate caches / JIT Triton.
run_one 4 warmup
rm -f "${OUTDIR}/c4_warmup.json"

for c in "${CONCURRENCIES[@]}"; do
    run_one "$c" run1
    run_one "$c" run2
    run_one "$c" run3
done

echo "Done. Results in $OUTDIR"
