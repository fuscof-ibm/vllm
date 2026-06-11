#!/usr/bin/env bash
# Sweep torch.compile contribution on Granite 4.
#
# Runs three server configs, each with a client-side concurrency sweep:
#   eager      : --enforce-eager           (no compile, no CUDA graphs)
#   nocompile  : -O.mode=0                 (no compile, CUDA graphs ON)
#   full       : default (mode=3)          (compile + piecewise CUDA graphs)
#
# `full` vs `nocompile` isolates the torch.compile contribution.
# `nocompile` vs `eager`  isolates the CUDA-graph contribution.
#
# Server logs are captured under bench_results/<model_tag>/logs/, and the
# `full` run also writes an FX/pass debug dump for parse_compile_dump.py.
#
# Usage:
#   ./bench_compile_sweep.sh                       # defaults below
#   MODEL=ibm-granite/granite-4.0-h-small \
#   PORT=8000 GPU=0 ./bench_compile_sweep.sh
set -euo pipefail

MODEL="${MODEL:-ibm-granite/granite-4.0-h-small}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL")}"
OUTROOT="./bench_results/compile_sweep/${MODEL_TAG}"
LOGDIR="${OUTROOT}/logs"
DUMPDIR="${OUTROOT}/compile_dump"
mkdir -p "$LOGDIR" "$DUMPDIR"

# Client-side workload (kept small to keep the 3x server boot cost reasonable).
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-200}"
CONCURRENCIES=(${CONCURRENCIES:-1 4 16})

PYBIN=".venv/bin/python"
[[ -x "$PYBIN" ]] || { echo "ERR: $PYBIN not found — activate uv venv first." >&2; exit 1; }

wait_for_ready() {
    local log="$1" deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        # Surface fatal errors fast.
        if grep -qE "Traceback|RuntimeError|CUDA error" "$log" 2>/dev/null; then
            echo "ERR: server failed to start, see $log" >&2
            return 1
        fi
        sleep 5
    done
    echo "ERR: server did not become ready within 1800s" >&2
    return 1
}

start_server() {
    local cfg="$1"; shift
    local logfile="${LOGDIR}/server_${cfg}.log"
    echo "=== starting server: cfg=${cfg} ==="
    # Kill any straggler on this port first.
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 2
    CUDA_VISIBLE_DEVICES="$GPU" \
    VLLM_LOGGING_LEVEL=DEBUG \
        "$PYBIN" -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --port "$PORT" \
            "$@" \
            >"$logfile" 2>&1 &
    SERVER_PID=$!
    if ! wait_for_ready "$logfile"; then
        kill -9 "$SERVER_PID" 2>/dev/null || true
        return 1
    fi
}

stop_server() {
    [[ -n "${SERVER_PID:-}" ]] || return 0
    kill -INT "$SERVER_PID" 2>/dev/null || true
    for _ in {1..30}; do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop_server EXIT

run_client_sweep() {
    local cfg="$1"
    local outdir="${OUTROOT}/${cfg}"
    mkdir -p "$outdir"
    # Discard a warmup at low concurrency.
    "$PYBIN" -m vllm.entrypoints.cli.main bench serve \
        --model "$MODEL" --base-url "http://127.0.0.1:${PORT}" \
        --dataset-name random --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" --num-prompts 32 \
        --max-concurrency 4 --request-rate inf --ignore-eos \
        >/dev/null 2>&1 || true

    for c in "${CONCURRENCIES[@]}"; do
        echo "--- bench cfg=${cfg} concurrency=${c} ---"
        "$PYBIN" -m vllm.entrypoints.cli.main bench serve \
            --model "$MODEL" \
            --base-url "http://127.0.0.1:${PORT}" \
            --dataset-name random \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$c" \
            --request-rate inf --ignore-eos \
            --percentile-metrics "ttft,tpot,itl,e2el" \
            --metric-percentiles "50,90,99" \
            --save-result --result-dir "$outdir" \
            --result-filename "c${c}.json" \
            --label "${cfg}-c${c}"
    done
}

run_cfg() {
    local cfg="$1"; shift
    start_server "$cfg" "$@"
    run_client_sweep "$cfg"
    stop_server
}

# 1) Eager: no compile, no CUDA graphs.
run_cfg eager --enforce-eager

# 2) No compile, CUDA graphs on.
run_cfg nocompile -O.mode=0

# 3) Full compile (default mode=3) with FX/pass dump enabled.
run_cfg full -O.debug_dump_path="${DUMPDIR}" \
             -O.compile_cache_save_format=unpacked

echo
echo "Done. Results: $OUTROOT"
echo "  ./parse_compile_dump.py ${LOGDIR}/server_full.log ${DUMPDIR}"
