#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_EXE=""
if [ -x "/data01/audio_group/d26_pengwenle/conda/bin/conda" ]; then
    CONDA_EXE="/data01/audio_group/d26_pengwenle/conda/bin/conda"
elif command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
fi

if [ -n "$CONDA_EXE" ]; then
    echo "[Env] Using conda env: dl"
    PYTHON_RUN=("$CONDA_EXE" run -n dl --no-capture-output python)
else
    echo "[Warn] conda not found in PATH, fallback to system python"
    PYTHON_RUN=(python)
fi

mkdir -p logs
mkdir -p artifacts

PROMPT_CACHE_PATH="artifacts/prompt_cache.json"

echo "[Clean] Removing old outputs..."
rm -rf outputs/*
rm -f metrics/results.csv
rm -f "$PROMPT_CACHE_PATH"

cleanup() {
    if [ -f logs/qwen_server.pid ]; then
        PID=$(cat logs/qwen_server.pid || true)
        if [ -n "${PID:-}" ]; then
            echo "[Clean] Stopping Qwen server: $PID"
            kill -9 "$PID" || true
        fi
        rm -f logs/qwen_server.pid
    fi
}
trap cleanup EXIT

# Kill old server
if [ -f logs/qwen_server.pid ]; then
    OLD_PID=$(cat logs/qwen_server.pid || true)
    if [ -n "${OLD_PID:-}" ]; then
        echo "[Clean] Killing old Qwen server: $OLD_PID"
        kill -9 "$OLD_PID" || true
    fi
fi

export HF_HUB_OFFLINE=1
export QWEN_SERVICE_URL_BASE="http://127.0.0.1:8008"
export QWEN_MAX_NEW_TOKENS=768
export QWEN_MIN_SAO_TOKENS=20
export QWEN_MAX_SAO_TOKENS=80

# 建议开启 PyTorch 显存碎片优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[Stage 1] Start Qwen server..."
nohup "${PYTHON_RUN[@]}" -m uvicorn Qwen2AudioInstruct.qwen2audio_server:app \
    --host 127.0.0.1 \
    --port 8008 \
    > logs/qwen_server.log 2>&1 &

QWEN_PID=$!
echo "$QWEN_PID" > logs/qwen_server.pid

echo "[Stage 1] Wait Qwen server..."
SERVER_READY=0
for i in {1..30}; do
    if curl -s "${QWEN_SERVICE_URL_BASE}/docs" > /dev/null; then
        echo "[Stage 1] Server ready."
        SERVER_READY=1
        break
    fi
    sleep 2
done

if [ "$SERVER_READY" -ne 1 ]; then
    echo "[Error] Qwen server did not become ready in time."
    echo "[Hint] Last 50 lines of logs/qwen_server.log:"
    tail -n 50 logs/qwen_server.log || true
    exit 1
fi

echo "[Stage 1] Prepare prompts..."
"${PYTHON_RUN[@]}" scripts/prepare_prompts.py \
    --config config.yaml \
    --output "$PROMPT_CACHE_PATH"

echo "[Stage 1] Prompt cache saved: $PROMPT_CACHE_PATH"

echo "[Stage 2] Stop Qwen server to free GPU memory..."
kill -9 "$QWEN_PID" || true
rm -f logs/qwen_server.pid

# 给 CUDA 驱动一点时间回收显存
sleep 5

echo "[Stage 3] Run SAO experiments with cached prompts..."
export CUDA_VISIBLE_DEVICES=1
"${PYTHON_RUN[@]}" scripts/run_experiments.py \
    --config config.yaml \
    --prompt-cache "$PROMPT_CACHE_PATH"

echo "[Done] Results saved."