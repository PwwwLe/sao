#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
LOG_DIR="$ROOT_DIR/experiments/logs"
QWEN_DIR="$ROOT_DIR/Qwen2AudioInstruct"
SAO_DIR="$ROOT_DIR/StableAudioOpen"

QWEN_HOST="${QWEN_HOST:-0.0.0.0}"
QWEN_PORT="${QWEN_PORT:-8008}"
GRADIO_HOST="${GRADIO_HOST:-0.0.0.0}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
QWEN_SERVICE_URL="${QWEN_SERVICE_URL:-http://127.0.0.1:${QWEN_PORT}/refine_prompt}"
QWEN_START_TIMEOUT="${QWEN_START_TIMEOUT:-600}"
GRADIO_START_TIMEOUT="${GRADIO_START_TIMEOUT:-60}"

HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_CACHE_DIR="${HF_HUB_CACHE:-$HF_HOME_DIR/hub}"
QWEN_HF_CACHE_MODEL_DIR_DEFAULT="$HF_HUB_CACHE_DIR/models--Qwen--Qwen2-Audio-7B-Instruct"
QWEN_LOCAL_MODEL_DIR="${QWEN_LOCAL_MODEL_DIR:-}"

QWEN_PID_FILE="$RUN_DIR/qwen.pid"
GRADIO_PID_FILE="$RUN_DIR/gradio.pid"
QWEN_LOG="$LOG_DIR/qwen.log"
GRADIO_LOG="$LOG_DIR/gradio.log"

PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$RUN_DIR" "$LOG_DIR"

print_help() {
  cat <<'EOF'
Usage:
  ./labctl.sh start      Start Qwen API and Gradio UI
  ./labctl.sh stop       Stop both services
  ./labctl.sh restart    Restart both services
  ./labctl.sh status     Show service status
  ./labctl.sh logs       Tail both logs
  ./labctl.sh check      Check Python dependencies only

Optional environment variables:
  PYTHON_BIN      Python executable path (default: python)
  QWEN_HOST       Qwen server host (default: 0.0.0.0)
  QWEN_PORT       Qwen server port (default: 8008)
  QWEN_LOCAL_MODEL_DIR Local Qwen snapshot dir (auto-detected if empty)
  QWEN_START_TIMEOUT Seconds to wait for Qwen startup (default: 600)
  GRADIO_HOST     Gradio server host (default: 0.0.0.0)
  GRADIO_PORT     Gradio server port (default: 7860)
  GRADIO_START_TIMEOUT Seconds to wait for Gradio startup (default: 60)
  QWEN_SERVICE_URL Override URL used by Gradio to call Qwen
EOF
}

is_running() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  kill -0 "$pid" 2>/dev/null
}

read_pid() {
  local file="$1"
  if [[ -f "$file" ]]; then
    cat "$file"
  fi
}

ensure_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $cmd"
    exit 1
  fi
}

detect_qwen_local_model_dir() {
  local snapshots_dir="$QWEN_HF_CACHE_MODEL_DIR_DEFAULT/snapshots"
  local candidate

  if [[ ! -d "$snapshots_dir" ]]; then
    return 1
  fi

  while IFS= read -r candidate; do
    if [[ -f "$candidate/config.json" ]]; then
      echo "$candidate"
      return 0
    fi
  done < <(ls -td "$snapshots_dir"/* 2>/dev/null || true)

  return 1
}

resolve_qwen_local_model_dir() {
  local detected

  if [[ -n "$QWEN_LOCAL_MODEL_DIR" ]]; then
    if [[ -d "$QWEN_LOCAL_MODEL_DIR" ]]; then
      return 0
    fi
    echo "[ERROR] QWEN_LOCAL_MODEL_DIR does not exist: $QWEN_LOCAL_MODEL_DIR"
    exit 1
  fi

  detected="$(detect_qwen_local_model_dir || true)"
  if [[ -n "$detected" ]]; then
    QWEN_LOCAL_MODEL_DIR="$detected"
  fi
}

validate_qwen_local_model_dir() {
  if [[ -z "$QWEN_LOCAL_MODEL_DIR" ]]; then
    return 0
  fi

  "$PYTHON_BIN" - "$QWEN_LOCAL_MODEL_DIR" <<'PY'
import json
import pathlib
import sys

snapshot = pathlib.Path(sys.argv[1])
idx = snapshot / "model.safetensors.index.json"

if not snapshot.is_dir():
    print(f"[ERROR] Qwen snapshot dir not found: {snapshot}")
    sys.exit(2)

if not idx.exists():
    print(f"[ERROR] Missing index file: {idx}")
    sys.exit(2)

data = json.loads(idx.read_text())
required = sorted(set(data.get("weight_map", {}).values()))
missing = [name for name in required if not (snapshot / name).exists()]

if missing:
    print("[ERROR] Qwen snapshot is incomplete. Missing weight shards:")
    for name in missing:
        print(f" - {name}")
    print("[HINT] Re-download model weights before starting Qwen.")
    sys.exit(3)

print("[OK] Qwen local snapshot integrity check passed")
PY
}

check_port_free() {
  local port="$1"
  local name="$2"
  if lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
    echo "[ERROR] Port $port is already in use ($name)"
    lsof -iTCP:"$port" -sTCP:LISTEN -n -P || true
    exit 1
  fi
}

check_python_deps() {
  "$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = [
    "torch",
    "torchaudio",
    "transformers",
    "fastapi",
    "uvicorn",
    "gradio",
    "requests",
    "einops",
    "stable_audio_tools",
]

missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)

if missing:
    print("[ERROR] Missing Python packages:")
    for mod in missing:
        print(" -", mod)
    print("Install with:")
    print("pip install " + " ".join(missing))
    sys.exit(1)

print("[OK] Python dependency check passed")
PY
}

wait_for_port() {
  local port="$1"
  local timeout_sec="$2"
  local label="$3"
  local pid="$4"
  local elapsed=0
  while (( elapsed < timeout_sec )); do
    if lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
      echo "[OK] $label is listening on port $port"
      return 0
    fi
    if [[ -n "$pid" ]] && ! is_running "$pid"; then
      echo "[ERROR] $label process exited before opening port $port"
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    if (( elapsed % 15 == 0 )); then
      echo "[INFO] Waiting for $label startup... ${elapsed}s/${timeout_sec}s"
    fi
  done
  echo "[ERROR] $label did not start within ${timeout_sec}s"
  return 1
}

start_qwen() {
  local pid
  resolve_qwen_local_model_dir
  validate_qwen_local_model_dir

  pid="$(read_pid "$QWEN_PID_FILE")"
  if is_running "$pid"; then
    echo "[INFO] Qwen already running (pid=$pid)"
    return 0
  fi

  check_port_free "$QWEN_PORT" "Qwen"

  (
    cd "$QWEN_DIR"
    QWEN_LOCAL_MODEL_DIR="$QWEN_LOCAL_MODEL_DIR" \
    nohup "$PYTHON_BIN" -m uvicorn qwen2audio_server:app --host "$QWEN_HOST" --port "$QWEN_PORT" >"$QWEN_LOG" 2>&1 &
    echo $! > "$QWEN_PID_FILE"
  )

  pid="$(read_pid "$QWEN_PID_FILE")"
  echo "[INFO] Starting Qwen (pid=$pid)"
  if [[ -n "$QWEN_LOCAL_MODEL_DIR" ]]; then
    echo "[INFO] Qwen local model dir: $QWEN_LOCAL_MODEL_DIR"
  else
    echo "[INFO] Qwen local model dir: (not found, will use remote model id)"
  fi
  wait_for_port "$QWEN_PORT" "$QWEN_START_TIMEOUT" "Qwen" "$pid" || {
    echo "[ERROR] Qwen failed. See log: $QWEN_LOG"
    echo "[INFO] Last 40 lines of Qwen log:"
    tail -n 40 "$QWEN_LOG" || true
    return 1
  }
}

start_gradio() {
  local pid
  pid="$(read_pid "$GRADIO_PID_FILE")"
  if is_running "$pid"; then
    echo "[INFO] Gradio already running (pid=$pid)"
    return 0
  fi

  check_port_free "$GRADIO_PORT" "Gradio"

  (
    cd "$SAO_DIR"
    QWEN_SERVICE_URL="$QWEN_SERVICE_URL" \
    GRADIO_SERVER_NAME="$GRADIO_HOST" \
    GRADIO_SERVER_PORT="$GRADIO_PORT" \
    nohup "$PYTHON_BIN" gradio_lab.py >"$GRADIO_LOG" 2>&1 &
    echo $! > "$GRADIO_PID_FILE"
  )

  pid="$(read_pid "$GRADIO_PID_FILE")"
  echo "[INFO] Starting Gradio (pid=$pid)"
  wait_for_port "$GRADIO_PORT" "$GRADIO_START_TIMEOUT" "Gradio" "$pid" || {
    echo "[ERROR] Gradio failed. See log: $GRADIO_LOG"
    echo "[INFO] Last 40 lines of Gradio log:"
    tail -n 40 "$GRADIO_LOG" || true
    return 1
  }
}

stop_service() {
  local label="$1"
  local pid_file="$2"
  local pid

  pid="$(read_pid "$pid_file")"
  if ! is_running "$pid"; then
    echo "[INFO] $label is not running"
    rm -f "$pid_file"
    return 0
  fi

  echo "[INFO] Stopping $label (pid=$pid)"
  kill "$pid" >/dev/null 2>&1 || true

  for _ in {1..10}; do
    if ! is_running "$pid"; then
      rm -f "$pid_file"
      echo "[OK] $label stopped"
      return 0
    fi
    sleep 1
  done

  echo "[WARN] Force killing $label (pid=$pid)"
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$pid_file"
}

status_service() {
  local label="$1"
  local pid_file="$2"
  local port="$3"
  local pid

  pid="$(read_pid "$pid_file")"
  if is_running "$pid"; then
    echo "[RUNNING] $label pid=$pid port=$port"
  else
    echo "[STOPPED] $label"
  fi
}

cmd_start() {
  ensure_command lsof
  ensure_command "$PYTHON_BIN"
  check_python_deps
  start_qwen
  start_gradio

  echo
  echo "[READY] Services are up"
  echo "Qwen API:    http://127.0.0.1:${QWEN_PORT}/docs"
  echo "Gradio Lab:  http://127.0.0.1:${GRADIO_PORT}"
  if [[ -n "$QWEN_LOCAL_MODEL_DIR" ]]; then
    echo "Qwen model:  $QWEN_LOCAL_MODEL_DIR"
  else
    echo "Qwen model:  Qwen/Qwen2-Audio-7B-Instruct (remote)"
  fi
  echo "Qwen log:    $QWEN_LOG"
  echo "Gradio log:  $GRADIO_LOG"
}

cmd_stop() {
  stop_service "Gradio" "$GRADIO_PID_FILE"
  stop_service "Qwen" "$QWEN_PID_FILE"
}

cmd_restart() {
  cmd_stop
  cmd_start
}

cmd_status() {
  resolve_qwen_local_model_dir
  status_service "Qwen" "$QWEN_PID_FILE" "$QWEN_PORT"
  status_service "Gradio" "$GRADIO_PID_FILE" "$GRADIO_PORT"
  if [[ -n "$QWEN_LOCAL_MODEL_DIR" ]]; then
    echo "Qwen local model dir: $QWEN_LOCAL_MODEL_DIR"
  else
    echo "Qwen local model dir: (not found, remote id mode)"
  fi
  echo "Qwen log:   $QWEN_LOG"
  echo "Gradio log: $GRADIO_LOG"
}

cmd_logs() {
  touch "$QWEN_LOG" "$GRADIO_LOG"
  tail -n 80 -f "$QWEN_LOG" "$GRADIO_LOG"
}

cmd_check() {
  ensure_command "$PYTHON_BIN"
  check_python_deps
}

ACTION="${1:-help}"
case "$ACTION" in
  start)
    cmd_start
    ;;
  stop)
    cmd_stop
    ;;
  restart)
    cmd_restart
    ;;
  status)
    cmd_status
    ;;
  logs)
    cmd_logs
    ;;
  check)
    cmd_check
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    echo "[ERROR] Unknown command: $ACTION"
    print_help
    exit 1
    ;;
esac
