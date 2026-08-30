#!/usr/bin/env bash
# shellcheck disable=SC1090   # GPU profile path is selected at runtime
# =============================================================================
# vLLM OpenAI-compatible service manager
#
# Usage:
#   ./launch.sh start     Start service in background
#   ./launch.sh stop      Stop service
#   ./launch.sh restart   Restart service
#   ./launch.sh status    Show process and health status
#   ./launch.sh test      Send a test chat request
#   ./launch.sh logs      Follow service logs
#   ./launch.sh keep      Restart the service after a crash
#
# Environment variables:
#   VLLM_MODEL_REPO      HF model ID or local path; falls back to MODEL_REPO
#   VLLM_MODEL_ROOT      Local model root; falls back to MODEL_ROOT, then /content/models
#   VLLM_MODEL_DIR       Local model directory (default <ROOT>/<repo-name>)
#   VLLM_SERVED_NAME     API model name (default lower-case repo-name)
#   VLLM_HOST / VLLM_PORT  Listen address and port (0.0.0.0 / 30000)
#   VLLM_API_KEY         API key; falls back to API_KEY; explicit empty disables auth
#   VLLM_GPU_MEMORY_UTILIZATION  GPU memory fraction (default 0.90)
#   VLLM_MAX_MODEL_LEN   Optional maximum context length
#   VLLM_MAX_NUM_SEQS    Optional maximum concurrent sequences
#   VLLM_MAX_NUM_BATCHED_TOKENS  Optional batch token limit
#   VLLM_TENSOR_PARALLEL_SIZE    Tensor parallel size (default 1)
#   VLLM_DTYPE           Optional dtype (for example auto or bfloat16)
#   VLLM_QUANTIZATION    Optional quantization backend
#   VLLM_DOWNLOAD_DIR    vLLM download cache (default <ROOT>/.cache/vllm)
#   VLLM_TRUST_REMOTE_CODE=1      Pass --trust-remote-code
#   VLLM_ENABLE_PREFIX_CACHING=1  Pass --enable-prefix-caching
#   VLLM_ENFORCE_EAGER=1          Pass --enforce-eager
#   VLLM_GENERATION_CONFIG   Sampling defaults source: empty=model generation_config.json
#                            (default), vllm=use vLLM defaults instead
#   VLLM_ENABLE_AUTO_TOOL_CHOICE  Enable auto tool choice (default 1)
#   VLLM_TOOL_CALL_PARSER         Tool-call parser; auto-detected from config.json
#   VLLM_VENV_DIR        Python virtual environment (default /tmp/vllm/venv)
#   FLASHINFER_CUDA_ARCH_LIST      FlashInfer JIT arch; auto-detected when unset
#   CUDA_HOME / CUDA_PATH          CUDA toolkit used to detect nvcc version
# =============================================================================

if [[ -n "${DEBUG:-}" ]]; then
  set -eux
else
  set -euo pipefail
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly WORKDIR="$SCRIPT_DIR"
readonly VENV_DIR="${VLLM_VENV_DIR:-/tmp/vllm/venv}"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)"
readonly ROOT_DIR
readonly LOG_DIR="${ROOT_DIR}/logs"
readonly LOG_FILE="${LOG_DIR}/vllm_server.log"
readonly CMD_LOG="${LOG_DIR}/launch_cmd.log"
readonly PID_FILE="${WORKDIR}/vllm.pid"
readonly KEEP_LOG="${LOG_DIR}/keeper.log"

load_gpu_profile() {
  local profile="${GPU_PROFILE:-}"
  if [[ -z "$profile" ]]; then
    local gpu
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    case "$gpu" in
      *T4*)                  profile=t4 ;;
      *G4*|*L4*|*Blackwell*) profile=g4 ;;
    esac
  fi
  [[ -n "$profile" ]] || return 0
  local file="${SCRIPT_DIR}/.env.${profile}"
  [[ -r "$file" ]] || return 0
  echo ">> [vllm] 加载 GPU profile: ${profile} (${file})" >&2
  # shellcheck disable=SC1090
  source "$file"
}

# Return the system nvcc version as major.minor. Do not point CUDA_HOME at a
# venv CUDA package: nvcc and its headers must remain from the same toolkit.
detect_nvcc_version() {
  local nvcc
  nvcc="${CUDA_HOME:-${CUDA_PATH:-}}/bin/nvcc"
  [[ -x "$nvcc" ]] || nvcc="$(command -v nvcc 2>/dev/null || true)"
  [[ -n "$nvcc" && -x "$nvcc" ]] || return 0
  "$nvcc" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1 | awk '{print $2}'
}

nvcc_at_least() {
  local version
  version="$(detect_nvcc_version)"
  [[ -n "$version" ]] || return 1
  awk -v version="$version" -v required="$1" 'BEGIN { exit !(version >= required) }'
}

# FlashInfer rejects SM 12.x when an older system nvcc is used for probing.
# A suffixed arch bypasses that probe while still selecting the right kernels.
configure_flashinfer_arch() {
  if [[ -n "${FLASHINFER_CUDA_ARCH_LIST:-}" ]]; then
    export FLASHINFER_CUDA_ARCH_LIST
    return 0
  fi

  local compute_cap
  compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 | tr -d ' ' || true)"
  case "$compute_cap" in
    12.0)
      if nvcc_at_least 12.9; then
        FLASHINFER_CUDA_ARCH_LIST="12.0f"
      else
        FLASHINFER_CUDA_ARCH_LIST="12.0a"
      fi
      ;;
    12.*|9*|10*|11*)
      FLASHINFER_CUDA_ARCH_LIST="${compute_cap}a"
      ;;
    *.*)
      FLASHINFER_CUDA_ARCH_LIST="$compute_cap"
      ;;
    *)
      FLASHINFER_CUDA_ARCH_LIST=""
      ;;
  esac
  export FLASHINFER_CUDA_ARCH_LIST
}

case "${1:-}" in
  start | restart)
    load_gpu_profile
    configure_flashinfer_arch
    ;;
esac

readonly VLLM_MODEL_REPO="${VLLM_MODEL_REPO:-${MODEL_REPO:-}}"
readonly VLLM_REPO_NAME="${VLLM_MODEL_REPO##*/}"

MODEL_ROOT_SOURCE=""
if [[ -n "${VLLM_MODEL_ROOT:-}" ]]; then
  MODEL_ROOT_SOURCE="VLLM_MODEL_ROOT"
elif [[ -n "${MODEL_ROOT:-}" ]]; then
  VLLM_MODEL_ROOT="$MODEL_ROOT"
  MODEL_ROOT_SOURCE="MODEL_ROOT"
else
  VLLM_MODEL_ROOT="/content/models"
  MODEL_ROOT_SOURCE="默认值(兜底)"
fi
readonly VLLM_MODEL_ROOT MODEL_ROOT_SOURCE

VLLM_MODEL_DIR="${VLLM_MODEL_DIR:-${VLLM_REPO_NAME:+$VLLM_MODEL_ROOT/$VLLM_REPO_NAME}}"
readonly VLLM_MODEL_DIR
readonly VLLM_DOWNLOAD_DIR="${VLLM_DOWNLOAD_DIR:-${VLLM_MODEL_ROOT}/.cache/vllm}"

MODEL_PATH="$VLLM_MODEL_REPO"
SERVED_NAME_DEFAULT="$(basename -- "$MODEL_PATH" 2>/dev/null | tr '[:upper:]' '[:lower:]' | tr '/' '-')"
readonly SERVED_NAME_DEFAULT
readonly SERVED_NAME="${VLLM_SERVED_NAME:-$SERVED_NAME_DEFAULT}"
readonly HOST="${VLLM_HOST:-0.0.0.0}"
readonly PORT="${VLLM_PORT:-30000}"
readonly GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
readonly MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
# vLLM defaults to 1024, which hybrid/mamba models cannot back with Mamba cache
# blocks at these GPU memory budgets; the engine then refuses to capture CUDA
# graphs. 512 stays below the observed per-GPU limit while keeping high batching.
readonly MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-512}"
readonly MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
readonly TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
readonly DTYPE="${VLLM_DTYPE:-auto}"
readonly QUANTIZATION="${VLLM_QUANTIZATION:-}"
# Agent/coding clients send tool_choice=auto; vLLM rejects it unless both
# --enable-auto-tool-choice and --tool-call-parser are set. Enabled by default
# (only affects requests that pass tools); set VLLM_ENABLE_AUTO_TOOL_CHOICE=0 to opt out.
readonly ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
# By default vLLM inherits sampling defaults from the model's generation_config.json
# (Qwen ships temperature 1.0 / top_k 20 / top_p 0.95) and logs a warning. Those are
# the model author's recommended values, so keep them; set VLLM_GENERATION_CONFIG=vllm
# to use vLLM's own defaults instead, or =auto to silence any doubt explicitly.
readonly GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-}"

# Preserve the distinction between an unset key (fall back) and an explicit empty key (disable auth).
if [[ -v VLLM_API_KEY ]]; then
  readonly VLLM_API_KEY_VALUE="$VLLM_API_KEY"
elif [[ -v API_KEY ]]; then
  readonly VLLM_API_KEY_VALUE="$API_KEY"
else
  readonly VLLM_API_KEY_VALUE=""
fi

readonly PROC_PATTERN="vllm [s]erve"

pgrep_server() {
  pgrep -f "$PROC_PATTERN" >/dev/null 2>&1
}

is_running() {
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

wait_stopped() {
  for _ in $(seq 1 30); do
    pgrep_server || return 0
    sleep 1
  done
  return 1
}

require_venv() {
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "错误: 未找到虚拟环境: ${VENV_DIR}" >&2
    echo "请先在项目根目录运行 './colab.sh install vllm'" >&2
    echo "或设置 VLLM_VENV_DIR 指向已有 venv" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  if ! command -v vllm >/dev/null 2>&1; then
    echo "错误: venv 中未找到 vllm 命令: ${VENV_DIR}" >&2
    echo "请重新运行 './colab.sh install vllm'" >&2
    exit 1
  fi
}

reject_drive_path() {
  local path="$1"
  if [[ "$path" == /content/drive || "$path" == /content/drive/* ]]; then
    echo "错误: vLLM 不支持 Google Drive 作为模型或缓存目录: $path" >&2
    echo "请先用 './colab.sh sync pull' 将模型复制到本地盘" >&2
    exit 1
  fi
}

resolve_model_path() {
  local repo="$MODEL_PATH"
  case "$repo" in
    /* | ./* | ../*)
      return 0
      ;;
  esac
  [[ -n "$VLLM_MODEL_DIR" ]] || {
    echo "错误: 未设置 VLLM_MODEL_DIR, 无法将 HF 模型保存到本地盘" >&2
    exit 1
  }
  if [[ -f "${VLLM_MODEL_DIR}/config.json" ]]; then
    echo ">> 命中本地模型目录: ${VLLM_MODEL_DIR}" >&2
    MODEL_PATH="$VLLM_MODEL_DIR"
    return 0
  fi
  if ! command -v hf >/dev/null 2>&1; then
    echo "错误: 未找到 hf 命令, 无法下载模型 $repo" >&2
    echo "请先运行 './colab.sh install vllm' 或安装 huggingface_hub" >&2
    exit 1
  fi
  echo ">> 下载模型 ${repo} -> ${VLLM_MODEL_DIR} ..." >&2
  if hf download "$repo" --local-dir "$VLLM_MODEL_DIR"; then
    MODEL_PATH="$VLLM_MODEL_DIR"
  else
    echo "错误: 模型下载失败, 未启动 vLLM" >&2
    exit 1
  fi
}

# Pick the vLLM tool-call parser for the served model. Explicit VLLM_TOOL_CALL_PARSER
# wins; otherwise use the model's config.json family. Unknown families return empty.
detect_tool_parser() {
  if [[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]; then
    printf '%s' "$VLLM_TOOL_CALL_PARSER"
    return 0
  fi
  local config="${MODEL_PATH}/config.json"
  [[ -f "$config" ]] || return 0
  local text
  text="$(python3 - "$config" <<'PY' 2>/dev/null || true
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(0)
parts = [str(data.get("architectures", "")), str(data.get("model_type", ""))]
print(" ".join(parts).lower())
PY
)"
  [[ -n "$text" ]] || return 0
  case "$text" in
    *qwen*)     printf 'qwen3_coder' ;;
    *deepseek*) printf 'deepseek_v3' ;;
    *glm47*)    printf 'glm47' ;;
    *glm*)      printf 'glm45' ;;
    *kimi*)     printf 'kimi_k2' ;;
    *minimax*)  printf 'minimax_m2' ;;
    *mistral*)  printf 'mistral' ;;
    *llama*)    printf 'llama3_json' ;;
    *)          printf '' ;;
  esac
}

# vLLM scans VLLM_* environment variables and warns about unknown ones. These
# names are this script's own configuration; their values are passed as CLI
# flags, so they are stripped from the server process with `env -u` (which also
# works for variables this script marks readonly, and so cannot unset).
readonly WRAPPER_ENV_NAMES=(
  VLLM_MODEL_REPO VLLM_MODEL_ROOT VLLM_MODEL_DIR VLLM_DOWNLOAD_DIR
  VLLM_SERVED_NAME VLLM_HOST VLLM_PORT VLLM_GPU_MEMORY_UTILIZATION
  VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
  VLLM_TENSOR_PARALLEL_SIZE VLLM_DTYPE VLLM_QUANTIZATION VLLM_VENV_DIR
  VLLM_ENABLE_AUTO_TOOL_CHOICE VLLM_TOOL_CALL_PARSER VLLM_TRUST_REMOTE_CODE
  VLLM_ENABLE_PREFIX_CACHING VLLM_ENFORCE_EAGER VLLM_GENERATION_CONFIG
)

check_start_config() {
  if [[ -z "$MODEL_PATH" ]]; then
    echo "错误: VLLM_MODEL_REPO / MODEL_REPO 均为空, 请设置 HF 模型 ID 或本地路径" >&2
    exit 1
  fi
  reject_drive_path "$VLLM_MODEL_ROOT"
  reject_drive_path "$VLLM_MODEL_DIR"
  reject_drive_path "$VLLM_DOWNLOAD_DIR"
  reject_drive_path "$MODEL_PATH"
  if [[ -z "${VLLM_API_KEY+x}" && -z "${API_KEY+x}" ]]; then
    echo "错误: 未设置 VLLM_API_KEY / API_KEY" >&2
    echo '若要临时关闭鉴权: VLLM_API_KEY="" ./launch.sh start' >&2
    exit 1
  fi
}

do_start() {
  if is_running; then
    echo "服务已在运行 (PID $(cat "$PID_FILE")), 如需重启请执行: $0 restart"
    exit 0
  fi
  if pgrep_server; then
    echo "检测到无 PID 文件的残留进程, 请先执行: $0 stop" >&2
    exit 1
  fi

  check_start_config
  require_venv
  resolve_model_path
  mkdir -p "$LOG_DIR" "$VLLM_DOWNLOAD_DIR"

  local args=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --host "$HOST"
    --port "$PORT"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --download-dir "$VLLM_DOWNLOAD_DIR"
    --dtype "$DTYPE"
  )
  [[ -n "$MAX_MODEL_LEN" ]] && args+=(--max-model-len "$MAX_MODEL_LEN")
  [[ -n "$MAX_NUM_SEQS" ]] && args+=(--max-num-seqs "$MAX_NUM_SEQS")
  [[ -n "$MAX_NUM_BATCHED_TOKENS" ]] && args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  [[ "$TENSOR_PARALLEL_SIZE" != "1" ]] && args+=(--tensor-parallel-size "$TENSOR_PARALLEL_SIZE")
  [[ -n "$QUANTIZATION" ]] && args+=(--quantization "$QUANTIZATION")
  [[ -n "$GENERATION_CONFIG" ]] && args+=(--generation-config "$GENERATION_CONFIG")
  [[ "${VLLM_TRUST_REMOTE_CODE:-0}" == "1" ]] && args+=(--trust-remote-code)
  [[ "${VLLM_ENABLE_PREFIX_CACHING:-0}" == "1" ]] && args+=(--enable-prefix-caching)
  [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]] && args+=(--enforce-eager)
  [[ -n "$VLLM_API_KEY_VALUE" ]] && args+=(--api-key "$VLLM_API_KEY_VALUE")

  # Auto tool choice requires a parser; without both, clients sending
  # tool_choice=auto get HTTP 400. Skipped when the model family is unknown.
  if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
    local tool_parser
    tool_parser="$(detect_tool_parser)"
    if [[ -n "$tool_parser" ]]; then
      args+=(--enable-auto-tool-choice --tool-call-parser "$tool_parser")
    else
      echo "警告: 未识别模型家族的工具调用解析器, 不启用 --enable-auto-tool-choice; 可用 VLLM_TOOL_CALL_PARSER 显式指定" >&2
    fi
  fi

  # Keep credentials out of both the service log and the command audit log.
  local -a log_args=("${args[@]}")
  local arg_index
  for ((arg_index = 0; arg_index + 1 < ${#log_args[@]}; arg_index++)); do
    if [[ "${log_args[$arg_index]}" == "--api-key" ]]; then
      log_args[$((arg_index + 1))]='<redacted>'
    fi
  done

  local -a env_clean=()
  local env_name
  for env_name in "${WRAPPER_ENV_NAMES[@]}"; do
    env_clean+=(-u "$env_name")
  done

  echo "启动 vLLM 服务... (日志: ${LOG_FILE})" | tee -a "$LOG_FILE"
  echo ">> 模型: $MODEL_PATH" | tee -a "$LOG_FILE"
  echo ">> 模型基础盘: $VLLM_MODEL_ROOT (来源: $MODEL_ROOT_SOURCE)" | tee -a "$LOG_FILE"
  echo ">> FlashInfer 架构: FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-<自动>} (nvcc $(detect_nvcc_version))" | tee -a "$LOG_FILE"
  echo ">> 启动命令: ${log_args[*]}" | tee -a "$LOG_FILE"
  {
    echo "===== $(date '+%F %T') [vllm] start ====="
    printf '  %s\n' "${log_args[@]}"
  } >>"$CMD_LOG"

  # shellcheck disable=SC2016
  setsid env "${env_clean[@]}" bash -c '
    echo $$ > "$1"; shift
    exec "$@"
  ' bash "$PID_FILE" "${args[@]}" >>"$LOG_FILE" 2>&1 </dev/null &

  sleep 1
  if is_running; then
    echo "已提交启动 (PID $(cat "$PID_FILE")), 首次加载可能需要较长时间; 请用 '$0 status' 查看就绪状态"
  else
    echo "启动失败, 请查看日志: $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
}

do_stop() {
  if ! pgrep_server; then
    echo "服务未在运行"
    rm -f "$PID_FILE"
    return 0
  fi
  echo "停止 vLLM 服务..."
  pkill -TERM -f "$PROC_PATTERN" 2>/dev/null || true
  if wait_stopped; then
    echo "已停止"
  else
    echo "优雅停止超时, 强制终止..."
    pkill -KILL -f "$PROC_PATTERN" 2>/dev/null || true
    sleep 2
    pgrep_server && echo "仍有关联进程, 请手动检查" >&2 || echo "已停止"
  fi
  rm -f "$PID_FILE"
}

do_status() {
  if is_running; then
    echo "进程: 运行中 (PID $(cat "$PID_FILE"))"
  elif pgrep_server; then
    echo "进程: 运行中 (无 PID 文件)"
  else
    echo "进程: 已停止"
  fi
  local -a auth=()
  [[ -n "$VLLM_API_KEY_VALUE" ]] && auth=(-H "Authorization: Bearer ${VLLM_API_KEY_VALUE}")
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "${auth[@]}" "http://localhost:${PORT}/health" 2>/dev/null || true)
  if [[ "$code" == "200" ]]; then
    echo "健康检查: HTTP 200 (就绪)"
  else
    echo "健康检查: HTTP ${code:-000} (启动中或异常, 查看 logs)"
  fi
}

do_test() {
  if ! is_running && ! pgrep_server; then
    echo "错误: vLLM 服务未在运行, 请先执行: $0 start" >&2
    exit 1
  fi
  local -a auth=()
  [[ -n "$VLLM_API_KEY_VALUE" ]] && auth=(-H "Authorization: Bearer ${VLLM_API_KEY_VALUE}")
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "${auth[@]}" "http://localhost:${PORT}/health" 2>/dev/null || true)
  if [[ "$code" != "200" ]]; then
    echo "错误: 服务尚未就绪(health HTTP ${code:-000}), 请稍后重试或查看 logs" >&2
    exit 1
  fi
  echo ">> 发送测试对话 (model=${SERVED_NAME}, port=${PORT}) ..."
  curl -s --max-time 60 "http://localhost:${PORT}/v1/chat/completions" \
    "${auth[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${SERVED_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"你好, 请用一句话回复\"}], \"max_tokens\": 64}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print("回复:", d["choices"][0]["message"].get("content", ""))' 2>/dev/null \
    || { echo "错误: 请求失败, 请检查服务状态和日志" >&2; exit 1; }
}

do_keep() {
  mkdir -p "$LOG_DIR"
  echo "进入守护模式 (PID $$), 日志: ${KEEP_LOG}"
  while true; do
    if ! pgrep_server; then
      echo "[$(date '+%F %T')] 服务进程消失, 自动重启..." >>"$KEEP_LOG"
      "$0" start >>"$KEEP_LOG" 2>&1 || true
    fi
    sleep 30
  done
}

usage() {
  grep -E '^# +(用法|  \.|  VLLM_|  API_KEY|    VLLM_)' "$0" | sed 's/^# \{1,\}//'
}

case "${1:-}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart)
    do_stop
    do_start
    ;;
  status)  do_status ;;
  test)    do_test ;;
  keep)    do_keep ;;
  logs)    tail -n 100 -f "$LOG_FILE" ;;
  help | -h | --help) usage ;;
  *)
    usage
    exit 1
    ;;
esac
