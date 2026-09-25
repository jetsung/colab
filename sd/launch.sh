#!/usr/bin/env bash
# ============================================================
# stable-diffusion.cpp 服务管理脚本 (sd-cli / sd-server)
#
# 用法:
#   ./launch.sh start     启动 sd-server(后台, setsid 托管; 首次自动下载模型)
#   ./launch.sh stop      停止服务
#   ./launch.sh restart   重启服务
#   ./launch.sh status    查看状态 + 健康检查
#   ./launch.sh test      调用服务生成一张测试图(需服务已就绪)
#   ./launch.sh generate [提示词...]  用 sd-cli 一次性出图(不依赖服务)
#   ./launch.sh logs      跟踪日志
#   ./launch.sh keep      守护模式(崩溃自动拉起)
#
# 依赖:
#   - install 已装好 sd 二进制(colab.sh install sd [--build])
#   - 首次自动下载模型时需要 `hf` 命令(huggingface_hub, 由 install 装入)
#
# 模型组件(每项都可用 <VAR> 直接给模型来源, 或 <VAR>_REPO/<VAR>_FILE 自动下载):
#   SD_MODEL            -> -m               全模型单文件(SD1.5/SD3/...)
#   SD_DIFFUSION_MODEL  -> --diffusion-model 组件式扩散主干(FLUX/Qwen-Image/...)
#   SD_VAE              -> --vae
#   SD_LLM              -> --llm            文本编码器(Qwen-Image 用 Qwen3-VL)
#   SD_LLM_VISION       -> --llm_vision     视觉投影器(图生图/编辑, 可选)
#   SD_CLIP_L / SD_CLIP_G / SD_T5XXL -> --clip_l / --clip_g / --t5xxl (SD3/FLUX, 可选)
#   SD_MODEL 与 SD_DIFFUSION_MODEL 至少配置一个(见 do_start 校验)
#
# <VAR> 单值来源支持四种写法(见 parse_model_source):
#   本地路径   /abs/path 或 ./rel 或 rel/path(相对当前目录)
#   file://    file:///abs/path(绝对) 或 file://rel/path(相对当前目录展开) — 直接使用
#   hf://      hf://<org>/<repo>/<file>
#   HF https   https://huggingface.co/<org>/<repo>/(blob|resolve)/<rev>/<file>
#              — 两者先归一为 hf://<org>/<repo>/<file>, 用 hf download 下载到 HF 标准缓存
#              ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/ 并定位真实文件路径
#
# 环境变量(可被外部/命令行覆盖):
#   SD_DIR           安装目录(默认 /content/stable-diffusion.cpp)
#   SD_SERVER / SD_CLI  二进制路径(默认 <SD_DIR>/build/bin/...; 亦从 PATH 查找)
#   HF 缓存目录遵循 HF_HUB_CACHE / HF_HOME(默认 ~/.cache/huggingface/hub)
#   SD_HOST / SD_PORT   监听地址与端口(默认 0.0.0.0 / 30000, 与 bore 隧道一致)
#   SD_STEPS / SD_CFG_SCALE / SD_SAMPLING_METHOD / SD_WIDTH / SD_HEIGHT / SD_SEED
#                    生成默认参数(留空则不传, 由 sd.cpp 按模型决定)
#   SD_DIFFUSION_FA / SD_OFFLOAD_TO_CPU / SD_VAE_TILING   1=启用对应开关
#   SD_THREADS       CPU 推理线程数(留空或 -1 交由 sd.cpp 按物理核心数决定)
#   SD_LORA_DIR      LoRA 目录(--lora-model-dir)
#   SD_OUTPUT_DIR    generate/test 出图目录(默认 /content/outputs)
#   SD_PROMPT        generate/test 默认提示词(默认 "a lovely cat")
#   SD_REQUEST_TIMEOUT  test 请求超时秒数(默认 1800; CPU 出图较慢)
#   SD_XET           1=启用 HuggingFace Xet 存储(默认), 0=禁用
# ============================================================

if [[ -n "${DEBUG:-}" ]]; then
  set -eux
else
  set -euo pipefail
fi

# ----------------------------- 可调配置 --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

# ------------------------- 平台探测(CPU / GPU) ------------------------------
has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  local gpu=""
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  [[ -n "$gpu" ]]
}

is_cpu_platform() {
  local profile="${GPU_PROFILE:-}"
  if [[ -n "$profile" ]]; then
    if [[ "$profile" == "cpu" ]]; then
      return 0
    fi
    return 1
  fi
  if has_nvidia_gpu; then
    return 1
  fi
  return 0
}

# profile 兜底加载(不依赖 direnv): 调用方未 cd 进本目录时按 GPU_PROFILE(未设置则探测)
# source 本目录 .env.<g4|t4|cpu>; profile 是纯 bash, 幂等且不覆盖已有值。
load_gpu_profile() {
  local profile="${GPU_PROFILE:-}"
  if [[ -z "$profile" ]]; then
    local gpu=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    fi
    case "$gpu" in
      *T4*)                  profile=t4 ;;
      *G4*|*L4*|*Blackwell*) profile=g4 ;;
      "")                    profile=cpu ;;
    esac
  fi
  [[ -n "$profile" ]] || return 0
  local f="${SCRIPT_DIR}/.env.${profile}"
  [[ -r "$f" ]] || return 0
  echo ">> [sd] 加载 profile: ${profile} (${f})" >&2
  # shellcheck disable=SC1090
  source "$f"
}
# 仅需要模型/生成参数的子命令才兜底, 避免 status/stop/logs 也去探测显卡
case "${1:-}" in
  start | restart | generate | test) load_gpu_profile ;;
esac

# ----------------------------- 路径与默认值 ----------------------------------
readonly SD_DIR="${SD_DIR:-/content/stable-diffusion.cpp}"
readonly ROOT_DIR="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)"   # 项目根目录 = sd/ 的父目录
readonly LOG_DIR="${ROOT_DIR}/logs"
readonly LOG_FILE="${LOG_DIR}/sd_server.log"
readonly CMD_LOG="${LOG_DIR}/launch_cmd.log"
readonly PID_FILE="${SCRIPT_DIR}/sd.pid"
readonly KEEP_LOG="${LOG_DIR}/keeper.log"

# 二进制解析: 显式 SD_SERVER/SD_CLI > PATH > <SD_DIR>/build/bin
if [[ -n "${SD_SERVER:-}" ]]; then
  :
elif command -v sd-server >/dev/null 2>&1; then
  SD_SERVER="$(command -v sd-server)"
else
  SD_SERVER="$SD_DIR/build/bin/sd-server"
fi
readonly SD_SERVER
if [[ -n "${SD_CLI:-}" ]]; then
  :
elif command -v sd-cli >/dev/null 2>&1; then
  SD_CLI="$(command -v sd-cli)"
else
  SD_CLI="$SD_DIR/build/bin/sd-cli"
fi
readonly SD_CLI

# 监听与运行时(仅 start/test/status 需要; 其余子命令无害)
readonly PORT="${SD_PORT:-30000}"
readonly HOST="${SD_HOST:-0.0.0.0}"
readonly SD_XET="${SD_XET:-1}"
readonly SD_OUTPUT_DIR="${SD_OUTPUT_DIR:-/content/outputs}"
readonly SD_PROMPT="${SD_PROMPT:-a lovely cat}"
readonly SD_REQUEST_TIMEOUT="${SD_REQUEST_TIMEOUT:-1800}"

# pgrep 匹配模式(括号防自匹配): 进程名 sd-server
readonly PROC_PATTERN="sd-serve[r]"

# ----------------------------- 内部函数 --------------------------------------
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

# 解析模型来源字符串为统一形态, stdout: local|<path> 或 hf|<org>/<repo>|<file>
# 支持写法:
#   本地路径  /abs 或 ./rel 或 rel/path(相对当前目录, 原样透传)
#   file://   file:///abs(绝对路径) 或 file://rel/path(相对当前目录展开)
#   hf://     hf://<org>/<repo>/<file>
#   HF https  https://huggingface.co/<org>/<repo>/(blob|resolve)/<rev>/<file>
parse_model_source() {
  local val="$1" rest org repo seg3
  case "$val" in
    file://*)
      printf 'local|%s' "${val#file://}"
      return 0
      ;;
    hf://*)
      rest="${val#hf://}"
      ;;
    https://huggingface.co/*)
      rest="${val#https://huggingface.co/}"
      org="${rest%%/*}"; rest="${rest#*/}"
      repo="${rest%%/*}"; rest="${rest#*/}"
      seg3="${rest%%/*}"
      if [[ "$seg3" != "blob" && "$seg3" != "resolve" ]]; then
        echo "ERROR: 不支持的 HF URL(仅支持 /blob/ 或 /resolve/ 文件直链): $val" >&2
        return 1
      fi
      rest="${rest#*/}"   # 跳过 blob|resolve
      rest="${rest#*/}"   # 跳过 revision
      if [[ -z "$rest" ]]; then
        echo "ERROR: HF URL 缺少文件路径: $val" >&2
        return 1
      fi
      printf 'hf|%s/%s|%s' "$org" "$repo" "$rest"
      return 0
      ;;
    *)
      printf 'local|%s' "$val"
      return 0
      ;;
  esac
  # hf:// 剩余部分: <org>/<repo>/<file>(file 可含子目录); 至少要有两段 /
  if [[ "$rest" != */*/* ]]; then
    echo "ERROR: hf:// 路径需形如 hf://<org>/<repo>/<file>: $val" >&2
    return 1
  fi
  org="${rest%%/*}"; rest="${rest#*/}"
  repo="${rest%%/*}"; rest="${rest#*/}"
  printf 'hf|%s/%s|%s' "$org" "$repo" "$rest"
}

# 在 HF hub 缓存的 snapshots/ 下定位已下载文件, 取最新一份
#   $1 = models--<org>--<repo> 缓存目录, $2 = 仓库内相对文件路径(可含子目录)
cache_find() {
  local found=""
  found="$(find -L "$1/snapshots" -type f -path "*/$2" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-)"
  [[ -n "$found" && -f "$found" ]] && printf '%s' "$found"
  return 0
}

# 解析单个模型组件: resolve_component <VAR名>
#   stdout: 解析后的本地路径(未配置则为空串)
#   stderr: 进度/错误信息
# 优先级: <VAR> 单值来源 > <VAR>_REPO/<VAR>_FILE 组合
#   - 本地路径 / file://    直接使用, 不联网不下载
#   - hf:// / HF https URL  归一为 hf://<org>/<repo>/<file>, 先查 HF hub 标准缓存
#     (~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/), 未命中则
#     hf download hf://<org>/<repo>/<file> 下载, 再从 snapshots/ 定位真实文件路径
resolve_component() {
  local name="$1"
  local repo="" file=""

  local explicit="${!name:-}"
  if [[ -n "$explicit" ]]; then
    local parsed
    parsed="$(parse_model_source "$explicit")" || return 1
    if [[ "$parsed" == 'local|'* ]]; then
      local local_path="${parsed#local|}"
      if [[ ! -f "$local_path" ]]; then
        echo "ERROR: ${name} 指向的文件不存在: $local_path" >&2
        return 1
      fi
      printf '%s' "$local_path"
      return 0
    fi
    parsed="${parsed#hf|}"
    repo="${parsed%%|*}"
    file="${parsed#*|}"
  else
    local repo_var="${name}_REPO" file_var="${name}_FILE"
    repo="${!repo_var:-}" file="${!file_var:-}"
    if [[ -z "$repo" && -z "$file" ]]; then
      return 0
    fi
    if [[ -z "$repo" || -z "$file" ]]; then
      echo "ERROR: ${name} 需同时设置 ${repo_var} 与 ${file_var}(或直接给 ${name} 单值来源)" >&2
      return 1
    fi
  fi

  # HF 来源: hub 缓存目录 models--<org>--<repo>(仓库路径中的 / 替换为 --)
  local hub_dir="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
  local model_dir="${hub_dir}/models--${repo//\//--}"
  local hf_url="hf://${repo}/${file}"

  local found
  found="$(cache_find "$model_dir" "$file")"
  if [[ -n "$found" ]]; then
    echo ">> [${name}] 缓存已存在: ${found}" >&2
    printf '%s' "$found"
    return 0
  fi

  if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 未找到 hf 命令(用于下载模型)。请先执行: ./colab.sh install sd" >&2
    return 1
  fi
  echo ">> [${name}] 下载 ${hf_url}" >&2
  local out=""
  if [[ "$SD_XET" == "1" ]]; then
    out="$(HF_HUB_ENABLE_XET=1 HF_TOKEN="${HF_TOKEN:-}" hf download "$hf_url")" || {
      echo "ERROR: hf download 失败: ${hf_url}" >&2
      return 1
    }
  else
    out="$(HF_TOKEN="${HF_TOKEN:-}" hf download "$hf_url")" || {
      echo "ERROR: hf download 失败: ${hf_url}" >&2
      return 1
    }
  fi
  # hf download 成功时 stdout 末行为下载文件路径(新版格式 "  path: <路径>", 旧版为裸路径),
  # 解析失效则回退到缓存查找
  out="${out##*$'\n'}"
  out="${out##*'path: '}"
  if [[ -n "$out" && -f "$out" ]]; then
    printf '%s' "$out"
    return 0
  fi
  found="$(cache_find "$model_dir" "$file")"
  if [[ -n "$found" ]]; then
    printf '%s' "$found"
    return 0
  fi
  echo "ERROR: 下载完成但未定位到文件: ${file} (缓存目录: ${model_dir})" >&2
  return 1
}

# 组装模型参数到全局 MODEL_ARGS; 校验至少配置一个主模型
MODEL_ARGS=()
build_model_args() {
  MODEL_ARGS=()
  local p

  p="$(resolve_component SD_MODEL)" || exit 1
  [[ -n "$p" ]] && MODEL_ARGS+=(-m "$p")

  p="$(resolve_component SD_DIFFUSION_MODEL)" || exit 1
  [[ -n "$p" ]] && MODEL_ARGS+=(--diffusion-model "$p")

  if [[ ${#MODEL_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: 未配置主模型。请设置 SD_MODEL(全模型)或 SD_DIFFUSION_MODEL(组件式扩散模型)。" >&2
    exit 1
  fi

  p="$(resolve_component SD_VAE)" || exit 1
  [[ -n "$p" ]] && MODEL_ARGS+=(--vae "$p")

  p="$(resolve_component SD_LLM)" || exit 1
  [[ -n "$p" ]] && MODEL_ARGS+=(--llm "$p")

  p="$(resolve_component SD_LLM_VISION)" || exit 1
  [[ -n "$p" ]] && MODEL_ARGS+=(--llm_vision "$p")

  p="$(resolve_component SD_CLIP_L)" || exit 1
  [[ -n "$p" ]] && MODEL_ARGS+=(--clip_l "$p")

  p="$(resolve_component SD_CLIP_G)" || exit 1
  [[ -n "$p" ]] && MODEL_ARGS+=(--clip_g "$p")

  p="$(resolve_component SD_T5XXL)" || exit 1
  [[ -n "$p" ]] && MODEL_ARGS+=(--t5xxl "$p")

  return 0
}

# 组装生成参数(server 默认值与 generate 共用)
GEN_ARGS=()
build_gen_args() {
  GEN_ARGS=()
  if [[ -n "${SD_NEGATIVE_PROMPT:-}" ]]; then GEN_ARGS+=(--negative-prompt "$SD_NEGATIVE_PROMPT"); fi
  if [[ -n "${SD_STEPS:-}" ]]; then GEN_ARGS+=(--steps "$SD_STEPS"); fi
  if [[ -n "${SD_CFG_SCALE:-}" ]]; then GEN_ARGS+=(--cfg-scale "$SD_CFG_SCALE"); fi
  if [[ -n "${SD_SAMPLING_METHOD:-}" ]]; then GEN_ARGS+=(--sampling-method "$SD_SAMPLING_METHOD"); fi
  if [[ -n "${SD_SCHEDULER:-}" ]]; then GEN_ARGS+=(--scheduler "$SD_SCHEDULER"); fi
  if [[ -n "${SD_WIDTH:-}" ]]; then GEN_ARGS+=(-W "$SD_WIDTH"); fi
  if [[ -n "${SD_HEIGHT:-}" ]]; then GEN_ARGS+=(-H "$SD_HEIGHT"); fi
  if [[ -n "${SD_SEED:-}" ]]; then GEN_ARGS+=(--seed "$SD_SEED"); fi
  if [[ -n "${SD_FLOW_SHIFT:-}" ]]; then GEN_ARGS+=(--flow-shift "$SD_FLOW_SHIFT"); fi
  if [[ "${SD_DIFFUSION_FA:-0}" == "1" ]]; then GEN_ARGS+=(--diffusion-fa); fi
  if [[ "${SD_OFFLOAD_TO_CPU:-0}" == "1" ]]; then GEN_ARGS+=(--offload-to-cpu); fi
  if [[ "${SD_VAE_TILING:-0}" == "1" ]]; then GEN_ARGS+=(--vae-tiling); fi
  if [[ -n "${SD_THREADS:-}" && "${SD_THREADS}" != "-1" ]]; then GEN_ARGS+=(-t "$SD_THREADS"); fi
  if [[ -n "${SD_LORA_DIR:-}" ]]; then GEN_ARGS+=(--lora-model-dir "$SD_LORA_DIR"); fi
  if [[ -n "${SD_LOG_LEVEL:-}" ]]; then GEN_ARGS+=(--log-level "$SD_LOG_LEVEL"); fi
  return 0
}

check_binary() {
  local bin="$1" hint="$2"
  if [[ ! -x "$bin" ]] && ! command -v "$bin" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: 找不到 sd 二进制: $bin

请先安装(二选一):
  ./colab.sh install sd            # GitHub Release Linux 通用预编译二进制(CPU, 快速)
  ./colab.sh install sd --build    # 源码编译; GPU/CUDA 必须用此方式

若已安装但仍找不到:
  - 在 sd/ 目录下重新执行(或 cd .. && cd sd 触发 direnv 刷新 PATH)
  - 检查 SD_DIR / SD_SERVER / SD_CLI 环境变量是否被覆盖
  - ${hint}
EOF
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

  check_binary "$SD_SERVER" "手动验证: command -v sd-server"
  build_model_args
  build_gen_args

  mkdir -p "$LOG_DIR"

  local LAUNCH_CMD=("$SD_SERVER" "${MODEL_ARGS[@]}" "${GEN_ARGS[@]}" -l "$HOST" --listen-port "$PORT")

  echo "启动 sd-server... (日志: ${LOG_FILE})" | tee -a "$LOG_FILE"
  echo ">> 启动命令: ${LAUNCH_CMD[*]}" | tee -a "$LOG_FILE"
  {
    echo "===== $(date '+%F %T') [sd] start ====="
    printf '  %s\n' "${LAUNCH_CMD[@]}"
  } >>"$CMD_LOG"

  # setsid 脱离终端, 子 shell 写入真实 PID 后 exec 替换为 sd-server
  # shellcheck disable=SC2016
  setsid bash -c '
    echo $$ > "$1"; shift
    exec "$@"
  ' bash "$PID_FILE" "${LAUNCH_CMD[@]}" >>"$LOG_FILE" 2>&1 </dev/null &

  sleep 1
  if is_running; then
    echo "已提交启动 (PID $(cat "$PID_FILE"))。查看状态: $0 status"
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
    exit 0
  fi
  echo "停止 sd-server..."
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
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "http://localhost:${PORT}/v1/models" 2>/dev/null || true)
  code="${code:-000}"
  if [[ "$code" == "200" ]]; then
    echo "健康检查: HTTP 200 (就绪)"
    echo "Web UI: http://localhost:${PORT}/"
  else
    echo "健康检查: HTTP ${code} (启动中或异常, 查看 logs)"
  fi
}

# 调用服务生成一张测试图(验证 server 端到端可用)
do_test() {
  if ! is_running && ! pgrep_server; then
    echo "错误: sd-server 未在运行。请先执行: $0 start" >&2
    exit 1
  fi
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "http://localhost:${PORT}/v1/models" 2>/dev/null || true)
  code="${code:-000}"
  if [[ "$code" != "200" ]]; then
    echo "错误: 服务尚未就绪(HTTP ${code})。请稍后重试或查看 logs。" >&2
    exit 1
  fi

  mkdir -p "$SD_OUTPUT_DIR"
  local out="${SD_OUTPUT_DIR}/sd_test_$(date '+%Y%m%d_%H%M%S').png"
  echo ">> 请求出图 (prompt=${SD_PROMPT}, port=${PORT}, 超时 ${SD_REQUEST_TIMEOUT}s) ..."

  local body
  body="$(python3 -c 'import json,sys; print(json.dumps({"prompt": sys.argv[1], "n": 1, "output_format": "png"}))' "$SD_PROMPT")"
  curl -s --max-time "$SD_REQUEST_TIMEOUT" \
    -X POST "http://localhost:${PORT}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d "$body" \
    | python3 -c '
import base64, json, sys
out = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception as e:
    sys.exit("错误: 响应不是合法 JSON: %s" % e)
if "error" in d:
    sys.exit("错误: 服务返回 %s" % d["error"])
items = d.get("data") or []
if not items or "b64_json" not in items[0]:
    sys.exit("错误: 响应中没有图片数据")
with open(out, "wb") as f:
    f.write(base64.b64decode(items[0]["b64_json"]))
print("已保存测试图:", out)
' "$out" || { echo "错误: 出图失败, 请检查服务状态" >&2; exit 1; }
}

# 用 sd-cli 一次性出图(不依赖 server)
do_generate() {
  local prompt="${*:-${SD_PROMPT}}"
  check_binary "$SD_CLI" "手动验证: command -v sd-cli"
  build_model_args
  build_gen_args

  mkdir -p "$SD_OUTPUT_DIR"
  local stamp out
  stamp="$(date '+%Y%m%d_%H%M%S')"
  local batch="${SD_BATCH_COUNT:-1}"
  if [[ "$batch" =~ ^[0-9]+$ ]] && (( batch > 1 )); then
    out="${SD_OUTPUT_DIR}/sd_${stamp}_%02d.png"
  else
    out="${SD_OUTPUT_DIR}/sd_${stamp}.png"
  fi

  local -a CMD=("$SD_CLI" -M img_gen "${MODEL_ARGS[@]}" "${GEN_ARGS[@]}" -p "$prompt" -o "$out")
  if [[ "$batch" =~ ^[0-9]+$ ]] && (( batch > 1 )); then
    CMD+=(-b "$batch")
  fi

  echo ">> 一次性出图: prompt=${prompt}"
  echo ">> 输出: ${out}"
  echo ">> 命令: ${CMD[*]}"
  "${CMD[@]}"
  ls -lh "$SD_OUTPUT_DIR"/sd_"${stamp}"*.png 2>/dev/null || true
}

do_keep() {
  mkdir -p "$LOG_DIR"
  echo "进入守护模式 (PID $$), 日志: ${KEEP_LOG}"
  while true; do
    if ! pgrep_server; then
      echo "[$(date '+%F %T')] 服务进程消失, 自动重启..." >>"${KEEP_LOG}"
      "$0" start >>"${KEEP_LOG}" 2>&1 || true
    fi
    sleep 30
  done
}

usage() {
  grep -E "^# +(用法|\./)" "$0" | sed 's/^# \{1,\}//'
}

# ----------------------------- 入口 ------------------------------------------
case "${1:-}" in
start)    do_start ;;
stop)     do_stop ;;
restart)
  do_stop
  do_start
  ;;
status)   do_status ;;
test)     do_test ;;
generate)
  shift
  do_generate "$@"
  ;;
keep)     do_keep ;;
logs)     tail -n 100 -f "$LOG_FILE" ;;
help | -h | --help) usage ;;
*)
  usage
  exit 1
  ;;
esac
