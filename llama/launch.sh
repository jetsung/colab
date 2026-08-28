#!/usr/bin/env bash
# ============================================================
# llama.cpp 服务管理脚本 (Qwen3.8-Flash-Next GGUF)
#
# 用法:
#   ./launch.sh start     启动服务(后台, setsid 托管; 首次自动下载模型)
#   ./launch.sh stop      停止服务
#   ./launch.sh restart   重启服务
#   ./launch.sh status    查看状态 + 健康检查
#   ./launch.sh test      发送一条测试对话(需服务已就绪)
#   ./launch.sh logs      跟踪日志
#   ./launch.sh keep      守护模式(崩溃自动拉起)
#
# 依赖:
#   - install 已装好 llama 二进制(编译或官方预编译)
#   - 环境变量 HF_TOKEN 已设置（且已在 HF 接受模型许可证）
#
# 环境变量(可被外部/命令行覆盖):
#   LLAMA_MODEL_REPO  模型仓库(不可为空; 未设置时回退 MODEL_REPO)
#   LLAMA_MODEL_NAME  模型名前缀(未设置时从 REPO 提取: / 后部分去 -GGUF); 服务 --alias 为其小写形式
#   LLAMA_QUANT       量化档(不可为空, 无默认; 由 .envrc / gpu profile 提供)
#   LLAMA_MODEL_DIR   模型目录(默认 /content/models/<repo名>/<quant>)
#   LLAMA_DIR       安装目录(默认 /content/llama.cpp; 由 .envrc 导出, 可覆盖)
#   LLAMA_SERVER     llama 二进制路径(默认从 PATH 查找 command -v llama; 未命中回退 <LLAMA_DIR>/build/bin)
#   LLAMA_HOST / LLAMA_PORT   监听地址与端口(内部变量 HOST/PORT, 默认 0.0.0.0 / 30000)
#   LLAMA_API_KEY     服务器 API 密钥(未设置时回退 API_KEY)
#   LLAMA_XET         1=启用 HF Xet 存储(默认), 0=禁用
#   LLAMA_MMPROJ      视觉投影器 mmproj 路径(可选; 缺省自动检测模型目录 mmproj-*.gguf, 再按需自动下载)
#   LLAMA_MMPROJ_REPO mmproj 自动下载源(默认同 LLAMA_MODEL_REPO; 空=禁用自动下载)
# ============================================================

if [[ -n "${DEBUG:-}" ]]; then
    set -eux
else
    set -euo pipefail
fi

# ----------------------------- 可调配置 --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

# 不再内置加载 GPU profile; 所需环境变量(LLAMA_QUANT/LLAMA_MODEL_REPO/... 等)
# 由外部环境提供(如 .envrc 经 direnv 注入), 未提供时沿用下方默认值。
# LLAMA_* 不存在时回退到通用变量(API_KEY / MODEL_REPO)
# 硬约束: LLAMA_MODEL_REPO / LLAMA_QUANT / LLAMA_MODEL_NAME 不可为空(在 do_start 内校验)

readonly LLAMA_MODEL_REPO="${LLAMA_MODEL_REPO:-${MODEL_REPO:-}}"
# 仓库名 = LLAMA_MODEL_REPO 取 / 后部分(保留 -GGUF 后缀), 供目录与模型名前缀复用
#   例: unsloth/Qwen3.8-Flash-Next-GGUF -> Qwen3.8-Flash-Next-GGUF
readonly LLAMA_REPO_NAME="${LLAMA_MODEL_REPO##*/}"
# 模型名前缀(用于本地分片文件匹配)
# 未显式设置时, 从仓库名去除末尾 -GGUF 后缀
#   例: Qwen3.8-Flash-Next-GGUF -> Qwen3.8-Flash-Next
if [[ -z "${LLAMA_MODEL_NAME:-}" ]]; then
  LLAMA_MODEL_NAME="${LLAMA_REPO_NAME%-GGUF}"   # 去除末尾 -GGUF 后缀
fi
readonly LLAMA_MODEL_NAME
# 服务别名: 模型名转小写(LLAMA_MODEL_NAME 保留原样, 用于分片文件匹配)
# 例: Qwen3.8-Flash-Next -> qwen3.8-flash-next
readonly LLAMA_MODEL_ALIAS="${LLAMA_MODEL_NAME,,}"
# 无默认值: 必须由外部提供(.envrc / gpu profile / 命令行); 此处仅声明空以规避 set -u
readonly LLAMA_QUANT="${LLAMA_QUANT:-}"
# 模型目录: 默认用仓库名(保留 -GGUF 后缀)
#   例: Qwen3.8-Flash-Next-GGUF -> /content/models/Qwen3.8-Flash-Next-GGUF/$LLAMA_QUANT
readonly LLAMA_MODEL_DIR="${LLAMA_MODEL_DIR:-/content/models/$LLAMA_REPO_NAME/$LLAMA_QUANT}"
# 安装目录(默认 /content/llama.cpp; .envrc 已 export, 此处兜底)
readonly LLAMA_DIR="${LLAMA_DIR:-/content/llama.cpp}"
# 统一二进制解析优先级: 显式 LLAMA_SERVER > PATH 查找 command -v llama(新版 llama serve) > 默认 <LLAMA_DIR>/build/bin/llama
if [[ -n "${LLAMA_SERVER:-}" ]]; then
  :   # 保持显式指定值
elif command -v llama >/dev/null 2>&1; then
  LLAMA_SERVER="$(command -v llama)"
else
  LLAMA_SERVER="$LLAMA_DIR/build/bin/llama"
fi
readonly LLAMA_SERVER
# 监听地址与端口: 内部变量 HOST/PORT, 可用 LLAMA_HOST / LLAMA_PORT 环境变量覆盖
readonly PORT="${LLAMA_PORT:-30000}"
readonly HOST="${LLAMA_HOST:-0.0.0.0}"
readonly LLAMA_NGL="${LLAMA_NGL:-999}"
readonly LLAMA_CTX="${LLAMA_CTX:-0}"
readonly LLAMA_API_KEY="${LLAMA_API_KEY:-${API_KEY:-}}"
readonly LLAMA_XET="${LLAMA_XET:-1}"   # 1=启用 HuggingFace Xet 存储(默认), 0=禁用
# 多模态视觉投影器(mmproj): 用于图片/视频输入(可选, 缺省不启用)
#   LLAMA_MMPROJ       显式指定 mmproj 文件路径(优先级最高; 不设置时自动在模型目录检测 mmproj-*.gguf)
#   LLAMA_MMPROJ_REPO  自动下载源(默认同 LLAMA_MODEL_REPO; 设为空字符串则禁用自动下载)
readonly LLAMA_MMPROJ="${LLAMA_MMPROJ:-}"
readonly LLAMA_MMPROJ_REPO="${LLAMA_MMPROJ_REPO:-${LLAMA_MODEL_REPO:-}}"

# 服务托管(与 sglang/launch.sh 一致): setsid 后台 + PID/日志文件
# 日志统一写到项目根目录的 logs/(无论从根目录还是 llama/ 下执行, 均落同一处)
ROOT_DIR="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)"   # 项目根目录 = llama/ 的父目录
readonly ROOT_DIR
readonly LOG_DIR="${ROOT_DIR}/logs"
readonly LOG_FILE="${LOG_DIR}/llama_server.log"
readonly CMD_LOG="${LOG_DIR}/launch_cmd.log"    # 启动命令日志(追加, 带时间戳)
readonly PID_FILE="${SCRIPT_DIR}/llama.pid"
readonly KEEP_LOG="${LOG_DIR}/keeper.log"

# pgrep 匹配模式(括号防自匹配): 新版统一命令 "llama serve"
# 前置边界 (^|[^a-z]) 排除 ollama serve(其 "llama" 前是字母 o)
readonly PROC_PATTERN="(^|[^a-z])llama serv[e]"

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

# 校验: llama 二进制与 HF_TOKEN
check_deps() {
  if ! command -v "$LLAMA_SERVER" >/dev/null 2>&1; then
    cat >&2 <<EOF
ERROR: 找不到 llama 二进制: $LLAMA_SERVER

请先安装(二选一):
  ./colab.sh install llama            # 官方预编译二进制(快速, 免编译)
  ./colab.sh install llama --build    # 源码编译(PR #27742, 支持 Qwen3.8-Flash-Next)

若已安装但仍找不到:
  - 在 llama/ 目录下重新执行(或 cd .. && cd llama 触发 direnv 刷新 PATH)
  - 检查 LLAMA_DIR / LLAMA_SERVER 环境变量是否被覆盖
  - 手动验证: command -v llama
EOF
    exit 1
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: 未设置 HF_TOKEN。请先 export HF_TOKEN=hf_xxx（并在 HF 接受模型许可证）。" >&2
    exit 1
  fi
}

# 定位本地模型分片文件(不足/缺失则下载)
resolve_model_file() {
  mkdir -p "$LLAMA_MODEL_DIR"

  # ---- 下载（按总分片数判断是否完整）----
  local first total existing
  # 分片 glob 模式(传给 find 展开, 避免赋值字面化)
  first=$(find "$LLAMA_MODEL_DIR" -maxdepth 1 -name "$LLAMA_MODEL_NAME-$LLAMA_QUANT-0000*-of-0000*.gguf" -printf '%f\n' 2>/dev/null | sort | head -1 || true)
  total=""
  if [[ -n "$first" ]]; then
    total=$(echo "$first" | sed -nE 's/.*of-0000([0-9]+)\.gguf/\1/p')
  fi
  existing=$(find "$LLAMA_MODEL_DIR" -maxdepth 1 -name "$LLAMA_MODEL_NAME-$LLAMA_QUANT-0000*-of-0000*.gguf" -printf '%f\n' 2>/dev/null | wc -l || true)
  if [[ -z "$first" || "${existing:-0}" -lt "${total:-3}" ]]; then
    echo ">> 本地模型不足或不完整（现有 ${existing:-0}/${total:-?} 分片），开始下载 $LLAMA_MODEL_REPO ($LLAMA_QUANT) ..." >&2
    echo "   (该仓库使用 Xet 存储，需要 hf_xet；install.sh 已安装)" >&2
    if [[ "$LLAMA_XET" == "1" ]]; then
      HF_HUB_ENABLE_XET=1 HF_TOKEN="$HF_TOKEN" \
        hf download "$LLAMA_MODEL_REPO" --include "$LLAMA_QUANT/*" --local-dir "$LLAMA_MODEL_DIR"
    else
      HF_TOKEN="$HF_TOKEN" \
        hf download "$LLAMA_MODEL_REPO" --include "$LLAMA_QUANT/*" --local-dir "$LLAMA_MODEL_DIR"
    fi
  else
    echo ">> 模型已存在 ($existing/$total 分片)，跳过下载。" >&2
  fi

  local MODEL_FILE
  MODEL_FILE=$(find "$LLAMA_MODEL_DIR" -maxdepth 1 -name "$LLAMA_MODEL_NAME-$LLAMA_QUANT-00001-of-0000*.gguf" -print -quit 2>/dev/null || true)
  if [[ -z "$MODEL_FILE" || ! -f "$MODEL_FILE" ]]; then
    echo "ERROR: 未找到模型分片文件 ($LLAMA_MODEL_DIR/$LLAMA_MODEL_NAME-$LLAMA_QUANT-*.gguf)。下载可能失败, 请查看日志。" >&2
    exit 1
  fi
  printf '%s' "$MODEL_FILE"
}

# 判断主模型 GGUF 是否声明多模态支持(存在图像 token / 视觉相关元数据键)。
# 返回 0=支持视觉, 1=不支持。
# llama-gguf 不可用时无法检测, 按"支持"处理(保持向后兼容, 不因检测而误伤)。
model_supports_vision() {
  local mf="$1"
  local gguf_tool=""
  if [[ -n "$LLAMA_SERVER" ]]; then
    gguf_tool="${LLAMA_SERVER%/*}/llama-gguf"
    [[ -x "$gguf_tool" ]] || gguf_tool="$(command -v llama-gguf 2>/dev/null || true)"
  else
    gguf_tool="$(command -v llama-gguf 2>/dev/null || true)"
  fi
  if [[ -z "$gguf_tool" || ! -x "$gguf_tool" ]]; then
    echo "警告: 未找到 llama-gguf, 跳过模型视觉能力检测(按支持视觉处理)。" >&2
    return 0
  fi
  # 注意: 不用 grep -q(匹配即关闭管道, 会让 llama-gguf 收到 SIGPIPE,
  # pipefail 下管道退出码变 141, 导致误判"不支持视觉"); 用 grep -c 读完全部输出
  if "$gguf_tool" "$mf" r n 2>/dev/null | grep -ciE 'image_token_id|has_vision|\.vision\.' >/dev/null; then
    return 0
  fi
  return 1
}

# 定位多模态视觉投影器(mmproj)。优先级: 显式 LLAMA_MMPROJ > 模型目录自动检测 > 自动下载。
# 仅在主模型声明支持视觉时启用(见 model_supports_vision); 否则直接跳过, 避免把不相干的
# mmproj 传给纯文本模型导致启动/请求失败。
# 未找到时输出空字符串(不报错): 文本服务照常可用, 仅图片/视频输入不可用。
# 仅当显式指定的 LLAMA_MMPROJ 路径不存在时返回非零(启动失败)。
resolve_mmproj_file() {
  local model_file="$1"

  # 0) 主模型不支持视觉 -> 直接禁用 mmproj
  if ! model_supports_vision "$model_file"; then
    echo ">> 主模型不支持视觉(无 image_token/视觉元数据), 跳过 mmproj。" >&2
    return 0
  fi

  # 1) 显式指定
  if [[ -n "$LLAMA_MMPROJ" ]]; then
    if [[ ! -f "$LLAMA_MMPROJ" ]]; then
      echo "ERROR: 指定的 LLAMA_MMPROJ 不存在: $LLAMA_MMPROJ" >&2
      return 1
    fi
    printf '%s' "$LLAMA_MMPROJ"
    return 0
  fi

  local mm=""
  # 2) 模型目录自动检测
  mm=$(find "$LLAMA_MODEL_DIR" -maxdepth 1 -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)
  if [[ -n "$mm" ]]; then
    printf '%s' "$mm"
    return 0
  fi

  # 3) 自动下载(未显式禁用; 失败仅告警, 不中断启动)
  if [[ -n "$LLAMA_MMPROJ_REPO" ]]; then
    echo ">> 未找到本地 mmproj, 开始下载 $LLAMA_MMPROJ_REPO (mmproj-*.gguf) ..." >&2
    if [[ "$LLAMA_XET" == "1" ]]; then
      HF_HUB_ENABLE_XET=1 HF_TOKEN="$HF_TOKEN" \
        hf download "$LLAMA_MMPROJ_REPO" --include 'mmproj-*.gguf' --local-dir "$LLAMA_MODEL_DIR" >/dev/null 2>&1 || true
    else
      HF_TOKEN="$HF_TOKEN" \
        hf download "$LLAMA_MMPROJ_REPO" --include 'mmproj-*.gguf' --local-dir "$LLAMA_MODEL_DIR" >/dev/null 2>&1 || true
    fi
    mm=$(find "$LLAMA_MODEL_DIR" -maxdepth 1 -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)
    if [[ -n "$mm" ]]; then
      printf '%s' "$mm"
      return 0
    fi
    echo "警告: mmproj 下载失败或未找到, 图片输入不可用(文本功能不受影响)。" >&2
  fi
  return 0
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

  # 硬约束: 核心变量不可为空(仅 start 需要, 故在启动时校验)
  if [[ -z "$LLAMA_MODEL_REPO" ]]; then
    echo "ERROR: LLAMA_MODEL_REPO 不可为空(且 MODEL_REPO 亦未提供)。请设置模型仓库(如 unsloth/Qwen3.8-Flash-Next-GGUF)。" >&2
    exit 1
  fi
  if [[ -z "$LLAMA_MODEL_NAME" ]]; then
    echo "ERROR: LLAMA_MODEL_NAME 不可为空(显式设置或从 LLAMA_MODEL_REPO 提取均无效)。" >&2
    exit 1
  fi
  if [[ -z "$LLAMA_QUANT" ]]; then
    echo "ERROR: LLAMA_QUANT 不可为空。请显式设置(如 LLAMA_QUANT=UD-Q4_K_XL, 或由 gpu profile/.envrc 提供)。" >&2
    exit 1
  fi

  check_deps
  local MODEL_FILE
  MODEL_FILE="$(resolve_model_file)"
  # 视觉投影器(可选): 解析后若非空则服务支持图片/视频输入
  local MMPROJ_FILE
  MMPROJ_FILE="$(resolve_mmproj_file "$MODEL_FILE")"

  mkdir -p "$LOG_DIR"

  # 可选：给服务器 API 加上访问密钥（客户端需通过 Authorization: Bearer $LLAMA_API_KEY 调用）
  local SERVER_ARGS=()
  if [[ -n "$LLAMA_API_KEY" ]]; then
    SERVER_ARGS+=(--api-key "$LLAMA_API_KEY")
  fi

  # 展开后的启动命令(回显 + 写入日志, 便于排查)
  # 官方统一命令: llama serve(参数与旧 llama-server 一致)
  # --alias: 模型别名(默认 = LLAMA_MODEL_ALIAS, 即模型名的小写形式),
  #          避免 API 中显示为实际文件路径
  local   LAUNCH_CMD=("$LLAMA_SERVER" serve
    -m "$MODEL_FILE"
    --alias "$LLAMA_MODEL_ALIAS"
    -ngl "$LLAMA_NGL"
    --host "$HOST"
    --port "$PORT"
    --ctx-size "$LLAMA_CTX")
  if [[ -n "$MMPROJ_FILE" ]]; then
    LAUNCH_CMD+=(--mmproj "$MMPROJ_FILE")
  fi
  LAUNCH_CMD+=("${SERVER_ARGS[@]}")

  echo "启动 llama 服务... (日志: ${LOG_FILE})" | tee -a "$LOG_FILE"
  echo ">> 加载模型: $MODEL_FILE" | tee -a "$LOG_FILE"
  if [[ -n "$MMPROJ_FILE" ]]; then
    echo ">> 视觉投影器(mmproj): $MMPROJ_FILE (图片输入已启用)" | tee -a "$LOG_FILE"
  else
    echo ">> 未加载 mmproj: 图片/视频输入不可用(文本功能正常)" | tee -a "$LOG_FILE"
  fi
  echo ">> 启动命令: ${LAUNCH_CMD[*]}" | tee -a "$LOG_FILE"

  # 启动命令单独追加写入命令日志(带时间戳), 便于事后查看实际启动参数
  {
    echo "===== $(date '+%F %T') [llama] start ====="
    printf '  %s\n' "${LAUNCH_CMD[@]}"
  } >>"$CMD_LOG"

  # setsid 脱离终端, 子 shell 写入真实 PID 后 exec 替换为 llama 服务进程
  # 命令与参数以 "$@" 透传, 规避手工转义
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
  echo "停止 llama 服务..."
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
    "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
  if [[ "$code" == "200" ]]; then
    echo "健康检查: HTTP 200 (就绪)"
  else
    echo "健康检查: HTTP ${code} (启动中或异常, 查看 logs)"
  fi
}

# 发送一条测试 chat 对话到平台(先确认服务已启动并就绪)
do_test() {
  if ! is_running && ! pgrep_server; then
    echo "错误: llama 服务未在运行。请先执行: $0 start" >&2
    exit 1
  fi
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
    "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
  if [[ "$code" != "200" ]]; then
    echo "错误: 服务尚未就绪(health HTTP ${code})。请稍后重试或查看 logs。" >&2
    exit 1
  fi

  local AUTH=()
  if [[ -n "$LLAMA_API_KEY" ]]; then
    AUTH=(-H "Authorization: Bearer $LLAMA_API_KEY")
  fi

  echo ">> 发送测试对话 (model=${LLAMA_MODEL_ALIAS}, port=${PORT}) ..."
  curl -s --max-time 30 "http://localhost:${PORT}/v1/chat/completions" \
    "${AUTH[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${LLAMA_MODEL_ALIAS}\", \"messages\": [{\"role\": \"user\", \"content\": \"你好, 请用一句话回复\"}], \"max_tokens\": 64}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print("回复:", d["choices"][0]["message"]["content"])' 2>/dev/null \
    || { echo "错误: 请求失败, 请检查服务状态" >&2; exit 1; }
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
