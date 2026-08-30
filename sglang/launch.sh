#!/usr/bin/env bash
# shellcheck disable=SC1090   # 动态 source GPU profile 文件, 无法静态解析(设计使然)
# =============================================================================
# SGLang 服务管理脚本 (通用模型 @ RTX PRO 6000 Blackwell 96GB)
#
# 启动参数根据缓存/本地模型目录的 config.json 自动推导(模型家族、上下文长度、
# 是否为 mamba 模型、是否支持 EAGLE 投机解码等), 不再写死某一模型。
# 可用环境变量覆盖任意自动推导结果(见下方可调配置)。
#
# 用法:
#   ./launch.sh start     启动服务(后台)
#   ./launch.sh stop      停止服务
#   ./launch.sh restart   重启服务
#   ./launch.sh status    查看状态
#   ./launch.sh test      发送一条测试对话(需服务已就绪)
#   ./launch.sh logs      跟踪日志
#   ./launch.sh keep      守护模式(崩溃自动拉起)
#
# 环境变量(可被外部/命令行覆盖):
#   SGLANG_MODEL_REPO    模型(HF ID 或本地路径); 未设置时回退 MODEL_REPO
#   SGLANG_MODEL_ROOT    模型基础盘前缀(未设置回退 MODEL_ROOT, 再兜底 /content/models)
#                        不支持 Google Drive(/content/drive): 指向 Drive 的路径会在启动时报错
#   SGLANG_MODEL_DIR     本仓库模型目录(默认 <ROOT>/<repo名>; 显式设置时原样使用)
#                        HF ID 启动时权重会下载到该目录(持久化), 命中本地目录则离线直接启动
#   SGLANG_SERVED_NAME   API 模型别名(默认取模型路径末段小写)
#   SGLANG_HOST / SGLANG_PORT   监听地址与端口(默认 0.0.0.0 / 30000)
#   SGLANG_DEVICE       设备(默认空=由 sglang 自动探测; .env.cpu 显式设为 cpu; 置空=交回自动探测)
#   SGLANG_CTX    最大上下文(默认0=由 config.json 推导)
#   SGLANG_SPECULATIVE_ALGORITHM    投机解码算法(默认按 MTP 层自动判断; 置空=关闭)
#   SGLANG_SPECULATIVE_NUM_STEPS / SGLANG_SPECULATIVE_EAGLE_TOPK / SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS
#   SGLANG_MAX_RUNNING_REQUESTS    投机解码时最大并发请求数(默认 48)
#   SGLANG_MAMBA_FULL_MEMORY_RATIO mamba 状态缓存占比(默认 0.6); mamba 模型实际并发
#                                  ≈ max_mamba_cache_size/4, 调大提升并发、缩小 KV 池
#   SGLANG_MEM_FRACTION_STATIC     静态显存占比(默认 0.90; 由 .env.g4/.env.t4 提供)
#   SGLANG_CUDA_GRAPH_BACKEND_PREFILL  prefill CUDA graph 后端(默认空=由 sglang 自选;
#                                Turing(T4, sm75) 上 flashinfer ragged prefill 捕获会崩,
#                                见 .env.t4 中设为 disabled)
#   SGLANG_CUDA_GRAPH_MAX_BS_PREFILL   prefill CUDA graph 最大 batch token(默认空=不限制)
#   FLASHINFER_CUDA_ARCH_LIST   FlashInfer JIT 目标架构(默认按 nvidia-smi + nvcc 版本推导,
#                               如 Blackwell: nvcc>=12.9 -> 12.0f, 否则 12.0a)
#   CUDA_HOME / CUDA_PATH       JIT 编译用的 CUDA 工具链(影响架构取值; 默认用系统 nvcc)
#   SGLANG_REASONING_PARSER / SGLANG_TOOL_CALL_PARSER / SGLANG_CHAT_TEMPLATE_KWARGS
#   SGLANG_API_KEY       鉴权密钥(未设置时回退 API_KEY; 两者均空则报错, 显式置空关闭鉴权)
#   SGLANG_VENV_DIR      Python 虚拟环境目录(默认 /tmp/sglang/venv, 见下)
# =============================================================================

if [[ -n "${DEBUG:-}" ]]; then
    set -eux
else
    set -euo pipefail
fi

# ----------------------------- 可调配置 --------------------------------------
# 工作目录 = 本脚本所在目录, 使脚本可随目录整体搬移而无需改路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

readonly WORKDIR="$SCRIPT_DIR"                                 # 工作目录(脚本/配置所在)
# venv 默认建在项目外: sglang 依赖数 GB(torch + CUDA 包), 放项目里会让整个目录难以复制/备份。
# 需要换位置(如放到持久化盘)时用 SGLANG_VENV_DIR 覆盖; 该默认值与 colab.sh install sglang 一致。
readonly VENV_DIR="${SGLANG_VENV_DIR:-/tmp/sglang/venv}"
# 日志统一写到项目根目录的 logs/(无论从根目录还是 sglang/ 下执行, 均落同一处)
ROOT_DIR="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)"             # 项目根目录 = sglang/ 的父目录
readonly ROOT_DIR
readonly LOG_DIR="${ROOT_DIR}/logs"
readonly LOG_FILE="${LOG_DIR}/sglang_server.log"
readonly CMD_LOG="${LOG_DIR}/launch_cmd.log"    # 启动命令日志(追加, 带时间戳)
readonly PID_FILE="${WORKDIR}/sglang.pid"
readonly KEEP_LOG="${LOG_DIR}/keeper.log"

# ------------------------- 平台探测(CPU / GPU) ------------------------------
# 是否存在可用的 NVIDIA GPU
has_nvidia_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  local gpu=""
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  [[ -n "$gpu" ]]
}

# 纯 CPU 平台: 显式 GPU_PROFILE=cpu, 或未设置 GPU_PROFILE 且探测不到 NVIDIA GPU
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

# profile 兜底加载(不依赖 direnv), 与 llama/launch.sh 同逻辑:
# 环境变量继承调用方 shell; 未 cd 进本目录(引擎 .envrc 未加载)时, 按 GPU_PROFILE
# (未设置则按 nvidia-smi 探测, 无显卡则按 cpu)source 本目录 .env.<g4|t4|cpu>,
# 补上 SGLANG_MEM_FRACTION_STATIC 等引擎专属默认值。
# profile 为纯 bash(${VAR:-默认}), 幂等且不覆盖已有值。
load_gpu_profile() {
  local profile="${GPU_PROFILE:-}"
  if [[ -z "$profile" ]]; then
    local gpu=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    fi
    case "$gpu" in
      *T4*)                  profile=t4 ;;
      *G4*|*L4*|*Blackwell*) profile=g4 ;;   # Blackwell 系(如 RTX PRO 6000)按 G4 处理
      "")                    profile=cpu ;;  # 无 nvidia-smi 或无输出: 纯 CPU 平台
      # 有显卡但型号未识别: 不套用任何 profile(避免把 A100 之类误判成 CPU 而静默降速)
    esac
  fi
  [[ -n "$profile" ]] || return 0
  local f="${SCRIPT_DIR}/.env.${profile}"
  [[ -r "$f" ]] || return 0
  echo ">> [sglang] 加载 profile: ${profile} (${f})" >&2
  # shellcheck disable=SC1090
  source "$f"
}
# 仅需要模型参数的子命令才兜底, 避免 status/stop/logs 也去探测显卡
case "${1:-}" in
  start | restart) load_gpu_profile ;;
esac

# 模型(HF ID 或本地路径): 优先 SGLANG_MODEL_REPO, 回退通用 MODEL_REPO
# 硬约束: 两者均空时 do_start 报错(参照 llama/launch.sh)
readonly SGLANG_MODEL_REPO="${SGLANG_MODEL_REPO:-${MODEL_REPO:-}}"
readonly SGLANG_MODEL_REPO_NAME="${SGLANG_MODEL_REPO##*/}"
# 模型路径(两级: 基础盘前缀 + 本仓库目录), 与 llama/launch.sh 的 LLAMA_MODEL_ROOT/DIR 对称
#   SGLANG_MODEL_ROOT  基础盘前缀(三级优先级, 见下方解析; 与 llama/launch.sh 对称)
#                      换持久化盘只改这一层(或改根 .envrc 的 MODEL_ROOT, 两引擎同时生效)
#   SGLANG_MODEL_DIR   本仓库模型目录(默认 <ROOT>/<repo名>, 按仓库隔离; 显式设置时原样使用)
#                      HF ID 启动时权重下载到该目录(见 resolve_model_path), 跨会话可复用;
#                      已给本地路径时不做拼接也不下载。
# 基础盘前缀解析(显式逐级判定, 便于看清优先级; 字面量仅作最后兜底):
#   1) SGLANG_MODEL_ROOT  引擎专属, 优先级最高
#   2) MODEL_ROOT         两引擎共用(根 .envrc 导出), 换持久化盘改这一处两引擎同时生效
#   3) /content/models    兜底默认值, 仅在上述两者均未设置(或为空)时使用
# "已设置" = 变量存在且非空(空串等同未设置, 继续回退); 来源记入 MODEL_ROOT_SOURCE 便于排查
MODEL_ROOT_SOURCE=""
if [[ -n "${SGLANG_MODEL_ROOT:-}" ]]; then
  MODEL_ROOT_SOURCE="SGLANG_MODEL_ROOT"
elif [[ -n "${MODEL_ROOT:-}" ]]; then
  SGLANG_MODEL_ROOT="$MODEL_ROOT"
  MODEL_ROOT_SOURCE="MODEL_ROOT"
else
  SGLANG_MODEL_ROOT="/content/models"
  MODEL_ROOT_SOURCE="默认值(兜底)"
fi
readonly SGLANG_MODEL_ROOT MODEL_ROOT_SOURCE
# 不支持 Google Drive 作为模型目录: Drive 是 FUSE 挂载, mmap 随机读极慢, 且不提供任何
# 冷存储/复制降级 —— 指向 /content/drive 的路径在启动时直接报错(见 do_start 内校验)。
SGLANG_MODEL_DIR="${SGLANG_MODEL_DIR:-${SGLANG_MODEL_REPO_NAME:+$SGLANG_MODEL_ROOT/$SGLANG_MODEL_REPO_NAME}}"
readonly SGLANG_MODEL_DIR
# 实际 --model-path: 初始为 SGLANG_MODEL_REPO, do_start 内由 resolve_model_path 改写为本地目录
MODEL_PATH="$SGLANG_MODEL_REPO"
# API 中使用的模型别名: 默认取模型路径末段(小写, 斜杠转连字符); 可用 SGLANG_SERVED_NAME 覆盖
SERVED_NAME_DEFAULT="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]' | tr '/' '-')"
readonly SERVED_NAME_DEFAULT
readonly SERVED_NAME="${SGLANG_SERVED_NAME:-$SERVED_NAME_DEFAULT}"
# 监听地址与端口: 可用 SGLANG_HOST / SGLANG_PORT 环境变量覆盖
readonly HOST="${SGLANG_HOST:-0.0.0.0}"
readonly PORT="${SGLANG_PORT:-30000}"
# API 鉴权密钥: 优先 SGLANG_API_KEY, 回退通用 API_KEY(由 .envrc/.env 加载), 脚本不内置任何密钥
# 未设置时 start 报错退出; 显式置空(SGLANG_API_KEY="")则关闭鉴权
readonly SGLANG_API_KEY="${SGLANG_API_KEY:-${API_KEY:-}}"
# 最大上下文: 默认0=由模型 config.json 的 max_position_embeddings 推导, 可用 SGLANG_CTX 覆盖
readonly CONTEXT_LENGTH="${SGLANG_CTX:-0}"
# 投机解码: 启动时按模型 config.json 是否内置 MTP 层自动判断 EAGLE 支持(见 detect_speculative_algorithm)
# 可用环境变量 SGLANG_SPECULATIVE_ALGORITHM 显式覆盖(置空=关闭)
readonly SPECULATIVE_NUM_STEPS="${SGLANG_SPECULATIVE_NUM_STEPS:-3}"
readonly SPECULATIVE_EAGLE_TOPK="${SGLANG_SPECULATIVE_EAGLE_TOPK:-1}"
readonly SPECULATIVE_NUM_DRAFT_TOKENS="${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
# 投机解码时的最大并发运行请求数: sglang 启用投机解码会自动重置为 48 并打印提示,
# 此处默认显式传入 48 使行为确定且消除启动提示; 可用 SGLANG_MAX_RUNNING_REQUESTS 覆盖
readonly MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-48}"
# 静态显存占比(KV池等): 优先 SGLANG_MEM_FRACTION_STATIC, 可由 .env.g4/.env.t4 提供
# (CPU 平台不使用该参数, 见 do_start)
readonly MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.90}"
# prefill CUDA graph 后端 / 最大 batch: 非空才注入, 由 .env.t4 等 profile 提供。
# 背景: Turing(sm75) 上 flashinfer 的 ragged prefill wrapper 在捕获图时报
#   "q.shape[0] (128) does not match qo_indptr[-1] (2048)" 导致 scheduler 退出;
#   按 sglang 自身建议传 --cuda-graph-backend-prefill=disabled 绕过。
readonly CUDA_GRAPH_BACKEND_PREFILL="${SGLANG_CUDA_GRAPH_BACKEND_PREFILL:-}"
readonly CUDA_GRAPH_MAX_BS_PREFILL="${SGLANG_CUDA_GRAPH_MAX_BS_PREFILL:-}"
# mamba 状态缓存占比(相对 KV 池): 直接决定 mamba 模型的并发上限 ——
# max_running_requests ≈ max_mamba_cache_size / 4(投机解码下每请求 4 个 state slot)。
# 0.2 时仅 ~29 slot(并发 7); 调大可线性提升并发, 代价是 KV 池变小(长上下文容量下降)。
readonly MAMBA_FULL_MEMORY_RATIO="${SGLANG_MAMBA_FULL_MEMORY_RATIO:-0.6}"
readonly CHUNKED_PREFILL_SIZE=2048                            # prefill 分块大小
# 以下参数默认由模型家族自动推导, 可用同名环境变量显式覆盖(见头部"环境变量"段)
readonly REASONING_PARSER="${SGLANG_REASONING_PARSER:-}"      # 推理解析器(如 qwen3 / deepseek_v3)
readonly TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-}"      # 工具调用解析器(如 qwen3_coder)
readonly CHAT_TEMPLATE_KWARGS="${SGLANG_CHAT_TEMPLATE_KWARGS:-}"  # chat template 参数(JSON, 如 {"enable_thinking":true})

# ------------------------ FlashInfer / CUDA 架构 ------------------------------
# 坑: FlashInfer 读的是 FLASHINFER_CUDA_ARCH_LIST(无 SGLANG_ 前缀), 且其
# CompilationContext 在 import 时即构造, 只能从进程环境继承 —— 必须由本脚本在
# 启动前 export。
# 自动探测时, SM 12.x(Blackwell)要求 CUDA >= 12.9 才能归一化出 12.0f; nvcc 更旧
# (如系统 CUDA 12.8)则抛 "SM 12.x requires CUDA >= 12.9" -> 架构集合为空 -> JIT
# 报 "FlashInfer requires GPUs with sm75 or higher"。
# 带后缀的值会被 FlashInfer 原样采用, 从而跳过该版本检查, 故这里按 nvcc 版本选:
#   SM 12.0 + nvcc >= 12.9 -> 12.0f; nvcc < 12.9 -> 12.0a(sm_120a, 与 flashinfer
#   自身对 CUDA<12.9 的回退一致, 见 jit/nvfp4_attention_sm120.py)。
# 注意: 不要为凑 12.0f 改指 CUDA_HOME 到 venv 的 cuda 包 —— 其头文件(CUDART 13.0)
# 与自带 nvcc(13.4)版本不一致, cccl 会直接 #error 拒绝编译。

# 输出 nvcc 版本(major.minor); 优先 CUDA_HOME/CUDA_PATH, 否则用 PATH 中的 nvcc
detect_nvcc_version() {
  local nvcc
  nvcc="${CUDA_HOME:-${CUDA_PATH:-}}/bin/nvcc"
  [[ -x "$nvcc" ]] || nvcc="$(command -v nvcc 2>/dev/null || true)"
  [[ -n "$nvcc" && -x "$nvcc" ]] || return
  "$nvcc" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | head -1 | awk '{print $2}'
}

nvcc_at_least() {  # $1=最低版本; 返回 0 表示满足
  local v
  v="$(detect_nvcc_version)"
  [[ -n "$v" ]] || return 1
  awk -v v="$v" -v req="$1" 'BEGIN{exit !(v >= req)}'
}

# 推导架构列表: 后缀规则与 flashinfer/compilation_context.py 一致
# SM 12.0 -> 12.0f/12.0a; 其它 SM 12.x -> 12.xa; SM 9/10/11 -> xa; SM <= 8 -> 无后缀
# 纯 CPU 平台不参与: FlashInfer 是 CUDA 专用, 且下方 --attention-backend 已跳过它
if ! is_cpu_platform; then
  if [[ -z "${FLASHINFER_CUDA_ARCH_LIST:-}" ]]; then
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
    case "$CC" in
      12.0)                                                   # Blackwell (RTX PRO 6000 / RTX 50)
        if nvcc_at_least 12.9; then
          FLASHINFER_CUDA_ARCH_LIST="12.0f"
        else
          FLASHINFER_CUDA_ARCH_LIST="12.0a"
        fi
        ;;
      12.*)           FLASHINFER_CUDA_ARCH_LIST="${CC}a" ;;    # 其它 SM 12.x
      9*|10*|11*)     FLASHINFER_CUDA_ARCH_LIST="${CC}a" ;;    # Hopper 及以后
      *.*)            FLASHINFER_CUDA_ARCH_LIST="${CC}" ;;     # Ada(8.9) / Turing(7.5) 等无后缀
      *)              FLASHINFER_CUDA_ARCH_LIST="" ;;          # 探测失败: 交给 FlashInfer 自行探测
    esac
    unset CC
  fi
fi
export FLASHINFER_CUDA_ARCH_LIST

# pgrep 匹配模式(括号防自匹配): 新入口 `sglang serve`, 兼容旧入口 `python -m sglang.launch_server`
readonly PROC_PATTERN="sglang [s]erve|launch_[s]erver"

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

require_venv() {
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "未找到虚拟环境: ${VENV_DIR}" >&2
    echo "请先在项目根目录运行 './colab.sh install sglang' 初始化环境" >&2
    echo "或设置 SGLANG_VENV_DIR 指向已有的 venv" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

# 定位模型 config.json, 输出路径到 stdout; 定位顺序: 本地路径 > HF 缓存 > HF API(远程)
# 远程拉取时写出临时文件, 调用方需负责清理(本函数不负责删除临时文件)
locate_model_config() {
  local config="" tmp=""
  # 1) 本地模型目录
  if [[ -f "${MODEL_PATH}/config.json" ]]; then
    printf '%s' "${MODEL_PATH}/config.json"
    return
  fi
  # 2) HF 缓存(models--org--name/snapshots/*/config.json)
  local snapshots="${HF_HOME:-${HOME}/.cache/huggingface}/hub/models--${MODEL_PATH//\//--}/snapshots"
  local snap
  snap="$(find "$snapshots" -maxdepth 2 -name config.json -print -quit 2>/dev/null || true)"
  if [[ -n "$snap" ]]; then
    printf '%s' "$snap"
    return
  fi
  # 3) HF API 远程拉取(需网络)
  if command -v curl >/dev/null 2>&1; then
    tmp="$(mktemp 2>/dev/null || true)"
    if [[ -n "$tmp" ]] && curl -fsSL --max-time 10 "https://huggingface.co/${MODEL_PATH}/raw/main/config.json" -o "$tmp" 2>/dev/null; then
      printf '%s' "$tmp"
      return
    fi
    [[ -n "$tmp" ]] && rm -f "$tmp"
  fi
  printf ''
}

# 读取 config.json 中某个字段(支持点号路径), 输出到 stdout; 不存在/解析失败输出空
read_config_value() {
  local config="$1" key="$2"
  [[ -z "$config" ]] && return
  python3 - "$config" "$key" <<'PY' 2>/dev/null || true
import sys, json
p, key = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception:
    sys.exit(0)
val = d
for part in key.split('.'):
    if isinstance(val, dict) and part in val:
        val = val[part]
    else:
        sys.exit(0)
if isinstance(val, (dict, list)):
    print(json.dumps(val, ensure_ascii=False))
else:
    print(val)
PY
}

# 根据 config.json 推导模型家族(qwen / deepseek / llama / glm / mamba / generic)
detect_model_family() {
  local config="$1"
  local arch mtype text
  arch="$(read_config_value "$config" architectures)"
  mtype="$(read_config_value "$config" model_type)"
  text="${arch} ${mtype}"
  if grep -qiE 'qwen' <<<"$text"; then
    printf 'qwen'
  elif grep -qiE 'deepseek' <<<"$text"; then
    printf 'deepseek'
  elif grep -qiE 'glm' <<<"$text"; then
    printf 'glm'
  elif grep -qiE 'llama' <<<"$text"; then
    printf 'llama'
  elif grep -qiE 'mamba|jamba|ssm' <<<"$text"; then
    printf 'mamba'
  elif grep -qiE 'muse' <<<"$text"; then
    printf 'muse'
  else
    printf 'generic'
  fi
}

# 判断是否为 mamba/SSM 类模型(需要 mamba 专属参数)
is_mamba_model() {
  local config="$1"
  local text
  text="$(read_config_value "$config" architectures) $(read_config_value "$config" model_type)"
  # 额外识别 config 中出现的 mamba/ssm 相关键(用前导引号锚定 JSON 键, 不要求结尾引号,
  # 以兼容 mamba_ssm_dtype 这类键名)
  if [[ -n "$config" ]] && grep -qiE '"(mamba|mamba2|hybrid_mamba|ssm|mamba_ssm|mamba_d_intermediate)' "$config"; then
    return 0
  fi
  grep -qiE 'mamba|jamba' <<<"$text"
}

# 推导默认上下文长度: 依次尝试 顶层 / text_config 下的 max_position_embeddings、
# max_sequence_length(多模态/ConditionalGeneration 模型常把文本配置嵌套在 text_config 中)
detect_context_length() {
  local config="$1" v
  for key in max_position_embeddings text_config.max_position_embeddings \
             max_sequence_length text_config.max_sequence_length; do
    v="$(read_config_value "$config" "$key")"
    if [[ -n "$v" ]]; then
      printf '%s' "$v"
      return
    fi
  done
  printf ''
}

# 检测投机解码算法: 显式设置 SGLANG_SPECULATIVE_ALGORITHM 优先;
# 否则按模型是否内置 MTP 层判断 EAGLE 支持(Qwen3 等自带 MTP 层, Muse-Glimmer-30B 等不支持)
# config.json 定位顺序: 本地路径 > HF 缓存 > HF API(远程, 需网络)
detect_speculative_algorithm() {
  if [[ -n "${SGLANG_SPECULATIVE_ALGORITHM+x}" ]]; then
    printf '%s' "${SGLANG_SPECULATIVE_ALGORITHM}"
    return
  fi
  local config
  config="$(locate_model_config)"
  # 内置 MTP 层 -> 支持 EAGLE(Qwen3.5 等字段名为 mtp_num_hidden_layers, 一并纳入匹配)
  if [[ -n "$config" ]] && grep -qE '"(num_mtp_layers|num_nextn_predict_layers|mtp_layers|mtp_num_hidden_layers)"' "$config"; then
    printf 'EAGLE'
    return
  fi
  printf ''
}

# 解析实际 --model-path(持久化布局):
#   本地路径(绝对/相对) -> 原样使用, 不拼接也不下载
#   HF ID -> 命中 <ROOT>/<repo名>/config.json 则直接用(离线可起);
#            否则 hf download 下载到该目录后再用, 使权重跨会话复用
#            下载失败(无 hf / 无网络 / 未授权)仅告警并回退 HF ID, 交由 sglang 自行下载
resolve_model_path() {
  local repo="$MODEL_PATH"
  case "$repo" in
    /* | ./* | ../*) return 0 ;;          # 已是本地路径
  esac
  [[ -n "$SGLANG_MODEL_DIR" ]] || return 0
  if [[ -f "$SGLANG_MODEL_DIR/config.json" ]]; then
    echo ">> 命中本地模型目录: $SGLANG_MODEL_DIR" >&2
    MODEL_PATH="$SGLANG_MODEL_DIR"
    return 0
  fi
  if ! command -v hf >/dev/null 2>&1; then
    echo "警告: 未找到 hf 命令, 回退 HF ID 启动(由 sglang 自行下载到 HF cache)" >&2
    return 0
  fi
  echo ">> 下载模型 $repo -> $SGLANG_MODEL_DIR ..." >&2
  if HF_HUB_ENABLE_XET=1 hf download "$repo" --local-dir "$SGLANG_MODEL_DIR"; then
    MODEL_PATH="$SGLANG_MODEL_DIR"
  else
    echo "警告: 下载失败, 回退 HF ID 启动(由 sglang 自行下载); 需持久化请检查 HF_TOKEN / 网络" >&2
  fi
}

do_start() {
  if is_running; then
    echo "服务已在运行 (PID $(cat "$PID_FILE")), 如需重启请执行: $0 restart"
    exit 0
  fi
  # 进程存在但 PID 文件丢失/失效 -> 兜底清理
  if pgrep_server; then
    echo "检测到无 PID 文件的残留进程, 请先执行: $0 stop" >&2
    exit 1
  fi

  mkdir -p "$LOG_DIR"
  require_venv

  # 硬约束: 模型仓库不可为空(参照 llama/launch.sh)
  if [[ -z "$MODEL_PATH" ]]; then
    echo "错误: SGLANG_MODEL_REPO / MODEL_REPO 均为空。请设置模型(HF ID 或本地路径)。" >&2
    exit 1
  fi

  # Google Drive 不支持作为模型目录(含通过 SGLANG_MODEL_ROOT / MODEL_ROOT 间接指向,
  # 以及把本地路径直接指定为 /content/drive 下的模型)
  if [[ "$SGLANG_MODEL_DIR" == /content/drive/* || "$SGLANG_MODEL_DIR" == /content/drive \
    || "$MODEL_PATH" == /content/drive/* || "$MODEL_PATH" == /content/drive ]]; then
    echo "错误: 不支持 Google Drive 作为模型目录: $SGLANG_MODEL_DIR" >&2
    exit 1
  fi

  # 鉴权密钥检查: 必须由 SGLANG_API_KEY 或回退的 API_KEY 提供(经 .envrc/.env 加载)
  # 两者均未设置时报错; 显式置空(SGLANG_API_KEY="" 或 API_KEY="")则关闭鉴权
  if [[ -z "${SGLANG_API_KEY+x}" && -z "${API_KEY+x}" ]]; then
    echo "错误: 未设置 SGLANG_API_KEY / API_KEY。请在项目目录的 .env 中写入密钥(经 .envrc 由 direnv 加载), 或手动 export。" >&2
    echo "若要临时关闭鉴权: SGLANG_API_KEY=\"\" $0 start" >&2
    exit 1
  fi

  # HF ID -> 本地持久化目录(命中则跳过下载; 失败仅告警, 回退 HF ID)
  resolve_model_path

  # ---- 基于缓存/本地模型 config.json 推导参数 ----
  local MODEL_CONFIG FAMILY CTX
  MODEL_CONFIG="$(locate_model_config || true)"
  if [[ -z "$MODEL_CONFIG" ]]; then
    echo "警告: 未找到模型 config.json(${MODEL_PATH}), 将仅依赖显式环境变量与模型默认值" >&2
  else
    echo "已定位模型配置: ${MODEL_CONFIG}" >&2
  fi
  FAMILY="$(detect_model_family "$MODEL_CONFIG")"

  # 推理/工具调用解析器: 优先显式环境变量, 否则按家族推导, 未知家族则不传(交由 sglang 自动识别)
  local REASONING_P TOOL_P CHAT_KW
  REASONING_P="$REASONING_PARSER"
  TOOL_P="$TOOL_CALL_PARSER"
  CHAT_KW="$CHAT_TEMPLATE_KWARGS"
  if [[ -z "$REASONING_P" || -z "$TOOL_P" || -z "$CHAT_KW" ]]; then
    case "$FAMILY" in
      qwen)
        REASONING_P="${REASONING_P:-qwen3}"
        TOOL_P="${TOOL_P:-qwen3_coder}"
        if [[ -z "$CHAT_KW" ]]; then
          CHAT_KW='{"enable_thinking": true}'
        fi
        ;;
      deepseek)
        REASONING_P="${REASONING_P:-deepseek_v3}"
        TOOL_P="${TOOL_P:-deepseek_v3}"
        CHAT_KW="${CHAT_KW:-}"
        ;;
      glm)
        REASONING_P="${REASONING_P:-glm45}"
        TOOL_P="${TOOL_P:-glm45}"
        CHAT_KW="${CHAT_KW:-}"
        ;;
      muse)
        # Muse-Glimmer: 原生 ATEM 工具协议(atem:function_calls/atem:invoke),
        # 需 muse 解析器把其转换为 OpenAI tool_calls; 推理通道 to=self 亦由 muse 解析
        REASONING_P="${REASONING_P:-muse}"
        TOOL_P="${TOOL_P:-muse}"
        CHAT_KW="${CHAT_KW:-}"
        ;;
      llama)
        # 无专用解析器, 交由 sglang 自动识别(不注入解析器参数)
        REASONING_P="${REASONING_P:-}"
        TOOL_P="${TOOL_P:-}"
        CHAT_KW="${CHAT_KW:-}"
        ;;
      *) REASONING_P="${REASONING_P:-}"; TOOL_P="${TOOL_P:-}"; CHAT_KW="${CHAT_KW:-}" ;;
    esac
  fi

  # 上下文长度: 优先环境变量(0 视为自动), 否则从 config 推导
  CTX="$CONTEXT_LENGTH"
  if [[ -z "$CTX" || "$CTX" == "0" ]]; then
    CTX="$(detect_context_length "$MODEL_CONFIG")"
  fi

  # mamba/SSM 专属参数仅对 mamba 类模型注入
  local IS_MAMBA=0
  if is_mamba_model "$MODEL_CONFIG"; then
    IS_MAMBA=1
  fi

  # 用数组组装启动参数, 避免引号拼接出错, 并便于按条件注入 --api-key
  local args=(
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_NAME"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mm-feature-transport cpu
    --enable-cache-report
    --enable-metrics
    --host "$HOST" --port "$PORT"
  )
  # GPU 专属参数: CPU 平台一律跳过(flashinfer 是 CUDA 专用, fp8 KV 与静态显存占比在
  # CPU 上无意义, 传了会直接启动失败)
  if is_cpu_platform; then
    # CPU 后端: 显式指定设备(置空则交回 sglang 自行探测), 并关掉 overlap schedule
    # (官方 CPU 服务器示例命令即带 --disable-overlap-schedule)
    if [[ -n "${SGLANG_DEVICE:-}" ]]; then
      args+=(--device "$SGLANG_DEVICE")
    fi
    args+=(--disable-overlap-schedule)
    # 官方要求: 必须设置 SGLANG_USE_CPU_ENGINE=1 才会真正启用 CPU 引擎
    # (正常情况下已由 .env.cpu 导出, 这里兜底 export, 保证子进程一定能读到)
    export SGLANG_USE_CPU_ENGINE="${SGLANG_USE_CPU_ENGINE:-1}"
  else
    args+=(
      --attention-backend flashinfer
      --kv-cache-dtype fp8_e4m3
      --mem-fraction-static "$MEM_FRACTION_STATIC"
    )
    # prefill CUDA graph 后端/最大 batch: 非空才注入, 由 .env.t4 等 GPU profile 提供。
    # 是 CUDA 专属(Turing 上 flashinfer ragged prefill 捕获会崩, 故置于 GPU 分支)
    if [[ -n "$CUDA_GRAPH_BACKEND_PREFILL" ]]; then
      args+=(--cuda-graph-backend-prefill "$CUDA_GRAPH_BACKEND_PREFILL")
    fi
    if [[ -n "$CUDA_GRAPH_MAX_BS_PREFILL" ]]; then
      args+=(--cuda-graph-max-bs-prefill "$CUDA_GRAPH_MAX_BS_PREFILL")
    fi
  fi
  if [[ "$IS_MAMBA" -eq 1 ]]; then
    args+=(
      --mamba-radix-cache-strategy extra_buffer_lazy
      --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO"
      --mamba-ssm-dtype bfloat16
    )
  fi
  # 仅在推导/显式指定了对应解析器时注入(避免给非 Qwen 模型硬塞 qwen3 解析器)
  if [[ -n "$REASONING_P" ]]; then
    args+=(--reasoning-parser "$REASONING_P")
  fi
  if [[ -n "$TOOL_P" ]]; then
    args+=(--tool-call-parser "$TOOL_P")
  fi
  if [[ -n "$CHAT_KW" ]]; then
    args+=(--default-chat-template-kwargs "$CHAT_KW")
  fi
  if [[ -n "${SGLANG_API_KEY}" ]]; then
    args+=(--api-key "${SGLANG_API_KEY}")
  fi
  # SGLANG_CTX(0=自动推导)/ 推导值非空时才注入 --context-length, 否则交给模型默认
  if [[ -n "$CTX" ]]; then
    args+=(--context-length "$CTX")
  fi
  # 投机解码: 按模型自动检测 EAGLE 支持(非空才注入; Muse-Glimmer-30B 等无 MTP 层自动关闭)
  local SPECULATIVE_ALGORITHM
  SPECULATIVE_ALGORITHM="$(detect_speculative_algorithm)"
  if [[ -n "$SPECULATIVE_ALGORITHM" ]]; then
    args+=(--speculative-algorithm "$SPECULATIVE_ALGORITHM")
    args+=(--speculative-num-steps "$SPECULATIVE_NUM_STEPS")
    args+=(--speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS")
    if [[ "$SPECULATIVE_ALGORITHM" == "EAGLE" ]]; then
      args+=(--speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK")
    fi
    # sglang 启用投机解码时会自动重置并发上限并打印提示, 显式传入消除提示
    args+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
  fi

  # 追加参数(空格分隔), 用于透传 launch.sh 未覆盖的选项, 例如:
  #   SGLANG_EXTRA_ARGS="--max-mamba-cache-size 132" make sglang-start
  # 混合 Mamba 模型 + 投机解码下, mamba state cache 会把并发压到
  # max_mamba_cache_size/4; ratio 调到 1.0 也只有 ~85 slot(并发 ~21),
  # 要更高并发只能直接指定 cache size(代价是 KV 池变小)。
  # shellcheck disable=SC2206
  if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then
    extra=($SGLANG_EXTRA_ARGS)
    args+=("${extra[@]}")
    unset extra
  fi

  # 启动入口: 优先官方推荐的 `sglang serve`(避免 launch_server 的 UserWarning),
  # 旧版本 venv 中无 sglang 命令时回退 python -m 方式
  local SERVER_CMD
  if command -v sglang >/dev/null 2>&1; then
    SERVER_CMD=(sglang serve)
  else
    echo "警告: venv 中未找到 sglang 命令, 回退到 python -m sglang.launch_server" >&2
    SERVER_CMD=(python -m sglang.launch_server)
  fi

  # 展开后的启动命令(回显 + 写入日志, 便于排查)
  local LAUNCH_CMD=("${SERVER_CMD[@]}" "${args[@]}")

  echo "启动 SGLang 服务... (日志: ${LOG_FILE})" | tee -a "$LOG_FILE"
  echo ">> 模型: $MODEL_PATH" | tee -a "$LOG_FILE"
  echo ">> 模型基础盘: $SGLANG_MODEL_ROOT (来源: $MODEL_ROOT_SOURCE)" | tee -a "$LOG_FILE"
  echo ">> FlashInfer 架构: FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-<自动>} (nvcc $(detect_nvcc_version), CUDA_HOME=${CUDA_HOME:-<系统默认>})" | tee -a "$LOG_FILE"
  echo ">> 启动命令: ${LAUNCH_CMD[*]}" | tee -a "$LOG_FILE"

  # 启动命令单独追加写入命令日志(带时间戳), 便于事后查看实际启动参数
  {
    echo "===== $(date '+%F %T') [sglang] start ====="
    printf '  %s\n' "${LAUNCH_CMD[@]}"
  } >>"$CMD_LOG"

  # 用 setsid 脱离终端; 由子 shell 自身写入真实 PID(避免记录到瞬退的 setsid 父进程)
  # 命令与参数以 "$@" 透传, 规避手工转义
  # shellcheck disable=SC2016
  setsid bash -c '
    echo $$ > "$1"; shift
    exec "$@"
  ' bash "$PID_FILE" "${LAUNCH_CMD[@]}" >>"$LOG_FILE" 2>&1 </dev/null &

  # 稍候确认子进程已起来并写入了有效 PID
  sleep 1
  if is_running; then
    echo "已提交启动 (PID $(cat "$PID_FILE"))。首次启动需 JIT 编译内核, 请用 '$0 status' 等待就绪。"
  else
    echo "启动失败, 请查看日志: $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
  fi
}

do_stop() {
  if ! pgrep_server; then
    # 用 return 而非 exit: restart 会在 stop 之后继续 start
    echo "服务未在运行"
    rm -f "$PID_FILE"
    return 0
  fi
  echo "停止 SGLang 服务..."
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
    echo "错误: SGLang 服务未在运行。请先执行: $0 start" >&2
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
  if [[ -n "$SGLANG_API_KEY" ]]; then
    AUTH=(-H "Authorization: Bearer $SGLANG_API_KEY")
  fi

  echo ">> 发送测试对话 (model=${SERVED_NAME}, port=${PORT}) ..."
  curl -s --max-time 30 "http://localhost:${PORT}/v1/chat/completions" \
    "${AUTH[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${SERVED_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"你好, 请用一句话回复\"}], \"max_tokens\": 64}" \
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
  grep -E "^# +(用法|环境变量|  \.|  SGLANG_|  API_KEY|    SGLANG_|    API_KEY)" "$0" | sed 's/^# \{1,\}//'
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
