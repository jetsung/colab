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
#   LLAMA_MODEL_REPO  模型来源(不可为空; 未设置时回退 MODEL_REPO), 支持四种写法:
#     本地路径   /abs/model.gguf 或 ./model.gguf — 直接使用, 不下载
#     file://    file:///abs/model.gguf(绝对) 或 file://rel/model.gguf(相对当前目录)
#     hf://      hf://<org>/<repo>/<file> 指向具体文件(直接用该文件, 跳过量化档选择)
#                或 hf://<org>/<repo> 裸仓库(按 LLAMA_QUANT 自适应选择)
#     HF https   https://huggingface.co/<org>/<repo>/(blob|resolve)/<rev>/<file>
#                HF 来源下载到标准缓存 ~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/
#                并定位真实文件路径; 缓存已存在则不联网
#   LLAMA_MODEL_NAME  模型名(未设置时从文件名/仓库推导); 服务 --alias 为其小写形式
#   LLAMA_QUANT       量化档(裸仓库来源时不可为空, 无默认; 由 .envrc / profile 提供;
#                     指向具体文件或本地文件时忽略)
#   LLAMA_NGL         GPU 卸载层数(默认 999=全部; .env.cpu 为 0=纯 CPU 推理)
#   LLAMA_CTX         上下文长度(默认 0=由 llama.cpp 按空闲显存拟合; .env.cpu 为 8192)
#   LLAMA_THREADS     CPU 推理线程数(默认 0=由 llama.cpp 按核心数自行决定, 即不传 -t)
#   LLAMA_DIR       安装目录(默认 /content/llama.cpp; 由 .envrc 导出, 可覆盖)
#   LLAMA_SERVER     llama 二进制路径(默认从 PATH 查找 command -v llama; 未命中回退 <LLAMA_DIR>/build/bin)
#   LLAMA_HOST / LLAMA_PORT   监听地址与端口(内部变量 HOST/PORT, 默认 0.0.0.0 / 30000)
#   LLAMA_API_KEY     服务器 API 密钥(未设置时回退 API_KEY)
#   LLAMA_XET         1=启用 HF Xet 存储(默认), 0=禁用
#   LLAMA_METRICS     1=开放 /metrics 端点(默认, 供根目录 bench.py 采样并发), 0=禁用
#   LLAMA_VISION      auto=按模型能力自动检测, 1=强制尝试启用, 0=禁用(默认 0)
#   LLAMA_MMPROJ      视觉投影器 mmproj 来源(可选; 与 LLAMA_MODEL_REPO 同样的四种写法;
#                     缺省自动检测模型缓存目录 mmproj-*.gguf, 再按需自动下载)
#   LLAMA_MMPROJ_REPO mmproj 自动下载源仓库(默认同模型仓库; 空=禁用自动下载)
#
# 裸仓库的自适应解析(不写死任何路径形式):
#   先扫描 HF 缓存快照目录, 本地分片齐全则直接启动; 否则拉取仓库文件清单,
#   按 LLAMA_QUANT 自动推导真实布局并下载。已验证的两种典型布局:
#     unsloth/Qwen3.8-Flash-Next-GGUF  -> UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
#     unsloth/Qwen3.8-27B-GGUF         -> Qwen3.8-27B-UD-Q4_K_XL.gguf  (根目录单文件)
#   同时兼容"根目录 + 分片"(*-<QUANT>-00001-of-000NN.gguf)。
#   推导结果: 布局目录 / 分片总数 / 模型名前缀(--alias) / 下载清单 / mmproj 路径。
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

# profile 兜底加载(不依赖 direnv):
# 环境变量一律继承调用方 shell(外层 direnv hook 注入)。若调用方未 cd 进本目录(引擎 .envrc
# 未加载), LLAMA_QUANT / LLAMA_MODEL_REPO 等引擎专属默认值会缺失, 这里按 GPU_PROFILE
# (未设置时按 nvidia-smi 探测, 无显卡则按 cpu)直接 source 本目录的 .env.<g4|t4|cpu>。
# profile 是纯 bash(只有 export VAR="${VAR:-默认}"), 不含 direnv stdlib, 可安全 source:
# 幂等且不覆盖已有值(direnv 已加载过再 source 一次无副作用)。
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
  echo ">> [llama] 加载 profile: ${profile} (${f})" >&2
  # shellcheck disable=SC1090
  source "$f"
}
# 仅需要模型参数的子命令才兜底, 避免 status/stop/logs 也去探测显卡
case "${1:-}" in
  start | restart) load_gpu_profile ;;
esac

# 其余环境变量(LLAMA_API_KEY / MODEL_REPO / MODEL_ROOT 等)由外部环境提供
# (外层 .envrc 经 direnv 注入, 或命令行 export); 未提供时沿用下方默认值。
# LLAMA_* 不存在时回退到通用变量(API_KEY / MODEL_REPO)
# 硬约束: LLAMA_MODEL_REPO / LLAMA_QUANT / LLAMA_MODEL_NAME 不可为空(在 do_start 内校验)

readonly LLAMA_MODEL_REPO="${LLAMA_MODEL_REPO:-${MODEL_REPO:-}}"
# 模型名前缀(用于本地分片文件匹配与 --alias 兜底):
# 未显式设置时, 从仓库值取末段并去除 -GGUF 后缀(指向具体文件时由文件名推导覆盖)
#   例: unsloth/Qwen3.8-Flash-Next-GGUF -> Qwen3.8-Flash-Next
if [[ -z "${LLAMA_MODEL_NAME:-}" ]]; then
  LLAMA_MODEL_NAME="${LLAMA_MODEL_REPO##*/}"    # 取 / 后部分
  LLAMA_MODEL_NAME="${LLAMA_MODEL_NAME%-GGUF}"  # 去除末尾 -GGUF 后缀
  readonly LLAMA_MODEL_NAME_EXPLICIT=0          # 未显式设置: 别名以解析推导结果为准
else
  readonly LLAMA_MODEL_NAME_EXPLICIT=1          # 显式设置: 别名与匹配提示均以其为准
fi
readonly LLAMA_MODEL_NAME
# 服务别名: 模型名转小写。未显式设置 LLAMA_MODEL_NAME 时, do_start 内改用解析推导的 PLAN_NAME
LLAMA_MODEL_ALIAS="${LLAMA_MODEL_NAME,,}"
# 无默认值: 裸仓库来源时必须由外部提供(.envrc / gpu profile / 命令行); 此处仅声明空以规避 set -u
readonly LLAMA_QUANT="${LLAMA_QUANT:-}"
# 注: HF 来源统一下载到 HF 标准缓存(HF_HUB_CACHE / HF_HOME, 默认 ~/.cache/huggingface/hub),
#     不再使用本地模型盘; 缓存目录遵循 HF_HUB_CACHE / HF_HOME 环境变量
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

# 动态库解析路径(脚本级 export, llama / llama-gguf 及 setsid 子进程均生效):
#   - 二进制同目录: 官方预编译包把 libllama-*.so / libggml-*.so 放在二进制旁,
#     但部分构建未写入 RPATH($ORIGIN), 直接运行会报 cannot open shared object file
#   - /usr/lib64-nvidia: 容器环境中 NVIDIA 驱动库(libcuda.so.1)的挂载位置
#   已存在于 LD_LIBRARY_PATH 中的目录不重复追加
_llama_lib_dirs=()
_llama_bin_dir="$(dirname "$LLAMA_SERVER")"
[[ -d "$_llama_bin_dir" ]] && _llama_lib_dirs+=("$_llama_bin_dir")
[[ -d /usr/lib64-nvidia ]] && _llama_lib_dirs+=("/usr/lib64-nvidia")
if ((${#_llama_lib_dirs[@]})); then
  _llama_lib_add="$(IFS=:; echo "${_llama_lib_dirs[*]}")"
  case ":${LD_LIBRARY_PATH:-}:" in
    *"$_llama_lib_add"*) ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_llama_lib_add" ;;
  esac
fi
unset _llama_bin_dir _llama_lib_add _llama_lib_dirs
# 监听地址与端口: 内部变量 HOST/PORT, 可用 LLAMA_HOST / LLAMA_PORT 环境变量覆盖
readonly PORT="${LLAMA_PORT:-30000}"
readonly HOST="${LLAMA_HOST:-0.0.0.0}"
readonly LLAMA_NGL="${LLAMA_NGL:-999}"
readonly LLAMA_CTX="${LLAMA_CTX:-0}"
# CPU 推理线程数: 0/空 = 不传 -t, 交给 llama.cpp 按核心数自行决定
readonly LLAMA_THREADS="${LLAMA_THREADS:-0}"
readonly LLAMA_API_KEY="${LLAMA_API_KEY:-${API_KEY:-}}"
readonly LLAMA_XET="${LLAMA_XET:-1}"   # 1=启用 HuggingFace Xet 存储(默认), 0=禁用
# Prometheus 指标端点 /metrics: 1=启用(默认, 供根目录 bench.py 采样并发), 0=禁用
readonly LLAMA_METRICS="${LLAMA_METRICS:-1}"
# 多模态视觉投影器(mmproj): 用于图片/视频输入(可选, 缺省不启用)
#   LLAMA_VISION      auto=按模型能力自动检测, 1=跳过检测并尝试启用, 0=完全禁用
#   LLAMA_MMPROJ       显式指定 mmproj 文件路径(优先级最高; 不设置时自动在模型目录检测 mmproj-*.gguf)
#   LLAMA_MMPROJ_REPO  自动下载源(默认同 LLAMA_MODEL_REPO; 设为空字符串则禁用自动下载)
readonly LLAMA_VISION="${LLAMA_VISION:-0}"
readonly LLAMA_MMPROJ="${LLAMA_MMPROJ:-}"
readonly LLAMA_MMPROJ_REPO="${LLAMA_MMPROJ_REPO:-}"

# 模型解析结果(由 resolve_model_file 填充; 此处先声明空值规避 set -u)
PLAN_NAME=""              # 由真实文件名/仓库清单推导的模型名前缀(用于 --alias)
PLAN_MMPROJ_INCLUDE=""    # 需下载的 mmproj 仓库内相对路径(无则空)
MODEL_FILE=""             # 首个分片(或单文件)的本地绝对路径
MAIN_SNAP=""              # 主模型所在的 HF 缓存快照目录(本地文件来源时为空)
MODEL_SOURCE_REPO=""      # 模型来源的仓库 ID(hf-repo / hf-file 时非空; 本地文件为空)

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
  ./colab.sh install llama            # GitHub 最新 prerelease Ubuntu 通用预编译二进制(快速, 免编译)
  ./colab.sh install llama --build    # 源码编译 CUDA 版本(PR #27742, 支持 Qwen3.8-Flash-Next)

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

# --------------------- HF 标准缓存(hub cache)辅助 ----------------------------
# 缓存根目录(遵循 HF_HUB_CACHE / HF_HOME, 默认 ~/.cache/huggingface/hub)
hub_root() {
  printf '%s' "${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
}
# 仓库的缓存目录: models--<org>--<repo>(仓库路径中的 / 替换为 --)
hub_model_dir() {
  printf '%s/models--%s' "$(hub_root)" "${1//\//--}"
}
# 当前主快照目录(refs/main 指向的 revision; 仓库从未下载过则为空串)
hub_snapshot_dir() {
  local d rev
  d="$(hub_model_dir "$1")"
  rev="$(cat "$d/refs/main" 2>/dev/null || true)"
  [[ -n "$rev" ]] && printf '%s/snapshots/%s' "$d" "$rev"
  return 0
}
# 在仓库缓存 snapshots/ 下定位文件(相对路径可含子目录, 取最新一份); 未找到输出空
cache_find_file() {
  local found=""
  found="$(find -L "$1/snapshots" -type f -path "*/$2" -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d' ' -f2-)"
  [[ -n "$found" && -f "$found" ]] && printf '%s' "$found"
  return 0
}
# hf download 到标准缓存(不带 --local-dir); 成功时全局 DL_PATH 为下载文件路径(解析失败为空)
DL_PATH=""
hf_download() {
  local url="$1" out
  DL_PATH=""
  if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: 未找到 hf 命令(用于下载模型)。请先执行: ./colab.sh install llama" >&2
    return 1
  fi
  echo ">> 下载 $url" >&2
  if [[ "$LLAMA_XET" == "1" ]]; then
    out="$(HF_HUB_ENABLE_XET=1 HF_TOKEN="${HF_TOKEN:-}" hf download "$url")" || {
      echo "ERROR: hf download 失败: $url" >&2
      return 1
    }
  else
    out="$(HF_TOKEN="${HF_TOKEN:-}" hf download "$url")" || {
      echo "ERROR: hf download 失败: $url" >&2
      return 1
    }
  fi
  # stdout 末行为下载文件路径(新版格式 "  path: <路径>", 旧版为裸路径)
  out="${out##*$'\n'}"
  out="${out##*'path: '}"
  [[ -n "$out" && -f "$out" ]] && DL_PATH="$out"
  return 0
}

# 归类模型来源字符串, stdout: <类别>|<值>
#   file|<path>                  本地文件(相对路径按当前目录展开), 直接使用
#   hf-file|<org>/<repo>|<file>  仓库内具体文件(可含子目录)
#   hf-repo|<org>/<repo>         裸仓库(按 LLAMA_QUANT 自适应选择)
classify_model_source() {
  local val="$1" rest org repo seg3
  case "$val" in
    file://*)
      printf 'file|%s' "${val#file://}"
      return 0
      ;;
    ./*|../*)
      printf 'file|%s' "$val"
      return 0
      ;;
    hf://*)
      rest="${val#hf://}"
      if [[ "$rest" == */*/* ]]; then
        org="${rest%%/*}"; rest="${rest#*/}"
        repo="${rest%%/*}"; rest="${rest#*/}"
        printf 'hf-file|%s/%s|%s' "$org" "$repo" "$rest"
      else
        printf 'hf-repo|%s' "$rest"
      fi
      return 0
      ;;
    https://huggingface.co/*)
      rest="${val#https://huggingface.co/}"
      org="${rest%%/*}"; rest="${rest#*/}"
      repo="${rest%%/*}"; rest="${rest#*/}"
      seg3="${rest%%/*}"
      if [[ "$seg3" == "blob" || "$seg3" == "resolve" ]]; then
        rest="${rest#*/}"; rest="${rest#*/}"   # 跳过 blob|resolve 与 revision
        if [[ -z "$rest" ]]; then
          echo "ERROR: HF URL 缺少文件路径: $val" >&2
          return 1
        fi
        printf 'hf-file|%s/%s|%s' "$org" "$repo" "$rest"
      elif [[ "$rest" == */* ]]; then
        echo "ERROR: 不支持的 HF URL(仅支持文件直链 /blob/ /resolve/ 或仓库主页): $val" >&2
        return 1
      else
        printf 'hf-repo|%s/%s' "$org" "$repo"
      fi
      return 0
      ;;
    https://*|http://*|ftp://*|s3://*)
      echo "ERROR: 不支持的来源协议: $val (仅支持 https://huggingface.co/ 与 hf://)" >&2
      return 1
      ;;
    /*)
      printf 'file|%s' "$val"
      return 0
      ;;
    *)
      if [[ "$val" == */*/* ]]; then
        echo "ERROR: 无法识别的模型来源: $val (本地路径请以 / 或 file:// 开头; HF 请用 hf://<org>/<repo>[/<file>])" >&2
        return 1
      elif [[ "$val" == */* ]]; then
        printf 'hf-repo|%s' "$val"    # 裸仓库 ID(org/repo)
      else
        printf 'file|%s' "$val"       # 无斜杠: 当前目录下的本地文件
      fi
      return 0
      ;;
  esac
}

# 解析 hf-file 来源并定位/下载文件, stdout: 本地路径(缓存命中则不联网)
resolve_hf_file() {
  local repo="$1" file="$2" found
  found="$(cache_find_file "$(hub_model_dir "$repo")" "$file")"
  if [[ -n "$found" ]]; then
    echo ">> 缓存已存在: $found" >&2
    printf '%s' "$found"
    return 0
  fi
  hf_download "hf://${repo}/${file}" || return 1
  if [[ -n "$DL_PATH" ]]; then
    printf '%s' "$DL_PATH"
    return 0
  fi
  found="$(cache_find_file "$(hub_model_dir "$repo")" "$file")"
  if [[ -n "$found" ]]; then
    printf '%s' "$found"
    return 0
  fi
  echo "ERROR: 下载完成但未定位到文件: $file (缓存目录: $(hub_model_dir "$repo"))" >&2
  return 1
}

# 从模型文件名推导名称前缀(用于 --alias): 去扩展名 / 分片序号 / 量化档后缀
derive_model_name() {
  local n
  n="$(basename "$1")"
  n="${n%.gguf}"
  n="${n%-?????-of-?????}"     # 去分片序号 -00001-of-00004
  if [[ -n "$LLAMA_QUANT" && "$n" == *-"$LLAMA_QUANT" ]]; then
    n="${n%-"$LLAMA_QUANT"}"
  fi
  printf '%s' "$n"
}

# 自适应解析模型布局: 不写死路径形式。
# 先扫描 HF 缓存快照目录(深度 2)判断本地分片是否齐全; 不齐全才联网拉仓库文件清单,
# 按 LLAMA_QUANT 推导布局(<QUANT>/ 子目录 | 根目录; 单文件 | 分片)并给出下载清单。
# 输出 KEY=value(值经 shlex 转义, 供 eval):
#   PLAN_ACTION         local=本地已齐全, download=需下载
#   PLAN_MODEL_REL      首个分片(或单文件)相对快照目录的路径
#   PLAN_DIR / PLAN_NAME / PLAN_TOTAL / PLAN_HAVE / PLAN_INCLUDE / PLAN_MMPROJ_INCLUDE
#   PLAN_ACTION         local=本地已齐全, download=需下载
#   PLAN_MODEL_REL      首个分片(或单文件)相对快照目录的路径
#   PLAN_DIR / PLAN_NAME / PLAN_TOTAL / PLAN_HAVE / PLAN_INCLUDE / PLAN_MMPROJ_INCLUDE
plan_model_scan() {
  REPO="$1" QUANT="$LLAMA_QUANT" SCAN_ROOT="$3" \
  NAME_HINT="$LLAMA_MODEL_NAME" VISION="$LLAMA_VISION" MMPROJ_REPO="$2" \
  HF_TOKEN="${HF_TOKEN:-}" HF_ENDPOINT="${HF_ENDPOINT:-}" \
  python3 - <<'PY'
import json, os, re, shlex, sys, urllib.error, urllib.request

REPO        = os.environ.get("REPO", "")
QUANT       = os.environ.get("QUANT", "")
SCAN_ROOT   = os.environ.get("SCAN_ROOT", "")   # HF 缓存快照目录(扫描根)
NAME_HINT   = os.environ.get("NAME_HINT", "")
VISION      = os.environ.get("VISION", "0")
MMPROJ_REPO = os.environ.get("MMPROJ_REPO", "")
TOKEN       = os.environ.get("HF_TOKEN", "")
ENDPOINT    = os.environ.get("HF_ENDPOINT", "").rstrip("/") or "https://huggingface.co"
MAX_DEPTH   = 2      # 本地扫描深度: 覆盖 <snap>/<file> 与 <snap>/<QUANT>/<file> 两种布局
AUX_DIRS    = {"MTP"}

def emit(k, v):
    print("%s=%s" % (k, shlex.quote("" if v is None else str(v))))

def warn(msg):
    sys.stderr.write("警告: %s\n" % msg)

def fail(msg):
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(1)

_Q        = re.escape(QUANT)
RE_SHARD  = re.compile(r"^(?P<p>.+)-%s-(?P<i>\d+)-of-(?P<t>\d+)$" % _Q, re.I)
RE_SINGLE = re.compile(r"^(?P<p>.+)-%s$" % _Q, re.I)

def is_aux(rel):
    """排除 mmproj / MTP 等附属文件(不参与主模型匹配)"""
    base = os.path.basename(rel).lower()
    parts = rel.split("/")
    return (base.startswith("mmproj") or base.startswith("mtp")
            or (len(parts) > 1 and parts[0] in AUX_DIRS))

def parse_model(rel):
    """按 QUANT 解析 gguf -> (仓库内目录, 模型名前缀, 分片序号, 分片总数); 不匹配返回 None"""
    if is_aux(rel) or not rel.lower().endswith(".gguf"):
        return None
    d, base = os.path.split(rel)
    name = base[: -len(".gguf")]
    m = RE_SHARD.match(name)       # <prefix>-<QUANT>-00001-of-000NN
    if m:
        return d, m.group("p"), int(m.group("i")), int(m.group("t"))
    m = RE_SINGLE.match(name)      # <prefix>-<QUANT>
    if m:
        return d, m.group("p"), 1, 1
    return None

def group(rels):
    """按 (目录, 前缀, 总数) 归组, 同组不同序号视为同一模型的不同分片"""
    gs = {}
    for rel in rels:
        r = parse_model(rel)
        if not r:
            continue
        d, p, i, t = r
        g = gs.setdefault((d, p, t), {"dir": d, "prefix": p, "total": t, "files": {}})
        g["files"][i] = rel
    return list(gs.values())

def complete(g):
    return set(range(1, g["total"] + 1)).issubset(g["files"])

def pick(gs):
    if not gs:
        return None
    def score(g):
        # 优先: 完整 > <QUANT>/ 子目录 > 根目录 > 其他目录; 再按与 NAME_HINT 一致
        layout = 2 if g["dir"] == QUANT else (1 if g["dir"] == "" else 0)
        hint = 1 if NAME_HINT and g["prefix"].lower() == NAME_HINT.lower() else 0
        return (1 if complete(g) else 0, layout, hint, g["total"])
    return max(gs, key=score)

def scan_local(root):
    if not root or not os.path.isdir(root):
        return []
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        rel = os.path.relpath(dirpath, root)
        if (0 if rel == "." else rel.count("/") + 1) >= MAX_DEPTH:
            dirnames[:] = []      # 不再深入
        out.extend(os.path.relpath(os.path.join(dirpath, f), root)
                   for f in filenames if f.lower().endswith(".gguf"))
    return out

_remote = None

def remote(fatal=False):
    """惰性拉取仓库文件清单; 非致命失败时降级为空清单(不影响已就绪的本地模型)"""
    global _remote
    if _remote is None:
        try:
            _remote = _list_remote(REPO)
        except Exception as e:
            if fatal:
                fail(str(e))
            warn("无法获取 %s 的远程文件清单(%s), 仅使用本地文件" % (REPO, e))
            _remote = []
    return _remote

def _list_remote(repo):
    url = "%s/api/models/%s" % (ENDPOINT, repo)
    req = urllib.request.Request(url, headers={"User-Agent": "llama/launch.sh"})
    if TOKEN:
        req.add_header("Authorization", "Bearer %s" % TOKEN)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError("无法列出 %s (HTTP %s): HF_TOKEN 无效或未接受该仓库许可证" % (repo, e.code))
        if e.code == 404:
            raise RuntimeError("仓库不存在或无权访问: %s" % repo)
        raise RuntimeError("列出 %s 的文件失败: HTTP %s" % (repo, e.code))
    except Exception as e:
        raise RuntimeError("无法连接 HuggingFace(%s): %s" % (url, e))
    return [s.get("rfilename", "") for s in data.get("siblings", [])]

def pick_mmproj(rels):
    c = [r for r in rels if r.lower().endswith(".gguf")
         and os.path.basename(r).lower().startswith("mmproj")]
    if not c:
        return None
    def rank(r):                  # F16 优先(体积小于 BF16), 其次 BF16, 其余最后
        b = os.path.basename(r).lower()
        return 1 if "bf16" in b else (0 if "f16" in b else 2)
    c.sort(key=lambda r: (rank(r), r.count("/"), r))
    return c[0]

def suggest_quants(rels):
    out = set()
    for rel in rels:
        if is_aux(rel) or not rel.lower().endswith(".gguf"):
            continue
        d, base = os.path.split(rel)
        if d:
            out.add(d)
        name = base[: -len(".gguf")]
        if NAME_HINT and name.lower().startswith(NAME_HINT.lower() + "-"):
            name = name[len(NAME_HINT) + 1:]
        out.add(re.sub(r"-\d+-of-\d+$", "", name))
    return sorted(x for x in out if x)

local_rels = scan_local(SCAN_ROOT)
chosen = pick(group(local_rels))
local_ok = bool(chosen) and complete(chosen)

if local_ok:
    have = len(chosen["files"])
    emit("PLAN_ACTION", "local")
    emit("PLAN_INCLUDE", "")
else:
    rels = remote(fatal=True)
    chosen = pick(group(rels))
    if not chosen:
        fail("仓库 %s 中未找到量化档 %s 的 .gguf 文件。可用档位: %s"
             % (REPO, QUANT, ", ".join(suggest_quants(rels)) or "(无法解析)"))
    if not complete(chosen):
        fail("仓库 %s 中 %s 的分片不完整: 仅见 %s/%s"
             % (REPO, QUANT, len(chosen["files"]), chosen["total"]))
    have = 0
    for g in group(local_rels):                       # 同一组的本地已有分片数
        if (g["dir"], g["prefix"], g["total"]) == (chosen["dir"], chosen["prefix"], chosen["total"]):
            have = len(g["files"])
    emit("PLAN_ACTION", "download")
    emit("PLAN_INCLUDE", "\n".join(sorted(chosen["files"].values())))

emit("PLAN_DIR", chosen["dir"])
emit("PLAN_NAME", chosen["prefix"])
emit("PLAN_TOTAL", chosen["total"])
emit("PLAN_HAVE", have)
emit("PLAN_MODEL_REL", chosen["files"][1])   # 快照内相对路径(下载后 refs/main 可能更新, 由 bash 重新定位)
# mmproj: 仅当与主模型同仓库时才能用清单给出精确路径(跨仓库由 bash 侧回退 glob 下载)
mm_inc = ""
if not pick_mmproj(local_rels) and VISION != "0" and MMPROJ_REPO \
        and MMPROJ_REPO.lower() == REPO.lower():
    mm_inc = pick_mmproj(remote()) or ""
emit("PLAN_MMPROJ_INCLUDE", mm_inc)
PY
}

# 解析并(按需)下载模型, 结果写入全局: MODEL_FILE / PLAN_NAME / MAIN_SNAP / MODEL_SOURCE_REPO
resolve_model_file() {
  local cls val
  cls="$(classify_model_source "$LLAMA_MODEL_REPO")" || exit 1
  val="${cls#*|}"; cls="${cls%%|*}"

  case "$cls" in
    file)
      if [[ ! -f "$val" ]]; then
        echo "ERROR: LLAMA_MODEL_REPO 指向的文件不存在: $val" >&2
        exit 1
      fi
      MODEL_FILE="$val"
      PLAN_NAME="$(derive_model_name "$val")"
      MAIN_SNAP=""
      MODEL_SOURCE_REPO=""
      echo ">> 使用本地模型文件: $MODEL_FILE" >&2
      ;;
    hf-file)
      local repo="${val%%|*}" file="${val#*|}" p
      MODEL_SOURCE_REPO="$repo"
      p="$(resolve_hf_file "$repo" "$file")" || exit 1
      MODEL_FILE="$p"
      PLAN_NAME="$(derive_model_name "$MODEL_FILE")"
      MAIN_SNAP="$(hub_snapshot_dir "$repo")"
      ;;
    hf-repo)
      resolve_repo_model "$val" || exit 1
      ;;
  esac
}

# 裸仓库来源: 自适应解析布局并(按需)下载, 写入 MODEL_FILE / PLAN_NAME / MAIN_SNAP / MODEL_SOURCE_REPO
resolve_repo_model() {
  local repo="$1" mdir snap plan fb_repo
  mdir="$(hub_model_dir "$repo")"
  MODEL_SOURCE_REPO="$repo"
  fb_repo="${LLAMA_MMPROJ_REPO:-$repo}"
  [[ -n "$LLAMA_QUANT" ]] || {
    echo "ERROR: LLAMA_QUANT 不可为空。请显式设置(如 LLAMA_QUANT=UD-Q4_K_XL, 或由 gpu profile/.envrc 提供; 或改用指向具体文件的来源如 hf://<org>/<repo>/<file>)。" >&2
    return 1
  }
  snap="$(hub_snapshot_dir "$repo")"
  if ! plan="$(plan_model_scan "$repo" "$fb_repo" "$snap")"; then
    echo "ERROR: 模型解析失败(见上方输出)。" >&2
    return 1
  fi
  eval "$plan"   # shellcheck disable=SC2091  # 值由 python shlex 转义

  local layout="${PLAN_DIR:-<仓库根目录>}"
  if [[ "$PLAN_ACTION" == "download" ]]; then
    echo ">> 仓库布局: $layout (${PLAN_NAME}-${LLAMA_QUANT}, ${PLAN_TOTAL} 分片)" >&2
    echo ">> 本地分片 ${PLAN_HAVE}/${PLAN_TOTAL}, 开始下载 $repo (缓存: $mdir) ..." >&2
    echo "   (该仓库使用 Xet 存储，需要 hf_xet；install.sh 已安装)" >&2
    local -a dl=(hf download "$repo")
    local inc
    while IFS= read -r inc; do
      [[ -n "$inc" ]] && dl+=(--include "$inc")
    done <<<"$PLAN_INCLUDE"
    if [[ "$LLAMA_VISION" == "1" && -n "$PLAN_MMPROJ_INCLUDE" ]]; then
      dl+=(--include "$PLAN_MMPROJ_INCLUDE")
    fi
    echo ">> 下载 $(( ${#dl[@]} - 1 )) 个 include 项 -> HF 标准缓存" >&2
    if [[ "$LLAMA_XET" == "1" ]]; then
      HF_HUB_ENABLE_XET=1 HF_TOKEN="${HF_TOKEN:-}" "${dl[@]}" >&2
    else
      HF_TOKEN="${HF_TOKEN:-}" "${dl[@]}" >&2
    fi
  else
    echo ">> 本地模型已齐全 (${PLAN_HAVE}/${PLAN_TOTAL} 分片, 布局: $layout), 跳过下载。" >&2
  fi

  # 定位首个分片(下载可能更新 refs/main, 重新读取快照)
  snap="$(hub_snapshot_dir "$repo")"
  if [[ -n "$snap" ]]; then
    MODEL_FILE="$snap/${PLAN_MODEL_REL}"
  else
    MODEL_FILE=""
  fi
  if [[ -z "$MODEL_FILE" || ! -f "$MODEL_FILE" ]]; then
    MODEL_FILE="$(cache_find_file "$mdir" "$PLAN_MODEL_REL")"
  fi
  if [[ -z "$MODEL_FILE" || ! -f "$MODEL_FILE" ]]; then
    echo "ERROR: 未找到模型文件: ${PLAN_MODEL_REL}(缓存目录: $mdir)。下载可能失败, 请查看日志。" >&2
    return 1
  fi
  MAIN_SNAP="$snap"
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

# 定位多模态视觉投影器(mmproj)。优先级: 显式 LLAMA_MMPROJ(四种来源写法)
#   > 模型所在目录/快照自动检测 > 按仓库清单推导的精确路径下载(同仓库) / glob 下载(跨仓库)。
# LLAMA_VISION=auto 时仅在主模型声明支持视觉时启用(见 model_supports_vision),
# 避免把不相干的 mmproj 传给纯文本模型导致启动/请求失败。
# LLAMA_VISION=1 跳过能力检测并尝试启用; LLAMA_VISION=0 直接禁用 mmproj。
# 未找到时输出空字符串(不报错): 文本服务照常可用, 仅图片/视频输入不可用。
# 仅当显式指定的 LLAMA_MMPROJ 无法解析/下载时返回非零(启动失败; LLAMA_VISION=0 除外)。
resolve_mmproj_file() {
  local model_file="$1"

  case "$LLAMA_VISION" in
    0)
      echo ">> LLAMA_VISION=0, 禁用视觉支持, 跳过 mmproj。" >&2
      return 0
      ;;
    auto)
      if ! model_supports_vision "$model_file"; then
        echo ">> 主模型不支持视觉(无 image_token/视觉元数据), 跳过 mmproj。" >&2
        return 0
      fi
      ;;
    1)
      echo ">> LLAMA_VISION=1, 跳过模型视觉能力检测, 尝试启用 mmproj。" >&2
      ;;
  esac

  # 1) 显式指定(本地路径 / file:// / hf:// / HF https URL)
  if [[ -n "$LLAMA_MMPROJ" ]]; then
    local cls val repo file p
    cls="$(classify_model_source "$LLAMA_MMPROJ")" || return 1
    val="${cls#*|}"; cls="${cls%%|*}"
    case "$cls" in
      file)
        if [[ ! -f "$val" ]]; then
          echo "ERROR: 指定的 LLAMA_MMPROJ 不存在: $val" >&2
          return 1
        fi
        printf '%s' "$val"
        return 0
        ;;
      hf-file)
        repo="${val%%|*}"; file="${val#*|}"
        p="$(resolve_hf_file "$repo" "$file")" || return 1
        printf '%s' "$p"
        return 0
        ;;
      hf-repo)
        echo "ERROR: LLAMA_MMPROJ 需指向具体文件(如 hf://<org>/<repo>/mmproj-F16.gguf), 而非仓库: $LLAMA_MMPROJ" >&2
        return 1
        ;;
    esac
  fi

  local mm=""
  local fb_repo="${LLAMA_MMPROJ_REPO:-$MODEL_SOURCE_REPO}"
  # 2) 模型所在目录/快照自动检测(覆盖根目录布局与 <QUANT>/ 子目录布局)
  if [[ -n "$MAIN_SNAP" && -d "$MAIN_SNAP" ]]; then
    mm="$(find -L "$MAIN_SNAP" -maxdepth 3 -type f -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)"
  fi
  if [[ -z "$mm" ]]; then
    mm="$(find -L "$(dirname "$model_file")" -maxdepth 2 -type f -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)"
  fi
  if [[ -n "$mm" ]]; then
    printf '%s' "$mm"
    return 0
  fi

  # 3) 自动下载(HF 标准缓存): 同仓库按清单精确路径, 跨仓库 glob mmproj-*.gguf
  #    (失败仅告警, 不中断启动)
  if [[ -n "$PLAN_MMPROJ_INCLUDE" ]]; then
    echo ">> 未找到本地 mmproj, 下载 $MODEL_SOURCE_REPO -> $PLAN_MMPROJ_INCLUDE ..." >&2
    if [[ "$LLAMA_XET" == "1" ]]; then
      HF_HUB_ENABLE_XET=1 HF_TOKEN="${HF_TOKEN:-}" \
        hf download "$MODEL_SOURCE_REPO" --include "$PLAN_MMPROJ_INCLUDE" >/dev/null 2>&1 || true
    else
      HF_TOKEN="${HF_TOKEN:-}" \
        hf download "$MODEL_SOURCE_REPO" --include "$PLAN_MMPROJ_INCLUDE" >/dev/null 2>&1 || true
    fi
    mm="$(cache_find_file "$(hub_model_dir "$MODEL_SOURCE_REPO")" "$PLAN_MMPROJ_INCLUDE")"
    if [[ -n "$mm" ]]; then
      printf '%s' "$mm"
      return 0
    fi
  elif [[ -n "$fb_repo" ]]; then
    echo ">> 未找到本地 mmproj, 开始下载 $fb_repo (mmproj-*.gguf) ..." >&2
    if [[ "$LLAMA_XET" == "1" ]]; then
      HF_HUB_ENABLE_XET=1 HF_TOKEN="${HF_TOKEN:-}" \
        hf download "$fb_repo" --include 'mmproj-*.gguf' >/dev/null 2>&1 || true
    else
      HF_TOKEN="${HF_TOKEN:-}" \
        hf download "$fb_repo" --include 'mmproj-*.gguf' >/dev/null 2>&1 || true
    fi
    local snap
    snap="$(hub_snapshot_dir "$fb_repo")"
    if [[ -n "$snap" ]]; then
      mm="$(find -L "$snap" -maxdepth 3 -type f -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)"
    fi
    if [[ -z "$mm" ]]; then
      mm="$(find -L "$(hub_model_dir "$fb_repo")/snapshots" -type f -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)"
    fi
    if [[ -n "$mm" ]]; then
      printf '%s' "$mm"
      return 0
    fi
  fi

  echo "警告: mmproj 下载失败或未找到, 图片输入不可用(文本功能不受影响)。" >&2
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
    echo "ERROR: LLAMA_MODEL_REPO 不可为空(且 MODEL_REPO 亦未提供)。请设置模型来源(如 unsloth/Qwen3.8-Flash-Next-GGUF 或 hf://<org>/<repo>/<file>)。" >&2
    exit 1
  fi
  if [[ -z "$LLAMA_MODEL_NAME" ]]; then
    echo "ERROR: LLAMA_MODEL_NAME 不可为空(显式设置或从 LLAMA_MODEL_REPO 提取均无效)。" >&2
    exit 1
  fi
  case "$LLAMA_VISION" in
    auto|1|0) ;;
    *)
      echo "ERROR: LLAMA_VISION 必须是 auto、1 或 0，当前值: $LLAMA_VISION" >&2
      exit 1
      ;;
  esac

  check_deps
  # 自适应解析: 本地齐全则直接用, 否则按仓库清单推导布局并下载(填充全局 MODEL_FILE / PLAN_*)
  resolve_model_file
  # 别名: 未显式指定模型名时, 用清单推导出的真实文件名前缀(如 Qwen3.8-27B / Qwen3.8-Flash-Next)
  if [[ "$LLAMA_MODEL_NAME_EXPLICIT" != "1" && -n "$PLAN_NAME" ]]; then
    LLAMA_MODEL_ALIAS="${PLAN_NAME,,}"
  fi
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
  # CPU 线程数: 仅显式指定(非 0)时传入, 否则交给 llama.cpp 自行决定
  if [[ "$LLAMA_THREADS" != "0" ]]; then
    LAUNCH_CMD+=(-t "$LLAMA_THREADS")
  fi
  # /metrics: 默认开启, 供根目录 bench.py 采样 requests_processing / requests_deferred
  # (llama-server 默认不开放该端点); LLAMA_METRICS=0 可关闭
  if [[ "$LLAMA_METRICS" == "1" ]]; then
    LAUNCH_CMD+=(--metrics)
  fi
  LAUNCH_CMD+=("${SERVER_ARGS[@]}")

  echo "启动 llama 服务... (日志: ${LOG_FILE})" | tee -a "$LOG_FILE"
  echo ">> 加载模型: $MODEL_FILE" | tee -a "$LOG_FILE"
  if [[ -n "$MAIN_SNAP" ]]; then
    echo ">> 模型缓存快照: $MAIN_SNAP" | tee -a "$LOG_FILE"
  fi
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

  echo ">> 发送测试对话 (port=${PORT}) ..."
  # 模型名: 优先查询服务实际加载的模型(与启动时的 --alias 一致)。
  # do_test 是独立进程, 无 profile/解析上下文, LLAMA_MODEL_ALIAS 回退值可能带量化档后缀
  # 而与服务器不一致(llama-server 虽宽容处理, 但保持一致更稳)。
  local model_name="$LLAMA_MODEL_ALIAS"
  local models_json queried
  models_json="$(curl -s --max-time 5 "http://localhost:${PORT}/v1/models" "${AUTH[@]}" 2>/dev/null || true)"
  if [[ -n "$models_json" ]]; then
    queried="$(python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["data"][0]["id"])' <<<"$models_json" 2>/dev/null || true)"
    [[ -n "$queried" ]] && model_name="$queried"
  fi

  # max_tokens 给足余量: 思考型模型(如 Qwen3)会先消耗大量 token 在 reasoning_content 上,
  # 太小会导致 content 为空(只有思考过程)。超时也相应放宽。
  echo ">> 测试对话: model=${model_name}"
  curl -s --max-time 60 "http://localhost:${PORT}/v1/chat/completions" \
    "${AUTH[@]}" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${model_name}\", \"messages\": [{\"role\": \"user\", \"content\": \"你好, 请用一句话回复\"}], \"max_tokens\": 512}" \
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
