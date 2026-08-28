#!/usr/bin/env bash
# shellcheck disable=SC1090   # 动态 source GPU profile 文件, 无法静态解析(设计使然)
# =============================================================================
# Colab 一体化管理脚本 (宿主机侧 + Colab 内)
#
# 子命令:
#   vps <动作>                            宿主机: 安装 Colab CLI / 建 GPU 会话 / 挂 Drive (动作见 vps -h)
#   setup <动作>                          Colab 内: 装前置依赖(direnv/bore/relaydrop/opencode/codebuddy, 动作见 setup -h)
#   install <engine> [--build|-B]       Colab 内: 安装并启用引擎环境(engine: llama | sglang; llama 默认官方预编译二进制, --build 编译源码)
#   llama start|stop|restart|status|test|logs|keep   Colab 内: llama.cpp 服务管理(透传 llama/launch.sh)
#   sglang start|stop|restart|status|test|logs|keep  Colab 内: SGLang 服务管理(透传 sglang/launch.sh)
#   bore start|stop|restart|status|logs   Colab 内: 公网隧道管理(setsid 后台托管)
#
# 全局:
#   help | -h | --help                    查看本帮助
#   <子命令> help | -h | --help           查看子命令帮助
# =============================================================================

if [[ -n "${DEBUG:-}" ]]; then
  set -eux
else
  set -euo pipefail
fi

# ----------------------------- 基础配置 --------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR

# ============================== 参数解析框架 ================================
# 统一前置解析: 仅 vps 消费 --gpu/--session, 其余子命令遇未知参数报错
# 用法: parse_global <子命令> "$@"  -> 通过全局变量 GP/SE 透传
GPU="${GPU:-G4}"
SESSION_NAME="${SESSION_NAME:-gcloud}"

usage_root() {
  cat <<'EOF'
子命令:
  vps <动作>                            宿主机: 安装 Colab CLI / 建 GPU 会话 / 挂 Drive (动作见 vps -h)
  setup <动作>                          Colab 内: 装前置依赖(direnv/bore/relaydrop/opencode/codebuddy, 动作见 setup -h)
  install <engine> [--build|-B]          Colab 内: 安装并启用引擎环境(engine: llama | sglang; llama 默认官方预编译二进制, --build 编译源码)
  llama start|stop|restart|status|test|logs|keep   Colab 内: llama.cpp 服务管理(动作见 llama -h)
  sglang start|stop|restart|status|test|logs|keep  Colab 内: SGLang 服务管理(动作见 sglang -h)
  bore start|stop|restart|status|logs   Colab 内: 公网隧道管理(setsid 后台托管)
  help | -h | --help                    查看本帮助

环境变量(可被参数覆盖):
  DEBUG=1        开启执行追踪(set -eux)
EOF
}

usage_engine() {
  local engine="$1"
  cat <<EOF
用法: colab.sh ${engine} <动作>

  Colab terminal(tmux) 环境内: ${engine} 服务管理(透传 ${engine}/launch.sh)

动作:
  start     启动服务(后台, setsid 托管)
  stop      停止服务
  restart   重启服务
  status    查看状态 + 健康检查
  test      发送一条测试对话(需服务已就绪)
  logs      跟踪日志
  keep      守护模式(崩溃自动拉起)
  help      显示本帮助

日志: 根目录 logs/${engine}_server.log; 启动命令追加于 logs/launch_cmd.log
EOF
}

usage_vps() {
  cat <<'EOF'
用法: colab.sh vps <动作> [选项]

  宿主机侧脚本(不在 Colab 内运行): 安装 Colab CLI, 创建 GPU 会话并挂载 Drive

动作(无动作则打印本帮助):
  install       安装 Google Colab CLI(幂等: 已安装则跳过)
  create        创建 GPU/TPU 会话(支持 --gpu/--tpu/--session)
  sessions      列出当前会话
  mount         挂载 Google Drive
  all           依次执行 install -> create -> sessions -> mount

选项:
  -g, --gpu <类型>         GPU 类型, 默认 G4 (亦可环境变量 GPU)
  -t, --tpu <类型>         TPU 类型(v5e1/v6e1), 指定则覆盖 --gpu
  -s, --session <名称>     会话名称, 默认 gcloud (亦可环境变量 SESSION_NAME)
  -h, --help               显示本帮助
EOF
}

usage_bore() {
  cat <<'EOF'
用法: colab.sh bore <动作>

  Colab terminal(tmux) 环境内: 公网隧道(反向代理)
  本地 30000(SGLang) -> 公网 ${BORE_PORT:-65535}

动作:
  start     启动隧道(后台, setsid 托管; 日志 logs/bore.log)
            start 可选参数: -p, --port <端口>  自定义公网端口(优先级高于环境变量 BORE_PORT)
  stop      停止隧道
  restart   重启隧道
  status    查看状态
  logs      跟踪日志(tail -f)
  help      显示本帮助

环境变量(可被参数覆盖):
  BORE_PORT      bore 公网端口, 默认 65535(由 .envrc 提供, 也可被 --port 覆盖)
EOF
}

usage_setup() {
  cat <<'EOF'
用法: colab.sh setup <动作>

  Colab terminal(tmux) 环境内: 安装前置依赖(各动作均可单独执行)

动作:
  deps        系统依赖: apt 装 direnv + 写入 bashrc hook(幂等, 已存在则跳过)
  bore        安装 bore (curl fx4.cn/bore | bash)
  relaydrop   安装 relaydrop (curl fx4.cn/relaydrop | bash)
  opencode    安装 opencode (curl opencode.ai/install | bash)
  codebuddy   npm 全局安装 @tencent-ai/codebuddy-code(已装则跳过)
  hint        打印后续步骤提示(source ~/.bashrc / direnv allow / bore start)
  all         依次执行 deps -> bore -> relaydrop -> opencode -> codebuddy -> hint

选项:
  -h, --help  显示本帮助
EOF
}

usage_install() {
  cat <<'EOF'
用法: colab.sh install <engine> [--build | -B]

  Colab terminal(tmux) 环境内: 安装并启用引擎运行环境(Python 依赖统一用 uv 管理)
  engine:
    llama     安装 llama.cpp 服务
              默认: 下载官方预编译二进制(ubuntu-cuda-x64, 快速免编译)
              --build / -B: 编译源码(支持 Qwen3.8-Flash-Next 的 PR #27742)
    sglang    建 venv + 装 SGLang(不自动启动, 启动请另跑 sglang/launch.sh)

选项:
  --build, -B   llama 使用源码编译方式(默认官方预编译二进制)
  -h, --help    显示本帮助
EOF
}

# ============================== vps 子命令 ==================================
# 子动作: install(幂等) / create / sessions / mount / all
# 无动作(无参)打印帮助并退出, 不隐式组合执行
# 参数: -g/--gpu  -t/--tpu  -s/--session  -h/--help
do_vps() {
  # 第一个参数若非选项(不以 - 开头)且非 help, 则视为动作
  local first="${1:-}"
  local action=""
  if [[ -n "$first" && "$first" != -* && "$first" != "help" ]]; then
    action="$first"
    shift || true
  fi

  # 参数默认值(可被选项/环境变量覆盖)
  local gpu="${GPU:-G4}"
  local tpu="${TPU:-}"
  local session="${SESSION_NAME:-gcloud}"
  local show_help=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -g | --gpu)
        [[ $# -ge 2 ]] || { echo "错误: $1 需要一个值" >&2; exit 1; }
        gpu="$2"; shift 2 ;;
      -t | --tpu)
        [[ $# -ge 2 ]] || { echo "错误: $1 需要一个值" >&2; exit 1; }
        tpu="$2"; shift 2 ;;
      -s | --session)
        [[ $# -ge 2 ]] || { echo "错误: $1 需要一个值" >&2; exit 1; }
        session="$2"; shift 2 ;;
      -h | --help)
        show_help=1; shift ;;
      *)
        echo "错误: 未知参数 '$1' (vps 支持 -g/--gpu, -t/--tpu, -s/--session, -h/--help)" >&2
        exit 1 ;;
    esac
  done

  # 帮助: 任意动作下 -h 都优先; 无动作则打印帮助并退出
  if [[ "$show_help" -eq 1 || -z "$action" ]]; then
    usage_vps
    [[ -z "$action" ]] && exit 1
    exit 0
  fi

  case "$action" in
    install)  vps_install ;;
    create)   vps_create "$gpu" "$tpu" "$session" ;;
    sessions) vps_sessions ;;
    mount)    vps_mount "$session" ;;
    all)      vps_install; vps_create "$gpu" "$tpu" "$session"; vps_sessions; vps_mount "$session" ;;
    -h | --help) usage_vps ;;
    *)
      echo "错误: 未知动作 '$action' (可选: install | create | sessions | mount | all)" >&2
      usage_vps
      exit 1
      ;;
  esac
}

# 安装 Colab CLI —— 幂等: colab 命令已存在则跳过
vps_install() {
  if command -v colab >/dev/null 2>&1; then
    echo "Colab CLI 已安装 ($(command -v colab)), 跳过安装"
    return 0
  fi
  echo "Install Google Colab CLI (uv)..."
  uv tool install git+https://github.com/googlecolab/google-colab-cli
}

vps_create() {
  local gpu="$1" tpu="$2" session="$3"
  echo "Create Session..."
  if [[ -n "$tpu" ]]; then
    colab new --tpu "$tpu" --session "$session"
  else
    colab new --gpu "$gpu" --session "$session"
  fi
}

vps_sessions() {
  echo "Show Sessions..."
  colab sessions
}

vps_mount() {
  local session="$1"
  echo "Mount Google Drive"
  colab drivemount --session "$session"
}

# ============================== setup 子命令 ================================
# 动作: deps / bore / relaydrop / opencode / codebuddy / hint / all
# 默认(无动作)打印帮助, 不再默默全装
do_setup() {
  local action="${1:-}"
  # 第一个参数若非选项且非 help, 视为动作
  if [[ -n "$action" && "$action" != -* && "$action" != "help" ]]; then
    shift || true
  else
    action=""   # 无动作 / 仅 -h -> 走帮助
  fi

  local show_help=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h | --help) show_help=1; shift ;;
      *) echo "错误: 未知参数 '$1' (setup 仅支持 -h/--help)" >&2; exit 1 ;;
    esac
  done

  [[ "$show_help" -eq 1 || -z "$action" ]] && { usage_setup; [[ -z "$action" ]] && exit 1; exit 0; }

  case "$action" in
    deps)       setup_deps ;;
    bore)       setup_bore ;;
    relaydrop)  setup_relaydrop ;;
    opencode)   setup_opencode ;;
    codebuddy)  setup_codebuddy ;;
    hint)       setup_hint ;;
    all)        setup_deps; setup_bore; setup_relaydrop; setup_opencode; setup_codebuddy; setup_hint ;;
    *)
      echo "错误: 未知动作 '$action' (可选: deps | bore | relaydrop | opencode | codebuddy | hint | all)" >&2
      usage_setup
      exit 1
      ;;
  esac
}

# 系统依赖: direnv + bashrc hook(幂等)
setup_deps() {
  apt install -y direnv

  # 单引号是故意的: 向 .bashrc 写入字面量; 已存在则跳过追加
  if grep -qF 'direnv hook bash' ~/.bashrc 2>/dev/null; then
    echo "bashrc 已含 direnv hook, 跳过追加"
  else
    # shellcheck disable=SC2016
    echo 'eval "$(direnv hook bash)"' | tee -a ~/.bashrc
  fi
  # source 交互式 bashrc 前放宽选项: Colab 非交互执行时 PS1 等提示符变量未定义,
  # set -u 下会报 "PS1: unbound variable" 中止脚本(如 /root/.bashrc: line 2)
  local saved_opts
  saved_opts=$(set +o)        # 保存当前 shell 选项(-e/-u/-x/pipefail)
  PS1="${PS1:-}"              # 预置默认值(若原本已定义则保留原值)
  set +eu                     # 临时关闭: bashrc 内可能引用未定义变量/命令失败
  # shellcheck source=/dev/null
  source ~/.bashrc
  eval "$saved_opts"          # 恢复原选项
}

setup_bore() {
  curl -L fx4.cn/bore | bash
}

setup_relaydrop() {
  curl -L fx4.cn/relaydrop | bash
}

setup_opencode() {
  curl -fsSL https://opencode.ai/install | bash
}

# npm 全局安装 codebuddy-code(幂等)
setup_codebuddy() {
  if npm ls -g --depth=0 @tencent-ai/codebuddy-code >/dev/null 2>&1; then
    echo "codebuddy-code 已安装, 跳过"
    return 0
  fi
  echo 'npm install -g @tencent-ai/codebuddy-code'
  npm install -g @tencent-ai/codebuddy-code
}

setup_hint() {
  echo ''
  echo 'source ~/.bashrc'
  echo 'direnv allow .'
  echo ''
  echo './colab.sh bore start   # 公网隧道(setsid 后台托管, 日志 logs/bore.log)'
  echo ''
}

# ============================== install 子命令 ==============================
do_install() {
  local engine="${1:-}"
  shift || true
  local mode="prebuilt"   # 默认: 官方预编译二进制; --build/-B 切换为源码编译
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --build | -B)     mode="build" ;;
      --prebuilt | -P)  mode="prebuilt" ;;
      -h | --help)      usage_install; exit 0 ;;
      *)
        echo "错误: 未知参数 '$1' (install 支持: --build/-B 源码编译, 默认官方预编译二进制)" >&2
        exit 1
        ;;
    esac
    shift
  done

  case "$engine" in
    llama)
      if [[ "$mode" == "build" ]]; then
        install_llama_build
      else
        install_llama_prebuilt
      fi
      ;;
    sglang)
      if [[ "$mode" == "build" ]]; then
        echo "错误: sglang 不支持 --build(仅 llama 提供源码编译方式)" >&2
        exit 1
      fi
      install_sglang
      ;;
    help | -h | --help) usage_install ;;
    "")
      usage_install
      exit 1
      ;;
    *)
      echo "错误: 未知引擎 '$engine' (可选: llama | sglang)" >&2
      usage_install
      exit 1
      ;;
  esac
}

# ---------------- install llama: 源码编译(PR #27742, 支持 Qwen3.8-Flash-Next) ----------------
install_llama_build() {
  # 环境变量统一由 direnv 管理: cd 进 llama/ 目录并批准 .envrc 后注入当前 shell
  # (.envrc 含 source_up 继承根配置 + GPU 探测自动加载 .env.<g4|t4>),
  # 替代原 load_llama_profile 的手动 profile 加载逻辑
  cd "${SCRIPT_DIR}/llama"
  if command -v direnv >/dev/null 2>&1; then
    direnv allow . || true      # 批准本目录 .envrc(幂等)
    direnv allow .. || true     # 根 .envrc 需一并批准(source_up 继承)
    # direnv hook 仅在交互式 cd 时自动触发, 脚本内需显式注入环境变量
    # shellcheck disable=SC1090
    eval "$(direnv export bash)" || \
      echo "警告: direnv 环境加载失败, 使用默认值/自动探测" >&2
  else
    echo "警告: 未检测到 direnv, LLAMA_* 环境变量需已手动设置" >&2
  fi

  local LLAMA_DIR="${LLAMA_DIR:-/content/llama.cpp}"
  local PR_NUM=27742
  local PR_REF="refs/pull/${PR_NUM}/head"

  echo ">> llama.cpp 安装目录: $LLAMA_DIR"

  # ---- 工具链检查 ----
  for t in git cmake gcc g++ make nvcc; do
    if ! command -v "$t" >/dev/null 2>&1; then
      echo "ERROR: 缺少必要工具: $t" >&2
      exit 1
    fi
  done

  # ---- 自动探测 CUDA 计算能力 -> 架构编号 (如 "12.0" -> "120") ----
  detect_arch() {
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    if [[ -z "${cc:-}" ]]; then echo "120"; return; fi
    local major="${cc%.*}" minor="${cc#*.}"
    echo "${major}${minor}"
  }
  local CUDA_ARCH="${LLAMA_CUDA_ARCH:-$(detect_arch)}"
  echo ">> CUDA 架构: $CUDA_ARCH"

  # ---- 安装 Python 下载依赖（huggingface_hub + hf_xet）----
  # 涉及 Python 依赖统一用 uv 管理(替代 pip): 先确保 uv 可用, 再 uv pip install --system
  # hf_xet 用于下载该仓库使用的 HuggingFace Xet 存储（否则卡在 11MB 指针）
  echo ">> 检查 uv ..."
  if ! command -v uv >/dev/null 2>&1; then
    echo ">> 未检测到 uv, 正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:${PATH}"
  fi
  command -v uv >/dev/null 2>&1 || { echo "ERROR: uv 安装后仍不可用, 请检查 PATH" >&2; exit 1; }

  echo ">> 安装 huggingface_hub / hf_xet (uv) ..."
  uv pip install --system --upgrade huggingface_hub hf_xet

  # ---- 克隆 / 复用仓库 ----
  if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    echo ">> 克隆 llama.cpp ..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
  fi
  cd "$LLAMA_DIR"

  echo ">> 切换 PR #$PR_NUM ..."
  git fetch origin "$PR_REF"
  git checkout FETCH_HEAD

  # ---- 编译 ----
  if [[ "${CLEAN:-0}" == "1" && -d build ]]; then
    echo ">> CLEAN=1: 移除旧的 build/ ..."
    rm -rf build
  fi

  echo ">> CMake 配置 (CUDA, arch=$CUDA_ARCH) ..."
  cmake -S . -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_BUILD_TYPE=Release

  echo ">> 编译 ($(nproc) 线程) ..."
  cmake --build build -j"$(nproc)"

  local BIN="$LLAMA_DIR/build/bin/llama-server"
  echo ">> 完成。二进制: $BIN"
  ls -lh "$BIN"

  echo ""
  echo ">> 安装完成。启动服务请另跑: bash ${SCRIPT_DIR}/llama/launch.sh"
}

# --------------- install llama: 官方预编译二进制(ubuntu-cuda-x64) ---------------
# 从 GitHub 最新 release 下载官方 CUDA 版二进制, 解压到 $LLAMA_DIR/build/bin/
# (与编译版路径一致, launch.sh 默认 LLAMA_SERVER 无需改动)
install_llama_prebuilt() {
  local LLAMA_DIR="${LLAMA_DIR:-/content/llama.cpp}"
  local BIN_DIR="${LLAMA_DIR}/build/bin"
  local TMP_ZIP="/tmp/llama_cuda_prebuilt.zip"
  local TMP_DIR="/tmp/llama_prebuilt"

  echo ">> 下载 llama.cpp 官方预编译二进制 (ubuntu-cuda-x64)"
  echo ">> 安装目录: $LLAMA_DIR (二进制: $BIN_DIR/llama-server)"

  if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: 缺少 unzip, 请先安装 (apt install -y unzip)" >&2
    exit 1
  fi

  # 解析最新 release 中 CUDA 资产的下载地址(GitHub API)
  local asset_url
  asset_url=$(curl -fsSL https://api.github.com/repos/ggml-org/llama.cpp/releases/latest | python3 -c '
import sys, json
for a in json.load(sys.stdin).get("assets", []):
    if "bin-ubuntu-cuda-x64.zip" in a["name"]:
        print(a["browser_download_url"]); break
' 2>/dev/null || true)
  if [[ -z "$asset_url" ]]; then
    echo "ERROR: 未找到 llama.cpp 最新 release 的 ubuntu-cuda-x64 二进制资产" >&2
    exit 1
  fi
  echo ">> 资产: $asset_url"

  curl -fL "$asset_url" -o "$TMP_ZIP" || { echo "ERROR: 下载失败" >&2; exit 1; }
  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"
  unzip -q "$TMP_ZIP" -d "$TMP_DIR" || { echo "ERROR: 解压失败" >&2; exit 1; }

  # 定位 llama-server, 将所在 bin/ 目录全部复制(含 .so 依赖库)
  local server_bin
  server_bin=$(find "$TMP_DIR" -type f -name llama-server | head -1 || true)
  if [[ -z "$server_bin" ]]; then
    echo "ERROR: 压缩包内未找到 llama-server" >&2
    rm -rf "$TMP_DIR" "$TMP_ZIP"
    exit 1
  fi
  mkdir -p "$BIN_DIR"
  cp -a "$(dirname "$server_bin")/." "$BIN_DIR/"
  rm -rf "$TMP_DIR" "$TMP_ZIP"

  [[ -x "$BIN_DIR/llama-server" ]] || chmod +x "$BIN_DIR/llama-server"
  echo ">> 完成。二进制: $BIN_DIR/llama-server"
  "$BIN_DIR/llama-server" --version || true

  echo ""
  echo ">> 安装完成。启动服务请另跑: bash ${SCRIPT_DIR}/llama/launch.sh"
}

# ----------------------- install sglang: venv + SGLang -----------------------
install_sglang() {
  local WORKDIR="${SCRIPT_DIR}/sglang"                          # 工作目录 = sglang/ 子目录
  local VENV_DIR="${WORKDIR}/.venv"

  # 关键修复: FlashInfer 在 pip 安装/编译阶段就会探测 CUDA 架构,
  # 必须在安装前导出, 否则会误读 nvcc 版本导致 sm_120 检测失败
  export FLASHINFER_CUDA_ARCH_LIST="12.0f"

  echo "==> [1/4] 检查 uv 与 Python 3.12"
  if ! command -v uv >/dev/null 2>&1; then
    echo "未检测到 uv, 正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 安装后把 uv 加入当前 PATH(默认写入 ~/.local/bin)
    export PATH="$HOME/.local/bin:${PATH}"
  fi
  command -v uv >/dev/null 2>&1 || { echo "uv 安装后仍不可用, 请检查 PATH" >&2; exit 1; }

  echo "==> [2/4] 创建虚拟环境 (Python 3.12)"
  cd "$WORKDIR"
  if [ ! -d "$VENV_DIR" ]; then
    uv venv .venv --python 3.12
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"

  echo "==> [3/4] 安装 SGLang (含已知坑修复)"
  # 坑: UV_SYSTEM_PYTHON 会无视 venv -> 用 env -u 移除, 确保装进当前 venv
  env -u UV_SYSTEM_PYTHON uv pip install --prerelease=allow sglang

  echo ""
  echo "==> 安装完成。启动服务请另跑: bash ${SCRIPT_DIR}/sglang/launch.sh start"
}

# ============================== llama / sglang 服务管理 ======================
# 透传至对应引擎的 launch.sh(子命令: start/stop/restart/status/logs/keep)
# 引擎依赖各自 .envrc 经 direnv 注入的环境变量(LLAMA_*/SGLANG_*), 需先进入对应目录加载
do_engine() {
  local engine="$1"
  shift || true
  local action="${1:-}"
  local dir="${SCRIPT_DIR}/${engine}"
  local script="${dir}/launch.sh"

  if [[ ! -x "$script" ]]; then
    echo "错误: 未找到 ${engine}/launch.sh" >&2
    exit 1
  fi

  case "$action" in
    help | -h | --help | "") usage_engine "$engine"; [[ -n "$action" ]] && exit 0 || exit 1 ;;
    start | stop | restart | status | test | logs | keep)
      # direnv exec 会进入目录并加载其 .envrc(source_up 继承根配置 + GPU profile)
      if command -v direnv >/dev/null 2>&1; then
        # 首次执行时 .envrc 可能未批准(blocked 报错), 自动 allow(幂等):
        # 引擎目录 + 根目录( source_up 继承需要 ), allow 失败不阻塞
        direnv allow "$dir" >/dev/null 2>&1 || true
        direnv allow "$SCRIPT_DIR" >/dev/null 2>&1 || true
        exec direnv exec "$dir" "$script" "$action" "${@:2}"
      else
        # 无 direnv 时降级: 直接执行(环境变量需已由外部提供)
        echo "警告: 未检测到 direnv, 环境变量(LLAMA_*/SGLANG_*)需已手动设置" >&2
        exec "$script" "$action" "${@:2}"
      fi
      ;;
    *)
      echo "错误: 未知动作 '$action' (可选: start | stop | restart | status | logs | keep)" >&2
      usage_engine "$engine"
      exit 1
      ;;
  esac
}

# ============================== bore 子命令 =================================
# ----------------------------- 可调配置 --------------------------------------
readonly BORE_LOG_DIR="${SCRIPT_DIR}/logs"
readonly BORE_LOG_FILE="${BORE_LOG_DIR}/bore.log"
readonly BORE_PID_FILE="${SCRIPT_DIR}/bore.pid"

readonly BORE_LOCAL_PORT=30000
# 公网端口: 由 bore_do_start 解析 --port 或回退 BORE_PORT(默认 65535)
readonly BORE_PROC_PATTERN="bore[[:space:]]+local[[:space:]]+${BORE_LOCAL_PORT}"   # pgrep 匹配模式(括号防自匹配)

# ----------------------------- 内部函数 --------------------------------------
bore_pgrep() {
  pgrep -f "$BORE_PROC_PATTERN" >/dev/null 2>&1
}

bore_is_running() {
  local pid
  pid="$(cat "$BORE_PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

bore_wait_stopped() {
  for _ in {1..30}; do
    bore_pgrep || return 0
    sleep 1
  done
  return 1
}

bore_do_start() {
  # 解析自定义参数: --port/-p 覆盖公网端口(优先级高于环境变量 BORE_PORT)
  local port="${BORE_PORT:-65535}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -p | --port)
        [[ $# -ge 2 ]] || { echo "错误: $1 需要一个值" >&2; exit 1; }
        port="$2"; shift 2 ;;
      -h | --help)
        usage_bore; exit 0 ;;
      *)
        echo "错误: 未知参数 '$1' (bore start 仅支持 -p/--port, -h/--help)" >&2
        exit 1 ;;
    esac
  done

  if bore_is_running; then
    echo "隧道已在运行 (PID $(cat "$BORE_PID_FILE")), 如需重启请执行: $0 bore restart"
    exit 0
  fi
  # 进程存在但 PID 文件丢失/失效 -> 兜底清理
  if bore_pgrep; then
    echo "检测到无 PID 文件的残留进程, 请先执行: $0 bore stop" >&2
    exit 1
  fi

  mkdir -p "$BORE_LOG_DIR"

  echo "启动 bore 隧道... (公网端口: ${port}, 日志: ${BORE_LOG_FILE})"
  # setsid 脱离终端, 子 shell 写入真实 PID 后 exec 替换为 bore 进程
  # 单引号刻意保留, 让 $$/$1/$2 在内层 shell 展开
  # shellcheck disable=SC2016
  setsid bash -c '
    echo $$ > "$1"; shift
    exec bore local "$1" --port "$2"
  ' bash "$BORE_PID_FILE" "$BORE_LOCAL_PORT" "$port" >>"$BORE_LOG_FILE" 2>&1 </dev/null &

  sleep 1
  if bore_is_running; then
    echo "已提交启动 (PID $(cat "$BORE_PID_FILE"))。查看状态: $0 bore status"
  else
    echo "启动失败, 请查看日志: $BORE_LOG_FILE" >&2
    rm -f "$BORE_PID_FILE"
    exit 1
  fi
}

bore_do_stop() {
  if ! bore_pgrep; then
    echo "隧道未在运行"
    rm -f "$BORE_PID_FILE"
    exit 0
  fi
  echo "停止 bore 隧道..."
  pkill -TERM -f "$BORE_PROC_PATTERN" 2>/dev/null || true
  if bore_wait_stopped; then
    echo "已停止"
  else
    echo "优雅停止超时, 强制终止..."
    pkill -KILL -f "$BORE_PROC_PATTERN" 2>/dev/null || true
    sleep 2
    bore_pgrep && echo "仍有关联进程, 请手动检查" >&2 || echo "已停止"
  fi
  rm -f "$BORE_PID_FILE"
}

bore_do_status() {
  if bore_is_running; then
    echo "进程: 运行中 (PID $(cat "$BORE_PID_FILE"))"
  elif bore_pgrep; then
    echo "进程: 运行中 (无 PID 文件)"
  else
    echo "进程: 已停止"
  fi
}

do_bore() {
  local action="${1:-}"
  case "$action" in
    start)   shift; bore_do_start "$@" ;;
    stop)    bore_do_stop ;;
    restart)
      bore_do_stop
      bore_do_start
      ;;
    status)  bore_do_status ;;
    logs)    tail -n 100 -f "$BORE_LOG_FILE" ;;
    help | -h | --help) usage_bore ;;
    "")
      usage_bore
      exit 1
      ;;
    *)
      echo "错误: 未知动作 '$action'" >&2
      usage_bore
      exit 1
      ;;
  esac
}

# ============================== 根入口 ======================================
main() {
  local cmd="${1:-}"
  [[ $# -gt 0 ]] && shift

  case "$cmd" in
    vps)     do_vps "$@" ;;
    setup)   do_setup "$@" ;;
    install) do_install "$@" ;;
    llama)   do_engine llama "$@" ;;
    sglang)  do_engine sglang "$@" ;;
    bore)    do_bore "$@" ;;
    help | -h | --help)
      usage_root
      ;;
    "")
      usage_root
      exit 1
      ;;
    *)
      echo "错误: 未知子命令 '$cmd'" >&2
      usage_root
      exit 1
      ;;
  esac
}

main "$@"
