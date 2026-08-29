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
#   LLAMA_MODEL_NAME  模型名(未设置时从 REPO 提取: / 后部分去 -GGUF); 服务 --alias 为其小写形式
#                     未显式设置时, 别名改用"仓库文件清单推导出的真实前缀"(见下方自适应说明)
#   LLAMA_QUANT       量化档(不可为空, 无默认; 由 .envrc / gpu profile 提供)
#   LLAMA_MODEL_ROOT  模型基础盘前缀(未设置回退 MODEL_ROOT, 再兜底 /content/models; 换持久化盘只改这一层)
#                     不支持 Google Drive(/content/drive): 指向 Drive 的路径会在启动时报错
#   LLAMA_MODEL_DIR   本仓库模型目录(默认 <ROOT>/<repo名>, 按仓库隔离; 显式设置时原样使用)
#   LLAMA_DIR       安装目录(默认 /content/llama.cpp; 由 .envrc 导出, 可覆盖)
#   LLAMA_SERVER     llama 二进制路径(默认从 PATH 查找 command -v llama; 未命中回退 <LLAMA_DIR>/build/bin)
#   LLAMA_HOST / LLAMA_PORT   监听地址与端口(内部变量 HOST/PORT, 默认 0.0.0.0 / 30000)
#   LLAMA_API_KEY     服务器 API 密钥(未设置时回退 API_KEY)
#   LLAMA_XET         1=启用 HF Xet 存储(默认), 0=禁用
#   LLAMA_METRICS     1=开放 /metrics 端点(默认, 供根目录 bench.py 采样并发), 0=禁用
#   LLAMA_VISION      auto=按模型能力自动检测, 1=强制尝试启用, 0=禁用(默认 0)
#   LLAMA_MMPROJ      视觉投影器 mmproj 路径(可选; 缺省自动检测模型目录 mmproj-*.gguf, 再按需自动下载)
#   LLAMA_MMPROJ_REPO mmproj 自动下载源(默认同 LLAMA_MODEL_REPO; 空=禁用自动下载)
#
# 自适应解析(不写死任何路径形式):
#   先扫描 LLAMA_MODEL_DIR(深度 2), 本地分片齐全则直接启动; 否则拉取仓库文件清单,
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

# GPU profile 兜底加载(不依赖 direnv):
# 环境变量一律继承调用方 shell(外层 direnv hook 注入)。若调用方未 cd 进本目录(引擎 .envrc
# 未加载), LLAMA_QUANT / LLAMA_MODEL_REPO 等引擎专属默认值会缺失, 这里按 GPU_PROFILE
# (未设置时按 nvidia-smi 探测)直接 source 本目录的 .env.<g4|t4>。
# profile 是纯 bash(只有 export VAR="${VAR:-默认}"), 不含 direnv stdlib, 可安全 source:
# 幂等且不覆盖已有值(direnv 已加载过再 source 一次无副作用)。
load_gpu_profile() {
  local profile="${GPU_PROFILE:-}"
  if [[ -z "$profile" ]]; then
    local gpu
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
    case "$gpu" in
      *T4*)                  profile=t4 ;;
      *G4*|*L4*|*Blackwell*) profile=g4 ;;   # Blackwell 系(如 RTX PRO 6000)按 G4 处理
    esac
  fi
  [[ -n "$profile" ]] || return 0
  local f="${SCRIPT_DIR}/.env.${profile}"
  [[ -r "$f" ]] || return 0
  echo ">> [llama] 加载 GPU profile: ${profile} (${f})" >&2
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
# 仓库名 = LLAMA_MODEL_REPO 取 / 后部分(保留 -GGUF 后缀), 供目录与模型名前缀复用
#   例: unsloth/Qwen3.8-Flash-Next-GGUF -> Qwen3.8-Flash-Next-GGUF
readonly LLAMA_REPO_NAME="${LLAMA_MODEL_REPO##*/}"
# 模型名前缀(用于本地分片文件匹配)
# 未显式设置时, 从仓库名去除末尾 -GGUF 后缀
#   例: Qwen3.8-Flash-Next-GGUF -> Qwen3.8-Flash-Next
if [[ -z "${LLAMA_MODEL_NAME:-}" ]]; then
  LLAMA_MODEL_NAME="${LLAMA_REPO_NAME%-GGUF}"   # 去除末尾 -GGUF 后缀
  readonly LLAMA_MODEL_NAME_EXPLICIT=0          # 未显式设置: 别名以清单推导结果为准
else
  readonly LLAMA_MODEL_NAME_EXPLICIT=1          # 显式设置: 别名与匹配提示均以其为准
fi
readonly LLAMA_MODEL_NAME
# 服务别名: 模型名转小写。未显式设置 LLAMA_MODEL_NAME 时, do_start 内改用清单推导的 PLAN_NAME
# 例: Qwen3.8-Flash-Next -> qwen3.8-flash-next
LLAMA_MODEL_ALIAS="${LLAMA_MODEL_NAME,,}"
# 无默认值: 必须由外部提供(.envrc / gpu profile / 命令行); 此处仅声明空以规避 set -u
readonly LLAMA_QUANT="${LLAMA_QUANT:-}"
# 模型路径(两级: 基础盘前缀 + 本仓库目录)
#   LLAMA_MODEL_ROOT  基础盘前缀(三级优先级, 见下方解析; 换持久化盘只改这一层)
#   LLAMA_MODEL_DIR   本仓库模型目录(默认 <ROOT>/<repo名>, 仓库名保留 -GGUF 后缀)
#     显式设置 LLAMA_MODEL_DIR 时原样使用, 不再拼接 ROOT(保留"自定义任意目录"的自由度)
#   按仓库分目录是刻意的隔离, 不是简单前缀: 脚本会扫描该目录(深度 2)判断分片是否齐全,
#   若把公共根(如 /content/models)直接当 MODEL_DIR, 其它仓库同量化档的文件会被判为
#   "本地已齐全"而误加载, mmproj 也可能抓到别模型的投影器。
#   目录内按仓库真实结构存放:
#     子目录布局 -> <dir>/<QUANT>/xxx.gguf ; 根目录布局 -> <dir>/xxx.gguf
#     例: Qwen3.8-Flash-Next-GGUF -> /content/models/Qwen3.8-Flash-Next-GGUF/{UD-Q4_K_XL/...}
# 基础盘前缀解析(显式逐级判定, 便于看清优先级; 字面量仅作最后兜底):
#   1) LLAMA_MODEL_ROOT  引擎专属, 优先级最高
#   2) MODEL_ROOT        两引擎共用(根 .envrc 导出), 换持久化盘改这一处两引擎同时生效
#   3) /content/models   兜底默认值, 仅在上述两者均未设置(或为空)时使用
# "已设置" = 变量存在且非空(空串等同未设置, 继续回退); 来源记入 MODEL_ROOT_SOURCE 便于排查
MODEL_ROOT_SOURCE=""
if [[ -n "${LLAMA_MODEL_ROOT:-}" ]]; then
  MODEL_ROOT_SOURCE="LLAMA_MODEL_ROOT"
elif [[ -n "${MODEL_ROOT:-}" ]]; then
  LLAMA_MODEL_ROOT="$MODEL_ROOT"
  MODEL_ROOT_SOURCE="MODEL_ROOT"
else
  LLAMA_MODEL_ROOT="/content/models"
  MODEL_ROOT_SOURCE="默认值(兜底)"
fi
readonly LLAMA_MODEL_ROOT MODEL_ROOT_SOURCE
# 不支持 Google Drive 作为模型目录: Drive 是 FUSE 挂载, mmap 随机读极慢, 且不提供任何
# 冷存储/复制降级 —— 指向 /content/drive 的路径在启动时直接报错(见 do_start 内校验)。
LLAMA_MODEL_DIR="${LLAMA_MODEL_DIR:-$LLAMA_MODEL_ROOT/$LLAMA_REPO_NAME}"
readonly LLAMA_MODEL_DIR
# 注: HF_ENDPOINT(huggingface_hub 通用变量, 镜像站可用)不在此声明, 避免 readonly 后无法透传
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
# Prometheus 指标端点 /metrics: 1=启用(默认, 供根目录 bench.py 采样并发), 0=禁用
readonly LLAMA_METRICS="${LLAMA_METRICS:-1}"
# 多模态视觉投影器(mmproj): 用于图片/视频输入(可选, 缺省不启用)
#   LLAMA_VISION      auto=按模型能力自动检测, 1=跳过检测并尝试启用, 0=完全禁用
#   LLAMA_MMPROJ       显式指定 mmproj 文件路径(优先级最高; 不设置时自动在模型目录检测 mmproj-*.gguf)
#   LLAMA_MMPROJ_REPO  自动下载源(默认同 LLAMA_MODEL_REPO; 设为空字符串则禁用自动下载)
readonly LLAMA_VISION="${LLAMA_VISION:-0}"
readonly LLAMA_MMPROJ="${LLAMA_MMPROJ:-}"
readonly LLAMA_MMPROJ_REPO="${LLAMA_MMPROJ_REPO:-${LLAMA_MODEL_REPO:-}}"

# 模型解析结果(由 resolve_model_file -> plan_model 填充; 此处先声明空值规避 set -u)
PLAN_NAME=""              # 由真实文件名推导的模型名前缀(用于 --alias)
PLAN_MMPROJ_INCLUDE=""    # 需下载的 mmproj 仓库内相对路径(无则空)
MODEL_FILE=""             # 首个分片(或单文件)的本地绝对路径

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

# 自适应解析模型布局: 不写死路径形式。
# 先扫描 LLAMA_MODEL_DIR(深度 2)判断本地分片是否齐全; 不齐全才联网拉仓库文件清单,
# 按 LLAMA_QUANT 推导布局(<QUANT>/ 子目录 | 根目录; 单文件 | 分片)并给出下载清单。
# 输出 KEY=value(值经 shlex 转义, 供 eval):
#   PLAN_ACTION         local=本地已齐全, download=需下载
#   PLAN_MODEL_FILE     首个分片(或单文件)的本地绝对路径
#   PLAN_DIR / PLAN_NAME / PLAN_TOTAL / PLAN_HAVE / PLAN_INCLUDE / PLAN_MMPROJ_INCLUDE
plan_model() {
  REPO="$LLAMA_MODEL_REPO" QUANT="$LLAMA_QUANT" MODEL_DIR="$LLAMA_MODEL_DIR" \
  NAME_HINT="$LLAMA_MODEL_NAME" VISION="$LLAMA_VISION" MMPROJ_REPO="$LLAMA_MMPROJ_REPO" \
  HF_TOKEN="${HF_TOKEN:-}" HF_ENDPOINT="${HF_ENDPOINT:-}" \
  python3 - <<'PY'
import json, os, re, shlex, sys, urllib.error, urllib.request

REPO        = os.environ.get("REPO", "")
QUANT       = os.environ.get("QUANT", "")
MODEL_DIR   = os.environ.get("MODEL_DIR", "")
NAME_HINT   = os.environ.get("NAME_HINT", "")
VISION      = os.environ.get("VISION", "0")
MMPROJ_REPO = os.environ.get("MMPROJ_REPO", "")
TOKEN       = os.environ.get("HF_TOKEN", "")
ENDPOINT    = os.environ.get("HF_ENDPOINT", "").rstrip("/") or "https://huggingface.co"
MAX_DEPTH   = 2      # 本地扫描深度: 覆盖 <dir>/<file> 与根目录两种布局
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

def abs_path(rel):
    return os.path.normpath(os.path.join(MODEL_DIR, *rel.split("/")))

local_rels = scan_local(MODEL_DIR)
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
emit("PLAN_MODEL_FILE", abs_path(chosen["files"][1]))
# mmproj: 仅当与主模型同仓库时才能用清单给出精确路径(跨仓库由 bash 侧回退 glob 下载)
mm_inc = ""
if not pick_mmproj(local_rels) and VISION != "0" and MMPROJ_REPO \
        and MMPROJ_REPO.lower() == REPO.lower():
    mm_inc = pick_mmproj(remote()) or ""
emit("PLAN_MMPROJ_INCLUDE", mm_inc)
PY
}

# 解析并(按需)下载模型, 结果写入全局: MODEL_FILE + PLAN_*(见 plan_model 注释)
resolve_model_file() {
  mkdir -p "$LLAMA_MODEL_DIR"
  local plan
  if ! plan="$(plan_model)"; then
    echo "ERROR: 模型解析失败(见上方输出)。" >&2
    exit 1
  fi
  eval "$plan"   # shellcheck disable=SC2091  # 值由 python shlex 转义

  local layout="${PLAN_DIR:-<仓库根目录>}"
  if [[ "$PLAN_ACTION" == "download" ]]; then
    echo ">> 仓库布局: $layout (${PLAN_NAME}-${LLAMA_QUANT}, ${PLAN_TOTAL} 分片)" >&2
    echo ">> 本地分片 ${PLAN_HAVE}/${PLAN_TOTAL}, 开始下载 $LLAMA_MODEL_REPO ..." >&2
    echo "   (该仓库使用 Xet 存储，需要 hf_xet；install.sh 已安装)" >&2
    local -a includes=()
    local line
    while IFS= read -r line; do
      [[ -n "$line" ]] && includes+=("$line")
    done <<<"$PLAN_INCLUDE"
    # VISION=1: 跳过能力检测, mmproj 与主模型一并下载(同仓库同 local-dir, 无额外请求)
    if [[ "$LLAMA_VISION" == "1" && -n "$PLAN_MMPROJ_INCLUDE" ]]; then
      includes+=("$PLAN_MMPROJ_INCLUDE")
    fi
    local -a dl=(hf download "$LLAMA_MODEL_REPO" --local-dir "$LLAMA_MODEL_DIR")
    local inc
    for inc in "${includes[@]}"; do
      dl+=(--include "$inc")
    done
    echo ">> 下载 ${#includes[@]} 个文件 -> $LLAMA_MODEL_DIR" >&2
    if [[ "$LLAMA_XET" == "1" ]]; then
      HF_HUB_ENABLE_XET=1 HF_TOKEN="$HF_TOKEN" "${dl[@]}"
    else
      HF_TOKEN="$HF_TOKEN" "${dl[@]}"
    fi
  else
    echo ">> 本地模型已齐全 (${PLAN_HAVE}/${PLAN_TOTAL} 分片, 布局: $layout), 跳过下载。" >&2
  fi

  MODEL_FILE="$PLAN_MODEL_FILE"
  if [[ -z "$MODEL_FILE" || ! -f "$MODEL_FILE" ]]; then
    echo "ERROR: 未找到模型文件: ${MODEL_FILE:-(未推导)}。下载可能失败, 请查看日志。" >&2
    exit 1
  fi
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

# 定位多模态视觉投影器(mmproj)。优先级: 显式 LLAMA_MMPROJ > 本地自动检测(深度 2)
#   > 按仓库清单推导的精确路径下载(同仓库) / glob 下载(跨仓库)。
# LLAMA_VISION=auto 时仅在主模型声明支持视觉时启用(见 model_supports_vision),
# 避免把不相干的 mmproj 传给纯文本模型导致启动/请求失败。
# LLAMA_VISION=1 跳过能力检测并尝试启用; LLAMA_VISION=0 直接禁用 mmproj。
# 未找到时输出空字符串(不报错): 文本服务照常可用, 仅图片/视频输入不可用。
# 仅当显式指定的 LLAMA_MMPROJ 路径不存在时返回非零(启动失败; LLAMA_VISION=0 除外)。
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
  # 2) 模型根目录自动检测(深度 2: 覆盖根目录布局与 <QUANT>/ 子目录布局)
  mm=$(find "$LLAMA_MODEL_DIR" -maxdepth 2 -type f -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)
  if [[ -n "$mm" ]]; then
    printf '%s' "$mm"
    return 0
  fi

  # 3) 按清单推导的精确路径下载(同仓库); 跨仓库时清单不覆盖, 回退 glob 匹配
  #    (失败仅告警, 不中断启动)
  if [[ -n "$PLAN_MMPROJ_INCLUDE" ]]; then
    echo ">> 未找到本地 mmproj, 下载 $LLAMA_MMPROJ_REPO -> $PLAN_MMPROJ_INCLUDE ..." >&2
    if [[ "$LLAMA_XET" == "1" ]]; then
      HF_HUB_ENABLE_XET=1 HF_TOKEN="$HF_TOKEN" \
        hf download "$LLAMA_MMPROJ_REPO" --include "$PLAN_MMPROJ_INCLUDE" --local-dir "$LLAMA_MODEL_DIR" >/dev/null 2>&1 || true
    else
      HF_TOKEN="$HF_TOKEN" \
        hf download "$LLAMA_MMPROJ_REPO" --include "$PLAN_MMPROJ_INCLUDE" --local-dir "$LLAMA_MODEL_DIR" >/dev/null 2>&1 || true
    fi
  elif [[ -n "$LLAMA_MMPROJ_REPO" ]]; then
    echo ">> 未找到本地 mmproj, 开始下载 $LLAMA_MMPROJ_REPO (mmproj-*.gguf) ..." >&2
    if [[ "$LLAMA_XET" == "1" ]]; then
      HF_HUB_ENABLE_XET=1 HF_TOKEN="$HF_TOKEN" \
        hf download "$LLAMA_MMPROJ_REPO" --include 'mmproj-*.gguf' --local-dir "$LLAMA_MODEL_DIR" >/dev/null 2>&1 || true
    else
      HF_TOKEN="$HF_TOKEN" \
        hf download "$LLAMA_MMPROJ_REPO" --include 'mmproj-*.gguf' --local-dir "$LLAMA_MODEL_DIR" >/dev/null 2>&1 || true
    fi
    mm=$(find "$LLAMA_MODEL_DIR" -maxdepth 2 -type f -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)
    if [[ -n "$mm" ]]; then
      printf '%s' "$mm"
      return 0
    fi
  fi

  mm=$(find "$LLAMA_MODEL_DIR" -maxdepth 2 -type f -name 'mmproj-*.gguf' -print -quit 2>/dev/null || true)
  if [[ -n "$mm" ]]; then
    printf '%s' "$mm"
    return 0
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
  # Google Drive 不支持作为模型目录(含通过 LLAMA_MODEL_ROOT / MODEL_ROOT 间接指向)
  if [[ "$LLAMA_MODEL_DIR" == /content/drive/* || "$LLAMA_MODEL_DIR" == /content/drive ]]; then
    echo "ERROR: 不支持 Google Drive 作为模型目录: $LLAMA_MODEL_DIR" >&2
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
  # /metrics: 默认开启, 供根目录 bench.py 采样 requests_processing / requests_deferred
  # (llama-server 默认不开放该端点); LLAMA_METRICS=0 可关闭
  if [[ "$LLAMA_METRICS" == "1" ]]; then
    LAUNCH_CMD+=(--metrics)
  fi
  LAUNCH_CMD+=("${SERVER_ARGS[@]}")

  echo "启动 llama 服务... (日志: ${LOG_FILE})" | tee -a "$LOG_FILE"
  echo ">> 加载模型: $MODEL_FILE" | tee -a "$LOG_FILE"
  echo ">> 模型目录: $LLAMA_MODEL_DIR (基础盘: $LLAMA_MODEL_ROOT, 来源: $MODEL_ROOT_SOURCE)" | tee -a "$LOG_FILE"
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
