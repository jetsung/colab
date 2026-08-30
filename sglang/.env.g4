# ============================================================
# GPU profile: G4 (NVIDIA Ada, sm_89, 显存较小)
# SGLang 引擎专用（sglang/ 目录）
#
# 用法:
#   由 sglang/.envrc 经 source_env_if_exists 自动加载(GPU_PROFILE=g4 时)
#   或手动: source .env.g4
# 以下变量均写成 ${VAR:-默认}: 外部(命令行/父 shell/.envrc)已设置时保持原值, 未设置才用本 profile 的默认值
# ============================================================

# FlashInfer / CUDA 架构列表：不在此硬编码。
# launch.sh 按 nvidia-smi 探测并加上正确后缀(Ada->8.9、Blackwell->12.0f)。
# 本 profile 也会被 .envrc 复用到 Blackwell 机型, 写死架构会编译出错误的内核。
# 确有需要时: export FLASHINFER_CUDA_ARCH_LIST=8.9

# 显存较小：降低静态显存占比防 OOM
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"

# 上下文: SGLANG_CTX=0 由 config.json 自动推导(按需调整)
export SGLANG_CTX="${SGLANG_CTX:-0}"

# 显存小时换更小模型（按需调整；不设置则用 .envrc/.env 的 SGLANG_MODEL_REPO / MODEL_REPO）
# export SGLANG_MODEL_REPO=Qwen/Qwen3-8B

# 透传 launch.sh 未覆盖的启动参数（空格分隔；设为空串可整体禁用）
# Ornith/Qwen3.5 这类混合 Mamba 模型 + 投机解码下, mamba state cache 会把
# 并发静默压到 max_mamba_cache_size/4(默认 64 slot -> 仅 16 并发, 即使
# --max-running-requests 传 48)。132 slot -> 33 并发, 32 并发压测不再排队。
# 注意: 132 是在 97GB Blackwell 上调的, mamba cache 多吃约 4GB, KV 池
# 726k -> 289k token。真跑在小显存 Ada 机型上如 OOM, 命令行设
# SGLANG_EXTRA_ARGS= 禁用, 或调小数值。
# 只调 --mamba-full-memory-ratio 到不了 32 并发(slot 数随 ratio 饱和,
# 1.0 也才 ~85), 必须直接指定 cache size。
# `${VAR+set}`(而非 `${VAR:-}`): 显式设空串 = 禁用, 不会被默认值覆盖。
if [[ -z "${SGLANG_EXTRA_ARGS+set}" ]]; then
  SGLANG_EXTRA_ARGS='--max-mamba-cache-size 132'
fi
export SGLANG_EXTRA_ARGS
