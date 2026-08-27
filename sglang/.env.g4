# ============================================================
# GPU profile: G4 (NVIDIA Ada, sm_89, 显存较小)
# SGLang 引擎专用（sglang/ 目录）
#
# 用法:
#   由 sglang/.envrc 经 source_env_if_exists 自动加载(GPU_PROFILE=g4 时)
#   或手动: source .env.g4
# 加载后仍可用环境变量临时覆盖
# ============================================================

# FlashInfer / CUDA 架构列表（launch.sh 使用；不设置时自动按 nvidia-smi 探测）
export SGLANG_FLASHINFER_CUDA_ARCH_LIST=8.9f

# 显存较小：降低静态显存占比防 OOM
export SGLANG_MEM_FRACTION_STATIC=0.85

# 显存有限：收敛上下文（按需调整）
export SGLANG_CONTEXT_LENGTH=16384

# 显存小时换更小模型（按需调整；不设置则用 .envrc/.env 的 SGLANG_MODEL_REPO / MODEL_REPO）
# export SGLANG_MODEL_REPO=Qwen/Qwen3-8B
