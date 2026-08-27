# ============================================================
# GPU profile: G4 (NVIDIA Ada, sm_89, 显存较小)
# llama.cpp 引擎专用（llama/ 目录）
#
# 用法:
#   由 llama/.envrc 经 source_env_if_exists 自动加载(GPU_PROFILE=g4 时)
#   或手动: source gpu_g4.env
# 以下变量均可由外部环境变量提供, 未提供时 launch.sh 使用自身默认值
# ============================================================

# llama.cpp 编译目标架构（install.sh 使用）
export LLAMA_CUDA_ARCH=89

# 显存有限：更小量化 + 更小上下文（按需调整）
export LLAMA_QUANT=UD-Q4_K_XL
export LLAMA_CTX=4096
export LLAMA_NGL=999

export LLAMA_MODEL_REPO=unsloth/Qwen3.8-Flash-Next-GGUF
