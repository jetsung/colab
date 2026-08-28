# ============================================================
# GPU profile: G4 (Blackwell, 显存充足)
# llama.cpp 引擎专用（llama/ 目录）
#
# 用法:
#   由 llama/.envrc 经 source_env_if_exists 自动加载(GPU_PROFILE=g4 时)
#   或手动: source gpu_g4.env
# 以下变量均可由外部环境变量提供, 未提供时 launch.sh 使用自身默认值
# ============================================================

# llama.cpp 编译目标架构（install.sh 使用）
export LLAMA_CUDA_ARCH=89

# 显存充足(96GB Blackwell): LLAMA_CTX=0 由 llama.cpp 按空闲显存自动拟合上下文
export LLAMA_QUANT=UD-Q4_K_XL
export LLAMA_CTX=0
export LLAMA_NGL=999

export LLAMA_MODEL_REPO=unsloth/Qwen3.8-Flash-Next-GGUF
