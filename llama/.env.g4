# ============================================================
# GPU profile: G4 (Blackwell, 显存充足)
# llama.cpp 引擎专用（llama/ 目录）
#
# 用法:
#   由 llama/.envrc 经 source_env_if_exists 自动加载(GPU_PROFILE=g4 时)
#   或手动: source gpu_g4.env
# 以下变量均写成 ${VAR:-默认}: 外部(命令行/父 shell/.envrc)已设置时保持原值, 未设置才用本 profile 的默认值
# ============================================================

# llama.cpp 编译目标架构（install.sh 使用）
export LLAMA_CUDA_ARCH="${LLAMA_CUDA_ARCH:-89}"

# 显存充足(96GB Blackwell): LLAMA_CTX=0 由 llama.cpp 按空闲显存自动拟合上下文
export LLAMA_QUANT="${LLAMA_QUANT:-UD-Q4_K_XL}"
export LLAMA_CTX="${LLAMA_CTX:-0}"
export LLAMA_NGL="${LLAMA_NGL:-999}"

# 模型仓库(外部已设置时保持原值, 与 .envrc 一致的 ${VAR:-默认} 写法):
#   两种仓库的量化文件布局不同(子目录分片 / 根目录单文件), launch.sh 会自动推导, 无需额外配置
export LLAMA_MODEL_REPO="${LLAMA_MODEL_REPO:-unsloth/Qwen3.8-Flash-Next-GGUF}"
#   unsloth/Qwen3.8-Flash-Next-GGUF -> UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf(子目录分片)
#   unsloth/Qwen3.8-27B-GGUF        -> Qwen3.8-27B-UD-Q4_K_XL.gguf(根目录单文件)
#
# 临时切换(不改动本文件):
#   LLAMA_MODEL_REPO=unsloth/Qwen3.8-27B-GGUF ./launch.sh start
