# ============================================================
# GPU profile: G4 (Blackwell, 显存充足)
# stable-diffusion.cpp 引擎专用（sd/ 目录）
#
# 用法:
#   由 sd/.envrc 经 source_env_if_exists 自动加载(GPU_PROFILE=g4 时)
#   或手动: source .env.g4
# 以下变量均写成 ${VAR:-默认}: 外部(命令行/父 shell/.envrc)已设置时保持原值, 未设置才用本 profile 的默认值
# ============================================================

# 编译/安装目录（colab.sh install sd 使用）
export SD_DIR="${SD_DIR:-/content/stable-diffusion.cpp}"

# ---- 默认模型: Qwen-Image-2.1 三件套 ----
# sd.cpp 的组件式模型拆成 3 个 HF 仓库: 扩散主干(GGUF) + VAE(safetensors) + 文本编码器(GGUF)。
# 每项都可用 *_REPO / *_FILE 覆盖; 也可用 SD_DIFFUSION_MODEL / SD_VAE / SD_LLM 直接给本地绝对路径。
export SD_DIFFUSION_MODEL_REPO="${SD_DIFFUSION_MODEL_REPO:-unsloth/Qwen-Image-2.1-GGUF}"
export SD_DIFFUSION_MODEL_FILE="${SD_DIFFUSION_MODEL_FILE:-qwen-image-2.1-Q4_K_M.gguf}"
export SD_VAE_REPO="${SD_VAE_REPO:-unsloth/Qwen-Image-2.1-FP8}"
export SD_VAE_FILE="${SD_VAE_FILE:-vae/qwen_image_2.1_vae_bf16.safetensors}"
export SD_LLM_REPO="${SD_LLM_REPO:-unsloth/Qwen3-VL-8B-Instruct-GGUF}"
export SD_LLM_FILE="${SD_LLM_FILE:-Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf}"

# 图生图/编辑用的视觉投影器(Qwen-Image-2.1 走 Qwen3-VL, 需 --llm_vision)。
# 默认不下载; 需要图片编辑时取消注释(约 1.7GB)。
# export SD_LLM_VISION_REPO="${SD_LLM_VISION_REPO:-unsloth/Qwen3-VL-8B-Instruct-GGUF}"
# export SD_LLM_VISION_FILE="${SD_LLM_VISION_FILE:-mmproj-F16.gguf}"

# ---- 生成默认参数(用户实测值) ----
export SD_STEPS="${SD_STEPS:-20}"
export SD_CFG_SCALE="${SD_CFG_SCALE:-6.0}"
export SD_SAMPLING_METHOD="${SD_SAMPLING_METHOD:-euler}"
export SD_WIDTH="${SD_WIDTH:-1024}"
export SD_HEIGHT="${SD_HEIGHT:-1024}"
export SD_DIFFUSION_FA="${SD_DIFFUSION_FA:-1}"

# ---- 服务监听(与 bore 隧道的本地端口一致) ----
export SD_HOST="${SD_HOST:-0.0.0.0}"
export SD_PORT="${SD_PORT:-30000}"

# 临时切换(不改动本文件):
#   SD_DIFFUSION_MODEL_FILE=qwen-image-2.1-Q8_0.gguf ./launch.sh start
#   SD_STEPS=30 ./launch.sh generate
