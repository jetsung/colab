# stable-diffusion.cpp (sd) 部署教程 — Qwen-Image-2.1 / SD1.5

用 **stable-diffusion.cpp** 在 Colab 上做纯 C/C++ 的扩散模型推理（文生图 / 图生图），对外提供
**OpenAI 兼容的图片接口** + 内置 Web UI，经 **bore** 隧道暴露到公网。脚本结构与 `llama/` 引擎完全一致。

本教程默认在 **单张 NVIDIA GPU（G4 / RTX PRO 6000 Blackwell）** 上运行 `unsloth/Qwen-Image-2.1`（GGUF + safetensors 三件套），
无显卡时自动回落到 **CPU + SD1.5 单文件**。

> 运行位置区分：
> - `colab.sh install sd` / `sd/launch.sh` 在 **Colab terminal** 运行（其中 `install` 在项目根目录，`launch.sh` 在 `sd/` 子目录）。
> - 长驻服务用 **setsid** 后台托管（脱离终端，SSH 断开不受影响），日志统一写入根目录 `logs/`。

---

## 0. 为什么是这套方案

- **stable-diffusion.cpp 是 ggml 系的纯 C/C++ 扩散推理器**，与 llama.cpp 同源，无 Python 运行时依赖，支持
  SD1.x/SDXL/SD3/FLUX/Qwen-Image/Z-Image/Wan 等大量模型，以及 CUDA / Vulkan / Metal / CPU 多后端。
- 权重支持 `.ckpt` / `.safetensors` / `.gguf`，可在加载时用 `--type` 量化，也可离线转成 GGUF。
- 自带 **`sd-server`**：提供 `/v1/images/generations`（OpenAI 风格）、`/sdapi/v1/txt2img`（A1111 风格）、
  `/sdcpp/v1/*`（原生异步 API）以及内嵌 Web UI，适合当作常驻出图服务。
- 本引擎默认模型 **Qwen-Image-2.1** 由三个组件拼装：扩散主干（GGUF）+ VAE（safetensors）+ 文本编码器 Qwen3-VL-8B（GGUF）。

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA，计算能力被 CUDA 支持；G4（Blackwell sm_120）需较新 CUDA 工具链 |
| 编译工具（仅 `--build`） | `git`、`cmake`(≥3.18)、`gcc`/`g++`、`make`、`nvcc`（GPU 时） |
| 运行时 | Python 3（仅用于下载），`huggingface_hub` + `hf_xet`（由 `install` 经 **uv** 安装） |
| HuggingFace | 默认仓库均为公开；若遇 gated 仓库需设置 `HF_TOKEN` 并接受许可证 |
| 磁盘 | Qwen-Image-2.1 三件套约 12GB；SD1.5 单文件约 4.3GB |
| 显存 | G4 可全量放下三件套；T4 建议开 `SD_OFFLOAD_TO_CPU=1`；纯 CPU 用 SD1.5 |

---

## 2. 安装 sd 二进制

### 方式一：官方预编译二进制（默认，仅 CPU）

```bash
./colab.sh install sd
```

下载 GitHub Release 的 `sd-master-*-bin-Linux-Ubuntu-24.04-x86_64.zip`（纯 CPU 构建），
解压 `sd-cli` / `sd-server` 到 `/content/stable-diffusion.cpp/build/bin/`。

> Linux 预编译包只有 CPU / Vulkan / ROCm，**没有 CUDA 版**；GPU 用户请用下面的源码编译。

### 方式二：源码编译（GPU/CUDA 必须）

```bash
./colab.sh install sd --build
```

脚本流程：检查工具链 → 安装 `huggingface_hub`/`hf_xet` → `git clone` 仓库 →
初始化子模块（`ggml` / `libwebp` / `libwebm`）→ CMake 配置 → 编译 →
产物 `build/bin/sd-cli`、`build/bin/sd-server`。

这一步等价于官方 [build.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md) 的 CUDA 构建：

```bash
mkdir build && cd build
cmake .. -DSD_CUDA=ON          # GPU(CUDA); 纯 CPU 去掉这一项
cmake --build . --config Release
```

可选环境变量：

| 变量 | 说明 |
|------|------|
| `SD_DIR` | 安装目录（默认 `/content/stable-diffusion.cpp`） |
| `SD_CUDA_ARCH` | 显式指定 CUDA 架构（如 `120`）；不填由 ggml 的 `native` 在构建时自动探测当前 GPU |
| `SD_SERVER_BUILD_FRONTEND=1` | 额外构建内嵌 Web UI（需 Node.js ≥20 + pnpm ≥10；默认关闭，服务仍可跑，只是根路径不返回网页） |
| `CLEAN=1 ./colab.sh install sd --build` | 清掉 `build/` 重新编译 |

验证：

```bash
ls /content/stable-diffusion.cpp/build/bin/     # sd-cli  sd-server
/content/stable-diffusion.cpp/build/bin/sd-server --version
```

---

## 3. 模型准备

### 默认（G4）：Qwen-Image-2.1 三件套

| 组件 | flag | HF 仓库 | 文件 |
|------|------|---------|------|
| 扩散主干 | `--diffusion-model` | `unsloth/Qwen-Image-2.1-GGUF` | `qwen-image-2.1-Q4_K_M.gguf` |
| VAE | `--vae` | `unsloth/Qwen-Image-2.1-FP8` | `vae/qwen_image_2.1_vae_bf16.safetensors` |
| 文本编码器 | `--llm` | `unsloth/Qwen3-VL-8B-Instruct-GGUF` | `Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf` |

`sd/launch.sh` 首次启动时会用 `hf download hf://<org>/<repo>/<file>` 自动下载到 **HF 标准缓存**
（`~/.cache/huggingface/hub`，仓库路径中的 `/` 替换为 `--`），再从 `snapshots/` 下定位真实文件路径：

```
~/.cache/huggingface/hub/
├── models--unsloth--Qwen-Image-2.1-GGUF/snapshots/<hash>/qwen-image-2.1-Q4_K_M.gguf
├── models--unsloth--Qwen-Image-2.1-FP8/snapshots/<hash>/vae/qwen_image_2.1_vae_bf16.safetensors
└── models--unsloth--Qwen3-VL-8B-Instruct-GGUF/snapshots/<hash>/Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf
```

> 缓存目录遵循 `HF_HUB_CACHE` / `HF_HOME`（默认 `~/.cache/huggingface/hub`）。已下载过则直接复用，不联网。

> Qwen-Image-2.1 必须配它自己的 VAE（`qwen_image_2.1_vae_bf16.safetensors`），旧的 Qwen-Image / Wan2.2 VAE **不通用**。

### 图生图 / 图像编辑（可选）

编辑需要视觉投影器。取消 `sd/.env.g4` 中这两行注释，或命令行覆盖：

```bash
SD_LLM_VISION=hf://unsloth/Qwen3-VL-8B-Instruct-GGUF/mmproj-F16.gguf ./launch.sh start
```

### CPU 默认：SD1.5 单文件

`sd/.env.cpu` 默认 `stable-diffusion-v1-5/stable-diffusion-v1-5` 的 `v1-5-pruned-emaonly.safetensors`，
用 `-m` 单文件加载（无需单独 VAE / 文本编码器）。

### HuggingFace Token（可选）

公开仓库无需 token；若换用 gated 仓库：

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

---

## 4. 启动服务

```bash
cd sd
direnv allow .        # 首次进入需允许本目录 .envrc(根目录的 allow 不覆盖引擎子目录)
./launch.sh start
```

> `sd/.envrc` 是独立文件，**根目录 `direnv allow .` 不会连带允许它**。首次 `cd sd` 时 direnv 会提示
> `direnv: error .../.envrc is blocked`，执行 `direnv allow .` 后 `SD_*` 变量与 profile 才会生效
> （其余引擎 `llama/` `sglang/` `vllm/` 同理）。`./colab.sh setup hint` 会打印该步骤。

脚本会：加载平台 profile → 解析（必要时下载）各模型组件 → 用 `sd-server` 后台加载并监听 `0.0.0.0:30000`。

服务管理：

```bash
./launch.sh start      # 启动（后台 setsid 托管；首次自动下载模型）
./launch.sh stop       # 优雅停止，超时强杀
./launch.sh restart    # 重启
./launch.sh status     # 进程 + 健康检查（curl /v1/models）
./launch.sh test       # 调用服务生成一张测试图
./launch.sh generate "一只戴墨镜的柴犬"   # 用 sd-cli 一次性出图（不依赖服务）
./launch.sh logs       # 实时跟踪日志
./launch.sh keep       # 守护模式：崩溃自动拉起
```

也可从项目根目录经 `colab.sh` / `make` 调用：

```bash
./colab.sh sd start
./colab.sh sd generate "一只戴墨镜的柴犬"
make sd-start
make sd-test
```

日志统一写在**项目根目录** `logs/sd_server.log`（PID 文件 `sd/sd.pid`，启动命令追加于 `logs/launch_cmd.log`）。

### 组件解析（统一模型来源格式）

每个组件用 `<VAR>` 一个变量表达完整来源，支持四种写法：

| 写法 | 示例 | 行为 |
|------|------|------|
| 本地路径 | `/path/to/model.gguf`、`./model.gguf` | 直接使用（相对路径按当前目录解析） |
| `file://` | `file:///abs/model.gguf`、`file://rel/model.gguf` | 已下载到本地的文件，直接使用不下载（三个 `/` 为绝对路径；两个 `/` 后接相对路径，按当前目录展开） |
| `hf://` | `hf://unsloth/Qwen-Image-2.1-GGUF/qwen-image-2.1-Q4_K_M.gguf` | 等价 `hf download hf://<org>/<repo>/<file>` |
| HF https URL | `https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/blob/main/qwen-image-2.1-Q4_K_M.gguf` | 去掉 domain 与 `blob/main`（或 `resolve/<rev>`），归一为 `hf://<org>/<repo>/<file>` 后按上一行下载 |

HF 来源（`hf://` 与 HF https URL）先查 HF 标准缓存
`~/.cache/huggingface/hub/models--<org>--<repo>/snapshots/`（已存在则不联网），
未命中则 `hf download hf://<org>/<repo>/<file>` 下载，再从 `snapshots/` 定位真实文件路径。

各 profile（`.env.g4/.env.t4/.env.cpu`）默认值即 `hf://` 形式。也保留旧式
`<VAR>_REPO` + `<VAR>_FILE` 组合作为回退（`<VAR>` 未设置时生效）。

> `SD_MODEL`（→`-m`）与 `SD_DIFFUSION_MODEL`（→`--diffusion-model`）**至少配置一个**，否则启动报错。

---

## 5. 环境变量（均带 `SD_` 前缀）

| 变量 | 默认 / 回退 | 说明 |
|------|-------------|------|
| `SD_DIFFUSION_MODEL` | `.env.g4`（`hf://…`） | 组件式扩散主干（→ `--diffusion-model`）；支持本地路径/`file://`/`hf://`/HF https URL |
| `SD_MODEL` | `.env.cpu` 用 SD1.5 | 单文件全模型（→ `-m`）；同上四种来源写法 |
| `SD_VAE` | `.env.g4` | 独立 VAE（→ `--vae`）；同上 |
| `SD_LLM` | `.env.g4` | 文本编码器（→ `--llm`）；同上 |
| `SD_LLM_VISION` | 空（关闭） | 视觉投影器（→ `--llm_vision`，图生图/编辑）；同上 |
| `SD_CLIP_L` / `SD_CLIP_G` / `SD_T5XXL` | 空 | SD3/FLUX 的文本编码器（→ `--clip_l/--clip_g/--t5xxl`）；同上 |
| `<VAR>_REPO` / `<VAR>_FILE` | 空 | 旧式组合写法回退（`<VAR>` 未设置时生效） |
| `HF_HUB_CACHE` / `HF_HOME` | `~/.cache/huggingface/hub` | HF 下载缓存目录（HF 标准布局，非 SD 专属变量） |
| `SD_HOST` / `SD_PORT` | `0.0.0.0` / `30000` | 监听地址与端口（与 bore 隧道一致） |
| `SD_STEPS` | profile 提供 | 采样步数（→ `--steps`） |
| `SD_CFG_SCALE` | profile 提供 | CFG 强度（→ `--cfg-scale`） |
| `SD_SAMPLING_METHOD` | profile 提供 | 采样器（→ `--sampling-method`，如 `euler` / `euler_a`） |
| `SD_WIDTH` / `SD_HEIGHT` | profile 提供 | 默认分辨率（→ `-W` / `-H`） |
| `SD_SEED` / `SD_SCHEDULER` / `SD_FLOW_SHIFT` / `SD_NEGATIVE_PROMPT` | 空 | 可选生成参数（留空则不传） |
| `SD_DIFFUSION_FA` | profile 提供 | 1=扩散模型启用 Flash Attention（→ `--diffusion-fa`） |
| `SD_OFFLOAD_TO_CPU` | `0`（T4 为 `1`） | 1=权重放内存按需换入显存（→ `--offload-to-cpu`） |
| `SD_VAE_TILING` | `0` | 1=VAE 分块解码省显存（→ `--vae-tiling`） |
| `SD_THREADS` | `-1` | CPU 线程数；`-1`/空=由 sd.cpp 按物理核心数决定 |
| `SD_LORA_DIR` | 空 | LoRA 目录（→ `--lora-model-dir`） |
| `SD_LOG_LEVEL` | 空（默认 `info`） | 日志级别（→ `--log-level`：`debug`/`verbose`/`info`/`warn`/`error`） |
| `SD_BATCH_COUNT` | `1` | `generate` 一次出图张数（→ `-b`；>1 时输出名带 `%02d`） |
| `SD_DIR` | `/content/stable-diffusion.cpp` | 安装目录 |
| `SD_SERVER` / `SD_CLI` | `<SD_DIR>/build/bin/...` | 二进制路径（亦从 `PATH` 查找） |
| `SD_OUTPUT_DIR` | `/content/outputs` | `generate` / `test` 出图目录 |
| `SD_PROMPT` | `a lovely cat` | `generate` / `test` 默认提示词 |
| `SD_REQUEST_TIMEOUT` | `1800` | `test` 请求超时秒数（CPU 出图较慢） |
| `SD_XET` | `1` | 1=启用 HF Xet 存储（默认），0=禁用 |
| `SD_CUDA_ARCH` / `SD_SERVER_BUILD_FRONTEND` | 空 | 仅安装期使用（见第 2 节） |

---

## 6. 调用示例

### 6.1 OpenAI 兼容图片接口

```bash
curl -s http://localhost:30000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a lovely cat holding a sign says '\''sd.cpp'\''", "size": "1024x1024", "n": 1, "output_format": "png"}' \
  | python3 -c 'import sys,json,base64; d=json.load(sys.stdin); open("out.png","wb").write(base64.b64decode(d["data"][0]["b64_json"]))'
```

返回 `data[].b64_json`（PNG 的 base64）。`size` 用 `宽x高`（建议能被 32 整除），`n` 为出图张数。

### 6.2 A1111 风格接口

```bash
curl -s http://localhost:30000/sdapi/v1/txt2img \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a lovely cat", "width": 1024, "height": 1024, "steps": 20, "cfg_scale": 6.0, "sampler_name": "Euler"}'
```

### 6.3 Web UI 与原生 API

- Web UI：浏览器打开 `http://localhost:30000/`（需安装时开启 `SD_SERVER_BUILD_FRONTEND=1`）
- 原生异步 API：`POST /sdcpp/v1/img_gen`、`GET /sdcpp/v1/capabilities`、`GET /sdcpp/v1/jobs/<id>`
- 模型列表：`GET /v1/models`（健康检查也用它）

### 6.4 公网访问（bore 隧道）

```bash
cd /content/colab
./colab.sh bore start          # 本地 30000 -> 公网 65535
./colab.sh bore logs           # 查看分配到的公网地址
```

### 6.5 一次性出图（sd-cli，不走服务）

```bash
cd sd
./launch.sh generate "一只戴墨镜的柴犬"          # 输出到 /content/outputs/sd_<时间戳>.png
SD_STEPS=30 SD_WIDTH=768 SD_HEIGHT=768 ./launch.sh generate "水彩风格的雪山"
```

---

## 7. 平台适配（g4 / t4 / cpu）

进入 `sd/` 时，`.envrc` 按 `GPU_PROFILE` 加载 profile；未设置时用 `nvidia-smi` 自动探测（G4→`g4`，T4→`t4`，无显卡→`cpu`）。
也可显式指定：`GPU_PROFILE=cpu ./launch.sh start`。

| profile | 默认模型 | 关键参数 |
|---------|----------|----------|
| `.env.g4` | Qwen-Image-2.1（Q4_K_M + Qwen3-VL UD-Q4_K_XL） | `1024x1024`、`steps 20`、`cfg 6.0`、`euler`、`--diffusion-fa` |
| `.env.t4` | Qwen-Image-2.1（Q4_K_S + Qwen3-VL UD-Q3_K_XL） | `768x768`、`--offload-to-cpu` |
| `.env.cpu` | SD1.5 单文件 safetensors | `512x512`、`steps 20`、`cfg 7.5`、`euler_a` |

---

## 8. 常见问题

- **GPU 会话里报 `no CUDA-capable device` / 跑得很慢**
  → 预编译包是纯 CPU 版。用 `./colab.sh install sd --build` 重新编 CUDA 版。
- **编译报找不到 CUDA**
  → 缺 `nvcc`；确认 Colab 会话带 GPU 且已装 CUDA toolkit（`nvcc --version`）。换架构可 `SD_CUDA_ARCH=120 ./colab.sh install sd --build`。
- **`WebP support enabled but no source found`**
  → 子模块没初始化。重跑安装（脚本会 `git submodule update --init --recursive ggml thirdparty/libwebp thirdparty/libwebm`）。
- **下载卡住 / 失败**
  → 确认网络与仓库名；gated 仓库需 `export HF_TOKEN=...` 并接受许可证。Xet 存储需 `hf_xet`（安装时已装）。
- **显存不足（T4）**
  → 开 `SD_OFFLOAD_TO_CPU=1`、`SD_VAE_TILING=1`，或降低 `SD_WIDTH/SD_HEIGHT`、换更小量化档。
- **出图是全黑 / 全白**
  → 多为分辨率或采样参数与该模型不匹配；Qwen-Image-2.1 参考 `cfg 6.0` / `steps 20` / `euler`，分辨率能被 32 整除。
- **端口冲突**
  → 四个引擎默认都用 30000（bore 隧道固定转发本地 30000），同一时刻只跑一个，或改 `SD_PORT`。
- **安全风险**
  → `sd-server` 默认无鉴权；对外暴露时请自行在隧道/反代层加访问控制。

---

## 9. 文件说明

```
colab/
├── colab.sh                 # 一体化入口: ./colab.sh install sd [--build] 安装; ./colab.sh sd <动作> 管理
├── logs/sd_server.log       # 服务日志（根目录 logs/）
├── sd/
│   ├── .envrc               # direnv: 继承根 .envrc + 按平台加载 .env.g4/.env.t4/.env.cpu
│   ├── .env.g4/.env.t4      # GPU profile（Qwen-Image-2.1 三件套 + 生成参数）
│   ├── .env.cpu             # CPU profile（SD1.5 单文件）
│   ├── launch.sh            # 服务管理: start/stop/restart/status/test/generate/logs/keep
│   └── sd.pid               # 运行 PID（自动生成）
└── /content/
    ├── stable-diffusion.cpp/        # 源码与编译产物（build/bin/sd-cli、sd-server）
    └── outputs/                     # generate / test 出图目录
~/.cache/huggingface/hub/            # 模型 HF 标准缓存（models--<org>--<repo>/snapshots/）
├── models--unsloth--Qwen-Image-2.1-GGUF/
├── models--unsloth--Qwen-Image-2.1-FP8/
└── models--unsloth--Qwen3-VL-8B-Instruct-GGUF/
```

参考：
- 项目主页：https://github.com/leejet/stable-diffusion.cpp
- 构建文档：[docs/build.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md)
- Qwen-Image-2.1 指南：[docs/qwen_image_2.1.md](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md)
