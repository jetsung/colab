# Google Colab 部署大模型完整教程

> 目标环境：**Google Colab**（GPU 会话，可能是 **G4** 或 **T4** 显卡）/ Ubuntu Linux
> 软件栈：三选一 —— **llama.cpp**（GGUF，轻量）、**SGLang**（高吞吐，多功能）或 **vLLM**（HF 模型，OpenAI 兼容服务）
> 服务形态：OpenAI 兼容 API，支持深度思考、工具调用、连续批处理和 Prometheus 指标
> 运行方式：宿主机用 Colab CLI 创建 GPU 会话，服务跑在 Colab terminal，经 bore 隧道暴露公网

本项目用于在 **Google Colab** 环境下部署开源大模型，并对内/对外提供 OpenAI 兼容 API。
推理引擎不限定某一种，可按需选择 **llama.cpp**、**SGLang** 或 **vLLM**（安装与使用教程见
[DOCS.md](./DOCS.md)）：

- **llama.cpp**：单文件 `llama-server`，模型为 GGUF 格式，显存占用低、上手快，
  非常适合 Colab 的 **G4 / T4** 等有限显存环境。
- **SGLang**：Python 生态，支持深度思考、工具调用、EAGLE(MTP) 投机解码、
  高并发与高吞吐，适合有更高性能/并发需求的场景。
- **vLLM**：Python 生态，直接运行 Hugging Face 模型，使用官方 `vllm serve` 提供
  OpenAI 兼容 API，并原生提供 Prometheus `/metrics`。

Colab 提供的 GPU 型号不定（可能分到 **G4** 或 **T4** 等），文档针对低显存场景给出适配建议
（llama.cpp 用 GGUF 量化、SGLang/vLLM 用显存利用率和上下文参数控制）。

## 目录

```
colab/
├── README.md              # 本文件：总体说明
├── DOCS.md                # 完整部署教程：llama.cpp / SGLang / vLLM 安装与使用、参数、API、FAQ
├── Makefile               # 常用操作封装（薄封装 colab.sh）
├── colab.sh               # 一体化管理入口：vps / setup / install / bore / sync 子命令
├── bench.py               # 并发压测：sglang / llama.cpp / vLLM 通用（吞吐、延迟、并发峰值）
├── .envrc / .env          # 环境变量（.env 含密钥，不入 git）
├── llama/                 # llama.cpp 引擎（GGUF）
│   ├── README.md          # 特定模型的 llama.cpp 部署教程（含 PR 需求说明）
│   ├── .envrc             # direnv：继承根 .envrc + 按 GPU_PROFILE 加载 .env.g4/.env.t4/.env.cpu
│   ├── .env.g4 / .env.t4  # GPU profile（LLAMA_* 前缀变量，按显卡自动/显式加载）
│   ├── .env.cpu           # CPU profile（无 NVIDIA GPU 时自动加载）
│   └── launch.sh          # 服务管理：start/stop/restart/status/logs/keep
├── sglang/                # SGLang 引擎
│   ├── README.md          # SGLang 快速上手
│   ├── .envrc             # direnv：继承根 .envrc + 按 GPU_PROFILE 加载 .env.g4/.env.t4/.env.cpu
│   ├── .env.g4 / .env.t4  # GPU profile（SGLANG_* 前缀变量）
│   ├── .env.cpu           # CPU profile（无 NVIDIA GPU 时自动加载）
│   ├── launch.sh          # 服务管理：start/stop/restart/status/logs/keep
│   └── sglang.ipynb       # Notebook 版一键部署
└── vllm/                  # vLLM 引擎（无 Notebook）
    ├── README.md          # vLLM 快速上手
    ├── .envrc             # direnv：继承根 .envrc + 按 GPU_PROFILE 加载 .env.g4/.env.t4/.env.cpu
    ├── .env.g4 / .env.t4  # GPU profile（VLLM_* 前缀变量）
    ├── .env.cpu           # CPU profile（无 NVIDIA GPU 时自动加载）
    └── launch.sh          # 服务管理：start/stop/restart/status/logs/keep
```

## 快速开始

> 完整安装与使用教程（含 llama.cpp、SGLang、vLLM 参数详解、API 示例、FAQ）见 [DOCS.md](./DOCS.md)。
> 所有操作均通过根目录 `colab.sh` 统一入口执行，子命令详情见 `./colab.sh --help`。

宿主机安装 Colab CLI 并创建 GPU 会话，连进 Colab terminal 后按需选择引擎：

### 通用前置（三个引擎都需要）

```bash
# 0. 宿主机: 创建 Colab GPU 会话并挂载 Drive, 然后连进 Colab terminal
./colab.sh vps create     # 安装 CLI + 创建会话(默认 G4, 可用 -g T4 指定)
./colab.sh vps mount      # 挂载 Drive

# 1. Colab terminal: 安装前置依赖, 加载环境变量
./colab.sh setup all
source ~/.bashrc
```

### 引擎一：SGLang（sglang/）

```bash
./colab.sh install sglang          # 装 uv → 建 venv → 装 SGLang
cd sglang && ./launch.sh start     # 启动服务
./launch.sh status                 # 等待就绪
cd .. && ./colab.sh bore start     # 暴露到公网
```

### 引擎二：llama.cpp（llama/）

```bash
./colab.sh install llama           # 下载最新 prerelease Ubuntu 通用预编译二进制
# ./colab.sh install llama --build   # GPU/CUDA 或需源码编译时使用
export HF_TOKEN=hf_xxxx            # 下载 GGUF 所需（需先接受模型许可证）
cd llama && ./launch.sh start      # 下载模型并启动 llama-server
./launch.sh status                 # 查看状态 + 健康检查
cd .. && ./colab.sh bore start     # 暴露到公网
```

### 引擎三：vLLM（vllm/）

```bash
./colab.sh install vllm            # 建 Python 3.12 venv，安装官方最新 vLLM
export VLLM_MODEL_REPO=Qwen/Qwen3-8B
export VLLM_API_KEY=sk_xxxx
./colab.sh vllm start               # 官方 vllm serve 启动 HF 模型
./colab.sh vllm status              # 查看状态 + 健康检查
./colab.sh vllm test                # 发送 OpenAI 兼容测试请求
./colab.sh bore start                # 暴露到公网（按需）
```

> 三个引擎默认均监听 `0.0.0.0:30000`，对外提供 OpenAI 兼容 API；公网入口为 bore 隧道地址。
> 引擎专属环境变量（`LLAMA_*` / `SGLANG_*` / `VLLM_*`）见各引擎目录 README；日志统一写在根目录 `logs/`。

## 关于运行平台（G4 / T4 / CPU）

Colab 免费/Pro 会话的 GPU 型号并不固定，常见有 **T4（16GB）**、**G4** 以及 L4、A100 等；
**CPU 会话（无 NVIDIA GPU）同样支持**。脚本已按“低显存/低内存可用”设计：

- 进入 `llama/`、`sglang/` 或 `vllm/` 目录时，direnv 按 `GPU_PROFILE` 加载对应 profile；
  未设置时自动探测：探测到 NVIDIA 显卡按型号加载 `.env.g4` / `.env.t4`，
  **没有 `nvidia-smi` 或无输出则加载 `.env.cpu`**；
- 也可显式指定（`GPU_PROFILE=g4|t4|cpu`），优先级高于自动探测；
  探测到显卡但型号未识别时不加载任何 profile（不会把 A100 之类误判成 CPU 而静默降速）；
- 显存不足时可通过各引擎环境变量降低上下文或显存利用率（详见 [DOCS.md](./DOCS.md)）。

### CPU 会话注意事项

- 安装：`./colab.sh install <engine>` 会自动按平台选依赖 —— GPU 装 CUDA 版 torch，
  **CPU 装 CPU 版 torch**；`install llama --build` 在有显卡时编 CUDA 版、无显卡时编纯 CPU 版
  （默认的官方预编译 `ubuntu-x64` 包本身就是纯 CPU 构建，两种会话都能用）。
- 启动：各 `launch.sh` 会自动跳过 GPU 专属参数（SGLang 的 `--attention-backend flashinfer` /
  `--kv-cache-dtype fp8_e4m3` / `--mem-fraction-static`，vLLM 的 `--gpu-memory-utilization`），
  并改为显式传 `--device cpu`。
- **SGLang 在 CPU 上需额外构建**：其 CPU 引擎不在 PyPI wheel 里，官方要求用
  `pyproject_cpu.toml` 源码构建（或直接用官方 `xeon.Dockerfile` 镜像），
  见 [SGLang CPU Server](https://docs.sglang.io/docs/platforms/cpu_server)。
  `install sglang` 只装好 venv + CPU 版 torch 并打印提示。想开箱即用优先选 llama.cpp 或 vLLM。
- **必须自己换小模型**：默认的 27B/80B-MoE 模型在 CPU 会话的内存（约 12GB）里装不下。
  SGLang/vLLM 建议 `Qwen/Qwen3-8B` 量级，llama.cpp 建议 `Qwen/Qwen3-8B-GGUF` + `Q4_K_M`；
  各 `.env.cpu` 里已给出注释开关。
- 上下文：CPU 上 KV cache 走系统内存，`.env.cpu` 统一限制到 `8192`，避免按模型的
  训练上下文（可达 256K）建池而 OOM；内存充裕可按需调大。
