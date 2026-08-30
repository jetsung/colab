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
│   ├── .envrc             # direnv：继承根 .envrc + 按 GPU_PROFILE 加载 .env.g4/.env.t4
│   ├── .env.g4 / .env.t4  # GPU profile（LLAMA_* 前缀变量，按显卡自动/显式加载）
│   └── launch.sh          # 服务管理：start/stop/restart/status/logs/keep
├── sglang/                # SGLang 引擎
│   ├── README.md          # SGLang 快速上手
│   ├── .envrc             # direnv：继承根 .envrc + 按 GPU_PROFILE 加载 .env.g4/.env.t4
│   ├── .env.g4 / .env.t4  # GPU profile（SGLANG_* 前缀变量）
│   ├── launch.sh          # 服务管理：start/stop/restart/status/logs/keep
│   └── sglang.ipynb       # Notebook 版一键部署
└── vllm/                  # vLLM 引擎（无 Notebook）
    ├── README.md          # vLLM 快速上手
    ├── .envrc             # direnv：继承根 .envrc + 按 GPU_PROFILE 加载 .env.g4/.env.t4
    ├── .env.g4 / .env.t4  # GPU profile（VLLM_* 前缀变量）
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

## 关于显卡（G4 / T4）

Colab 免费/Pro 会话的 GPU 型号并不固定，常见有 **T4（16GB）**、**G4** 以及 L4、A100 等。
脚本已按“低显存可用”设计：

- 进入 `llama/`、`sglang/` 或 `vllm/` 目录时，direnv 会按 `GPU_PROFILE`（或 `nvidia-smi` 自动探测）
  加载对应的 `.env.g4` / `.env.t4`，自动适配显卡（量化档位、上下文、显存占比等）；
- SGLang 默认使用 **fp8 KV cache**；vLLM 使用 `VLLM_GPU_MEMORY_UTILIZATION` 控制显存上限；
- 显存不足时可通过各引擎环境变量降低上下文或显存利用率（详见 [DOCS.md](./DOCS.md)）。
