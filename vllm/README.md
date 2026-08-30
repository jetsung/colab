# colab-vllm

基于 [vLLM](https://github.com/vllm-project/vllm) 的 OpenAI 兼容 API 渠道。启动入口采用官方 `vllm serve`，模型使用 Hugging Face 格式（如 safetensors），不新增 Notebook。

> 运行位置：`colab.sh install vllm`、`colab.sh vllm ...` 在 Colab terminal 的项目根目录执行；也可以在本目录直接执行 `./launch.sh ...`。服务由 `setsid` 后台托管，日志写入项目根目录 `logs/`。

## 快速开始

```bash
# 根目录：安装官方最新 vLLM（Python 3.12 venv，不自动启动）
./colab.sh install vllm

# 配置密钥和模型（也可以写入根目录 .env）
export VLLM_API_KEY=sk-your-key
export VLLM_MODEL_REPO=Qwen/Qwen3-8B

# 启动、查看状态、发送测试请求
./colab.sh vllm start
./colab.sh vllm status
./colab.sh vllm test

# 服务管理
./colab.sh vllm stop
./colab.sh vllm restart
./colab.sh vllm logs
./colab.sh vllm keep
```

也可以从 `vllm/` 目录执行：

```bash
cd vllm
./launch.sh start
./launch.sh status
```

默认服务地址为 `http://0.0.0.0:30000`，与其他渠道一样；同一时间只运行一个默认端口的引擎，或通过 `VLLM_PORT` 修改端口。

## 文件说明

| 文件 | 作用 |
|---|---|
| `launch.sh` | `start/stop/restart/status/test/logs/keep` 服务管理 |
| `.envrc` | 继承根配置并自动加载 GPU profile |
| `.env.g4` / `.env.t4` | G4/L4/Blackwell 与 T4 的显存默认值 |
| `README.md` | vLLM 渠道使用说明 |

虚拟环境默认位于 `/tmp/vllm/venv`，可用 `VLLM_VENV_DIR` 覆盖；依赖安装与启动脚本必须使用同一个路径。

## 模型存放

`VLLM_MODEL_REPO` 可以是 Hugging Face 模型 ID 或本地模型路径。HF ID 首次启动前会使用 `hf download` 下载到本地目录，默认布局为：

```text
${VLLM_MODEL_ROOT:-${MODEL_ROOT:-/content/models}}/<模型仓库名>/
```

命中目录中的 `config.json` 后会直接从本地加载，不重复下载。也可以显式指定：

```bash
VLLM_MODEL_REPO=/content/models/Qwen3-8B ./launch.sh start
VLLM_MODEL_ROOT=/content/models VLLM_MODEL_DIR=/content/models/my-model ./launch.sh start
```

不要把 `VLLM_MODEL_ROOT`、`VLLM_MODEL_DIR`、`VLLM_DOWNLOAD_DIR` 或模型路径指向 `/content/drive`。Google Drive 是冷存储，先在根目录使用：

```bash
./colab.sh sync pull <模型目录>
```

再从本地 `/content/models` 启动；脚本会在启动前拒绝 Drive 路径。

## 环境变量

| 变量 | 说明 |
|---|---|
| `VLLM_MODEL_REPO` | HF 模型 ID 或本地路径；未设置时回退 `MODEL_REPO` |
| `MODEL_ROOT` | 根目录共享本地模型盘，默认 `/content/models` |
| `VLLM_MODEL_ROOT` | vLLM 模型基础盘，优先于 `MODEL_ROOT` |
| `VLLM_MODEL_DIR` | 单个模型目录；默认 `<ROOT>/<repo-name>` |
| `VLLM_SERVED_NAME` | API 中的模型名，默认模型路径末段小写 |
| `VLLM_HOST` / `VLLM_PORT` | 监听地址和端口，默认 `0.0.0.0` / `30000` |
| `VLLM_API_KEY` | Bearer 密钥；未设置时回退 `API_KEY`，显式置空可关闭鉴权 |
| `VLLM_GPU_MEMORY_UTILIZATION` | GPU 显存使用比例，G4 默认 `0.90`、T4 默认 `0.80` |
| `VLLM_MAX_MODEL_LEN` | 最大上下文长度；留空使用 vLLM/模型默认值 |
| `VLLM_MAX_NUM_SEQS` | 最大并发序列数，默认 `512`；vLLM 默认 1024 会让混合 mamba 模型在 CUDA graph 阶段报 `max_num_seqs exceeds available Mamba cache blocks` |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | 单批最大 token 数 |
| `VLLM_TENSOR_PARALLEL_SIZE` | 张量并行 GPU 数，默认 `1` |
| `VLLM_DTYPE` | `auto`、`half`、`bfloat16` 等，默认 `auto` |
| `VLLM_QUANTIZATION` | 可选量化后端 |
| `VLLM_DOWNLOAD_DIR` | vLLM 下载缓存目录，默认 `<ROOT>/.cache/vllm` |
| `VLLM_TRUST_REMOTE_CODE=1` | 追加 `--trust-remote-code` |
| `VLLM_ENABLE_PREFIX_CACHING=1` | 追加 `--enable-prefix-caching` |
| `VLLM_ENFORCE_EAGER=1` | 追加 `--enforce-eager` |
| `VLLM_GENERATION_CONFIG` | 采样默认来源：留空使用模型 `generation_config.json`（默认，Qwen 为 temperature 1.0 / top_k 20 / top_p 0.95）；设为 `vllm` 改用 vLLM 默认 |
| `VLLM_ENABLE_AUTO_TOOL_CHOICE` | 自动工具调用，默认 `1`（仅影响带 tools 的请求）；设为 `0` 关闭 |
| `VLLM_TOOL_CALL_PARSER` | 工具调用解析器；留空时按模型 `config.json` 家族推导（qwen→`qwen3_coder`、deepseek→`deepseek_v3`、glm→`glm45`、kimi→`kimi_k2`、minimax→`minimax_m2`、mistral→`mistral`、llama→`llama3_json`） |

带 `tool_choice: "auto"` 的请求需要同时启用这两项，否则返回 HTTP 400。模型家族无法识别时脚本会打印警告并跳过这两个参数，此时用 `VLLM_TOOL_CALL_PARSER` 显式指定（可选值见 `vllm serve` 的 `--tool-call-parser`，如 `qwen3_coder`、`hermes`、`pythonic`）。
| `VLLM_VENV_DIR` | venv 路径，默认 `/tmp/vllm/venv` |

## API

服务提供 OpenAI 兼容接口。开启密钥时请求带 Bearer header：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.6,
    "max_tokens": 256
  }'
```

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key=os.environ.get("VLLM_API_KEY") or os.environ["API_KEY"],
)
response = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "你好"}],
)
print(response.choices[0].message.content)
```

## 监控与压测

```bash
curl http://localhost:30000/health
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:30000/metrics
./colab.sh vllm bench -n 8 --max-tokens 128
make vllm-bench BENCH_ARGS="-n 8 --max-tokens 128"
```

vLLM 原生提供 `/metrics`，`bench.py` 采样 `vllm:num_requests_running` 与 `vllm:num_requests_waiting`，不需要额外的 metrics 启动参数。日志位于 `logs/vllm_server.log`，启动命令追加记录到 `logs/launch_cmd.log`。

更多官方参数说明见 [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/)。
