# Google Colab 部署大模型完整教程（llama.cpp / SGLang / vLLM）

> 目标环境：**Google Colab**（GPU 会话，可能是 **G4** 或 **T4** 显卡）/ Ubuntu Linux
> 服务形态：OpenAI 兼容 API，支持深度思考、工具调用、投机解码等
>
> 本教程提供 **三套可选推理引擎**：
> - **llama.cpp**：单文件 `llama-server`，GGUF 模型，显存占用低、上手快，适合 G4/T4 有限显存；
> - **SGLang**：Python 生态，高吞吐、多功能（深度思考 / 工具调用 / EAGLE 投机解码）；
> - **vLLM**：Python 生态，直接运行 Hugging Face 模型，使用官方 `vllm serve` 提供高吞吐 OpenAI 兼容服务。
>
> 三者都能对外提供 OpenAI 兼容 API，可按需求选择（见下方「如何选择」）。

---

## 如何选择

| 维度 | llama.cpp | SGLang | vLLM |
|---|---|---|---|
| 安装 | 编译 CUDA 版或下载预编译二进制 | `pip install sglang`（Python venv） | 官方最新 vLLM（Python venv） |
| 模型格式 | **GGUF**（需量化/转换） | HF safetensors（原格式直接跑） | HF safetensors（原格式直接跑） |
| 显存占用 | 低（GGUF 量化到 Q4 等） | 中等（fp8 KV 量化） | 由 `--gpu-memory-utilization` 控制 |
| 上手难度 | ★（单命令启动） | ★★★（参数多、自动推导） | ★★（官方 `vllm serve`） |
| 深度思考 / 工具调用 | 支持（需对应模板） | 支持（解析器更完善） | 随模型和 vLLM 版本支持 |
| 高并发 / 高吞吐 | 一般 | 强（连续 batching、投机解码） | 强（连续 batching） |
| 适用场景 | G4/T4 低显存、快速起服务 | 追求性能、多并发、复杂功能 | HF 模型、高吞吐、标准 API |

> 简单判断：**图省事、显存小 → llama.cpp**；**要 SGLang 特性 → SGLang**；**要标准 vLLM 服务 → vLLM**。

---

## 目录

**Part A：llama.cpp**
- [A1 安装 llama.cpp](#a1-安装-llama.cpp)
- [A2 获取 GGUF 模型](#a2-获取-gguf-模型)
- [A3 启动 llama-server](#a3-启动-llama-server)
- [A4 llama.cpp 参数详解](#a4-llama.cpp-参数详解)
- [A5 llama.cpp API 使用示例](#a5-llama.cpp-api-使用示例)

**Part B：SGLang**
- [B1 安装 SGLang](#b1-安装-sglang)
- [B2 显卡的坑与修复](#b2-显卡的坑与修复)
- [B3 启动脚本](#b3-启动脚本)
- [B4 SGLang 参数详解](#b4-sglang-参数详解)
- [B5 SGLang API 使用示例](#b5-sglang-api-使用示例)

**Part C：vLLM**
- [C1 安装 vLLM](#c1-安装-vllm)
- [C2 vLLM 启动脚本](#c2-vllm-启动脚本)
- [C3 vLLM 参数详解](#c3-vllm-参数详解)
- [C4 vLLM API 与监控](#c4-vllm-api-与监控)
- [C5 vLLM 压测与调优](#c5-vllm-压测与调优)

**通用**
- [5. 模型存放：冷存储（Drive）vs 本地盘](#5-模型存放冷存储google-drive-vs-本地盘)
- [6. 运维与监控](#6-运维与监控)
- [7. 常见问题 FAQ](#7-常见问题-faq)

---

# Part A：llama.cpp

> **项目已内置一键脚本**：`./colab.sh install llama` 默认选择 llama.cpp GitHub 最新 prerelease，下载 Ubuntu 通用预编译二进制（不含 CUDA）；
> GPU/CUDA 用户应使用 `./colab.sh install llama --build` 源码编译。脚本同时安装 HuggingFace/Xet 依赖；`llama/launch.sh` 管理服务（下载 GGUF 模型、启动/停止/状态等）。
> 直接使用即可（见 [llama/README.md](./llama/README.md)）。下方为两种手动方式，便于理解与自定义。

## A1. 安装 llama.cpp

llama.cpp 提供两种方式：**下载预编译二进制**（最快）或 **源码编译 CUDA 版**（可针对 Colab GPU 优化）。

> 项目内置脚本默认对应「方式一」的自动化：`./colab.sh install llama` 会从 GitHub Releases
> 选择最新 prerelease 的 `ubuntu-x64.tar.gz` 通用资产，解压到 `/content/llama.cpp/build/bin/`。
> 该预编译包不含 CUDA；需要 GPU/CUDA 时使用 `./colab.sh install llama --build`。

### A1.1 方式一：Ubuntu 通用预编译二进制（最快；GPU/CUDA 请用源码编译）

从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 的最新 prerelease 下载
`llama-<build>-bin-ubuntu-x64.tar.gz`（Ubuntu 通用版，不含 CUDA），解压后即可用 `llama-server`：

```bash
cd ~
wget <llama-XXXX-bin-ubuntu-x64.tar.gz 链接>
mkdir -p llama.cpp
tar -xzf llama-*-bin-ubuntu-x64.tar.gz -C llama.cpp
cd llama.cpp/llama-*
chmod +x llama-server
./llama-server --version
```

### A1.2 方式二：源码编译 CUDA 版

Colab 的 GPU（G4/T4）均为 NVIDIA，需要 CUDA 后端。编译需 CMake 与 CUDA 工具链
（Colab 自带 nvcc，若缺失先 `apt install -y cmake nvidia-cuda-toolkit` 或用 pip 装 nvcc）：

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=on
cmake --build build --config Release -j
# 生成的可执行文件在 build/bin/ 下: llama-server / llama-cli / llama-quantize 等
```

> **Colab 显卡架构提示**：G4（Ada，sm_89）与 T4（Turing，sm_75）架构不同。
> 若 CMake 探测 CUDA 架构不准，可显式指定：
> `cmake -B build -DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES="89"`（G4）或 `"75"`（T4）。
> 也可用 `-DGGML_CUDA=on` 自动探测当前 GPU。

### A1.3 验证

```bash
./build/bin/llama-server --version   # 或预编译的 ./llama-server --version
# 应显示版本与 CUDA 支持信息
```

---

## A2. 获取 GGUF 模型

> 项目内置脚本已封装修拉流程：设置 `HF_TOKEN`（需已接受模型许可证）后执行
> `cd llama && ./launch.sh start` 会自动下载仓库中的量化档位 GGUF 并启动服务。下方为手动步骤。

llama.cpp 需要 **GGUF** 格式模型。有两种途径：

### A2.1 直接下载已量化好的 GGUF

Hugging Face 上许多作者发布现成 GGUF（如 `Qwen/Qwen3-8B-GGUF`、各 TheBloke 镜像），
用 `hf download`（来自 `huggingface_hub`，本项目经 **uv** 安装）下载即可：

```bash
uv pip install --system --upgrade huggingface_hub hf_xet
hf download Qwen/Qwen3-8B-GGUF \
  --include "Q4_K_M/*" \
  --local-dir ./models/qwen3-8b
# 得到 ./models/qwen3-8b/Q4_K_M/*.gguf
```

### A2.2 自己转换 + 量化（从 safetensors）

如果你只有 HF 原格式模型，先用 `llama.cpp` 的转换脚本转成 GGUF，再量化到目标位数：

```bash
# 1) 转换 safetensors -> GGUF (fp16)
python convert_hf_to_gguf.py ./models/qwen3-8b \
  --outfile ./models/qwen3-8b-f16.gguf

# 2) 量化到 Q4_K_M (显存小用 Q4; 显存充足可用 Q5/Q6 更高精度)
./build/bin/llama-quantize ./models/qwen3-8b-f16.gguf \
  ./models/qwen3-8b-q4km.gguf Q4_K_M
```

> **量化档位建议（G4/T4 显存有限）**：
> - `Q4_K_M`：约 4.5bit，显存占用最低，性价比最高，**推荐**；
> - `Q5_K_M` / `Q6_K`：精度略高、显存略增；
> - `Q8_0`：接近原精度，显存接近 fp16。
>
> 粗略估计：N bit 量化，模型显存 ≈ 参数量 × bit / 8（GB）。例如 8B 模型
> Q4 ≈ 4.5GB、Q8 ≈ 8GB；再加上下文 KV 缓存，T4(16GB)/G4 均能容纳 8B~30B 的 Q4。

---

## A3. 启动 llama-server

`llama-server` 是内置 HTTP 服务器，直接提供 OpenAI 兼容 API：

```bash
./llama-server \
  -m ./models/qwen3-8b-q4km.gguf \   # GGUF 模型路径
  --host 0.0.0.0 --port 30000 \        # 监听地址/端口
  --api-key "sk-你的密钥" \            # 鉴权(可选, 建议开启)
  --ctx-size 8192 \                   # 上下文长度
  -ngl 999                            # 全部层放 GPU(-ngl 999 即全量 GPU)
```

看到日志 `server is listening on ...` 即就绪，健康检查：

```bash
curl http://localhost:30000/health   # 返回 200 即正常
```

> 后台托管（Colab terminal 为 tmux，无法嵌套 tmux，用 setsid 脱离终端）：
> ```bash
> setsid ./llama-server -m ... --host 0.0.0.0 --port 30000 \
>   --api-key "sk-..." --ctx-size 8192 -ngl 999 \
>   >> logs/llama_server.log 2>&1 </dev/null &
> ```

---

## A4. llama.cpp 参数详解

| 参数 | 说明 |
|---|---|
| `-m, --model` | GGUF 模型路径（必填） |
| `--host / --port` | 监听地址与端口（默认 `127.0.0.1:30000`，对外需 `0.0.0.0`） |
| `--api-key` | Bearer 鉴权密钥；不设则无鉴权 |
| `--ctx-size` | 上下文长度（token）；越大越占显存 |
| `-ngl, --n-gpu-layers` | 放入 GPU 的层数；`999` = 全部放 GPU（显存够时最快） |
| `-b, --batch-size` | 推理批次大小，默认 2048 |
| `-ub, --ubatch-size` | 微批次大小（影响显存与速度），默认 512 |
| `--threads / --threads-batch` | CPU 线程数（KV 缓存、非 GPU 部分用） |
| `--mlock` | 锁内存防换页（可选） |
| `--jinja` | 用模型自带 chat template（GGUF 内含时推荐，支持工具调用） |
| `--parallel` | 并行请求数（并发），越多越占显存 |
| `--rope-scaling` / `--yarn-*` | 上下文外推（如超长上下文时） |

> **G4/T4 低显存调优**：
> - 显存紧张 → 用更低量化（Q4）、减小 `--ctx-size`、降 `-ngl` 让部分层跑 CPU；
> - 速度优先且显存够 → `-ngl 999` 全 GPU、`--parallel 1~4`、合理 `-ub`。

---

## A5. llama.cpp API 使用示例

llama-server 兼容 OpenAI API，base_url 为 `http://<主机>:30000/v1`（端口按启动参数）。

### 对话

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Authorization: Bearer sk-你的密钥" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.6, "top_p": 0.95, "max_tokens": 512
  }'
```

### Python 客户端

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="sk-你的密钥")
resp = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.6, top_p=0.95,
)
print(resp.choices[0].message.content)
```

---

# Part B：SGLang

## B1. 安装 SGLang

SGLang 默认发布 **CUDA 13** 版本。若 Colab 驱动足够（≥ 580，支持 CUDA 13）可直接用默认版本，
无需系统安装 CUDA Toolkit（pip 包自带运行时）。

> **Colab G4/T4 注意**：若驱动/环境不支持 CUDA 13，可安装适配当前 CUDA 的版本，
> 例如 `pip install sglang[all]` 或按 SGLang 官方指引选择对应 cu 版本。

### B1.1 创建虚拟环境（必须 Python 3.12，原因见下）

```bash
# venv 建在项目外(默认 /tmp/sglang/venv), 避免数 GB 依赖混进项目目录不便复制
uv venv /tmp/sglang/venv --python 3.12
```

> 路径由 `SGLANG_VENV_DIR` 控制（`colab.sh install sglang` 与 `sglang/launch.sh` 共用同一默认值
> `/tmp/sglang/venv`）。手动执行上面这条命令时请与脚本保持一致，否则 `launch.sh` 找不到 venv。

> **为什么不用系统的 Python 3.13？**
> 依赖 `outlines-core==0.1.26` 只发布到 cp312 的预编译 wheel。Python 3.13 会回退源码编译，
> 而编译需要 Rust 工具链（`error: can't find Rust compiler`）。用 3.12 直接装 wheel 最省事。

### B1.2 安装

```bash
source /tmp/sglang/venv/bin/activate
env -u UV_SYSTEM_PYTHON uv pip install --prerelease=allow sglang
```

### B1.3 验证

```bash
python -c "
import torch, sglang
print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
print('gpu:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('sglang:', sglang.__version__)
"
# 期望: torch: 2.x+cuXXX | cuda: 对应版本
#       gpu: NVIDIA T4 / NVIDIA ... (8,9 / 7,5 等)
#       sglang: 0.x
```

---

## B2. 显卡的坑与修复

### 坑一：环境变量 `UV_SYSTEM_PYTHON=true`

该变量会让 `uv pip install` **无视已激活的 venv**，始终安装到系统 Python。
表现为：明明激活了 3.12 的 venv，uv 却输出 `Using Python 3.13 environment at: /usr`。

**修复**：安装时去掉该变量：

```bash
env -u UV_SYSTEM_PYTHON uv pip install --prerelease=allow sglang
```

### 坑二：FlashInfer 架构检测失败

启动时报错：

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

根因：FlashInfer 查询 CUDA 版本时优先调用系统 nvcc。Blackwell（SM 12.x）需要
CUDA ≥ 12.9 才能归一化出 `12.0f`，系统 nvcc 更旧（如 12.8）时架构集合被置空，
JIT 编译器误判为"显卡太旧"。

**修复**：`launch.sh` 已内置处理，核心是导出 FlashInfer 真正读取的变量（无 `SGLANG_` 前缀），
并按 `nvidia-smi` 的 compute_cap + nvcc 版本推导架构后缀（带后缀的值会被原样采用，跳过版本检查）：

```bash
export FLASHINFER_CUDA_ARCH_LIST="12.0f"   # Blackwell SM 12.0 且 nvcc >= 12.9
export FLASHINFER_CUDA_ARCH_LIST="12.0a"   # Blackwell SM 12.0 但 nvcc < 12.9（如系统 CUDA 12.8）
# export FLASHINFER_CUDA_ARCH_LIST="8.9"    # G4 (Ada, sm_89, 无后缀)
# export FLASHINFER_CUDA_ARCH_LIST="7.5"    # T4 (Turing, sm_75, 无后缀)
```

> 一般无需手动设置：脚本按 `nvidia-smi` 自动推导，并在启动日志打印实际取值。
> 不要为凑 `12.0f` 把 `CUDA_HOME` 指到 venv 的 cuda 包 —— 其头文件（CUDART 13.0）与自带
> nvcc（13.4）版本不一致，cccl 会 `#error` 拒绝编译；用系统 CUDA 12.8 + `12.0a` 即可正常 JIT。

---

## B3. 启动脚本

项目在 `sglang/` 子目录提供 `launch.sh` 一键管理服务（先 `cd sglang` 再执行）：

```bash
cd sglang
./launch.sh start     # 后台启动（setsid 脱离终端）
./launch.sh stop      # 优雅停止，超时后强杀
./launch.sh restart   # 重启
./launch.sh status    # 进程 + 健康检查
./launch.sh logs      # 实时跟踪日志
./launch.sh keep      # 守护模式：崩溃自动拉起
```

一键初始化环境（装 uv / 建 venv / 装 sglang，**不自动启动**）：

```bash
./colab.sh install sglang
```

venv 默认建在**项目外**的 `/tmp/sglang/venv`（依赖数 GB，放项目里不便复制/备份），用 `SGLANG_VENV_DIR` 可改位置：

```bash
SGLANG_VENV_DIR=/content/venvs/sglang ./colab.sh install sglang
```

并发压测（根目录 `bench.py`，sglang / llama.cpp 通用；参数透传）：

```bash
./colab.sh sglang bench -n 32 --max-tokens 256
make sglang-bench BENCH_ARGS="-n 32 --max-tokens 256"
```

脚本为**通用模型**设计：served 模型名、推理/工具调用解析器、chat template、
上下文长度、mamba 参数、EAGLE 投机解码等均由模型 `config.json` 自动推导，
也可用同名 `SGLANG_*` 环境变量覆盖（见 B4）。进入 `sglang/` 时 `.envrc` 自动加载
GPU profile（`.env.g4`/`.env.t4`），日志统一写在根目录 `logs/sglang_server.log`。

### API 密钥（鉴权）

服务通过 `--api-key` 启用 Bearer 鉴权，密钥来自环境变量 `SGLANG_API_KEY`
（未设置时回退 `API_KEY`；建议写入 `.env`，由 direnv 的 `.envrc` 加载）。
**脚本自身不内置密钥，两个变量均未设置时 `start` 会报错退出**：

```bash
cd sglang
SGLANG_API_KEY='sk-你的密钥' ./launch.sh start
# 或写入 .env 后(经 direnv 加载): ./launch.sh start
```

> 客户端请求**必须**带 `Authorization: Bearer <密钥>`，否则返回 401。
> 临时关闭鉴权：`SGLANG_API_KEY="" ./launch.sh start`。

---

## B4. SGLang 参数详解

最终配置综合了 llama.cpp 实践参数与 SGLang 官方推荐。核心参数（脚本按模型自动推导）：

| 参数 | 取值 | 说明 |
|---|---|---|
| `--model-path` | `Qwen/Qwen3.8-27B`（默认） | 由 `SGLANG_MODEL_REPO` 控制（未设置回退 `MODEL_REPO`）；HF ID 自动下载或本地路径 |
| `--served-model-name` | 模型路径末段（默认） | API 里的模型别名，**自动推导**；可用 `SGLANG_SERVED_NAME` 覆盖 |
| `--attention-backend` | `flashinfer` | 默认注意力后端 |
| `--kv-cache-dtype` | `fp8_e4m3` | KV 量化为 8bit，KV 容量翻倍（省显存，适合 G4/T4） |
| `--chunked-prefill-size` | `2048` | prefill 分块，降低长输入首 token 延迟抖动 |
| `--context-length` | 模型 `max_position_embeddings`（默认） | **自动推导**；可用 `SGLANG_CTX` 覆盖（默认 0=自动）；KV 池按剩余显存分配，不预占 |
| `--mem-fraction-static` | `0.90` | 静态显存占比。**G4/T4 OOM 时降到 0.85 试探**（由 `SGLANG_MEM_FRACTION_STATIC` 控制；`.env.g4`/`.env.t4` 提供 0.85/0.80） |
| `--reasoning-parser` | 按家族推导 | **自动推导**（qwen→`qwen3`、deepseek→`deepseek_v3`、glm→`glm45`）；可用 `SGLANG_REASONING_PARSER` 覆盖 |
| `--tool-call-parser` | 按家族推导 | **自动推导**（qwen→`qwen3_coder` 等）；可用 `SGLANG_TOOL_CALL_PARSER` 覆盖 |
| `--default-chat-template-kwargs` | 按家族推导 | **自动推导**（qwen→`{"enable_thinking": true}`）；可用 `SGLANG_CHAT_TEMPLATE_KWARGS` 覆盖 |
| `--mm-feature-transport` | `cpu` | 多模态特征经 CPU 中转 |
| `--speculative-algorithm EAGLE` 等 | steps=3/topk=1/draft=4 | **自动推导**：仅当模型含 MTP 层时注入 EAGLE；可用 `SGLANG_SPECULATIVE_ALGORITHM` 覆盖 |
| `--max-running-requests` | `48`（仅启用投机解码时注入） | 投机解码时的并发上限；可用 `SGLANG_MAX_RUNNING_REQUESTS` 覆盖 |
| `--enable-cache-report` | — | usage 中报告前缀缓存命中 |
| `--enable-metrics` | — | Prometheus 指标 `/metrics` |

> **G4/T4 低显存调优**：
> - 显存不够 → `--mem-fraction-static` 降到 `0.85`（防 OOM 但减少 KV 容量）；
> - KV 已是 fp8，若仍紧张且模型不支持则回退 bf16 排查（见 FAQ Q5）；
> - 小显存时选更小的模型（如 Qwen3-8B 而非 27B）。

**自动推导规则（来自 HF 缓存/本地 `config.json`）**：

| 参数 | 推导逻辑 | 覆盖变量 |
|---|---|---|
| `--served-model-name` | 默认=模型路径末段（小写、`/`→`-`） | `SGLANG_SERVED_NAME` |
| `--reasoning-parser` / `--tool-call-parser` | 按 `architectures`/`model_type` 家族 | `SGLANG_REASONING_PARSER` / `SGLANG_TOOL_CALL_PARSER` |
| `--default-chat-template-kwargs` | qwen 家族默认 `{"enable_thinking": true}` | `SGLANG_CHAT_TEMPLATE_KWARGS` |
| `--context-length` | 读取 `max_position_embeddings` | `SGLANG_CTX`（默认 0=自动） |
| `--mamba-*` 三参数 | 仅当模型为 mamba/SSM 类时注入 | — |
| `--speculative-algorithm EAGLE` | 仅当 config 含 MTP 层时注入 | `SGLANG_SPECULATIVE_ALGORITHM` |
| `--max-running-requests` | 仅当启用投机解码时注入，默认 48 | `SGLANG_MAX_RUNNING_REQUESTS` |

> 上述变量在 `.envrc` 中均为空默认值，留空即走自动推导；在 `.env` 或运行时设置同名变量可强制覆盖。

首次启动流程（权重已缓存约需 2~4 分钟；冷启动另需下载权重）：

```
下载/加载权重 → 初始化分布式 → 分配 KV 缓存(fp8) → 捕获 CUDA graph
→ FlashInfer JIT 编译(仅首次, 几分钟) → warmup 请求 → 就绪
```

看到日志中的 `The server is fired up and ready to roll!` 即就绪，
或 `curl http://localhost:30000/health` 返回 200 确认。

---

## B5. SGLang API 使用示例

服务兼容 OpenAI API，base_url 为 `http://<主机>:30000/v1`。
**所有请求都需带 `Authorization: Bearer <密钥>`**。

### B5.1 对话（思考模式）

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Authorization: Bearer $SGLANG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "证明 sqrt(2) 是无理数"}],
    "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096
  }'
```

返回中 `reasoning_content` 为思考过程，`content` 为最终回答。

### B5.2 单次请求关闭思考

```bash
# messages 不变，追加:
"chat_template_kwargs": {"enable_thinking": false}
```

### B5.3 流式输出

```bash
curl -N http://localhost:30000/v1/chat/completions \
  -H "Authorization: Bearer $SGLANG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b","stream":true,"max_tokens":512,"messages":[{"role":"user","content":"写一首诗"}]}'
```

### B5.4 Python 客户端

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key=os.environ.get("SGLANG_API_KEY") or os.environ["API_KEY"])
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.6, top_p=0.95,
)
print(resp.choices[0].message.reasoning_content)  # 思考
print(resp.choices[0].message.content)            # 回答
```

---

# Part C：vLLM

## C1. 安装 vLLM

项目通过 uv 创建独立的 Python 3.12 venv，并按官方 quickstart 用 `--torch-backend=auto` 自动选择匹配后端，安装最新 vLLM，不固定版本：

```bash
./colab.sh install vllm
```

默认 venv 为 `/tmp/vllm/venv`，可通过 `VLLM_VENV_DIR` 修改：

```bash
VLLM_VENV_DIR=/content/venvs/vllm ./colab.sh install vllm
```

安装脚本会在完成后验证 `vllm serve --help`。vLLM 对 Python、PyTorch、CUDA 驱动和 GPU 架构较敏感；如果官方最新版本与当前 Colab 驱动不兼容，应根据驱动环境选择合适的 vLLM/CUDA 安装组合，而不是复用 SGLang 的 venv。

## C2. vLLM 启动脚本

`vllm/launch.sh` 使用官方推荐的 `vllm serve`，并提供与 SGLang 对称的服务管理：

```bash
./colab.sh vllm start
./colab.sh vllm stop
./colab.sh vllm restart
./colab.sh vllm status
./colab.sh vllm test
./colab.sh vllm logs
./colab.sh vllm keep
```

启动前会把 HF ID 下载到本地模型目录；已有 `config.json` 时直接复用，不会让 vLLM 在后台把权重下载到不可控位置。服务默认监听 `0.0.0.0:30000`，日志为 `logs/vllm_server.log`，启动命令追加记录到 `logs/launch_cmd.log`。

示例：

```bash
export VLLM_MODEL_REPO=Qwen/Qwen3-8B
export VLLM_API_KEY=sk-your-key
./colab.sh vllm start
./colab.sh vllm status
```

`VLLM_API_KEY` 未设置时回退 `API_KEY`；显式设置 `VLLM_API_KEY=""` 可关闭鉴权。两个变量都未设置时，`start` 会拒绝启动，避免无意中暴露公网服务。

## C3. vLLM 参数详解

启动脚本根据环境变量构造参数数组，避免 shell 字符串拼接造成参数错位：

| vLLM 参数 | 环境变量 | 说明 |
|---|---|---|
| `--model`（位置参数） | `VLLM_MODEL_REPO` | HF 模型 ID 或本地模型目录 |
| `--served-model-name` | `VLLM_SERVED_NAME` | API 中的模型名，默认取路径末段小写 |
| `--host` / `--port` | `VLLM_HOST` / `VLLM_PORT` | 默认 `0.0.0.0:30000` |
| `--api-key` | `VLLM_API_KEY` / `API_KEY` | Bearer 鉴权；空值不传参数 |
| `--gpu-memory-utilization` | `VLLM_GPU_MEMORY_UTILIZATION` | GPU 显存使用比例；G4 默认 0.90、T4 默认 0.80 |
| `--max-model-len` | `VLLM_MAX_MODEL_LEN` | 最大上下文长度；留空使用模型/vLLM 默认值 |
| `--max-num-seqs` | `VLLM_MAX_NUM_SEQS` | 限制并发序列数，默认 `512`；显存不足时可继续降低 |
| `--max-num-batched-tokens` | `VLLM_MAX_NUM_BATCHED_TOKENS` | 限制单批 token，控制显存与延迟 |
| `--tensor-parallel-size` | `VLLM_TENSOR_PARALLEL_SIZE` | 多 GPU 张量并行，默认 1 |
| `--dtype` | `VLLM_DTYPE` | 默认 `auto`，也可用 `bfloat16` 等 |
| `--quantization` | `VLLM_QUANTIZATION` | 可选量化后端 |
| `--enable-prefix-caching` | `VLLM_ENABLE_PREFIX_CACHING=1` | 启用前缀缓存 |
| `--trust-remote-code` | `VLLM_TRUST_REMOTE_CODE=1` | 模型需要自定义代码时显式开启 |
| `--enforce-eager` | `VLLM_ENFORCE_EAGER=1` | CUDA graph 不兼容时排查使用 |
| `--generation-config` | `VLLM_GENERATION_CONFIG` | 留空用模型 `generation_config.json`（默认）；设为 `vllm` 用 vLLM 默认采样参数 |
| `--enable-auto-tool-choice` | `VLLM_ENABLE_AUTO_TOOL_CHOICE` | 自动工具调用，默认开启；设为 `0` 关闭 |
| `--tool-call-parser` | `VLLM_TOOL_CALL_PARSER` | 工具调用解析器；留空时按 `config.json` 模型家族推导 |

带 `tool_choice: "auto"` 的 coding agent 请求需要同时开启 `--enable-auto-tool-choice` 和 `--tool-call-parser`，否则 vLLM 返回 `400 --enable-auto-tool-choice and --tool-call-parser to be set`。脚本已默认开启两者；模型家族无法识别时会打印警告并跳过，此时用 `VLLM_TOOL_CALL_PARSER` 显式指定（如 `qwen3_coder`、`hermes`、`pythonic`）。

例如显存不足时：

```bash
VLLM_MAX_MODEL_LEN=8192 \
VLLM_GPU_MEMORY_UTILIZATION=0.75 \
./colab.sh vllm restart
```

## C4. vLLM API 与监控

vLLM 提供 OpenAI 兼容 API：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 256
  }'
```

就绪和指标端点：

```bash
curl http://localhost:30000/health
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:30000/metrics
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:30000/v1/models
```

`/health` 返回 HTTP 200 表示服务已就绪。vLLM 原生提供 Prometheus `/metrics`，不需要添加 SGLang 的 `--enable-metrics` 参数；`bench.py` 采样 `num_requests_running` 和 `num_requests_waiting`。

## C5. vLLM 压测与调优

```bash
./colab.sh vllm bench -n 8 --max-tokens 128
make vllm-bench BENCH_ARGS="-n 8 --max-tokens 128"
```

压测请求走 `/v1/chat/completions`，模型未显式指定时从 `/v1/models` 自动获取。vLLM 开启 API key 时，压测脚本会将 `VLLM_API_KEY`（或 `--api-key`）带到 API 和 `/metrics` 请求中。

调优顺序建议：

1. 先选能完整放入本地 GPU 的 HF 模型/量化后端。
2. OOM 时先降低 `VLLM_MAX_MODEL_LEN`，再降低 `VLLM_GPU_MEMORY_UTILIZATION` 或 `VLLM_MAX_NUM_SEQS`。
3. 追求吞吐时在显存允许范围内增加并发和批 token 限制，并比较聚合 tok/s、最大延迟和 waiting 峰值。
4. 使用 `VLLM_TRUST_REMOTE_CODE=1` 前确认模型仓库可信；默认不启用该选项。

### 两个常见启动失败

**`FlashInfer requires GPUs with sm75 or higher`**：Blackwell（SM 12.x）在系统 nvcc 较旧时会探测失败。`launch.sh` 会按 compute capability 与 nvcc 版本导出 `FLASHINFER_CUDA_ARCH_LIST`（SM 12.0 + nvcc ≥ 12.9 为 `12.0f`，否则 `12.0a`）；日志会打印实际取值。

**`max_num_seqs (1024) exceeds available Mamba cache blocks`**：混合 mamba 模型需要为每个并发序列预留一个 Mamba cache block，vLLM 默认 1024 超过可用数量。脚本默认传 `--max-num-seqs 512`；仍失败时降低 `VLLM_MAX_NUM_SEQS`，或提高 `VLLM_GPU_MEMORY_UTILIZATION`。

**`Unknown vLLM environment variable detected: VLLM_*`**：`VLLM_GPU_MEMORY_UTILIZATION` 等只是本脚本的配置项，会被 vLLM 当作未知环境变量告警。脚本启动服务时用 `env -u` 剥离这些变量（值已通过命令行参数传入），因此新启动日志不应再出现该告警。

**`Default vLLM sampling parameters have been overridden by the model's generation_config.json`**：这是模型作者推荐的采样参数（Qwen 为 temperature 1.0 / top_k 20 / top_p 0.95），按默认保留即可；想改用 vLLM 默认采样时设 `VLLM_GENERATION_CONFIG=vllm`。

## 5. 模型存放：冷存储（Google Drive）vs 本地盘

> **结论：不要让引擎直接从 Drive 加载权重。** Drive 在 Colab 里是 FUSE 挂载，
> 顺序读只有几十 MB/s 且抖动大；而 llama.cpp / SGLang 加载权重用的是 `mmap` 随机读，
> 延迟在 FUSE 上会被放大 —— 30GB 的权重可能从数十秒变成十几分钟，表现得像卡死
> （挂载再抖一下，进程还可能进入 uninterruptible sleep，kill 都杀不掉）。

因此引擎**不支持把 Drive 用作模型目录**：模型目录（含通过 `MODEL_ROOT` / `LLAMA_MODEL_ROOT` /
`SGLANG_MODEL_ROOT` 间接指向）以 `/content/drive` 开头时，`launch.sh start` 直接报错退出；
脚本也**不会自动复制/降级**任何文件。Drive 只作为冷存储，权重的搬运完全由手动
`./colab.sh sync` 完成（见下）。

```bash
# 根 .envrc（两引擎共用）—— 模型一律放本地盘
export MODEL_ROOT="/content/models"                    # 本地模型盘：引擎从这里加载

# ./colab.sh sync 专用（默认值即可用，无需显式设置）
export MODEL_DRIVE_ROOT="/content/drive/MyDrive/hf-models"   # Drive 冷存储：只存权重
export MODEL_LOCAL_ROOT="/content/models"                    # 本地工作盘：sync 的本地端
```

行为要点：

| 场景 | 做法 |
|---|---|
| Drive 冷存储里有模型，本地没有 | `./colab.sh sync pull <模型名> --quant <档位>` 拉到本地盘后再启动 |
| 本地下好了模型，要持久化到 Drive | `./colab.sh sync push <模型名>` 回存冷存储 |
| 引擎启动 | 只读本地盘（`MODEL_ROOT`），不读 Drive，不自动复制 |

### 手动双向同步：`./colab.sh sync`

`sync` 是本地盘与 Drive 冷存储之间唯一的搬运途径（逐个模型目录 rsync，手动触发）：

```bash
./colab.sh sync pull -n                        # 预览：Drive 上有哪些模型会拉到本地
./colab.sh sync pull Qwen3.8-27B-GGUF --quant UD-Q8_K_XL   # 只拉这一个量化档（推荐）
./colab.sh sync push Qwen3.8-27B-GGUF          # 把本地下好的权重回存到 Drive
./colab.sh sync all                            # 双向各取较新的一方（先 pull 再 push）
```

**从云端取回时请带 `--quant <档位>`**：只同步 `*-<档位>-*.gguf` / `*-<档位>.gguf`，
免得把目录里 BF16 等几十 GB 的其它档位一起搬下来。省略 `--quant` 会同步整个模型目录，
此时脚本会打印提示——SGLang 的 safetensors 权重没有档位概念，那种情况就该省略。

| 动作 | 方向 |
|---|---|
| `pull` | Drive → 本地盘 |
| `push` | 本地盘 → Drive |
| `all` | 先 pull 再 push，两边各取较新的一方 |

安全约定（权重是几十 GB 的资产，宁可少同步也绝不覆盖/删除更新的那一份）：
- 全程 `rsync -u`：**目标端已有且比源端新的文件一律跳过**，双向都不会用旧版本覆盖新版本；
- **不用 `--delete`**：目标端多出来的文件保留，只补不删；
- 跳过 `.cache/`（HF 下载产生的临时数据，本地续传用，不参与同步）；
- 源目录不存在（该模型只在另一端）时跳过不报错。

建议大批量操作前先加 `-n/--dry-run` 预览（预览不会创建任何目录、不传输任何文件）。

`pull` / `all` 在本地工作盘目录不存在时会自动创建（首次同步的常见情况）；`push` 时本地盘是源
目录，不存在则报错——没有模型可推。

---

## 6. 运维与监控

```bash
# llama.cpp (端口 30000)
tail -f ./logs/llama_server.log           # 日志(根目录 logs/)
curl http://localhost:30000/health        # 健康

# SGLang (端口 30000)
tail -f ./logs/sglang_server.log          # 日志(根目录 logs/)
curl http://localhost:30000/health        # 健康
curl http://localhost:30000/metrics       # Prometheus 指标

# vLLM (端口 30000)
tail -f ./logs/vllm_server.log            # 日志(根目录 logs/)
curl http://localhost:30000/health        # 健康
curl -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:30000/metrics

nvidia-smi                               # 显存占用
```

---

## 7. 常见问题 FAQ

**Q1: Colab GPU 到底是 G4 还是 T4？**
型号不固定，用 `nvidia-smi` 或 `python -c "import torch;print(torch.cuda.get_device_name(0))"`
查询。不同型号架构不同（G4=Ada sm_89、T4=Turing sm_75、Blackwell=sm_120），影响
`FLASHINFER_CUDA_ARCH_LIST` 与 llama.cpp 的 `-DCMAKE_CUDA_ARCHITECTURES` 取值。

**Q2: llama.cpp 启动报 CUDA 相关错误？**
确认使用的是 CUDA 源码编译版；执行 `./colab.sh install llama --build`（或用 `-DGGML_CUDA=on` 手动重编）；
`-ngl 999` 全 GPU 前确认显存足够，不足则减小 `-ngl` 或降量化。

**Q3: 显存不足（OOM）怎么办？**
- llama.cpp：降 GGUF 量化（Q4）、减小 `--ctx-size`、减小 `-ngl`、降 `--parallel`；
- SGLang：降 `--mem-fraction-static` 到 0.85、KV 用 fp8、换更小模型；
- vLLM：降低 `VLLM_MAX_MODEL_LEN`、`VLLM_GPU_MEMORY_UTILIZATION` 或 `VLLM_MAX_NUM_SEQS`，必要时换更小/量化模型。

**Q4: 选 llama.cpp、SGLang 还是 vLLM？**
显存小/图省事选 llama.cpp；要 SGLang 的解析器、投机解码等功能选 SGLang；要直接运行 HF 模型并使用标准 vLLM 服务选 vLLM。

**Q5: SGLang 启动报 `FlashInfer requires GPUs with sm75 or higher`？**
架构没传对，或 nvcc 太旧编不出 `compute_120f`：需 `FLASHINFER_CUDA_ARCH_LIST`（注意无
`SGLANG_` 前缀）+ CUDA ≥ 12.9 的 nvcc。见 B2 坑二。

**Q6: uv 安装时装到了系统 Python？**
`UV_SYSTEM_PYTHON=true` 在作怪，用 `env -u UV_SYSTEM_PYTHON` 前缀。见 B2 坑一。

**Q7: outlines-core 构建失败 `can't find Rust compiler`？**
Python 3.13 无预编译 wheel。改用 Python 3.12 建 venv。见 B1.1。

**Q8: SGLang 并发数只有 7，怎么提高？**
Mamba 状态缓存限制。调大 `--mamba-full-memory-ratio`（如 0.5），或显式指定
`--max-mamba-cache-size`。代价是 KV 池变小、长上下文容量下降。

**Q9: SGLang 日志警告 "Using FP8 KV cache but no scaling factors provided"？**
模型 checkpoint 未携带 KV 量化系数，SGLang 按 1.0 处理。通常可用；
若发现长文注意力异常，可去掉 `--kv-cache-dtype fp8_e4m3` 回退 bf16。

**Q10: 首次请求特别慢？**
FlashInfer 首次 JIT 编译内核（几分钟，之后有缓存）；llama.cpp 首次加载 GGUF、vLLM 首次加载权重和 CUDA graph warmup 也需时间。
均属正常。

**Q11: 请求返回 `401 Unauthorized`？**
服务已启用 Bearer 鉴权，客户端必须带 `Authorization: Bearer <密钥>`。
密钥由 `SGLANG_API_KEY`、`VLLM_API_KEY` 或 `LLAMA_API_KEY`（各自回退 `API_KEY`）决定。显式设置引擎密钥为空可关闭该引擎鉴权。

**Q12: 上下文不够长？**
- llama.cpp：增大 `--ctx-size`（显存有限，配合 YARN/rope-scaling 可外推）；
- SGLang：`--context-length` 按模型自动推导，可用 `SGLANG_CTX` 覆盖（默认 0=自动）；
- vLLM：用 `VLLM_MAX_MODEL_LEN` 设置上限；上下文越长 KV 越占显存，需在显存范围内权衡。

**Q13: 模型放在 Google Drive，加载特别慢 / 看着像卡死？**
权重不该从 Drive 加载（FUSE 上的 mmap 随机读极慢），引擎也**不支持** Drive 作为模型目录——
模型目录以 `/content/drive` 开头时启动直接报错。先用 `./colab.sh sync pull <模型名>
--quant <档位>` 把权重取到本地盘（默认 `/content/models`）再启动。详见
[第 5 节](#5-模型存放冷存储google-drive-vs-本地盘)。
