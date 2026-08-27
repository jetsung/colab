# colab-sglang

基于 **SGLang** 部署大模型并对外提供 **OpenAI 兼容 API** 的一键脚本集。

脚本为**通用模型**设计：启动参数（served 模型名、推理/工具调用解析器、chat template、上下文长度、是否 mamba）均依据 HF 缓存/本地目录中模型的 `config.json` **自动推导**，不再写死某一具体模型；上述任一参数也都能用同名 `SGLANG_*` 环境变量显式覆盖。默认模型为 `Qwen/Qwen3.8-27B`。

支持深度思考（thinking）、工具调用、EAGLE(MTP) 投机解码。宿主机通过 Colab CLI 创建 GPU 会话作为运行环境，服务跑在 Colab terminal 中，经 **bore** 隧道（反向代理）把本地 30000 端口暴露到公网。

> 运行位置区分：`colab.sh vps` 在**宿主机**运行（根目录）；`colab.sh setup` / `colab.sh bore` 在 **Colab terminal** 运行（根目录）；`launch.sh` 在 **Colab terminal** 运行（`sglang/` 子目录）。
> Colab terminal 本身是 tmux 环境（无法再嵌套 tmux），长驻服务统一用 **setsid** 后台托管（脱离终端，SSH 断开不受影响）：`launch.sh` 与 `colab.sh bore` 均采用此方式，日志统一写入根目录 `logs/`。

## 文件说明

| 文件 | 位置 | 运行位置 | 作用 |
|---|---|---|---|
| `colab.sh vps` | 根目录 | 宿主机 | 安装 Google Colab CLI，创建 GPU 会话并挂载 Google Drive |
| `colab.sh setup` | 根目录 | Colab terminal | 前置依赖安装：direnv、bore、relaydrop、opencode |
| `colab.sh install sglang` | 根目录 | Colab terminal | 装环境：装 uv → 建 Python 3.12 venv → 装 SGLang（含已知坑修复）；**不自动启动服务**，启动请另跑 `./launch.sh start` |
| `launch.sh` | `sglang/` | Colab terminal | 服务管理：`start` / `stop` / `restart` / `status` / `logs` / `keep`（守护） |
| `.envrc` | `sglang/` | Colab terminal | direnv：继承根 `.envrc`，按 `GPU_PROFILE`（或自动探测）加载 `.env.g4`/`.env.t4`，并声明 `SGLANG_*` 空默认（留空即自动推导） |
| `.env.g4` / `.env.t4` | `sglang/` | Colab terminal | GPU profile（`SGLANG_FLASHINFER_CUDA_ARCH_LIST` / `SGLANG_MEM_FRACTION_STATIC` / `SGLANG_CONTEXT_LENGTH` 等） |
| `colab.sh bore` | 根目录 | Colab terminal | bore 公网隧道：`30000 → ${BORE_PORT:-65535}`，setsid 后台托管，日志 `logs/bore.log` |
| `sglang.ipynb` | `sglang/` | Colab | 部署流程 Notebook 版：等价于在 terminal 依次执行 setup/install/launch/bore，另含 .env 加载、就绪等待与 API 验证单元格；上传后 Run All 即可，项目路径可自定义（第 0 节填写）或按 环境变量→工作目录→子目录→Drive 自动探测 |
| `.envrc`（根） | 根目录 | Colab terminal | direnv 入口：`PATH_add`、`API_KEY` / `MODEL_REPO` / `BORE_PORT` 默认值、`dotenv` 加载 `.env`（`.env` 不入 git） |
| `.env` | 根目录 | — | 实际密钥/凭据值（bore / relaydrop / SGLang / opencode），已被 `.gitignore` 忽略 |
| `DOCS.md` | 根目录 | — | 完整部署教程（llama.cpp / SGLang）：安装坑修复、参数详解、API 示例、FAQ |

## 快速开始

```bash
# 0. 宿主机: 创建 Colab GPU 会话并挂载 Drive, 然后连进 Colab terminal
./colab.sh vps create

# 1. Colab terminal(根目录): 安装前置依赖, 加载环境变量
./colab.sh setup all
source ~/.bashrc

# 2. 安装 SGLang 环境(uv → venv → sglang; 不自动启动)
#    需先在根目录 .env 中写入密钥(SGLANG_API_KEY 或 API_KEY, 未设置时启动会报错)
./colab.sh install sglang

# 3. 启动服务(进入 sglang/ 后 direnv 自动加载 GPU profile)
cd sglang
./launch.sh start

# 4. 等待就绪(看到 HTTP 200 / "ready to roll" 即可)
./launch.sh status

# 5. 暴露到公网(回根目录, 隧道输出中的公网地址即入口)
cd ..
./colab.sh bore start
./colab.sh bore logs    # 查看隧道输出(含公网地址)
```

### 服务管理

```bash
./launch.sh start    # 后台启动（setsid，SSH 断开不受影响）
./launch.sh stop     # 优雅停止，超时强杀
./launch.sh restart  # 重启
./launch.sh status   # 进程 + 健康检查
./launch.sh logs     # 实时跟踪日志
./launch.sh keep     # 守护模式：崩溃自动拉起（每 30s 检查）
```

日志：根目录 `logs/sglang_server.log`（PID 文件 `sglang/sglang.pid`）。

### bore 隧道管理

```bash
./colab.sh bore start     # 后台启动（setsid，SSH 断开不受影响）
./colab.sh bore stop      # 优雅停止，超时强杀
./colab.sh bore restart   # 重启
./colab.sh bore status    # 进程状态
./colab.sh bore logs      # 实时跟踪日志（含公网地址）
```

公网端口由环境变量 `BORE_PORT` 控制，默认 `65535`（在 `.envrc` 中定义，可在 `.env` 或运行时覆盖）；也可用 `--port` 临时指定（优先级高于 `BORE_PORT`）：

```bash
./colab.sh bore start --port 12345   # 临时改用 12345
# 或写入 .env 持久化: BORE_PORT=12345
```

### 调用 API

服务监听 `0.0.0.0:30000`，走公网时 base_url 为 `http://<bore 公网地址>:65535/v1`。
所有请求必须携带 `Authorization: Bearer <密钥>`（密钥为 `SGLANG_API_KEY`，未设置时回退 `.env` 中的 `API_KEY`）。
请求里的 `model` 字段需与 served 名一致——默认取**模型路径末段**（如 `Qwen/Qwen3.8-27B`→`qwen3.8-27b`），可用 `SGLANG_SERVED_NAME` 覆盖。

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Authorization: Bearer $SGLANG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.6, "top_p": 0.95
  }'
```

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key=os.environ.get("SGLANG_API_KEY") or os.environ["API_KEY"])
resp = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.6, top_p=0.95,
)
print(resp.choices[0].message.reasoning_content)  # 思考过程
print(resp.choices[0].message.content)            # 最终回答
```

## 环境变量

**根 `.envrc`** 保留 `PATH_add`、`API_KEY` / `MODEL_REPO` / `BORE_PORT` 默认值与 `dotenv`，其余变量值从同目录 `.env` 加载（`.env` 含密钥，已被 `.gitignore` 忽略，**勿提交**）。修改 `.envrc` 后需重新执行 `direnv allow .`。

**`sglang/.envrc`** 继承根配置后，按 `GPU_PROFILE`（未设置时用 `nvidia-smi` 自动探测 G4/T4）加载 `.env.g4` / `.env.t4`，并声明以下 `SGLANG_*` 空默认（留空即由脚本按模型 `config.json` 自动推导）。

| 变量 | 说明 |
|---|---|
| `SGLANG_MODEL_REPO` | 模型路径（HF ID 或本地路径）；未设置时回退 `MODEL_REPO`（默认 `Qwen/Qwen3.8-27B`）；**两者均空时启动报错** |
| `SGLANG_SERVED_NAME` | API 中的模型别名，默认=模型路径末段（小写、斜杠转连字符） |
| `SGLANG_HOST` / `SGLANG_PORT` | 监听地址与端口（默认 `0.0.0.0` / `30000`） |
| `SGLANG_CONTEXT_LENGTH` | 最大上下文长度，留空时由 `config.json` 的 `max_position_embeddings` 推导，再回退模型默认 |
| `SGLANG_API_KEY` | 鉴权密钥；未设置时回退 `API_KEY`；两者均**未设置**时 `start` 报错（显式置空任一可关闭鉴权） |
| `SGLANG_REASONING_PARSER` / `SGLANG_TOOL_CALL_PARSER` | 推理/工具调用解析器，留空时由脚本按模型家族推导（qwen→`qwen3`/`qwen3_coder`、deepseek→`deepseek_v3`、glm→`glm45`，其余不注入交由 SGLang 自动识别）；显式设置可覆盖 |
| `SGLANG_CHAT_TEMPLATE_KWARGS` | chat template 参数（JSON），留空时按家族推导（qwen→`{"enable_thinking": true}`） |
| `SGLANG_SPECULATIVE_ALGORITHM` | 投机解码算法；留空时按 config.json 是否含 MTP 层自动判断（Qwen3→EAGLE）；置空=关闭 |
| `SGLANG_SPECULATIVE_NUM_STEPS` / `SGLANG_SPECULATIVE_EAGLE_TOPK` / `SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS` | EAGLE 投机解码参数（默认 3 / 1 / 4） |
| `SGLANG_MAX_RUNNING_REQUESTS` | 投机解码时的最大并发请求数（默认 48） |
| `SGLANG_MEM_FRACTION_STATIC` | 静态显存占比（默认 0.90；G4/T4 profile 提供 0.85/0.80，防 OOM） |
| `SGLANG_FLASHINFER_CUDA_ARCH_LIST` | FlashInfer/CUDA 架构（默认按 `nvidia-smi` 探测：G4→`8.9f`、T4→`7.5f`、Blackwell→`12.0f`） |
| `BORE_SECRET` / `BORE_SERVER` | bore 隧道凭据与中继服务器 |
| `RELAYDROP_RELAY` / `RELAYDROP_PASSWORD` | relaydrop 凭据 |
| `OPENCODE_API_KEY` | opencode 客户端密钥 |

## 文件传输（relaydrop）

`relaydrop` 用于在**宿主机 ↔ Colab** 之间快速传文件，典型场景是把本地的 `.env`（含密钥/凭据）推送到 Colab，省去手动粘贴。凭据为 `.env` 中的 `RELAYDROP_RELAY` / `RELAYDROP_PASSWORD`。

```bash
# 宿主机: 把 .env 推到 Colab 工作目录(需先 export 凭据或用 relaydrop 配置)
relaydrop put .env

# Colab terminal: 拉取
relaydrop get .env
```

> 传完 `.env` 后执行 `direnv allow .` 即可让 `SGLANG_API_KEY` / `BORE_PORT` 等变量生效。

## 运维

```bash
tail -f ./logs/sglang_server.log   # 服务日志(根目录 logs/)
curl http://localhost:30000/health # 健康检查
curl http://localhost:30000/metrics# Prometheus 指标
nvidia-smi                         # 显存占用
```

## 详细文档

安装踩坑（FlashInfer sm_120 架构检测、`UV_SYSTEM_PYTHON` 等）、完整启动参数详解、采样参数建议与 FAQ 见 [DOCS.md](../DOCS.md)。
