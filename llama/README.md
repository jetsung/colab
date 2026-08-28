# Qwen3.8-Flash-Next (GGUF) 部署教程 — llama.cpp (PR #27742)

在单张 NVIDIA GPU 上用 **llama.cpp** 运行 `unsloth/Qwen3.8-Flash-Next-GGUF`。
本教程已在 **RTX PRO 6000 Blackwell（96GB，sm_120）单卡** 上验证通过。

---

## 0. 为什么是这套方案

- **GGUF 是 llama.cpp 的格式，sglang 不支持 GGUF。** 要跑 GGUF 必须用 llama.cpp 系运行器（llama.cpp / Ollama / LM Studio / KoboldCpp）。
- **Qwen3.8-Flash-Next 的架构是 `qwen4exp`（混合注意力 QSA / Gated DeltaNet / n-gram embedding + 视觉编码器）。** 主线 `llama.cpp` 还不支持，必须编译 **PR #27742**（`model: add Qwen3.8-Flash-Next (qwen4exp)`，截至编写时仍未合并）。
- 该模型约 **125B 参数**（含 51B n-gram embedding + 4B MTP），全量 FP8 约 93GB，单卡 96GB 放不下；GGUF 量化后可放下。

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA，计算能力需被 CUDA 支持（Blackwell sm_120 需 CUDA ≥ 12.8） |
| 编译工具 | `git`、`cmake`(≥3.20)、`gcc`/`g++`、`make`、`nvcc` |
| 运行时 | Python 3（仅用于下载），`huggingface_hub` + `hf_xet`（经 **uv** 安装） |
| HuggingFace | 一个 **已接受模型许可证** 的 token（`HF_TOKEN`），否则 LFS/Xet 下载会被拒 |
| 磁盘 | 模型约 74GB（`UD-Q2_K_XL`）~105GB（`UD-Q4_K_XL`），请预留 ≥120GB（仅 Q2）或 ≥180GB（Q2+Q4 并存） |
| 显存 | 单卡 96GB 实测可跑 `UD-Q2_K_XL`（~74GiB VRAM）与 `UD-Q4_K_XL`（~80GiB VRAM，含 ~18GiB KV 余量）；中间档 `UD-Q3_K_XL` 同样可行 |

> ⚠️ **Xet 存储坑**：该仓库用 HuggingFace 新一代 **Xet** 存储。若没装 `hf_xet`，`hf download` 会卡在 11MB 的"指针文件"不动。`colab.sh install llama` 已自动安装 `hf_xet`（用 `uv pip install --system`）。

---

## 2. 安装（编译 llama.cpp）

```bash
./colab.sh install llama
```

脚本会：检查工具链 → 自动探测 CUDA 架构（如 12.0→`120`）→ `uv pip install huggingface_hub hf_xet` → 克隆 `ggml-org/llama.cpp` → checkout **PR #27742** → CMake 配置（CUDA）→ 编译。

产物：`/content/llama.cpp/build/bin/llama-server`。

可选环境变量：
- `LLAMA_DIR`：llama.cpp 目录（默认 `/content/llama.cpp`）
- `LLAMA_CUDA_ARCH`：强制指定架构（如 `120` / `90` / `89`），不填则自动探测
- `CLEAN=1 ./colab.sh install llama`：清掉 `build/` 重新编译

---

## 3. 准备 HuggingFace Token

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

1. 在 https://huggingface.co/settings/tokens 创建一个 **read** 权限的 token。
2. 打开 https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF （如需跳转到基础模型页也一并操作），点击 **Agree** 接受 `qwen-community-1.0` 许可证。
3. 把 token 导出到环境变量（或写进你的 shell profile）。

---

## 4. 启动服务

```bash
cd llama
./launch.sh start
```

脚本会：校验必要变量 → 若本地缺模型则自动下载（首次约 74GB）→ 用 `llama-server` 后台加载并监听 `0.0.0.0:30000`。

服务管理（与 `sglang/launch.sh` 一致）：

```bash
./launch.sh start      # 启动（后台 setsid 托管；首次自动下载模型）
./launch.sh stop       # 优雅停止，超时强杀
./launch.sh restart    # 重启
./launch.sh status     # 进程 + 健康检查（curl /health）
./launch.sh logs       # 实时跟踪日志
./launch.sh keep       # 守护模式：崩溃自动拉起
```

日志统一写在**项目根目录** `logs/llama_server.log`（PID 文件 `llama/llama.pid`）。

### 环境变量（均带 `LLAMA_` 前缀，未设置时回退通用变量）

| 变量 | 默认/回退 | 说明 |
|------|-----------|------|
| `LLAMA_MODEL_REPO` | 回退 `MODEL_REPO` | HF 仓库（**不可为空**） |
| `LLAMA_MODEL_NAME` | 从 REPO 提取（`/` 后去 `-GGUF`） | 模型名前缀（分片文件匹配） |
| `LLAMA_QUANT` | 无默认（**不可为空**） | 量化档位（由 `.env.g4`/`.env.t4` 或命令行提供） |
| `LLAMA_MODEL_DIR` | `/content/models/<repo名>/<quant>` | 本地模型目录 |
| `LLAMA_SERVER` | `/content/llama.cpp/build/bin/llama-server` | llama-server 二进制路径 |
| `LLAMA_HOST` / `LLAMA_PORT` | `0.0.0.0` / `30000` | 监听地址与端口 |
| `LLAMA_CTX` / `LLAMA_NGL` | `0` / `999` | 上下文长度（0=自动按空闲显存拟合）/ GPU 层数 |
| `LLAMA_API_KEY` | 回退 `API_KEY` | 服务器 API 鉴权密钥 |
| `LLAMA_XET` | `1` | 1=启用 HF Xet 存储（默认），0=禁用 |
| `LLAMA_MMPROJ` | 自动检测 `模型目录/mmproj-*.gguf` | 视觉投影器路径（图片输入）；缺省自动下载 `mmproj-*.gguf` |
| `LLAMA_MMPROJ_REPO` | 同 `LLAMA_MODEL_REPO` | mmproj 自动下载源；设为空串禁用自动下载 |

> mmproj 采用**动态能力检测**：启动前解析主模型 GGUF 元数据，仅当存在 `image_token_id`/视觉键
> （如 qwen4exp 的 `qwen4exp.ple.image_token_id`）时才加载 mmproj；换用纯文本模型时会自动跳过，
> 不会把不相干的 mmproj 传给文本模型。检测依赖 `llama-gguf`（与 `llama` 同目录），缺失时按支持处理。

> GPU 适配：进入 `llama/` 目录时，`.envrc` 按 `GPU_PROFILE`（或 `nvidia-smi` 自动探测）
> 加载 `.env.g4` / `.env.t4`（如 G4 → `LLAMA_QUANT=UD-Q2_K_XL`、T4 → `UD-IQ1_M` 等）。

启动成功后日志应有：
```
load_model: loading model '.../Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf'
llama_server: model loaded
llama_server: listening on http://0.0.0.0:30000
```

---

## 5. 调用示例

OpenAI 兼容接口：

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": "用一句话介绍你自己。"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

## 6. 量化档位选择（单卡 96GB 参考）

| 档位 | 磁盘大小 | 单卡 96GB 适配 | 说明 |
|------|----------|----------------|------|
| `UD-IQ1_M` | ~25–35GB | ✅ 最稳 | 量化损失最大 |
| `UD-Q2_K_XL` | ~74GB | ✅ 可选 | 体积/质量平衡，最省显存与磁盘（回退档） |
| `UD-Q3_K_XL` | ~55–65GB | ✅ 可跑 | 质量更好 |
| `UD-Q4_K_XL` | ~105GB(磁盘) / ~80GiB(VRAM) | ✅ 实测可跑（默认） | 质量最好；单卡 96GB 放得下（KV 余量 ~18GiB，已验证生成正常） |

> 注：GGUF **磁盘体积**与**显存占用(VRAM)**不同。`UD-Q4_K_XL` 磁盘约 105GB，但权重加载进显存仅 ~80GiB，因此单卡 96GB 可全量卸载（`-ngl 999`）。切换档位：`LLAMA_QUANT=UD-Q4_K_XL ./launch.sh start`

---

## 7. 常见问题

- **下载卡在 11MB / "Entry not found"**
  → 没装 `hf_xet`（Xet 存储）。重跑 `./colab.sh install llama`；并确认 `HF_TOKEN` 已设置且**已接受许可证**。
- **`qwen4exp` / 架构不支持**
  → 用的是主线 llama.cpp。必须来自 **PR #27742**（`./colab.sh install llama` 已处理）。
- **CUDA out of memory**
  → 模型或上下文太大。换更小量化（`UD-Q2_K_XL`→`UD-IQ1_M`）、降低 `LLAMA_CTX`，或多卡用 `--tensor-split`。
- **安全风险**
  → 默认 CORS 允许所有来源且无 API key。对外暴露时设置 `LLAMA_API_KEY`（或通用 `API_KEY`）。
- **多卡**
  → `--tensor-split 50,50`（2 卡等分）。注意本教程默认按单卡 `NGL=999` 全卸载。

---

## 8. 文件说明

```
colab/
├── colab.sh            # 一体化入口: ./colab.sh install llama 编译引擎
├── logs/llama_server.log   # 服务日志（根目录 logs/）
├── llama/
│   ├── .envrc          # direnv: 继承根 .envrc + 按 GPU 加载 .env.g4/.env.t4
│   ├── .env.g4/.env.t4 # GPU profile（LLAMA_* 变量: 量化/上下文/架构）
│   ├── launch.sh       # 服务管理: start/stop/restart/status/logs/keep
│   └── llama.pid       # 运行 PID（自动生成）
└── /content/
    ├── llama.cpp/      # 源码与编译产物（build/bin/llama-server）
    └── models/
        └── Qwen3.8-Flash-Next-GGUF/
            ├── UD-Q2_K_XL/   # 3 个 .gguf 分片（~74GB）
            └── UD-Q4_K_XL/   # 4 个 .gguf 分片（~105GB，实测单卡可跑）
```
