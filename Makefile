# =============================================================================
# Colab 管理 — colab.sh 的薄封装
#
# 所有功能均透传给根目录 colab.sh(一体化入口):
#   [宿主机]  vps:   install / create / sessions / mount / all
#   [Colab]   setup: deps / bore / relaydrop / opencode / codebuddy / hint / all
#   [Colab]   install: llama / sglang / vllm(安装并启用引擎环境)
#   [Colab]   llama: start / stop / restart / status / test / logs / keep
#   [Colab]   sglang: start / stop / restart / status / test / bench / logs / keep
#   [Colab]   vllm: start / stop / restart / status / test / bench / logs / keep
#   [Colab]   bore:  start / stop / restart / status / logs
#
# 实际逻辑见 colab.sh, 此处仅配置参数透传
# =============================================================================

SCRIPT ?= ./colab.sh

# vps 会话参数
SESSION_NAME ?= gcloud
GPU ?= G4
# TPU: 留空则用 GPU; 指定则用 TPU(v5e1 / v6e1)
TPU ?=

# setup: 安装工具后回显的提示信息
CODEBUDDY_PKG ?= @tencent-ai/codebuddy-code
OPENCODE_INSTALL ?= https://opencode.ai/install

# bore 隧道参数(由 .envrc 或命令行覆盖)
BORE_PORT ?= 65535

.PHONY: install create sessions mount all \
        setup setup-deps setup-tools setup-bore setup-relaydrop \
        setup-opencode setup-codebuddy setup-hint \
        install-llama install-llama-build install-sglang install-vllm \
        llama llama-start llama-stop llama-restart llama-status llama-test llama-bench llama-logs llama-keep \
        sglang sglang-start sglang-stop sglang-restart sglang-status sglang-test sglang-bench sglang-logs sglang-keep \
        vllm vllm-start vllm-stop vllm-restart vllm-status vllm-test vllm-bench vllm-logs vllm-keep \
        bore bore-start bore-stop bore-restart bore-status bore-logs \
        help list

# 默认目标: 显示帮助
.DEFAULT_GOAL := help

# 查看所有功能列表(从注释自动解析 "## [一级/二级] target: 说明")
help list:
	@echo 'Colab 管理 — 可用功能:'
	@echo ''
	@awk '/^## /{ \
	       line=substr($$0,4); \
	       grp=line; sub(/].*/,"",grp); sub(/^\[/,"",grp); \
	       subgrp=grp; if(i=index(subgrp,"/")){ grp=substr(subgrp,1,i-1); subgrp=substr(subgrp,i+1) } else { subgrp="" } \
	       tgt=line; sub(/^\[[^]]*\][ \t]*/,"",tgt); sub(/:.*/,"",tgt); \
	       desc=substr(line,index(line,":")+1); sub(/^[ \t]+/,"",desc); \
	       key=grp SUBSEP subgrp; \
	       item[key]=item[key] sprintf("    \033[36mmake %-16s\033[0m %s\n", tgt, desc); \
	       if(!(key in seen)){ seen[key]=1; order[++n]=key } \
	     } \
	     END{ \
	       lastgrp=""; \
	       for(k=1;k<=n;k++){ \
	         key=order[k]; \
	         split(key, p, SUBSEP); \
	         if(p[1]!=lastgrp){ printf "\033[33m[%s]\033[0m\n", p[1]; lastgrp=p[1] } \
	         if(p[2]!=""){ printf "  \033[32m%s\033[0m\n", p[2] } \
	         printf "%s", item[key]; \
	       } \
	     }' $(MAKEFILE_LIST)

# ================== [宿主机] vps: Colab CLI 与会话管理 ==================

## [宿主机] install: 安装 Google Colab CLI(幂等: 已装则跳过)
install:
	bash $(SCRIPT) vps install

## [宿主机] create: 创建 GPU/TPU 会话(GPU=$(GPU) / TPU=$(TPU), 会话 $(SESSION_NAME))
create:
	@if [ -n "$(TPU)" ]; then \
	  bash $(SCRIPT) vps create -t $(TPU) -s $(SESSION_NAME); \
	else \
	  bash $(SCRIPT) vps create -g $(GPU) -s $(SESSION_NAME); \
	fi

## [宿主机] sessions: 列出会话
sessions:
	bash $(SCRIPT) vps sessions

## [宿主机] mount: 挂载 Google Drive
mount:
	bash $(SCRIPT) vps mount -s $(SESSION_NAME)

## [宿主机] all: 一键执行 安装->创建->列出->挂载
all:
	bash $(SCRIPT) vps all -g $(GPU) -s $(SESSION_NAME)

# ================== [Colab/setup] 环境前置 ==================

## [Colab/setup] setup: 安装全部前置依赖(等价 setup all)
setup:
	bash $(SCRIPT) setup all

## [Colab/setup] setup-deps: 系统依赖 direnv + bashrc hook(幂等)
setup-deps:
	bash $(SCRIPT) setup deps

## [Colab/setup] setup-tools: 安装命令行工具(bore/relaydrop)组合入口
setup-tools: setup-bore setup-relaydrop

## [Colab/setup] setup-bore: 安装 bore
setup-bore:
	bash $(SCRIPT) setup bore

## [Colab/setup] setup-relaydrop: 安装 relaydrop
setup-relaydrop:
	bash $(SCRIPT) setup relaydrop

## [Colab/setup] setup-opencode: 安装 opencode(OPENCODE_INSTALL 可覆盖地址)
setup-opencode:
	bash $(SCRIPT) setup opencode

## [Colab/setup] setup-codebuddy: 安装 codebuddy-code(CODEBUDDY_PKG 可覆盖包名, 已装则跳过)
setup-codebuddy:
	bash $(SCRIPT) setup codebuddy

## [Colab/setup] setup-hint: 安装完成后的后续步骤提示
setup-hint:
	bash $(SCRIPT) setup hint

# ================== [Colab/install] 引擎环境安装 ==================

# llama 安装参数，可传 --build 或 -B；默认为空(下载预编译二进制)
LLAMA_INSTALL_ARGS ?=

## [Colab/install] install-llama: 安装 llama.cpp(LLAMA_INSTALL_ARGS=--build 或 -B 可源码编译)
install-llama:
	bash $(SCRIPT) install llama $(LLAMA_INSTALL_ARGS)

## [Colab/install] install-llama-build: 使用源码编译方式安装 llama.cpp(等价 --build)
install-llama-build:
	bash $(SCRIPT) install llama --build

## [Colab/install] install-sglang: 建 venv + 装 SGLang(不自动启动)
install-sglang:
	bash $(SCRIPT) install sglang

## [Colab/install] install-vllm: 建 venv + 安装官方最新 vLLM(不自动启动)
install-vllm:
	bash $(SCRIPT) install vllm

# ================== [Colab/llama] llama.cpp 服务管理 ==================

## [Colab/llama] llama: 默认动作(启动服务)
llama: llama-start

## [Colab/llama] llama-start: 启动服务(后台 setsid 托管; 首次自动下载模型)
llama-start:
	bash $(SCRIPT) llama start

## [Colab/llama] llama-stop: 停止服务
llama-stop:
	bash $(SCRIPT) llama stop

## [Colab/llama] llama-restart: 重启服务
llama-restart:
	bash $(SCRIPT) llama restart

## [Colab/llama] llama-status: 查看状态 + 健康检查
llama-status:
	bash $(SCRIPT) llama status

## [Colab/llama] llama-test: 发送一条测试对话(需服务已就绪)
llama-test:
	bash $(SCRIPT) llama test

## [Colab/llama] llama-bench: 并发压测(参数用 BENCH_ARGS 传, 如 -n 8 --max-tokens 256)
llama-bench:
	bash $(SCRIPT) llama bench $(BENCH_ARGS)

## [Colab/llama] llama-logs: 跟踪日志
llama-logs:
	bash $(SCRIPT) llama logs

## [Colab/llama] llama-keep: 守护模式(崩溃自动拉起)
llama-keep:
	bash $(SCRIPT) llama keep

# ================== [Colab/sglang] SGLang 服务管理 ==================

## [Colab/sglang] sglang: 默认动作(启动服务)
sglang: sglang-start

## [Colab/sglang] sglang-start: 启动服务(后台 setsid 托管)
sglang-start:
	bash $(SCRIPT) sglang start

## [Colab/sglang] sglang-stop: 停止服务
sglang-stop:
	bash $(SCRIPT) sglang stop

## [Colab/sglang] sglang-restart: 重启服务
sglang-restart:
	bash $(SCRIPT) sglang restart

## [Colab/sglang] sglang-status: 查看状态 + 健康检查
sglang-status:
	bash $(SCRIPT) sglang status

## [Colab/sglang] sglang-test: 发送一条测试对话(需服务已就绪)
sglang-test:
	bash $(SCRIPT) sglang test

## [Colab/sglang] sglang-bench: 并发压测(参数用 BENCH_ARGS 传, 如 -n 64 --max-tokens 512)
sglang-bench:
	bash $(SCRIPT) sglang bench $(BENCH_ARGS)

## [Colab/sglang] sglang-logs: 跟踪日志
sglang-logs:
	bash $(SCRIPT) sglang logs

## [Colab/sglang] sglang-keep: 守护模式(崩溃自动拉起)
sglang-keep:
	bash $(SCRIPT) sglang keep

# ================== [Colab/vllm] vLLM 服务管理 ==================

## [Colab/vllm] vllm: 默认动作(启动服务)
vllm: vllm-start

## [Colab/vllm] vllm-start: 启动服务(后台 setsid 托管)
vllm-start:
	bash $(SCRIPT) vllm start

## [Colab/vllm] vllm-stop: 停止服务
vllm-stop:
	bash $(SCRIPT) vllm stop

## [Colab/vllm] vllm-restart: 重启服务
vllm-restart:
	bash $(SCRIPT) vllm restart

## [Colab/vllm] vllm-status: 查看状态 + 健康检查
vllm-status:
	bash $(SCRIPT) vllm status

## [Colab/vllm] vllm-test: 发送一条测试对话(需服务已就绪)
vllm-test:
	bash $(SCRIPT) vllm test

## [Colab/vllm] vllm-bench: 并发压测(参数用 BENCH_ARGS 传, 如 -n 8 --max-tokens 256)
vllm-bench:
	bash $(SCRIPT) vllm bench $(BENCH_ARGS)

## [Colab/vllm] vllm-logs: 跟踪日志
vllm-logs:
	bash $(SCRIPT) vllm logs

## [Colab/vllm] vllm-keep: 守护模式(崩溃自动拉起)
vllm-keep:
	bash $(SCRIPT) vllm keep

# ================== [Colab/bore] 公网隧道 ==================

## [Colab/bore] bore: 隧道管理组合入口(start/stop/restart/status/logs)
bore: bore-start

## [Colab/bore] bore-start: 启动隧道(后台 setsid 托管; BORE_PORT 可覆盖公网端口)
bore-start:
	BORE_PORT=$(BORE_PORT) bash $(SCRIPT) bore start

## [Colab/bore] bore-stop: 停止隧道
bore-stop:
	bash $(SCRIPT) bore stop

## [Colab/bore] bore-restart: 重启隧道(stop+start)
bore-restart:
	BORE_PORT=$(BORE_PORT) bash $(SCRIPT) bore restart

## [Colab/bore] bore-status: 查看隧道状态
bore-status:
	bash $(SCRIPT) bore status

## [Colab/bore] bore-logs: 跟踪隧道日志(含公网地址)
bore-logs:
	bash $(SCRIPT) bore logs
