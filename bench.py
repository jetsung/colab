#!/usr/bin/env python3
"""本地推理服务并发压测(sglang / llama.cpp 通用, 仅用标准库, 无需额外依赖)。

请求层走 OpenAI 兼容 API(/v1/chat/completions、/v1/models), 两个引擎通用;
并发层额外采样 /metrics 的 Prometheus 指标, 指标名按引擎自动识别:
    sglang    num_running_reqs      / num_queue_reqs
    llama.cpp requests_processing   / requests_deferred    (需 --metrics 启动)
采样不到时只跳过并发统计, 不影响吞吐/延迟结果。

用法:
    ./bench.py                              # 默认 32 并发 / 256 tokens / 自动识别引擎
    ./bench.py -n 64 --max-tokens 512
    ./bench.py -n 16 --port 30000 --model qwen3.8-27b
    ./bench.py --engine llama -n 8          # 显式指定引擎

对比引擎参数(如 sglang 的 --mamba-full-memory-ratio、llama 的 -np/--parallel)时,
保持 -n / --max-tokens 不变, 重点看: 聚合吞吐、最大延迟、queue 峰值。
鉴权密钥从 SGLANG_API_KEY / LLAMA_API_KEY / API_KEY 读取(经 .envrc 由 direnv 加载)。
"""

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_PROMPT = "用一句话介绍你自己"


def build_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def get_model(host: str, port: int, api_key: str, timeout: float) -> str:
    """未指定模型时, 从 /v1/models 取第一个可用模型名"""
    req = urllib.request.Request(
        build_url(host, port, "/v1/models"),
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception as e:
        sys.exit(f"无法获取模型列表({e}); 服务未就绪? 或用 --model 显式指定")


def make_request(host, port, api_key, model, max_tokens, temperature, prompt, idx, timeout):
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": f"{idx}: {prompt}"}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        build_url(host, port, "/v1/chat/completions"),
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return time.time() - t0, data["usage"]["completion_tokens"], None
    except Exception as e:  # HTTPError / URLError / 超时都记为失败
        return time.time() - t0, 0, str(e)


# 各引擎 /metrics 中"正在处理/排队中"的指标名(按后缀匹配, 忽略前缀与 label)
ENGINE_METRICS = {
    "sglang": ("num_running_reqs", "num_queue_reqs"),
    "llama": ("requests_processing", "requests_deferred"),
}

# llama.cpp 的 /metrics 需 --metrics 启动; 采样不到时给出这个提示
METRICS_HINT = "llama.cpp 需加 --metrics 启动才有 /metrics(见 llama/launch.sh 的 LLAMA_METRICS)"


def parse_metrics(text):
    """解析 Prometheus 文本 -> {指标名: 值}; 跳过注释与非法行"""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].strip()
        try:
            out[name] = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return out


def detect_engine(host, port, timeout=2.0):
    """取一次 /metrics, 按指标名判断引擎; 取不到或都没有则返回 None"""
    try:
        with urllib.request.urlopen(build_url(host, port, "/metrics"), timeout=timeout) as r:
            names = set(parse_metrics(r.read().decode(errors="ignore")))
    except (urllib.error.URLError, OSError):
        return None
    for engine, keys in ENGINE_METRICS.items():
        if any(any(n.endswith(k) for n in names) for k in keys):
            return engine
    return None


class MetricsSampler(threading.Thread):
    """周期性采样 /metrics, 记录 running / queue 峰值"""

    def __init__(self, host, port, engine, interval=0.3):
        super().__init__(daemon=True)
        self.url = build_url(host, port, "/metrics")
        self.interval = interval
        # 未识别引擎时按全部已知指标名匹配, 保证换引擎也能采到
        self.running_keys = [k for k, _ in ENGINE_METRICS.values()]
        self.queue_keys = [k for _, k in ENGINE_METRICS.values()]
        if engine in ENGINE_METRICS:
            self.running_keys = [ENGINE_METRICS[engine][0]]
            self.queue_keys = [ENGINE_METRICS[engine][1]]
        self.running_peak = 0.0
        self.queue_peak = 0.0
        self.sampled = False
        self._stop = threading.Event()

    @staticmethod
    def _matches(name, keys):
        return any(name.endswith(k) for k in keys)

    def run(self):
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=2) as r:
                    for name, value in parse_metrics(
                        r.read().decode(errors="ignore")
                    ).items():
                        if self._matches(name, self.running_keys):
                            self.running_peak = max(self.running_peak, value)
                            self.sampled = True
                        elif self._matches(name, self.queue_keys):
                            self.queue_peak = max(self.queue_peak, value)
                            self.sampled = True
            except (urllib.error.URLError, OSError):
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


def main():
    p = argparse.ArgumentParser(
        description="本地推理服务并发压测(sglang / llama.cpp 通用, 走 OpenAI 兼容 API)"
    )
    p.add_argument("-n", "--concurrency", type=int, default=32, help="并发请求数(默认 32)")
    p.add_argument("--max-tokens", type=int, default=256, help="每请求最大生成 token(默认 256)")
    p.add_argument("--model", default=None, help="模型名; 默认从 /v1/models 自动获取")
    p.add_argument("--engine", choices=("auto", "sglang", "llama"), default="auto",
                   help="引擎类型, 仅影响并发指标的指标名(默认 auto: 按 /metrics 自动识别)")
    p.add_argument("--host", default="localhost")
    p.add_argument("-p", "--port", type=int, default=30000)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=1, help="预热请求数(默认 1, 避开首次 CUDA graph/JIT)")
    p.add_argument("--timeout", type=float, default=300.0, help="单请求超时秒数(默认 300)")
    args = p.parse_args()

    api_key = (
        os.environ.get("SGLANG_API_KEY")
        or os.environ.get("LLAMA_API_KEY")
        or os.environ.get("API_KEY", "")
    )
    model = args.model or get_model(args.host, args.port, api_key, args.timeout)

    # 预热: 不计入统计, 避免首次请求触发 JIT/CUDA graph 捕获而拉低第一波成绩
    for i in range(args.warmup):
        make_request(args.host, args.port, api_key, model, args.max_tokens,
                     args.temperature, args.prompt, -i - 1, args.timeout)

    engine = args.engine
    if engine == "auto":
        engine = detect_engine(args.host, args.port) or "unknown"

    sampler = MetricsSampler(args.host, args.port, engine)
    sampler.start()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(
            lambda i: make_request(args.host, args.port, api_key, model, args.max_tokens,
                                   args.temperature, args.prompt, i, args.timeout),
            range(args.concurrency),
        ))
    wall = time.time() - t0
    sampler.stop()
    sampler.join(timeout=2)

    latencies = [r[0] for r in results]
    tokens = sum(r[1] for r in results)
    failures = [r[2] for r in results if r[2]]

    print(f"引擎={engine} 模型={model} 并发={args.concurrency} max_tokens={args.max_tokens}")
    print(f"  墙钟           {wall:.1f}s")
    print(f"  聚合吞吐       {tokens / wall:.1f} tok/s (完成 {tokens} tokens)")
    print(f"  单请求延迟     平均 {statistics.mean(latencies):.1f}s "
          f"中位 {statistics.median(latencies):.1f}s 最大 {max(latencies):.1f}s")
    if sampler.sampled:
        print(f"  服务端并发     running 峰值 {sampler.running_peak:.0f} / queue 峰值 {sampler.queue_peak:.0f}")
    else:
        print(f"  服务端并发     无数据(未从 /metrics 采到并发指标); {METRICS_HINT}")
    if failures:
        print(f"  失败 {len(failures)}/{len(results)}: {failures[0]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
