"""Async online serving benchmark for CoSpec.

Sends concurrent requests to a running vLLM server and measures throughput,
TTFT, TPOT, and ITL. Supports multiple datasets via vllm.benchmarks.datasets.

Usage:
    python3 bench_serving.py --model Qwen/Qwen3-8B --dataset sharegpt \
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 200 --request-rate 4
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from vllm.benchmarks.datasets import (
    BurstGPTDataset,
    RandomDataset,
    ShareGPTDataset,
    SonnetDataset,
)
from vllm.transformers_utils.tokenizer import get_tokenizer

DATASETS = {
    "sharegpt": ShareGPTDataset,
    "random": RandomDataset,
    "sonnet": SonnetDataset,
    "burstgpt": BurstGPTDataset,
}


@dataclass
class RequestResult:
    prompt_len: int = 0
    output_len: int = 0
    ttft: float = 0.0  # time to first token
    latency: float = 0.0  # end-to-end
    token_timestamps: list = field(default_factory=list)
    success: bool = True
    error: str = ""


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    prompt_len: int,
    expected_output_len: int,
    temperature: float,
) -> RequestResult:
    """Send a single streaming completion request and collect timing."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": expected_output_len,
        "temperature": temperature,
        "stream": True,
    }
    result = RequestResult(prompt_len=prompt_len)
    t_start = time.perf_counter()
    first_token_time = None
    token_times = []

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                result.success = False
                result.error = f"HTTP {resp.status}: {await resp.text()}"
                return result

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data: "):
                    continue
                data = decoded[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choice = chunk["choices"][0]
                    if choice.get("text", "") or choice.get("delta", {}).get(
                        "content", ""
                    ):
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_times.append(now)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    except Exception as e:
        result.success = False
        result.error = str(e)
        return result

    t_end = time.perf_counter()
    result.latency = t_end - t_start
    result.output_len = len(token_times)
    result.ttft = (first_token_time - t_start) if first_token_time else 0.0
    result.token_timestamps = token_times
    if result.output_len == 0:
        result.success = False
        result.error = "No tokens received"
    return result


async def benchmark(
    url: str,
    model: str,
    requests: list,
    request_rate: float,
    max_concurrency: Optional[int],
) -> list[RequestResult]:
    """Run the benchmark with Poisson-distributed request arrivals."""
    sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout
    ) as session:
        tasks = []
        pbar = tqdm(total=len(requests), desc="Requests")

        async def run_one(prompt, prompt_len, output_len):
            if sem:
                async with sem:
                    r = await send_request(
                        session, url, model, prompt, prompt_len, output_len, 0.0
                    )
            else:
                r = await send_request(
                    session, url, model, prompt, prompt_len, output_len, 0.0
                )
            pbar.update(1)
            return r

        for prompt, prompt_len, output_len in requests:
            tasks.append(asyncio.create_task(run_one(prompt, prompt_len, output_len)))
            if request_rate < float("inf"):
                await asyncio.sleep(np.random.exponential(1.0 / request_rate))

        results = await asyncio.gather(*tasks)
        pbar.close()
    return list(results)


def print_metrics(results: list[RequestResult], elapsed: float):
    """Compute and print benchmark metrics."""
    ok = [r for r in results if r.success]
    failed_results = [r for r in results if not r.success]
    failed = len(failed_results)
    if not ok:
        print("All requests failed!")
        errors = set(r.error for r in failed_results[:10])
        for e in errors:
            print(f"  Error: {e[:200]}")
        return

    total_input = sum(r.prompt_len for r in ok)
    total_output = sum(r.output_len for r in ok)

    ttfts = [r.ttft for r in ok if r.ttft > 0]
    latencies = [r.latency for r in ok]

    # Inter-token latencies
    itls = []
    for r in ok:
        ts = r.token_timestamps
        for i in range(1, len(ts)):
            itls.append(ts[i] - ts[i - 1])

    # Time per output token
    tpots = []
    for r in ok:
        if r.output_len > 1:
            tpots.append((r.latency - r.ttft) / (r.output_len - 1))

    def percentiles(data, ps=(50, 90, 95, 99)):
        if not data:
            return {p: 0 for p in ps}
        return {p: float(np.percentile(data, p)) for p in ps}

    print("\n" + "=" * 60)
    print(f"{'Benchmark Results':^60}")
    print("=" * 60)
    print(f"  Completed requests:    {len(ok)}")
    print(f"  Failed requests:       {failed}")
    print(f"  Total time:            {elapsed:.2f} s")
    print(f"  Request throughput:    {len(ok) / elapsed:.2f} req/s")
    print(f"  Input token throughput:  {total_input / elapsed:.2f} tok/s")
    print(f"  Output token throughput: {total_output / elapsed:.2f} tok/s")
    print(f"  Total token throughput:  {(total_input + total_output) / elapsed:.2f} tok/s")

    print(f"\n  {'Metric':<30} {'Mean':>10} {'P50':>10} {'P90':>10} {'P99':>10}")
    print(f"  {'-'*70}")

    for name, data, unit in [
        ("TTFT", ttfts, "ms"),
        ("TPOT", tpots, "ms"),
        ("ITL", itls, "ms"),
        ("E2E Latency", latencies, "ms"),
    ]:
        if not data:
            continue
        scale = 1000.0  # convert s → ms
        p = percentiles(data)
        mean = float(np.mean(data))
        print(
            f"  {name + ' (' + unit + ')':<30} "
            f"{mean*scale:>10.2f} {p[50]*scale:>10.2f} "
            f"{p[90]*scale:>10.2f} {p[99]*scale:>10.2f}"
        )
    print("=" * 60)

    return {
        "completed": len(ok),
        "failed": failed,
        "total_time_s": elapsed,
        "request_throughput": len(ok) / elapsed,
        "input_tok_throughput": total_input / elapsed,
        "output_tok_throughput": total_output / elapsed,
        "mean_ttft_ms": float(np.mean(ttfts)) * 1000 if ttfts else 0,
        "mean_tpot_ms": float(np.mean(tpots)) * 1000 if tpots else 0,
        "mean_itl_ms": float(np.mean(itls)) * 1000 if itls else 0,
        "mean_e2e_ms": float(np.mean(latencies)) * 1000,
    }


def load_requests(args, tokenizer) -> list[tuple[str, int, int]]:
    """Load dataset and return list of (prompt, prompt_len, output_len)."""
    cls = DATASETS[args.dataset]

    kwargs = {"random_seed": args.seed}
    if args.dataset == "random":
        kwargs["dataset_path"] = None
    else:
        kwargs["dataset_path"] = args.dataset_path

    dataset = cls(**kwargs)

    sample_kwargs = {"tokenizer": tokenizer, "num_requests": args.num_prompts}
    if args.dataset == "random":
        sample_kwargs["input_len"] = args.random_input_len
        sample_kwargs["output_len"] = args.random_output_len
        sample_kwargs["range_ratio"] = args.random_range_ratio
        sample_kwargs["prefix_len"] = args.random_prefix_len
    elif args.dataset == "sharegpt" and args.sharegpt_output_len:
        sample_kwargs["output_len"] = args.sharegpt_output_len

    samples = dataset.sample(**sample_kwargs)
    return [(s.prompt, s.prompt_len, s.expected_output_len) for s in samples]


def parse_args():
    parser = argparse.ArgumentParser(description="CoSpec online serving benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="sharegpt",
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--result-dir", default="results")

    # Random dataset options
    parser.add_argument("--random-input-len", type=int, default=256)
    parser.add_argument("--random-output-len", type=int, default=128)
    parser.add_argument("--random-range-ratio", type=float, default=1.0)
    parser.add_argument("--random-prefix-len", type=int, default=0)

    # ShareGPT options
    parser.add_argument("--sharegpt-output-len", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    url = f"http://{args.host}:{args.port}/v1/completions"

    if args.dataset != "random" and args.dataset_path is None:
        raise ValueError(f"--dataset-path required for dataset '{args.dataset}'")

    print(f"Loading tokenizer for {args.model}...")
    tokenizer = get_tokenizer(args.model)

    print(f"Loading {args.dataset} dataset...")
    requests = load_requests(args, tokenizer)
    print(f"Loaded {len(requests)} requests")

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Benchmarking {url} @ {args.request_rate} req/s")
    t0 = time.perf_counter()
    results = asyncio.run(
        benchmark(url, args.model, requests, args.request_rate, args.max_concurrency)
    )
    elapsed = time.perf_counter() - t0

    metrics = print_metrics(results, elapsed)

    if args.save_result and metrics:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        out = Path(args.result_dir) / "results.json"
        with open(out, "w") as f:
            json.dump({**metrics, "args": vars(args)}, f, indent=2)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
