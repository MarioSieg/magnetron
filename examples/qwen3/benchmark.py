from __future__ import annotations

import argparse
import gc
import time

from magnetron import context

from inference import InferenceConfig, InferenceEngine


def _set_mode(mode: str) -> None:
    if mode == "eager":
        context.stop_lazy_execution()
        context.stop_full_graph_trace()
    elif mode == "lazy":
        context.start_lazy_execution()
        context.stop_full_graph_trace()
    elif mode == "lazy_fullgraph":
        context.start_lazy_execution()
        context.start_full_graph_trace()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _run_once(engine: InferenceEngine, prompt: str) -> tuple[float, str]:
    gc.collect()
    start = time.perf_counter()
    out = engine.one_shot_answer(prompt)
    elapsed = time.perf_counter() - start
    return elapsed, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3 across execution modes.")
    parser.add_argument("--prompt", type=str, default="Hey", help="Prompt to run for timing.")
    parser.add_argument("--max_tokens", type=int, default=32, help="Max generated tokens.")
    parser.add_argument("--temp", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=200, help="Top-k sampling.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.", help="System prompt.")
    parser.add_argument("--max_ctx", type=int, default=4096, help="Max context tokens.")
    parser.add_argument("--reserve_gen", type=int, default=1024, help="Reserved generation headroom.")
    args = parser.parse_args()

    cfg = InferenceConfig(
        system=args.system,
        max_ctx=args.max_ctx,
        reserve_gen=args.reserve_gen,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_k=args.top_k,
        seed=args.seed,
    )
    engine = InferenceEngine(cfg)

    modes = ("eager", "lazy", "lazy_fullgraph")
    results: list[tuple[str, float, str]] = []

    for mode in modes:
        _set_mode(mode)
        # Warmup run for model/allocator/cache/threadpool effects.
        _run_once(engine, args.prompt)
        elapsed, out = _run_once(engine, args.prompt)
        results.append((mode, elapsed, out))

    print("\nQwen3 benchmark (lower is better):")
    for mode, elapsed, out in results:
        preview = out.replace("\n", " ")[:80]
        print(f"- {mode:14s} {elapsed:8.3f}s  |  {preview}")


if __name__ == "__main__":
    main()

