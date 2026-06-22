# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import gc
import time

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from magnetron import Tensor, context, dtype
from rich.console import Console
from tokenizers import Tokenizer
from model import Qwen3Model, Qwen3HyperParams

console = Console()
REPO_ID: str = 'mario-sieg/qwen3.0-4b-2507-instruct-magnetron'


def _download_or_ensure_hf_file(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    console.print(f'Downloading {filename}', style='dim')
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type='model',
    )


class HFTokenizer:
    def __init__(self, repo_id: str) -> None:
        tok_path = _download_or_ensure_hf_file(repo_id=repo_id, filename='tokenizer.json')
        self.tok = Tokenizer.from_file(tok_path)

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, tok_id: list[int]) -> str:
        return self.tok.decode(tok_id)


@dataclass
class InferenceConfig:
    system: str = 'You are a helpful assistant.'
    device: str = 'cuda'
    max_ctx: int = 4096
    max_tokens: int = 1024
    temp: float = 0.6
    top_k: int = 200
    seed: int = 3407
    snapshot: str | None = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'InferenceConfig':
        dtype_map: dict[str, dtype.DType] = {}
        for T in dtype.floating:
            dtype_map[T.name] = T
        return cls(
            system=args.system,
            device=args.device,
            max_ctx=args.max_ctx,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_k=args.top_k,
            seed=args.seed,
            snapshot=args.snapshot,
        )


class InferenceEngine:
    def __init__(self, config: InferenceConfig) -> None:
        snapshot: str | None = config.snapshot
        if snapshot is None:
            snapshot = _download_or_ensure_hf_file(
                repo_id=REPO_ID,
                filename='qwen3-4b-instruct-2507-bf16.mag',
            )
        assert snapshot is not None
        start = time.perf_counter()
        context.stop_grad_recorder()
        context.set_default_dtype(dtype.bfloat16)
        context.manual_seed(config.seed)
        if context.is_device_available(config.device):
            context.set_default_device(config.device)
        console.print(f'Loading model from snapshot: {snapshot}', style='dim')
        self.model = Qwen3Model.from_pretrained_snapshot(snapshot, Qwen3HyperParams())
        self.tokenizer = HFTokenizer(REPO_ID)
        self.config = config
        end = time.perf_counter()
        console.print(f'Ready in {end - start:.2f}s', style='dim')
        gc.collect()

    def gen_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temp: float | None = None,
        top_k: int | None = None,
        reset_cache: bool = False,
    ) -> Iterator[str]:
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temp is None:
            temp = self.config.temp
        if top_k is None:
            top_k = self.config.top_k
        model_input_ids = Tensor([self.tokenizer.encode(prompt)], dtype=dtype.int64)
        for chunk in self.model.generate_stream(
            model_input_ids,
            self.tokenizer,
            max_tokens=max_tokens,
            temp=temp,
            top_k=top_k,
            reset_cache=reset_cache,
        ):
            yield chunk
        gc.collect()

    async def gen_stream_async(
        self, prompt: str, max_tokens: int | None = None, temp: float | None = None, top_k: int | None = None
    ) -> AsyncIterator[str]:
        import asyncio

        for chunk in self.gen_stream(prompt, max_tokens, temp, top_k):
            yield chunk
            await asyncio.sleep(0)

    def gen_one_shot(self, prompt: str, max_tokens: int | None = None, temp: float | None = None, top_k: int | None = None) -> str:
        parts: list[str] = []
        for chunk in self.gen_stream(prompt, max_tokens, temp, top_k):
            parts.append(chunk)
        reply: str = ''.join(parts)
        return reply
