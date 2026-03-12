# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import asyncio
import gc
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass

from magnetron import Tensor, context
from rich.console import Console
from tokenizers import Tokenizer

from model import build_prompt, Qwen3Model, dtype

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


def clamp_history_by_tokens(
    tokenizer: HFTokenizer,
    system: str,
    history: list[tuple[str, str]],
    max_ctx: int,
    reserve_gen: int,
) -> list[tuple[str, str]]:
    budget = max(256, max_ctx - reserve_gen)
    prompt = build_prompt(system, history)
    n = len(tokenizer.encode(prompt))
    if n <= budget:
        return history
    trimmed = history[:]
    while trimmed:
        trimmed.pop(0)
        if trimmed and trimmed[0][0] == 'assistant':
            trimmed.pop(0)
        prompt = build_prompt(system, trimmed)
        n = len(tokenizer.encode(prompt))
        if n <= budget:
            return trimmed
    return []


@dataclass
class InferenceConfig:
    system: str = 'You are a helpful assistant.'
    max_ctx: int = 4096
    reserve_gen: int = 1024
    max_tokens: int = 1024
    temp: float = 0.6
    top_k: int = 200
    seed: int = 3407

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'InferenceConfig':
        return cls(
            system=args.system,
            max_ctx=args.max_ctx,
            reserve_gen=args.reserve_gen,
            max_tokens=args.max_tokens,
            temp=args.temp,
            top_k=args.top_k,
            seed=args.seed,
        )


class InferenceEngine:
    def __init__(self, config: InferenceConfig, snapshot: str | None = None) -> None:
        if snapshot is None:
            snapshot = _download_or_ensure_hf_file(
                repo_id=REPO_ID,
                filename='qwen3-4b-instruct-2507-bfloat16.mag',
            )
        start = time.perf_counter()
        context.stop_grad_recorder()
        context.manual_seed(config.seed)
        console.print(f'Loading model from snapshot: {snapshot}', style='dim')
        self.model = Qwen3Model.from_pretrained_snapshot(snapshot)
        self.tokenizer = HFTokenizer(REPO_ID)
        self.config = config
        end = time.perf_counter()
        console.print(f'Ready in {end - start:.2f}s', style='dim')
        gc.collect()

    def stream_chat(
        self,
        history: list[tuple[str, str]],
        system_override: str | None = None,
        mode: str | None = None,
    ) -> Iterator[str]:
        c = self.config
        system = (system_override or c.system).strip() or c.system
        max_tokens = c.max_tokens
        temp = c.temp
        if mode == 'proactive':
            max_tokens = min(80, c.max_tokens)
            temp = 0.8
        yield from self._stream_chat_impl(history, system, max_tokens=max_tokens, temp=temp)

    def _stream_chat_impl(
        self,
        history: list[tuple[str, str]],
        system: str,
        max_tokens: int,
        temp: float,
    ) -> Iterator[str]:
        c = self.config
        history = clamp_history_by_tokens(self.tokenizer, system, history, max_ctx=c.max_ctx, reserve_gen=c.reserve_gen)
        prompt = build_prompt(system, history)
        model_input_ids = Tensor([self.tokenizer.encode(prompt)], dtype=dtype.int64)
        for chunk in self.model.generate_stream(model_input_ids, self.tokenizer, max_tokens=max_tokens, temp=temp, top_k=c.top_k):
            yield chunk
        gc.collect()

    async def async_stream_chat(
        self,
        history: list[tuple[str, str]],
        system_override: str | None = None,
        mode: str | None = None,
    ) -> AsyncIterator[str]:
        c = self.config
        system = (system_override or c.system).strip() or c.system
        max_tokens = c.max_tokens
        temp = c.temp
        if mode == 'proactive':
            max_tokens = min(80, c.max_tokens)
            temp = 0.8
        for chunk in self._stream_chat_impl(history, system, max_tokens=max_tokens, temp=temp):
            yield chunk
            await asyncio.sleep(0)

    def one_shot_answer(self, prompt: str) -> str:
        prompt = build_prompt(self.config.system, [('user', prompt)])
        model_input_ids = Tensor([self.tokenizer.encode(prompt)], dtype=dtype.int64)
        gc.collect()
        reply_parts: list[str] = []
        c = self.config
        for chunk in self.model.generate_stream(model_input_ids, self.tokenizer, max_tokens=c.max_tokens, temp=c.temp, top_k=c.top_k):
            reply_parts.append(chunk)
        reply = ''.join(reply_parts)
        gc.collect()
        return reply
