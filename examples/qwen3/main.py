# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import gc
import time
import argparse

from magnetron import Tensor, context
from tokenizers import Tokenizer
from model import build_prompt, Qwen3Model, dtype
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.prompt import Prompt

REPO_ID: str = 'mario-sieg/qwen3.0-4b-2507-instruct-magnetron'  # HF repo we download the data from


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


console = Console()


def _clamp_history_by_tokens(
    tokenizer: HFTokenizer,
    system: str,
    history: list[tuple[str, str]],
    max_ctx: int,
    reserve_gen: int,
) -> list[tuple[str, str]]:
    budget = max(256, max_ctx - reserve_gen)  # avoid too-small budgets
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


class GenerationContext:
    def __init__(self, snapshot: str, args: argparse.Namespace) -> None:
        start = time.perf_counter()
        context.stop_grad_recorder()
        context.manual_seed(args.seed)
        console.print(f'Loading model from snapshot: {snapshot}', style='dim')
        self.model = Qwen3Model.from_pretrained_snapshot(snapshot)
        self.tokenizer = HFTokenizer(REPO_ID)
        self.args = args
        end = time.perf_counter()
        console.print(f'Ready in {end - start:.2f}s', style='dim')
        gc.collect()

    def repl(self) -> None:
        console.print(
            Panel.fit(
                Text('Qwen3 REPL', style='bold white') + Text('\n/exit  /reset', style='dim'),
                border_style='cyan',
            )
        )
        history: list[tuple[str, str]] = []
        last_ctx_used: int = 0
        while True:
            user = Prompt.ask('[bold cyan]You[/]').strip()
            if not user:
                continue
            if user == '/exit':
                break
            if user == '/reset':
                history.clear()
                console.print('[dim]History cleared.[/dim]')
                continue
            history.append(('user', user))
            history = _clamp_history_by_tokens(
                self.tokenizer,
                self.args.system,
                history,
                max_ctx=self.args.max_ctx,
                reserve_gen=self.args.reserve_gen,
            )
            prompt = build_prompt(self.args.system, history)
            model_input_ids = Tensor.of([self.tokenizer.encode(prompt)], dtype=dtype.int64)
            console.print(Rule(style='dim'))
            console.print('[bold magenta]Assistant[/]:', end=' ')
            start = time.perf_counter()
            count = 0
            reply_parts: list[str] = []
            try:
                for chunk in self.model.generate_stream(
                    model_input_ids,
                    self.tokenizer,
                    max_tokens=self.args.max_tokens,
                    temp=self.args.temp,
                    top_k=self.args.top_k,
                ):
                    reply_parts.append(chunk)
                    console.print(chunk, style='bold white', end='')
                    count += 1
            except KeyboardInterrupt:
                console.print('\n[dim]Interrupted.[/dim]')
                continue
            reply = ''.join(reply_parts)
            if count > 0:
                elapsed = time.perf_counter() - start
                console.print(f'\n[dim]Tokens/s: {count / elapsed:.2f}, {count} tokens in {elapsed:.3f}s[/dim]')
            else:
                console.print()
            history.append(('assistant', reply))
            history = _clamp_history_by_tokens(
                self.tokenizer,
                self.args.system,
                history,
                max_ctx=self.args.max_ctx,
                reserve_gen=self.args.reserve_gen,
            )
            try:
                ctx_used = len(self.tokenizer.encode(build_prompt(self.args.system, history)))
                last_ctx_used = ctx_used
            except Exception:
                ctx_used = last_ctx_used
            console.print()
            console.print(f'[dim]ctx: {ctx_used}/{self.args.max_ctx}, reserve: {self.args.reserve_gen}[/dim]')
            gc.collect()

    def one_shot_answer(self, prompt: str) -> str:
        prompt = build_prompt(self.args.system, [('user', prompt)])
        model_input_ids = Tensor.of([self.tokenizer.encode(prompt)], dtype=dtype.int64)
        gc.collect()
        reply_parts: list[str] = []
        for chunk in self.model.generate_stream(
            model_input_ids,
            self.tokenizer,
            max_tokens=self.args.max_tokens,
            temp=self.args.temp,
            top_k=self.args.top_k,
        ):
            reply_parts.append(chunk)
        reply = ''.join(reply_parts)
        gc.collect()
        return reply


def _main() -> None:
    args = argparse.ArgumentParser(description='Run Qwen-3 model inference')
    args.add_argument('--prompt', type=str, help='Prompt to start generation')
    args.add_argument('--repl', action='store_true', help='Run interactive chat REPL')
    args.add_argument('--max_tokens', type=int, default=256, help='Maximum number of new tokens to generate')
    args.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    args.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')
    args.add_argument('--temp', type=float, default=0.6, help='Sampling temperature')
    args.add_argument('--system', type=str, default='You are a helpful assistant.', help='System prompt')
    args.add_argument('--max_ctx', type=int, default=4096, help='Max prompt context tokens (including system)')
    args.add_argument('--reserve_gen', type=int, default=512, help='Reserve tokens for generation headroom')
    args = args.parse_args()

    snapshot_file = _download_or_ensure_hf_file(repo_id=REPO_ID, filename='qwen3-4b-instruct-2507-bfloat16.mag')

    if not args.repl and not args.prompt:
        args.error('the --prompt argument is required when not running in REPL mode')

    model_context = GenerationContext(snapshot_file, args)

    if args.repl:  # REPL
        model_context.repl()
    else:  # Single shot mode
        reply = model_context.one_shot_answer(args.prompt)
        console.print(f'\n\nAnswer: {reply}', style='bold green')


if __name__ == '__main__':
    _main()
