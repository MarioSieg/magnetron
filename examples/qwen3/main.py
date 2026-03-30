# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

from inference import InferenceConfig, InferenceEngine, clamp_history_by_tokens
from model import build_prompt

console = Console()


def repl(engine: InferenceEngine) -> None:
    console.print(
        Panel.fit(
            Text('Magnetron Qwen3 REPL', style='bold white') + Text('\n/exit  /reset', style='dim'),
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
        reply_parts: list[str] = []
        console.print(Rule(style='dim'))
        console.print('[bold magenta]Assistant[/]:', end=' ')
        start = time.perf_counter()
        count = 0
        try:
            for chunk in engine.stream_chat(history):
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
        c = engine.config
        history = clamp_history_by_tokens(engine.tokenizer, c.system, history, max_ctx=c.max_ctx, reserve_gen=c.reserve_gen)
        try:
            last_ctx_used = len(engine.tokenizer.encode(build_prompt(c.system, history)))
        except Exception:
            pass
        console.print(f'[dim]ctx: {last_ctx_used}/{c.max_ctx}, reserve: {c.reserve_gen}[/dim]\n')


def _main() -> None:
    args = argparse.ArgumentParser(description='Run Qwen-3 model inference')
    args.add_argument('--prompt', type=str, help='Prompt to start generation')
    args.add_argument('--repl', action='store_true', help='Run interactive chat REPL')
    args.add_argument('--max_tokens', type=int, default=1024, help='Maximum number of new tokens to generate')
    args.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    args.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')
    args.add_argument('--temp', type=float, default=0.6, help='Sampling temperature')
    args.add_argument('--system', type=str, default='You are a helpful assistant.', help='System prompt')
    args.add_argument('--max_ctx', type=int, default=4096, help='Max prompt context tokens (including system)')
    args.add_argument('--reserve_gen', type=int, default=1024, help='Reserve tokens for generation headroom')
    args = args.parse_args()

    if not args.repl and not args.prompt:
        args.error('the --prompt argument is required when not running in REPL mode')

    engine = InferenceEngine(InferenceConfig.from_args(args))

    if args.repl:
        repl(engine)
    else:
        reply = engine.one_shot_answer(args.prompt)
        console.print(f'\n\nAnswer: {reply}', style='bold green')


if __name__ == '__main__':
    _main()
