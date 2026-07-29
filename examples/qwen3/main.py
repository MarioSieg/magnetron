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

from inference import InferenceConfig, InferenceEngine
from model import build_prompt

console = Console()


class Conversation:
    def __init__(self) -> None:
        self.history: list[tuple[str, str]] = []

    def push_assistant(self, reply: str) -> None:
        self.history.append(('assistant', reply))

    def push_user(self, reply: str) -> None:
        self.history.append(('user', reply))

    def clear(self) -> None:
        self.history.clear()

    def build_prompt(self, system: str) -> str:
        return build_prompt(system, self.history)

    def build_initial_prompt(self, system: str) -> str:
        return f'<|im_start|>system\n{system}<|im_end|>\n'

    def build_user_turn(self, user: str) -> str:
        return f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n'

    def build_assistant_end(self) -> str:
        return '<|im_end|>\n'


def repl(engine: InferenceEngine) -> None:
    console.print(
        Panel.fit(
            Text('Magnetron Qwen3 REPL', style='bold white') + Text('\n/exit', style='dim'),
            border_style='cyan',
        )
    )
    cfg = engine.config
    conv = Conversation()
    while True:
        user = Prompt.ask('[bold cyan]You[/]').strip()
        if not user:
            continue
        if user == '/exit':
            break
        is_first_turn = len(conv.history) == 0
        conv.push_user(user)
        console.print(Rule(style='dim'))
        console.print('[bold magenta]Assistant[/]:', end=' ')
        start = time.perf_counter()
        parts: list[str] = []
        count = 0
        try:
            prompt: str = conv.build_initial_prompt(cfg.system) + conv.build_user_turn(user) if is_first_turn else conv.build_user_turn(user)
            for chunk in engine.gen_stream(prompt, reset_cache=is_first_turn):
                parts.append(chunk)
                console.print(chunk, style='bold white', end='')
                count += 1
            conv.push_assistant(''.join(parts))
        except KeyboardInterrupt:
            console.print('\n[dim]Interrupted.[/dim]')
            continue
        if count > 0:
            elapsed = time.perf_counter() - start
            console.print(f'\n[dim]Tokens/s: {count / elapsed:.2f}, {count} tokens in {elapsed:.3f}s[/dim]')
        else:
            console.print()


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
    args.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args.add_argument('--snapshot', type=str, default=None, help='Choose local .mag snapshot file instead of HF repo')
    args.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Data type to run the model in',
    )
    args = args.parse_args()

    if not args.repl and not args.prompt:
        args.error('the --prompt argument is required when not running in REPL mode')

    engine = InferenceEngine(InferenceConfig.from_args(args))

    if args.repl:
        repl(engine)
    else:
        reply = engine.gen_one_shot(args.prompt)
        console.print(f'\n\nAnswer: {reply}', style='bold green')


if __name__ == '__main__':
    _main()
