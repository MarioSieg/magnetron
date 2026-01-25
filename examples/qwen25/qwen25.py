# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import gc
import argparse
import math
import time

from magnetron import Tensor, Snapshot, nn, dtype, context
from dataclasses import dataclass
from transformers import AutoTokenizer
from rich.console import Console

console = Console()


@dataclass
class Qwen25HyperParams:
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    sliding_window: int = 131072
    max_window_layers: int = 70
    bos_token_id: int = 151643
    eos_token_id: int = 151645


class MLP(nn.Module):
    def __init__(self, config: Qwen25HyperParams) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.inter_size: int = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.inter_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.inter_size, bias=False)
        self.down_proj = nn.Linear(self.inter_size, self.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    if n_rep == 1:
        return x
    B, n_kv, T, D = x.shape
    chunks: list[Tensor] = []
    for h in range(n_kv):  # TODO: repeat
        xh: Tensor = x[:, h : h + 1, :, :]
        for _ in range(n_rep):
            chunks.append(xh)
    return Tensor.cat(chunks, dim=1)


def _precompute_freq_cache(dim: int, theta: float, max_seq_len: int) -> tuple[Tensor, Tensor]:
    idx: Tensor = Tensor.arange(0, dim, 2, dtype=dtype.float32) / dim
    log_theta: float = math.log(theta)
    inv_freq: Tensor = Tensor.exp(-idx * log_theta)  # TODO: use pow
    seq: Tensor = Tensor.arange(stop=max_seq_len, dtype=dtype.float32)
    freqs: Tensor = seq.reshape(max_seq_len, 1) * inv_freq.reshape(1, -1)  # TODO: outer product
    cos_half: Tensor = Tensor.cos(freqs)
    sin_half: Tensor = Tensor.sin(freqs)
    cos: Tensor = Tensor.cat([cos_half, cos_half], dim=-1)
    sin: Tensor = Tensor.cat([sin_half, sin_half], dim=-1)
    return cos, sin


def _apply_rope(q: Tensor, k: Tensor, freq_cos: Tensor, freq_sin: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
    def _rot_half(x: Tensor) -> Tensor:
        half: int = x.shape[-1] >> 1
        x1: Tensor = x[:, :, :, :half]
        x2: Tensor = x[:, :, :, half:]
        return Tensor.cat([-x2, x1], dim=-1)

    cos: Tensor = freq_cos[idx]
    sin: Tensor = freq_sin[idx]
    batch_size, seq_len, head_size = cos.shape
    cos = cos.reshape(batch_size, 1, seq_len, head_size)
    sin = sin.reshape(batch_size, 1, seq_len, head_size)
    q_embed: Tensor = (q * cos) + (_rot_half(q) * sin)
    k_embed: Tensor = (k * cos) + (_rot_half(k) * sin)
    return q_embed, k_embed


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: Qwen25HyperParams) -> None:
        super().__init__()
        self.head_size = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.sliding_window = config.sliding_window
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_size)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_size)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_size)
        self.o_proj = nn.Linear(self.num_heads * self.head_size, config.hidden_size)

    def forward(
        self, x: Tensor, cos_freq: Tensor, sin_freq: Tensor, idx: Tensor, prev_kv: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        block_size, seq_len, _ = x.shape
        q: Tensor = self.q_proj(x)
        k_cur: Tensor = self.k_proj(x)
        v_cur: Tensor = self.v_proj(x)
        q = q.reshape(block_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k_cur = k_cur.reshape(block_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        v_cur = v_cur.reshape(block_size, seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        q, k_cur = _apply_rope(q, k_cur, cos_freq, sin_freq, idx)
        if prev_kv is not None:
            past_k, past_v = prev_kv
            if self.sliding_window is not None:
                max_past = max(0, self.sliding_window - k_cur.shape[2])
                if past_k.shape[2] > max_past:
                    past_k = past_k[:, :, -max_past:, :]
                    past_v = past_v[:, :, -max_past:, :]
            k: Tensor = Tensor.cat([past_k, k_cur], dim=2)
            v: Tensor = Tensor.cat([past_v, v_cur], dim=2)
        else:
            k: Tensor = k_cur
            v: Tensor = v_cur
        curr_kv: tuple[Tensor, Tensor] = (k, v)
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)
        scores: Tensor = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(self.head_size))
        q_len: int = q.shape[2]
        k_len: int = k.shape[2]
        k_pos_indices: Tensor = Tensor.arange(k_len).reshape(1, -1)
        q_pos_indices: Tensor = Tensor.arange(start=(k_len - q_len), stop=k_len).reshape(-1, 1)
        causal_mask: Tensor = k_pos_indices <= q_pos_indices
        keep: Tensor = causal_mask.cast(scores.dtype)
        additive_mask: Tensor = (1.0 - keep) * -1e4
        additive_mask = additive_mask.reshape(1, 1, q_len, k_len)
        masked_scores: Tensor = scores + additive_mask
        attn_weights: Tensor = masked_scores.softmax(dim=-1)
        out: Tensor = (attn_weights @ v).transpose(1, 2).reshape(block_size, seq_len, -1)
        return self.o_proj(out), curr_kv


class Block(nn.Module):
    def __init__(self, config: Qwen25HyperParams) -> None:
        super().__init__()
        self.self_attn = SlidingWindowAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, x: Tensor, freq_cos: Tensor, freq_sin: Tensor, idx: Tensor, prev_kv: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        ln_out: Tensor = self.input_layernorm(x)
        attn_out, present_kv = self.self_attn(ln_out, freq_cos, freq_sin, idx, prev_kv)
        h: Tensor = x + attn_out
        return h + self.mlp(self.post_attention_layernorm(h)), present_kv


class Qwen25Model(nn.Module):
    def __init__(self, config: Qwen25HyperParams) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos_cache, sin_cache = _precompute_freq_cache(
            config.hidden_size // config.num_attention_heads, config.rope_theta, config.max_position_embeddings
        )
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def _load_from_snapshot(self, snapshot_file: str) -> None:
        with Snapshot.read(snapshot_file) as snap:
            for name, param in self.named_parameters():
                tensor = snap.get_tensor(name)
                if tuple(tensor.shape) != tuple(param.x.shape):
                    raise RuntimeError(f'Shape mismatch for {name}: {tensor.shape} != {param.shape}')
                if tensor.dtype != param.x.dtype:
                    raise RuntimeError(f'Dtype mismatch for {name}: {tensor.dtype} != {param.dtype}')
                param.x = tensor

    @staticmethod
    def from_pretrained_snapshot(snapshot_file: str) -> 'Qwen25Model':
        console.print(f'Loading QWEN-2.5 model from snapshot: {snapshot_file}', style='dim')
        model = Qwen25Model(Qwen25HyperParams())
        model._load_from_snapshot(snapshot_file)
        return model

    def forward(
        self, x: Tensor, idx: Tensor, prev_kv: list[tuple[Tensor, Tensor]] | None = None, use_cache: bool = True
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]] | None:
        h = self.embed_tokens(x)

        next_kv = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = prev_kv[i] if prev_kv is not None else None
            h, present_kv = layer(h, self.cos_cache, self.sin_cache, idx=idx, prev_kv=layer_cache)
            if use_cache:
                next_kv.append(present_kv)

        h = self.norm(h)
        return self.lm_head(h), next_kv

    def generate(self, idx: Tensor, tokenizer: AutoTokenizer, max_tokens: int, temp: float = 1.0, top_k: int = 10) -> str:
        start: float = time.perf_counter()
        idx = idx.reshape(1, -1)
        in_len: int = idx.shape[1]
        logits, prev_kv = self(idx, idx=Tensor.arange(stop=in_len).reshape(1, -1), prev_kv=None)
        next_logits: Tensor = logits[:, -1, :] / temp
        tokens: str = ''
        curr_len: int = in_len
        count: int = 0
        for _ in range(max_tokens):
            logits_1d: Tensor = next_logits.reshape(-1)
            top_vals, top_idx = logits_1d.topk(top_k, dim=0, largest=True, sorted=False)
            pick: Tensor = top_vals.softmax(dim=-1).reshape(1, -1).multinomial(num_samples=1)
            tok_id: int = int(top_idx[pick[0, 0]].item())
            if tok_id == self.config.eos_token_id:
                break
            token: str = tokenizer.decode(tok_id, skip_special_tokens=True)
            tokens += token
            console.print(token, style='bold white', end='')
            input_ids = Tensor.of([tok_id], dtype=dtype.int64).reshape(1, 1)
            logits, prev_kv = self(input_ids, idx=Tensor.of([curr_len], dtype=dtype.int64).reshape(1, 1), prev_kv=prev_kv)
            next_logits = logits[:, -1, :] / temp
            curr_len += 1
            count += 1
        if count > 0:
            elapsed = time.perf_counter() - start
            console.print(f'\nTokens/s: {count / elapsed:.2f}, {count} tokens in {elapsed:.3f}s', style='dim')
        return tokens


def _build_prompt(system: str, messages: list[tuple[str, str]]) -> str:
    out = [f'<|im_start|>system\n{system}<|im_end|>\n']
    for role, content in messages:
        out.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
    out.append('<|im_start|>assistant\n')
    return ''.join(out)


def _clamp_history_by_tokens(
    tokenizer: AutoTokenizer,
    system: str,
    history: list[tuple[str, str]],
    max_ctx: int,
    reserve_gen: int,
) -> list[tuple[str, str]]:
    budget = max(256, max_ctx - reserve_gen)  # avoid too-small budgets
    prompt = _build_prompt(system, history)
    n = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    if n <= budget:
        return history
    trimmed = history[:]
    while trimmed:
        trimmed.pop(0)
        if trimmed and trimmed[0][0] == 'assistant':
            trimmed.pop(0)
        prompt = _build_prompt(system, trimmed)
        n = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        if n <= budget:
            return trimmed
    return []


class GenerationContext:
    def __init__(self, args: argparse.Namespace) -> None:
        start = time.perf_counter()
        context.stop_grad_recorder()
        context.manual_seed(args.seed)
        self.model = Qwen25Model.from_pretrained_snapshot(args.snapshot)
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
        self.args = args
        end = time.perf_counter()
        console.print(f'Ready in {end - start:.2f}s', style='dim')
        gc.collect()

    def repl(self) -> None:
        from rich.panel import Panel
        from rich.rule import Rule
        from rich.text import Text
        from rich.prompt import Prompt
        console.print(
            Panel.fit(
                Text('Qwen2.5 REPL', style='bold white') + Text('\n/exit  /reset', style='dim'),
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
            prompt = _build_prompt(self.args.system, history)
            model_inputs = self.tokenizer([prompt], return_tensors='np', add_special_tokens=False)
            model_input_ids = Tensor.of(model_inputs.input_ids.tolist(), dtype=dtype.int64)
            console.print(Rule(style='dim'))
            console.print('[bold magenta]Assistant[/]:', end=' ')
            try:
                reply = self.model.generate(
                    model_input_ids,
                    self.tokenizer,
                    max_tokens=self.args.max_tokens,
                    temp=self.args.temp,
                    top_k=self.args.top_k,
                )
            except KeyboardInterrupt:
                console.print('\n[dim]Interrupted.[/dim]')
                continue
            history.append(('assistant', reply))
            history = _clamp_history_by_tokens(
                self.tokenizer,
                self.args.system,
                history,
                max_ctx=self.args.max_ctx,
                reserve_gen=self.args.reserve_gen,
            )
            try:
                ctx_used = len(self.tokenizer(_build_prompt(self.args.system, history), add_special_tokens=False).input_ids)
                last_ctx_used = ctx_used
            except Exception:
                ctx_used = last_ctx_used
            console.print()
            console.print(f'[dim]ctx: {ctx_used}/{self.args.max_ctx}, reserve: {self.args.reserve_gen}[/dim]')
            gc.collect()

    def one_shot_answer(self, prompt: str) -> str:
        prompt = _build_prompt(self.args.system, [('user', prompt)])
        model_input_ids = Tensor.of(self.tokenizer([prompt], return_tensors='np').input_ids.tolist(), dtype=dtype.int64)
        gc.collect()
        reply = self.model.generate(model_input_ids, self.tokenizer, max_tokens=self.args.max_tokens, temp=self.args.temp, top_k=self.args.top_k)
        gc.collect()
        return reply


def _main() -> None:
    args = argparse.ArgumentParser(description='Run QWEN-2.5 model inference')
    args.add_argument('--prompt', type=str, help='Prompt to start generation')
    args.add_argument('--repl', action='store_true', help='Run interactive chat REPL')
    args.add_argument('--max_tokens', type=int, default=256, help='Maximum number of new tokens to generate')
    args.add_argument('--top_k', type=int, default=200, help='Top-k sampling')
    args.add_argument('--seed', type=int, default=3407, help='Random seed for reproducibility')
    args.add_argument('--temp', type=float, default=0.6, help='Sampling temperature')
    args.add_argument('--snapshot', type=str, default='qwen2.5-3b-instruct-float32.mag', help='Path to model snapshot file')
    args.add_argument('--system', type=str, default='You are a helpful assistant.', help='System prompt')
    args.add_argument('--max_ctx', type=int, default=4096, help='Max prompt context tokens (including system)')
    args.add_argument('--reserve_gen', type=int, default=512, help='Reserve tokens for generation headroom')
    args = args.parse_args()

    if not args.repl and not args.prompt:
        args.error('the --prompt argument is required when not running in REPL mode')

    model_context = GenerationContext(args)

    if args.repl:  # REPL
        model_context.repl()
    else:  # Single shot mode
        reply = model_context.one_shot_answer(args.prompt)
        console.print(f'\n\nAnswer: {reply}', style='bold green')


if __name__ == '__main__':
    _main()
