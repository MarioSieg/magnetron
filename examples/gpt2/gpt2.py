from __future__ import annotations

import codecs
import math
import time
import argparse
import magnetron as mag
import magnetron.nn as nn
import tiktoken
from collections.abc import Iterator
from dataclasses import dataclass
from magnetron import FFI, C
from rich.console import Console

console = Console()

tok = tiktoken.get_encoding('gpt2')

EOS: str = '<|endoftext|>'
encode = lambda x: tok.encode(x, allowed_special={EOS})
decode = lambda x: tok.decode(x)
EOS_ID: int = encode(EOS)[0]


@dataclass
class GPT2HyperParams:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2HyperParams) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias, init_weights=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias, init_weights=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            'bias',
            mag.Tensor.ones(config.block_size, config.block_size).tril().view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = att.softmax(dim=-1)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPT2HyperParams) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias, init_weights=False)
        self.gelu = nn.GeLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias, init_weights=False)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPT2HyperParams) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, config: GPT2HyperParams) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False, init_weights=False)
        self.transformer.wte.weight = self.lm_head.weight
        console.print(f'Parameter count: {self.get_num_params(False) // 1e6}M', style='dim')

    @classmethod
    @mag.no_grad()
    def from_pretrained(cls, model_type: str = 'gpt2') -> GPT2:
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        cfg = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        cfg['vocab_size'] = 50257
        cfg['block_size'] = 1024
        cfg['bias'] = True
        cfg = GPT2HyperParams(**cfg)
        console.print(f'Loading {model_type} with config: {cfg}', style='dim')
        model = cls(cfg)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f'mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}'

        def copy(r: mag.Tensor, x) -> None:  # TODO
            assert x.is_contiguous and r.is_contiguous
            assert r.shape == x.shape, f'Shape mismatch: {r.shape} != {x.shape}'
            assert r.is_contiguous and x.is_contiguous, 'Both tensors must be contiguous for copy operation'
            bytes = x.numel() * x.element_size()
            C.mag_tensor_fill_from_raw_bytes(r._ptr, FFI.cast('void*', x.data_ptr()), bytes)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with mag.no_grad():
                    copy(sd[k], sd_hf[k].T.contiguous())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with mag.no_grad():
                    copy(sd[k], sd_hf[k].contiguous())

        return model

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.x.numel for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.x.numel
        return n_params

    def forward(self, idx: mag.Tensor) -> mag.Tensor:
        b, t = idx.shape
        assert t <= self.config.block_size, f'Block size {self.config.block_size} exceeded by input length {t}'
        pos = mag.Tensor.arange(0, t, dtype=mag.int32)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])
        return logits

    @mag.no_grad()
    def generate(self, prompt: str, max_tokens: int, temp: float = 1.0) -> str:
        tokens = encode(prompt)
        for _ in range(max_tokens):
            ctx = tokens[-self.config.block_size :] if len(tokens) > self.config.block_size else tokens
            idx = mag.Tensor.of(ctx, dtype=mag.int32)[None, ...]
            logits = self(idx)[:, -1, :] / temp
            probs = logits.softmax(dim=-1)
            tokens.append(probs.multinomial(num_samples=1).item())
        return decode(tokens)

    @mag.no_grad()
    def generate_stream(self, prompt: str, max_tokens: int, temp: float = 1.0) -> Iterator[str]:
        tokens = encode(prompt)
        dec = codecs.getincrementaldecoder('utf-8')()
        start = time.perf_counter()
        n = 0
        for _ in range(max_tokens):
            ctx = tokens[-self.config.block_size :] if len(tokens) > self.config.block_size else tokens
            idx = mag.Tensor.of(ctx, dtype=mag.int32)[None, ...]
            logits = self(idx)[:, -1, :] / temp
            probs = logits.softmax(dim=-1)
            next_id = probs.multinomial(num_samples=1).item()
            tokens.append(next_id)
            n += 1
            delta = dec.decode(tok.decode_single_token_bytes(next_id))
            if delta:
                yield delta
        tail = dec.decode(b'', final=True)
        if tail:
            yield tail
        elapsed = time.perf_counter() - start
        if n > 0:
            console.print(f'\nTokens/s: {n / elapsed:.2f}, {n} tokens in {elapsed:.3f}s', style='dim')


if __name__ == '__main__':
    mag.active_context().stop_grad_recorder()

    args = argparse.ArgumentParser(description='Run GPT-2 model inference')
    args.add_argument('prompt', type=str, help='Prompt to start generation')
    args.add_argument('--model', type=str, default='gpt2', help='Model type (gpt2, gpt2-medium, gpt2-large, gpt2-xl)')
    args.add_argument('--max_tokens', type=int, default=128, help='Maximum number of new tokens to generate')
    args.add_argument('--temp', type=float, default=0.6, help='Temperature for sampling')
    args.add_argument('--no-stream', action='store_true', help='Disable streaming output')
    args = args.parse_args()

    model = GPT2.from_pretrained(args.model)
    puts = lambda s: console.print(s, style='bold white', end='')
    puts(args.prompt)
    if not args.no_stream:
        for chunk in model.generate_stream(args.prompt, max_tokens=args.max_tokens, temp=args.temp):
            puts(chunk)
    else:
        puts(model.generate(args.prompt, max_tokens=args.max_tokens, temp=args.temp))
