# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import gc
import math
from collections.abc import Iterator
from typing import Any

from magnetron import Tensor, Snapshot, nn, dtype
from dataclasses import dataclass

_EOS: set[int] = {151645, 151643}


@dataclass
class Qwen3HyperParams:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 8192  # 262144
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    rope_theta: float = 5_000_000.0
    sliding_window: int | None = None
    bos_token_id: int = 151643
    eos_token_id: int = 151645


class MLP(nn.Module):
    def __init__(self, config: Qwen3HyperParams) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.inter_size: int = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.inter_size, bias=False, dtype=dtype.bfloat16, init=False)
        self.up_proj = nn.Linear(self.hidden_size, self.inter_size, bias=False, dtype=dtype.bfloat16, init=False)
        self.down_proj = nn.Linear(self.inter_size, self.hidden_size, bias=False, dtype=dtype.bfloat16, init=False)

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
    idx = Tensor.arange(0, dim, 2, dtype=dtype.float32) / dim
    inv_freq = Tensor.exp(-idx * math.log(theta))  # TODO: pow
    seq = Tensor.arange(stop=max_seq_len, dtype=dtype.float32)
    freqs = seq.reshape(max_seq_len, 1) * inv_freq.reshape(1, -1)
    cos_half = Tensor.cos(freqs)
    sin_half = Tensor.sin(freqs)
    cos = Tensor.cat([cos_half, cos_half], dim=-1).cast(dtype.bfloat16)
    sin = Tensor.cat([sin_half, sin_half], dim=-1).cast(dtype.bfloat16)
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
    def __init__(self, config: Qwen3HyperParams) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.sliding_window = config.sliding_window
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=dtype.bfloat16, init=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype.bfloat16, init=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype.bfloat16, init=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False, dtype=dtype.bfloat16, init=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype.bfloat16, init=False)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype.bfloat16, init=False)

    def forward(
        self, x: Tensor, cos_freq: Tensor, sin_freq: Tensor, idx: Tensor, prev_kv: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k_cur = self.k_proj(x)
        v_cur = self.v_proj(x)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_cur = k_cur.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_cur = v_cur.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k_cur = self.k_norm(k_cur)
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
            k = k_cur
            v = v_cur
        curr_kv: tuple[Tensor, Tensor] = (k, v)
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)
        scores: Tensor = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(self.head_dim))
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
        out: Tensor = (attn_weights @ v).transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), curr_kv


class Block(nn.Module):
    def __init__(self, config: Qwen3HyperParams) -> None:
        super().__init__()
        self.self_attn = SlidingWindowAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, init=False, dtype=dtype.bfloat16)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps, init=False, dtype=dtype.bfloat16)

    def forward(
        self, x: Tensor, freq_cos: Tensor, freq_sin: Tensor, idx: Tensor, prev_kv: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        ln_out: Tensor = self.input_layernorm(x)
        attn_out, present_kv = self.self_attn(ln_out, freq_cos, freq_sin, idx, prev_kv)
        h: Tensor = x + attn_out
        return h + self.mlp(self.post_attention_layernorm(h)), present_kv


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3HyperParams) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, init=False, dtype=dtype.bfloat16)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps, init=False, dtype=dtype.bfloat16)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype.bfloat16, init=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        cos_cache, sin_cache = _precompute_freq_cache(config.head_dim, config.rope_theta, config.max_position_embeddings)
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    def _load_from_snapshot(self, snapshot_file: str) -> None:
        with Snapshot.read(snapshot_file) as snap:
            for name, param in self.named_parameters():
                tensor = snap.get_tensor(name)
                if tuple(tensor.shape) != tuple(param.x.shape):
                    raise RuntimeError(f'Shape mismatch for {name}: {tensor.shape} != {param.shape}')
                if tensor.dtype != param.x.dtype:
                    raise RuntimeError(f'Dtype mismatch for {name}: {tensor.dtype} != {param.x.dtype}')
                param.x = tensor

    @staticmethod
    def from_pretrained_snapshot(snapshot_file: str) -> 'Qwen3Model':
        model = Qwen3Model(Qwen3HyperParams())
        model._load_from_snapshot(snapshot_file)
        gc.collect()
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

    def generate_stream(
        self,
        idx: Tensor,
        tokenizer: Any,
        max_tokens: int,
        temp: float = 1.0,
        top_k: int = 10,
    ) -> Iterator[str]:
        idx = idx.reshape(1, -1)
        in_len: int = idx.shape[1]
        logits, prev_kv = self(idx, idx=Tensor.arange(stop=in_len).reshape(1, -1), prev_kv=None)
        next_logits: Tensor = logits[:, -1, :] / temp
        curr_len: int = in_len
        gen_ids: list[int] = []
        concated: str = ''
        for _ in range(max_tokens):
            logits_1d: Tensor = next_logits.reshape(-1)
            top_vals, top_idx = logits_1d.topk(top_k, dim=0, largest=True, sorted=False)
            pick: Tensor = top_vals.softmax(dim=-1).reshape(1, -1).multinomial(num_samples=1)
            tok_id: int = int(top_idx[pick[0, 0]].item())
            if tok_id == self.config.eos_token_id or tok_id in _EOS:
                return
            gen_ids.append(tok_id)
            text: str = tokenizer.decode(gen_ids)
            if len(text) > len(concated):
                delta = text[len(concated) :]
                delta = delta.replace('\ufffd', '')
                concated = text
                yield delta
            input_ids = Tensor.of([tok_id], dtype=dtype.int64).reshape(1, 1)
            logits, prev_kv = self(
                input_ids,
                idx=Tensor.of([curr_len], dtype=dtype.int64).reshape(1, 1),
                prev_kv=prev_kv,
            )
            next_logits = logits[:, -1, :] / temp
            curr_len += 1


def build_prompt(system: str, messages: list[tuple[str, str]]) -> str:
    out = [f'<|im_start|>system\n{system}<|im_end|>\n']
    for role, content in messages:
        out.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
    out.append('<|im_start|>assistant\n')
    return ''.join(out)
