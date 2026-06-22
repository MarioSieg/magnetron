# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
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
from enum import Enum, unique

from magnetron import Tensor, Snapshot, nn, dtype, context
from dataclasses import dataclass

_EOS: set[int] = {151645, 151643}


@unique
class SamplingStrategy(Enum):
    GREEDY = 'greedy'
    TOPK = 'topk'


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
    sampling_strategy: SamplingStrategy = SamplingStrategy.GREEDY


class KVLayerCache:
    def __init__(self, batch_size: int, num_kv_heads: int, max_seq_len: int, head_dim: int) -> None:
        self.k: Tensor = Tensor.zeros(batch_size, num_kv_heads, max_seq_len, head_dim)
        self.v: Tensor = Tensor.zeros(batch_size, num_kv_heads, max_seq_len, head_dim)
        self.pos: int = 0

    def append(self, curr_k: Tensor, curr_v: Tensor, sliding_window: int | None = None) -> tuple[Tensor, Tensor]:
        T: int = curr_k.shape[2]
        start: int = self.pos
        end: int = start + T
        if end > self.k.shape[2]:
            raise RuntimeError(f'KV cache overflow: {end} > {self.k.shape[2]}')
        self.k[:, :, start:end, :] = curr_k
        self.v[:, :, start:end, :] = curr_v
        self.pos = end
        view_start: int = 0
        if sliding_window is not None:
            view_start = max(end - sliding_window, 0)
        return self.k[:, :, view_start:end, :], self.v[:, :, view_start:end, :]

    def clear(self) -> None:
        self.pos = 0


class KVCache:
    def __init__(self, cfg: Qwen3HyperParams, batch_size: int = 1, max_seq_len: int | None = None) -> None:
        max_seq_len = max_seq_len or cfg.max_position_embeddings
        self.layers: list[KVLayerCache] = [
            KVLayerCache(batch_size=batch_size, num_kv_heads=cfg.num_key_value_heads, max_seq_len=max_seq_len, head_dim=cfg.head_dim)
            for _ in range(cfg.num_hidden_layers)
        ]
        assert all(layer.pos == self.cache_pos for layer in self.layers)

    def __getitem__(self, idx: int) -> KVLayerCache:
        return self.layers[idx]

    @property
    def cache_pos(self) -> int:
        return self.layers[0].pos

    def clear(self) -> None:
        for layer in self.layers:
            layer.clear()


class MLP(nn.Module):
    def __init__(self, cfg: Qwen3HyperParams) -> None:
        super().__init__()
        self.hidden_size: int = cfg.hidden_size
        self.inter_size: int = cfg.intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size,
            self.inter_size,
            bias=False,
            weight_init=nn.init.EmptyInitStrategy(),
            bias_init=nn.init.EmptyInitStrategy(),
        )
        self.up_proj = nn.Linear(
            self.hidden_size,
            self.inter_size,
            bias=False,
            weight_init=nn.init.EmptyInitStrategy(),
            bias_init=nn.init.EmptyInitStrategy(),
        )
        self.down_proj = nn.Linear(
            self.inter_size,
            self.hidden_size,
            bias=False,
            weight_init=nn.init.EmptyInitStrategy(),
            bias_init=nn.init.EmptyInitStrategy(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


def _precompute_freq_cache(dim: int, theta: float, max_seq_len: int) -> tuple[Tensor, Tensor]:
    inv_freq = (theta ** -(Tensor.arange(0, dim, 2, dtype=dtype.float32) / dim)).reshape(1, -1)
    freqs = Tensor.arange(stop=max_seq_len, dtype=dtype.float32).reshape(max_seq_len, 1) * inv_freq
    cos_half = freqs.cos()
    sin_half = freqs.sin()
    cos = Tensor.cat([cos_half, cos_half], dim=-1).cast(context.get_default_dtype())
    sin = Tensor.cat([sin_half, sin_half], dim=-1).cast(context.get_default_dtype())
    return cos, sin


def _apply_rope(q: Tensor, k: Tensor, freq_cos: Tensor, freq_sin: Tensor, idx: Tensor) -> tuple[Tensor, Tensor]:
    def _rot_half(x: Tensor) -> Tensor:
        half: int = x.shape[-1] >> 1
        x1 = x[:, :, :, :half]
        x2 = x[:, :, :, half:]
        return Tensor.cat([-x2, x1], dim=-1)

    cos = freq_cos[idx]
    sin = freq_sin[idx]
    batch_size, seq_len, head_size = cos.shape
    cos = cos.reshape(batch_size, 1, seq_len, head_size)
    sin = sin.reshape(batch_size, 1, seq_len, head_size)
    q_embed = (q * cos) + (_rot_half(q) * sin)
    k_embed = (k * cos) + (_rot_half(k) * sin)
    return q_embed, k_embed


class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg: Qwen3HyperParams) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.sliding_window = cfg.sliding_window
        self.q_proj = nn.Linear(
            cfg.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            weight_init=nn.init.EmptyInitStrategy(),
            bias_init=nn.init.EmptyInitStrategy(),
        )
        self.k_proj = nn.Linear(
            cfg.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            weight_init=nn.init.EmptyInitStrategy(),
            bias_init=nn.init.EmptyInitStrategy(),
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            weight_init=nn.init.EmptyInitStrategy(),
            bias_init=nn.init.EmptyInitStrategy(),
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            cfg.hidden_size,
            bias=False,
            weight_init=nn.init.EmptyInitStrategy(),
            bias_init=nn.init.EmptyInitStrategy(),
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, weight_init=nn.init.EmptyInitStrategy())
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, weight_init=nn.init.EmptyInitStrategy())

    def forward(self, x: Tensor, cos_freq: Tensor, sin_freq: Tensor, idx: Tensor, cache: KVLayerCache | None = None) -> Tensor:
        B, T, _ = x.shape
        curr_k = self.k_norm(self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2))
        curr_v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2))
        q, curr_k = _apply_rope(q, curr_k, cos_freq, sin_freq, idx)
        if cache is not None:
            k, v = cache.append(curr_k, curr_v, self.sliding_window)
        else:
            k, v = curr_k, curr_v
        qg = q.reshape(B, self.num_kv_heads, self.n_rep, T, self.head_dim)
        scores = Tensor.einsum('bkgtd, bksd -> bkgts', qg, k)
        scores *= 1.0 / math.sqrt(self.head_dim)
        q_len: int = q.shape[2]
        k_len: int = k.shape[2]
        if q_len == 1:
            attn = scores.softmax(dim=-1)
        else:
            k_pos_indices = Tensor.arange(k_len).reshape(1, -1)
            q_pos_indices = Tensor.arange(start=(k_len - q_len), stop=k_len).reshape(-1, 1)
            mask = Tensor.where(k_pos_indices <= q_pos_indices, 0.0, -1e4)
            mask = mask.cast(scores.dtype).reshape(1, 1, 1, q_len, k_len)
            attn = (scores + mask).softmax(dim=-1)
        out = Tensor.einsum('bkgts, bksd -> bkgtd', attn, v)
        out = out.reshape(B, self.num_heads, T, self.head_dim)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class Block(nn.Module):
    def __init__(self, cfg: Qwen3HyperParams) -> None:
        super().__init__()
        self.self_attn = SlidingWindowAttention(cfg)
        self.mlp = MLP(cfg)
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, weight_init=nn.init.EmptyInitStrategy())
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, weight_init=nn.init.EmptyInitStrategy())

    def forward(self, x: Tensor, freq_cos: Tensor, freq_sin: Tensor, idx: Tensor, cache: KVLayerCache | None = None) -> Tensor:
        h = x + self.self_attn(
            self.input_layernorm(x),
            freq_cos,
            freq_sin,
            idx,
            cache,
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3HyperParams) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, weight_init=nn.init.EmptyInitStrategy())
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, weight_init=nn.init.EmptyInitStrategy())
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                cfg.hidden_size,
                cfg.vocab_size,
                bias=False,
                weight_init=nn.init.EmptyInitStrategy(),
                bias_init=nn.init.EmptyInitStrategy(),
            )
        cos_cache, sin_cache = _precompute_freq_cache(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings)
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache
        self.cache = self._alloc_kv_cache()

    def _alloc_kv_cache(self) -> KVCache:
        return KVCache(self.cfg)

    def _load_from_snapshot(self, snapshot_file: str) -> None:
        with Snapshot.read(snapshot_file) as snap:
            for name, param in self.named_parameters():
                tensor = snap.get_tensor(name)
                if tuple(tensor.shape) != tuple(param.shape):
                    raise RuntimeError(f'Shape mismatch for {name}: {tensor.shape} != {param.shape}')
                if tensor.dtype != param.dtype:
                    raise RuntimeError(f'Dtype mismatch for {name}: {tensor.dtype} != {param.dtype}')
                if context.get_default_device() != 'cpu:0':
                    param.data = tensor.transfer(context.get_default_device())
                else:
                    param.data = tensor

    @staticmethod
    def from_pretrained_snapshot(snapshot_file: str, params: Qwen3HyperParams) -> 'Qwen3Model':
        model = Qwen3Model(params)
        model._load_from_snapshot(snapshot_file)
        gc.collect()
        return model

    def forward(
        self,
        x: Tensor,
        idx: Tensor,
    ) -> Tensor:
        h = self.embed_tokens(x)
        for i, layer in enumerate(self.layers):
            layer_cache = self.cache[i] if self.cache is not None else None
            h = layer(h, self.cos_cache, self.sin_cache, idx, cache=layer_cache)
        h = self.norm(h)
        return Tensor.einsum('...h, vh -> ...v', h, self.embed_tokens.weight) if self.cfg.tie_word_embeddings else self.lm_head(h)

    def generate_stream(
        self,
        idx: Tensor,
        tokenizer: Any,
        max_tokens: int,
        temp: float = 1.0,
        top_k: int = 10,
        reset_cache: bool = False,
    ) -> Iterator[str]:

        def sample(logits: Tensor, strategy: SamplingStrategy) -> int:  # Sample according to strategy
            match strategy:
                case SamplingStrategy.GREEDY:
                    return int(logits.argmax(dim=0).item())
                case SamplingStrategy.TOPK:
                    top_vals, top_idx = logits.topk(top_k, dim=0, largest=True, sorted=False)
                    return int(top_idx[top_vals.softmax(dim=-1).reshape(1, -1).multinomial(num_samples=1)[0, 0]].item())
                case _:
                    raise RuntimeError(f'Invalid sampling strategy: {strategy}')

        if reset_cache:
            self.cache.clear()
        idx = idx.reshape(1, -1)
        start_pos: int = self.cache.cache_pos
        T: int = idx.shape[1]
        logits = self(
            idx,
            idx=Tensor.arange(start=start_pos, stop=start_pos + T).reshape(1, -1),
        )
        next_logits = logits[:, -1, :] / temp
        curr_len: int = start_pos + T
        pending: list[int] = []

        for _ in range(max_tokens):
            tok_id: int = sample(next_logits.reshape(-1), self.cfg.sampling_strategy)
            if tok_id == self.cfg.eos_token_id or tok_id in _EOS:
                return
            pending.append(tok_id)
            delta: str = tokenizer.decode(pending)
            if delta and '\ufffd' not in delta:
                yield delta
                pending.clear()
            input_ids = Tensor([tok_id], dtype=dtype.int64).reshape(1, 1)
            logits = self(input_ids, idx=Tensor([curr_len], dtype=dtype.int64).reshape(1, 1))
            next_logits = logits[:, -1, :] / temp
            curr_len += 1


def build_prompt(system: str, messages: list[tuple[str, str]]) -> str:
    out = [f'<|im_start|>system\n{system}<|im_end|>\n']
    for role, content in messages:
        out.append(f'<|im_start|>{role}\n{content}<|im_end|>\n')
    out.append('<|im_start|>assistant\n')
    return ''.join(out)
