import math
import torch

import magnetron as mag
import magnetron.nn as nn
from dataclasses import dataclass

from magnetron import FFI, C


@dataclass
class GPTHParams:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTHParams) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
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
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTHParams) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GeLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTHParams) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTHParams) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        print(f'Parameter count: {self.get_num_params(False) // 1e6}M')

    @classmethod
    def from_pretrained(cls, model_type):
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
        cfg = GPTHParams(**cfg)
        print(f'Loading {model_type} with config: {cfg}')
        model = GPT(cfg)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        def copy(r: mag.Tensor, x: torch.tensor) -> None: # TODO
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
    def generate(self, idx: mag.Tensor, max_tokens: int, temp: float = 1.0) -> mag.Tensor:
        for _ in range(max_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temp
            probs = logits.softmax(dim=-1)
            idx_next = torch.multinomial(torch.tensor(probs.tolist()), num_samples=1) # TODO
            idx = mag.Tensor.of(torch.cat((torch.tensor(idx.tolist()), idx_next), dim=1).tolist()) # TODO
        return idx
