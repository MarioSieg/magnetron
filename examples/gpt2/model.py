import math
import torch

import magnetron as mag
import magnetron.nn as nn
from dataclasses import dataclass

class LayerNorm(nn.Module): # Torch helping layer norm till we have fixed reductions
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(mag.Tensor.ones(ndim))
        self.bias = nn.Parameter(mag.Tensor.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: mag.Tensor) -> mag.Tensor: # TODO: ASAP
        x = torch.tensor(x.tolist())
        xm = x - x.mean(dim=-1, keepdim=True)
        var = (xm * xm).mean(dim=-1, keepdim=True)
        x_hat = xm / (var + self.eps).sqrt()
        y = torch.tensor(self.weight.x.tolist()) * x_hat
        if self.bias is not None:
            y = y + torch.tensor(self.bias.x.tolist())
        return mag.Tensor.of(y.tolist())


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            'mask',
            mag.Tensor.ones(config.block_size, config.block_size).tril().view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = att.softmax(dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
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
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: mag.Tensor) -> mag.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
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
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                p.x.fill_random_normal_(mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print(f'Parameter count: {self.get_num_params(False) // 1e6}M')

    def get_num_params(self, non_embedding: bool = False) -> int:
        n_params = sum(p.x.numel for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.x.numel
        return n_params

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.x.fill_random_normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.x.zeros_()
        elif isinstance(module, nn.Embedding):
            module.weight.x.fill_random_normal_(mean=0.0, std=0.02)

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
            idx_next = torch.multinomial(torch.tensor(probs.tolist()), num_samples=1)
            idx = mag.Tensor.of(torch.cat((torch.tensor(idx.tolist()), idx_next), dim=1).tolist())
        return idx
