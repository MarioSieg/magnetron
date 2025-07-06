import torch
import torch.nn as tnn
import magnetron.nn as nn

def test_state_dict_save_load_simple() -> None:
    model = nn.ModuleList([nn.Linear(3, 4), nn.Linear(4, 2)])
    sd = model.state_dict()
    model2 = nn.ModuleList([nn.Linear(3, 4), nn.Linear(4, 2)])
    model2.load_state_dict(sd)

def test_state_dict_save_load_complex() -> None:
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 2),
        nn.Tanh(),
        nn.Linear(2, 1)
    )
    sd = model.state_dict()
    model2 = nn.Sequential(
        nn.Linear(3, 4),
        nn.Tanh(),
        nn.Linear(4, 2),
        nn.Tanh(),
        nn.Linear(2, 1)
    )
    model2.load_state_dict(sd)

class MagMLP(nn.Module):
    def __init__(self, n_embd: int, n_inner: int | None = None):
        super().__init__()
        n_inner = n_inner or 4 * n_embd
        self.c_fc   = nn.Linear(n_embd, n_inner)
        self.c_proj = nn.Linear(n_inner, n_embd)

    def forward(self, x):
        pass

class MagSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.split_size = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        pass

class MagGPT2Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, n_inner: int | None = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn = MagSelfAttention(n_embd, n_head)
        self.ln_2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.mlp  = MagMLP(n_embd, n_inner)

    def forward(self, x):
        pass

class MagGPT2Model(nn.Module):
    def __init__(
            self,
            vocab_size: int   = 50_257,
            n_ctx: int        = 1_024,
            n_embd: int       = 768,
            n_layer: int      = 12,
            n_head: int       = 12,
            n_inner: int | None = None,
    ):
        super().__init__()
        self.wte   = nn.Embedding(vocab_size, n_embd)
        self.wpe   = nn.Embedding(n_ctx,    n_embd)
        self.drop  = nn.Dropout(0.1)
        self.h = nn.ModuleList([
            MagGPT2Block(n_embd, n_head, n_inner) for _ in range(n_layer)
        ])
        self.ln_f  = nn.LayerNorm(n_embd, eps=1e-5)

    def forward(self, input_ids):
        pass

class TorchMLP(tnn.Module):
    def __init__(self, n_embd: int, n_inner: int | None = None):
        super().__init__()
        n_inner = n_inner or 4 * n_embd
        self.c_fc   = tnn.Linear(n_embd, n_inner)
        self.c_proj = tnn.Linear(n_inner, n_embd)

    def forward(self, x):
        pass

class TorchSelfAttention(tnn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.split_size = n_embd
        self.c_attn = tnn.Linear(n_embd, 3 * n_embd)
        self.c_proj = tnn.Linear(n_embd, n_embd)

    def forward(self, x):
        pass

class TorchGPT2Block(tnn.Module):
    def __init__(self, n_embd: int, n_head: int, n_inner: int | None = None):
        super().__init__()
        self.ln_1 = tnn.LayerNorm(n_embd, eps=1e-5)
        self.attn = TorchSelfAttention(n_embd, n_head)
        self.ln_2 = tnn.LayerNorm(n_embd, eps=1e-5)
        self.mlp  = TorchMLP(n_embd, n_inner)

    def forward(self, x):
        pass

class TorchGPT2Model(tnn.Module):
    def __init__(
            self,
            vocab_size: int   = 50_257,
            n_ctx: int        = 1_024,
            n_embd: int       = 768,
            n_layer: int      = 12,
            n_head: int       = 12,
            n_inner: int | None = None,
    ):
        super().__init__()
        self.wte   = tnn.Embedding(vocab_size, n_embd)
        self.wpe   = tnn.Embedding(n_ctx,    n_embd)
        self.drop  = tnn.Dropout(0.1)
        self.h = tnn.ModuleList([
            TorchGPT2Block(n_embd, n_head, n_inner) for _ in range(n_layer)
        ])
        self.ln_f  = tnn.LayerNorm(n_embd, eps=1e-5)

    def forward(self, input_ids):
        pass

def test_state_dict_save_load_gpt2_vs_torch() -> None:
    gpt2_mag = MagGPT2Model()
    gpt2_torch = TorchGPT2Model()
    sd_mag = gpt2_mag.state_dict()
    sd_torch = gpt2_torch.state_dict()
    assert len(sd_mag) == len(sd_torch), f'State dicts should have the same number of parameters but got {len(sd_mag)} vs {len(sd_torch)}'
    for key in sd_mag:
        assert key in sd_torch, f'Key {key} not found in Torch state dict'
        assert sd_mag[key].shape == sd_torch[key].shape, f'Shape mismatch for key {key}: {sd_mag[key].shape} vs {sd_torch[key].shape}'


    gpt2_mag2 = MagGPT2Model()
    gpt2_mag2.load_state_dict(sd_mag)