# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch
import torch.nn as tnn
import torch.nn.functional as F

import magnetron as mag
import magnetron.nn as nn
from magnetron.nn.init import inplace_init, UniformInitStrategy

from .common import *

_GN_CASES = [
    # (shape, num_groups)
    ((2, 6), 2),
    ((2, 6, 5), 2),
    ((3, 8, 4, 4), 4),
    ((1, 4, 3, 5), 1),
    ((2, 4, 2, 3, 3), 4),
]


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('affine', [True, False])
@pytest.mark.parametrize('case', _GN_CASES)
def test_group_norm(device: str, affine: bool, case) -> None:
    shape, groups = case
    channels = shape[1]
    with mag.device(device):
        layer = nn.GroupNorm(groups, channels, eps=1e-5, affine=affine)
        if affine:
            inplace_init(layer.weight, UniformInitStrategy(0.5, 1.5))
            inplace_init(layer.bias, UniformInitStrategy(-0.5, 0.5))
        x = random_tensor(shape, dtype.float32, device)
        tx = totorch(x).clone()
        x.requires_grad = True
        tx.requires_grad = True
        y = layer(x)
        tw = totorch(layer.weight).clone().requires_grad_(True) if affine else None
        tb = totorch(layer.bias).clone().requires_grad_(True) if affine else None
        ty = F.group_norm(tx, groups, tw, tb, eps=1e-5)
        assert y.shape == tuple(ty.shape)
        assert_close_mag_torch(y, ty, dtype.float32)
        w = random_tensor(y.shape, dtype.float32, device)
        (y * w).sum().backward()
        (ty * totorch(w)).sum().backward()
        assert_close_mag_torch(x.grad, tx.grad, dtype.float32)
        if affine:
            assert set(dict(layer.named_parameters()).keys()) == {'weight', 'bias'}
            assert_close_mag_torch(layer.weight.grad, tw.grad, dtype.float32, rtol=1e-4, atol=1e-3)
            assert_close_mag_torch(layer.bias.grad, tb.grad, dtype.float32, rtol=1e-4, atol=1e-3)
        else:
            assert layer.weight is None and layer.bias is None
            assert dict(layer.named_parameters()) == {}


def test_group_norm_invalid_args() -> None:
    with pytest.raises(ValueError):
        nn.GroupNorm(4, 6)
    with pytest.raises(ValueError):
        nn.GroupNorm(0, 6)
    with pytest.raises(ValueError):
        nn.GroupNorm(2, 6)(Tensor.uniform(2, 4, 3))


def _leaf_like(x: Tensor) -> torch.Tensor:
    return totorch(x).clone().requires_grad_(True)


def _weighted_backward(y: Tensor, ty: torch.Tensor, device: str) -> None:
    assert y.shape == tuple(ty.shape)
    assert_close_mag_torch(y, ty.detach(), dtype.float32)
    w = random_tensor(y.shape, dtype.float32, device)
    (y * w).sum().backward()
    (ty * totorch(w)).sum().backward()


def _assert_grad(x: Tensor, tx: torch.Tensor) -> None:
    assert x.grad is not None
    assert x.grad.shape == tuple(tx.grad.shape)
    assert_close_mag_torch(x.grad, tx.grad, dtype.float32)


def _randomize(layer: nn.Module) -> None:
    with mag.no_grad():
        for p in layer.parameters():
            p.copy_(uniform_tensor(p.shape, -1.0, 1.0))


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('in_shape', [(5, 8), (2, 3, 8), (8,), (1, 8)])
def test_linear(device: str, bias: bool, in_shape) -> None:
    with mag.device(device):
        layer = nn.Linear(8, 6, bias=bias)
        _randomize(layer)
        x = random_tensor(in_shape, dtype.float32, device)
        tx = _leaf_like(x)
        x.requires_grad = True
        tw = _leaf_like(layer.weight)
        tb = _leaf_like(layer.bias) if bias else None
        y = layer(x)
        ty = F.linear(tx, tw, tb)
        if len(in_shape) == 1:
            assert_close_mag_torch(y, ty.detach(), dtype.float32)
            return
        _weighted_backward(y, ty, device)
        _assert_grad(x, tx)
        _assert_grad(layer.weight, tw)
        if bias:
            _assert_grad(layer.bias, tb)
        else:
            assert layer.bias is None


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('in_shape', [(4, 16), (2, 3, 16), (16,)])
def test_layer_norm(device: str, bias: bool, in_shape) -> None:
    with mag.device(device):
        layer = nn.LayerNorm(16, bias=bias, eps=1e-5)
        _randomize(layer)
        x = random_tensor(in_shape, dtype.float32, device)
        tx = _leaf_like(x)
        x.requires_grad = True
        tw = _leaf_like(layer.weight)
        tb = _leaf_like(layer.bias) if bias else None
        y = layer(x)
        ty = F.layer_norm(tx, (16,), tw, tb, eps=1e-5)
        _weighted_backward(y, ty, device)
        _assert_grad(x, tx)
        _assert_grad(layer.weight, tw)
        if bias:
            _assert_grad(layer.bias, tb)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('in_shape', [(4, 16), (2, 3, 16), (16,)])
def test_rms_norm(device: str, in_shape) -> None:
    with mag.device(device):
        layer = nn.RMSNorm(16, eps=1e-5)
        _randomize(layer)
        x = random_tensor(in_shape, dtype.float32, device)
        tx = _leaf_like(x)
        x.requires_grad = True
        tw = _leaf_like(layer.weight)
        y = layer(x)
        ty = F.rms_norm(tx, (16,), tw, eps=1e-5)
        _weighted_backward(y, ty, device)
        _assert_grad(x, tx)
        _assert_grad(layer.weight, tw)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('idx_shape', [(3,), (2, 4), (2, 2, 2)])
def test_embedding_layer(device: str, idx_shape) -> None:
    with mag.device(device):
        layer = nn.Embedding(9, 5)
        _randomize(layer)
        tidx = torch.randint(0, 9, idx_shape)
        idx = Tensor(tidx.tolist(), device=device)
        tw = _leaf_like(layer.weight)
        y = layer(idx)
        ty = F.embedding(tidx, tw)
        _weighted_backward(y, ty, device)
        _assert_grad(layer.weight, tw)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('shape', [(4,), (3, 5), (2, 3, 4)])
def test_mse_loss(device: str, shape) -> None:
    y_hat = random_tensor(shape, dtype.float32, device)
    y = random_tensor(shape, dtype.float32, device)
    ty_hat, ty = _leaf_like(y_hat), totorch(y)
    y_hat.requires_grad = True
    loss = nn.MSELoss()(y_hat, y)
    ref = F.mse_loss(ty_hat, ty)
    assert loss.shape == ()
    assert_close_mag_torch(loss, ref.detach(), dtype.float32)
    loss.backward()
    ref.backward()
    _assert_grad(y_hat, ty_hat)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('n, c', [(8, 10), (1, 4), (32, 3)])
def test_cross_entropy_hard_targets(device: str, n: int, c: int) -> None:
    logits = uniform_tensor((n, c), low=-4.0, high=4.0, device=device)
    tlogits = _leaf_like(logits)
    logits.requires_grad = True
    ttargets = torch.randint(0, c, (n,))
    targets = Tensor(ttargets.tolist(), device=device).one_hot(c).cast(dtype.float32)
    loss = nn.CrossEntropyLoss()(logits, targets)
    ref = F.cross_entropy(tlogits, ttargets)
    assert_close_mag_torch(loss, ref.detach(), dtype.float32)
    loss.backward()
    ref.backward()
    _assert_grad(logits, tlogits)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_cross_entropy_soft_targets(device: str) -> None:
    logits = uniform_tensor((6, 7), low=-4.0, high=4.0, device=device)
    tlogits = _leaf_like(logits)
    logits.requires_grad = True
    targets = uniform_tensor((6, 7), low=0.0, high=1.0, device=device).softmax()
    ttargets = totorch(targets)
    loss = nn.CrossEntropyLoss()(logits, targets)
    ref = F.cross_entropy(tlogits, ttargets)
    assert_close_mag_torch(loss, ref.detach(), dtype.float32)
    loss.backward()
    ref.backward()
    _assert_grad(logits, tlogits)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_cross_entropy_batched_rank3(device: str) -> None:
    logits = uniform_tensor((3, 4, 5), low=-4.0, high=4.0, device=device)
    tlogits = _leaf_like(logits)
    logits.requires_grad = True
    ttargets = torch.randint(0, 5, (3, 4))
    targets = Tensor(ttargets.tolist(), device=device).one_hot(5).cast(dtype.float32)
    loss = nn.CrossEntropyLoss()(logits, targets)
    ref = F.cross_entropy(tlogits.reshape(12, 5), ttargets.reshape(12))
    assert_close_mag_torch(loss, ref.detach(), dtype.float32)
    loss.backward()
    ref.backward()
    _assert_grad(logits, tlogits)


_ACTIVATIONS: tuple[tuple[str, Callable[[], nn.Module], Callable[[torch.Tensor], torch.Tensor]], ...] = (
    ('Softmax', lambda: nn.Softmax(), lambda t: torch.softmax(t, -1)),
    ('Sigmoid', lambda: nn.Sigmoid(), torch.sigmoid),
    ('HardSigmoid', lambda: nn.HardSigmoid(), F.hardsigmoid),
    ('SiLU', lambda: nn.SiLU(), F.silu),
    ('Tanh', lambda: nn.Tanh(), torch.tanh),
    ('ReLU', lambda: nn.ReLU(), torch.relu),
    ('GeLU', lambda: nn.GeLU(), F.gelu),
    ('Identity', lambda: nn.Identity(), lambda t: t),
)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('name, make, ref', _ACTIVATIONS, ids=[a[0] for a in _ACTIVATIONS])
@pytest.mark.parametrize('shape', [(7,), (2, 3, 5), (4, 33)])
def test_activation_modules(device: str, name: str, make: Callable, ref: Callable, shape) -> None:
    with mag.device(device):
        x = uniform_tensor(shape, low=-5.0, high=5.0, device=device)
        tx = _leaf_like(x)
        x.requires_grad = True
        y = make()(x)
        ty = ref(tx)
        _weighted_backward(y, ty, device)
        _assert_grad(x, tx)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_flatten_unflatten_pad_modules(device: str) -> None:
    with mag.device(device):
        x = random_tensor((2, 3, 4, 5), dtype.float32, device)
        tx = _leaf_like(x)
        x.requires_grad = True
        for start, end in ((1, -1), (0, 1), (2, 3), (1, 2)):
            assert_close_mag_torch(nn.Flatten(start, end)(x), torch.flatten(tx, start, end).detach(), dtype.float32)
        for dim, sizes in ((1, (3, 1)), (-1, (5, 1)), (2, (2, 2)), (0, (1, 2))):
            assert_close_mag_torch(nn.Unflatten(dim, sizes)(x), tx.unflatten(dim, sizes).detach(), dtype.float32)
        for pad, mode, value in (((1, 2), 'constant', 0.0), ((1, 1, 2, 0), 'constant', -2.0), ((1, 2, 2, 1), 'reflect', 0.0), ((2, 2, 1, 1), 'replicate', 0.0)):
            assert_close_mag_torch(nn.Pad(list(pad), mode, value)(x), F.pad(tx, list(pad), mode, value).detach(), dtype.float32)
        y = nn.Unflatten(1, (3, 4))(nn.Flatten(1, 2)(x))
        ty = torch.flatten(tx, 1, 2).unflatten(1, (3, 4))
        _weighted_backward(y, ty, device)
        _assert_grad(x, tx)


def _mlp_pair(device: str) -> tuple[nn.Module, tnn.Module]:
    with mag.device(device):
        model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4), nn.Sigmoid())
        _randomize(model)
    ref = tnn.Sequential(tnn.Linear(8, 16), tnn.Tanh(), tnn.Linear(16, 4), tnn.Sigmoid())
    with torch.no_grad():
        for (name, p), (rname, rp) in zip(model.named_parameters(), ref.named_parameters()):
            assert name == rname
            rp.copy_(totorch(p))
    return model, ref


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_sequential_mlp_forward_backward(device: str) -> None:
    model, ref = _mlp_pair(device)
    x = random_tensor((5, 8), dtype.float32, device)
    tx = _leaf_like(x)
    x.requires_grad = True
    y = model(x)
    ty = ref(tx)
    _weighted_backward(y, ty, device)
    _assert_grad(x, tx)
    for (name, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
        _assert_grad(p, rp)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_state_dict_roundtrip_preserves_outputs(device: str) -> None:
    model, ref = _mlp_pair(device)
    x = random_tensor((5, 8), dtype.float32, device)
    with mag.device(device):
        other = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4), nn.Sigmoid())
        _randomize(other)
    assert not torch.equal(totorch(other(x)), totorch(model(x)))
    other.load_state_dict(model.state_dict())
    assert torch.equal(totorch(other(x)), totorch(model(x)))
    assert_close_mag_torch(other(x), ref(totorch(x)).detach(), dtype.float32)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
def test_load_torch_state_dict_values(device: str) -> None:
    ref = tnn.Sequential(tnn.Linear(8, 16), tnn.Tanh(), tnn.Linear(16, 4), tnn.Sigmoid())
    with mag.device(device):
        model = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4), nn.Sigmoid())
    model.load_state_dict({k: Tensor(v.detach().numpy(), device=device) for k, v in ref.state_dict().items()})
    for (name, p), (rname, rp) in zip(model.named_parameters(), ref.named_parameters()):
        assert name == rname
        assert torch.equal(totorch(p), rp.detach())
    x = random_tensor((6, 8), dtype.float32, device)
    assert_close_mag_torch(model(x), ref(totorch(x)).detach(), dtype.float32)
