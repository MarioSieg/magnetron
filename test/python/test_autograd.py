# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

import random

import torch

import magnetron as mag


def test_autograd_simple() -> None:
    x = mag.Tensor(3.0, requires_grad=True)
    y = mag.Tensor(2.0, requires_grad=True)
    assert x.requires_grad
    assert y.requires_grad
    y = (x + y) * (x - y)
    y.backward()
    magx, magy = x, y

    x = torch.Tensor([3.0])
    x.requires_grad = True
    y = torch.Tensor([2.0])
    y.requires_grad = True
    y = (x + y) * (x - y)
    y.backward()
    torchx, torchy = x, y

    assert magy.item() == torchy.data.item()
    assert magx.grad.item() == torchx.grad.item()


def test_autograd_simple2() -> None:
    x = mag.Tensor(-4.0, requires_grad=True)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    magx, magy = x, y

    x = torch.Tensor([-4.0])
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    torchx, torchy = x, y

    assert magy.item() == torchy.data.item()
    assert magx.grad.item() == torchx.grad.item()


def test_autograd_inherit() -> None:
    xi1 = random.random() * 128.0
    xi2 = random.random() * 512.0
    x = mag.Tensor(xi1, requires_grad=True)
    y = mag.Tensor(xi2, requires_grad=True)
    t1 = x + y
    t2 = x - y
    t3 = t1 * t2
    y = t3.relu()
    assert x.requires_grad
    assert y.requires_grad
    assert t1.requires_grad
    assert t2.requires_grad
    assert t3.requires_grad
    assert y.requires_grad
    y.backward()
    magx, magy = x, y

    x = torch.Tensor([xi1])
    x.requires_grad = True
    y = torch.Tensor([xi2])
    y.requires_grad = True
    t1 = x + y
    t2 = x - y
    t3 = t1 * t2
    y = t3.relu()
    y.backward()
    torchx, torchy = x, y

    assert magy.item() == torchy.data.item()
    assert magx.grad.item() == torchx.grad.item()


def test_autograd_inherit_nograd() -> None:
    xi1 = random.random() * 128.0
    xi2 = random.random() * 512.0
    with mag.no_grad():
        x = mag.Tensor(xi1, requires_grad=True)
        y = mag.Tensor(xi2, requires_grad=True)
        t1 = x + y
        t2 = x - y
        t3 = t1 * t2
        yy = t3.relu()
        assert x.requires_grad  # Overriding the no_grad context
        assert y.requires_grad  # Overriding the no_grad context
        assert not t1.requires_grad
        assert not t2.requires_grad
        assert not t3.requires_grad
        assert not yy.requires_grad
        magy = yy

    with torch.no_grad():
        x = torch.Tensor([xi1])
        x.requires_grad = True
        y = torch.Tensor([xi2])
        y.requires_grad = True
        t1 = x + y
        t2 = x - y
        t3 = t1 * t2
        y = t3.relu()
        torchy = y

    assert magy.item() == torchy.data.item()


def test_detach_clears_requires_grad_and_shares_storage() -> None:
    t = mag.Tensor([1.0, 2.0, 3.0, 4.0], dtype=mag.dtype.float32)
    t.requires_grad = True

    d = t.detach()
    assert not d.requires_grad
    assert d.data_ptr == t.data_ptr
    assert t.requires_grad, 'detach() must not disturb the base'

    d.requires_grad = True
    assert d.requires_grad and t.requires_grad

    assert t.clone().requires_grad, 'clone() keeps the flag, unlike detach()'


def test_reduction_backward_without_keepdim() -> None:
    for shape, dim, op in [((32, 2), -1, 'sum'), ((32, 2), -1, 'mean'), ((4, 3, 2), 1, 'sum'), ((4, 3, 2), (0, 2), 'mean'), ((32, 2), 0, 'sum')]:
        xt = torch.rand(*shape, requires_grad=True)
        wt = torch.rand(*shape)
        x = mag.Tensor(xt.tolist(), requires_grad=True)
        w = mag.Tensor(wt.tolist())
        r = getattr(x * w, op)(dim=dim)
        (r * r).sum().backward()
        rt = getattr(xt * wt, op)(dim=dim)
        (rt * rt).sum().backward()
        assert tuple(x.grad.shape) == tuple(shape)
        torch.testing.assert_close(torch.tensor(x.grad.tolist()), xt.grad, rtol=1e-4, atol=1e-5)


def test_cross_entropy_matches_torch_and_is_stable() -> None:
    logits = torch.tensor([[0.0, 200.0], [-300.0, 5.0], [1.0, -1.0], [0.5, 0.5]], requires_grad=True)
    targets = torch.tensor([1, 0, 1, 0])
    x = mag.Tensor(logits.tolist(), requires_grad=True)
    y = mag.Tensor(targets.tolist(), dtype=mag.dtype.int64).one_hot(2).cast(mag.dtype.float32)
    loss = mag.nn.CrossEntropyLoss()(x, y)
    loss.backward()
    ref = torch.nn.functional.cross_entropy(logits, targets)
    ref.backward()
    torch.testing.assert_close(torch.tensor(loss.item()), ref.detach(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.tensor(x.grad.tolist()), logits.grad, rtol=1e-4, atol=1e-5)


def test_unbind_backward_matches_torch() -> None:
    x = mag.Tensor.uniform(3, 4, requires_grad=True)
    tx = torch.tensor(x.tolist(), dtype=torch.float32, requires_grad=True)
    a, b, c = x.unbind(0)
    ta, tb, tc = tx.unbind(0)
    y = (a * 2.0 + b * 3.0 + c * 5.0).sum()
    ty = (ta * 2.0 + tb * 3.0 + tc * 5.0).sum()
    y.backward()
    ty.backward()
    torch.testing.assert_close(torch.tensor(x.grad.tolist()), tx.grad)
