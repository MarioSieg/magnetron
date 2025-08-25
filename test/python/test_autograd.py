# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

import random

import magnetron as mag
import torch


def test_autograd_1() -> None:
    x = mag.Tensor.of([3.0], requires_grad=True)
    y = mag.Tensor.of([2.0], requires_grad=True)
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


def test_autograd_2() -> None:
    x = mag.Tensor.of([-4.0], requires_grad=True)
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
    x = mag.Tensor.of([xi1], requires_grad=True)
    y = mag.Tensor.of([xi2], requires_grad=True)
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
        x = mag.Tensor.of([xi1], requires_grad=True)
        y = mag.Tensor.of([xi2], requires_grad=True)
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


def test_autograd_broadcast() -> None:
    x = torch.full(size=(3, 3, 3, 3), fill_value=3.0, requires_grad=True)
    y = torch.full(size=(3, 3), fill_value=2.0, requires_grad=True)
    k = torch.full(size=(1,), fill_value=10.0, requires_grad=True)
    z = (x + y) * (x - y) / k
    z.sum().backward()
    print('\nx grad:' + str(x.grad[0][0][0][0]))
    print('\ny grad:' + str(y.grad[0][0]))
