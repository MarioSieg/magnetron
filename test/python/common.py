# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import torch
import magnetron as mag

DTYPE_TORCH_MAP: dict[mag.DataType, torch.dtype] = {
    mag.float16: torch.float16,
    mag.float32: torch.float32,
    mag.int32: torch.int32,
    mag.boolean: torch.bool
}

def totorch(t: mag.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is None:
        dtype = DTYPE_TORCH_MAP[t.dtype]
    return torch.tensor(t.tolist(), dtype=dtype).reshape(t.shape)

def square_shape_permutations(f: callable, lim: int) -> None:
    lim += 1
    for i0 in range(1, lim):
        for i1 in range(1, lim):
            for i2 in range(1, lim):
                for i3 in range(1, lim):
                    for i4 in range(1, lim):
                        for i5 in range(1, lim):
                            f((i0, i1, i2, i3, i4, i5))

def binary_op_square(dtype: mag.DataType, f: callable, lim: int = 4, from_: float | int | None = None, to: float | int | None = None) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def func(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
            y = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
            y = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
        r = f(x, y)
        torch.testing.assert_close(
            totorch(r, torch_dt),
            f(totorch(x, torch_dt), totorch(y, torch_dt))
        )

    square_shape_permutations(func, lim)

def binary_cmp_op(dtype: mag.DataType, f: callable, lim: int = 4, from_: float | int | None = None, to: float | int | None = None) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def func(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
            y = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
            y = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
        r = f(x, y)
        assert r.dtype == mag.boolean
        torch.testing.assert_close(
            totorch(r, torch.bool),
            f(totorch(x, torch_dt), totorch(y, torch_dt))
        )

    square_shape_permutations(func, lim)

def unary_op(
    dtype: mag.DataType,
    magf: callable,
    torchf: callable,
    lim: int = 4,
    from_: float | int | None = None,
    to: float | int | None = None
) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def func(shape: tuple[int, ...]) -> None:
        if dtype == mag.boolean:
            x = mag.Tensor.bernoulli(shape)
        else:
            x = mag.Tensor.uniform(shape, dtype=dtype, from_=from_, to=to)
        r = magf(x.clone())
        torch.testing.assert_close(totorch(r, torch_dt), torchf(totorch(x, torch_dt)))

    square_shape_permutations(func, lim)


def scalar_op(dtype: mag.DataType, f: callable, rhs: bool = True, lim: int = 4) -> None:
    torch_dt = DTYPE_TORCH_MAP[dtype]

    def func(shape: tuple[int, ...]) -> None:  # x op scalar
        xi: float = random.uniform(-1.0, 1.0)
        x = mag.Tensor.uniform(shape, dtype=dtype)
        r = f(x, xi)
        torch.testing.assert_close(totorch(r, torch_dt), f(totorch(x, torch_dt), xi))

    square_shape_permutations(func, lim)

    if not rhs:
        return

    def func(shape: tuple[int, ...]) -> None:  # scalar op x
        xi: float = random.uniform(-1.0, 1.0)
        x = mag.Tensor.uniform(shape)
        r = f(xi, x)
        torch.testing.assert_close(totorch(r, torch_dt), f(xi, totorch(x, torch_dt)))

    square_shape_permutations(func, lim)
