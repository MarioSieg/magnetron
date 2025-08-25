# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>
import torch

from ..common import *

@pytest.mark.parametrize('dtype', [float16, float32])
def test_matmul_squared(dtype: DataType) -> None:
    binary_op_square(dtype, lambda x, y: x + y, kind=BinaryOpParamKind.TENSOR)

@pytest.mark.parametrize('dtype', [float16, float32])
def test_matmul_full(dtype: DataType) -> None:
    for A, B in matmul_shape_pairs(lim=3, max_total_rank=MAX_DIMS):
        a = Tensor.uniform(A, dtype=dtype)
        b = Tensor.uniform(B, dtype=dtype)
        r = a @ b
        rt = torch.matmul(totorch(a), totorch(b))
        assert r.rank == rt.dim(), f"Expected rank {rt.dim()}, got {r.rank}"
        assert r.shape == rt.shape, f"Expected shape {rt.shape}, got {r.shape}"
        assert torch.allclose(totorch(r), rt, atol=1e-4, rtol=1e-4)

def test_matmul_simple_mlp() -> None:
    truth_table = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

    W1 = Tensor.uniform((2, 4))
    b1 = Tensor.uniform((1, 4))
    W2 = Tensor.uniform((4, 1))
    b2 = Tensor.uniform((1, 1))

    nW1 = totorch(W1)
    nb1 = totorch(b1)
    nW2 = totorch(W2)
    nb2 = totorch(b2)

    def sigmoid(x: torch.Tensor) -> None:
        return 1 / (1 + torch.exp(-x))

    np_data = []
    for x in truth_table:
        z1 = torch.tensor(x) @ nW1 + nb1
        a1 = sigmoid(z1)
        z2 = a1 @ nW2 + nb2
        a2 = sigmoid(z2)
        np_data.append(a2)

    mag_data = []
    for x in truth_table:
        x = Tensor.of([x])

        z1 = x @ W1 + b1
        a1 = z1.sigmoid()
        z2 = a1 @ W2 + b2
        a2 = z2.sigmoid()
        mag_data.append(a2)

    for mag, np_ in zip(mag_data, np_data):
        torch.testing.assert_close(totorch(mag), np_, atol=1e-4, rtol=1e-4)


def test_matmul_squared() -> None:
    shapes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for shape in shapes:
        mag_a = Tensor.uniform((shape, shape))
        mag_b = Tensor.uniform((shape, shape))
        np_a = totorch(mag_a)
        np_b = totorch(mag_b)
        mag_result = mag_a @ mag_b
        np_result = torch.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        torch.testing.assert_close(totorch(mag_result), np_result, atol=1e-4, rtol=1e-4)


def test_matmul() -> None:
    shapes = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]
    for shape in shapes:
        mag_a = Tensor.uniform(shape)
        mag_b = Tensor.uniform((shape[1], shape[0]))
        np_a = totorch(mag_a)
        np_b = totorch(mag_b)
        mag_result = mag_a @ mag_b
        np_result = torch.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        torch.testing.assert_close(totorch(mag_result), np_result, atol=1e-4, rtol=1e-4)


def test_matmul_matrix_by_vector() -> None:
    shapes = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]
    for shape in shapes:
        mag_a = Tensor.uniform(shape)
        mag_b = Tensor.uniform((shape[1], 1))
        np_a = totorch(mag_a)
        np_b = totorch(mag_b)
        mag_result = mag_a @ mag_b
        np_result = torch.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        torch.testing.assert_close(totorch(mag_result), np_result, atol=1e-4, rtol=1e-4)


def test_matmul_vector_by_matrix() -> None:
    shapes = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]
    for shape in shapes:
        mag_a = Tensor.uniform((1, shape[0]))
        mag_b = Tensor.uniform(shape)
        np_a = totorch(mag_a)
        np_b = totorch(mag_b)
        mag_result = mag_a @ mag_b
        np_result = torch.matmul(np_a, np_b)
        assert mag_result.shape == np_result.shape
        torch.testing.assert_close(totorch(mag_result), np_result, atol=1e-4, rtol=1e-4)


def test_matmul_scalar_by_matrix() -> None:
    shapes = [
        (4, 8),
        (8, 16),
        (16, 32),
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]
    for shape in shapes:
        scalar = random.random() * 10.0
        mag_b = Tensor.uniform(shape)
        np_b = totorch(mag_b)
        mag_result = scalar * mag_b
        np_result = scalar * np_b
        assert mag_result.shape == np_result.shape
        torch.testing.assert_close(totorch(mag_result), np_result, atol=1e-4, rtol=1e-4)


def test_matmul_x_transposed() -> None:
    shape_a = (4, 2)
    shape_b = (4, 4)
    mag_a = Tensor.uniform(shape_a)
    mag_b = Tensor.uniform(shape_b)
    np_a = totorch(mag_a)
    np_b = totorch(mag_b)
    mag_result = mag_a.T @ mag_b
    np_result = torch.matmul(np_a.T, np_b)
    assert mag_result.shape == np_result.shape
    assert mag_result.shape == (2, 4)
    torch.testing.assert_close(totorch(mag_result), np_result, atol=1e-4, rtol=1e-4)
