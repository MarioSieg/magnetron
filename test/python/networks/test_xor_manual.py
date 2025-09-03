# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

import numpy as np
from magnetron import Tensor, Context, active_context

np.random.seed(932002)
active_context().seed(932002)

LR: float = 0.1
EPOCHS: int = 10000
INPUT: list[list[float]] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
TARGET: list[list[float]] = [[0.0], [1.0], [1.0], [0.0]]
HIDDEN_DIM: int = 4


def xor_nn_np() -> list[float]:
    def sigmoid(x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x: np.array) -> np.array:
        return x * (1 - x)

    x = np.array(INPUT, dtype=np.float32)
    y = np.array(TARGET, dtype=np.float32)

    w1 = np.random.randn(2, HIDDEN_DIM).astype(np.float32)
    b1 = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
    w2 = np.random.randn(HIDDEN_DIM, 1).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)

    for epoch in range(EPOCHS):
        z1 = np.matmul(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.matmul(a1, w2) + b2
        a2 = sigmoid(z2)

        loss = np.mean((y - a2) ** 2)

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, loss: {loss}')

        d_a2 = -(y - a2)
        d_z2 = d_a2 * sigmoid_derivative(a2)
        d_w2 = np.matmul(a1.T, d_z2)
        d_b2 = np.sum(d_z2)

        d_a1 = np.matmul(d_z2, w2.T)
        d_z1 = d_a1 * sigmoid_derivative(a1)
        d_w1 = np.matmul(x.T, d_z1)
        d_b1 = np.sum(d_z1)

        w2 -= LR * d_w2
        b2 -= LR * d_b2
        w1 -= LR * d_w1
        b1 -= LR * d_b1

    def predict(x: np.array) -> np.array:
        z1 = np.matmul(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.matmul(a1, w2) + b2
        a2 = sigmoid(z2)
        return a2

    return [float(predict(xr)[0][0]) for xr in x]


def xor_nn_mag() -> list[Tensor]:
    x = Tensor.of(INPUT)
    y = Tensor.of(TARGET)

    def sigmoid_derivative(x: Tensor) -> Tensor:
        return x * (1 - x)

    w1 = Tensor.uniform(2, HIDDEN_DIM)
    b1 = Tensor.zeros(1, HIDDEN_DIM)
    w2 = Tensor.uniform(HIDDEN_DIM, 1)
    b2 = Tensor.zeros(1, 1)

    for epoch in range(EPOCHS):
        a1 = (x @ w1 + b1).sigmoid()
        a2 = (a1 @ w2 + b2).sigmoid()

        loss = (y - a2).sqr_().mean()[0]

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, loss: {loss}')

        d_a2 = -(y - a2)
        d_z2 = d_a2 * sigmoid_derivative(a2)
        d_w2 = a1.T.clone() @ d_z2
        d_b2 = d_z2.sum()
        d_z1 = (d_z2 @ w2.T.clone()) * sigmoid_derivative(a1)
        d_w1 = x.T @ d_z1
        d_b1 = d_z1.sum()

        w2 -= LR * d_w2
        b2 -= LR * d_b2
        w1 -= LR * d_w1
        b1 -= LR * d_b1

    def predict(x: Tensor) -> Tensor:
        z1 = x @ w1 + b1
        a1 = z1.sigmoid()
        z2 = a1 @ w2 + b2
        a2 = z2.sigmoid()
        return a2

    return [predict(Tensor.of([xr]))[0].item() for xr in INPUT]


def test_xor_nn() -> None:
    np_out = xor_nn_np()
    mag_out = xor_nn_mag()
    assert [round(x) for x in np_out] == [0, 1, 1, 0]
    assert [round(x) for x in mag_out] == [0, 1, 1, 0]
    assert np.allclose(np_out, mag_out, atol=0.1)