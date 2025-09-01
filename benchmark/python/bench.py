# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

from bench_tool import (
    benchmark,
    BenchParticipant,
    generate_matmul_shapes,
    generate_elementwise_shapes,
    generate_square_shapes,
)

import magnetron as mag
import numpy as np
import torch


class NumpyBenchmark(BenchParticipant):
    def __init__(self) -> None:
        super().__init__('Numpy')

    def allocate_args(self, shape_a: tuple[int, int], shape_b: tuple[int, int]) -> None:
        x = np.full(shape_a, fill_value=1.0, dtype=np.float32)
        y = np.full(shape_b, fill_value=2.0, dtype=np.float32)
        return x, y


class PyTorchBenchmark(BenchParticipant):
    def __init__(self) -> None:
        super().__init__('PyTorch')
        self.device = torch.device('cpu')

    def allocate_args(self, shape_a: tuple[int, int], shape_b: tuple[int, int]) -> None:
        x = torch.full(shape_a, fill_value=1.0, dtype=torch.float32).to(self.device)
        y = torch.full(shape_b, fill_value=2.0, dtype=torch.float32).to(self.device)
        return x, y


class MagnetronBenchmark(BenchParticipant):
    def __init__(self) -> None:
        super().__init__('Magnetron')

    def allocate_args(self, shape_a: tuple[int, int], shape_b: tuple[int, int]) -> None:
        x = mag.Tensor.full(shape_a, fill_value=1.0, dtype=mag.float32)
        y = mag.Tensor.full(shape_b, fill_value=2.0, dtype=mag.float32)
        return x, y


participants = [
    MagnetronBenchmark(),
    # NumpyBenchmark(),
    PyTorchBenchmark(),
]

elementwise_ops = [
    ('Addition', lambda x, y: x + y),
    ('Subtraction', lambda x, y: x - y),
    ('Hadamard Product', lambda x, y: x * y),
    ('Division', lambda x, y: x / y),
]

matmul_ops = [
    ('Matrix Multiplication', lambda x, y: x @ y),
]

print('Running performance benchmark...')
print('Magnetron VS')
for participant in participants:
    if not isinstance(participant, MagnetronBenchmark):
        print(f'    {participant.name}')


def bench_square_bin_ops(dim_lim: int = 2048, step: int = 32) -> None:
    square_shapes = generate_square_shapes(dim_lim, step)
    for op in elementwise_ops:
        name, fn = op
        print(f'Benchmarking {name} Operator')
        benchmark(name, participants, fn, square_shapes, plot_style='lines')


def bench_square_matmul(dim_lim: int = 2048, step: int = 32) -> None:
    square_shapes = generate_square_shapes(dim_lim, step)
    for op in matmul_ops:
        name, fn = op
        print(f'Benchmarking {name} Operator')
        benchmark(name, participants, fn, square_shapes, plot_style='lines')


def bench_permuted_bin_ops(dim_lim: int = 2048, step: int = 32) -> None:
    elementwise_shapes = generate_elementwise_shapes(dim_lim, step)
    for op in elementwise_ops:
        name, fn = op
        print(f'Benchmarking {name} Operator')
        benchmark(name, participants, fn, elementwise_shapes, plot_style='bars')


def bench_permuted_matmul(dim_lim: int = 2048, step: int = 32) -> None:
    matmul_shapes = generate_matmul_shapes(dim_lim, step)
    for op in matmul_ops:
        name, fn = op
        print(f'Benchmarking {name} Operator')
        benchmark(name, participants, fn, matmul_shapes, plot_style='bars')
