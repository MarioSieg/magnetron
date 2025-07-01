# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag


def test_tensor_creation() -> None:
    tensor = mag.empty(1, 2, 3, 4, 5, 6)
    assert tensor.shape == (1, 2, 3, 4, 5, 6)
    assert tensor.numel == (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.data_size == 4 * (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.data_ptr != 0
    assert tensor.is_contiguous is True
    assert tensor.dtype == magfloat32


def test_tensor_scalar_get_set_physical() -> None:
    tensor = mag.empty(4, 4)
    tensor[0, 0] = 128
    assert tensor[0, 0] == 128
    tensor[3, 3] = 3.14
    assert abs(tensor[3, 3] - 3.14) < 1e-6


def test_tensor_scalar_get_set_virtual() -> None:
    tensor = mag.empty(4, 4)
    tensor[0] = 128
    assert tensor[0] == 128
    tensor[15] = 3.14
    assert abs(tensor[15] - 3.14) < 1e-6
