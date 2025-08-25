# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag


def test_tensor_creation() -> None:
    tensor = mag.Tensor.empty(1, 2, 3, 4, 5, 6)
    assert tensor.shape == (1, 2, 3, 4, 5, 6)
    assert tensor.numel == (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.data_size == 4 * (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.data_ptr != 0
    assert tensor.is_contiguous is True
    assert tensor.dtype == mag.float32
