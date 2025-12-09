# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch.nn.functional

from ..common import *

def cast(
    dtype: DataType,
    mag_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor],
    torch_callback: Callable[[Tensor | torch.Tensor], Tensor | torch.Tensor]
) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dtype=dtype)
        r = mag_callback(x)
        torch.testing.assert_close(totorch(r), torch_callback(totorch(x)), equal_nan=True)

    for_all_shapes(test)

@pytest.mark.parametrize('src_dtype', ALL_DTYPES)
@pytest.mark.parametrize('dst_dtype', ALL_DTYPES)
def test_unary_operators(src_dtype: DataType, dst_dtype: DataType) -> None:
    cast(src_dtype, lambda x: x.cast(dst_dtype), lambda x: x.to(totorch_dtype(dst_dtype)))
