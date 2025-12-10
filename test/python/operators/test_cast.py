# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch.nn.functional

from ..common import *

@pytest.mark.parametrize('src_dtype', ALL_DTYPES)
@pytest.mark.parametrize('dst_dtype', ALL_DTYPES)
def test_cast_op(src_dtype: DataType, dst_dtype: DataType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dtype=src_dtype)
        r = x.cast(dst_dtype)
        torch.testing.assert_close(totorch(r), totorch(x).to(totorch_dtype(dst_dtype)), equal_nan=True)

    for_all_shapes(test)
