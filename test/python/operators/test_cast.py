# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import torch.nn.functional

from ..common import *


@pytest.mark.parametrize('src_dtype', dtype.all)
@pytest.mark.parametrize('dst_dtype', dtype.all)
def test_cast_op(src_dtype: dtype.DType, dst_dtype: dtype.DType) -> None:
    def test(shape: tuple[int, ...]) -> None:
        x = random_tensor(shape, dt=src_dtype)
        r = x.cast(dst_dtype)
        tdst = totorch_dtype(dst_dtype)
        if src_dtype.is_floating_point() and dst_dtype.is_integer():
            info = torch.iinfo(tdst)
            # Magnetron saturates on all casts, torch has UB (:
            expected = totorch(x).to(torch.float64).nan_to_num(0.0, posinf=info.max, neginf=info.min).clamp(info.min, info.max).to(tdst)
        else:
            expected = totorch(x).to(tdst)
        assert_close_mag_torch(r, expected, src_dtype, equal_nan=True)

    for_all_shapes(test)
