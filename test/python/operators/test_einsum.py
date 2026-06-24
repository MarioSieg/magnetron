# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations
from dataclasses import dataclass

from ..common import *


@dataclass(frozen=True)
class EinsumCase:
    equation: str
    shapes: tuple[tuple[int, ...], ...]


_EINSUM_CASES: tuple[EinsumCase, ...] = (
    EinsumCase('i->i', ((5,),)),
    EinsumCase('i->', ((5,),)),
    EinsumCase('ij->ji', ((2, 4),)),
    EinsumCase('ij->i', ((2, 4),)),
    EinsumCase('ij->j', ((2, 4),)),
    EinsumCase('ij,j->i', ((2, 4), (4,))),
    EinsumCase('i,j->ij', ((3,), (5,))),
    EinsumCase('ij,jk->ik', ((2, 4), (4, 3))),
    EinsumCase('ab,bc->ac', ((2, 4), (4, 3))),
    EinsumCase('abc,cd,de,bef->af', ((2, 3, 4), (4, 5), (5, 6), (3, 6, 7))),
    EinsumCase('abc,cde,ef->abdf', ((2, 3, 4), (4, 5, 6), (6, 7))),
    EinsumCase('abcd,ad->bc', ((2, 3, 4, 5), (2, 5))),
    EinsumCase('bij,bjk->bik', ((2, 3, 4), (2, 4, 5))),
    EinsumCase('bij,jk->bik', ((2, 3, 4), (4, 5))),
    EinsumCase('...ij->...ji', ((2, 3, 4, 5),)),
    EinsumCase('...ij,jk->...ik', ((2, 3, 4), (4, 5))),
    EinsumCase('ij,jk', ((2, 4), (4, 3))),
    EinsumCase('...ij->...', ((2, 3, 4, 5),)),
)

# we use all float types except float8_e4m3fn as torch doesnt support bmm for it on the CPU
_TYPES: set[dtype.DType] = dtype.floating - {dtype.float8_e4m3fn}


def _make_small_positive_tensor(shape: tuple[int, ...], dt: dtype.DType, device: str) -> Tensor:
    return Tensor.uniform(*shape, dtype=dt, device=device) * Tensor.full(shape, fill_value=0.25, dtype=dt, device=device)


def _assert_einsum_close(
    equation: str,
    mag_args: list[Tensor],
    torch_args: list[torch.Tensor],
    dt: dtype.DType,
) -> None:
    r = Tensor.einsum(equation, *mag_args)
    ref = torch.einsum(equation, *torch_args)
    got = totorch(r)
    if dt == dtype.float8_e4m3fn:
        torch.testing.assert_close(
            got.float(),
            ref.float(),
            rtol=0.25,
            atol=0.25,
            equal_nan=True,
        )
    elif dt in (dtype.float16, dtype.bfloat16):
        torch.testing.assert_close(
            got.float(),
            ref.float(),
            rtol=2e-2,
            atol=2e-2,
            equal_nan=True,
        )
    else:
        torch.testing.assert_close(
            got,
            ref,
            rtol=2e-2,
            atol=2e-2,
            equal_nan=True,
        )


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _TYPES)
@pytest.mark.parametrize('case', _EINSUM_CASES)
def test_einsum_cases(device: str, dt: dtype.DType, case: EinsumCase) -> None:
    mag_args = [_make_small_positive_tensor(shape, dt, device) for shape in case.shapes]
    torch_args = [totorch(x) for x in mag_args]
    _assert_einsum_close(case.equation, mag_args, torch_args, dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _TYPES)
def test_einsum_many_random_reduction_shapes(device: str, dt: dtype.DType) -> None:
    cases = [
        EinsumCase('abc,cd,de,bef->af', ((1, 2, 3), (3, 2), (2, 4), (2, 4, 3))),
        EinsumCase('abc,cd,de,bef->af', ((2, 1, 4), (4, 3), (3, 2), (1, 2, 5))),
        EinsumCase('abc,cde,ef->abdf', ((1, 2, 3), (3, 4, 2), (2, 5))),
        EinsumCase('abcd,ce,df->abef', ((2, 1, 3, 4), (3, 2), (4, 5))),
        EinsumCase('ab,bc,cd,de->ae', ((2, 3), (3, 4), (4, 2), (2, 5))),
    ]

    for case in cases:
        mag_args = [_make_small_positive_tensor(shape, dt, device) for shape in case.shapes]
        torch_args = [totorch(x) for x in mag_args]
        _assert_einsum_close(case.equation, mag_args, torch_args, dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _TYPES)
def test_einsum_repeated_labels(device: str, dt: dtype.DType) -> None:
    cases = (
        EinsumCase('ii->i', ((4, 4),)),
        EinsumCase('ii->', ((4, 4),)),
        EinsumCase('ijj->i', ((2, 3, 3),)),
        EinsumCase('ijij->ij', ((2, 3, 2, 3),)),
    )

    for case in cases:
        mag_args = [_make_small_positive_tensor(shape, dt, device) for shape in case.shapes]
        torch_args = [totorch(x) for x in mag_args]
        _assert_einsum_close(case.equation, mag_args, torch_args, dt)


@pytest.mark.parametrize('device', AVAILABLE_DEVICES)
@pytest.mark.parametrize('dt', _TYPES)
def test_einsum_invalid_cases(device: str, dt: dtype.DType) -> None:
    x = _make_small_positive_tensor((2, 3), dt, device)
    y = _make_small_positive_tensor((4, 5), dt, device)

    with pytest.raises(Exception):
        Tensor.einsum('ij->ii', x)

    with pytest.raises(Exception):
        Tensor.einsum('ij,jk->ik', x, y)

    with pytest.raises(Exception):
        Tensor.einsum('ij->ik', x)
