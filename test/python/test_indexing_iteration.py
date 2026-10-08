# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

import math

import pytest
import torch

from magnetron import Tensor, dtype

from .common import totorch


def _pair(*shape: int) -> tuple[Tensor, torch.Tensor]:
    ref = torch.arange(float(math.prod(shape))).reshape(*shape) if shape else torch.tensor(3.0)
    return Tensor(ref.tolist()), ref


def _assert_same(got: Tensor, ref: torch.Tensor) -> None:
    assert tuple(got.shape) == tuple(ref.shape)
    assert got.rank == ref.dim()
    torch.testing.assert_close(totorch(got), ref)


@pytest.mark.parametrize('shape', [(4,), (2, 3), (2, 3, 4), (3, 1, 2)])
def test_iteration_matches_torch(shape: tuple[int, ...]) -> None:
    mag_t, ref_t = _pair(*shape)
    mag_rows = list(mag_t)
    ref_rows = list(ref_t)
    assert len(mag_rows) == len(ref_rows) == shape[0]
    for got, ref in zip(mag_rows, ref_rows, strict=True):
        _assert_same(got, ref)


@pytest.mark.parametrize('shape', [(4,), (2, 3), (2, 3, 4)])
def test_len_matches_torch(shape: tuple[int, ...]) -> None:
    mag_t, ref_t = _pair(*shape)
    assert len(mag_t) == len(ref_t)


def test_iteration_over_1d_yields_scalars() -> None:
    mag_t, ref_t = _pair(3)
    for got, ref in zip(mag_t, ref_t, strict=True):
        assert got.rank == 0 == ref.dim()
        assert got.item() == ref.item()
        assert float(got) == float(ref)


def test_iteration_over_0d_raises_type_error_like_torch() -> None:
    mag_t, ref_t = _pair()
    with pytest.raises(TypeError):
        iter(ref_t)
    with pytest.raises(TypeError):
        iter(mag_t)
    with pytest.raises(TypeError):
        len(ref_t)
    with pytest.raises(TypeError):
        len(mag_t)


def test_unpacking_matches_torch() -> None:
    mag_t, ref_t = _pair(2, 3)
    a, b = mag_t
    ra, rb = ref_t
    _assert_same(a, ra)
    _assert_same(b, rb)
    with pytest.raises(ValueError):
        a, b, c = mag_t
    with pytest.raises(ValueError):
        a, b, c = ref_t


def test_enumerate_and_zip_over_rows() -> None:
    mag_t, ref_t = _pair(3, 2)
    for i, (got, ref) in enumerate(zip(mag_t, ref_t, strict=True)):
        _assert_same(got, ref_t[i])
        _assert_same(got, ref)


def test_iterated_rows_are_views_sharing_storage() -> None:
    mag_t, ref_t = _pair(2, 3)
    for got, ref in zip(mag_t, ref_t, strict=True):
        assert got.is_view
        assert got.data_storage_ptr == mag_t.data_storage_ptr
        got.fill_(7.0)
        ref.fill_(7.0)
    _assert_same(mag_t, ref_t)


@pytest.mark.parametrize(
    'index',
    [
        0,
        -1,
        (1, 2),
        (0, -1, 1),
        (slice(None), 1),
        (1, slice(None)),
        (slice(1, None), 0),
        (0, slice(None, None, 2)),
        (None, 0),
        (0, None),
        (Ellipsis, 1),
        (1, Ellipsis),
        (Ellipsis, 0, 0),
        (1, 1, slice(1, 3)),
    ],
)
def test_integer_indexing_shapes_match_torch(index) -> None:
    mag_t, ref_t = _pair(2, 3, 4)
    _assert_same(mag_t[index], ref_t[index])


def test_full_integer_index_is_scalar() -> None:
    mag_t, ref_t = _pair(2, 3)
    got = mag_t[1, 2]
    ref = ref_t[1, 2]
    _assert_same(got, ref)
    assert got.rank == 0
    assert got.item() == ref.item()


def test_chained_indexing_matches_torch() -> None:
    mag_t, ref_t = _pair(2, 3, 4)
    _assert_same(mag_t[1][2], ref_t[1][2])
    _assert_same(mag_t[1][2][3], ref_t[1][2][3])
    assert mag_t[1][2][3].item() == ref_t[1][2][3].item()


def test_out_of_bounds_index_raises_like_torch() -> None:
    mag_t, ref_t = _pair(2, 3)
    with pytest.raises(IndexError):
        ref_t[2]
    with pytest.raises(IndexError):
        mag_t[2]
    with pytest.raises(IndexError):
        ref_t[-3]
    with pytest.raises(IndexError):
        mag_t[-3]


def test_setitem_with_integer_index_matches_torch() -> None:
    mag_t, ref_t = _pair(2, 3)
    mag_t[0, 1] = 9.0
    ref_t[0, 1] = 9.0
    mag_t[1] = 5.0
    ref_t[1] = 5.0
    _assert_same(mag_t, ref_t)
    row_mag, row_ref = _pair(3)
    mag_t[0] = row_mag
    ref_t[0] = row_ref
    _assert_same(mag_t, ref_t)


def test_gradient_flows_through_iteration_like_torch() -> None:
    ref_x = torch.arange(6.0).reshape(2, 3).requires_grad_(True)
    mag_x = Tensor(ref_x.tolist(), requires_grad=True)
    ref_loss = sum((i + 1) * (row * row).sum() for i, row in enumerate(ref_x))
    mag_loss = sum((i + 1) * (row * row).sum() for i, row in enumerate(mag_x))
    ref_loss.backward()
    mag_loss.backward()
    assert mag_loss.item() == pytest.approx(ref_loss.item())
    _assert_same(mag_x.grad, ref_x.grad)


def test_gradient_flows_through_scalar_indexing_like_torch() -> None:
    ref_x = torch.arange(4.0).requires_grad_(True)
    mag_x = Tensor(ref_x.tolist(), requires_grad=True)
    (ref_x[1] * ref_x[3]).backward()
    (mag_x[1] * mag_x[3]).backward()
    _assert_same(mag_x.grad, ref_x.grad)


def test_list_conversion_of_rows_matches_tolist() -> None:
    mag_t, ref_t = _pair(3, 2)
    assert [row.tolist() for row in mag_t] == ref_t.tolist()
    assert [x.item() for x in mag_t[0]] == ref_t[0].tolist()


def test_index_with_0d_int64_tensor_matches_torch() -> None:
    got, ref = _pair(5, 3)
    got_i = Tensor([2]).cast(dtype.int64).reshape(1, 1)[0, 0]
    ref_i = torch.tensor([[2]])[0, 0]
    assert got_i.rank == 0 == ref_i.dim()
    _assert_same(got[got_i], ref[ref_i])
    _assert_same(got[got_i, 1], ref[ref_i, 1])
    got1, ref1 = _pair(5)
    _assert_same(got1[got_i], ref1[ref_i])
    assert got1[got_i].item() == ref1[ref_i].item()


def test_embedding_with_0d_index_matches_torch() -> None:
    got, ref = _pair(5, 3)
    got_i = Tensor([4]).cast(dtype.int64)[0]
    assert got_i.rank == 0
    _assert_same(got.embedding(got_i), torch.nn.functional.embedding(torch.tensor(4), ref))
