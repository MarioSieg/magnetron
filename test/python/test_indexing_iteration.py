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


def _mask_pair(*shape: int) -> tuple[Tensor, torch.Tensor]:
    ref = torch.arange(math.prod(shape)) % 3 == 0
    return Tensor(ref.tolist()).reshape(*shape), ref.reshape(*shape)


@pytest.mark.parametrize('shape', [(5,), (2, 3), (2, 3, 4)])
def test_boolean_mask_indexing_matches_torch(shape: tuple[int, ...]) -> None:
    got, ref = _pair(*shape)
    got_m, ref_m = _mask_pair(*shape)
    _assert_same(got[got_m], ref[ref_m])


def test_boolean_mask_on_leading_axis_matches_torch() -> None:
    got, ref = _pair(4, 3)
    got_m, ref_m = _mask_pair(4)
    _assert_same(got[got_m], ref[ref_m])
    _assert_same(got[got_m, 1:], ref[ref_m, 1:])


def test_boolean_mask_on_inner_axis_matches_torch() -> None:
    got, ref = _pair(2, 3, 4)
    got_m, ref_m = _mask_pair(3)
    _assert_same(got[:, got_m], ref[:, ref_m])
    _assert_same(got[1, got_m], ref[1, ref_m])
    got_m2, ref_m2 = _mask_pair(4)
    _assert_same(got[..., got_m2], ref[..., ref_m2])
    _assert_same(got[:, got_m, ::2], ref[:, ref_m, ::2])


def test_boolean_mask_consumes_multiple_axes_like_torch() -> None:
    got, ref = _pair(2, 3, 4)
    got_m, ref_m = _mask_pair(3, 4)
    _assert_same(got[:, got_m], ref[:, ref_m])
    _assert_same(got[1, got_m], ref[1, ref_m])
    got_m0, ref_m0 = _mask_pair(2, 3)
    _assert_same(got[got_m0], ref[ref_m0])


def test_boolean_mask_from_comparison_matches_torch() -> None:
    got, ref = _pair(3, 4)
    _assert_same(got[got > 5.0], ref[ref > 5.0])
    got1, ref1 = _pair(10)
    _assert_same(got1[(got1 % 2.0) == 0.0], ref1[(ref1 % 2.0) == 0.0])


def test_all_false_boolean_mask_yields_empty_like_torch() -> None:
    got, ref = _pair(3, 4)
    _assert_same(got[got > 100.0], ref[ref > 100.0])
    got_m = Tensor([False, False, False])
    assert tuple(got[got_m].shape) == tuple(ref[torch.tensor([False, False, False])].shape)


def test_boolean_mask_shape_mismatch_raises_like_torch() -> None:
    got, ref = _pair(2, 3, 4)
    got_m, ref_m = _mask_pair(3, 4)
    with pytest.raises(IndexError):
        got[got_m]
    with pytest.raises(IndexError):
        ref[ref_m]


def test_gradient_flows_through_boolean_mask_like_torch() -> None:
    got, ref = _pair(2, 3, 4)
    got.requires_grad = True
    ref.requires_grad_(True)
    got_m, ref_m = _mask_pair(3, 4)
    got[:, got_m].sum().backward()
    ref[:, ref_m].sum().backward()
    _assert_same(got.grad, ref.grad)


def test_boolean_mask_with_ellipsis_and_none_matches_torch() -> None:
    got, ref = _pair(2, 3, 4)
    got_m, ref_m = _mask_pair(3, 4)
    _assert_same(got[..., got_m], ref[..., ref_m])
    got_m0, ref_m0 = _mask_pair(2, 3)
    _assert_same(got[got_m0, ...], ref[ref_m0, ...])
    _assert_same(got[got_m0, 1:], ref[ref_m0, 1:])
    _assert_same(got[None, got_m0], ref[None, ref_m0])
    _assert_same(got[got_m0, None], ref[ref_m0, None])
    with pytest.raises(IndexError):
        got[:, got_m, 0]


def test_0d_boolean_mask_matches_torch() -> None:
    got, ref = _pair(2, 3, 4)
    _assert_same(got[Tensor(True)], ref[torch.tensor(True)])
    _assert_same(got[Tensor(True), 0], ref[torch.tensor(True), 0])
    assert tuple(got[Tensor(False)].shape) == tuple(ref[torch.tensor(False)].shape)
