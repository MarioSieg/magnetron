# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import numpy as np
import pytest

from magnetron import Tensor, dtype


def _u8(values: list[int]) -> Tensor:
    return Tensor(values, dtype=dtype.uint8)


def _bytes_of(t: Tensor) -> bytes:
    return t.tobytes()


def test_widens_bytes_into_floats() -> None:
    src = np.arange(16, dtype=np.uint8)
    t = _u8(src.tolist())
    f = t.reinterpret_view(dtype.float32, [4])
    assert f.dtype == dtype.float32
    assert tuple(f.shape) == (4,)
    assert f.is_view
    # No conversion happened: these are the same bytes, which is what numpy's own view() does.
    np.testing.assert_array_equal(f.numpy(), src.view(np.float32))


def test_narrows_floats_into_bytes() -> None:
    src = np.array([1.5, -2.25, 3.75, 0.0], dtype=np.float32)
    t = Tensor(src.tolist(), dtype=dtype.float32)
    b = t.reinterpret_view(dtype.uint8, [16])
    assert b.dtype == dtype.uint8
    np.testing.assert_array_equal(b.numpy(), src.view(np.uint8))


def test_round_trips_through_every_size() -> None:
    src = np.arange(64, dtype=np.uint8)
    t = _u8(src.tolist())
    for dt, np_dt in [
        (dtype.uint8, np.uint8),
        (dtype.int8, np.int8),
        (dtype.uint16, np.uint16),
        (dtype.int16, np.int16),
        (dtype.uint32, np.uint32),
        (dtype.int32, np.int32),
        (dtype.uint64, np.uint64),
        (dtype.int64, np.int64),
        (dtype.float32, np.float32),
    ]:
        wide = t.reinterpret_view(dt, [64 // dt.size])
        np.testing.assert_array_equal(wide.numpy(), src.view(np_dt))
        # ... and all the way back, byte for byte.
        assert _bytes_of(wide.reinterpret_view(dtype.uint8, [64])) == _bytes_of(t)


def test_shares_storage_with_its_base() -> None:
    t = _u8(list(range(16)))
    f = t.reinterpret_view(dtype.float32, [4])
    assert f.data_ptr == t.data_ptr
    assert f.is_view


def test_shape_forms_agree() -> None:
    t = _u8(list(range(32)))
    varargs = t.reinterpret_view(dtype.float32, 2, 4)
    sequence = t.reinterpret_view(dtype.float32, [2, 4])
    inferred = t.reinterpret_view(dtype.float32, 2, -1)
    assert tuple(varargs.shape) == tuple(sequence.shape) == tuple(inferred.shape) == (2, 4)


def test_inference_is_against_the_reinterpreted_count() -> None:
    # 48 uint8 elements are 12 float32 ones, so -1 must resolve to 4, not 16.
    t = _u8(list(range(48)))
    f = t.reinterpret_view(dtype.float32, 3, -1)
    assert tuple(f.shape) == (3, 4)
    assert f.numel == 12


def test_without_a_shape_the_last_dim_absorbs_it() -> None:
    # torch's one-argument Tensor.view(dtype).
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype.float32)
    assert tuple(t.reinterpret_view(dtype.uint8).shape) == (2, 8)
    assert tuple(t.reinterpret_view(dtype.uint16).shape) == (2, 4)
    wide = _u8(list(range(16))).view(2, 8).reinterpret_view(dtype.float32)
    assert tuple(wide.shape) == (2, 2)


def test_empty_sequence_gives_a_rank_zero_view() -> None:
    t = _u8(list(range(4)))
    scalar = t.reinterpret_view(dtype.int32, [])
    assert tuple(scalar.shape) == ()
    assert scalar.numel == 1


def test_same_dtype_is_an_ordinary_view() -> None:
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype.float32)
    v = t.reinterpret_view(dtype.float32, [4])
    assert tuple(v.shape) == (4,)
    assert v.data_ptr == t.data_ptr
    np.testing.assert_array_equal(v.numpy(), t.numpy().reshape(4))


def test_tracks_the_offset_of_a_slice() -> None:
    src = np.arange(64, dtype=np.uint8)
    t = _u8(src.tolist())
    sliced = t.view_slice(0, 32, 16, 1)  # a 64-aligned byte range, as a snapshot produces
    f = sliced.reinterpret_view(dtype.float32, [4])
    assert f.data_ptr == t.data_ptr + 32
    np.testing.assert_array_equal(f.numpy(), src[32:48].view(np.float32))


def test_writes_through_to_the_base() -> None:
    t = _u8([0] * 8)
    f = t.reinterpret_view(dtype.float32, [2])
    f.numpy()[:] = np.array([1.0, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(t.numpy(), np.array([1.0, 2.0], dtype=np.float32).view(np.uint8))


def test_rejects_a_byte_count_that_does_not_divide() -> None:
    with pytest.raises(RuntimeError, match='not a whole number'):
        _u8(list(range(6))).reinterpret_view(dtype.float32, [1])


def test_rejects_a_misaligned_offset() -> None:
    # Fine for uint8, but byte 2 is not where a float32 may start.
    sliced = _u8(list(range(64))).view_slice(0, 2, 16, 1)
    with pytest.raises(RuntimeError, match='not a multiple of'):
        sliced.reinterpret_view(dtype.float32, [4])


def test_rejects_a_non_contiguous_base() -> None:
    t = _u8(list(range(64))).view(8, 8).transpose(0, 1)
    assert not t.is_contiguous
    with pytest.raises(RuntimeError, match='must be contiguous'):
        t.reinterpret_view(dtype.float32, [16])


def test_rejects_a_shape_that_does_not_match() -> None:
    with pytest.raises(RuntimeError, match='requested shape has 5 elements'):
        _u8(list(range(16))).reinterpret_view(dtype.float32, [5])


def test_rejects_a_tensor_that_requires_grad() -> None:
    # Reinterpreting bits has no derivative to record.
    t = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype.float32)
    t.requires_grad = True
    with pytest.raises(RuntimeError, match='requires grad'):
        t.reinterpret_view(dtype.int64, [2])
    # detach() is the way out, and it leaves the base alone.
    assert tuple(t.detach().reinterpret_view(dtype.int64, [2]).shape) == (2,)
    assert t.requires_grad
