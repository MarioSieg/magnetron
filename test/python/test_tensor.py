# (c) 2026 Mario Sieg. <mario.sieg.64@gmail.com>

from .common import *


def test_tensor_creation() -> None:
    tensor = Tensor.empty(1, 2, 3, 4, 5, 6)
    assert tensor.shape == (1, 2, 3, 4, 5, 6)
    assert tensor.numel == (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.numbytes == 4 * (1 * 2 * 3 * 4 * 5 * 6)
    assert tensor.data_ptr != 0
    assert tensor.is_contiguous is True
    assert tensor.dtype == dtype.float32


_NP_DTYPES = tuple(sorted(NUMPY_DTYPE_MAP, key=lambda d: d.name))


@pytest.mark.parametrize('dt', _NP_DTYPES, ids=[d.name for d in _NP_DTYPES])
def test_tensor_numpy_roundtrip(dt: dtype.DType) -> None:
    np_dt = tonumpy_dtype(dt)
    for shape in BASE_TEST_SHAPES:
        if dt == dtype.boolean:
            a = np.asarray(np.random.uniform(0, 1, size=shape) > 0.5)
        elif dt.is_integer():
            a = np.random.randint(0 if dt.is_unsigned_integer() else -100, 100, size=shape).astype(np_dt)
        else:
            a = np.random.uniform(-100, 100, size=shape).astype(np_dt)
        t = Tensor(a)
        assert t.dtype == dt
        assert t.shape == shape
        assert t.numel == a.size
        back = t.numpy()
        assert back.dtype == np_dt
        np.testing.assert_array_equal(back, a)
        assert t.tolist() == a.tolist()
        torch.testing.assert_close(totorch(t), torch.from_numpy(a.copy()), rtol=0, atol=0)


def test_numbytes_is_the_extent_not_the_storage() -> None:
    # numbytes is numel*itemsize, so it shrinks with a view; storage_numbytes is the buffer the
    # view shares with its base and does not. Conflating the two makes a view's transfer, copy or
    # bounds check reach past the end of the tensor.
    base = Tensor.empty(64, dtype=dtype.uint8)
    assert base.numbytes == 64
    assert base.storage_numbytes == 64

    v = base.view_slice(0, 32, 16, 1)
    assert v.numbytes == 16
    assert v.storage_numbytes == 64
    assert v.data_ptr == base.data_ptr + 32

    f32 = Tensor.empty(8, dtype=dtype.float32)
    assert f32.numbytes == 32
    assert f32.view(2, 4).numbytes == 32, 'a reshape spans the same bytes'
