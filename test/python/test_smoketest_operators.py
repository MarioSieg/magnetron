# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>
# This file tests each operator once against a manual, correct result

from magnetron import dtype, Tensor

def test_empty() -> None:
    x = Tensor.empty((2, 3), dtype=dtype.int64)
    assert x.shape == (2,3)
    assert x.dtype == dtype.int64

def test_as_strided() -> None:
    x = Tensor.arange(9).reshape(3, 3)
    t = Tensor.as_strided(x, (2, 2), (1, 2))
    assert t.shape == (2, 2)
    assert t.tolist() == [
        [0, 2],
        [1, 3],
    ]

def test_broadcast_to() -> None:
    x = Tensor([1, 2, 3])
    y = x.broadcast_to((3, 3))
    assert y.tolist() == [[1, 2, 3],
                          [1, 2, 3],
                          [1, 2, 3]]
def test_expand() -> None:
    x = Tensor([[1], [2], [3]])
    assert x.shape == (3, 1)
    y = x.expand((3, 4))
    assert y.tolist() == [[ 1,  1,  1,  1],
                          [ 2,  2,  2,  2],
                          [ 3,  3,  3,  3]]
    y = x.expand(-1, 4)
    assert y.tolist() == [[ 1,  1,  1,  1],
                          [ 2,  2,  2,  2],
                          [ 3,  3,  3,  3]]

def test_empty_like() -> None:
    x = Tensor.empty(2, 4, dtype=dtype.float8_e4m3fn)
    y = Tensor.empty_like(x)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() == y.tolist()

def test_scalar() -> None:
    x = Tensor.scalar(2.5)
    assert x.rank == 0
    assert x.shape == ()
    assert x.strides == ()
    assert x.numel == 1
    assert x.item() == 2.5

def test_full() -> None:
    x = Tensor.full(2, 3, fill_value=3.141592)
    assert x.shape == (2, 3)
    assert x.numel == 2*3
    assert x.rank == 2
    assert (x == 3.141592).all()

def test_full_like() -> None:
    x = Tensor.full(2, 4, fill_value=3.141592)
    y = Tensor.full_like(x, -3.141592)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() != y.tolist()
    assert (y == -3.141592).all()

def test_zeros() -> None:
    x = Tensor.zeros(2, 3)
    assert x.shape == (2, 3)
    assert x.numel == 2*3
    assert x.rank == 2
    assert (x == 0).all()

def test_zeros_like() -> None:
    x = Tensor.uniform(2, 4, fill_value=3.141592)
    y = Tensor.zeros_like(x)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() != y.tolist()
    assert (y == 0).all()

def test_ones() -> None:
    x = Tensor.ones(2, 3)
    assert x.shape == (2, 3)
    assert x.numel == 2*3
    assert x.rank == 2
    assert (x == 1).all()

def test_ones_like() -> None:
    x = Tensor.uniform(2, 4)
    y = Tensor.ones_like(x)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() != y.tolist()
    assert (y == 1).all()

def test_uniform() -> None:
    x = Tensor.uniform(2, 4, low=-10.0, hi=13.0)
    assert x.shape == (2, 4)
    assert x.numel == 2*4
    assert x.rank == 2
    assert (x <= 13.0).all() and (x >= -10.0).all()

def test_uniform_like() -> None:
    x = Tensor.uniform(2, 4, low=-10.0, hi=13.0)
    y = Tensor.uniform_like(x, low=-10.0, hi=13.0)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert (x <= 13.0).all() and (x >= -10.0).all()
    assert (y <= 13.0).all() and (y >= -10.0).all()

def test_normal() -> None:
    x = Tensor.normal(2, 4, mean=0.3, std=0.6)
    assert x.shape == (2, 4)
    assert x.numel == 2*4
    assert x.rank == 2

def test_normal_like() -> None:
    x = Tensor.normal(2, 4, mean=0.3, std=0.6)
    y = Tensor.normal_like(x, mean=0.3, std=0.6)
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel

def test_bernoulli() -> None:
    x = Tensor.bernoulli(2, 4, p=0.5)
    assert x.dtype == dtype.boolean
    assert x.shape == (2, 4)
    assert x.numel == 2*4
    assert x.rank == 2

def test_bernoulli_like() -> None:
    x = Tensor.bernoulli(2, 4, p=0.5)
    y = Tensor.bernoulli_like(x, p=0.5)
    assert x.dtype == dtype.boolean
    assert y.dtype == dtype.boolean
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel

def test_arange() -> None:
    x = Tensor.arange(5)
    assert x.dtype == dtype.int64
    assert x.tolist() == [ 0,  1,  2,  3,  4]
    x = Tensor.arange(1, 4)
    assert x.tolist() == [ 1,  2,  3]
    x = Tensor.arange(1.0, 2.5, 0.5)
    assert x.tolist() == [ 1.0000,  1.5000,  2.0000]

def test_linspace() -> None:
    x = Tensor.linspace(3, 10, steps=5)
    assert x.tolist() == [  3.0000,   4.7500,   6.5000,   8.2500,  10.0000]
    x = Tensor.linspace(-10, 10, steps=5)
    assert x.tolist() == [-10.,  -5.,   0.,   5.,  10.]
    x = Tensor.linspace(start=-10, end=10, steps=5)
    assert x.tolist() == [-10.,  -5.,   0.,   5.,  10.]
    x = Tensor.linspace(start=-10, end=10, steps=1)
    assert x.tolist() == [-10]

def test_meshgrid() -> None:
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    gx, gy = Tensor.meshgrid(x, y, indexing='ij')
    assert gx.tolist() == [[1, 1, 1],
                           [2, 2, 2],
                           [3, 3, 3]]
    assert gy.tolist() == [[4, 5, 6],
                           [4, 5, 6],
                           [4, 5, 6]]

def test_one_hot() -> None:
    x = (Tensor.arange(0, 5) % 3).one_hot()
    assert x.dtype == dtype.int64
    assert x.tolist() == [[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]]
    x = (Tensor.arange(0, 5) % 3).one_hot(num_classes=5)
    assert x.tolist() == [[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0]]
    x = (Tensor.arange(0, 6).view(3, 2) % 3).one_hot()
    assert x.tolist() == [[[1, 0, 0],
                           [0, 1, 0]],
                          [[0, 0, 1],
                           [1, 0, 0]],
                          [[0, 1, 0],
                           [0, 0, 1]]]

def test_rand_perm() -> None:
    x = Tensor.rand_perm(4)
    assert x.dtype == dtype.int64
    for i in range(x.numel):
        assert i in x.tolist()

def test_inplace_copy() -> None:
    x = Tensor.uniform(2, 4, 6)
    y = Tensor.uniform_like(x)
    x.copy_(y)
    assert (x == y).all()

def test_inplace_zeros() -> None:
    x = Tensor.uniform(2, 4, 6)
    x.zeros_()
    assert (x == 0).all()

def test_inplace_ones() -> None:
    x = Tensor.uniform(2, 4, 6)
    x.ones_()
    assert (x == 1).all()

def test_inplace_fill() -> None:
    x = Tensor.uniform(2, 4, 6)
    x.fill_(3.1415)
    assert (x == 3.1415).all()

def test_inplace_masked_fill() -> None:
    x = Tensor.uniform(2, 4, 6)
    mask = Tensor.ones_like(x).tril().cast(dtype.boolean)
    x.masked_fill_(mask, 3.1415)
    assert (x == 3.1415).any()

def test_inplace_uniform() -> None:
    x = Tensor.zeros(2, 4, 6)
    x.uniform_(low=-1.0, high=1.0)
    assert (x != 0).all()
    assert (x >= -1.0).all() and (x <= 1.0).all()

def test_inplace_normal() -> None:
    x = Tensor.zeros(2, 4, 6)
    x.normal_(mean=0.5, std=1.0)
    assert (x != 0).all()

def test_inplace_bernoulli() -> None:
    x = Tensor.ones(2, 4, 6).cast(dtype.boolean)
    x.bernoulli_(p=0.5)
    assert ((x ^ x) == 0).all()

def test_clone() -> None:
    x = Tensor.uniform(2, 4, 8, 3)
    y = x.clone()
    assert x.shape == y.shape
    assert x.rank == y.rank
    assert x.strides == y.strides
    assert x.dtype == y.dtype
    assert x.device == y.device
    assert x.numel == y.numel
    assert x.tolist() == y.tolist()
    assert x.data_storage_ptr != y.data_storage_ptr
    assert x.is_contiguous
    assert y.is_contiguous
    assert not x.is_view
    assert not y.is_view

def test_cast() -> None:
    x = Tensor.uniform(2, 4, 8, 3, low=-10.0, high=10.0, dtype=dtype.float32)
    vals = x.flatten().tolist()
    y = x.cast(dtype.int16)
    assert y.dtype == dtype.int16
    assert y.data_storage_ptr != x.data_storage_ptr
    for i,v in enumerate(y.flatten().tolist()):
        assert v == int(vals[i])

# TODO: transfer

def test_view() -> None:
    x = Tensor.normal(4, 4)
    assert not x.is_view
    assert x.shape == (4, 4)
    y = x.view(16)
    assert y.is_view
    assert y.shape == (16,)
    z = x.view(-1, 8)
    assert z.is_view
    assert z.shape == (2, 8)
    a = Tensor.normal(1, 2, 3, 4)
    assert a.shape == (1, 2, 3, 4)
    b = a.transpose(1, 2)
    assert b.shape == (1, 3, 2, 4)
    c = a.view(1, 2, 3, 4)
    assert a.shape == c.shape
    assert b.tolist() != c.tolist()

# TODO: view_slice

def test_reshape() -> None:
    a = Tensor.arange(4.0)
    b = a.reshape((2, 2))
    assert b.tolist() == [[ 0.,  1.],
                          [ 2.,  3.]]
    b = Tensor([[0, 1], [2, 3]])
    c = b.reshape((-1,))
    assert c.tolist() == [ 0,  1,  2,  3]

def test_transpose() -> None:
    x = Tensor([[ 33, 4, -10],
                [12, 100, -666]])
    assert x.shape == (2, 3)
    assert x.tolist() == [[ 33, 4, -10],
                          [12, 100, -666]]
    y = x.transpose(0, 1)
    assert y.shape == (3, 2)
    assert y.tolist() == [[ 33, 12],
                          [4,  100],
                          [-10,  -666]]

def test_T() -> None:
    x = Tensor.normal(())
    x.shape == ()
    x.numel == 1
    assert x.T.shape == x.shape
    assert x.T.numel == x.numel
    assert x.T == x
    x = Tensor([[ 33, 4, -10],
                [12, 100, -666]])
    assert x.shape == (2, 3)
    assert x.tolist() == [[ 33, 4, -10],
                          [12, 100, -666]]
    y = x.T
    assert y.shape == (3, 2)
    assert y.tolist() == [[ 33, 12],
                          [4,  100],
                          [-10,  -666]]

def test_permute() -> None:
    x = Tensor.normal(2, 3, 5)
    assert x.shape == (2, 3, 5)
    y = x.permute((2, 0, 1))
    assert y.shape == (5, 2, 3)

def test_contiguous() -> None:
    x = Tensor.normal(2, 3, 5)
    assert x.is_contiguous
    y = x.permute((2, 0, 1))
    assert not y.is_contiguous
    assert (y.contiguous() == y).all()
    assert y.contiguous().is_contiguous
    assert (y.contiguous() == y).all()

def test_squeeze() -> None:
    x = Tensor.zeros(2, 1, 2, 1, 2)
    assert x.shape == (2, 1, 2, 1, 2)
    y = x.squeeze()
    assert y.shape == (2, 2, 2)
    y = x.squeeze(0)
    assert y.shape == (2, 1, 2, 1, 2)
    y = x.squeeze(1)
    assert y.shape == (2, 2, 1, 2)
    # TODO: squeeze with tuple

def test_unsqueeze() -> None:
    x = Tensor([1, 2, 3, 4])
    y = x.unsqueeze(0)
    assert y.tolist() == [[ 1,  2,  3,  4]]
    y = x.unsqueeze(1)
    assert y.tolist() == [[ 1],
                          [ 2],
                          [ 3],
                          [ 4]]

def test_flatten() -> None:
    x = Tensor([[[1, 2],
                 [3, 4]],
                [[5, 6],
                 [7, 8]]])
    y = x.flatten()
    assert y.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    y = x.flatten(start_dim=1)
    assert y.tolist() == [[1, 2, 3, 4],
                          [5, 6, 7, 8]]

def test_unflatten() -> None:
    x = Tensor.normal(3, 4, 1)
    y = x.unflatten(1, (2, 2))
    assert y.shape == (3, 2, 2, 1)
    y = x.unflatten(1, (-1, 2))
    assert y.shape == (3, 2, 2, 1)
    x = Tensor.normal(5, 12, 3)
    y = x.unflatten(-2, (2, 2, 3, 1, 1))
    assert y.shape == (5, 2, 2, 3, 1, 1, 3)

def test_narrow() -> None:
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert x.narrow(0, 0, 2).tolist() == [[ 1,  2,  3],
                                          [ 4,  5,  6]]
    assert x.narrow(1, 1, 2).tolist() == [[ 2,  3],
                                          [ 5,  6],
                                          [ 8,  9]]
    assert x.narrow(-1,-1, 1).tolist() == [[3],
                                           [6],
                                           [9]]

def test_movedim() -> None:
    x = Tensor.normal(3, 2, 1)
    assert x.shape == (3, 2, 1)
    y = x.movedim(1, 0)
    assert y.shape == (2, 3, 1)
    # TODO: tuple movedim
    #y = x.movedim((1, 2), (0, 1))
    #assert y.shape == (2, 1, 3)

def test_select() -> None:
    pass # TODO

def test_split() -> None:
    x = Tensor.arange(10).reshape(5, 2)
    assert x.tolist() == [[0, 1],
                          [2, 3],
                          [4, 5],
                          [6, 7],
                          [8, 9]]
    y: tuple[Tensor] = x.split(2)
    should: tuple[Tensor] = (Tensor([[0, 1],
                     [2, 3]]),
             Tensor([[4, 5],
                     [6, 7]]),
             Tensor([[8, 9]]))
    assert len(y) == len(should)
    for i in range(len(y)):
        assert (y[i] == should[i]).all()
    # TODO: tuple split
