from .common import *

def _rand_int(dim_size: int) -> int:
    """Safe version – never crashes when dim_size == 0."""
    if dim_size == 0:
        return 0                    # will be out-of-bounds and caught later
    return random.randrange(-dim_size, dim_size)

def _rand_slice(dim_size: int) -> slice:
    """Generate a random Python slice that is always valid."""
    if dim_size == 0:                       # any slice is fine on a 0-length dim
        return slice(None)

    step_choices = [1, 2, -1]
    step = random.choice(step_choices) if random.random() < 0.5 else None

    if step is None or step > 0:            # forward slice
        start = random.randrange(-dim_size, dim_size)
        stop  = random.randrange(start + 1, dim_size + 1)
    else:                                   # backward slice
        # make sure there is room for stop < start
        start = random.randrange(-dim_size, dim_size)
        stop  = random.randrange(-1, start) if start > -1 else None

    if random.random() < 0.3:
        start = None
    if random.random() < 0.3:
        stop = None

    return slice(start, stop, step)

ITER_PER_SHAPE: int = 20

@pytest.mark.parametrize("dtype", [float16, float32, int32, boolean])
def test_tensor_getitem_basic(dtype: DataType) -> None:
    def func(shape: tuple[int, ...]) -> None:
        if len(shape) == 0:
            return
        if dtype == boolean:
            x = Tensor.bernoulli(shape)
        else:
            x = Tensor.uniform(shape, dtype=dtype)
        tx = totorch(x)
        for _ in range(ITER_PER_SHAPE):
            idx_components = []
            for dim_sz in shape:
                comp = random.random()
                if comp < 0.35:
                    idx_components.append(_rand_int(dim_sz))
                elif comp < 0.75:
                    idx_components.append(_rand_slice(dim_sz))
                else:
                    idx_components.append(None)
            if random.random() < 0.3:
                pos = random.randrange(0, len(idx_components) + 1)
                idx_components.insert(pos, Ellipsis)
            index = tuple(idx_components)
            if len(index) == 0:
                index = (slice(None),)

            try:
                y_m = x[index]
            except NotImplementedError:
                continue
            except IndexError:
                with pytest.raises(IndexError):
                    _ = tx[index]
                continue
            try:
                y_t = tx[index]
                if y_t.dim() == 0:
                    y_t = y_t.unsqueeze(0) # TODO: Because we don't support 0-dim tensors yet
            except IndexError:
                pytest.fail(f"torch raised IndexError for {index} but custom did not")
            torch.testing.assert_close(totorch(y_m), y_t, rtol=1e-3, atol=1e-3)

    square_shape_permutations(func, lim=4)