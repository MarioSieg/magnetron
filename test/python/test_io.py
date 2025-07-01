# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from magnetron.io import *


def test_open_close_stream() -> None:
    with StorageStream() as stream:
        assert len(stream.tensor_keys()) == 0


def test_write_read_tensor() -> None:
    with StorageStream() as stream:
        tensor = Tensor.of([1.0, 2.0, 3.0])
        stream['test_tensor'] = tensor
        assert 'test_tensor' in stream.tensor_keys()

        read_tensor = stream['test_tensor']
        assert read_tensor.tolist() == [1.0, 2.0, 3.0]


def test_write_read_multiple_tensors() -> None:
    with StorageStream() as stream:
        tensor1 = Tensor.of([1.0, 2.0])
        tensor2 = Tensor.of([3.0, 4.0])
        stream['tensor1'] = tensor1
        stream['tensor2'] = tensor2

        assert 'tensor1' in stream.tensor_keys()
        assert 'tensor2' in stream.tensor_keys()

        read_tensor1 = stream['tensor1']
        read_tensor2 = stream['tensor2']

        assert read_tensor1.tolist() == [1.0, 2.0]
        assert read_tensor2.tolist() == [3.0, 4.0]
