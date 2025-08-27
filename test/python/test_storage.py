# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

from magnetron import *
import magnetron.io as io

def test_storage_write_tensor() -> None:
    a = Tensor.uniform(4, 4)
    with io.StorageArchive('meow.mag', 'w') as f:
        f.put_tensor("a", a)
