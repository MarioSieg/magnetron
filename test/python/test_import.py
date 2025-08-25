# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>


def test_import_magnetron() -> None:
    import magnetron

    assert magnetron.__version__ is not None


def test_simple_exec() -> None:
    import magnetron as mag

    a = mag.Tensor.of([1.0, 4.0, 1.0])
    assert a.max()[0] == 4
