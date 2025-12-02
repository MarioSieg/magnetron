# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

from magnetron import context


def test_context_creation() -> None:
    # Test that a context can be created and defaults are correct.
    ctx = context.native_ptr()
    ctx = context.native_ptr()
    ctx = context.native_ptr()
    ctx = context.native_ptr()
    assert isinstance(context.os_name(), str)
    assert isinstance(context.cpu_name(), str)
    assert context.cpu_virtual_cores() >= 1
    assert context.cpu_physical_cores() >= 1
