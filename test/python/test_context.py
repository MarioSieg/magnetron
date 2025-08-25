# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

from magnetron import *

Context.verbose = True


def test_context_creation() -> None:
    # Test that a context can be created and defaults are correct.
    ctx = active_context()
    ctx = active_context()
    ctx = active_context()
    assert isinstance(ctx.os_name, str)
    assert isinstance(ctx.cpu_name, str)
    assert ctx.cpu_virtual_cores >= 1
    assert ctx.cpu_physical_cores >= 1
