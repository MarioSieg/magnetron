# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from collections.abc import Iterator

import pytest
from magnetron import context, no_grad


def test_no_grad_function() -> None:
    @no_grad()
    def f() -> bool:
        return context.is_grad_recording()

    assert context.is_grad_recording()
    assert not f()
    assert context.is_grad_recording()


def test_no_grad_generator_body_runs_without_grad() -> None:
    @no_grad()
    def gen() -> Iterator[bool]:
        for _ in range(3):
            yield context.is_grad_recording()

    assert context.is_grad_recording()
    assert list(gen()) == [False, False, False]
    assert context.is_grad_recording()


def test_no_grad_generator_restores_state_between_items() -> None:
    @no_grad()
    def gen() -> Iterator[int]:
        yield 1
        yield 2

    seen = []
    for _ in gen():
        seen.append(context.is_grad_recording())  # consumer code between items keeps its own state
    assert seen == [True, True]


def test_no_grad_generator_send_return_and_yield_from() -> None:
    @no_grad()
    def echo() -> Iterator[bool]:
        got = yield context.is_grad_recording()
        assert got == 'ping'
        yield context.is_grad_recording()
        return 'done'

    g = echo()
    assert next(g) is False
    assert g.send('ping') is False

    def outer() -> Iterator[bool]:
        result = yield from echo_two()
        assert result == 'done'

    def echo_two() -> Iterator[bool]:
        r = yield from echo_wrapper()
        return r

    @no_grad()
    def echo_wrapper() -> Iterator[bool]:
        yield context.is_grad_recording()
        return 'done'

    assert list(outer()) == [False]


def test_no_grad_generator_throw_and_close() -> None:
    log = []

    @no_grad()
    def gen() -> Iterator[int]:
        try:
            yield 1
        except ValueError:
            log.append(('caught', context.is_grad_recording()))
            yield 2
        finally:
            log.append(('finally', context.is_grad_recording()))

    g = gen()
    assert next(g) == 1
    assert g.throw(ValueError()) == 2
    g.close()
    assert log == [('caught', False), ('finally', False)]
    assert context.is_grad_recording()

    g = gen()
    next(g)
    with pytest.raises(KeyError):
        g.throw(KeyError())
    assert context.is_grad_recording()


def test_no_grad_generator_inside_no_grad_block() -> None:
    @no_grad()
    def gen() -> Iterator[bool]:
        yield context.is_grad_recording()

    with no_grad():
        assert list(gen()) == [False]
        assert not context.is_grad_recording()  # outer no_grad must survive the generator's restore
    assert context.is_grad_recording()
