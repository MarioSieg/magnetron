# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Loopback conformance run of every collective over the host backends: four ranks on 127.0.0.1."""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

WORLD_SIZE = 4
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_collective_worker.py')


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run_world(backend: str, size: int = WORLD_SIZE, timeout: float = 180.0) -> None:
    port = free_port()
    procs = [
        subprocess.Popen(
            [sys.executable, WORKER, '--rank', str(r), '--size', str(size), '--backend', backend, '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for r in range(size)
    ]
    outputs: dict[int, tuple[int | None, str]] = {}
    try:
        for r, p in enumerate(procs):
            try:
                out, _ = p.communicate(timeout=timeout)
                outputs[r] = (p.returncode, out)
            except subprocess.TimeoutExpired:
                outputs[r] = (None, 'timed out')
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    failed = {r: v for r, v in outputs.items() if v[0] != 0}
    assert not failed, '\n'.join(f'--- rank {r} (rc={rc}) ---\n{out}' for r, (rc, out) in failed.items())


@pytest.mark.parametrize('backend', ['tcp', 'shm'])
def test_collectives_world4(backend: str) -> None:
    run_world(backend)


@pytest.mark.parametrize('backend', ['tcp', 'shm'])
def test_collectives_world1(backend: str) -> None:
    run_world(backend, size=1)
