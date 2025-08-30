# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import atexit
import threading
import time

from dataclasses import dataclass
from typing import Any, Callable, Sequence

@dataclass
class Stat:
    total_ns: int = 0
    max_ns: int = 0
    count: int = 0
    def add(self, dt_ns: int) -> None:
        self.total_ns += dt_ns
        self.count += 1
        if dt_ns > self.max_ns:
            self.max_ns = dt_ns

Key = tuple[
    str,
    tuple[tuple[int, ...], ...],
    tuple[str | None, ...],
]

class OpProfiler:
    def __init__(self, enabled: bool = True, top_k: int | None = None) -> None:
        self.enabled = enabled
        self.top_k = top_k
        self._stats: dict[Key, Stat] = {}
        self._lock = threading.Lock()
        atexit.register(self._report)

    def wrap(self, op: str, operands: Sequence[Any], fn: Callable[[], Any]) -> Any:
        if not self.enabled:
            return fn()
        key = self._make_key(op, operands)
        t0 = time.perf_counter_ns()
        try:
            return fn()
        finally:
            dt = time.perf_counter_ns() - t0
            with self._lock:
                self._stats.setdefault(key, Stat()).add(dt)

    def unary(self, op: str, x: Any, fn: Callable[[], Any]) -> Any:
        return self.wrap(op, (x,), fn)

    def binary(self, op: str, x: Any, y: Any, fn: Callable[[], Any]) -> Any:
        return self.wrap(op, (x, y), fn)

    @staticmethod
    def _shape_of(t: Any) -> tuple[int, ...]:
        shp = getattr(t, "shape", None)
        if shp is None:
            return ()
        return tuple(int(d) for d in shp)

    @staticmethod
    def _dtype_of(t: Any) -> str | None:
        dt = getattr(t, "dtype", None)
        return str(dt) if dt is not None else None

    def _make_key(self, op: str, operands: Sequence[Any]) -> Key:
        shapes: tuple[tuple[int, ...], ...] = tuple(self._shape_of(o) for o in operands)
        dtypes: tuple[str | None, ...] = tuple(self._dtype_of(o) for o in operands)
        return op, shapes, dtypes

    def _report(self) -> None:
        if not self.enabled or not self._stats:
            return
        rows: list[tuple[int, int, int, int, Key]] = []
        with self._lock:
            for k, s in self._stats.items():
                avg = s.total_ns // max(1, s.count)
                rows.append((avg, s.max_ns, s.total_ns, s.count, k))
        rows.sort(key=lambda r: r[0], reverse=True)
        if self.top_k is not None:
            rows = rows[: self.top_k]

        print("\n=== Op Profiler (slowest average first) ===")
        print(f"{'Avg ms':>10}  {'Max ms':>10}  {'Total ms':>10}  {'Count':>7}  Op / Shapes / Dtypes")
        for avg_ns, max_ns, tot_ns, cnt, key in rows:
            op, shapes, dtypes = key
            avg_ms = avg_ns / 1e6
            max_ms = max_ns / 1e6
            tot_ms = tot_ns / 1e6
            print(f"{avg_ms:10.3f}  {max_ms:10.3f}  {tot_ms:10.3f}  {cnt:7d}  {op}  {shapes}  {dtypes}")
        print("===========================================\n")

OP_PROFILER = OpProfiler(enabled=True)
