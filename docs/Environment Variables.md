# Environment Variables

All variables listed here are declared and parsed in [`magnetron/core/mag_envcfg.h`](../magnetron/core/mag_envcfg.h)
and [`magnetron/core/mag_envcfg.c`](../magnetron/core/mag_envcfg.c). Nothing else in the codebase
calls `getenv` - call sites ask for an already parsed value through an accessor declared in that
header. When adding, renaming or changing the accepted values of a variable, update both.

## MAG_LOG_LEVEL

Read by `mag_envcfg_apply_log_level()`.

Controls the global log verbosity (case-insensitive).

Allowed values:
- `off`   – disable all logging
- `error` – errors only (default)
- `warn`  – errors and warnings
- `info`  – errors, warnings, and info logs
- `debug` – all logs, including debug information

Example:
```bash
export MAG_LOG_LEVEL=info
```

## MAG_CPU_SPECIALIZATION_LEVEL

Read by `mag_envcfg_cpu_specialization_level()`.

Pins the CPU specialization level instead of auto-detecting the best one for the host CPU
(case-insensitive). Intended for benchmarking one specialization level against another on the same
machine.

Allowed values:
- a full specialization level name, e.g. `amd64-v3`, `arm64-v86_sve`
- the bare level suffix, e.g. `v3`, `v86_sve`
- `fallback` or `generic` – force the portable, non-specialized kernels

A specialization level the host CPU cannot execute is refused (auto-detection runs instead) rather
than crashing on an illegal instruction. Unknown names are also ignored; the available levels are
then logged at `info` level.

Example:
```bash
export MAG_CPU_SPECIALIZATION_LEVEL=v3
```

## MAG_CPU_ASYNC

Read by `mag_envcfg_cpu_async()`.

Selects how the CPU backend executes operators. In async mode, operators are recorded into a command
queue and return immediately. A dedicated executor thread (worker 0 of the thread pool) runs them in
order, so Python overhead overlaps with compute. The host only waits where it has to: `item()`,
`tolist()`, `numpy()`, printing, transfers, or an explicit `magnetron.sync()` / `Tensor.sync()`.
Errors raised while executing a queued operator are reported by the next sync point (see
`MAG_LAUNCH_BLOCKING`). Small single threaded operators still run inline on the calling thread
while nothing is queued, so `item()` heavy loops don't pay for a thread handoff.
See [Async CPU Backend](Async%20CPU%20Backend.md).

In async mode the default worker count leaves 2 logical CPUs free: one for the submitting (Python)
thread, which no longer computes, and one spare so OS noise doesn't preempt a spinning worker and
stall the fork-join barrier. Setting `MAG_CPU_THREADS` overrides this.

Allowed values:
- `auto` – async when there are more than 2 usable logical CPUs (default)
- `1` / `on` – always async, even with a single worker thread
- `0` / `off` – synchronous: the calling thread executes each operator before returning

Example:
```bash
export MAG_CPU_ASYNC=0
```

## MAG_CPU_QUEUE_DEPTH

Read by `mag_envcfg_cpu_queue_depth()`.

Capacity of the async CPU command queue, in operators. Bounds how far the host can run ahead of the
executor, and with it how many temporaries queued operators keep alive. A full queue blocks the
submitting thread until a slot frees up.

Allowed values:
- an integer in `[1, 1048576]` (default: `1024`)

Example:
```bash
export MAG_CPU_QUEUE_DEPTH=256
```

## MAG_CPU_QUEUE_MB

Read by `mag_envcfg_cpu_queue_mb()`.

Budget for the output bytes of queued, not yet completed async CPU operators, in MiB. Every queued
operator keeps its output alive, so running far ahead of the executor trades cache locality for
overlap. Above the cache size this costs more than it gains (a chain of 4 MiB elementwise ops is
several times slower with an unbounded queue). A submit that would exceed the budget waits for older
operators to finish; a single operator larger than the budget runs alone.

Allowed values:
- an integer in `[1, 1048576]` (default: `16`)

Example:
```bash
export MAG_CPU_QUEUE_MB=64
```

## MAG_LAUNCH_BLOCKING

Read by `mag_envcfg_launch_blocking()`.

Makes every operator wait for its own completion before returning, on all backends (async CPU and
CUDA). Errors raised while executing an operator then surface at the call that submitted it, instead
of at a later sync point. This is meant for debugging and is slow.

Allowed values:
- `1` / `on` – wait after every operator
- `0` / `off` – asynchronous (default)

Example:
```bash
export MAG_LAUNCH_BLOCKING=1
```

## MAGNETRON_LEAK_WARNINGS

Read directly by the Python bindings in [`magnetron/bindings/_module.cpp`](../magnetron/bindings/_module.cpp),
not through `mag_envcfg`; the core library never sees it.

Enables nanobind's exit-time report of binding objects that were never freed. It is off by default
because pytest with numpy or torch loaded keeps parametrized `DType` values alive through interpreter
shutdown, which the report flags as a leak even though the bindings are correct. Set it when hunting
a real reference-counting bug in the bindings from a plain Python script.

Allowed values:
- any non-empty value – enable the report
- unset – disable (default)

Example:
```bash
export MAGNETRON_LEAK_WARNINGS=1
```

