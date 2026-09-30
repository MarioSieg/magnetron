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
