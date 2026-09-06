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

## MAG_CPU_INTRAOP_MIN_ELEMS

Read by `mag_envcfg_cpu_intraop_min_elems()`.

Overrides, for every operator, the element count at which the CPU backend starts spreading an
operation across the threadpool. Normally each operator has its own threshold in the table in
[`magnetron/cpu/mag_cpu_autotune.c`](../magnetron/cpu/mag_cpu_autotune.c); this variable replaces
all of them with one value.

This exists for tuning, not for production use. The thresholds are machine-specific: fan-out and
barrier cost roughly a fixed amount regardless of tensor size, so the crossover depends on how
fast the host executes the kernel and how much memory bandwidth a single core can already
saturate. Setting a very large value forces everything single-threaded; setting `0` forces
everything through the threadpool.

Allowed values:
- a non-negative integer element count

To re-derive the table on a new machine, sweep this with the tuning tool, which runs each
operator both ways at a range of sizes and prints a suggested table:

```bash
python benchmark/python/tune_intraop.py
```

Example:
```bash
export MAG_CPU_INTRAOP_MIN_ELEMS=0   # force multithreading on, to compare against the table
```
