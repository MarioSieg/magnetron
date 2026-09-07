# Environment Variables

All variables listed here are declared and parsed in [`magnetron/core/mag_envcfg.h`](../magnetron/core/mag_envcfg.h)
and [`magnetron/core/mag_envcfg.c`](../magnetron/core/mag_envcfg.c). Nothing else in the codebase
calls `getenv` - call sites ask for a value through an accessor declared in that header, either a
parsed one or `mag_envcfg_raw` for the few variables whose value is a bare string. When adding,
renaming or changing the accepted values of a variable, update both, and this file with them.

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

## MAG_JIT

Read by `mag_envcfg_jit_enabled()`.

Enables the pointwise fusion JIT. On by default.

Inside a `mag.fuse()` region the runtime records a chain of elementwise operations instead of
executing them one at a time, generates C for the whole chain, and builds it into a shared object
with the host compiler the first time that chain is seen. Results are identical either way: the
generated code is built with `-ffp-contract=off` and rounds at the same points the eager kernels
do, so fusing a chain does not change its arithmetic.

Turn it off to keep the runtime from invoking a compiler at all, which matters on a machine
without a toolchain, in a sandbox where `system()` is unavailable, and when bisecting a numerical
difference to determine whether a fused kernel is responsible.

Allowed values:
- `on`, `1`, `true`  – compile fused kernels (default)
- `off`, `0`, `false` – record and run the operations eagerly instead

Example:
```bash
export MAG_JIT=off
```

## MAG_JIT_CC

Read directly through `mag_envcfg_raw()` in [`magnetron/core/mag_fusion.c`](../magnetron/core/mag_fusion.c).

The host compiler used to build fused kernels. Defaults to `cc` on POSIX and `cl` on Windows.

The value is passed to `system()` as the start of a command line, so it may carry flags
(`MAG_JIT_CC='clang -march=native'`). Everything after it is fixed:
`-O3 -ffp-contract=off -fPIC -shared`. Those are POSIX compiler-driver flags, so on Windows the
default does not in fact work and the JIT falls back to eager execution; set this to a
clang-compatible driver there.

When the compiler cannot be run the JIT reports it once and every chain runs eagerly, so an
absent or misconfigured compiler costs performance, not correctness.

Example:
```bash
export MAG_JIT_CC=clang
```

## MAG_JIT_CACHE_DIR

Read directly through `mag_envcfg_raw()` in [`magnetron/core/mag_fusion.c`](../magnetron/core/mag_fusion.c).

Where generated kernels are written and looked up. Defaults to `/tmp` on POSIX and the working
directory on Windows.

A kernel is named after the hash of its generated source, so an object left behind by an earlier
run is reused and a repeated session pays no compile at all. Point this at a persistent directory
to keep that cache across reboots, or at a scratch directory to isolate one process from another.

Example:
```bash
export MAG_JIT_CACHE_DIR=~/.cache/magnetron
```

## MAG_JIT_POISON

Read by `mag_envcfg_jit_poison()`.

Fills every value a fused chain declined to write with a NaN pattern. Off by default, and always on
in debug builds.

A fused chain skips writing an intermediate when nothing can read it again. Deciding that requires
knowing whether an operator's backward reads an operand's data or only its shape, which is declared
per operator in [`magnetron/core/mag_op_grads.c`](../magnetron/core/mag_op_grads.c). A wrong
declaration there does not crash: the backward reads whatever the buffer happened to contain and
produces gradients that are slightly off, which is the hardest kind of mistake to notice.

With this on, the same mistake yields NaN on the first run. Turn it on when adding an operator to
the table, or when a gradient looks subtly wrong under `mag.fuse()`.

Allowed values:
- `on`, `1`, `true`  – poison elided values
- `off`, `0`, `false` – leave them (default)

Example:
```bash
MAG_JIT_POISON=on pytest test/python/test_fusion_gradients.py
```
