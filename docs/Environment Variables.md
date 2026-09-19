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

## MAG_FUSE_COMPILE

Read by `mag_envcfg_fuse_compile_enabled()`.

Whether a backend may turn a fused chain into compiled code. Turning it off leaves every chain to
whatever the backend does without a compiler, which for the CPU backend means interpreting it.
Intended for measuring one lowering against the other, and for machines where invoking a compiler at
runtime is unwelcome.

Allowed values: `on` (default), `off`. `1`/`0` and `true`/`false` are accepted too.

A chain produces the same bits either way, so this changes speed and nothing else.

Example:
```bash
export MAG_FUSE_COMPILE=off
```

## MAG_FUSE_CC

Read by `mag_envcfg_fuse_cc()`.

The compiler used to build fused chains. Defaults to `cc`.

This is a program name, not a command line: it is passed as the first element of an argument vector
and looked up on `PATH`, never handed to a shell. `MAG_FUSE_CC='clang -march=native'` will not work
and is not meant to - a compiler name that can carry arguments is a compiler name that can carry
anything else.

If the named program cannot be run, or the compile fails, the chain runs interpreted and a warning
says so. Raise `MAG_LOG_LEVEL` to `warn` or above to see it.

Example:
```bash
export MAG_FUSE_CC=clang
```

## MAG_FUSE_CACHE_DIR

Read by `mag_envcfg_fuse_cache_dir()`.

Where compiled chains are kept between runs. Defaults to `$HOME/.cache/magnetron/fused`, created
with owner-only permissions.

Compiling a chain costs tens of milliseconds the first time its shape is seen; a later process that
finds the object already built pays a fraction of a millisecond. Objects are named by a hash of the
generated source, so a change to code generation cannot pick up one built by an older version.

Point this somewhere a process the user does not control can write to and that process chooses what
this one loads, so prefer leaving it alone.

Example:
```bash
export MAG_FUSE_CACHE_DIR=/var/tmp/my-magnetron-cache
```
