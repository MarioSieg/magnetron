/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#ifndef MAG_CPU_KERNEL_DATA_H
#define MAG_CPU_KERNEL_DATA_H

#include <core/mag_backend.h>
#include <core/mag_prng_philox4x32.h>

#include "mag_cpu_autotune.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CPU Compute kernel payload passed to each CPU thread. */
typedef struct mag_kernel_payload_t {
    const mag_command_t *cmd;
    int64_t thread_num;
    int64_t thread_idx;
    mag_philox4x32_stream_t *prng;
    volatile mag_atomic64_t *mm_next_tile;
    mag_matmul_block_params_t mm_params;
} mag_kernel_payload_t;

/*
** Stores function-pointer lookup table for all compute kernels.
** The lookup table is used to dispatch the correct kernel for each operation by indexing with the opcode.
** The CPU runtime dynamically fills these arrays with the best fitting kernel depending on the detected CPU.
** See magnetron_cpu.c for details.
*/
typedef struct mag_kernel_registry_t {
    void (*init)(void);
    void (*deinit)(void);
    void (*operators[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_kernel_payload_t *);      /* Eval operator kernels. */
    size_t (*vreg_width)(void);
} mag_kernel_registry_t;

#ifdef __cplusplus
}
#endif

#endif
