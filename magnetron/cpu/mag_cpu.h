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

#ifndef MAGNETRON_CPU_H
#define MAGNETRON_CPU_H

#include <core/mag_backend.h>
#include <core/mag_thread.h>

#include "mag_cpu_threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_CPU_BACKEND_VERSION mag_ver_encode(0, 1, 0)

mag_backend_decl_interface();

typedef struct mag_cpu_device_t {
    mag_context_t *ctx;
    mag_thread_pool_t *pool;                /* Thread pool. NULL if num_allocated_workers <= 1 */
    uint32_t num_allocated_workers;         /* Amount of worker thread used. if == 1 then single threaded mode and thread pool is not created */
    mag_kernel_registry_t kernels;          /* Compute kernels. Specialized by arch optimized version at boot (e.g. AVX, AVX512 etc..) */
    mag_philox4x32_stream_t primary_prng;   /* Primary prng context. */
} mag_cpu_device_t;

#ifdef __cplusplus
}
#endif

#endif
