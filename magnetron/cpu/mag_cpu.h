/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
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
#include <core/mag_threadlib.h>

#include "mag_cpu_numa.h"
#include "mag_cpu_async_stream.h"
#include "mag_cpu_threadpool.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_CPU_BACKEND_VERSION mag_ver_encode(0, 1, 0)
#define MAG_CPU_INLINE_MAX_NUMEL 8192   /* Async mode: single threaded commands with numel <= this run on the submitting thread when the stream is idle as they complete small */
#define MAG_CPU_ASYNC_RESERVED_CPUS 2   /* Logical CPUs left free in async mode */

mag_backend_decl_interface();

typedef struct mag_cpu_device_t {
  mag_context_t *ctx;
  mag_device_t *device;
  mag_lock_t submit_lock;
  mag_cpu_stream_t *stream;
  mag_thread_pool_t *pool;
  uint32_t num_allocated_workers;
  uint32_t num_core_workers;
  mag_kernel_registry_t kernels;
  mag_philox4x32_stream_t primary_prng;
  mag_numa_node_controller_t numa_ctrl;
} mag_cpu_device_t;

#ifdef __cplusplus
}
#endif

#endif
