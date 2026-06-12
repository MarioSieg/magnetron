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

#ifndef MAG_CPU_AUTOTUNE_H
#define MAG_CPU_AUTOTUNE_H

#include <core/mag_operator.h>
#include <core/mag_backend.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_op_thread_scaling_info {
  double growth;        /* Logarithmic growth factor for the number of threads */
  int64_t thread_treshold;    /* Number of elements after which multithreading kicks in */
} mag_op_thread_scaling_info;
extern mag_op_thread_scaling_info mag_cpu_get_op_thread_scaling_info(mag_opcode_t op);

extern uint32_t mag_cpu_tune_eager_intra_op_worker_count(const mag_command_t *cmd, mag_device_t *dvc);

#ifdef __cplusplus
}
#endif

#endif
