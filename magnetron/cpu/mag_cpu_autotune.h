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

typedef struct mag_matmul_block_tune_info_t {
    int64_t nthreads;
    int64_t elsize;
    int64_t vecreg_width;
    int64_t M;
    int64_t N;
    int64_t K;
    int64_t l1_size;
    int64_t l2_size;
    double l1_load_factor;
    double l2_load_factor;
    int64_t min_tile_flops;
    double split_a;
    int64_t min_n_factor;
    int64_t min_m_factor;
} mag_matmul_block_tune_info_t;

typedef struct mag_matmul_block_params_t {
    int64_t MR;
    int64_t NR;
    int64_t MC;
    int64_t KC;
    int64_t NC;
} mag_matmul_block_params_t;

extern void mag_mm_autotune_block_params(const mag_matmul_block_tune_info_t *info, mag_matmul_block_params_t *params);
extern uint32_t mag_cpu_dynamic_work_scaling(uint32_t allocated_workers, mag_opcode_t op, int64_t numel);
extern uint32_t mag_cpu_tune_heuristics_intraop_workers(const mag_command_t *cmd, mag_device_t *dvc);

#ifdef __cplusplus
}
#endif

#endif
