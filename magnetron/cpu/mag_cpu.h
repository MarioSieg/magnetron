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

#include <mag_backend.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_CPU_BACKEND_VERSION mag_ver_encode(0, 1, 0)

mag_backend_decl_interface();

typedef struct mag_matmul_block_tune_info_t {
    int64_t nthreads;
    int64_t elsize;
    int64_t vecreg_width;
    int64_t M;
    int64_t N;
    int64_t K;
    int64_t l1_size;
    int64_t l2_size;
    mag_e11m52_t l1_load_factor;
    mag_e11m52_t l2_load_factor;
    int64_t min_tile_flops;
    mag_e11m52_t split_a;
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

#define MAG_PHILOX_ROUNDS 10
typedef struct mag_philox4x32_ctr_t {
    uint32_t v[4];
} mag_philox4x32_ctr_t;
typedef struct mag_philox4x32_key_t {
    uint32_t v[2];
} mag_philox4x32_key_t;
typedef struct mag_philox4x32_stream_t {
    mag_philox4x32_ctr_t ctr;
    mag_philox4x32_key_t key;
} mag_philox4x32_stream_t;

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
    void (*vector_cast)(size_t nb, const void *src, mag_dtype_t src_t, void *dst, mag_dtype_t dst_t); /* Vector cast (dtype conversion) kernel. */
    size_t (*vreg_width)(void);
} mag_kernel_registry_t;

#ifdef __cplusplus
}
#endif

#endif
