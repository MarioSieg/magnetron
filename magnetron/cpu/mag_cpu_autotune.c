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

#include "mag_cpu_autotune.h"
#include "mag_cpu.h"

#include <core/mag_context.h>
#include <core/mag_tensor.h>

mag_op_thread_scaling_info mag_cpu_get_op_thread_scaling_info(mag_opcode_t op) {
    static const mag_op_thread_scaling_info scaling_table[MAG_OP__NUM] = {
        [MAG_OP_NOP] = {0.0, 0},
        [MAG_OP_FILL] = {0.5, 10000},
        [MAG_OP_MASKED_FILL] = {0.5, 10000},
        [MAG_OP_RAND_UNIFORM] = {0.8, 10000},
        [MAG_OP_RAND_NORMAL] = {1.0, 10000},
        [MAG_OP_RAND_BERNOULLI] = {0.0, 0},
        [MAG_OP_RAND_PERM] = {0.0, 0},
        [MAG_OP_ARANGE] = {0.4, 10000},
        [MAG_OP_ONE_HOT] = {0.4, 10000},
        [MAG_OP_CLONE] = {0.4, 10000},
        [MAG_OP_CAST] = {0.4, 10000},
        [MAG_OP_VIEW] = {0.0, 0},
        [MAG_OP_TRANSPOSE] = {0.0, 0},
        [MAG_OP_PERMUTE] = {0.0, 0},
        [MAG_OP_MEAN] = {0.0, 0},
        [MAG_OP_MIN] = {0.0, 0},
        [MAG_OP_MAX] = {0.0, 0},
        [MAG_OP_ARGMIN] = {0.0, 0},
        [MAG_OP_ARGMAX] = {0.0, 0},
        [MAG_OP_SUM] = {0.0, 0},
        [MAG_OP_PROD] = {0.0, 0},
        [MAG_OP_ALL] = {0.0, 0},
        [MAG_OP_ANY] = {0.0, 0},
        [MAG_OP_ABS] = {0.5, 25000},
        [MAG_OP_SGN] = {0.5, 25000},
        [MAG_OP_NEG] = {0.5, 25000},
        [MAG_OP_LOG] = {0.5, 25000},
        [MAG_OP_LOG10] = {0.5, 25000},
        [MAG_OP_LOG1P] = {0.5, 25000},
        [MAG_OP_LOG2] = {0.5, 25000},
        [MAG_OP_SQR] = {0.5, 25000},
        [MAG_OP_RCP] = {0.5, 25000},
        [MAG_OP_SQRT] = {0.5, 25000},
        [MAG_OP_RSQRT] = {0.5, 25000},
        [MAG_OP_SIN] = {0.5, 25000},
        [MAG_OP_COS] = {0.5, 25000},
        [MAG_OP_TAN] = {0.5, 25000},
        [MAG_OP_SINH] = {0.5, 25000},
        [MAG_OP_COSH] = {0.5, 25000},
        [MAG_OP_TANH] = {0.5, 25000},
        [MAG_OP_ASIN] = {0.5, 25000},
        [MAG_OP_ACOS] = {0.5, 25000},
        [MAG_OP_ATAN] = {0.5, 25000},
        [MAG_OP_ASINH] = {0.5, 25000},
        [MAG_OP_ACOSH] = {0.5, 25000},
        [MAG_OP_ATANH] = {0.5, 25000},
        [MAG_OP_STEP] = {0.5, 25000},
        [MAG_OP_ERF] = {0.5, 25000},
        [MAG_OP_ERFC] = {0.5, 25000},
        [MAG_OP_EXP] = {0.5, 25000},
        [MAG_OP_EXP2] = {0.5, 25000},
        [MAG_OP_EXPM1] = {0.5, 25000},
        [MAG_OP_FLOOR] = {0.5, 25000},
        [MAG_OP_CEIL] = {0.5, 25000},
        [MAG_OP_ROUND] = {0.5, 25000},
        [MAG_OP_TRUNC] = {0.5, 25000},
        [MAG_OP_SOFTMAX] = {0.9, 25000},
        [MAG_OP_SOFTMAX_DV] = {0.5, 25000},
        [MAG_OP_SIGMOID] = {0.5, 25000},
        [MAG_OP_SIGMOID_DV] = {0.5, 25000},
        [MAG_OP_HARD_SIGMOID] = {0.5, 25000},
        [MAG_OP_SILU] = {0.5, 25000},
        [MAG_OP_SILU_DV] = {0.5, 25000},
        [MAG_OP_TANH_DV] = {0.5, 25000},
        [MAG_OP_RELU] = {0.5, 25000},
        [MAG_OP_RELU_DV] = {0.5, 25000},
        [MAG_OP_GELU] = {0.5, 25000},
        [MAG_OP_GELU_APPROX] = {0.5, 25000},
        [MAG_OP_GELU_DV] = {0.5, 25000},
        [MAG_OP_TRIL] = {0.5, 10000},
        [MAG_OP_TRIU] = {0.5, 10000},
        [MAG_OP_MULTINOMIAL] = {0.5, 25000},
        [MAG_OP_CAT] = {0.8, 10000},
        [MAG_OP_ADD] = {3.5, 10000},
        [MAG_OP_SUB] = {3.5, 10000},
        [MAG_OP_MUL] = {3.5, 10000},
        [MAG_OP_DIV] = {3.5, 10000},
        [MAG_OP_MOD] = {3.5, 10000},
        [MAG_OP_MATMUL] = {0.4, 1000},
        [MAG_OP_REPEAT_BACK] = {0.5, 25000},
        [MAG_OP_GATHER] = {0.0, 0},
        [MAG_OP_AND] = {3.5, 10000},
        [MAG_OP_OR] = {3.5, 10000},
        [MAG_OP_XOR] = {3.5, 10000},
        [MAG_OP_NOT] = {3.5, 10000},
        [MAG_OP_SHL] = {3.5, 10000},
        [MAG_OP_SHR] = {3.5, 10000},
        [MAG_OP_EQ] = {3.5, 10000},
        [MAG_OP_NE] = {3.5, 10000},
        [MAG_OP_LE] = {3.5, 10000},
        [MAG_OP_GE] = {3.5, 10000},
        [MAG_OP_LT] = {3.5, 10000},
        [MAG_OP_GT] = {3.5, 10000},
    };
    return scaling_table[op];
}

void mag_mm_autotune_block_params(const mag_matmul_block_tune_info_t *info, mag_matmul_block_params_t *params) {
    if (!info->l1_size || !info->l2_size || !info->elsize) {
        *params = (mag_matmul_block_params_t) {
            .MR = 8,
            .NR = 16,
            .MC = 256,
            .KC = 256,
            .NC = 128
        };
        return;
    }
    int64_t nt = info->nthreads;
    int64_t M = info->M;
    int64_t N = info->N;
    int64_t K = info->K;
    int64_t MR;
    int64_t NR;
    int64_t KC;
    int64_t VW = info->vecreg_width;
    int64_t W = VW >= 64 ? 64 : VW >= 32 ? 32 : 16;
    MR = VW / info->elsize;
    int64_t NR_cap = W == 64 ? 32 : W == 32 ? 32 : 16;
    NR = mag_clamp((MR)<<1, MR, NR_cap);
    if (W == 64) MR = 16, NR = 32;
    double aL1 = info->l1_load_factor ? info->l1_load_factor : W == 64 ? 0.55 : W == 32 ? 0.60 : 0.65;
    double aL2 = info->l2_load_factor ? info->l2_load_factor : W == 64 ? 0.40 : W == 32 ? 0.45 : 0.50;
    double L1e = aL1 * (double)info->l1_size;
    double L2e = aL2 * (double)info->l2_size;
    if (nt >= 2) {
        L1e *= 0.85;
        L2e *= 0.85;
    }
    double nb = (double)info->elsize;
    int64_t kc = (int64_t)(L1e / (nb*(double)(MR + NR)));
    kc = mag_rd_down(kc, 8);
    int64_t KC_lo = W == 64 ? 384 : W == 32 ? 256 : 192;
    int64_t KC_hi = W == 64 ? 1024 : W == 32 ? 768 : 512;
    kc = mag_clamp(kc, KC_lo, KC_hi);
    if (K >= 2048) kc = mag_clamp(kc + 128, KC_lo, KC_hi);
    KC = kc;
    int64_t MC = (int64_t)(info->split_a*L2e / (nb*(double)KC));
    int64_t NC = (int64_t)((1.0-info->split_a)*L2e / (nb*(double)KC));
    MC = mag_rd_down(MC, MR);
    NC = mag_rd_down(NC, NR);
    MR = mag_xmax(8, MR);
    NR = mag_xmax(8, NR);
    if (MC < MR) MC = MR;
    if (NC < NR) NC = NR;
    int64_t NC_cap = W == 64 ? 256 : 128;
    if (N < 8192) NC_cap = 128;
    if (NC > NC_cap) NC = mag_rd_down(NC_cap, NR);
    int64_t tic = (M + MC - 1)/MC;
    int64_t tjc = (N + NC - 1)/NC;
    int64_t tiles = tic * tjc;
    int64_t flops_call = (M*N*K)<<1;
    int64_t min_tiles_core = flops_call >= 0x10000000ll ? 1 : flops_call >= 0x2000000ll ? 2 : 4;
    int64_t tiles_needed = min_tiles_core * nt;
    if (tiles_needed < (nt<<1)+nt) tiles_needed = (nt<<1)+nt;
    while (tiles < tiles_needed && (MC > MR<<4 || NC > NR<<4)) {
        bool changed = false;
        int64_t nMC = MC>>1;
        if (!changed && nMC >= MR && (nMC*NC*KC)<<1 >= info->min_tile_flops) {
            MC = mag_rd_down(nMC, MR);
            changed = true;
        }
        int64_t nNC = NC>>1;
        if (!changed && nNC >= NR && (MC*nNC*KC)<<1 >= info->min_tile_flops) {
            NC = mag_rd_down(nNC, NR);
            changed = true;
        }
        if (!changed) break;
        tic = (M + MC - 1)/MC;
        tjc = (N + NC - 1)/NC;
        tiles = tic * tjc;
    }
    if (N >= 512 && NC < NR<<1) NC = NR<<1;
    *params = (mag_matmul_block_params_t) {
        .MR = MR,
        .NR = NR,
        .MC = MC,
        .KC = KC,
        .NC = NC
    };
}

/*
** Computes how many workers to use for intra-op parallelism depending on the number of elements.
** A logarithmic scaling is used, see: https://www.desmos.com/calculator/xiunrskpwu
** TODO: This can be improved by using a more sophisticated heuristic and a benchmarked, numerical approach.
*/
uint32_t mag_cpu_dynamic_work_scaling(uint32_t allocated_workers, mag_opcode_t op, int64_t numel) {
    const mag_op_traits_t *meta = mag_op_traits(op);
    mag_op_thread_scaling_info info = mag_cpu_get_op_thread_scaling_info(op);
    if (allocated_workers <= 1 || !(meta->flags & MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING) || numel < info.thread_treshold)  /* Use a single worker (main thread). */
        return 1;
    numel -= info.thread_treshold;                                                             /* Saturate threshold */
    uint32_t workers = (uint32_t)ceil(info.growth * log2((double)numel));         /* Logarithmic scaling */
    workers = mag_xmin(allocated_workers, mag_xmax(1, workers));
    return workers;
}

static uint32_t mag_mm_choose_workers(uint64_t flops, uint32_t tiles_total, uint32_t max_threads) {
    if (flops < 0x80000) return 1;
    (void)tiles_total;
    return max_threads;
}

uint32_t mag_cpu_tune_heuristics_intraop_workers(const mag_command_t *cmd, mag_device_t *dvc) {
    mag_cpu_device_t *cpu_dvc = dvc->impl;
    if (cmd->op == MAG_OP_MATMUL) {
        const mag_tensor_t *x = cmd->in[0];
        const mag_tensor_t *y = cmd->in[1];
        int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
        int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
        int64_t K = x->coords.shape[x->coords.rank-1];
        int64_t L1 = dvc->ctx->machine.cpu_l1_size;
        int64_t L2 = dvc->ctx->machine.cpu_l2_size;
        const mag_matmul_block_tune_info_t tune_info = {
            .nthreads = cpu_dvc->num_allocated_workers,
            .elsize = (int64_t)x->storage->granularity,
            .vecreg_width = (int64_t)(*cpu_dvc->kernels.vreg_width)(),
            .M = M,
            .N = N,
            .K = K,
            .l1_size = L1,
            .l2_size = L2,
            .l1_load_factor = 0.8,
            .l2_load_factor = 0.8,
            .min_tile_flops = (16*1024*1024),
            .split_a = 0.8,
            .min_n_factor = 16,
            .min_m_factor = 16,
        };
        mag_matmul_block_params_t tuned;
        mag_mm_autotune_block_params(&tune_info, &tuned); /* Tune block params */
        for (uint32_t i=0; i < cpu_dvc->pool->num_allocated_workers; ++i) { /* Set up payload */
            mag_kernel_payload_t *payload = &cpu_dvc->pool->workers[i].payload;
            payload->mm_params = tuned;
        }
        if (M == 1 && K >= 128 && N >= 4096 && y->coords.rank == 2 && y->coords.strides[y->coords.rank-1] == 1) /* Special case for GEMV */
            return mag_xmax(cpu_dvc->num_allocated_workers, 4)>>1;
        int64_t flops = M*N*K;
        uint32_t tiles_total = (uint32_t)(((M + tuned.MC - 1)/tuned.MC)*((N + tuned.NC - 1)/tuned.NC));
        uint32_t nt = mag_mm_choose_workers(flops, tiles_total, cpu_dvc->num_allocated_workers);
        /*printf("MM nt: %u, M=%" PRId64 ", N=%" PRId64 ", K=%" PRId64 ", MC=%" PRId64 ", NC=%" PRId64 ", KC=%" PRId64 ", MR=%" PRId64 ", NR=%" PRId64 "\n", nt, M, N, K, MC, NC, KC, MR, NR);*/
        return nt;
    }
    int64_t max_numel = INT64_MIN;
    for (uint32_t i=0; i < cmd->num_in; ++i) max_numel = mag_xmax(max_numel, cmd->in[i]->numel);
    for (uint32_t i=0; i < cmd->num_out; ++i) max_numel = mag_xmax(max_numel, cmd->out[i]->numel);
    return mag_cpu_dynamic_work_scaling(cpu_dvc->num_allocated_workers, cmd->op, max_numel);
}
