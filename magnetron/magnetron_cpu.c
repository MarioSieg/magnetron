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

#include "magnetron_internal.h"

#include <math.h>

typedef struct mag_op_cpu_info_t {
    mag_e11m52_t growth;        /* Logarithmic growth factor for the number of threads */
    int64_t thread_treshold;    /* Number of elements after which multithreading kicks in */
} mag_op_cpu_info_t;

static const mag_op_cpu_info_t mag_op_cpu_info_lut[MAG_OP__NUM] = {
    [MAG_OP_NOP] = {0.1, 250000},
    [MAG_OP_FILL] = {0.1, 250000},
    [MAG_OP_MASKED_FILL] = {0.1, 250000},
    [MAG_OP_RAND_UNIFORM] = {0.1, 250000},
    [MAG_OP_RAND_NORMAL] = {0.1, 250000},
    [MAG_OP_RAND_BERNOULLI] = {0.1, 250000},
    [MAG_OP_ARANGE] = {0.1, 250000},
    [MAG_OP_CLONE] = {0.1, 250000},
    [MAG_OP_VIEW] = {0.1, 250000},
    [MAG_OP_TRANSPOSE] = {0.1, 250000},
    [MAG_OP_PERMUTE] = {0.1, 250000},
    [MAG_OP_MEAN] = {0.1, 250000},
    [MAG_OP_MIN] = {0.1, 250000},
    [MAG_OP_MAX] = {0.1, 250000},
    [MAG_OP_SUM] = {0.1, 250000},
    [MAG_OP_ABS] = {0.1, 250000},
    [MAG_OP_SGN] = {0.1, 250000},
    [MAG_OP_NEG] = {0.1, 250000},
    [MAG_OP_LOG] = {0.1, 250000},
    [MAG_OP_SQR] = {0.1, 250000},
    [MAG_OP_SQRT] = {0.1, 250000},
    [MAG_OP_SIN] = {0.1, 250000},
    [MAG_OP_COS] = {0.1, 250000},
    [MAG_OP_STEP] = {0.1, 250000},
    [MAG_OP_EXP] = {0.1, 250000},
    [MAG_OP_FLOOR] = {0.1, 250000},
    [MAG_OP_CEIL] = {0.1, 250000},
    [MAG_OP_ROUND] = {0.1, 250000},
    [MAG_OP_SOFTMAX] = {0.1, 250000},
    [MAG_OP_SOFTMAX_DV] = {0.1, 250000},
    [MAG_OP_SIGMOID] = {0.1, 250000},
    [MAG_OP_SIGMOID_DV] = {0.1, 250000},
    [MAG_OP_HARD_SIGMOID] = {0.1, 250000},
    [MAG_OP_SILU] = {0.1, 250000},
    [MAG_OP_SILU_DV] = {0.1, 250000},
    [MAG_OP_TANH] = {0.1, 250000},
    [MAG_OP_TANH_DV] = {0.1, 250000},
    [MAG_OP_RELU] = {0.1, 250000},
    [MAG_OP_RELU_DV] = {0.1, 250000},
    [MAG_OP_GELU] = {0.1, 250000},
    [MAG_OP_GELU_APPROX] = {0.1, 250000},
    [MAG_OP_GELU_DV] = {0.1, 250000},
    [MAG_OP_TRIL] = {0.1, 250000},
    [MAG_OP_TRIU] = {0.1, 250000},
    [MAG_OP_MULTINOMIAL] = {0.1, 250000},
    [MAG_OP_ADD] = {3.5, 10000},
    [MAG_OP_SUB] = {3.5, 10000},
    [MAG_OP_MUL] = {3.5, 10000},
    [MAG_OP_DIV] = {3.5, 10000},
    [MAG_OP_MATMUL] = {0.4, 1000},
    [MAG_OP_REPEAT_BACK] = {0.1, 250000},
    [MAG_OP_GATHER] = {0.1, 250000},
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

extern void mag_cpu_blas_specialization_fallback(mag_kernel_registry_t* kernels); /* Generic any CPU impl */

#if defined(__x86_64__) || defined(_M_X64) /* Specialized impls for x86-64 with runtime CPU detection */

typedef struct mag_amd64_blas_spec_t {
    const char* name;
    mag_amd64_cap_bitset_t (*get_feature_permutation)(void);
    void (*inject_kernels)(mag_kernel_registry_t* kernels);
} mag_amd64_blas_spec_t;

#define mag_amd64_blas_spec_decl(feat) \
    mag_amd64_cap_bitset_t mag_cpu_blas_specialization_amd64_v##feat##_features(void); \
    extern void mag_cpu_blas_specialization_amd64_v##feat(mag_kernel_registry_t* kernels)

#define mag_amd64_blas_spec_permute(feat) \
    (mag_amd64_blas_spec_t) { \
        .name = "amd64-v."#feat, \
        .get_feature_permutation = &mag_cpu_blas_specialization_amd64_v##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_amd64_v##feat \
    }

mag_amd64_blas_spec_decl(4_5);
mag_amd64_blas_spec_decl(4);
mag_amd64_blas_spec_decl(3);
mag_amd64_blas_spec_decl(2_5);
mag_amd64_blas_spec_decl(2);

static bool mag_blas_detect_gen_optimal_spec(const mag_context_t* host_ctx, mag_kernel_registry_t* kernels) {
    const mag_amd64_blas_spec_t impls[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
        mag_amd64_blas_spec_permute(4_5),
        mag_amd64_blas_spec_permute(4),
        mag_amd64_blas_spec_permute(3),
        mag_amd64_blas_spec_permute(2_5),
        mag_amd64_blas_spec_permute(2),
    };

    mag_amd64_cap_bitset_t cap_machine = host_ctx->machine.amd64_cpu_caps;
    for (size_t i=0; i < sizeof(impls)/sizeof(*impls); ++i) { /* Find best blas spec for the host CPU */
        const mag_amd64_blas_spec_t* spec = impls+i;
        mag_amd64_cap_bitset_t cap_required = (*spec->get_feature_permutation)(); /* Get requires features */
        if ((cap_machine & cap_required) == cap_required) { /* Since specializations are sorted by score, we found the perfect spec. */
            (*spec->inject_kernels)(kernels);
            mag_log_info("Using tuned BLAS specialization: %s", spec->name);
            return true;
        }
    }
    /* No matching specialization found, use generic */
    mag_cpu_blas_specialization_fallback(kernels);
    mag_log_info("Using fallback BLAS specialization");
    return false; /* No spec used, fallback is active */
}

#undef mag_amd64_blas_spec_permute
#undef mag_cpu_blas_spec_decl

#elif defined(__aarch64__) || defined(_M_ARM64)

typedef struct mag_arm64_blas_spec_t {
    const char* name;
    mag_arm64_cap_bitset_t (*get_cap_permutation)(void);
    void (*inject_kernels)(mag_kernel_registry_t* kernels);
} mag_arm64_blas_spec_t;

#define mag_arm64_blas_spec_decl(feat) \
    mag_arm64_cap_bitset_t mag_cpu_blas_specialization_arm64_v_##feat##_features(void); \
    extern void mag_cpu_blas_specialization_arm64_v_##feat(mag_kernel_registry_t* kernels)

#define mag_arm64_blas_spec_permute(feat) \
    (mag_arm64_blas_spec_t) { \
        .name = "arm64-v."#feat, \
        .get_cap_permutation = &mag_cpu_blas_specialization_arm64_v_##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_arm64_v_##feat \
}

mag_arm64_blas_spec_decl(9);
mag_arm64_blas_spec_decl(8_2);

static bool mag_blas_detect_gen_optimal_spec(const mag_context_t* ctx, mag_kernel_registry_t* kernels) {
    const mag_arm64_blas_spec_t impls[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
        mag_arm64_blas_spec_permute(9),
        mag_arm64_blas_spec_permute(8_2),
    };

    mag_arm64_cap_bitset_t cap_avail = ctx->machine.arm64_cpu_caps;
    for (size_t i=0; i < sizeof(impls)/sizeof(*impls); ++i) { /* Find best blas spec for the host CPU */
        const mag_arm64_blas_spec_t* spec = impls+i;
        mag_arm64_cap_bitset_t cap_required = (*spec->get_cap_permutation)(); /* Get requires features */
        if ((cap_avail & cap_required) == cap_required) { /* Since specializations are sorted by score, we found the perfect spec. */
            (*spec->inject_kernels)(kernels);
            mag_log_info("Using tuned BLAS specialization: %s", spec->name);
            return true;
        }
    }
    /* No matching specialization found, use generic */
    mag_cpu_blas_specialization_fallback(kernels);
    mag_log_info("Using fallback BLAS specialization");
    return false; /* No spec used, fallback is active */
}

#undef mag_cpu_blas_spec_decl

#endif

static bool mag_blas_detect_optimal_specialization(const mag_context_t* ctx, mag_kernel_registry_t* kernels) {
    if (mag_likely(mag_blas_detect_gen_optimal_spec(ctx, kernels))) return true;
    mag_cpu_blas_specialization_fallback(kernels);
    return false; /* No spec used, fallback is active */
}

typedef struct mag_phase_fence_t {
    mag_atomic32_t phase;
    mag_atomic32_t remaining;
} mag_phase_fence_t;
static inline void mag_phase_fence_init(mag_phase_fence_t* f) {
    f->phase = f->remaining = 0;
}
static inline void mag_phase_fence_kick(mag_phase_fence_t* f, int32_t workers_active) {
    mag_atomic32_store(&f->remaining, workers_active, MAG_MO_RELAXED);
    int32_t cur = mag_atomic32_load(&f->phase, MAG_MO_RELAXED);
    mag_atomic32_store(&f->phase, cur + 1, MAG_MO_RELEASE);
    mag_futex_wakeall(&f->phase);
}
static inline void mag_phase_fence_wait(mag_phase_fence_t* f, int32_t* pha) {
    int32_t p = *pha;
    for (int spin=0; spin < 20; ++spin) {
        if (mag_atomic32_load(&f->phase, MAG_MO_ACQUIRE) != p) goto ready;
        mag_cpu_pause();
    }
    while (mag_atomic32_load(&f->phase, MAG_MO_ACQUIRE) == p)
        mag_futex_wait(&f->phase, p);
ready:
    *pha = p+1;
}
static inline void mag_phase_fence_done(mag_phase_fence_t* f) {
    if (mag_atomic32_fetch_sub(&f->remaining, 1, MAG_MO_ACQ_REL) == 1)
        mag_futex_wake1(&f->remaining);
}
static inline void mag_phase_fence_barrier(mag_phase_fence_t* f) {
    int32_t val = mag_atomic32_load(&f->remaining, MAG_MO_ACQUIRE);
    while (val != 0) {
        mag_futex_wait(&f->remaining, val);
        val = mag_atomic32_load(&f->remaining, MAG_MO_ACQUIRE);
    }
}

typedef struct mag_worker_t mag_worker_t;
typedef struct mag_thread_pool_t {
    mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile bool interrupt;   /* Interrupt flag, 1=stop */
    mag_phase_fence_t fence;
    int32_t num_allocated_workers;                 /* Number of intra-op workers allocated */
    uint32_t num_active_workers;                    /* Number of intra-op workers that are actively used in this compute step. */
    volatile mag_atomic32_t num_workers_online;       /* Number of workers that are online */
    mag_worker_t* workers;                          /* Array of workers */
    const mag_kernel_registry_t* kernels;           /* Specialized compute kernel registry */
    mag_thread_prio_t sched_prio;             /* Scheduling priority */
    mag_context_t* host_ctx;                            /* Host context */
} mag_thread_pool_t;

struct mag_worker_t {
    int32_t phase;                         /* Current compute phase */
    mag_kernel_payload_t payload;          /* Compute op payload */
    mag_prng_state_t prng;                  /* Thread local prng */
    mag_thread_pool_t* pool;                 /* Host thread pool */
    bool is_async;                          /* True if worker is async (executed on a different thread)  */
    mag_thread_t thread;                    /* Thread handle */
} mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE);

typedef struct mag_cpu_device_t {
    mag_context_t* ctx;
    mag_thread_pool_t* pool;               /* Thread pool. NULL if num_allocated_workers <= 1 */
    uint32_t num_allocated_workers;     /* Amount of worker thread used. if == 1 then single threaded mode and thread pool is not created */
    mag_kernel_registry_t kernels;      /* Compute kernels. Specialized by arch optimized version at boot (e.g. AVX, AVX512 etc..) */
    mag_prng_state_t primary_prng;         /* Primary prng context. */
} mag_cpu_device_t;

/* Await signal to start work */
static bool mag_worker_await_work(mag_worker_t* worker, mag_thread_pool_t* pool) {
    if (mag_unlikely(pool->interrupt))
        return false;
    mag_phase_fence_wait(&pool->fence, &worker->phase);
    return !pool->interrupt;
}

/* Execute the operation on the current thread */
static void mag_worker_exec_thread_local(const mag_kernel_registry_t* kernels, mag_kernel_payload_t* payload) {
    if (mag_unlikely(!payload->cmd)) return;
    mag_opcode_t op = payload->cmd->op;
    mag_dtype_t dtype = *payload->cmd->in ? (*payload->cmd->in)->dtype : (*payload->cmd->out)->dtype;
    mag_assert2(op >= 0 && op < MAG_OP__NUM);
    mag_assert2(dtype >= 0 && dtype < MAG_DTYPE__NUM);
    void (*kernel)(const mag_kernel_payload_t*) = kernels->operators[op][dtype];
    mag_assert(kernel, "no kernel found for op '%s'  with dtype %s", mag_op_meta_of(op)->mnemonic, mag_dtype_meta_of(dtype)->name);
    (*kernel)(payload);
    payload->cmd = NULL;
}

/* Execute the operation and broadcast completion if last chunk was done */
static void mag_worker_exec_and_broadcast(mag_thread_pool_t* pool, const mag_kernel_registry_t* kernels, mag_kernel_payload_t* payload) {
    if (mag_likely(payload->thread_idx < pool->num_active_workers))
        mag_worker_exec_thread_local(kernels, payload);

    /* signal completion to master */
    mag_phase_fence_done(&pool->fence);
}

/* Worker thread entry point */
static MAG_HOTPROC void* mag_worker_thread_exec_op(void* arg) {
    mag_worker_t* worker = arg;
    mag_thread_pool_t* pool = worker->pool;
    mag_kernel_payload_t* payload = &worker->payload;
    const mag_kernel_registry_t* kernels = pool->kernels;
    char name[32];
    snprintf(name, sizeof(name), "mag_worker_%" PRIx64, payload->thread_idx);
    mag_thread_set_name(name);
    /*mag_thread_set_prio(pool->sched_prio);*/
    mag_prng_seed(&worker->prng, pool->host_ctx->prng_algo, rand()^mag_thread_id()^((uintptr_t)worker>>4)^((uintptr_t)kernels->vector_cast>>4));
    mag_atomic32_fetch_add(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    while (mag_likely(mag_worker_await_work(worker, pool)))  /* Main work loop: wait, work, signal status */
        mag_worker_exec_and_broadcast(pool, kernels, payload);
    mag_atomic32_fetch_sub(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    return MAG_THREAD_RET_NONE;
}

/* Create thread pool and allocate threads */
static mag_thread_pool_t* mag_threadpool_create(mag_context_t* host_ctx, uint32_t num_workers, const mag_kernel_registry_t* kernels, mag_thread_prio_t prio) { /* Create a thread pool */
    mag_thread_pool_t* pool = (*mag_alloc)(NULL, sizeof(*pool), __alignof(mag_thread_pool_t));
    memset(pool, 0, sizeof(*pool));
    mag_worker_t* workers = (*mag_alloc)(NULL, num_workers*sizeof(*workers), __alignof(mag_worker_t));
    memset(workers, 0, num_workers*sizeof(*workers));
    *pool = (mag_thread_pool_t){
        .interrupt = false,
        .num_allocated_workers = num_workers,
        .num_active_workers = num_workers,
        .num_workers_online = 0,  /* Main thread as worker 0 */
        .workers = workers,
        .kernels = kernels,
        .sched_prio = prio,
        .host_ctx = host_ctx
    };
    mag_phase_fence_init(&pool->fence);
    for (uint32_t ti=0; ti < num_workers; ++ti) { /* Initialize workers */
        mag_worker_t* worker = workers+ti;
        *worker = (mag_worker_t){
            .phase = 0,
            .payload = (mag_kernel_payload_t){
                .cmd = NULL, /* Will be set later */
                .thread_num = num_workers,
                .thread_idx = ti,
                .local_prng = NULL
            },
            .pool = pool,
            .is_async = ti != 0 /* Main thread is worker but without thread */
        };
        worker->payload.local_prng = &worker->prng;
        if (worker->is_async)
            mag_thread_create(&worker->thread, &mag_worker_thread_exec_op, workers+ti);
    }
    while (mag_atomic32_load(&pool->num_workers_online, MAG_MO_SEQ_CST) != num_workers-1)  /* Wait for all workers to come online */
        mag_thread_yield();
    return pool;
}

/* Destroy thread pool */
static void mag_threadpool_destroy(mag_thread_pool_t* pool) {
    pool->interrupt = true;
    mag_phase_fence_kick(&pool->fence, pool->num_allocated_workers);
    while (mag_atomic32_load(&pool->num_workers_online, MAG_MO_SEQ_CST))  /* Wait for all workers to exit */
        mag_thread_yield();
    for (uint32_t i=0; i < pool->num_allocated_workers; ++i) /* Join all worker threads */
        if (pool->workers[i].is_async)
            mag_thread_join(pool->workers[i].thread);
    (*mag_alloc)(pool->workers, 0, __alignof(mag_worker_t));
    (*mag_alloc)(pool, 0, __alignof(mag_thread_pool_t));
}

/* Submits work payload and awakens all threads */
static void mag_threadpool_kickoff(mag_thread_pool_t* pool, const mag_command_t* cmd, uint32_t num_active_workers, volatile mag_atomic64_t* next_tile) {
    pool->num_active_workers = num_active_workers;
    for (uint32_t i=0; i < pool->num_allocated_workers; ++i) { /* Set up payload */
        mag_kernel_payload_t* payload = &pool->workers[i].payload;
        payload->cmd = cmd;
        payload->thread_num = num_active_workers;
        payload->mm_next_tile = next_tile;
    }
    mag_phase_fence_kick(&pool->fence, pool->num_allocated_workers);
}

/* Blocks until all threads have completed their work */
static void mag_threadpool_barrier(mag_thread_pool_t* pool) {
    mag_phase_fence_barrier(&pool->fence);
}

/* Execute an operator tensor on the CPU */
static MAG_HOTPROC void mag_threadpool_parallel_compute(mag_thread_pool_t* pool, const mag_command_t* cmd, uint32_t num_active_workers) {
    mag_assert2(pool != NULL);
    volatile mag_atomic64_t next_tile = 0;
    mag_threadpool_kickoff(pool, cmd, num_active_workers, &next_tile);                  /* Kick off workers */
    mag_worker_exec_and_broadcast(pool, pool->kernels, &pool->workers->payload);        /* Main thread does work too */
    mag_threadpool_barrier(pool);                                                       /* Wait for all workers to finish */
}

static uint32_t mag_cpu_dynamic_work_scaling(mag_cpu_device_t* dvc, mag_opcode_t op, int64_t numel);

static uint32_t mag_mm_choose_workers(uint64_t flops, uint32_t tiles_total, uint32_t max_threads) {
    if (flops < 0x800000) return 1;
    uint32_t threads = mag_xmin(max_threads, tiles_total);
    uint32_t ideal_threads = (uint32_t)(flops / 0x400000);
    if (ideal_threads < 1) ideal_threads = 1;
    threads = mag_xmin(threads, ideal_threads);
    if (threads & (threads-1))
        threads = 1u<<mag_fls(threads);
    return threads ? threads : 1;
}

static uint32_t mag_cpu_tune_heuristics_intraop_workers(const mag_command_t* cmd, mag_idevice_t* dvc) {
    mag_cpu_device_t* cpu_dvc = dvc->impl;
    if (cmd->op == MAG_OP_MATMUL) {
        const mag_tensor_t* x = cmd->in[0];
        const mag_tensor_t* y = cmd->in[1];
        int64_t M = x->rank == 1 ? 1 : x->shape[x->rank-2];
        int64_t N = y->rank == 1 ? 1 : y->shape[y->rank-1];
        int64_t K = x->shape[x->rank-1];
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
        mag_matmul_tune_block_params(&tune_info, &tuned); /* Tune block params */
        for (uint32_t i=0; i < cpu_dvc->pool->num_allocated_workers; ++i) { /* Set up payload */
            mag_kernel_payload_t* payload = &cpu_dvc->pool->workers[i].payload;
            payload->mm_params = tuned;
        }
        if (M == 1 && K >= 128 && N >= 4096 && y->rank == 2 && y->strides[y->rank-1] == 1) /* Special case for GEMV */
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
    return mag_cpu_dynamic_work_scaling(cpu_dvc, cmd->op, max_numel);
}

static MAG_HOTPROC void mag_cpu_submit(mag_idevice_t* dvc, const mag_command_t* cmd) {
    mag_cpu_device_t* cpu_dvc = dvc->impl;
    uint32_t intraop_workers = mag_cpu_tune_heuristics_intraop_workers(cmd, dvc); /* Determine number of intra-op workers */
    if (intraop_workers <= 1) { /* Main thread does the work (single threaded mode). */
        volatile mag_atomic64_t next_tile = 0; /* Tile index for the next tile to process. */
        mag_kernel_payload_t* yy = &cpu_dvc->pool->workers[0].payload; /* TODO: Ugly */
        mag_kernel_payload_t payload = {
            .cmd = cmd,
            .thread_idx = 0,
            .thread_num = 1,
            .local_prng = &cpu_dvc->primary_prng,
            .mm_next_tile = &next_tile,
            .mm_params = yy->mm_params
        };
        mag_worker_exec_thread_local(&cpu_dvc->kernels, &payload);
        return; /* We're done */
    }
    mag_threadpool_parallel_compute(cpu_dvc->pool, cmd, intraop_workers); /* Multithreaded exec + barrier */
}

static void mag_cpu_broadcast(mag_istorage_t* sto, size_t offs, const void* x, size_t stride) {
    mag_assert2(offs <= sto->size);
    size_t len = sto->size-offs;
    uint8_t* dst8 = (uint8_t*)sto->base+offs;
    switch (stride) {
        case 1: memset(dst8, *(const uint8_t*)x, len); return;
        case 2: {
            mag_assert2(!(offs & 1) && !(len & 1));     /* Offsets and len must be aligned to granularity */
            uint16_t v = *(const uint16_t*)x;
            uint16_t* restrict p = (uint16_t*)dst8;
            uint16_t* restrict end = p + (len/sizeof(v));
            for (; p < end; ++p) *p = v;
        } return;
        case 4: {
            mag_assert2(!(offs & 3) && !(len & 3));     /* Offsets and len must be aligned to granularity */
            uint32_t v = *(const uint32_t*)x;
            uint32_t* restrict p = (uint32_t*)dst8;
            uint32_t* restrict end = p + (len/sizeof(v));
            for (; p < end; ++p) *p = v;
        } return;
        case 8: {
            mag_assert2(!(offs & 7) && !(len & 7));     /* Offsets and len must be aligned to granularity */
            uint64_t v = *(const uint64_t*)x;
            uint64_t* restrict p = (uint64_t*)dst8;
            uint64_t* restrict end = p + (len/sizeof(v));
            for (; p < end; ++p) *p = v;
        } return;
        default: {
            while (len >= stride) {
                memcpy(dst8, x, stride);
                dst8 += stride;
                len -= stride;
            }
            if (len) memcpy(dst8, x, len);  /* handle tail */
        }
    }
}

#define mag_ranges_overlap(a, asz, b, bsz) (((uintptr_t)(a)+(asz)) > (uintptr_t)(b) && ((uintptr_t)(b)+(bsz)) > (uintptr_t)(a))

static void mag_cpu_transfer(mag_istorage_t* sto, mag_transfer_dir_t dir, size_t offs, void* inout, size_t size) {
    mag_assert2(inout && size);
    mag_assert2(!(size & (sto->granularity-1)));
    uintptr_t base = sto->base;
    uintptr_t dp = base+offs;
    uintptr_t dpe = dp+size;
    mag_assert2(dpe > dp);
    mag_assert2(dpe <= base+sto->size);
    mag_assert2(!(offs & (sto->granularity-1)));
    uintptr_t hp = (uintptr_t)inout;
    uintptr_t hpe = hp+size;
    mag_assert2(hpe > hp);
    mag_assert2(!mag_ranges_overlap((void*)hp, size, (void*)dp, size));
    void* restrict device = (void*)dp;
    void* restrict host = inout;
    if (dir == MAG_TRANSFER_DIR_H2D) memcpy(device, host, size);
    else memcpy(host, device, size);
}

static void mag_cpu_convert(mag_istorage_t* sto, mag_transfer_dir_t dir, size_t offs, void* restrict host, size_t size, mag_dtype_t hdt) {
    mag_assert2(host && size);
    mag_dtype_t ddt = sto->dtype;
    if (ddt == hdt) { /* identical dtype – delegate to raw copy */
        (*sto->transfer)(sto, dir, offs, host, size);
        return;
    }
    uintptr_t base = sto->base;
    size_t hsz = mag_dtype_meta_of(hdt)->size;
    size_t dsz = mag_dtype_meta_of(ddt)->size;
    mag_assert2(!(hsz & (hsz-1)));              /* pow2 */
    mag_assert2(!(size & (hsz-1)));             /* multiple of dtype size */
    mag_assert2(!((uintptr_t)host & (hsz-1)));  /* aligned */
    mag_assert2(!(dsz & (dsz-1)));              /* pow2 */
    mag_assert2(!(offs & (dsz-1)));             /* multiple of dtype size */
    size_t helem = size / hsz;                  /* host numel */
    mag_assert2(helem > 0);
    mag_assert2(!(sto->granularity & (dsz-1))); /* sane granularity */
    size_t tnb = helem * sto->granularity;      /* device bytes */
    mag_assert2(tnb <= sto->size-offs);
    void* dvb = (void*)(base+offs);
    void (*kern)(size_t, const void*, mag_dtype_t, void*, mag_dtype_t)
        = ((mag_cpu_device_t*)sto->host->impl)->kernels.vector_cast;
    mag_assert2(kern);
    mag_assert2(!mag_ranges_overlap(host, size, dvb, tnb));
    void* restrict device = dvb;
    if (dir == MAG_TRANSFER_DIR_H2D) (*kern)(size, host, hdt, device, ddt);
    else (*kern)(tnb, device, ddt, host, hdt);
}

static void mag_cpu_storage_dtor(void* self) {
    mag_istorage_t* buf = self;
    mag_context_t* ctx = buf->ctx;
    mag_assert(ctx->num_storages > 0, "double freed storage");
    --ctx->num_storages;
    (*mag_alloc)((void*)buf->base, 0, MAG_CPU_BUF_ALIGN);
    mag_fixed_pool_free_block(&ctx->storage_pool, buf);
}

static void mag_cpu_alloc_storage(mag_idevice_t* host, mag_istorage_t** out, size_t size, mag_dtype_t dtype) {
    mag_context_t* ctx = host->ctx;
    void* block = (*mag_alloc)(NULL, size, MAG_CPU_BUF_ALIGN);
    *out = mag_fixed_pool_alloc_block(&ctx->storage_pool);
    **out = (mag_istorage_t){ /* Set up storage buffer. */
        .ctx = ctx,
        .rc_control = mag_rc_control_init(*out, &mag_cpu_storage_dtor),
        .base = (uintptr_t)block,
        .size = size,
        .alignment = MAG_CPU_BUF_ALIGN,
        .dtype = dtype,
        .granularity = mag_dtype_meta_of(dtype)->size,
        .host = host,
        .broadcast = &mag_cpu_broadcast,
        .transfer = &mag_cpu_transfer,
        .convert = &mag_cpu_convert
    };
    ++host->ctx->num_storages;
}

static mag_cpu_device_t* mag_cpu_init_device(mag_context_t* ctx, uint32_t num_threads) {
    mag_thread_prio_t sched_prio = MAG_THREAD_PRIO_HIGH;
    mag_cpu_device_t* dvc = (*mag_alloc)(NULL, sizeof(*dvc), 0);
    memset(dvc, 0, sizeof(*dvc));
    *dvc = (mag_cpu_device_t) {
        .ctx = ctx,
        .pool = NULL,
        .num_allocated_workers = 0,
        .kernels = {},
        .primary_prng = {}
    };
    mag_blas_detect_optimal_specialization(ctx, &dvc->kernels);
    mag_prng_seed(&dvc->primary_prng, ctx->prng_algo, rand()^((uintptr_t)ctx>>4)^((uintptr_t)dvc>>4));
    if (num_threads > 1) {
        dvc->pool = mag_threadpool_create(ctx, num_threads, &dvc->kernels, sched_prio);
        dvc->num_allocated_workers = num_threads;
    }
    if (*dvc->kernels.init) (*dvc->kernels.init)();
    return dvc;
}

/*
** Computes how many workers to use for intra-op parallelism depending on the number of elements.
** A logarithmic scaling is used, see: https://www.desmos.com/calculator/xiunrskpwu
** TODO: This can be improved by using a more sophisticated heuristic and a benchmarked, numerical approach.
*/
static uint32_t mag_cpu_dynamic_work_scaling(mag_cpu_device_t* dvc, mag_opcode_t op, int64_t numel) {
    const mag_opmeta_t* meta = mag_op_meta_of(op);
    const mag_op_cpu_info_t* info = mag_op_cpu_info_lut+op;
    if (!dvc->pool || !(meta->flags & MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING) || numel < info->thread_treshold)  /* Use a single worker (main thread). */
        return 1;
    numel -= info->thread_treshold;                                                             /* Saturate threshold */
    uint32_t workers = (uint32_t)ceil(info->growth * log2((mag_e11m52_t)numel));         /* Logarithmic scaling */
    workers = mag_xmin(dvc->num_allocated_workers, mag_xmax(1, workers));
    return workers;
}

static void mag_cpu_destroy_device(mag_cpu_device_t* dvc) {
    if (*dvc->kernels.deinit) (*dvc->kernels.deinit)();
    if (dvc->pool) mag_threadpool_destroy(dvc->pool);
    (*mag_alloc)(dvc, 0, 0);
}

static mag_idevice_t* mag_cpu_init_interface(mag_context_t* ctx, uint32_t num_threads) {
    mag_cpu_device_t* cpu_dvc = mag_cpu_init_device(ctx, num_threads);
    mag_idevice_t* dvc = (*mag_alloc)(NULL, sizeof(*dvc), 0);
    *dvc = (mag_idevice_t){ /* Initialize device interface */
        .ctx = ctx,
        .name = "CPU",
        .impl = cpu_dvc,
        .is_async = false,
        .type = MAG_DEVICE_TYPE_CPU,
        .submit = &mag_cpu_submit,
        .alloc_storage = &mag_cpu_alloc_storage,
    };
    snprintf(dvc->name, sizeof(dvc->name), "%s", ctx->machine.cpu_name);
    return dvc;
}

static void mag_cpu_release_interface(mag_idevice_t* ctx) {
    mag_cpu_device_t* cpu_dvc = ctx->impl;
    mag_cpu_destroy_device(cpu_dvc);
    (*mag_alloc)(ctx, 0, 0); /* Free all memory */
}

mag_idevice_t* mag_init_device_cpu(mag_context_t* ctx, const mag_device_desc_t* desc) {
    uint32_t hw_concurrency = mag_xmax(1, ctx->machine.cpu_virtual_cores);
    uint32_t num_threads = desc->cpu_thread_count;
    num_threads = num_threads ? num_threads : hw_concurrency;
    mag_idevice_t* dvc = mag_cpu_init_interface(ctx, num_threads);
    return dvc;
}

void mag_destroy_device_cpu(mag_idevice_t* dvc) {
    mag_cpu_release_interface(dvc);
}
