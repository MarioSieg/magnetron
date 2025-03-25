/*
** (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
**
**
** Implement CPU dispatch to specialized runtime kernels generated from magnetron_cpu_blas.inl and multithreading logic for parallel computations.
** Each CPU kernel has an optional pre- and post-kernel, which can be used to initialize and destroy temporary data for the op.
** A tensor can also have an init op, to fill it with a fixed value or with random numbers. The operation order is as follows:
** 1. Init OP (if any)
** 2. Pre Kernel (if any)
** 3. Primary Kernel
** 4. Post Kernel (if any)
*/

#include "magnetron_internal.h"

#include <math.h>

extern void mag_cpu_blas_specialization_fallback(mag_kernel_registry_t* kernels); /* Generic any CPU impl */

#if defined(__x86_64__) || defined(_M_X64) /* Specialized impls for x86-64 with runtime CPU detection */

typedef struct mag_amd64_blas_specialization {
    const char* name;
    uint64_t (*get_feature_permutation)(void);
    void (*inject_kernels)(mag_kernel_registry_t* kernels);
} mag_amd64_blas_specialization;

#define mag_amd64_blas_spec_decl(feat) \
    uint64_t mag_cpu_blas_specialization_amd64_v##feat##_features(void); \
    extern void mag_cpu_blas_specialization_amd64_v##feat(mag_kernel_registry_t* kernels)

#define mag_amd64_blas_spec_permute(feat) \
    (mag_amd64_blas_specialization) { \
        .name = "amd64-v."#feat, \
        .get_feature_permutation = &mag_cpu_blas_specialization_amd64_v##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_amd64_v##feat \
    }

mag_amd64_blas_spec_decl(4_5);
mag_amd64_blas_spec_decl(4);
mag_amd64_blas_spec_decl(3);
mag_amd64_blas_spec_decl(2_5);
mag_amd64_blas_spec_decl(2);

static bool mag_blas_detect_gen_optimal_spec(const mag_ctx_t* host_ctx, mag_kernel_registry_t* kernels) {
    const mag_amd64_blas_specialization mag_amd64_blas_specializations[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
        mag_amd64_blas_spec_permute(4_5),
        mag_amd64_blas_spec_permute(4),
        mag_amd64_blas_spec_permute(3),
        mag_amd64_blas_spec_permute(2_5),
        mag_amd64_blas_spec_permute(2),
    };

    uint64_t cap_avail = host_ctx->machine.amd64_cpu_caps;
    for (size_t i=0; i < sizeof(mag_amd64_blas_specializations)/sizeof(*mag_amd64_blas_specializations); ++i) { /* Find best blas spec for the host CPU */
        const mag_amd64_blas_specialization* spec = mag_amd64_blas_specializations+i;
        uint64_t cap_required = (*spec->get_feature_permutation)(); /* Get requires features */
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

#undef mag_amd64_blas_spec_permute
#undef mag_cpu_blas_spec_decl

#elif defined(__aarch64__) || defined(_M_ARM64)

typedef struct mag_arm64_blas_specialization {
    const char* name;
    uint64_t (*get_cap_permutation)(void);
    void (*inject_kernels)(mag_kernel_registry_t* kernels);
} mag_arm64_blas_specialization;

#define mag_arm64_blas_spec_decl(feat) \
    uint64_t mag_cpu_blas_specialization_arm64_v_##feat##_features(void); \
    extern void mag_cpu_blas_specialization_arm64_v_##feat(mag_kernel_registry_t* kernels)

#define mag_arm64_blas_spec_permute(feat) \
    (mag_arm64_blas_specialization) { \
        .name = "arm64-v."#feat, \
        .get_cap_permutation = &mag_cpu_blas_specialization_arm64_v_##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_arm64_v_##feat \
}

mag_arm64_blas_spec_decl(9);
mag_arm64_blas_spec_decl(8_2);

static bool mag_blas_detect_gen_optimal_spec(const mag_ctx_t* ctx, mag_kernel_registry_t* kernels) {
    const mag_arm64_blas_specialization mag_arm64_blas_specializations[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
        mag_arm64_blas_spec_permute(9),
        mag_arm64_blas_spec_permute(8_2),
    };

    uint64_t cap_avail = ctx->machine.arm64_cpu_caps;
    for (size_t i=0; i < sizeof(mag_arm64_blas_specializations)/sizeof(*mag_arm64_blas_specializations); ++i) { /* Find best blas spec for the host CPU */
        const mag_arm64_blas_specialization* spec = mag_arm64_blas_specializations+i;
        uint64_t cap_required = (*spec->get_cap_permutation)(); /* Get requires features */
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

static bool mag_blas_detect_optimal_specialization(const mag_ctx_t* ctx, mag_kernel_registry_t* kernels) {
    if (mag_likely(mag_blas_detect_gen_optimal_spec(ctx, kernels))) return true;
    mag_cpu_blas_specialization_fallback(kernels);
    return false; /* No spec used, fallback is active */
}

typedef struct mag_cpu_op_info_t {
    bool mt_support;
    double growth;
    int64_t threshold;
} mag_cpu_op_info_t;

static const mag_cpu_op_info_t mag_cpu_op_infos[MAG_OP__NUM] = {
    [MAG_OP_NOP]            = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_CLONE]          = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_VIEW]           = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_TRANSPOSE]      = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_PERMUTE]        = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_MEAN]           = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_MIN]            = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_MAX]            = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_SUM]            = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_ABS]            = {.mt_support = true,  .growth = 0.1, .threshold = 0},
    [MAG_OP_NEG]            = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_LOG]            = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SQR]            = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SQRT]           = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SIN]            = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_COS]            = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_STEP]           = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SOFTMAX]        = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SOFTMAX_DV]     = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SIGMOID]        = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SIGMOID_DV]     = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_HARD_SIGMOID]   = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SILU]           = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_SILU_DV]        = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_TANH]           = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_TANH_DV]        = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_RELU]           = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_RELU_DV]        = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_GELU]           = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_GELU_DV]        = {.mt_support = true,  .growth = 0.1, .threshold = 250000},
    [MAG_OP_ADD]            = {.mt_support = true,  .growth = 0.2, .threshold = 250000},
    [MAG_OP_SUB]            = {.mt_support = true,  .growth = 0.2, .threshold = 250000},
    [MAG_OP_MUL]            = {.mt_support = true,  .growth = 0.2, .threshold = 250000},
    [MAG_OP_DIV]            = {.mt_support = true,  .growth = 0.2, .threshold = 250000},
    [MAG_OP_ADDS]           = {.mt_support = true,  .growth = 0.2, .threshold = 250000},
    [MAG_OP_SUBS]           = {.mt_support = true,  .growth = 0.2, .threshold = 250000},
    [MAG_OP_MULS]           = {.mt_support = true,  .growth = 0.2, .threshold = 250000},
    [MAG_OP_DIVS]           = {.mt_support = true,  .growth = 0.2, .threshold = 250000},
    [MAG_OP_MATMUL]         = {.mt_support = true,  .growth = 3.0, .threshold =  10000},
};

typedef struct mag_worker_t mag_worker_t;
typedef struct mag_threadpool_t {
    mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile bool interrupt;   /* Interrupt flag, 1=stop */
    mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) uint64_t phase;            /* Current compute phase */
    mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) uint64_t num_completed;    /* Number of workers that have completed their work */
    mag_cond_var_t cv;                              /* Condition variable for thread wakeup */
    mag_mutex_t mtx;                                /* Mutex for synchronization */
    uint32_t num_allocated_workers;                 /* Number of intra-op workers allocated */
    uint32_t num_active_workers;                    /* Number of intra-op workers that are actively used in this compute step. */
    volatile mag_atomic_t num_workers_online;       /* Number of workers that are online */
    mag_worker_t* workers;                          /* Array of workers */
    const mag_kernel_registry_t* kernels;           /* Specialized compute kernel registry */
    mag_kernel_context_t* kernel_ctx;               /* Kernel context */
    mag_thread_sched_prio_t sched_prio;             /* Scheduling priority */
    mag_ctx_t* host_ctx;                                 /* Host context */
} mag_threadpool_t;

struct mag_worker_t {
    uint64_t phase;                         /* Current compute phase */
    mag_compute_payload_t payload;          /* Compute op payload */
    mag_prng_state_t prng;                  /* Thread local prng */
    mag_threadpool_t* pool;                 /* Host thread pool */
    bool is_async;                          /* True if worker is async (executed on a different thread)  */
    mag_thread_t thread;                    /* Thread handle */
} mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE);

typedef struct mag_cpu_device_t {
    mag_ctx_t* ctx;
    mag_threadpool_t* pool;             /* Thread pool. NULL if num_allocated_workers <= 1 */
    uint32_t num_allocated_workers;     /* Amount of worker thread used. if == 1 then single threaded mode and thread pool is not created */
    mag_kernel_registry_t kernels;      /* Compute kernels. Specialized by arch optimized version at boot (e.g. AVX, AVX512 etc..) */
    mag_kernel_context_t kernel_ctx;    /* Kernel context */
    mag_prng_state_t primary_prng;      /* Primary prng context. */
} mag_cpu_device_t;

/* Await signal to start work */
static bool mag_worker_await_work(mag_worker_t* worker, mag_threadpool_t* pool) {
    mag_mutex_lock(&pool->mtx);
    while (!(pool->interrupt || pool->phase > worker->phase)) /* Wait for work 🥱*/
        mag_cv_wait(&pool->cv, &pool->mtx);
    if (mag_unlikely(pool->interrupt)) { /* Exit if interrupted */
        mag_mutex_unlock(&pool->mtx);
        return false;
    }
    worker->phase = pool->phase;
    mag_mutex_unlock(&pool->mtx);
    return true;
}

/* Execute the operation on the current thread */
static void mag_worker_exec_thread_local(const mag_kernel_registry_t* kernels, mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    if (mag_unlikely(!payload->node)) return;
    mag_op_t op = payload->node->op;
    mag_init_op_t iop = payload->node->init_op;
    mag_dtype_t dtype = payload->node->dtype;
    mag_assert2(op >= 0 && op < MAG_OP__NUM);
    mag_assert2(iop >= 0 && iop < MAG_IOP__NUM);
    mag_assert2(dtype >= 0 && dtype < MAG_DTYPE__NUM);
    void (*kernel)(const mag_compute_payload_t*, mag_kernel_context_t* ctx)
        = payload->gra_eval == MAG_GRA_INIT
        ? kernels->init[iop][dtype] : payload->gra_eval == MAG_GRA_FWD
        ? kernels->fwd[op][dtype] : kernels->bwd[op][dtype];
    (*kernel)(payload, ctx);
    payload->node = NULL;
}

/* Execute the operation and broadcast completion if last chunk was done */
static void mag_worker_exec_and_broadcast(mag_threadpool_t* pool, const mag_kernel_registry_t* kernels, mag_compute_payload_t* payload, mag_kernel_context_t* ctx) {
    if (mag_likely(payload->thread_idx < pool->num_active_workers)) /* Execute the operation if we are an active thread. */
        mag_worker_exec_thread_local(kernels, payload, ctx);
    mag_mutex_lock(&pool->mtx);
    if (++pool->num_completed == pool->num_allocated_workers) /* If we are the last to finish, wake the main thread */
        mag_cv_broadcast(&pool->cv);
    mag_mutex_unlock(&pool->mtx);
}

/* Worker thread entry point */
static MAG_HOTPROC void* mag_worker_thread_exec_op(void* arg) {
    mag_worker_t* worker = arg;
    mag_threadpool_t* pool = worker->pool;
    mag_compute_payload_t* payload = &worker->payload;
    const mag_kernel_registry_t* kernels = pool->kernels;
    mag_kernel_context_t* ctx = pool->kernel_ctx;
    char name[32];
    snprintf(name, sizeof(name), "mag_worker_%" PRIx64, payload->thread_idx);
    mag_thread_set_name(name);
    /*mag_thread_set_prio(pool->sched_prio);*/
    mag_prng_init(&worker->prng, pool->host_ctx->prng_algo, mag_thread_id()^(uintptr_t)worker);
    mag_atomic_fetch_add(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    while (mag_likely(mag_worker_await_work(worker, pool)))  /* Main work loop: wait, work, signal status */
        mag_worker_exec_and_broadcast(pool, kernels, payload, ctx);
    mag_atomic_fetch_sub(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    return MAG_THREAD_RET_NONE;
}

/* Create thread pool and allocate threads */
static mag_threadpool_t* mag_threadpool_create(mag_ctx_t* host_ctx, uint32_t num_workers, const mag_kernel_registry_t* kernels, mag_kernel_context_t* ctx, mag_thread_sched_prio_t prio) { /* Create a thread pool */
    mag_threadpool_t* pool = mag_alloc_aligned(sizeof(*pool), __alignof(mag_threadpool_t));
    memset(pool, 0, sizeof(*pool));
    mag_worker_t* workers = mag_alloc_aligned(num_workers*sizeof(*workers), __alignof(mag_worker_t));
    memset(workers, 0, num_workers*sizeof(*workers));
    *pool = (mag_threadpool_t){
        .interrupt = false,
        .phase = 0,
        .num_completed = 0,
        .num_allocated_workers = num_workers,
        .num_active_workers = num_workers,
        .num_workers_online = 0,  /* Main thread as worker 0 */
        .workers = workers,
        .kernels = kernels,
        .kernel_ctx = ctx,
        .sched_prio = prio,
        .host_ctx = host_ctx
    };
    mag_cv_create(&pool->cv);
    mag_mutex_create(&pool->mtx);
    for (uint32_t ti=0; ti < num_workers; ++ti) { /* Initialize workers */
        mag_worker_t* worker = workers+ti;
        *worker = (mag_worker_t){
            .phase = 0,
            .payload = (mag_compute_payload_t){
                .thread_num = num_workers,
                .thread_idx = ti,
                .node = NULL,
                .gra_eval = MAG_GRA_FWD,
                .local_prng = NULL
            },
            .pool = pool,
            .is_async = ti != 0 /* Main thread is worker but without thread */
        };
        worker->payload.local_prng = &worker->prng;
        if (worker->is_async)
            mag_thread_create(&worker->thread, &mag_worker_thread_exec_op, workers+ti);
    }
    while (mag_atomic_load(&pool->num_workers_online, MAG_MO_SEQ_CST) != num_workers-1)  /* Wait for all workers to come online */
        mag_thread_yield();
    return pool;
}

/* Destroy thread pool */
static void mag_threadpool_destroy(mag_threadpool_t* pool) {
    mag_mutex_lock(&pool->mtx);
        pool->interrupt = true;
        ++pool->phase;
    mag_mutex_unlock(&pool->mtx);
    mag_cv_broadcast(&pool->cv); /* Wake up all workers to exit */
    while (mag_atomic_load(&pool->num_workers_online, MAG_MO_SEQ_CST))  /* Wait for all workers to exit */
        mag_thread_yield();
    for (uint32_t i=0; i < pool->num_allocated_workers; ++i) /* Join all worker threads */
        if (pool->workers[i].is_async)
            mag_thread_join(pool->workers[i].thread);
    mag_cv_destroy(&pool->cv);
    mag_mutex_destroy(&pool->mtx);
    mag_free_aligned(pool->workers);
    mag_free_aligned(pool);
}

/* Submits work payload and awakens all threads */
static void mag_threadpool_kickoff(mag_threadpool_t* pool, mag_tensor_t* node, mag_gra_eval_t gra_eval, uint32_t num_active_workers) {
    mag_mutex_lock(&pool->mtx);
    pool->num_active_workers = num_active_workers;
    for (uint32_t i=0; i < pool->num_allocated_workers; ++i) { /* Set up payload */
        mag_compute_payload_t* payload = &pool->workers[i].payload;
        payload->thread_num = num_active_workers;
        payload->node = node;
        payload->gra_eval = gra_eval;
    }
    ++pool->phase;
    pool->num_completed = 0; /* Reset completion counter */
    mag_mutex_unlock(&pool->mtx);
}

/* Blocks until all threads have completed their work */
static void mag_threadpool_barrier(mag_threadpool_t* pool) {
    mag_mutex_lock(&pool->mtx);
    while (pool->num_completed != pool->num_allocated_workers) /* Wait for all workers to finish */
        mag_cv_wait(&pool->cv, &pool->mtx);
    #ifdef MAG_DEBUG
        for (uint32_t i=0; i < pool->num_workers; ++i) /* Verify phases executed */
            mag_assert2(pool->workers[i].phase == pool->phase);
    #endif
    mag_mutex_unlock(&pool->mtx);
}

/* Execute an operator tensor on the CPU */
static MAG_HOTPROC void mag_threadpool_parallel_compute(mag_threadpool_t* pool, mag_tensor_t* node, mag_gra_eval_t gra_eval, uint32_t num_active_workers) {
    mag_assert2(pool != NULL);
    mag_threadpool_kickoff(pool, node, gra_eval, num_active_workers);                                     /* Kick off workers */
    mag_cv_broadcast(&pool->cv);                                                                        /* Wake up all workers */
    mag_worker_exec_and_broadcast(pool, pool->kernels, &pool->workers->payload, pool->kernel_ctx);      /* Main thread does work too */
    mag_threadpool_barrier(pool);                                                                       /* Wait for all workers to finish */
}

static uint32_t mag_cpu_dynamic_work_scaling(mag_cpu_device_t* dvc, mag_op_t op, int64_t numel);

static MAG_HOTPROC void mag_cpu_exec(mag_compute_device_t* dvc, mag_gra_eval_t gra_eval, mag_tensor_t* node) {
    mag_cpu_device_t* cpu_dvc = dvc->impl;
    const mag_kernel_registry_t* kernels = &cpu_dvc->kernels;
    mag_kernel_context_t* kctx = &cpu_dvc->kernel_ctx; /* Setup pre/post kernel context */
    kctx->node = node;
    kctx->alloced_threads = cpu_dvc->num_allocated_workers;
    uint32_t (*pre)(mag_kernel_context_t*) = gra_eval == MAG_GRA_FWD ? kernels->fwd_pre[node->op] : kernels->bwd_pre[node->op];      /* Fetch pre exec kernel */
    void (*post)(mag_kernel_context_t*) = gra_eval == MAG_GRA_BWD ? kernels->fwd_post[node->op] : kernels->bwd_post[node->op];       /* Fetch post exec kernel */
    uint32_t intraop_workers = gra_eval == MAG_GRA_INIT ? 0 : pre ? (*pre)(kctx) : mag_cpu_dynamic_work_scaling(cpu_dvc, node->op, node->numel);   /* Use thread count recommended by pre-kernel or compute general thread count heuristic. */
    if (intraop_workers <= 1) { /* Main thread does the work (single threaded mode). */
        mag_compute_payload_t payload = {
            .node = node,
            .thread_idx = 0,
            .thread_num = 1,
            .gra_eval = gra_eval,
            .local_prng = &cpu_dvc->primary_prng
        };
        mag_worker_exec_thread_local(&cpu_dvc->kernels, &payload, kctx);
        goto epilogue;
    }
    mag_threadpool_parallel_compute(cpu_dvc->pool, node, gra_eval, intraop_workers); /* Multithreaded exec + barrier */
    epilogue: if (post) (*post)(kctx); /* Post-exec */
}

static void mag_cpu_exec_init(mag_compute_device_t* dvc, mag_tensor_t* node) {
    mag_cpu_exec(dvc, MAG_GRA_INIT, node);
}

static void mag_cpu_exec_fwd(mag_compute_device_t* dvc, mag_tensor_t* node) {
    mag_cpu_exec(dvc, MAG_GRA_FWD, node);
}

static void mag_cpu_exec_bwd(mag_compute_device_t* dvc, mag_tensor_t* node) {
    mag_cpu_exec(dvc, MAG_GRA_BWD, node);
}

static void mag_cpu_buf_broadcast(mag_storage_buffer_t* sto, size_t offs, const void* src, size_t stride) {
    mag_assert2(sto->base+offs <= sto->base+sto->size);
    switch (stride) {
        case 1: {
            uint8_t* p = (uint8_t*)(sto->base+offs);
            memset(p, *(uint8_t*)src, sto->size);
        } return;
        case 2: {
            uint16_t* p = (uint16_t*)(sto->base+offs);
            const uint16_t* end = p+(sto->size/sizeof(*p));
            uint16_t x = *(uint16_t*)src;
            for (; p < end; ++p) *p = x;
        } return;
        case 4: {
            uint32_t* p = (uint32_t*)(sto->base+offs);
            const uint32_t* end = p+(sto->size/sizeof(*p));
            uint32_t x = *(uint32_t*)src;
            for (; p < end; ++p) *p = x;
        } return;
        default: {
            uint8_t* p = (uint8_t*)(sto->base+offs);
            const uint8_t* end = p+sto->size;
            while (p < end) {
                memcpy(p, src, stride);
                p += stride;
            }
        }  return;
    }
}

mag_static_assert(sizeof(mag_e8m23_t) == 4);

static void mag_cpu_transfer(mag_storage_buffer_t* sto, mag_transfer_dir_t dir, mag_transfer_op_t op, size_t offs, void* inout, size_t inout_size) {
    mag_assert2(inout && inout_size);
    uintptr_t base = sto->base;
    mag_assert2(offs <= sto->size);     /* Validate that range is within device memory. */
    if (sto->dtype == MAG_DTYPE_E8M23 && op == MAG_TRANSFER_OP_CVT_E8M23)   /* If both are same dtype already, no conversion is needed. */
        goto direct_memcpy;
    switch (op) {
        case MAG_TRANSFER_OP_CPY: direct_memcpy: {      /* Copy raw bytes. */
            mag_assert2(inout_size <= sto->size-offs);  /* Check that byte range is valid. */
            switch (dir) {
                case MAG_TRANSFER_DIR_H2D: memcpy((void*)(base+offs), inout, inout_size); return; /* Host -> Device */
                case MAG_TRANSFER_DIR_D2H: memcpy(inout, (void*)(base+offs), inout_size); return; /* Device -> Host */
                default: mag_panic("invalid transfer dir"); return;
            }
        } return;
        case MAG_TRANSFER_OP_CVT_E8M23: {                       /* Convert to/from float. */
            mag_assert2((inout_size&3) == 0);                   /* Must be float array byte size (rem 4). */
            mag_assert2(((uintptr_t)inout&3) == 0);             /* Must be aligned to 4. */
            int64_t nf32 = (int64_t)inout_size>>2;              /* Number of inout f32s */
            int64_t nb = nf32*(int64_t)sto->granularity;        /* Number of bytes on device */
            mag_assert2(nb && offs + nb <= sto->size);          /* At least 1 byte must be transferred + bounds check. */
            uintptr_t pa = base + offs;
            uintptr_t pb = pa + nb;
            mag_assert2(pa >= base);                            /* Bounds check memory interval */
            mag_assert2(pb <= base + sto->size);
            uint8_t* dev_ptr = (uint8_t*)pa;
            mag_cpu_device_t* cpu_dvc = sto->host->impl;
            void (*vcast)(int64_t, const void*, mag_dtype_t, void*, mag_dtype_t) = cpu_dvc->kernels.vector_cast;
            switch (dir) { /* Vector cast ranges */
                case MAG_TRANSFER_DIR_H2D: (*vcast)(nb, inout, MAG_DTYPE_E8M23, dev_ptr, sto->dtype); return;
                case MAG_TRANSFER_DIR_D2H: (*vcast)(nb, dev_ptr, sto->dtype, inout, MAG_DTYPE_E8M23); return;
                default: mag_panic("invalid transfer dir"); return;
            }
        } return;
        default: mag_panic("invalid transfer op"); return;
    }
}

/* Align CPU buffer to cache line size, which should also be satisfy alignment requirements for SSE, AVX and AVX512 on x86-64. */
#define MAG_CPU_BUF_ALIGN MAG_DESTRUCTIVE_INTERFERENCE_SIZE
mag_static_assert((MAG_CPU_BUF_ALIGN & 15)==0);
mag_static_assert((MAG_CPU_BUF_ALIGN & 31)==0);
mag_static_assert((MAG_CPU_BUF_ALIGN & 63)==0);

static void mag_cpu_alloc_storage(mag_compute_device_t* host, mag_storage_buffer_t* out, size_t size, mag_dtype_t dtype) {
    mag_assert2(size);
    void* block = mag_alloc_aligned(size, MAG_CPU_BUF_ALIGN);
    *out = (mag_storage_buffer_t){ /* Set up storage buffer. */
        .base = (uintptr_t)block,
        .size = size,
        .alignment = MAG_CPU_BUF_ALIGN,
        .dtype = dtype,
        .granularity = mag_dtype_meta_of(dtype)->size,
        .host = host,
        .broadcast = &mag_cpu_buf_broadcast,
        .transfer = &mag_cpu_transfer
    };
}

static void mag_cpu_free_storage(mag_compute_device_t* dvc, mag_storage_buffer_t* buf) {
    mag_free_aligned((void*)buf->base);
    memset(buf, 0, sizeof(*buf)); /* Set to zero. */
}

static mag_cpu_device_t* mag_cpu_init_device(mag_ctx_t* ctx, uint32_t num_threads) {
    mag_thread_sched_prio_t sched_prio = MAG_THREAD_SCHED_PRIO_HIGH;
    mag_cpu_device_t* dvc = (*mag_alloc)(NULL, sizeof(*dvc));
    memset(dvc, 0, sizeof(*dvc));
    *dvc = (mag_cpu_device_t) {
        .ctx = ctx,
        .pool = NULL,
        .num_allocated_workers = 0,
        .kernels = {},
        .kernel_ctx = {},
        .primary_prng = {}
    };
    mag_blas_detect_optimal_specialization(ctx, &dvc->kernels);
    mag_prng_init(&dvc->primary_prng, ctx->prng_algo, num_threads);
    if (num_threads > 1) {
        dvc->pool = mag_threadpool_create(ctx, num_threads, &dvc->kernels, &dvc->kernel_ctx, sched_prio);
        dvc->num_allocated_workers = num_threads;
    }
    return dvc;
}

/*
** Computes how many workers to use for intra-op parallelism depending on the number of elements.
** A logarithmic scaling is used, see: https://www.desmos.com/calculator/xiunrskpwu
** TODO: This can be improved by using a more sophisticated heuristic and a benchmarked, numerical approach.
*/
static uint32_t mag_cpu_dynamic_work_scaling(mag_cpu_device_t* dvc, mag_op_t op, int64_t numel) {
    const mag_cpu_op_info_t* info = mag_cpu_op_infos+op;
    if (!dvc->pool || !info->mt_support || numel < info->threshold) return 1;       /* Use a single worker (main thread). */
    numel -= info->threshold;                                                       /* Saturate threshold */
    uint32_t workers = (uint32_t)ceil(info->growth * log2((double)numel));       /* Logarithmic scaling */
    workers = mag_xmin(dvc->num_allocated_workers, mag_xmax(1, workers));
    return workers;
}

static void mag_cpu_destroy_device(mag_cpu_device_t* dvc) {
    if (dvc->pool)
        mag_threadpool_destroy(dvc->pool);
    (*mag_alloc)(dvc, 0);
}

static mag_compute_device_t* mag_cpu_init_interface(mag_ctx_t* ctx, uint32_t num_threads) {
    mag_cpu_device_t* cpu_dvc = mag_cpu_init_device(ctx, num_threads);
    mag_compute_device_t* dvc = (*mag_alloc)(NULL, sizeof(*dvc));
    *dvc = (mag_compute_device_t){ /* Initialize device interface */
        .name = "CPU",
        .impl = cpu_dvc,
        .is_async = false,
        .type = MAG_COMPUTE_DEVICE_TYPE_CPU,
        .eager_exec_init = &mag_cpu_exec_init,
        .eager_exec_fwd = &mag_cpu_exec_fwd,
        .eager_exec_bwd = &mag_cpu_exec_bwd,
        .alloc_storage = &mag_cpu_alloc_storage,
        .free_storage = &mag_cpu_free_storage,
    };
    snprintf(dvc->name, sizeof(dvc->name), "%s", ctx->machine.cpu_name);
    return dvc;
}

static void mag_cpu_release_interface(mag_compute_device_t* ctx) {
    mag_cpu_device_t* cpu_dvc = ctx->impl;
    mag_cpu_destroy_device(cpu_dvc);
    (*mag_alloc)(ctx, 0); /* Free all memory */
}

mag_compute_device_t* mag_init_device_cpu(mag_ctx_t* ctx, const mag_device_descriptor_t* desc) {
    uint32_t hw_concurrency = mag_xmax(1, ctx->machine.cpu_virtual_cores);
    uint32_t num_threads = desc->thread_count;
    num_threads = num_threads ? num_threads : hw_concurrency;
    mag_compute_device_t* dvc = mag_cpu_init_interface(ctx, num_threads);
    return dvc;
}

void mag_destroy_device_cpu(mag_compute_device_t* dvc) {
    mag_cpu_release_interface(dvc);
}
