/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
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

extern void mag_cpu_blas_specialization_fallback(mag_CPUKernelRegistry* kernels); /* Generic any CPU impl */

#if defined(__x86_64__) || defined(_M_X64) /* Specialized impls for x86-64 with runtime CPU detection */

typedef struct mag_AMD64BLASSpec {
    const char* name;
    uint64_t (*get_feature_permutation)(void);
    void (*inject_kernels)(mag_CPUKernelRegistry* kernels);
} mag_AMD64BLASSpec;

#define mag_amd64_blas_spec_decl(feat) \
    uint64_t mag_cpu_blas_specialization_amd64_v##feat##_features(void); \
    extern void mag_cpu_blas_specialization_amd64_v##feat(mag_CPUKernelRegistry* kernels)

#define mag_amd64_blas_spec_permute(feat) \
    (mag_AMD64BLASSpec) { \
        .name = "amd64-v."#feat, \
        .get_feature_permutation = &mag_cpu_blas_specialization_amd64_v##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_amd64_v##feat \
    }

mag_amd64_blas_spec_decl(4_5);
mag_amd64_blas_spec_decl(4);
mag_amd64_blas_spec_decl(3);
mag_amd64_blas_spec_decl(2_5);
mag_amd64_blas_spec_decl(2);

static bool mag_blas_detect_gen_optimal_spec(const mag_Context* host_ctx, mag_CPUKernelRegistry* kernels) {
    const mag_AMD64BLASSpec mag_AMD64BLASSpecs[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
        mag_amd64_blas_spec_permute(4_5),
        mag_amd64_blas_spec_permute(4),
        mag_amd64_blas_spec_permute(3),
        mag_amd64_blas_spec_permute(2_5),
        mag_amd64_blas_spec_permute(2),
    };

    uint64_t cap_avail = host_ctx->machine.amd64_cpu_caps;
    for (size_t i=0; i < sizeof(mag_AMD64BLASSpecs)/sizeof(*mag_AMD64BLASSpecs); ++i) { /* Find best blas spec for the host CPU */
        const mag_AMD64BLASSpec* spec = mag_AMD64BLASSpecs+i;
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

typedef struct mag_ARM64BLASSpec {
    const char* name;
    uint64_t (*get_cap_permutation)(void);
    void (*inject_kernels)(mag_CPUKernelRegistry* kernels);
} mag_ARM64BLASSpec;

#define mag_arm64_blas_spec_decl(feat) \
    uint64_t mag_cpu_blas_specialization_arm64_v_##feat##_features(void); \
    extern void mag_cpu_blas_specialization_arm64_v_##feat(mag_CPUKernelRegistry* kernels)

#define mag_arm64_blas_spec_permute(feat) \
    (mag_ARM64BLASSpec) { \
        .name = "arm64-v."#feat, \
        .get_cap_permutation = &mag_cpu_blas_specialization_arm64_v_##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_arm64_v_##feat \
}

mag_arm64_blas_spec_decl(9);
mag_arm64_blas_spec_decl(8_2);

static bool mag_blas_detect_gen_optimal_spec(const mag_Context* ctx, mag_CPUKernelRegistry* kernels) {
    const mag_ARM64BLASSpec mag_ARM64BLASSpecs[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
        mag_arm64_blas_spec_permute(9),
        mag_arm64_blas_spec_permute(8_2),
    };

    uint64_t cap_avail = ctx->machine.arm64_cpu_caps;
    for (size_t i=0; i < sizeof(mag_ARM64BLASSpecs)/sizeof(*mag_ARM64BLASSpecs); ++i) { /* Find best blas spec for the host CPU */
        const mag_ARM64BLASSpec* spec = mag_ARM64BLASSpecs+i;
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

static bool mag_blas_detect_optimal_specialization(const mag_Context* ctx, mag_CPUKernelRegistry* kernels) {
    if (mag_likely(mag_blas_detect_gen_optimal_spec(ctx, kernels))) return true;
    mag_cpu_blas_specialization_fallback(kernels);
    return false; /* No spec used, fallback is active */
}

typedef struct mag_WorkerThread mag_WorkerThread;
typedef struct mag_ThreadPool {
    mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) volatile bool interrupt;   /* Interrupt flag, 1=stop */
    mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) uint64_t phase;            /* Current compute phase */
    mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE) uint64_t num_completed;    /* Number of workers that have completed their work */
    mag_CondVar cv;                              /* Condition variable for thread wakeup */
    mag_Mutex mtx;                                /* Mutex for synchronization */
    uint32_t num_allocated_workers;                 /* Number of intra-op workers allocated */
    uint32_t num_active_workers;                    /* Number of intra-op workers that are actively used in this compute step. */
    volatile mag_Atomic64 num_workers_online;       /* Number of workers that are online */
    mag_WorkerThread* workers;                          /* Array of workers */
    const mag_CPUKernelRegistry* kernels;           /* Specialized compute kernel registry */
    mag_ThreadPrio sched_prio;             /* Scheduling priority */
    mag_Context* host_ctx;                            /* Host context */
} mag_ThreadPool;

struct mag_WorkerThread {
    uint64_t phase;                         /* Current compute phase */
    mag_CPUKernelPayload payload;          /* Compute op payload */
    mag_PRNGState prng;                  /* Thread local prng */
    mag_ThreadPool* pool;                 /* Host thread pool */
    bool is_async;                          /* True if worker is async (executed on a different thread)  */
    mag_Thread thread;                    /* Thread handle */
} mag_alignas(MAG_DESTRUCTIVE_INTERFERENCE_SIZE);

typedef struct mag_CPUComputeDeviceImpl {
    mag_Context* ctx;
    mag_ThreadPool* pool;               /* Thread pool. NULL if num_allocated_workers <= 1 */
    uint32_t num_allocated_workers;     /* Amount of worker thread used. if == 1 then single threaded mode and thread pool is not created */
    mag_CPUKernelRegistry kernels;      /* Compute kernels. Specialized by arch optimized version at boot (e.g. AVX, AVX512 etc..) */
    mag_PRNGState primary_prng;         /* Primary prng context. */
} mag_CPUComputeDeviceImpl;

/* Await signal to start work */
static bool mag_worker_await_work(mag_WorkerThread* worker, mag_ThreadPool* pool) {
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
static void mag_worker_exec_thread_local(const mag_CPUKernelRegistry* kernels, mag_CPUKernelPayload* payload) {
    if (mag_unlikely(!payload->node)) return;
    mag_Operator op = payload->node->op;
    mag_InitOperator iop = payload->node->init_op;
    mag_DType dtype = *payload->node->op_inputs ? (*payload->node->op_inputs)->dtype : payload->node->dtype;
    mag_assert2(op >= 0 && op < MAG_OP__NUM);
    mag_assert2(iop >= 0 && iop < MAG_IOP__NUM);
    mag_assert2(dtype >= 0 && dtype < MAG_DTYPE__NUM);
    void (*kernel)(const mag_CPUKernelPayload*) = payload->stage == MAG_STAGE_INIT ? kernels->init[iop][dtype] : kernels->eval[op][dtype];
    mag_assert(kernel, "no kernel found for op '%s' (iop #%d) with dtype %s", mag_op_meta_of(op)->mnemonic, iop, mag_dtype_meta_of(dtype)->name);
    (*kernel)(payload);
    payload->node = NULL;
}

/* Execute the operation and broadcast completion if last chunk was done */
static void mag_worker_exec_and_broadcast(mag_ThreadPool* pool, const mag_CPUKernelRegistry* kernels, mag_CPUKernelPayload* payload) {
    if (mag_likely(payload->thread_idx < pool->num_active_workers)) /* Execute the operation if we are an active thread. */
        mag_worker_exec_thread_local(kernels, payload);
    mag_mutex_lock(&pool->mtx);
    if (++pool->num_completed == pool->num_allocated_workers) /* If we are the last to finish, wake the main thread */
        mag_cv_broadcast(&pool->cv);
    mag_mutex_unlock(&pool->mtx);
}

/* Worker thread entry point */
static MAG_HOTPROC void* mag_worker_thread_exec_op(void* arg) {
    mag_WorkerThread* worker = arg;
    mag_ThreadPool* pool = worker->pool;
    mag_CPUKernelPayload* payload = &worker->payload;
    const mag_CPUKernelRegistry* kernels = pool->kernels;
    char name[32];
    snprintf(name, sizeof(name), "mag_worker_%" PRIx64, payload->thread_idx);
    mag_thread_set_name(name);
    /*mag_thread_set_prio(pool->sched_prio);*/
    mag_prng_seed(&worker->prng, pool->host_ctx->prng_algo, mag_thread_id()^(uintptr_t)worker);
    mag_atomic_fetch_add(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    while (mag_likely(mag_worker_await_work(worker, pool)))  /* Main work loop: wait, work, signal status */
        mag_worker_exec_and_broadcast(pool, kernels, payload);
    mag_atomic_fetch_sub(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    return MAG_THREAD_RET_NONE;
}

/* Create thread pool and allocate threads */
static mag_ThreadPool* mag_threadpool_create(mag_Context* host_ctx, uint32_t num_workers, const mag_CPUKernelRegistry* kernels, mag_ThreadPrio prio) { /* Create a thread pool */
    mag_ThreadPool* pool = mag_alloc_aligned(sizeof(*pool), __alignof(mag_ThreadPool));
    memset(pool, 0, sizeof(*pool));
    mag_WorkerThread* workers = mag_alloc_aligned(num_workers*sizeof(*workers), __alignof(mag_WorkerThread));
    memset(workers, 0, num_workers*sizeof(*workers));
    *pool = (mag_ThreadPool){
        .interrupt = false,
        .phase = 0,
        .num_completed = 0,
        .num_allocated_workers = num_workers,
        .num_active_workers = num_workers,
        .num_workers_online = 0,  /* Main thread as worker 0 */
        .workers = workers,
        .kernels = kernels,
        .sched_prio = prio,
        .host_ctx = host_ctx
    };
    mag_cv_create(&pool->cv);
    mag_mutex_create(&pool->mtx);
    for (uint32_t ti=0; ti < num_workers; ++ti) { /* Initialize workers */
        mag_WorkerThread* worker = workers+ti;
        *worker = (mag_WorkerThread){
            .phase = 0,
            .payload = (mag_CPUKernelPayload){
                .thread_num = num_workers,
                .thread_idx = ti,
                .node = NULL,
                .stage = MAG_STAGE_EVAL,
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
static void mag_threadpool_destroy(mag_ThreadPool* pool) {
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
static void mag_threadpool_kickoff(mag_ThreadPool* pool, mag_Tensor* node, mag_ExecStage stage, uint32_t num_active_workers) {
    mag_mutex_lock(&pool->mtx);
    pool->num_active_workers = num_active_workers;
    for (uint32_t i=0; i < pool->num_allocated_workers; ++i) { /* Set up payload */
        mag_CPUKernelPayload* payload = &pool->workers[i].payload;
        payload->thread_num = num_active_workers;
        payload->node = node;
        payload->stage = stage;
    }
    ++pool->phase;
    pool->num_completed = 0; /* Reset completion counter */
    mag_mutex_unlock(&pool->mtx);
}

/* Blocks until all threads have completed their work */
static void mag_threadpool_barrier(mag_ThreadPool* pool) {
    mag_mutex_lock(&pool->mtx);
    while (pool->num_completed != pool->num_allocated_workers) /* Wait for all workers to finish */
        mag_cv_wait(&pool->cv, &pool->mtx);
    #ifdef MAG_DEBUG
        for (uint32_t i=0; i < pool->num_active_workers; ++i) /* Verify phases executed */
            mag_assert2(pool->workers[i].phase == pool->phase);
    #endif
    mag_mutex_unlock(&pool->mtx);
}

/* Execute an operator tensor on the CPU */
static MAG_HOTPROC void mag_threadpool_parallel_compute(mag_ThreadPool* pool, mag_Tensor* node, mag_ExecStage stage, uint32_t num_active_workers) {
    mag_assert2(pool != NULL);
    mag_threadpool_kickoff(pool, node, stage, num_active_workers);               /* Kick off workers */
    mag_cv_broadcast(&pool->cv);                                                    /* Wake up all workers */
    mag_worker_exec_and_broadcast(pool, pool->kernels, &pool->workers->payload);    /* Main thread does work too */
    mag_threadpool_barrier(pool);                                                   /* Wait for all workers to finish */
}

static uint32_t mag_cpu_dynamic_work_scaling(mag_CPUComputeDeviceImpl* dvc, mag_Operator op, int64_t numel);

static MAG_HOTPROC void mag_cpu_exec(mag_IComputeDevice* dvc, mag_ExecStage stage, mag_Tensor* node) {
    mag_CPUComputeDeviceImpl* cpu_dvc = dvc->impl;
    uint32_t intraop_workers = stage == MAG_STAGE_INIT ? 0 : mag_cpu_dynamic_work_scaling(cpu_dvc, node->op, node->numel);   /* Use thread count recommended by pre-kernel or compute general thread count heuristic. */
    if (intraop_workers <= 1) { /* Main thread does the work (single threaded mode). */
        mag_CPUKernelPayload payload = {
            .node = node,
            .thread_idx = 0,
            .thread_num = 1,
            .stage = stage,
            .local_prng = &cpu_dvc->primary_prng
        };
        mag_worker_exec_thread_local(&cpu_dvc->kernels, &payload);
        return; /* We're done */
    }
    mag_threadpool_parallel_compute(cpu_dvc->pool, node, stage, intraop_workers); /* Multithreaded exec + barrier */
}

static void mag_cpu_exec_init(mag_IComputeDevice* dvc, mag_Tensor* node) {
    mag_cpu_exec(dvc, MAG_STAGE_INIT, node);
}

static void mag_cpu_exec_fwd(mag_IComputeDevice* dvc, mag_Tensor* node) {
    mag_cpu_exec(dvc, MAG_STAGE_EVAL, node);
}

static void mag_cpu_buf_broadcast(mag_IStorageBuffer* sto, size_t offs, const void* src, size_t stride) {
    mag_assert2(offs <= sto->size);
    size_t len = sto->size - offs;
    if (mag_unlikely(!len)) return;
    uint8_t* dst8 = (uint8_t*)sto->base+offs;
    switch (stride) {
        case 1: memset(dst8, *(const uint8_t*)src, len); return;
        case 2: {
            mag_assert2(!(offs&1) && !(len&1));     /* Offsets and len must be aligned to granularity */
            uint16_t v = *(const uint16_t*)src;
            uint16_t* p = (uint16_t*)dst8;
            uint16_t* end = p + (len/sizeof(v));
            for (; p < end; ++p) *p = v;
            return;
        }
        case 4: {
            mag_assert2(!(offs&3) && !(len&3));     /* Offsets and len must be aligned to granularity */
            uint32_t v = *(const uint32_t*)src;
            uint32_t* p = (uint32_t*)dst8;
            uint32_t* end = p + (len/sizeof(v));
            for (; p < end; ++p) *p = v;
            return;
        }
        default: {
            while (len >= stride) {
                memcpy(dst8, src, stride);
                dst8 += stride;
                len -= stride;
            }
            if (len) memcpy(dst8, src, len);  /* handle tail */
        }
    }
}

mag_static_assert(sizeof(mag_E8M23) == 4);

static void mag_cpu_transfer(mag_IStorageBuffer* sto, mag_TransferDir dir, mag_TransferOP op, size_t offs, void* inout, size_t inout_size) {
    mag_assert2(inout && inout_size);
    uintptr_t base = sto->base;
    mag_assert2(offs <= sto->size);     /* Validate that range is within device memory. */
    if (sto->dtype == MAG_DTYPE_E8M23 && op == MAG_TRANSFER_OP_CONVERT_E8M23)   /* If both are same dtype already, no conversion is needed. */
        goto direct_memcpy;
    switch (op) {
        case MAG_TRANSFER_OP_COPY: direct_memcpy: {      /* Copy raw bytes. */
            mag_assert2(inout_size <= sto->size-offs);  /* Check that byte range is valid. */
            switch (dir) {
                case MAG_TRANSFER_DIR_H2D: memcpy((void*)(base+offs), inout, inout_size); return; /* Host -> Device */
                case MAG_TRANSFER_DIR_D2H: memcpy(inout, (void*)(base+offs), inout_size); return; /* Device -> Host */
                default: mag_panic("invalid transfer dir"); return;
            }
        } return;
        case MAG_TRANSFER_OP_CONVERT_E8M23: {                       /* Convert to/from float. */
            mag_assert2((inout_size&3) == 0);                       /* Must be float array byte size (rem 4). */
            mag_assert2(((uintptr_t)inout&3) == 0);                 /* Must be aligned to 4. */
            size_t host_nb = inout_size;                            /* Total host bytes (4 bytes per element) */
            size_t device_nb = (host_nb>>2)*sto->granularity;       /* Total device bytes (e.g. 2 bytes per element for fp16) */
            mag_assert2(device_nb && offs+device_nb <= sto->size);
            uintptr_t pa = base + offs;
            void* inout2 = (void*)pa;
            mag_CPUComputeDeviceImpl* cpu_dvc = sto->host->impl;
            void (*vcast)(size_t, const void*, mag_DType, void*, mag_DType) = cpu_dvc->kernels.vector_cast;
            switch (dir) { /* Vector cast ranges */
                case MAG_TRANSFER_DIR_H2D: (*vcast)(host_nb, inout, MAG_DTYPE_E8M23, inout2, sto->dtype); return;
                case MAG_TRANSFER_DIR_D2H: (*vcast)(device_nb, inout2, sto->dtype, inout, MAG_DTYPE_E8M23); return;
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

static void mag_cpu_storage_dtor(void* self) {
    mag_IStorageBuffer* buf = self;
    mag_Context* ctx = buf->ctx;
    mag_assert(ctx->num_storages > 0, "double freed storage");
    --ctx->num_storages;
    mag_free_aligned((void*)buf->base);
    mag_fixed_intrusive_pool_free(&ctx->storage_pool, buf);
}

static void mag_cpu_alloc_storage(mag_IComputeDevice* host, mag_IStorageBuffer** out, size_t size, mag_DType dtype) {
    mag_Context* ctx = host->ctx;
    void* block = mag_alloc_aligned(size, MAG_CPU_BUF_ALIGN);
    *out = mag_fixed_intrusive_pool_malloc(&ctx->storage_pool);
    **out = (mag_IStorageBuffer){ /* Set up storage buffer. */
        .ctx = ctx,
        .rc_control = mag_rc_control_init(*out, &mag_cpu_storage_dtor),
        .base = (uintptr_t)block,
        .size = size,
        .alignment = MAG_CPU_BUF_ALIGN,
        .dtype = dtype,
        .granularity = mag_dtype_meta_of(dtype)->size,
        .host = host,
        .broadcast = &mag_cpu_buf_broadcast,
        .transfer = &mag_cpu_transfer
    };
    ++host->ctx->num_storages;
}

static mag_CPUComputeDeviceImpl* mag_cpu_init_device(mag_Context* ctx, uint32_t num_threads) {
    mag_ThreadPrio sched_prio = MAG_THREAD_PRIO_HIGH;
    mag_CPUComputeDeviceImpl* dvc = (*mag_alloc)(NULL, sizeof(*dvc));
    memset(dvc, 0, sizeof(*dvc));
    *dvc = (mag_CPUComputeDeviceImpl) {
        .ctx = ctx,
        .pool = NULL,
        .num_allocated_workers = 0,
        .kernels = {},
        .primary_prng = {}
    };
    mag_blas_detect_optimal_specialization(ctx, &dvc->kernels);
    mag_prng_seed(&dvc->primary_prng, ctx->prng_algo, num_threads);
    if (num_threads > 1) {
        dvc->pool = mag_threadpool_create(ctx, num_threads, &dvc->kernels, sched_prio);
        dvc->num_allocated_workers = num_threads;
    }
    return dvc;
}

/*
** Computes how many workers to use for intra-op parallelism depending on the number of elements.
** A logarithmic scaling is used, see: https://www.desmos.com/calculator/xiunrskpwu
** TODO: This can be improved by using a more sophisticated heuristic and a benchmarked, numerical approach.
*/
static uint32_t mag_cpu_dynamic_work_scaling(mag_CPUComputeDeviceImpl* dvc, mag_Operator op, int64_t numel) {
    const mag_OPMetadata* meta = mag_op_meta_of(op);
    if (!dvc->pool || !(meta->flags & MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING) || numel < meta->cpu.thread_treshold)  /* Use a single worker (main thread). */
        return 1;
    numel -= meta->cpu.thread_treshold;                                                             /* Saturate threshold */
    uint32_t workers = (uint32_t)ceil(meta->cpu.thread_growth * log2((mag_E11M52)numel));         /* Logarithmic scaling */
    workers = mag_xmin(dvc->num_allocated_workers, mag_xmax(1, workers));
    return workers;
}

static void mag_cpu_destroy_device(mag_CPUComputeDeviceImpl* dvc) {
    if (dvc->pool)
        mag_threadpool_destroy(dvc->pool);
    (*mag_alloc)(dvc, 0);
}

static mag_IComputeDevice* mag_cpu_init_interface(mag_Context* ctx, uint32_t num_threads) {
    mag_CPUComputeDeviceImpl* cpu_dvc = mag_cpu_init_device(ctx, num_threads);
    mag_IComputeDevice* dvc = (*mag_alloc)(NULL, sizeof(*dvc));
    *dvc = (mag_IComputeDevice){ /* Initialize device interface */
        .ctx = ctx,
        .name = "CPU",
        .impl = cpu_dvc,
        .is_async = false,
        .type = MAG_COMPUTE_DEVICE_TYPE_CPU,
        .eager_exec_init = &mag_cpu_exec_init,
        .eager_exec_fwd = &mag_cpu_exec_fwd,
        .alloc_storage = &mag_cpu_alloc_storage,
    };
    snprintf(dvc->name, sizeof(dvc->name), "%s", ctx->machine.cpu_name);
    return dvc;
}

static void mag_cpu_release_interface(mag_IComputeDevice* ctx) {
    mag_CPUComputeDeviceImpl* cpu_dvc = ctx->impl;
    mag_cpu_destroy_device(cpu_dvc);
    (*mag_alloc)(ctx, 0); /* Free all memory */
}

mag_IComputeDevice* mag_init_device_cpu(mag_Context* ctx, const mag_ComputeDeviceDesc* desc) {
    uint32_t hw_concurrency = mag_xmax(1, ctx->machine.cpu_virtual_cores);
    uint32_t num_threads = desc->cpu_thread_count;
    num_threads = num_threads ? num_threads : hw_concurrency;
    mag_IComputeDevice* dvc = mag_cpu_init_interface(ctx, num_threads);
    return dvc;
}

void mag_destroy_device_cpu(mag_IComputeDevice* dvc) {
    mag_cpu_release_interface(dvc);
}
