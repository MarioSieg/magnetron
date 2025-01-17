/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "magnetron_internal.h"

#include <math.h>

extern void mag_cpu_blas_specialization_fallback(mag_kernel_registry_t* kernels); /* Generic any CPU impl */

#if defined(__x86_64__) || defined(_M_X64) /* Specialized impls for x86-64 with runtime CPU detection */

typedef struct mag_amd64_blas_specialization {
    const char* name;
    const mag_x86_64_feature_t* (*get_feature_permutation)(size_t* out_num);
    void (*inject_kernels)(mag_kernel_registry_t* kernels);
} mag_amd64_blas_specialization;

#define mag_cpu_blas_spec_decl(feat) \
    const mag_x86_64_feature_t* mag_cpu_blas_specialization_amd64_##feat##_features(size_t* out_num); \
    extern void mag_cpu_blas_specialization_amd64_##feat(mag_kernel_registry_t* kernels)

#define mag_amd64_blas_spec_permute(feat) \
    (mag_amd64_blas_specialization) { \
        .name = "amd64_"#feat, \
        .get_feature_permutation = &mag_cpu_blas_specialization_amd64_##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_amd64_##feat \
    }

mag_cpu_blas_spec_decl(avx512f);
mag_cpu_blas_spec_decl(avx2);
mag_cpu_blas_spec_decl(avx);
mag_cpu_blas_spec_decl(sse41);

static const mag_amd64_blas_specialization mag_amd64_blas_specializations[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
    mag_amd64_blas_spec_permute(avx512f),
    mag_amd64_blas_spec_permute(avx2),
    mag_amd64_blas_spec_permute(avx),
    mag_amd64_blas_spec_permute(sse41),
};

static bool mag_blas_detect_gen_optimal_spec(const mag_ctx_t* ctx, mag_kernel_registry_t* kernels) {
    for (size_t i=0; i < sizeof(mag_amd64_blas_specializations)/sizeof(*mag_amd64_blas_specializations); ++i) { /* Find best blas spec for the host CPU */
        const mag_amd64_blas_specialization* spec = mag_amd64_blas_specializations+i;
        size_t num_features = 0;
        const mag_x86_64_feature_t* features = (*spec->get_feature_permutation)(&num_features); /* Get requires features */
        if (mag_unlikely(!num_features || !features)) continue;
        bool has_all_features = true;
        for (size_t j=0; j < num_features; ++j) /* For each requested feature, check if host CPU supports it */
            has_all_features &= mag_ctx_x86_64_cpu_has_feature(ctx, features[j]);
        if (has_all_features) { /* Since specializations are sorted by score, we found the perfect spec. */
            (*spec->inject_kernels)(kernels);
            mag_log_info("Using BLAS specialization: %s", spec->name);
            return true;
        }
    }
    /* No matching specialization found, use generic */
    mag_cpu_blas_specialization_fallback(kernels);
    return false; /* No spec used, fallback is active */
}

#undef mag_amd64_blas_spec_permute
#undef mag_cpu_blas_spec_decl

#elif defined(__aarch64__) || defined(_M_ARM64)

#define mag_cpu_blas_spec_name(feat) mag_cpu_blas_specialization_arm64_##feat
#define mag_cpu_blas_spec_decl(feat) extern void mag_cpu_blas_spec_name(feat)(mag_kernel_registry_t* kernels)

mag_cpu_blas_spec_decl(82);

static bool mag_blas_detect_gen_optimal_spec(const mag_ctx_t* ctx, mag_kernel_registry_t* kernels) {
#ifdef __linux__
    long hwcap = ctx->sys.cpu_arm64_hwcap;
    if (hwcap & HWCAP_FPHP) && (hwcap & HWCAP_ASIMDHP) && (hwcap && HWCAP_ASIMDDP)) { /* ARM v.8.2 f16 scalar + f16 vec + dotprod */
        mag_cpu_blas_spec_name(82)(kernels);
        return true;
    }
#elif defined(__APPLE__)
    /* TODO - currently using ARM v8 baseline but Apple M2/M3 have newer arm versions we could target */
    /* mag_cpu_blas_spec_name(82)(kernels); */
#endif
    /* No matching specialization found, use generic */
    mag_cpu_blas_specialization_fallback(kernels);
    return false; /* No spec used, fallback is active */
}

#undef mag_cpu_blas_spec_decl

#endif

static bool mag_blas_detect_optimal_specialization(const mag_ctx_t* ctx, mag_kernel_registry_t* kernels) {
    if (mag_likely(mag_blas_detect_gen_optimal_spec(ctx, kernels))) return true;
    mag_cpu_blas_specialization_fallback(kernels);
    return false; /* No spec used, fallback is active */
}

typedef struct mag_worker_t mag_worker_t;
typedef struct mag_threadpool_t {
    mag_alignas(MAG_HDI) volatile bool interrupt;   /* Interrupt flag, 1=stop */
    mag_alignas(MAG_HDI) uint64_t phase;            /* Current compute phase */
    mag_alignas(MAG_HDI) uint64_t num_completed;    /* Number of workers that have completed their work */
    mag_cond_var_t cv;                              /* Condition variable for thread wakeup */
    mag_mutex_t mtx;                                /* Mutex for synchronization */
    uint32_t num_allocated_workers;                 /* Number of intra-op workers allocated */
    uint32_t num_active_workers;                    /* Number of intra-op workers that are actively used in this compute step. */
    volatile mag_atomic_t num_workers_online;       /* Number of workers that are online */
    mag_worker_t* workers;                          /* Array of workers */
    const mag_kernel_registry_t* kernels;           /* Specialized compute kernel registry */
    mag_thread_sched_prio_t sched_prio;             /* Scheduling priority */
} mag_threadpool_t;

struct mag_worker_t {
    uint64_t phase;                         /* Current compute phase */
    mag_compute_payload_t payload;          /* Compute op payload */
    mag_prng_state_t local_prng;            /* Local PRNG state */
    mag_threadpool_t* pool;                 /* Host thread pool */
    bool is_async;                          /* True if worker is async (executed on a different thread)  */
    mag_thread_t thread;                    /* Thread handle */
} mag_alignas(MAG_HDI);

typedef struct mag_cpu_device_t {
    mag_ctx_t* ctx;
    mag_threadpool_t* pool;             /* Thread pool. NULL if num_allocated_workers <= 1 */
    mag_compute_payload_t st_payload;   /* Single threaded payload */
    uint32_t num_allocated_workers;     /* Amount of worker thread used. if == 1 then single threaded mode and thread pool is not created */
    mag_kernel_registry_t kernels;      /* Compute kernels. Specialized by arch optimized version at boot (e.g. AVX, AVX512 etc..) */
    double growth_scale;                /* Growth scale for dynamic work scaling. TODO: Find better value and benchmark. */
    int64_t numel_threshold;            /* Numel treshold from where to enable dynamic work scaling. */
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
static void mag_worker_exec_thread_local(const mag_kernel_registry_t* kernels, mag_compute_payload_t* payload) {
    if (mag_likely(payload->node)) { /* Do the work 🦾 */
        (*kernels->fwd[payload->node->op])(payload);
        payload->node = NULL;
    }
}

/* Execute the operation and broadcast completion if last chunk was done */
static void mag_worker_exec_and_broadcast(mag_threadpool_t* pool, const mag_kernel_registry_t* kernels, mag_compute_payload_t* payload) {
    if (mag_likely(payload->thread_idx < pool->num_active_workers)) /* Execute the operation if we are an active thread. */
        mag_worker_exec_thread_local(kernels, payload);
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
    char name[32];
    snprintf(name, sizeof(name), "mag_worker_%" PRIx64, payload->thread_idx);
    mag_thread_set_name(name);
    uint64_t seed = payload->thread_idx ^ mag_thread_id() ^ (uintptr_t)arg ^ (uintptr_t)&arg;
    mag_prng_init(&payload->local_prng, MAG_PRNG_MERSENNE_TWISTER, seed);
    /*mag_thread_set_prio(pool->sched_prio);*/
    mag_atomic_fetch_add(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);

    while (mag_likely(mag_worker_await_work(worker, pool)))  /* Main work loop: wait, work, signal status */
        mag_worker_exec_and_broadcast(pool, kernels, payload);

    mag_atomic_fetch_sub(&pool->num_workers_online, 1, MAG_MO_SEQ_CST);
    return MAG_THREAD_RET_NONE;
}

/* Create thread pool and allocate threads */
static mag_threadpool_t* mag_threadpool_create(uint32_t num_workers, const mag_kernel_registry_t* kernels, mag_thread_sched_prio_t prio) { /* Create a thread pool */
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
        .sched_prio = prio
    };
    mag_cv_create(&pool->cv);
    mag_mutex_create(&pool->mtx);
    for (uint32_t ti=0; ti < num_workers; ++ti) { /* Initialize workers */
        workers[ti] = (mag_worker_t){
            .phase = 0,
            .payload = (mag_compute_payload_t){.thread_num = num_workers, .thread_idx = ti, .local_prng = {0}, .node = NULL, },
            .pool = pool,
            .is_async = ti != 0 /* Main thread is worker but without thread */
        };
        if (workers[ti].is_async)
            mag_thread_create(&workers[ti].thread, &mag_worker_thread_exec_op, workers+ti);
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
static void mag_threadpool_kickoff(mag_threadpool_t* pool, mag_tensor_t* node, uint32_t num_active_workers) {
    mag_mutex_lock(&pool->mtx);
    pool->num_active_workers = num_active_workers;
    for (uint32_t i=0; i < pool->num_allocated_workers; ++i) { /* Set up payload */
        mag_compute_payload_t* payload = &pool->workers[i].payload;
        payload->node = node;
        payload->thread_num = num_active_workers;
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
static MAG_HOTPROC void mag_threadpool_parallel_compute(mag_threadpool_t* pool, mag_tensor_t* node, uint32_t num_active_workers) {
    mag_assert2(pool != NULL);
    mag_threadpool_kickoff(pool, node, num_active_workers);                         /* Kick off workers */
    mag_cv_broadcast(&pool->cv);                                  /* Wake up all workers */
    mag_worker_exec_and_broadcast(pool, pool->kernels, &pool->workers->payload);    /* Main thread does work too */
    mag_threadpool_barrier(pool);                                                   /* Wait for all workers to finish */
}

static uint32_t mag_cpu_dynamic_work_scaling(mag_cpu_device_t* dvc, int64_t numel);
static MAG_HOTPROC void mag_cpu_exec_fwd(mag_compute_device_t* dvc, mag_tensor_t* node) {
    mag_cpu_device_t* cpu_dvc = dvc->impl;
    mag_threadpool_t* pool = cpu_dvc->pool;
    uint32_t num_active_workers = mag_cpu_dynamic_work_scaling(cpu_dvc, node->numel);
    if (!pool || num_active_workers <= 1) /* Main thread does the work (single threaded mode). */
        mag_worker_exec_thread_local(&cpu_dvc->kernels, &cpu_dvc->st_payload);
    else
        mag_threadpool_parallel_compute(pool, node, num_active_workers);
}

static MAG_HOTPROC void mag_cpu_exec_bwd(mag_compute_device_t* dvc, mag_tensor_t* root) {
    (void)dvc, (void)root;
    mag_panic("NYI");
}

static void mag_cpu_buf_set(mag_storage_buffer_t* sto, size_t offs, uint8_t x) {
    mag_assert2(sto->base+offs <= sto->base+sto->size);
    memset((void*)(sto->base+offs), x, sto->size-offs); /* On CPU just plain old memset with offset. */
}

static void mag_cpu_buf_cpy_host_device(mag_storage_buffer_t* sto, size_t offs, const void* src, size_t n) {
    mag_assert2(sto->base+offs+n <= sto->base+sto->size);
    memcpy((void*)(sto->base+offs), src, n); /* On CPU just plain old memcpy with offset. */
}

static void mag_cpu_buf_cpy_device_host(mag_storage_buffer_t* sto, size_t offs, void* dst, size_t n) {
    mag_assert2(sto->base+offs+n <= sto->base+sto->size);
    memcpy(dst, (void*)(sto->base+offs), n); /* On CPU just plain old memcpy with offset. */
}

static void mag_cpu_alloc_storage(mag_compute_device_t* host, mag_storage_buffer_t* out, size_t size) {
    mag_assert2(size);
    size_t align = 16; /* Align to cache line size. */
    void* block = mag_alloc_aligned(size, align);
    *out = (mag_storage_buffer_t){ /* Set up storage buffer. */
        .base = (uintptr_t)block,
        .size = size,
        .alignment = align,
        .host = host,
        .set = &mag_cpu_buf_set,
        .cpy_host_device = &mag_cpu_buf_cpy_host_device,
        .cpy_device_host = &mag_cpu_buf_cpy_device_host
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
        .st_payload = {
            .thread_idx = 0,
            .thread_num = 1,
            .local_prng = {0},
            .node = NULL
        },
        .num_allocated_workers = 0,
        .kernels = {},
        .growth_scale = 0.3, /* TODO: better value and heuristic */
        .numel_threshold = 250000 /* TODO: better value and heuristic */
    };
    mag_prng_init(&dvc->st_payload.local_prng, MAG_PRNG_MERSENNE_TWISTER, 0);
    mag_blas_detect_optimal_specialization(ctx, &dvc->kernels);
    if (num_threads > 1) {
        dvc->pool = mag_threadpool_create(num_threads, &dvc->kernels, sched_prio);
        dvc->num_allocated_workers = num_threads;
    }
    return dvc;
}

/*
** Computes how many workers to use for intra-op parallelism depending on the number of elements.
** A logarithmic scaling is used, see: https://www.desmos.com/calculator/xiunrskpwu
** TODO: This can be improved by using a more sophisticated heuristic and a benchmarked, numerical approach.
*/
static uint32_t mag_cpu_dynamic_work_scaling(mag_cpu_device_t* dvc, int64_t numel) {
    if (!dvc->pool || numel < dvc->numel_threshold) return 1; /* If numel is small, use a single worker. */
    numel -= dvc->numel_threshold;
    uint32_t workers = (uint32_t)ceil(dvc->growth_scale * log2((int64_t)numel)); /* Logarithmic scaling */
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
        .eager_exec_fwd = &mag_cpu_exec_fwd,
        .eager_exec_bwd = &mag_cpu_exec_bwd,
        .alloc_storage = &mag_cpu_alloc_storage,
        .free_storage = &mag_cpu_free_storage
    };
    snprintf(dvc->name, sizeof(dvc->name), "%s - %s - Using %u Compute Threads", mag_device_type_get_name(dvc->type), ctx->sys.cpu_name, num_threads);
    return dvc;
}

static void mag_cpu_release_interface(mag_compute_device_t* ctx) {
    mag_cpu_device_t* cpu_dvc = ctx->impl;
    mag_cpu_destroy_device(cpu_dvc);
    (*mag_alloc)(ctx, 0); /* Free all memory */
}

mag_compute_device_t* mag_init_device_cpu(mag_ctx_t* ctx, const mag_device_descriptor_t* desc) {
    uint32_t hw_concurrency = mag_xmax(1, ctx->sys.cpu_virtual_cores);
    uint32_t num_threads = desc->thread_count;
    num_threads = num_threads ? num_threads : hw_concurrency;
    mag_compute_device_t* dvc = mag_cpu_init_interface(ctx, num_threads);
    return dvc;
}

void mag_destroy_device_cpu(mag_compute_device_t* dvc) {
    mag_cpu_release_interface(dvc);
}
