/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "magnetron_internal.h"

#include <math.h>

#include "pool.hpp"

extern "C" void mag_cpu_blas_specialization_fallback(mag_kernel_registry_t* kernels); /* Generic any CPU impl */

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

mag_cpu_blas_spec_decl(znver4);
mag_cpu_blas_spec_decl(avx512f);
mag_cpu_blas_spec_decl(avx2);
mag_cpu_blas_spec_decl(avx);
mag_cpu_blas_spec_decl(sse41);

static const mag_amd64_blas_specialization mag_amd64_blas_specializations[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
    mag_amd64_blas_spec_permute(znver4),
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
    [MAG_OP_ABS]            = {.mt_support = false, .growth = 0.0, .threshold = 0},
    [MAG_OP_NEG]            = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_LOG]            = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SQR]            = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SQRT]           = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SIN]            = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_COS]            = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_STEP]           = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SOFTMAX]        = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SOFTMAX_DV]     = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SIGMOID]        = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SIGMOID_DV]     = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_HARD_SIGMOID]   = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SILU]           = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_SILU_DV]        = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_TANH]           = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_TANH_DV]        = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_RELU]           = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_RELU_DV]        = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_GELU]           = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_GELU_DV]        = {.mt_support = true, .growth = 0.1, .threshold = 250000},
    [MAG_OP_ADD]            = {.mt_support = true, .growth = 0.2, .threshold = 250000},
    [MAG_OP_SUB]            = {.mt_support = true, .growth = 0.2, .threshold = 250000},
    [MAG_OP_MUL]            = {.mt_support = true, .growth = 0.2, .threshold = 250000},
    [MAG_OP_DIV]            = {.mt_support = true, .growth = 0.2, .threshold = 250000},
    [MAG_OP_ADDS]           = {.mt_support = true, .growth = 0.2, .threshold = 250000},
    [MAG_OP_SUBS]           = {.mt_support = true, .growth = 0.2, .threshold = 250000},
    [MAG_OP_MULS]           = {.mt_support = true, .growth = 0.2, .threshold = 250000},
    [MAG_OP_DIVS]           = {.mt_support = true, .growth = 0.2, .threshold = 250000},
    [MAG_OP_MATMUL]         = {.mt_support = true, .growth = 3.0, .threshold =  10000},
};

static mag_kernel_registry_t kernels;

static MAG_HOTPROC void mag_cpu_exec_fwd(mag_compute_device_t* dvc, mag_tensor_t* node) {
    if (mag_cpu_op_infos[node->op].mt_support) { /* Main thread does the work (single threaded mode). */
        const mag_compute_payload_t payload = {
            .thread_idx = 0,
            .thread_num = 1,
            .node = node
        };
        kernels.fwd[node->op](&payload);
    } else {
        auto* pool = static_cast<BS::thread_pool<>*>(dvc->impl);
        auto nt = pool->get_thread_count();
        for (int i=0; i < nt; ++i) {
                pool->detach_task([=] {
                    const mag_compute_payload_t payload = {
                    .thread_idx = i,
                    .thread_num = (int64_t)nt,
                    .node = node
                };
                kernels.fwd[node->op](&payload);
            });
        }
        pool->wait();
    }
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
    size_t align = MAG_CACHE_LINE_SIZE; /* Align to cache line size. */
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

static mag_compute_device_t* mag_cpu_init_interface(mag_ctx_t* ctx, uint32_t num_threads) {
    mag_compute_device_t* dvc = (mag_compute_device_t*)(*mag_alloc)(NULL, sizeof(*dvc));
    *dvc = (mag_compute_device_t){ /* Initialize device interface */
        .name = "CPU",
        .impl = new BS::thread_pool<>{num_threads},
        .is_async = false,
        .type = MAG_COMPUTE_DEVICE_TYPE_CPU,
        .eager_exec_fwd = &mag_cpu_exec_fwd,
        .eager_exec_bwd = &mag_cpu_exec_bwd,
        .alloc_storage = &mag_cpu_alloc_storage,
        .free_storage = &mag_cpu_free_storage
    };
    mag_blas_detect_optimal_specialization(ctx, &kernels);
    snprintf(dvc->name, sizeof(dvc->name), "%s - %s - Using %u Compute Threads", mag_device_type_get_name(dvc->type), ctx->sys.cpu_name, num_threads);
    return dvc;
}

static void mag_cpu_release_interface(mag_compute_device_t* ctx) {
    delete static_cast<BS::thread_pool<>*>(ctx->impl);
    (*mag_alloc)(ctx, 0); /* Free all memory */
}

extern "C" mag_compute_device_t* mag_init_device_cpu(mag_ctx_t* ctx, const mag_device_descriptor_t* desc) {
    uint32_t hw_concurrency = mag_xmax(1, ctx->sys.cpu_virtual_cores);
    uint32_t num_threads = desc->thread_count;
    num_threads = num_threads ? num_threads : hw_concurrency;
    mag_compute_device_t* dvc = mag_cpu_init_interface(ctx, num_threads);
    return dvc;
}

extern "C" void mag_destroy_device_cpu(mag_compute_device_t* dvc) {
    mag_cpu_release_interface(dvc);
}
