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

#ifndef MAG_CONTEXT_H
#define MAG_CONTEXT_H

#include "mag_def.h"
#include "mag_cpuid.h"
#include "mag_pool.h"
#include "mag_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Context specific flags. */
typedef enum mag_context_flags_t {
    MAG_CTX_FLAG_NONE = 0,
    MAG_CTX_FLAG_GRAD_RECORDER = 1<<0,     /* Gradient recording is currently active. */
} mag_context_flags_t;

struct mag_context_t {
    struct {
        char os_name[128];                      /* OS name. */
        char cpu_name[128];                     /* CPU name. */
        uint32_t cpu_virtual_cores;             /* Virtual CPUs. */
        uint32_t cpu_physical_cores;            /* Physical CPU cores. */
        uint32_t cpu_sockets;                   /* CPU sockets. */
        size_t cpu_l1_size;                    /* L1 data cache size in bytes. */
        size_t cpu_l2_size;                     /* L2 cache size in bytes. */
        size_t cpu_l3_size;                     /* L3 cache size in bytes. */
        size_t phys_mem_total;                  /* Total physical memory in bytes. */
        size_t phys_mem_free;                   /* Free physical memory in bytes. */
#if defined(__x86_64__) || defined(_M_X64)
        mag_amd64_cap_bitset_t amd64_cpu_caps;  /* x86-64 CPU capability bits. */
        uint32_t amd64_avx10_ver;               /* x86-64 AVX10 version. */
#elif defined (__aarch64__) || defined(_M_ARM64)
        mag_arm64_cap_bitset_t arm64_cpu_caps;  /* ARM64 CPU features. */
        int64_t arm64_cpu_sve_width;            /* ARM64 SVE vector register width. */
#endif
    } machine;
    size_t num_tensors;                         /* Total tensor instances allocated. */
    size_t num_storages;                        /* Total storage buffers allocated. */
    mag_fixed_pool_t tensor_pool;               /* Tensor header memory pool. */
    mag_fixed_pool_t storage_pool;              /* Storage header memory pool. */
    mag_fixed_pool_t view_meta_pool;            /* View metadata header memory pool. */
    mag_fixed_pool_t au_state_pool;             /* Autodiff state memory pool. */
    mag_context_flags_t flags;                  /* Context flags. */
    uintptr_t tr_id;                            /* Context thread ID. */
    size_t sh_len;                              /* Number of shutdown hooks. */
    size_t sh_cap;                              /* Maximum number of shutdown hooks. */
    mag_device_type_t device_type;              /* Active compute device type. */
    mag_backend_registry_t *backend_registry;   /* Compute backend registry */
    mag_backend_t *backend;                     /* Active compute backend. */
    mag_device_t *device;                       /* Active compute device. */
    void *ud;                                   /* User data. */
#ifdef MAG_DEBUG
    mag_tensor_t *alive_head;                   /* List of alive tensors used for leak detection. */
#endif
};

#ifdef __cplusplus
}
#endif

#endif
