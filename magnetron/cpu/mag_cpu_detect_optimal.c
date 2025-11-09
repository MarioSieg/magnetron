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

#include "mag_cpu_detect_optimal.h"

#include <core/mag_context.h>
#include <core/mag_cpuid.h>

extern void mag_cpu_blas_specialization_fallback(mag_kernel_registry_t *kernels); /* Generic any CPU impl */

#if defined(__x86_64__) || defined(_M_X64) /* Specialized impls for x86-64 with runtime CPU detection */

typedef struct mag_amd64_specialization_dispatch_t {
    const char *name;
    mag_amd64_cap_bitset_t (*get_feature_permutation)(void);
    void (*inject_kernels)(mag_kernel_registry_t *kernels);
} mag_amd64_specialization_dispatch_t;

#define mag_amd64_blas_spec_decl(feat) \
    mag_amd64_cap_bitset_t mag_cpu_blas_specialization_amd64_##feat##_features(void); \
    extern void mag_cpu_blas_specialization_amd64_##feat(mag_kernel_registry_t *kernels)

#define mag_amd64_blas_spec_permute(feat) \
    (mag_amd64_specialization_dispatch_t) { \
        .name = "amd64-"#feat, \
        .get_feature_permutation = &mag_cpu_blas_specialization_amd64_##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_amd64_##feat \
    }

#ifdef MAG_HAVE_CPU_ALDERLAKE
mag_amd64_blas_spec_decl(alderlake);
#endif
#ifdef MAG_HAVE_CPU_ARROWLAKE
mag_amd64_blas_spec_decl(arrowlake);
#endif
#ifdef MAG_HAVE_CPU_CANNONLAKE
mag_amd64_blas_spec_decl(cannonlake);
#endif
#ifdef MAG_HAVE_CPU_CASCADELAKE
mag_amd64_blas_spec_decl(cascadelake);
#endif
#ifdef MAG_HAVE_CPU_COOPERLAKE
mag_amd64_blas_spec_decl(cooperlake);
#endif
#ifdef MAG_HAVE_CPU_CORE2
mag_amd64_blas_spec_decl(core2);
#endif
#ifdef MAG_HAVE_CPU_HASWELL
mag_amd64_blas_spec_decl(haswell);
#endif
#ifdef MAG_HAVE_CPU_ICELAKE
mag_amd64_blas_spec_decl(icelake);
#endif
#ifdef MAG_HAVE_CPU_IVYBRIDGE
mag_amd64_blas_spec_decl(ivybridge);
#endif
#ifdef MAG_HAVE_CPU_NEHALEM
mag_amd64_blas_spec_decl(nehalem);
#endif
#ifdef MAG_HAVE_CPU_SANDYBRIDGE
mag_amd64_blas_spec_decl(sandybridge);
#endif
#ifdef MAG_HAVE_CPU_SAPPHIRERAPIDS
mag_amd64_blas_spec_decl(sapphirerapids);
#endif
#ifdef MAG_HAVE_CPU_SIERRAFOREST
mag_amd64_blas_spec_decl(sierraforest);
#endif
#ifdef MAG_HAVE_CPU_SKYLAKE_AVX512
mag_amd64_blas_spec_decl(skylake_avx512);
#endif
#ifdef MAG_HAVE_CPU_TIGERLAKE
mag_amd64_blas_spec_decl(tigerlake);
#endif
#ifdef MAG_HAVE_CPU_ZNVER1
mag_amd64_blas_spec_decl(zn1);
#endif
#ifdef MAG_HAVE_CPU_ZNVER2
mag_amd64_blas_spec_decl(zn2);
#endif
#ifdef MAG_HAVE_CPU_ZNVER3
mag_amd64_blas_spec_decl(zn3);
#endif
#ifdef MAG_HAVE_CPU_ZNVER4
mag_amd64_blas_spec_decl(zn4);
#endif
#ifdef MAG_HAVE_CPU_ZNVER5
mag_amd64_blas_spec_decl(zn5);
#endif

static bool mag_blas_detect_gen_optimal_spec(const mag_context_t *host_ctx, mag_kernel_registry_t *kernels) {
    static const mag_amd64_specialization_dispatch_t specializations_intel[] = {
        #ifdef MAG_HAVE_CPU_SAPPHIRERAPIDS
                mag_amd64_blas_spec_permute(sapphirerapids),
        #endif
        #ifdef MAG_HAVE_CPU_ICELAKE
                mag_amd64_blas_spec_permute(icelake),
        #endif
        #ifdef MAG_HAVE_CPU_COOPERLAKE
                mag_amd64_blas_spec_permute(cooperlake),
        #endif
        #ifdef MAG_HAVE_CPU_CASCADELAKE
                mag_amd64_blas_spec_permute(cascadelake),
        #endif
        #ifdef MAG_HAVE_CPU_TIGERLAKE
                mag_amd64_blas_spec_permute(tigerlake),
        #endif
        #ifdef MAG_HAVE_CPU_SKYLAKE_AVX512
                mag_amd64_blas_spec_permute(skylake_avx512),
        #endif
        #ifdef MAG_HAVE_CPU_HASWELL
                mag_amd64_blas_spec_permute(haswell),
        #endif
        #ifdef MAG_HAVE_CPU_ALDERLAKE
                mag_amd64_blas_spec_permute(alderlake),
        #endif
        #ifdef MAG_HAVE_CPU_ARROWLAKE
                mag_amd64_blas_spec_permute(arrowlake),
        #endif
        #ifdef MAG_HAVE_CPU_CANNONLAKE
                mag_amd64_blas_spec_permute(cannonlake),
        #endif
        #ifdef MAG_HAVE_CPU_IVYBRIDGE
                mag_amd64_blas_spec_permute(ivybridge),
        #endif
        #ifdef MAG_HAVE_CPU_SANDYBRIDGE
                mag_amd64_blas_spec_permute(sandybridge),
        #endif
        #ifdef MAG_HAVE_CPU_NEHALEM
                mag_amd64_blas_spec_permute(nehalem),
        #endif
        #ifdef MAG_HAVE_CPU_CORE2
                mag_amd64_blas_spec_permute(core2),
        #endif
        #ifdef MAG_HAVE_CPU_SIERRAFOREST
                mag_amd64_blas_spec_permute(sierraforest),
        #endif
    };

    static const mag_amd64_specialization_dispatch_t specializations_amd[] = {
        #ifdef MAG_HAVE_CPU_ZNVER5
                mag_amd64_blas_spec_permute(zn5),
        #endif
        #ifdef MAG_HAVE_CPU_ZNVER4
                mag_amd64_blas_spec_permute(zn4),
        #endif
        #ifdef MAG_HAVE_CPU_ZNVER3
                mag_amd64_blas_spec_permute(zn3),
        #endif
        #ifdef MAG_HAVE_CPU_ZNVER2
                mag_amd64_blas_spec_permute(zn2),
        #endif
        #ifdef MAG_HAVE_CPU_ZNVER1
                mag_amd64_blas_spec_permute(zn1),
        #endif
    };
    bool is_amd = host_ctx->machine.amd64_cpu_caps & mag_amd64_cap(AMD);
    const mag_amd64_specialization_dispatch_t *impls = is_amd ? specializations_amd : specializations_intel;
    size_t num_impls = is_amd ? sizeof(specializations_amd)/sizeof(*specializations_amd) : sizeof(specializations_intel)/sizeof(*specializations_intel);

    mag_amd64_cap_bitset_t cap_machine = host_ctx->machine.amd64_cpu_caps;
    for (size_t i=0; i < num_impls; ++i) { /* Find best blas spec for the host CPU */
        const mag_amd64_specialization_dispatch_t *spec = impls+i;
        mag_amd64_cap_bitset_t cap_required = (*spec->get_feature_permutation)(); /* Get requires features */
        if ((cap_machine & cap_required) == cap_required) { /* Since specializations are sorted by score, we found the perfect spec. */
            (*spec->inject_kernels)(kernels);
            mag_log_info("Using tuned specialization: %s", spec->name);
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

typedef struct mag_arm64_specialization_dispatch_t {
    const char *name;
    mag_arm64_cap_bitset_t (*get_cap_permutation)(void);
    void (*inject_kernels)(mag_kernel_registry_t *kernels);
} mag_arm64_specialization_dispatch_t;

#define mag_arm64_spec_extern(feat) \
    mag_arm64_cap_bitset_t mag_cpu_blas_specialization_arm64_v##feat##_features(void); \
    extern void mag_cpu_blas_specialization_arm64_v##feat(mag_kernel_registry_t* kernels)

#define mag_arm64_spec_dispatch(feat) \
    (mag_arm64_specialization_dispatch_t) { \
        .name = "arm64-v."#feat, \
        .get_cap_permutation = &mag_cpu_blas_specialization_arm64_v##feat##_features, \
        .inject_kernels = &mag_cpu_blas_specialization_arm64_v##feat \
}

#ifdef MAG_HAVE_CPU_ARMV9_A_SVE2
mag_arm64_spec_extern(9_sve2);
#endif
#ifdef MAG_HAVE_CPU_ARMV8_2_A_SVE
mag_arm64_spec_extern(82_sve);
#endif
#ifdef MAG_HAVE_CPU_ARMV8_6_A_BF16_I8MM_FP16_DOTPROD
mag_arm64_spec_extern(86);
#endif
#ifdef MAG_HAVE_CPU_ARMV8_2_A_DOTPROD_FP16
mag_arm64_spec_extern(82);
#endif

static bool mag_blas_detect_gen_optimal_spec(const mag_context_t *ctx, mag_kernel_registry_t *kernels) {
    const mag_arm64_specialization_dispatch_t impls[] = { /* Dynamic selectable BLAS permutations, sorted from best to worst score. */
        #ifdef MAG_HAVE_CPU_ARMV9_A_SVE2
            mag_arm64_spec_dispatch(9_sve2),
        #endif
        #ifdef MAG_HAVE_CPU_ARMV8_2_A_SVE
            mag_arm64_spec_dispatch(82_sve),
        #endif
        #ifdef MAG_HAVE_CPU_ARMV8_6_A_BF16_I8MM_FP16_DOTPROD
            mag_arm64_spec_dispatch(86),
        #endif
        #ifdef MAG_HAVE_CPU_ARMV8_2_A_DOTPROD_FP16
            mag_arm64_spec_dispatch(82),
        #endif
    };

    mag_arm64_cap_bitset_t cap_avail = ctx->machine.arm64_cpu_caps;
    for (size_t i=0; i < sizeof(impls)/sizeof(*impls); ++i) { /* Find best blas spec for the host CPU */
        const mag_arm64_specialization_dispatch_t *spec = impls+i;
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

bool mag_blas_detect_optimal_specialization(const mag_context_t *ctx, mag_kernel_registry_t *kernels) {
    if (mag_likely(mag_blas_detect_gen_optimal_spec(ctx, kernels))) return true;
    mag_cpu_blas_specialization_fallback(kernels);
    return false; /* No spec used, fallback is active */
}
