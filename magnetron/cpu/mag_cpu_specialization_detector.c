/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "mag_cpu_specialization_detector.h"

#include <core/mag_context.h>
#include <core/mag_cpuid.h>
#include <core/mag_envcfg.h>
#include <core/mag_sstream.h>

extern void mag_cpu_blas_specialization_fallback(mag_kernel_registry_t *kernels); /* Generic any CPU impl */

#define mag_cpu_specialization_extern(arch, flag) \
extern uint64_t mag_cpu_blas_specialization_##arch##_##flag##_features(void); \
extern void mag_cpu_blas_specialization_##arch##_##flag(mag_kernel_registry_t *kernels)

#define mag_cpu_specialization_configure(arch, flag) \
(mag_cpu_specialization_t) { \
.name = #arch"-"#flag, \
.get_feature_bitset = &mag_cpu_blas_specialization_##arch##_##flag##_features, \
.inject_kernels = &mag_cpu_blas_specialization_##arch##_##flag \
}

#if defined(__x86_64__) || defined(_M_X64) /* Specialized impls for x86-64 with runtime CPU detection */

  static uint64_t mag_get_cpu_host_caps(const mag_context_t *ctx) { return ctx->machine.amd64_cpu_caps; }

  static const char *const mag_cpu_cap_names[MAG_AMD64_CAP__NUM] = {
  #define _(name) #name
    mag_amd64_capdef(_, MAG_SEP)
  #undef _
  };

  const mag_cpu_specialization_t *mag_cpu_specializations(size_t *num) {
  #ifdef MAG_HAVE_CPU_V4_BF16
    mag_cpu_specialization_extern(amd64, v4_bf16);
  #endif
  #ifdef MAG_HAVE_CPU_V4
    mag_cpu_specialization_extern(amd64, v4);
  #endif
  #ifdef MAG_HAVE_CPU_V3
    mag_cpu_specialization_extern(amd64, v3);
  #endif
  #ifdef MAG_HAVE_CPU_V2_AVX
    mag_cpu_specialization_extern(amd64, v2_avx);
  #endif
  #ifdef MAG_HAVE_CPU_V2
    mag_cpu_specialization_extern(amd64, v2);
  #endif

  static const mag_cpu_specialization_t specializations[] = {  /* Order matters, sorted from best to worst */
    #ifdef MAG_HAVE_CPU_V4_BF16
      mag_cpu_specialization_configure(amd64, v4_bf16),
    #endif
    #ifdef MAG_HAVE_CPU_V4
      mag_cpu_specialization_configure(amd64, v4),
    #endif
    #ifdef MAG_HAVE_CPU_V3
      mag_cpu_specialization_configure(amd64, v3),
    #endif
    #ifdef MAG_HAVE_CPU_V2_AVX
      mag_cpu_specialization_configure(amd64, v2_avx),
    #endif
    #ifdef MAG_HAVE_CPU_V2
      mag_cpu_specialization_configure(amd64, v2),
    #endif
  };
  *num = sizeof(specializations)/sizeof(*specializations);
  return specializations;
  }

#elif defined(__aarch64__) || defined(_M_ARM64)

  static uint64_t mag_get_cpu_host_caps(const mag_context_t *ctx) { return ctx->machine.arm64_cpu_caps; }

  static const char *const mag_cpu_cap_names[MAG_ARM64_CAP__NUM] = {
  #define _(name) #name
    mag_armd64_capdef(_, MAG_SEP)
  #undef _
  };

  const mag_cpu_specialization_t *mag_cpu_specializations(size_t *num) {
    #ifdef MAG_HAVE_CPU_ARM_V9_SVE2
      mag_cpu_specialization_extern(arm64, v9_sve2);
    #endif
    #ifdef MAG_HAVE_CPU_ARM_V86_SVE
      mag_cpu_specialization_extern(arm64, v86_sve);
    #endif
    #ifdef MAG_HAVE_CPU_ARM_V86_CRYPTO
      mag_cpu_specialization_extern(arm64, v86_crypto);
    #endif
    #ifdef MAG_HAVE_CPU_ARM_V86
      mag_cpu_specialization_extern(arm64, v86);
    #endif
    #ifdef MAG_HAVE_CPU_ARM_V82
      mag_cpu_specialization_extern(arm64, v82);
    #endif
    #ifdef MAG_HAVE_CPU_ARM_V82_SVE
      mag_cpu_specialization_extern(arm64, v82_sve);
    #endif

    static const mag_cpu_specialization_t specializations[] = {  /* Order matters, sorted from best to worst */
      #ifdef MAG_HAVE_CPU_ARM_V9_SVE2
        mag_cpu_specialization_configure(arm64, v9_sve2),
      #endif
      #ifdef MAG_HAVE_CPU_ARM_V86_SVE
        mag_cpu_specialization_configure(arm64, v86_sve),
      #endif
      #ifdef MAG_HAVE_CPU_ARM_V86_CRYPTO
        mag_cpu_specialization_configure(arm64, v86_crypto),
      #endif
      #ifdef MAG_HAVE_CPU_ARM_V86
        mag_cpu_specialization_configure(arm64, v86),
      #endif
      #ifdef MAG_HAVE_CPU_ARM_V82
        mag_cpu_specialization_configure(arm64, v82),
      #endif
      #ifdef MAG_HAVE_CPU_ARM_V82_SVE
        mag_cpu_specialization_configure(arm64, v82_sve),
      #endif
    };
    *num = sizeof(specializations)/sizeof(*specializations);
    return specializations;
  }

#elif defined(__loongarch64) /* Loongson / Godson */

  static uint64_t mag_get_cpu_host_caps(const mag_context_t *ctx) { return ctx->machine.loongarch64_cpu_caps; }

  static const char *const mag_cpu_cap_names[MAG_LOONGARCH64_CAP__NUM] = {
  #define _(name) #name
    mag_loongarch64_capdef(_, MAG_SEP)
  #undef _
  };

  const mag_cpu_specialization_t *mag_cpu_specializations(size_t *num) {
    #ifdef MAG_HAVE_CPU_LSX
      mag_cpu_specialization_extern(loongarch64, lsx);
    #endif
    #ifdef MAG_HAVE_CPU_LASX
      mag_cpu_specialization_extern(loongarch64, lasx);
    #endif

    static const mag_cpu_specialization_t specializations[] = { /* Order matters, sorted from best to worst */
      #ifdef MAG_HAVE_CPU_LASX
        mag_cpu_specialization_configure(loongarch64, lasx),
      #endif
      #ifdef MAG_HAVE_CPU_LSX
        mag_cpu_specialization_configure(loongarch64, lsx),
      #endif
    };
    *num = sizeof(specializations)/sizeof(*specializations);
    return specializations;
  }

#else

static uint64_t mag_get_cpu_host_caps(const mag_context_t *ctx) { (void)ctx; return 0; }

static const char *const mag_cpu_cap_names[1] = {"NONE"};

const mag_cpu_specialization_t *mag_cpu_specializations(size_t *num) {
  *num = 0;
  return NULL;
}

#endif

const char *mag_cpu_cap_name(uint32_t bit) {
  size_t num = sizeof(mag_cpu_cap_names)/sizeof(*mag_cpu_cap_names);
  return bit < num ? mag_cpu_cap_names[bit] : NULL;
}

size_t mag_cpu_format_caps(uint64_t caps, char *buf, size_t cap) {
  size_t len = 0;
  if (!cap) return 0;
  *buf = '\0';
  for (uint32_t bit=0; bit < (sizeof(caps)<<3) && caps; ++bit) {
    if (!(caps & (1ull<<bit))) continue;
    caps &= ~(1ull<<bit);
    const char *name = mag_cpu_cap_name(bit);
    int n = name ? snprintf(buf+len, cap-len, "%s%s", len ? " " : "", name) : snprintf(buf+len, cap-len, "%sbit%u", len ? " " : "", bit);
    if (n < 0) break;
    len += (size_t)n;
    if (len >= cap) {
      len = cap-1;
      break;
    }
  }
  return len;
}

const mag_cpu_specialization_t *mag_cpu_select_specialization(const mag_cpu_specialization_t *impls, size_t num, uint64_t host_caps) {
  for (size_t i=0; i < num; ++i) {
    const mag_cpu_specialization_t *spec = impls+i;
    uint64_t spec_caps = (*spec->get_feature_bitset)();
    uint64_t missing = spec_caps & ~host_caps;
    if (mag_log_level() >= MAG_LOG_LEVEL_DEBUG) {
      char required[512], lacking[512];
      mag_cpu_format_caps(spec_caps, required, sizeof(required));
      mag_cpu_format_caps(missing, lacking, sizeof(lacking));
      mag_log_debug(
        "Checked specialization %s:\n\trequired: %s\n\tmissing:  %s\n\tmatches: %s",
        spec->name, required, missing ? lacking : "(none)", missing ? "no" : "yes"
      );
    }
    if (!missing) return spec;
  }
  return NULL;
}

static MAG_COLDPROC bool mag_specialization_name_matches(const char *name, const char *want) {
  if (mag_casecmp(name, want)) return true;
  const char *suffix = strchr(name, '-');
  return suffix && mag_casecmp(suffix+1, want);
}

static MAG_COLDPROC void mag_log_available_specializations(const mag_cpu_specialization_t *impls, size_t num) {
  for (size_t i=0; i < num; ++i)
    mag_log_info("  %s", impls[i].name);
  mag_log_info("  fallback");
}

static const mag_cpu_specialization_t *mag_get_pinned_specialization(
  const mag_cpu_specialization_t *impls,
  size_t num,
  uint64_t host_caps,
  const char *want
) {
  for (size_t i=0; i < num; ++i) {
    const mag_cpu_specialization_t *spec = impls+i;
    if (!mag_specialization_name_matches(spec->name, want)) continue;
    uint64_t spec_caps = (*spec->get_feature_bitset)();
    uint64_t missing = spec_caps & ~host_caps;
    if (missing) {
      char lacking[512];
      mag_cpu_format_caps(missing, lacking, sizeof(lacking));
      mag_log_warn(
        MAG_ENV_CPU_SPECIALIZATION_LEVEL "=%s requests specialization %s, but this CPU lacks the required features: %s "
        "(requires 0x%" PRIx64 ", machine caps 0x%" PRIx64 "). Auto-detecting instead.",
        want, spec->name, lacking, spec_caps, host_caps
      );
      return NULL;
    }
    return spec;
  }
  mag_log_warn(MAG_ENV_CPU_SPECIALIZATION_LEVEL "=%s is not a specialization level built into this binary. Available:", want);
  mag_log_available_specializations(impls, num);
  return NULL;
}

bool mag_blas_detect_optimal_specialization(const mag_context_t *ctx, mag_kernel_registry_t *kernels) {
  size_t num_impls=0;
  const mag_cpu_specialization_t *impls = mag_cpu_specializations(&num_impls);
  if (mag_unlikely(!num_impls || !impls)) goto fallback;
  uint64_t host_caps = mag_get_cpu_host_caps(ctx);
  if (mag_log_level() >= MAG_LOG_LEVEL_DEBUG) {
    char present[512];
    mag_cpu_format_caps(host_caps, present, sizeof(present));
    mag_log_debug("Available CPU specializations: %zu, machine caps 0x%" PRIx64 ": %s", num_impls, host_caps, present);
  }
  const char *pinned_name = NULL;
  switch (mag_envcfg_cpu_specialization_level(&pinned_name)) { /* Honor a level pinned via the environment. */
    case MAG_ENVCFG_CPU_SPECIALIZATION_FALLBACK:
      mag_log_info("Using fallback BLAS specialization (pinned by " MAG_ENV_CPU_SPECIALIZATION_LEVEL ")");
      mag_cpu_blas_specialization_fallback(kernels);
      return false;
    case MAG_ENVCFG_CPU_SPECIALIZATION_PINNED: {
      const mag_cpu_specialization_t *pinned = mag_get_pinned_specialization(impls, num_impls, host_caps, pinned_name);
      if (pinned) {
        (*pinned->inject_kernels)(kernels);
        mag_log_info("Using pinned specialization: %s (" MAG_ENV_CPU_SPECIALIZATION_LEVEL ")", pinned->name);
        return true;
      }
    } break;
    case MAG_ENVCFG_CPU_SPECIALIZATION_AUTO: break;
  }
  const mag_cpu_specialization_t *best = mag_cpu_select_specialization(impls, num_impls, host_caps);
  if (best) {
    (*best->inject_kernels)(kernels);
    mag_log_info("Found tuned specialization: %s", best->name);
    return true;
  }
fallback:
  mag_cpu_blas_specialization_fallback(kernels);
  mag_log_info("Using fallback BLAS specialization");
  return false;
}
