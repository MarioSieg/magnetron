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

#include "mag_cpuid.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <cpuid.h>

const char *const mag_amd64_cpu_cap_names[MAG_AMD64_CAP__NUM] = {
#define _(name) #name
    mag_amd64_capdef(_, MAG_SEP)
#undef _
};

static void mag_cpuid_ex(uint32_t (*o)[4], uint32_t eax, uint32_t ecx) {
#ifdef _WIN32
    __cpuidex((int *)(*o), eax, ecx);
#else
    __cpuid_count(eax, ecx, (*o)[0], (*o)[1], (*o)[2], (*o)[3]);
#endif
}
static void mag_cpuid(uint32_t (*o)[4], uint32_t eax) {
    mag_cpuid_ex(o, eax, 0);
}
static bool mag_cpuid_streq(uint32_t ebx, uint32_t ecx, uint32_t edx, const char str[12]) {
#define mag_strbe(x) ((x)[0] | ((x)[1]<<8) | ((x)[2]<<16) | ((x)[3]<<24))
    return ebx == mag_strbe(str) && edx == mag_strbe(str+4) && ecx == mag_strbe(str+8);
#undef mag_strbe
}
static uint64_t MAG_AINLINE mag_xgetbv(void) { /* Query extended control register value. */
#ifdef _MSC_VER
    return _xgetbv(0);
#else
    uint32_t eax, edx;
    __asm__ volatile(".byte 0x0f,0x01,0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return (uint64_t)edx<<32 | eax;
#endif
}
void mag_probe_cpu_amd64(mag_amd64_cap_bitset_t *o, uint32_t *avx10ver) {
    memset(o, 0, sizeof(*o));
    uint32_t id[4] = {0};
    const uint32_t *eax = id+0, *ebx = id+1, *ecx = id+2, *edx = id+3;
    mag_cpuid(&id, 0);
    uint32_t max = *eax;
    if (mag_cpuid_streq(*ebx, *ecx, *edx, "AuthenticAMD")) *o|=mag_amd64_cap(AMD);
    else if (mag_cpuid_streq(*ebx, *ecx, *edx, "GenuineIntel")) *o|=mag_amd64_cap(INTEL);
    mag_cpuid(&id, 0x80000000);
#define mag_captest(reg, bit, cp) if (*(reg)&(1u<<((bit)&31))) *o|=mag_amd64_cap(cp)
    uint32_t max_ex = *eax;
    if (max_ex >= 0x80000001) {
        mag_cpuid(&id, 0x80000001);
        mag_captest(ecx, 0, SSE4A);
    }
    mag_cpuid(&id, 1);
    mag_captest(ecx, 0, SSE3);
    mag_captest(ecx, 9, SSSE3);
    mag_captest(ecx, 19, SSE41);
    mag_captest(ecx, 20, SSE42);
    mag_captest(ecx, 27, OSXSAVE);
    mag_captest(ecx, 29, F16C);
    mag_captest(edx, 25, SSE);
    mag_captest(edx, 26, SSE2);
    if (*o & mag_amd64_cap(OSXSAVE)) {
        uint64_t cr = mag_xgetbv();
        if ((cr&6) == 6) {
            mag_captest(ecx, 12, FMA);
            mag_captest(ecx, 28, AVX);
            if (((cr>>5)&7) == 7) {
                mag_cpuid_ex(&id, 7, 0);
                mag_captest(ebx, 16, AVX512_F);
                if (*o & mag_amd64_cap(AVX512_F)) {
                    mag_captest(ebx, 17, AVX512_DQ);
                    mag_captest(ebx, 21, AVX512_IFMA);
                    mag_captest(ebx, 26, AVX512_PF);
                    mag_captest(ebx, 27, AVX512_ER);
                    mag_captest(ebx, 28, AVX512_CD);
                    mag_captest(ebx, 30, AVX512_BW);
                    mag_captest(ebx, 31, AVX512_VL);
                    mag_captest(ecx, 1, AVX512_VBMI);
                    mag_captest(ecx, 6, AVX512_VBMI2);
                    mag_captest(ecx, 11, AVX512_VNNI);
                    mag_captest(ecx, 12, AVX512_BITALG);
                    mag_captest(ecx, 14, AVX512_VPOPCNTDQ);
                    mag_captest(edx, 2, AVX512_4VNNIW);
                    mag_captest(edx, 3, AVX512_4FMAPS);
                    mag_captest(edx, 8, AVX512_VP2INTERSECT);
                    if (*o & mag_amd64_cap(AVX512_BW))
                        mag_captest(edx, 23, AVX512_FP16);
                }
            }
        }
    }
    if (max >= 7) {
        mag_cpuid_ex(&id, 7, 0);
        uint32_t max_sub = *eax;
        if (*o & mag_amd64_cap(AVX) && (*ebx & 1u<<5))
            *o |= mag_amd64_cap(AVX2);
        mag_captest(ebx, 3, BMI1);
        mag_captest(ebx, 8, BMI2);
        mag_captest(ecx, 8, GFNI);
        mag_captest(edx, 22, AMX_BF16);
        mag_captest(edx, 24, AMX_TILE);
        mag_captest(edx, 25, AMX_INT8);
        if (max_sub >= 1) {
            mag_cpuid_ex(&id, 7, 1);
            mag_captest(eax, 4, AVX_VNNI);
            if (*o & mag_amd64_cap(AVX512_F))
                mag_captest(eax, 5, AVX512_BF16);
            mag_captest(edx, 22, AMX_FP16);
            mag_captest(edx, 4, AVX_VNNI_INT8);
            mag_captest(edx, 5, AVX_NE_CONVERT);
            mag_captest(edx, 10, AVX_VNNI_INT16);
            mag_captest(edx, 19, AVX10);
            mag_captest(edx, 21, APX_F);
            mag_cpuid_ex(&id, 0x1e, 1);
            mag_captest(eax, 4, AMX_FP8);
            mag_captest(eax, 5, AMX_TRANSPOSE);
            mag_captest(eax, 6, AMX_TF32);
            mag_captest(eax, 7, AMX_AVX512);
            mag_captest(eax, 8, AMX_MOVRS);
        }
    }
#undef mag_captest
    if (*o & mag_amd64_cap(AVX10)) {
        mag_cpuid_ex(&id, 0x24, 0);
        *avx10ver = *ebx & 127;
    }
}

#define mag_bextract(x, b, e) (((x)>>(b))&((1u<<((e)+1-(b)))-1))

typedef enum mag_cpu_topology_level {
    MAG_CPU_TOPO_STMT = 1,
    MAG_CPU_TOPO_CORE = 2
} mag_cpu_topology_level;

static void mag_probe_cpu_core_topology(mag_amd64_cap_bitset_t caps, uint32_t (*num_cores)[MAG_MAX_CPU_TOPO_DEPTH]) {
    uint32_t id[4] = {0};
    mag_cpuid(&id, 0x0);
    if (*id >= 0xB) {
        mag_cpuid_ex(&id, 0xb, 0);
        if (*id || id[1]) {
            for (uint32_t i=0; i < MAG_MAX_CPU_TOPO_DEPTH; ++i) {
                mag_cpuid_ex(&id, 0xb, i);
                mag_cpu_topology_level level = (mag_cpu_topology_level)mag_bextract(id[2], 8, 15);
                if (level == MAG_CPU_TOPO_STMT || level == MAG_CPU_TOPO_CORE)
                    (*num_cores)[level-1] = mag_bextract(id[1], 0, 15);
            }
            (*num_cores)[MAG_CPU_TOPO_STMT-1] = mag_xmax(1u, (*num_cores)[MAG_CPU_TOPO_STMT-1]);
            (*num_cores)[MAG_CPU_TOPO_CORE-1] = mag_xmax((*num_cores)[MAG_CPU_TOPO_STMT-1], (*num_cores)[MAG_CPU_TOPO_CORE-1]);
            return;
        }
    }
    if (caps & mag_amd64_cap(AMD)) {
        int32_t ptc = 0;
        mag_cpuid(&id, 0x1);
        int32_t ltc = mag_bextract(id[1], 16, 23);
        int32_t htn = mag_bextract(id[3], 28, 28);
        mag_cpuid(&id, 0x80000000);
        uint32_t max_leaf = *id;
        if (max_leaf >= 0x80000008) {
            mag_cpuid(&id, 0x80000008);
            ptc = mag_bextract(id[2], 0, 7) + 1;
        }
        if (!htn) {
            (*num_cores)[MAG_CPU_TOPO_STMT-1] = 1;
            (*num_cores)[MAG_CPU_TOPO_CORE-1] = 1;
        } else if (ptc > 1) {
            mag_cpuid(&id, 1);
            int32_t fam_ext = mag_bextract(*id, 20, 27);
            int32_t fam = mag_bextract(*id, 8, 11);
            int32_t dis_fam = fam;
            if (dis_fam == 0x0f) dis_fam += fam_ext;
            if (dis_fam >= 0x17 && max_leaf >= 0x8000001e) {
                mag_cpuid(&id, 0x8000001e);
                ptc /= mag_bextract(id[1], 8, 15)+1;
            }
            (*num_cores)[MAG_CPU_TOPO_STMT-1] = ltc/ptc;
            (*num_cores)[MAG_CPU_TOPO_CORE-1] = ltc;
        } else {
            (*num_cores)[MAG_CPU_TOPO_STMT-1] = 1;
            (*num_cores)[MAG_CPU_TOPO_CORE-1] = ltc > 1 ? ltc : 2;
        }
    } else if (caps & mag_amd64_cap(INTEL)) {
        int32_t ptc = 0;
        mag_cpuid(&id, 0x1);
        int32_t lpc = mag_bextract(id[1], 16, 23);
        int32_t htt = mag_bextract(id[3], 28, 28);
        mag_cpuid(&id, 0);
        if (*id >= 0x4) {
            mag_cpuid(&id, 0x4);
            ptc = mag_bextract(id[0], 26, 31)+1;
        }
        if (!htt) {
            (*num_cores)[MAG_CPU_TOPO_STMT-1] = 1;
            (*num_cores)[MAG_CPU_TOPO_CORE-1] = 1;
        } else if (ptc > 1) {
            (*num_cores)[MAG_CPU_TOPO_STMT-1] = lpc/ptc;
            (*num_cores)[MAG_CPU_TOPO_CORE-1] = lpc;
        } else {
            (*num_cores)[MAG_CPU_TOPO_STMT-1] = 1;
            (*num_cores)[MAG_CPU_TOPO_CORE-1] = lpc > 0 ? lpc : 1;
        }
    }
}

void mag_probe_cpu_cache_topology(mag_amd64_cap_bitset_t caps, size_t *ol1, size_t *ol2, size_t *ol3) {
    uint32_t levels = 0;
    uint32_t data_cache[MAG_MAX_CPU_CACHE_DEPTH] = {0};
    uint32_t shared_cache[MAG_MAX_CPU_CACHE_DEPTH] = {0};
    uint32_t num_cores[MAG_MAX_CPU_TOPO_DEPTH] = {0};
    mag_probe_cpu_core_topology(caps, &num_cores);
    uint32_t id[4] = {0};
    if (caps & mag_amd64_cap(AMD)) {
        mag_cpuid(&id, 0x80000000);
        if (*id >= 0x8000001d) {
            levels = 0;
            for (uint32_t leaf=0; levels < MAG_MAX_CPU_CACHE_DEPTH; ++leaf) {
                mag_cpuid_ex(&id, 0x8000001d, leaf);
                int32_t type = mag_bextract(*id, 0, 4);
                if (!type) break;
                if (type == 0x2) continue;
                int32_t assoc = mag_bextract(*id, 9, 9);
                int32_t sharing = mag_bextract(*id, 14, 25)+1;
                int32_t ways = mag_bextract(id[1], 22, 31)+1;
                int32_t partitions = mag_bextract(id[1], 12, 21)+1;
                int32_t line = mag_bextract(id[1], 0, 11)+1;
                int32_t sets = id[2]+1;
                data_cache[levels] = line*partitions*ways;
                if (!assoc) data_cache[levels] *= sets;
                if (leaf > 0) {
                    sharing = mag_xmin(sharing, num_cores[1]);
                    sharing /= mag_xmax(1u, *shared_cache);
                }
                shared_cache[levels] = sharing;
                ++levels;
            }
            *shared_cache = mag_xmin(1u, *shared_cache);
        } else if (*id >= 0x80000006) {
            levels = 1;
            mag_cpuid(&id, 0x80000005);
            int32_t l1dc = mag_bextract(id[2], 24, 31);
            *data_cache = l1dc<<10;
            *shared_cache = 1;
            mag_cpuid(&id, 0x80000006);
            int32_t l2 = mag_bextract(id[2], 12, 15);
            if (l2 > 0) {
                levels = 2;
                int32_t l2s = mag_bextract(id[2], 16, 31);
                data_cache[1] = l2s<<10;
                shared_cache[1] = 1;
            }
            int32_t l3 = mag_bextract(id[3], 12, 15);
            if (l3 > 0) {
                levels = 3;
                int32_t l3s = mag_bextract(id[3], 18, 31);
                data_cache[2] = l3s<<19;
                shared_cache[2] = num_cores[1];
            }
        }
    } else if (caps & mag_amd64_cap(INTEL)) {
        uint32_t smt_width = *num_cores;
        uint32_t logical_cores = num_cores[1];
        for (uint32_t i=0; levels < MAG_MAX_CPU_CACHE_DEPTH; ++i) {
            mag_cpuid_ex(&id, 0x4, i);
            uint32_t type = mag_bextract(*id, 0, 4);
            if (!type) break;
            if (type == 1 || type == 3) {
                uint32_t actual_logical_cores = mag_bextract(*id, 14, 25)+1;
                if (logical_cores != 0) actual_logical_cores = mag_xmin(actual_logical_cores, logical_cores);
                mag_assert2(actual_logical_cores);
                data_cache[levels] =
                    (mag_bextract(id[1], 22, 31)+1)
                    * (mag_bextract(id[1], 12, 21)+1)
                    * (mag_bextract(id[1], 0, 11)+1)
                    * (id[2]+1);
                if (type == 1 && smt_width == 0) smt_width = actual_logical_cores;
                mag_assert2(smt_width != 0);
                shared_cache[levels] = mag_xmax(actual_logical_cores / smt_width, 1u);
                ++levels;
            }
        }
    }
    if (levels) {
        *ol1 = data_cache[0]/shared_cache[0];
        *ol2 = data_cache[1]/shared_cache[1];
        *ol3 = data_cache[2]/shared_cache[2];
    } else {
        *ol1 = 32ull<<10;
        *ol2 = 512ull<<10;
        *ol3 = 1024ull<<10;
    }
}

#undef mag_bextract

#elif defined(__aarch64__)

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#define _(ident) #ident
const char *const mag_arm64_cpu_cap_names[MAG_ARM64_CAP__NUM] = {
    mag_armd64_capdef(_, MAG_SEP)
};

void mag_probe_cpu_arm64(mag_arm64_cap_bitset_t *o, int64_t *sve_width) {
    *o = MAG_ARM64_CAP_NONE;
#ifdef __linux__
    unsigned long hwcap = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);
    (void)hwcap2;
    *o|=mag_arm64_cap(NEON); /* NEON is always required by build */
#ifdef HWCAP_ASIMD
    if (hwcap & HWCAP_ASIMD) *o|=mag_arm64_cap(NEON);
#endif
#ifdef HWCAP_ASIMDDP
    if (hwcap & HWCAP_ASIMDDP) *o|=mag_arm64_cap(DOTPROD);
#endif
#ifdef HWCAP2_I8MM
    if (hwcap2 & HWCAP2_I8MM) *o|=mag_arm64_cap(I8MM);
#endif
#ifdef HWCAP_FPHP
    if (hwcap & HWCAP_FPHP) *o|=mag_arm64_cap(F16SCA);
#endif
#ifdef HWCAP_ASIMDHP
    if (hwcap & HWCAP_ASIMDHP) *o|=mag_arm64_cap(F16VEC);
#endif
#ifdef HWCAP2_BF16
    if (hwcap2 & HWCAP2_BF16) *o|=mag_arm64_cap(BF16);
#endif
#ifdef HWCAP_SVE
    if (hwcap & HWCAP_SVE) *o|=mag_arm64_cap(SVE);
#endif
#ifdef HWCAP2_SVE2
    if (hwcap2 & HWCAP2_SVE2) *o|=mag_arm64_cap(SVE2);
#endif
    *sve_width = 0; /* NYI */
#elif defined(_WIN32)
    if (IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE)) *o|=mag_arm64_cap(NEON);
    if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)) *o|=mag_arm64_cap(DOTPROD);
    /* Other features not supported by IsProcessorFeaturePresent*/
    *sve_width = 0; /* NYI */
#elif defined(__APPLE__)
    *o |= mag_arm64_cap(NEON);  /* AArch64 baseline */
    int v = 0;
    size_t n = sizeof(v);
    bool has_fp16 = false;
    v = 0; n = sizeof(v);
    if (sysctlbyname("hw.optional.arm.FEAT_FP16", &v, &n, NULL, 0) == 0 && n == sizeof(v) && v)
        has_fp16 = true;
    else {
        v = 0; n = sizeof(v);
        if (sysctlbyname("hw.optional.neon_fp16", &v, &n, NULL, 0) == 0 && n == sizeof(v) && v)
            has_fp16 = true;
    }
    if (has_fp16) {
        *o |= mag_arm64_cap(F16SCALAR);
        *o |= mag_arm64_cap(F16VECTOR);
        *o |= mag_arm64_cap(F16CVT);
    } else {
        v = 0; n = sizeof(v);
        if (sysctlbyname("hw.optional.AdvSIMD_HPFPCvt", &v, &n, NULL, 0) == 0 && n == sizeof(v) && v) {
            *o |= mag_arm64_cap(F16CVT);
        }
    }
    v = 0; n = sizeof(v);
    if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &n, NULL, 0) == 0 && n == sizeof(v) && v)
        *o |= mag_arm64_cap(DOTPROD);
    v = 0; n = sizeof(v);
    if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &v, &n, NULL, 0) == 0 && n == sizeof(v) && v)
        *o |= mag_arm64_cap(I8MM);
    v = 0; n = sizeof(v);
    if (sysctlbyname("hw.optional.arm.FEAT_BF16", &v, &n, NULL, 0) == 0 && n == sizeof(v) && v)
        *o |= mag_arm64_cap(BF16);
    v = 0; n = sizeof(v);
    if (sysctlbyname("hw.optional.arm.FEAT_SVE", &v, &n, NULL, 0) == 0 && n == sizeof(v) && v)
        *o |= mag_arm64_cap(SVE);
    *sve_width = 0;
#endif
}

void mag_probe_cpu_cache_topology(mag_arm64_cap_bitset_t caps, size_t *ol1, size_t *ol2, size_t *ol3) {
#ifdef __APPLE__
    size_t sz;
    uint64_t v = 0;
    uint32_t pcores = 0;
    sz = sizeof(pcores);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &sz, NULL, 0) != 0) {
        sz = sizeof(pcores);
        if (sysctlbyname("hw.physicalcpu", &pcores, &sz, NULL, 0) != 0) pcores = 1;
    }
    sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.l1dcachesize", &v, &sz, NULL, 0) == 0) *ol1 = v;
    else if (sysctlbyname("hw.l1dcachesize", &v, &sz, NULL, 0) == 0) *ol1 = v;
    else *ol1 = 128ull<<10;
    sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.l2cachesize", &v, &sz, NULL, 0) == 0) *ol2 = v / (pcores*2);
    else if (sysctlbyname("hw.l2cachesize", &v, &sz, NULL, 0) == 0) *ol2 = v / (pcores*2);
    else *ol2 = 3ull<<20;
    sz = sizeof(v);
    if (sysctlbyname("hw.perflevel0.l3cachesize", &v, &sz, NULL, 0) == 0) *ol3 = v;
    else if (sysctlbyname("hw.l3cachesize", &v, &sz, NULL, 0) == 0) *ol3 = v;
    else *ol3 = 64ull<<20;
#else
    *ol1 = 32ull<<10;
    *ol2 = 512ull<<10;
    *ol3 = 1024ull<<10;
#endif
}

#endif
