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

#ifndef MAG_MACHINE_H
#define MAG_MACHINE_H

#include "mag_cpuid.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_machine_info_t {
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
} mag_machine_info_t;

extern void mag_machine_info_probe(mag_machine_info_t *ma);

#ifdef __cplusplus
}
#endif

#endif
