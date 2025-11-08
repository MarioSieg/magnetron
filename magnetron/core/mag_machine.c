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

#include "mag_machine.h"
#include "mag_cpuid.h"

#include <string.h>
#include <ctype.h>
#include <errno.h>

#ifdef _WIN32
#else
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif
#include <unistd.h>
#endif

#ifdef __APPLE__
static bool mag_sysctl_mib01(uint8_t (*out)[256], size_t *o_len, int mib0, int mib1) { /* Get sysctl data */
    memset(out, 0, sizeof(*out));
    *o_len = 0;
    int name[2] = {mib0, mib1};
    size_t len = 0;
    if (mag_unlikely(sysctl(name, sizeof(name) / sizeof(*name), NULL, &len, NULL, 0))) return false; /* Get length */
    if (mag_unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
    if (mag_unlikely(sysctl(name, sizeof(name) / sizeof(*name), *out, &len, NULL, 0))) return false; /* Get data */
    *o_len = len;
    return true;
}
static bool mag_sysctl_key(uint8_t (*out)[256], size_t *o_len, const char *key) { /* Get sysctl data */
    memset(out, 0, sizeof(*out));
    *o_len = 0;
    size_t len = 0;
    if (mag_unlikely(sysctlbyname(key, NULL, &len, NULL, 0))) return false; /* Get length */
    if (mag_unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
    if (mag_unlikely(sysctlbyname(key, *out, &len, NULL, 0))) return false; /* Get data */
    *o_len = len;
    return true;
}
static uint64_t mag_sysctl_unpack_int(const uint8_t (*in)[256], size_t len) { /* Unpack sysctl data */
    switch (len) {
    case sizeof(uint16_t): {
        uint16_t r;
        memcpy(&r, *in, sizeof(r));
        return r;
    }
    case sizeof(uint32_t): {
        uint32_t r;
        memcpy(&r, *in, sizeof(r));
        return r;
    }
    case sizeof(uint64_t): {
        uint64_t r;
        memcpy(&r, *in, sizeof(r));
        return r;
    }
    default:
        return 0;
    }
}
#else
static bool mag_cpuinfo_parse_value(const char *key, char (*out)[128]) {
    FILE *cpuinfo = mag_fopen("/proc/cpuinfo", "rt");
    if (mag_unlikely(!cpuinfo)) return false;
    size_t key_len = strlen(key);
    char line[128];
    while (fgets(line, sizeof(line), cpuinfo)) {
        size_t line_len = strlen(line);
        if (line_len > 0 && line[line_len-1] == '\n') line[line_len-1] = '\0';
        if (strncmp(line, key, key_len) == 0 && (isspace((unsigned char)line[key_len]) || line[key_len] == ':')) {
            char *colon = strchr(line, ':');
            if (!colon) continue;
            char *value = colon+1;
            while (isspace((unsigned char)*value)) ++value;
            char *end = value + strlen(value);
            for (; end > value && isspace((unsigned char)*(end-1)); --end);
            *end = '\0';
            size_t value_len = llabs(end-value);
            if (mag_unlikely(!value_len || value_len >= sizeof(*out))) {
                fclose(cpuinfo);
                return false;
            }
            snprintf(*out, sizeof(*out), "%s", value);
            fclose(cpuinfo);
            return true;
        }
    }
    fclose(cpuinfo);
    return false;
}
static uint64_t mag_parse_meminfo_value(const char *line) {
    const char *p = strchr(line, ':');
    if (mag_unlikely(!p)) return 0;
    ++p;
    p += strspn(p, " \t");
    errno = 0;
    char *end;
    uint64_t value = strtoull(p, &end, 10);
    if (mag_unlikely(errno != 0 || p == end)) return 0;
    return value<<10;
}
#endif

#ifdef __linux__
static void mag_trim_quotes(char *in) {
    if (in == NULL || *in == '\0') return;
    size_t len = strlen(in);
    if (in[len - 1] == '"') {
        in[len - 1] = '\0';
        len--;
    }
    if (in[0] == '"') {
        memmove(in, in + 1, len);
    }
}
#endif

static void mag_machine_probe_os_name(char (*out_os_name)[128]) { /* Get OS name */
#ifdef _WIN32

#elif defined(__APPLE__)
    size_t len;
    uint8_t tmp[256];
    if (mag_likely(mag_sysctl_mib01(&tmp, &len, CTL_KERN, KERN_VERSION) && len && *tmp)) {
        char *colon = strchr((const char *)tmp, ':');
        if (colon) *colon = '\0';
        snprintf(*out_os_name, sizeof(*out_os_name), "%s", (const char *)tmp);
    }
#elif defined (__linux__)
    FILE *f = mag_fopen("/etc/os-release", "r");
    if (!f) {
        f = mag_fopen("/usr/lib/os-release", "r");
        if (!f) {
            f = mag_fopen("/etc/lsb-release", "r");
            if (mag_unlikely(!f)) return;
            char line[256];
            while (fgets(line, sizeof(line), f) != NULL) {
                size_t len = strlen(line);
                if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
                if (strncmp(line, "DISTRIB_ID", sizeof("DISTRIB_ID")-1) == 0) {
                    char *equals_sign = strchr(line, '=');
                    if (equals_sign && *(equals_sign+1)) {
                        strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                        (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                    }
                } else if (strncmp(line, "DISTRIB_DESCRIPTION", sizeof("DISTRIB_DESCRIPTION")-1) == 0) {
                    char *equals_sign = strchr(line, '=');
                    if (equals_sign && *(equals_sign+1)) {
                        char *start_quote = strchr(equals_sign+1, '"');
                        if (start_quote) {
                            char *end_quote = strchr(start_quote+1, '"');
                            if (end_quote) {
                                size_t desc_len = end_quote-start_quote-1;
                                if (desc_len >= sizeof(*out_os_name)) desc_len = sizeof(*out_os_name)-1;
                                strncpy(*out_os_name, start_quote+1, desc_len);
                                (*out_os_name)[desc_len] = '\0';
                            } else {
                                strncpy(*out_os_name, start_quote+1, sizeof(*out_os_name)-1);
                                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                            }
                        } else {
                            strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                            (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                        }
                    }
                }
            }
            fclose(f);
            return;
        }
    }
    char line[256];
    while (fgets(line, sizeof(line), f) != NULL) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (strncmp(line, "NAME", sizeof("NAME")-1) == 0) {
            char *equals_sign = strchr(line, '=');
            if (equals_sign && *(equals_sign+1)) {
                strncpy(*out_os_name, equals_sign + 1, sizeof(*out_os_name)-1);
                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
            }
        } else if (strncmp(line, "PRETTY_NAME", sizeof("PRETTY_NAME")-1) == 0) {
            char *equals_sign = strchr(line, '=');
            if (equals_sign && *(equals_sign+1)) {
                strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
            }
        }
    }
    fclose(f);
    mag_trim_quotes(*out_os_name);
#endif
}

static void mag_machine_probe_cpu_name(char (*out_cpu_name)[128]) { /* Get CPU name */
#ifdef _WIN32
    HKEY key;
    if (mag_unlikely(RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key))) return;
    char tmp[64+1] = {0};
    DWORD len = sizeof(tmp);
    if (mag_unlikely(RegQueryValueExA(key, "ProcessorNameString", NULL, NULL, (LPBYTE)tmp, &len))) return;
    if (mag_likely(strlen(tmp))) tmp[strlen(tmp)-1] = '\0';
    snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", tmp);
#elif defined(__APPLE__)
    size_t len;
    uint8_t tmp[256];
    if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.brand_string") && len && *tmp))
        snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", (const char *)tmp);
#else
    char cpu_name[128];
    if (mag_likely((mag_cpuinfo_parse_value("model name", &cpu_name) && *cpu_name) || (mag_cpuinfo_parse_value("Model", &cpu_name) && *cpu_name)))
        snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", cpu_name);
#endif
}

static void mag_machine_probe_cpu_cores(uint32_t *out_virtual, uint32_t *out_physical, uint32_t *out_sockets) { /* Get CPU virtual (logical) cores. */
#ifdef _WIN32
    DWORD size = 0;
    GetLogicalProcessorInformation(NULL, &size);
    if (mag_unlikely(!size)) return;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *info = (*mag_alloc)(NULL, size, 0);
    if (mag_unlikely(!GetLogicalProcessorInformation(info, &size))) goto end;
    for (DWORD i=0; i < size/sizeof(*info); ++i) {
        switch (info[i].Relationship) {
        default:
            continue;
        case RelationProcessorPackage:
            ++*out_sockets;
            continue;
        case RelationProcessorCore: {
            ++*out_physical;
            uintptr_t m = (uintptr_t)info[i].ProcessorMask;
            m = m - ((m>>1) & 0x5555555555555555);
            m = (m & 0x3333333333333333) + ((m>>2) & 0x3333333333333333);
            *out_virtual += (((m + (m>>4)) & 0xf0f0f0f0f0f0f0f) * 0x101010101010101)>>56;
        }
        continue;
        }
    }
end:
    (*mag_alloc)(info, 0, 0);
#elif defined(__APPLE__)
    uint8_t tmp[256];
    size_t len;
    if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.thread_count") && len))
        *out_virtual = mag_sysctl_unpack_int(&tmp, len);
    if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.core_count") && len))
        *out_physical = mag_sysctl_unpack_int(&tmp, len);
    if (mag_likely(mag_sysctl_key(&tmp, &len, "hw.packages") && len))
        *out_sockets = mag_sysctl_unpack_int(&tmp, len);
#else
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    FILE *cpuinfo = mag_fopen("/proc/cpuinfo", "r");
    if (mag_unlikely(!cpuinfo)) return;
    uint32_t physical_ids[MAG_MAX_CPUS];
    uint32_t core_ids[MAG_MAX_CPUS];
    uint32_t package_ids[MAG_MAX_CPUS];
    uint32_t cpu_count = 0;
    uint32_t package_count = 0;
    uint32_t current_physical_id = 0;
    uint32_t current_core_id = 0;
    bool got_physical_id = false;
    bool got_core_id = false;
    char line[256];
    while (fgets(line, sizeof(line), cpuinfo) != NULL) {
        if (strncmp(line, "physical id", sizeof("physical id")-1) == 0) {
            char *ptr = strchr(line, ':');
            if (ptr) {
                ++ptr;
                for (; *ptr && !isdigit((unsigned char)*ptr); ++ptr);
                if (*ptr) {
                    current_physical_id = (uint32_t)strtoul(ptr, NULL, 10);
                    got_physical_id = true;
                }
            }
        } else if (strncmp(line, "core id", sizeof("core id")-1) == 0) {
            char *ptr = strchr(line, ':');
            if (ptr) {
                ++ptr;
                for (; *ptr && !isdigit((unsigned char)*ptr); ++ptr);
                if (*ptr) {
                    current_core_id = (uint32_t)strtoul(ptr, NULL, 10);
                    got_core_id = true;
                }
            }
        } else if (*line == '\n') {
            if (got_physical_id && got_core_id) {
                bool is_unique = true;
                for (int32_t i=0; i < cpu_count; ++i) if (physical_ids[i] == current_physical_id && core_ids[i] == current_core_id) {
                        is_unique = false;
                        break;
                    }
                if (is_unique) {
                    if (cpu_count < MAG_MAX_CPUS) {
                        physical_ids[cpu_count] = current_physical_id;
                        core_ids[cpu_count] = current_core_id;
                        ++cpu_count;
                    } else break;
                }
                is_unique = true;
                for (int32_t i=0; i < package_count; ++i) if (package_ids[i] == current_physical_id) {
                        is_unique = false;
                        break;
                    }
                if (is_unique) {
                    if (package_count < MAG_MAX_CPUS) package_ids[package_count++] = current_physical_id;
                    else break;
                }
            }
            got_physical_id = false;
            got_core_id = false;
        }
    }
    fclose(cpuinfo);
    *out_virtual = nprocs > 0 ? (uint32_t)nprocs : 0;
    if (!cpu_count && *out_virtual) cpu_count = *out_virtual;
    *out_physical = mag_xmax(1, cpu_count);
    *out_virtual = nprocs > 0 ? (uint32_t)nprocs : *out_physical;
    *out_sockets = mag_xmax(1, package_count);
#endif
}

static void mag_machine_probe_memory(size_t *out_phys_mem_total, size_t *out_phys_mem_free) { /* Get physical memory */
#ifdef _WIN32
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (mag_likely(GlobalMemoryStatusEx(&mem))) {
        *out_phys_mem_total = mem.ullTotalPhys;
        *out_phys_mem_free = mem.ullAvailPhys;
    }
#elif defined(__APPLE__)
    uint8_t tmp[256];
    size_t len;
    if (mag_likely(mag_sysctl_mib01(&tmp, &len, CTL_HW, HW_MEMSIZE) && len))
        *out_phys_mem_total = mag_sysctl_unpack_int(&tmp, len);
    struct vm_statistics64 stats;
    natural_t count = HOST_VM_INFO64_COUNT;
    if (mag_likely(host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)(&stats), &count) == KERN_SUCCESS))
        *out_phys_mem_free = stats.free_count * getpagesize();
#else
    FILE *meminfo = mag_fopen("/proc/meminfo", "r");
    if (mag_unlikely(!meminfo)) return;
    char line[256];
    while (fgets(line, sizeof(line), meminfo)) {
        if (strncmp(line, "MemTotal:", sizeof("MemTotal:")-1) == 0)
            *out_phys_mem_total = mag_parse_meminfo_value(line);
        else if (strncmp(line, "MemAvailable:", sizeof("MemAvailable:")-1) == 0)
            *out_phys_mem_free = mag_parse_meminfo_value(line);
    }
    fclose(meminfo);
#endif
}

void mag_machine_info_probe(mag_machine_info_t *ma) {
    mag_machine_probe_os_name(&ma->os_name);
    mag_machine_probe_cpu_name(&ma->cpu_name);
    mag_machine_probe_cpu_cores(&ma->cpu_virtual_cores, &ma->cpu_physical_cores, &ma->cpu_sockets);
    mag_machine_probe_memory(&ma->phys_mem_total, &ma->phys_mem_free);
    uint64_t caps = 0;
#if defined(__x86_64__) || defined(_M_X64)
    mag_probe_cpu_amd64(&ma->amd64_cpu_caps, &ma->amd64_avx10_ver);
    caps = ma->amd64_cpu_caps;
#elif defined(__aarch64__)
    mag_probe_cpu_arm64(&ma->arm64_cpu_caps, &ma->arm64_cpu_sve_width);
    caps = ma->arm64_cpu_caps;
#endif
    mag_probe_cpu_cache_topology(caps, &ma->cpu_l1_size, &ma->cpu_l2_size, &ma->cpu_l3_size);
    if (mag_unlikely(!*ma->os_name)) snprintf(ma->os_name, sizeof(ma->os_name), "Unknown");
    if (mag_unlikely(!*ma->cpu_name)) snprintf(ma->cpu_name, sizeof(ma->cpu_name), "Unknown");
}
