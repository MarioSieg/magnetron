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

#include "mag_numa.h"

#ifdef __gnu_linux__
#include <syscall.h>
#include <pthread.h>
#include <sys/stat.h>

static cpu_set_t mag_query_numa_affinity(void) {
    cpu_set_t cpuset;
    pthread_t thread = pthread_self();
    CPU_ZERO(&cpuset);
    int ret = pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (mag_unlikely(ret)) {
        CPU_ZERO(&cpuset);
        mag_log_warn("pthread_getaffinity_np() failed: %s", strerror(ret));
    }
    return cpuset;
}

#else
static int mag_query_numa_affinity(void) {
   return 0; /* TODO */
}
#endif

bool mag_numa_init(mag_numa_node_controller_t *nodes, mag_numa_strategy_t strategy) {
    if (nodes->num_nodes > 0) return true;
    memset(nodes, 0, sizeof(*nodes));
    nodes->strategy = strategy;
    nodes->cpuset = mag_query_numa_affinity();
#ifdef __gnu_linux__
    int ret;
    char pathbuf[512];
    struct stat st;
    while (nodes->num_nodes < MAG_MAX_NUMA_NODES) {
        ret = snprintf(pathbuf, sizeof(pathbuf), "/sys/devices/system/node/node%u", nodes->num_nodes);
        if (mag_unlikely(!(ret > 0 && (unsigned)ret < sizeof(pathbuf)))) return false;
        if (stat(pathbuf, &st) != 0) { break; }
        ++nodes->num_nodes;
    }
    while (nodes->num_cpus < MAG_MAX_CPUS) {
        ret = snprintf(pathbuf, sizeof(pathbuf), "/sys/devices/system/cpu/cpu%u", nodes->num_cpus);
        if (mag_unlikely(!(ret > 0 && (unsigned)ret < sizeof(pathbuf)))) return false;
        if (stat(pathbuf, &st) != 0) { break; }
        ++nodes->num_cpus;
    }
    unsigned curr_cpu;
    #if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ > 33) || defined(__COSMOPOLITAN__)
        ret = getcpu(&curr_cpu, &nodes->curr_node);
    #else
        #if !defined(SYS_getcpu) && defined(SYS_get_cpu)
        #define SYS_getcpu SYS_get_cpu
        #endif
        ret = syscall(SYS_getcpu, &curr_cpu, &nodes->curr_node);
    #endif
    if (nodes->num_nodes < 1 || nodes->num_cpus < 1 || ret != 0) {
        nodes->num_nodes = 0; /* We just have no numa nodes */
        mag_log_info("NUMA topology set to single node");
        return true;
    }
    mag_log_debug("Found current process on NUMA node %u, CPU %u", nodes->curr_node, curr_cpu);
    for (uint32_t i=0; i < nodes->num_nodes; ++i) {
        mag_numa_node_t *node = nodes->nodes+i;
        mag_log_debug("CPUs on node %u:", i);
        node->num_cpus = 0;
        for (uint32_t k=0; k < nodes->num_cpus; ++k) {
            ret = snprintf(pathbuf, sizeof(pathbuf), "/sys/devices/system/node/node%u/cpu%u", i, k);
            if (mag_unlikely(!(ret > 0 && (unsigned)ret < sizeof(pathbuf)))) return false;
            if (stat(pathbuf, &st) == 0) {
                node->cpus[node->num_cpus++] = k;
                mag_log_debug("\t%u", k);
            }
        }
    }
    if (nodes->num_nodes > 0) {
        FILE *f = mag_fopen("/proc/sys/kernel/numa_balancing", "r");
        if (f) {
            char buf[42];
            if (fgets(buf, sizeof(buf), f) && strncmp(buf, "0\n", sizeof("0\n")-1) != 0) {
                mag_log_warn("/proc/sys/kernel/numa_balancing is enabled, this has been observed to reduce perf");
            }
            fclose(f);
        }
    }
    return true;
#else
    /* TODO */
    return true;
#endif
}

bool mag_numa_is_numa(const mag_numa_node_controller_t *nodes) {
    return nodes->num_nodes > 1;
}

void mag_numa_pin_thread_affinity(mag_numa_node_controller_t *nodes, uint32_t id) {
    if (!mag_numa_is_numa(nodes)) return;
#ifdef __gnu_linux__
    uint32_t node_num=0;
    int ret;
    size_t setsize = CPU_ALLOC_SIZE(nodes->num_cpus);
    switch(nodes->strategy) {
        case MAG_NUMA_STRATEGY_DISTRIBUTE: node_num = id % nodes->num_nodes; break;
        case MAG_NUMA_STRATEGY_ISOLATE: node_num = nodes->curr_node; break;
        case MAG_NUMA_STRATEGY_NUMACTL:
            ret = pthread_setaffinity_np(pthread_self(), sizeof(nodes->cpuset), &nodes->cpuset);
            if (mag_unlikely(ret)) mag_log_warn("pthread_setaffinity_np() failed: %s", strerror(ret));
            return;
        default: return;
    }
    mag_numa_node_t *node = nodes->nodes+node_num;
    cpu_set_t *set = CPU_ALLOC(nodes->num_cpus);
    if (mag_unlikely(!set)) return;
    CPU_ZERO_S(setsize, set);
    for (size_t i=0; i < node->num_cpus; ++i)
        CPU_SET_S(node->cpus[i], setsize, set);
    ret = pthread_setaffinity_np(pthread_self(), setsize, set);
    if (mag_unlikely(ret)) mag_log_warn("pthread_setaffinity_np() failed: %s", strerror(ret));
    CPU_FREE(set);
#endif
}

void mag_numa_clear_thread_affinity(mag_numa_node_controller_t *nodes) {
    if (!mag_numa_is_numa(nodes)) return;
#ifdef __gnu_linux__
    int ret = pthread_setaffinity_np(pthread_self(), sizeof(nodes->cpuset), &nodes->cpuset);
    if (mag_unlikely(ret)) {
        mag_log_warn("pthread_setaffinity_np() failed: %s", strerror(ret));
    }
#endif
}
