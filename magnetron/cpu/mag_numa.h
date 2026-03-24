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

#ifndef MAG_NUMA_H
#define MAG_NUMA_H

#include <core/mag_def.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __gnu_linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#endif

typedef struct mag_numa_node_t {
    uint32_t cpus[MAG_MAX_CPUS];
    uint32_t num_cpus;
} mag_numa_node_t;

typedef enum mag_numa_strategy_t {
    MAG_NUMA_STRATEGY_DISABLED,
    MAG_NUMA_STRATEGY_DISTRIBUTE,
    MAG_NUMA_STRATEGY_ISOLATE,
    MAG_NUMA_STRATEGY_NUMACTL
} mag_numa_strategy_t;

typedef struct mag_numa_node_controller_t {
    mag_numa_strategy_t strategy;
    mag_numa_node_t nodes[MAG_MAX_NUMA_NODES];
    uint32_t num_nodes;
    uint32_t num_cpus;
    uint32_t curr_node;
#ifdef __gnu_linux__
    cpu_set_t cpuset;
#else  /* Only Linux support for NUMA right now. TODO: Win32 numa support */
    int cpuset;
#endif
} mag_numa_node_controller_t;

extern bool mag_numa_init(mag_numa_node_controller_t *nodes, mag_numa_strategy_t strategy);
extern bool mag_numa_is_numa(const mag_numa_node_controller_t *nodes);
extern void mag_numa_pin_thread_affinity(mag_numa_node_controller_t *nodes, uint32_t id);
extern void mag_numa_clear_thread_affinity(mag_numa_node_controller_t *nodes);

#ifdef __cplusplus
}
#endif

#endif

