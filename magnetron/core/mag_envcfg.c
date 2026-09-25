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

#include "mag_envcfg.h"

const char *mag_envcfg_raw(const char *name) {
  const char *v = getenv(name);
  return v && *v ? v : NULL;
}

void mag_envcfg_apply_log_level(void) {
  const char *v = mag_envcfg_raw(MAG_ENV_LOG_LEVEL);
  if (!v) return;
  if (mag_casecmp(v, "off")) mag_set_log_level(MAG_LOG_LEVEL_NONE);
  else if (mag_casecmp(v, "error")) mag_set_log_level(MAG_LOG_LEVEL_ERROR);
  else if (mag_casecmp(v, "warn") || mag_casecmp(v, "warning")) mag_set_log_level(MAG_LOG_LEVEL_WARN);
  else if (mag_casecmp(v, "info")) mag_set_log_level(MAG_LOG_LEVEL_INFO);
  else if (mag_casecmp(v, "debug")) mag_set_log_level(MAG_LOG_LEVEL_DEBUG);
  else mag_log_error("Invalid " MAG_ENV_LOG_LEVEL " value '%s' (valid: off, error, warn, info, debug)", v);
}

mag_envcfg_cpu_specialization_t mag_envcfg_cpu_specialization_level(const char **out_name) {
  const char *v = mag_envcfg_raw(MAG_ENV_CPU_SPECIALIZATION_LEVEL);
  if (!v) return MAG_ENVCFG_CPU_SPECIALIZATION_AUTO;
  if (mag_casecmp(v, "fallback") || mag_casecmp(v, "generic")) return MAG_ENVCFG_CPU_SPECIALIZATION_FALLBACK;
  *out_name = v;
  return MAG_ENVCFG_CPU_SPECIALIZATION_PINNED;
}

uint32_t mag_envcfg_cpu_threads(uint32_t fallback) {
  const char *v = mag_envcfg_raw(MAG_ENV_CPU_THREADS);
  if (!v) return fallback;
  char *end = NULL;
  unsigned long n = strtoul(v, &end, 10);
  if (end == v || *end || n < 1 || n > 0xffffu) {
    mag_log_error("Invalid " MAG_ENV_CPU_THREADS " value '%s' (expected int >= 0)", v);
    return fallback;
  }
  return (uint32_t)n;
}

uint64_t mag_envcfg_readahead_mb(uint64_t fallback) {
  const char *v = mag_envcfg_raw(MAG_ENV_READAHEAD_MB);
  if (!v) return fallback;
  char *end = NULL;
  unsigned long long n = strtoull(v, &end, 10);
  if (end == v || *end || n > 0x100000ull) {
    mag_log_error("Invalid " MAG_ENV_READAHEAD_MB " value '%s' (expected int in [0, 1048576])", v);
    return fallback;
  }
  return n;
}

int mag_envcfg_readahead_release(void) {
  const char *v = mag_envcfg_raw(MAG_ENV_READAHEAD_RELEASE);
  if (!v || mag_casecmp(v, "auto")) return -1;
  if (!strcmp(v, "0")) return 0;
  if (!strcmp(v, "1")) return 1;
  mag_log_error("Invalid " MAG_ENV_READAHEAD_RELEASE " value '%s' (expected 0, 1 or auto)", v);
  return -1;
}

int mag_envcfg_numa_strategy(int fallback) {
  const char *v = mag_envcfg_raw(MAG_ENV_NUMA_STRATEGY);
  if (!v) return fallback;
  if (mag_casecmp(v, "disabled") || mag_casecmp(v, "off")) return 0;
  if (mag_casecmp(v, "distribute")) return 1; /* vals of mag_numa_strategy_t in cpu */
  if (mag_casecmp(v, "isolate")) return 2;
  if (mag_casecmp(v, "numactl")) return 3;
  mag_log_error("Invalid " MAG_ENV_NUMA_STRATEGY " value '%s' (valid: disabled, distribute, isolate, numactl)", v);
  return fallback;
}
