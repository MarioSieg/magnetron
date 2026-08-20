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
