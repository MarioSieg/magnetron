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

#ifndef MAG_OS_H
#define MAG_OS_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

extern char *mag_current_module_path(void);
extern bool mag_sec_crypto_entropy(void *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif
