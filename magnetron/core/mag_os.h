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

#ifndef MAG_OS_H
#define MAG_OS_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

extern MAG_EXPORT char *mag_os_chmod_path(void); /* Current module dir */
extern MAG_EXPORT bool mag_os_sample_crypto_entropy(void *buf, size_t len); /* Sample secure OS bytes */
extern MAG_EXPORT uint64_t mag_os_boot_uid(void); /* Unique system boot ID (used for collective ops, VM safe too) */

#ifdef __cplusplus
}
#endif

#endif
