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

#ifndef MAG_SSTREAM_H
#define MAG_SSTREAM_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Dynamic zero-terminated string buffer. */
typedef struct mag_sstream_t {
    char *buf;
    size_t len;
    size_t cap;
} mag_sstream_t;

extern void mag_sstream_init(mag_sstream_t *ss);
extern void mag_sstream_free(mag_sstream_t *ss);
extern void mag_sstream_reserve_more(mag_sstream_t *ss, size_t extra);
extern void mag_sstream_vappend(mag_sstream_t *ss, const char *fmt, va_list ap);
extern void mag_sstream_append(mag_sstream_t *ss, const char *fmt, ...);
extern void mag_sstream_append_strn(mag_sstream_t *ss, const char *str, size_t len);
extern void mag_sstream_putc(mag_sstream_t *ss, char c);
extern void mag_sstream_flushf(mag_sstream_t *ss, FILE *f);
extern bool mag_sstream_flush(mag_sstream_t *ss, const char *file);

#ifdef __cplusplus
}
#endif

#endif
