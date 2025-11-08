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

#include "mag_sstream.h"
#include "mag_alloc.h"

void mag_sstream_init(mag_sstream_t *ss) {
    memset(ss, 0, sizeof(*ss));
    ss->cap = 0x200;
    ss->len = 0;
    ss->buf = (*mag_alloc)(NULL, ss->cap, 0);
    *ss->buf = '\0';
}

void mag_sstream_free(mag_sstream_t *ss) {
    (*mag_alloc)(ss->buf, 0, 0);
    memset(ss, 0, sizeof(*ss));
}

void mag_sstream_reserve_more(mag_sstream_t *ss, size_t extra) {
    size_t want = ss->len+extra+1; /* +1 for terminator */
    if (want <= ss->cap) return;
    while (ss->cap < want) ss->cap <<= 1; /* geometric growth */
    ss->buf = (*mag_alloc)(ss->buf, ss->cap, 0);
}

void mag_sstream_vappend(mag_sstream_t *ss, const char *fmt, va_list ap0) {
    va_list ap;
    va_copy(ap, ap0);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (mag_unlikely(need < 0)) return;
    size_t want = ss->len + (size_t)need+1; /* +1 for terminator */
    if (want > ss->cap) {
        while (ss->cap < want) ss->cap <<= 1; /* geometric growth */
        ss->buf = (*mag_alloc)(ss->buf, ss->cap, 0);
    }
    va_copy(ap, ap0);
    vsnprintf(ss->buf + ss->len, ss->cap - ss->len, fmt, ap);
    va_end(ap);
    ss->len += (size_t)need;
}

void mag_sstream_append(mag_sstream_t *ss, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    mag_sstream_vappend(ss, fmt, ap);
    va_end(ap);
}

void mag_sstream_append_strn(mag_sstream_t *ss, const char *str, size_t len) {
    if (mag_unlikely(!len)) return;
    mag_sstream_reserve_more(ss, len);
    memcpy(ss->buf + ss->len, str, len);
    ss->len += len;
    ss->buf[ss->len] = '\0';
}

void mag_sstream_putc(mag_sstream_t *ss, char c) {
    mag_sstream_reserve_more(ss, 1);
    ss->buf[ss->len++] = c;
    ss->buf[ss->len] = '\0';
}

void mag_sstream_flushf(mag_sstream_t *ss, FILE *f) {
    fputs(ss->buf, f);
}

bool mag_sstream_flush(mag_sstream_t *ss, const char *file){
    FILE *f = mag_fopen(file, "wt");
    if (mag_unlikely(!f)) return false;
    mag_sstream_flushf(ss, f);
    fclose(f);
    return true;
}
