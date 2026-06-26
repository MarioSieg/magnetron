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

#ifndef MAG_TCP_SOCKET_H
#define MAG_TCP_SOCKET_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* todo: win32 */

typedef int mag_tcp_socket_t;

extern MAG_EXPORT mag_tcp_socket_t mag_tcp_socket_invalid(void);
extern MAG_EXPORT bool mag_tcp_socket_is_open(mag_tcp_socket_t sock);
extern MAG_EXPORT void mag_tcp_socket_close(mag_tcp_socket_t sock);
extern MAG_EXPORT bool mag_tcp_socket_listen(mag_tcp_socket_t *out_sock, uint16_t port, int backlog);
extern MAG_EXPORT bool mag_tcp_socket_accept(mag_tcp_socket_t *out_sock, mag_tcp_socket_t listener);
extern MAG_EXPORT bool mag_tcp_socket_connect(mag_tcp_socket_t *out_sock, const char *host, uint16_t port);
extern MAG_EXPORT bool mag_tcp_socket_send_all(mag_tcp_socket_t sock, const void *buf, size_t nb);
extern MAG_EXPORT bool mag_tcp_socket_recv_all(mag_tcp_socket_t sock, void *buf, size_t nb);
extern MAG_EXPORT void mag_tcp_socket_sleep_ms(unsigned ms);

#ifdef __cplusplus
}

#endif

#endif
