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

typedef void mag_tcp_socket_t;

#define MAG_TCP_ADDR_STRLEN 64

extern MAG_EXPORT void mag_tcp_socket_close(mag_tcp_socket_t *sock);
extern MAG_EXPORT bool mag_tcp_socket_listen(mag_tcp_socket_t **out_sock, uint16_t port, int backlog);
extern MAG_EXPORT uint16_t mag_tcp_socket_local_port(mag_tcp_socket_t *sock);
extern MAG_EXPORT bool mag_tcp_socket_peer_addr(mag_tcp_socket_t *sock, char (*buf)[MAG_TCP_ADDR_STRLEN]);
extern MAG_EXPORT bool mag_tcp_socket_accept(mag_tcp_socket_t **out_sock, mag_tcp_socket_t *listener);
extern MAG_EXPORT bool mag_tcp_socket_connect(mag_tcp_socket_t **out_sock, const char *host, uint16_t port, uint32_t timeout_ms);
extern MAG_EXPORT bool mag_tcp_socket_send_all(mag_tcp_socket_t *sock, const void *buf, size_t nb);
extern MAG_EXPORT bool mag_tcp_socket_recv_all(mag_tcp_socket_t *sock, void *buf, size_t nb);
extern MAG_EXPORT bool mag_tcp_socket_exchange(mag_tcp_socket_t *send_sock, const void *send_buf, size_t send_nb, mag_tcp_socket_t *recv_sock, void *recv_buf, size_t recv_nb);
extern MAG_EXPORT void mag_tcp_socket_sleep_ms(unsigned ms);

#ifdef __cplusplus
}

#endif

#endif
