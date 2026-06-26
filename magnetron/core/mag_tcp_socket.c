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

#include "mag_tcp_socket.h"

#ifdef _WIN32
#error "TODO!"
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#endif

mag_tcp_socket_t mag_tcp_socket_invalid(void) {
  return -1;
}

bool mag_tcp_socket_is_open(mag_tcp_socket_t sock) {
  return sock >= 0;
}

void mag_tcp_socket_close(mag_tcp_socket_t sock) {
  if (mag_tcp_socket_is_open(sock))
    close(sock);
}

bool mag_tcp_socket_set_ops(mag_tcp_socket_t sock) {
  volatile int yes = 1;
  if (mag_unlikely(setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&yes, sizeof(yes)) < 0)) return false;
  if (mag_unlikely(setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (const char *)&yes, sizeof(yes)) < 0)) return false;
  return true;
}

bool mag_tcp_socket_listen(mag_tcp_socket_t *out_sock, uint16_t port, int backlog) {
  mag_tcp_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
  if (mag_unlikely(!mag_tcp_socket_is_open(sock))) return false;
  volatile int yes = 1;
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (mag_unlikely(bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)) {
    mag_tcp_socket_close(sock);
    return false;
  }
  if (mag_unlikely(listen(sock, backlog)) < 0) {
    mag_tcp_socket_close(sock);
    return false;
  }
  *out_sock = sock;
  return true;
}

bool mag_tcp_socket_accept(mag_tcp_socket_t *out_sock, mag_tcp_socket_t listener) {
  mag_tcp_socket_t sock;
  for (;;) {
    sock = accept(listener, NULL, NULL);
    if (mag_tcp_socket_is_open(sock)) break;
    if (errno == EINTR) continue;
    return false;
  }
  if (mag_unlikely(!mag_tcp_socket_set_ops(sock))) {
    mag_tcp_socket_close(sock);
    return false;
  }
  *out_sock = sock;
  return true;
}

bool mag_tcp_socket_connect(mag_tcp_socket_t *out_sock, const char *host, uint16_t port) {
  mag_tcp_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
  if (mag_unlikely(!mag_tcp_socket_is_open(sock))) return false;
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (mag_unlikely(inet_pton(AF_INET, host, &addr.sin_addr)) != 1) {
    mag_tcp_socket_close(sock);
    return false;
  }
  for (uint32_t retries=0;;) {
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) break;
    if (errno == EINTR) continue;
    if ((retries++ % 10) == 0) {
      mag_log_level_t log_level_before = mag_log_level();
      mag_set_log_level(MAG_LOG_LEVEL_INFO);
      mag_log_info("Waiting for %s:%u (%s)\n", host, port, strerror(errno));
      fflush(stdout);
      mag_set_log_level(log_level_before);
    }
    mag_tcp_socket_sleep_ms(100);
  }
  if (mag_unlikely(!mag_tcp_socket_set_ops(sock))) {
    mag_tcp_socket_close(sock);
    return false;
  }
  *out_sock = sock;
  return true;
}

bool mag_tcp_socket_send_all(mag_tcp_socket_t sock, const void *buf, size_t nb) {
  const uint8_t *p = buf;
  while (nb > 0) {
    ssize_t r = send(sock, p, nb, MSG_NOSIGNAL);
    if (mag_unlikely(r < 0)) {
      if (errno == EINTR) continue;
      return false;
    }
    if (mag_unlikely(r == 0)) return false;
    p += r;
    nb -= (size_t)r;
  }
  return true;
}

bool mag_tcp_socket_recv_all(mag_tcp_socket_t sock, void *buf, size_t nb) {
  uint8_t *p = buf;
  while (nb > 0) {
    ssize_t r = recv(sock, p, nb, 0);
    if (mag_unlikely(r < 0)) {
      if (errno == EINTR) continue;
      return false;
    }
    if (mag_unlikely(r == 0)) return false;
    p += r;
    nb -= (size_t)r;
  }
  return true;
}

void mag_tcp_socket_sleep_ms(unsigned ms) {
  usleep(1000*(useconds_t)ms);
}

