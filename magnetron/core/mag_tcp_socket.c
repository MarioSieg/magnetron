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

void mag_tcp_socket_close(mag_tcp_socket_t *sock) {
  if (sock)
    close((int)(intptr_t)sock);
}

static bool mag_tcp_socket_set_options(mag_tcp_socket_t *sock) {
  volatile int yes = 1;
  if (mag_unlikely(setsockopt((int)(intptr_t)sock, IPPROTO_TCP, TCP_NODELAY, (const char *)&yes, sizeof(yes)) < 0)) return false;
  if (mag_unlikely(setsockopt((int)(intptr_t)sock, SOL_SOCKET, SO_KEEPALIVE, (const char *)&yes, sizeof(yes)) < 0)) return false;
  return true;
}

bool mag_tcp_socket_listen(mag_tcp_socket_t **out_sock, uint16_t port, int backlog) {
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (mag_unlikely(sock < 0)) return false;
  volatile int yes = 1;
  setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (mag_unlikely(bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0)) {
    close(sock);
    return false;
  }
  if (mag_unlikely(listen(sock, backlog)) < 0) {
    close(sock);
    return false;
  }
  *out_sock = (void *)(intptr_t)sock;
  return true;
}

bool mag_tcp_socket_accept(mag_tcp_socket_t **out_sock, mag_tcp_socket_t *listener) {
  int sock;
  for (;;) {
    sock = accept((int)(intptr_t)listener, NULL, NULL);
    if (sock >= 0) break;
    if (errno == EINTR) continue;
    return false;
  }
  if (mag_unlikely(!mag_tcp_socket_set_options((void *)(intptr_t)sock))) {
    mag_tcp_socket_close((void *)(intptr_t)sock);
    return false;
  }
  *out_sock = (void *)(intptr_t)sock;
  return true;
}

bool mag_tcp_socket_connect(mag_tcp_socket_t **out_sock, const char *host, uint16_t port) {
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (mag_unlikely(inet_pton(AF_INET, host, &addr.sin_addr)) != 1)
    return false;
  uint32_t retries=0;
  for (;;) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (mag_unlikely(sock < 0)) return false;
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      if (mag_unlikely(!mag_tcp_socket_set_options((void *)(intptr_t)sock))) {
        close(sock);
        return false;
      }
      *out_sock = (void *)(intptr_t)sock;
      return true;
    }
    int e = errno;
    close(sock);
    if ((retries++ % 10) == 0) {
      printf("[magnetron] Waiting for %s:%u (%s)\n", host, port, strerror(e));
      fflush(stdout);
    }
    mag_tcp_socket_sleep_ms(100);
  }
}

bool mag_tcp_socket_send_all(mag_tcp_socket_t *sock, const void *buf, size_t nb) {
  const uint8_t *p = buf;
  while (nb > 0) {
    ssize_t r = send((int)(intptr_t)sock, p, nb, MSG_NOSIGNAL);
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

bool mag_tcp_socket_recv_all(mag_tcp_socket_t *sock, void *buf, size_t nb) {
  uint8_t *p = buf;
  while (nb > 0) {
    ssize_t r = recv((int)(intptr_t)sock, p, nb, 0);
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

