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
#include <sys/time.h>
#include <poll.h>
#include <unistd.h>
#include <errno.h>
#endif

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define mag_sock_fd(s) ((int)(intptr_t)(s))
#define mag_sock_of(fd) ((mag_tcp_socket_t *)(intptr_t)(fd))

void mag_tcp_socket_close(mag_tcp_socket_t *sock) {
  if (sock)
    close(mag_sock_fd(sock));
}

static bool mag_tcp_socket_set_options(mag_tcp_socket_t *sock) {
  volatile int yes = 1;
  if (mag_unlikely(setsockopt(mag_sock_fd(sock), IPPROTO_TCP, TCP_NODELAY, (const char *)&yes, sizeof(yes)) < 0)) return false;
  if (mag_unlikely(setsockopt(mag_sock_fd(sock), SOL_SOCKET, SO_KEEPALIVE, (const char *)&yes, sizeof(yes)) < 0)) return false;
#ifdef SO_NOSIGPIPE
  if (mag_unlikely(setsockopt(mag_sock_fd(sock), SOL_SOCKET, SO_NOSIGPIPE, (const char *)&yes, sizeof(yes)) < 0)) return false;
#endif
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
  *out_sock = mag_sock_of(sock);
  return true;
}

uint16_t mag_tcp_socket_local_port(mag_tcp_socket_t *sock) {
  struct sockaddr_in addr = {0};
  socklen_t len = sizeof(addr);
  if (mag_unlikely(getsockname(mag_sock_fd(sock), (struct sockaddr *)&addr, &len) < 0)) return 0;
  return ntohs(addr.sin_port);
}

bool mag_tcp_socket_peer_addr(mag_tcp_socket_t *sock, char (*buf)[MAG_TCP_ADDR_STRLEN]) {
  struct sockaddr_in addr = {0};
  socklen_t len = sizeof(addr);
  if (mag_unlikely(getpeername(mag_sock_fd(sock), (struct sockaddr *)&addr, &len) < 0)) return false;
  return inet_ntop(AF_INET, &addr.sin_addr, *buf, sizeof(*buf)) != NULL;
}

bool mag_tcp_socket_accept(mag_tcp_socket_t **out_sock, mag_tcp_socket_t *listener) {
  int sock;
  for (;;) {
    sock = accept(mag_sock_fd(listener), NULL, NULL);
    if (sock >= 0) break;
    if (errno == EINTR) continue;
    return false;
  }
  if (mag_unlikely(!mag_tcp_socket_set_options(mag_sock_of(sock)))) {
    mag_tcp_socket_close(mag_sock_of(sock));
    return false;
  }
  *out_sock = mag_sock_of(sock);
  return true;
}

static uint64_t mag_tcp_now_ms(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t)tv.tv_sec*1000ull + (uint64_t)tv.tv_usec/1000ull;
}

bool mag_tcp_socket_connect(mag_tcp_socket_t **out_sock, const char *host, uint16_t port, uint32_t timeout_ms) {
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (mag_unlikely(inet_pton(AF_INET, host, &addr.sin_addr)) != 1)
    return false;
  uint64_t deadline = timeout_ms ? mag_tcp_now_ms() + timeout_ms : 0;
  uint32_t retries=0;
  for (;;) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (mag_unlikely(sock < 0)) return false;
    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      if (mag_unlikely(!mag_tcp_socket_set_options(mag_sock_of(sock)))) {
        close(sock);
        return false;
      }
      *out_sock = mag_sock_of(sock);
      return true;
    }
    int e = errno;
    close(sock);
    if (deadline && mag_tcp_now_ms() >= deadline) return false;
    if ((retries++ % 50) == 0)
      mag_log_info("Waiting for %s:%u (%s)", host, port, strerror(e));
    mag_tcp_socket_sleep_ms(100);
  }
}

bool mag_tcp_socket_send_all(mag_tcp_socket_t *sock, const void *buf, size_t nb) {
  const uint8_t *p = buf;
  while (nb > 0) {
    ssize_t r = send(mag_sock_fd(sock), p, nb, MSG_NOSIGNAL);
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
    ssize_t r = recv(mag_sock_fd(sock), p, nb, 0);
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

bool mag_tcp_socket_exchange(mag_tcp_socket_t *send_sock, const void *send_buf, size_t send_nb, mag_tcp_socket_t *recv_sock, void *recv_buf, size_t recv_nb) {
  const uint8_t *sp = send_buf;
  uint8_t *rp = recv_buf;
  int sfd = mag_sock_fd(send_sock), rfd = mag_sock_fd(recv_sock);
  while (send_nb || recv_nb) {
    struct pollfd pfd[2];
    int n = 0;
    int si = -1, ri = -1;
    if (sfd == rfd) {
      pfd[0].fd = sfd;
      pfd[0].events = (short)((send_nb ? POLLOUT : 0)|(recv_nb ? POLLIN : 0));
      pfd[0].revents = 0;
      si = ri = 0;
      n = 1;
    } else {
      if (send_nb) {
        pfd[n].fd = sfd;
        pfd[n].events = POLLOUT;
        pfd[n].revents = 0;
        si = n++;
      }
      if (recv_nb) {
        pfd[n].fd = rfd;
        pfd[n].events = POLLIN;
        pfd[n].revents = 0;
        ri = n++;
      }
    }
    int pr = poll(pfd, (nfds_t)n, -1);
    if (mag_unlikely(pr < 0)) {
      if (errno == EINTR) continue;
      return false;
    }
    if (send_nb && si >= 0 && (pfd[si].revents&(POLLOUT|POLLERR|POLLHUP))) {
      ssize_t r = send(sfd, sp, send_nb, MSG_NOSIGNAL|MSG_DONTWAIT);
      if (r < 0 && mag_unlikely(errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)) return false;
      if (mag_unlikely(!r)) return false;
      sp += r;
      send_nb -= (size_t)r;
    }
    if (recv_nb && ri >= 0 && (pfd[ri].revents&(POLLIN|POLLERR|POLLHUP))) {
      ssize_t r = recv(rfd, rp, recv_nb, MSG_DONTWAIT);
      if (r < 0 && mag_unlikely(errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)) return false;
      if (mag_unlikely(!r)) return false;
      rp += r;
      recv_nb -= (size_t)r;
    }
  }
  return true;
}

void mag_tcp_socket_sleep_ms(unsigned ms) {
  usleep(1000*(useconds_t)ms);
}
