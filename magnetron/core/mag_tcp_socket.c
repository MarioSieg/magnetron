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
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <fcntl.h>
#include <poll.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#endif

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#define mag_sock_fd(s) ((int)(intptr_t)(s))
#define mag_sock_of(fd) ((mag_tcp_socket_t *)(intptr_t)(fd))

#define MAG_TCP_KEEPALIVE_IDLE_S 30
#define MAG_TCP_KEEPALIVE_INTVL_S 5
#define MAG_TCP_KEEPALIVE_CNT 6
#define MAG_TCP_CONNECT_ATTEMPT_MS 5000u
#define MAG_TCP_CONNECT_RETRY_MS 100u

uint64_t mag_tcp_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec*1000ull + (uint64_t)ts.tv_nsec/1000000ull;
}

static uint64_t mag_tcp_deadline(uint32_t timeout_ms) {
  return timeout_ms ? mag_tcp_now_ms() + timeout_ms : 0;
}

static int mag_tcp_poll_timeout(uint64_t deadline) {
  if (!deadline) return -1;
  uint64_t now = mag_tcp_now_ms();
  if (now >= deadline) return 0;
  uint64_t rem = deadline - now;
  return rem > INT32_MAX ? INT32_MAX : (int)rem;
}

void mag_tcp_socket_close(mag_tcp_socket_t *sock) {
  if (sock) close(mag_sock_fd(sock));
}

void mag_tcp_socket_shutdown(mag_tcp_socket_t *sock) {
  if (sock) shutdown(mag_sock_fd(sock), SHUT_RDWR);
}

static bool mag_tcp_socket_set_options(int fd) {
  int yes = 1;
  if (mag_unlikely(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) < 0)) return false;
  if (mag_unlikely(setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes)) < 0)) return false;
#ifdef SO_NOSIGPIPE
  if (mag_unlikely(setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes)) < 0)) return false;
#endif
  int idle = MAG_TCP_KEEPALIVE_IDLE_S, intvl = MAG_TCP_KEEPALIVE_INTVL_S, cnt = MAG_TCP_KEEPALIVE_CNT;
#if defined(TCP_KEEPIDLE)
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
#elif defined(TCP_KEEPALIVE)
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE, &idle, sizeof(idle));
#endif
#ifdef TCP_KEEPINTVL
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
#endif
#ifdef TCP_KEEPCNT
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
#endif
  (void)idle, (void)intvl, (void)cnt;
  return true;
}

static int mag_tcp_bind_listen(int family, uint16_t port, uint32_t backlog) {
  int fd = socket(family, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int yes = 1, no = 0;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  int rc;
  if (family == AF_INET6) {
    setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &no, sizeof(no));
    struct sockaddr_in6 addr = {0};
    addr.sin6_family = AF_INET6;
    addr.sin6_addr = in6addr_any;
    addr.sin6_port = htons(port);
    rc = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  } else {
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    rc = bind(fd, (struct sockaddr *)&addr, sizeof(addr));
  }
  if (rc < 0 || listen(fd, (int)(backlog > INT32_MAX ? INT32_MAX : backlog)) < 0) {
    int e = errno;
    close(fd);
    errno = e;
    return -1;
  }
  return fd;
}

bool mag_tcp_socket_listen(mag_tcp_socket_t **out_sock, uint16_t port, uint32_t backlog) {
  *out_sock = NULL;
  int fd = mag_tcp_bind_listen(AF_INET6, port, backlog);
  if (fd < 0 && errno != EADDRINUSE) fd = mag_tcp_bind_listen(AF_INET, port, backlog);
  if (fd < 0) return false;
  *out_sock = mag_sock_of(fd);
  return true;
}

uint16_t mag_tcp_socket_local_port(mag_tcp_socket_t *sock) {
  struct sockaddr_storage addr = {0};
  socklen_t len = sizeof(addr);
  if (mag_unlikely(getsockname(mag_sock_fd(sock), (struct sockaddr *)&addr, &len) < 0)) return 0;
  if (addr.ss_family == AF_INET6) return ntohs(((struct sockaddr_in6 *)&addr)->sin6_port);
  return ntohs(((struct sockaddr_in *)&addr)->sin_port);
}

static bool mag_tcp_format_addr(const struct sockaddr *sa, socklen_t len, char (*buf)[MAG_TCP_ADDR_STRLEN]) {
  if (sa->sa_family == AF_INET6) {
    const struct sockaddr_in6 *s6 = (const struct sockaddr_in6 *)sa;
    if (IN6_IS_ADDR_V4MAPPED(&s6->sin6_addr))
      return inet_ntop(AF_INET, &s6->sin6_addr.s6_addr[12], *buf, sizeof(*buf)) != NULL;
  }
  return getnameinfo(sa, len, *buf, sizeof(*buf), NULL, 0, NI_NUMERICHOST) == 0;
}

bool mag_tcp_socket_peer_addr(mag_tcp_socket_t *sock, char (*buf)[MAG_TCP_ADDR_STRLEN]) {
  struct sockaddr_storage addr = {0};
  socklen_t len = sizeof(addr);
  if (mag_unlikely(getpeername(mag_sock_fd(sock), (struct sockaddr *)&addr, &len) < 0)) return false;
  return mag_tcp_format_addr((struct sockaddr *)&addr, len, buf);
}

bool mag_tcp_socket_local_addr(mag_tcp_socket_t *sock, char (*buf)[MAG_TCP_ADDR_STRLEN]) {
  struct sockaddr_storage addr = {0};
  socklen_t len = sizeof(addr);
  if (mag_unlikely(getsockname(mag_sock_fd(sock), (struct sockaddr *)&addr, &len) < 0)) return false;
  return mag_tcp_format_addr((struct sockaddr *)&addr, len, buf);
}

bool mag_tcp_iface_addr(const char *ifname, char (*buf)[MAG_TCP_ADDR_STRLEN]) {
  struct ifaddrs *list = NULL;
  if (mag_unlikely(getifaddrs(&list) < 0)) return false;
  const struct ifaddrs *v4 = NULL, *v6 = NULL;
  for (const struct ifaddrs *it = list; it; it = it->ifa_next) {
    if (!it->ifa_addr || !it->ifa_name || strcmp(it->ifa_name, ifname) != 0) continue;
    if (it->ifa_addr->sa_family == AF_INET && !v4) v4 = it;
    else if (it->ifa_addr->sa_family == AF_INET6 && !v6) v6 = it;
  }
  const struct ifaddrs *pick = v4 ? v4 : v6;
  bool ok = false;
  if (pick) {
    socklen_t len = pick->ifa_addr->sa_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
    ok = mag_tcp_format_addr(pick->ifa_addr, len, buf);
  } else {
    errno = ENODEV;
  }
  freeifaddrs(list);
  return ok;
}

bool mag_tcp_socket_accept(mag_tcp_socket_t **out_sock, mag_tcp_socket_t *listener, uint32_t timeout_ms) {
  *out_sock = NULL;
  uint64_t deadline = mag_tcp_deadline(timeout_ms);
  int lfd = mag_sock_fd(listener);
  int fd;
  for (;;) {
    struct pollfd pfd = {.fd=lfd, .events=POLLIN, .revents=0};
    int tmo = mag_tcp_poll_timeout(deadline);
    if (tmo == 0 && deadline) {
      errno = ETIMEDOUT;
      return false;
    }
    int pr = poll(&pfd, 1, tmo);
    if (pr < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (pr == 0) {
      errno = ETIMEDOUT;
      return false;
    }
    if (pfd.revents & POLLNVAL) {
      errno = EBADF;
      return false;
    }
    fd = accept(lfd, NULL, NULL);
    if (fd >= 0) break;
    if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK || errno == ECONNABORTED) continue;
    return false;
  }
  if (mag_unlikely(!mag_tcp_socket_set_options(fd))) {
    int e = errno;
    close(fd);
    errno = e;
    return false;
  }
  *out_sock = mag_sock_of(fd);
  return true;
}

static bool mag_tcp_set_nonblocking(int fd, bool on) {
  int fl = fcntl(fd, F_GETFL, 0);
  if (fl < 0) return false;
  fl = on ? fl|O_NONBLOCK : fl&~O_NONBLOCK;
  return fcntl(fd, F_SETFL, fl) == 0;
}

static int mag_tcp_try_connect(const struct addrinfo *ai, int attempt_ms) {
  int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
  if (fd < 0) return -1;
  if (!mag_tcp_set_nonblocking(fd, true)) goto fail;
  if (connect(fd, ai->ai_addr, ai->ai_addrlen) < 0) {
    if (errno != EINPROGRESS && errno != EINTR) goto fail;
    for (;;) {
      struct pollfd pfd = {.fd=fd, .events=POLLOUT, .revents=0};
      int pr = poll(&pfd, 1, attempt_ms);
      if (pr < 0 && errno == EINTR) continue;
      if (pr <= 0) {
        if (pr == 0) errno = ETIMEDOUT;
        goto fail;
      }
      break;
    }
    int soerr = 0;
    socklen_t len = sizeof(soerr);
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &len) < 0) goto fail;
    if (soerr) {
      errno = soerr;
      goto fail;
    }
  }
  if (!mag_tcp_set_nonblocking(fd, false)) goto fail;
  if (!mag_tcp_socket_set_options(fd)) goto fail;
  return fd;
fail:;
  int e = errno;
  close(fd);
  errno = e;
  return -1;
}

bool mag_tcp_socket_connect(mag_tcp_socket_t **out_sock, const char *host, uint16_t port, uint32_t timeout_ms) {
  *out_sock = NULL;
  if (mag_unlikely(!host || !*host)) {
    errno = EINVAL;
    return false;
  }
  char service[8];
  snprintf(service, sizeof(service), "%u", (unsigned)port);
  uint64_t deadline = mag_tcp_deadline(timeout_ms);
  uint32_t retries = 0;
  for (;;) {
    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int gai = getaddrinfo(host, service, &hints, &res);
    int e = 0;
    if (gai == 0) {
      for (const struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        int tmo = mag_tcp_poll_timeout(deadline);
        if (tmo < 0 || tmo > (int)MAG_TCP_CONNECT_ATTEMPT_MS) tmo = MAG_TCP_CONNECT_ATTEMPT_MS;
        if (tmo == 0) tmo = 1;
        int fd = mag_tcp_try_connect(ai, tmo);
        if (fd >= 0) {
          freeaddrinfo(res);
          *out_sock = mag_sock_of(fd);
          return true;
        }
        e = errno;
      }
      freeaddrinfo(res);
    } else {
      e = gai == EAI_SYSTEM ? errno : EHOSTUNREACH;
    }
    if (deadline && mag_tcp_now_ms() >= deadline) {
      errno = e ? e : ETIMEDOUT;
      return false;
    }
    if ((retries++ % 50) == 0)
      mag_log_info("Waiting for %s:%u (%s)", host, (unsigned)port, gai == 0 ? strerror(e) : gai_strerror(gai));
    mag_tcp_socket_sleep_ms(MAG_TCP_CONNECT_RETRY_MS);
  }
}

static bool mag_tcp_io(int sfd, const uint8_t *sp, size_t snb, int rfd, uint8_t *rp, size_t rnb, uint32_t timeout_ms) {
  uint64_t deadline = mag_tcp_deadline(timeout_ms);
  while (snb || rnb) {
    bool progress = false;
    if (snb) {
      ssize_t r = send(sfd, sp, snb, MSG_NOSIGNAL|MSG_DONTWAIT);
      if (r > 0) {
        sp += r;
        snb -= (size_t)r;
        progress = true;
      } else if (r < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
        return false;
      }
    }
    if (rnb) {
      ssize_t r = recv(rfd, rp, rnb, MSG_DONTWAIT);
      if (r > 0) {
        rp += r;
        rnb -= (size_t)r;
        progress = true;
      } else if (r == 0) {
        errno = ECONNRESET;
        return false;
      } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
        return false;
      }
    }
    if (progress || (!snb && !rnb)) continue;
    struct pollfd pfd[2];
    nfds_t n = 0;
    if (snb && rnb && sfd == rfd) {
      pfd[n++] = (struct pollfd){.fd=sfd, .events=POLLOUT|POLLIN, .revents=0};
    } else {
      if (snb) pfd[n++] = (struct pollfd){.fd=sfd, .events=POLLOUT, .revents=0};
      if (rnb) pfd[n++] = (struct pollfd){.fd=rfd, .events=POLLIN, .revents=0};
    }
    int tmo = mag_tcp_poll_timeout(deadline);
    if (deadline && tmo == 0) {
      errno = ETIMEDOUT;
      return false;
    }
    int pr = poll(pfd, n, tmo);
    if (pr < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (pr == 0) {
      errno = ETIMEDOUT;
      return false;
    }
    for (nfds_t i=0; i < n; ++i) {
      if (pfd[i].revents & POLLNVAL) {
        errno = EBADF;
        return false;
      }
    }
  }
  return true;
}

bool mag_tcp_socket_send_all(mag_tcp_socket_t *sock, const void *buf, size_t nb, uint32_t timeout_ms) {
  return mag_tcp_io(mag_sock_fd(sock), buf, nb, -1, NULL, 0, timeout_ms);
}

bool mag_tcp_socket_recv_all(mag_tcp_socket_t *sock, void *buf, size_t nb, uint32_t timeout_ms) {
  return mag_tcp_io(-1, NULL, 0, mag_sock_fd(sock), buf, nb, timeout_ms);
}

bool mag_tcp_socket_exchange(
  mag_tcp_socket_t *send_sock,
  const void *send_buf,
  size_t send_nb,
  mag_tcp_socket_t *recv_sock,
  void *recv_buf,
  size_t recv_nb,
  uint32_t timeout_ms
) {
  return mag_tcp_io(mag_sock_fd(send_sock), send_buf, send_nb, mag_sock_fd(recv_sock), recv_buf, recv_nb, timeout_ms);
}

bool mag_tcp_reserve_fds(uint32_t needed) {
  struct rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) < 0) return true;
  if (rl.rlim_cur == RLIM_INFINITY || rl.rlim_cur >= needed) return true;
  if (rl.rlim_max != RLIM_INFINITY && rl.rlim_max < needed) {
    errno = EMFILE;
    return false;
  }
  rl.rlim_cur = needed;
  if (setrlimit(RLIMIT_NOFILE, &rl) < 0) return false;
  return true;
}

void mag_tcp_socket_sleep_ms(unsigned ms) {
  usleep(1000*(useconds_t)ms);
}
