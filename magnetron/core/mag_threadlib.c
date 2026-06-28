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

#include "mag_threadlib.h"
#include "mag_alloc.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <synchapi.h>
#elif defined __APPLE__
#include <unistd.h>
#include <pthread.h>
/* Stuff for macOS's not entirely public ulock API which provides futex like functionality (: */
#define UL_COMPARE_AND_WAIT 1
#define UL_UNFAIR_LOCK 2
#define UL_COMPARE_AND_WAIT_SHARED 3
#define UL_UNFAIR_LOCK64_SHARED 4
#define UL_COMPARE_AND_WAIT64 5
#define UL_COMPARE_AND_WAIT64_SHARED 6
#define UL_OSSPINLOCK UL_COMPARE_AND_WAIT
#define UL_HANDOFFLOCK UL_UNFAIR_LOCK
#define ULF_WAKE_ALL 0x00000100
#define ULF_WAKE_THREAD 0x00000200
#define ULF_WAKE_ALLOW_NON_OWNER 0x00000400
__attribute__((weak_import)) extern int __ulock_wait(uint32_t op, void *addr, uint64_t value, uint32_t timeout);
__attribute__((weak_import)) extern int __ulock_wake(uint32_t op, void *addr, uint64_t value);
#elif defined(__linux__)
#include <unistd.h>
#include <pthread.h>
#include <linux/futex.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif

typedef struct mag_thread_entry_params_t {
  void (*entry)(void *);
  void *arg;
  mag_thread_prio_t prio;
  char name[64];
} mag_thread_entry_params_t;

static void *mag_thread_entry_stub(void *arg) {
  mag_thread_entry_params_t params = *( mag_thread_entry_params_t*)arg;
  (*mag_alloc)(arg, 0, 0);
  if (params.prio != MAG_THREAD_PRIO_NORMAL) mag_curr_thread_set_prio(params.prio);
  if (*params.name) mag_curr_thread_set_name(params.name);
  void (*e)(void *) = params.entry;
  (*e)(params.arg);
  return NULL;
}

mag_status_t mag_thread_create(
  mag_error_t *err,
  mag_thread_t **out,
  void (*entry)(void *),
  mag_thread_prio_t prio,
  const char *name,
  void *arg
) {
  mag_thread_entry_params_t *params = (*mag_try_alloc)(err, sizeof(mag_thread_entry_params_t), 0);
  if (mag_unlikely(!params))
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "mag_threadlib: failed to allocate thread entry params.");
  *params = (mag_thread_entry_params_t){
    .entry = entry,
    .arg = arg,
    .prio = prio,
    .name = ""
  };
  if (name && *name) snprintf(params->name, sizeof(params->name), "%s", name);
#ifdef _WIN32
#else
  pthread_t tr;
  if (mag_unlikely(pthread_create(&tr, NULL, &mag_thread_entry_stub, params) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to create thread.");
#endif
  *out = tr;
  return MAG_STATUS_OK;
}

mag_status_t mag_thread_join(mag_error_t *err, mag_thread_t *thr) {
#ifdef _WIN32
#else
  if (mag_unlikely(pthread_join(thr, NULL) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: Failed to join thread.");
#endif
  return MAG_STATUS_OK;
}

mag_status_t mag_mutex_create(mag_error_t *err, mag_mutex_t **out) {
  pthread_mutex_t *mtx = (*mag_try_alloc)(NULL, sizeof(pthread_mutex_t), __alignof(pthread_mutex_t));
  if (mag_unlikely(!mtx))
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "mag_threadlib: failed to allocate mutex");
  if (mag_unlikely(pthread_mutex_init(mtx, NULL) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to create mutex");
  *out = mtx;
  return MAG_STATUS_OK;
}

mag_status_t mag_mutex_destroy(mag_error_t *err, mag_mutex_t *mtx) {
  if (mag_unlikely(pthread_mutex_destroy(mtx) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to destroy mutex");
  (*mag_try_alloc)(mtx, 0, 0);
  return MAG_STATUS_OK;
}

mag_status_t mag_mutex_lock(mag_error_t *err, mag_mutex_t *mtx) {
  if (mag_unlikely(pthread_mutex_lock(mtx) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to lock mutex.");
  return MAG_STATUS_OK;
}

mag_status_t mag_mutex_unlock(mag_error_t *err, mag_mutex_t *mtx) {
  if (mag_unlikely(pthread_mutex_unlock(mtx) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to unlock mutex.");
  return MAG_STATUS_OK;
}

mag_status_t mag_condvar_create(mag_error_t *err, mag_condvar_t **out) {
  pthread_cond_t *cv = (*mag_try_alloc)(NULL, sizeof(pthread_cond_t), __alignof(pthread_cond_t));
  if (mag_unlikely(!cv))
    return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "mag_threadlib: failed to allocate condvar");
  if (mag_unlikely(pthread_cond_init(cv, NULL) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to create condvar");
  *out = cv;
  return MAG_STATUS_OK;
}

mag_status_t mag_condvar_destroy(mag_error_t *err, mag_condvar_t *cv) {
  if (mag_unlikely(pthread_cond_destroy(cv) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to destroy condvar");
  (*mag_try_alloc)(cv, 0, 0);
  return MAG_STATUS_OK;
}

mag_status_t mag_condvar_wait(mag_error_t *err, mag_condvar_t *cv, mag_mutex_t *mtx) {
  if (mag_unlikely(pthread_cond_wait(cv, mtx) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to wait on condvar.");
  return MAG_STATUS_OK;
}

mag_status_t mag_condvar_signal(mag_error_t *err, mag_condvar_t *cv) {
  if (mag_unlikely(pthread_cond_signal(cv) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to signal condvar.");
  return MAG_STATUS_OK;
}

mag_status_t mag_condvar_broadcast(mag_error_t *err, mag_condvar_t *cv) {
  if (mag_unlikely(pthread_cond_broadcast(cv) != 0))
    return mag_set_error(err, MAG_STATUS_ERR_OS_ERROR, "mag_threadlib: failed to broadcast condvar.");
  return MAG_STATUS_OK;
}

/* Set scheduling priority for current thread. */
void mag_curr_thread_set_prio(mag_thread_prio_t prio) {
#ifdef _WIN32
  DWORD policy = THREAD_PRIORITY_NORMAL;
  switch (prio) {
  case MAG_THREAD_PRIO_NORMAL:
    policy = THREAD_PRIORITY_NORMAL;
    break;
  case MAG_THREAD_PRIO_MEDIUM:
    policy = THREAD_PRIORITY_ABOVE_NORMAL;
    break;
  case MAG_THREAD_PRIO_HIGH:
    policy = THREAD_PRIORITY_HIGHEST;
    break;
  case MAG_THREAD_PRIO_REALTIME:
    policy = THREAD_PRIORITY_TIME_CRITICAL;
    break;
  }
  if (mag_unlikely(!SetThreadPriority(GetCurrentThread(), policy))) {
    mag_log_warn("Failed to set thread scheduling priority: %d", prio);
  }
#else
  int32_t policy = SCHED_OTHER;
  struct sched_param p;
  switch (prio) {
  case MAG_THREAD_PRIO_NORMAL:
    p.sched_priority = 0;
    policy = SCHED_OTHER;
    break;
  case MAG_THREAD_PRIO_MEDIUM:
    p.sched_priority = 40;
    policy = SCHED_FIFO;
    break;
  case MAG_THREAD_PRIO_HIGH:
    p.sched_priority = 80;
    policy = SCHED_FIFO;
    break;
  case MAG_THREAD_PRIO_REALTIME:
    p.sched_priority = 90;
    policy = SCHED_FIFO;
    break;
  }
  int status = pthread_setschedparam(pthread_self(), policy, &p);
  if (mag_unlikely(status)) {
    mag_log_warn("Failed to set thread scheduling priority: %d, error: %x", prio, status);
  }
#endif
}

/* Set thread name for current thread. */
void mag_curr_thread_set_name(const char *name) {
#if defined(__linux__)
  prctl(PR_SET_NAME, name);
#elif defined(__APPLE__) && defined(__MACH__)
  pthread_setname_np(name);
#endif
}

/* Yield current thread. */
void mag_curr_thread_yield(void) {
#if defined(_WIN32)
  YieldProcessor();
#else
  sched_yield();
#endif
}

int mag_futex_wait(volatile mag_atomic32_t *addr, mag_atomic32_t expect) {
#ifdef __linux__
  return syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expect, NULL, NULL, 0);
#elif defined(__APPLE__)
  mag_assert2(__ulock_wait);
  return __ulock_wait(UL_COMPARE_AND_WAIT, (void *)addr, expect, 0);
#elif defined(_WIN32)
  BOOL ok = WaitOnAddress((volatile VOID *)addr, &expect, sizeof(expect), INFINITE);
  if (mag_likely(ok)) return 0;
  errno = GetLastError() == ERROR_TIMEOUT ? ETIMEDOUT : EAGAIN;
  return -1;
#else
#error "Not implemented for this platform"
#endif
}

void mag_futex_wake1(volatile mag_atomic32_t *addr) {
#ifdef __linux__
  syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
#elif defined(__APPLE__)
  mag_assert2(__ulock_wake);
  __ulock_wake(UL_COMPARE_AND_WAIT, (void *)addr, 0);
#elif defined(_WIN32)
  WakeByAddressSingle((PVOID)addr);
#else
#error "Not implemented for this platform"
#endif
}

void mag_futex_wakeall(volatile mag_atomic32_t *addr) {
#ifdef __linux__
  syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, 0x7fffffff, NULL, NULL, 0);
#elif defined(__APPLE__)
  mag_assert2(__ulock_wake);
  __ulock_wake(UL_COMPARE_AND_WAIT|ULF_WAKE_ALL, (void *)addr, 0);
#elif defined(_WIN32)
  WakeByAddressAll((PVOID)addr);
#else
#error "Not implemented for this platform"
#endif
}
