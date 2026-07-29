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

#ifndef MAG_THREAD_H
#define MAG_THREAD_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__amd64__) || defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#ifdef _MSC_VER
#define mag_cpu_pause() _mm_pause()
#else
#define mag_cpu_pause() __asm__ __volatile__("pause" ::: "memory")
#endif
#elif defined(__aarch64__)
#define mag_cpu_pause() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__loongarch64)
#define mag_cpu_pause() __asm__ __volatile__("nop" ::: "memory")
#else
#error "Unsupported architecture for mag_cpu_pause()"
#endif

typedef enum mag_thread_prio_t {      /* Thread scheduling priority for CPU compute */
  MAG_THREAD_PRIO_NORMAL = 0,         /* Normal thread priority */
  MAG_THREAD_PRIO_MEDIUM = 1,         /* Medium thread priority */
  MAG_THREAD_PRIO_HIGH = 2,           /* High thread priority */
  MAG_THREAD_PRIO_REALTIME = 3,       /* Real-time thread priority */
} mag_thread_prio_t;

typedef void mag_thread_t;
extern MAG_EXPORT mag_status_t mag_thread_create(
  mag_error_t *err,
  mag_thread_t **out,
  void (*entry)(void *),
  mag_thread_prio_t prio,
  const char *name,
  void *arg
);
extern MAG_EXPORT mag_status_t mag_thread_join(mag_error_t *err, mag_thread_t *thr);

typedef void mag_mutex_t;
extern MAG_EXPORT mag_status_t mag_mutex_create(mag_error_t *err, mag_mutex_t **out);
extern MAG_EXPORT mag_status_t mag_mutex_destroy(mag_error_t *err, mag_mutex_t *mtx);
extern MAG_EXPORT mag_status_t mag_mutex_lock(mag_error_t *err, mag_mutex_t *mtx);
extern MAG_EXPORT mag_status_t mag_mutex_unlock(mag_error_t *err, mag_mutex_t *mtx);

typedef void mag_condvar_t;
extern MAG_EXPORT mag_status_t mag_condvar_create(mag_error_t *err, mag_condvar_t **out);
extern MAG_EXPORT mag_status_t mag_condvar_destroy(mag_error_t *err, mag_condvar_t *cv);
extern MAG_EXPORT mag_status_t mag_condvar_wait(mag_error_t *err, mag_condvar_t *cv, mag_mutex_t *mtx);
extern MAG_EXPORT mag_status_t mag_condvar_signal(mag_error_t *err, mag_condvar_t *cv);
extern MAG_EXPORT mag_status_t mag_condvar_broadcast(mag_error_t *err, mag_condvar_t *cv);

extern MAG_EXPORT void mag_curr_thread_set_prio(mag_thread_prio_t prio); /* Set thread scheduling priority of current thread. */
extern MAG_EXPORT void mag_curr_thread_set_name(const char *name); /* Set thread name. */
extern MAG_EXPORT void mag_curr_thread_yield(void); /* Yield current thread. */
extern MAG_EXPORT int mag_futex_wait(volatile mag_atomic32_t *addr, mag_atomic32_t expect);
extern MAG_EXPORT void mag_futex_wake1(volatile mag_atomic32_t *addr);
extern MAG_EXPORT void mag_futex_wakeall(volatile mag_atomic32_t *addr);

#ifdef __cplusplus
}
#endif

#endif
