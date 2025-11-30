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

#ifndef MAG_THREAD_H
#define MAG_THREAD_H

#include "mag_def.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32 /* WIN32 specific threading and synchronization. */

typedef DWORD mag_thread_ret_t;
#define MAG_THREAD_RET_NONE 0

typedef HANDLE mag_thread_t;

static void mag_thread_create(mag_thread_t *out, mag_thread_ret_t (*f)(void *), void *arg) { /* WIN32 -> pthread style wrapper. */
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)f, arg, 0, NULL);
    mag_assert2(handle != 0);
    *out = handle;
}

static void mag_thread_join(mag_thread_t th) { /* WIN32 -> pthread style wrapper. */
    int ret = (int)WaitForSingleObject(th, INFINITE);
    CloseHandle(th);
    mag_assert2(ret == 0);
}

typedef SRWLOCK mag_mutex_t;
#define mag_mutex_create(mtx) InitializeSRWLock(mtx)
#define mag_mutex_destroy(mtx)
#define mag_mutex_lock(mtx) AcquireSRWLockExclusive(mtx)
#define mag_mutex_unlock(mtx) ReleaseSRWLockExclusive(mtx)

typedef CONDITION_VARIABLE mag_condvar_t;
#define mag_cv_create(cv) InitializeConditionVariable(cv)
#define mag_cv_destroy(cv)
#define mag_cv_wait(cv, mtx) SleepConditionVariableSRW(cv, mtx, INFINITE, 0)
#define mag_cv_signal(cv) WakeConditionVariable(cv)
#define mag_cv_broadcast(cv) WakeAllConditionVariable(cv)

#else /* POSIX threading and synchronization. */

typedef void *mag_thread_ret_t;
#define MAG_THREAD_RET_NONE NULL

typedef pthread_t mag_thread_t;
#define mag_thread_create(out, fn, arg) mag_assert2(pthread_create((out), NULL, (fn), (arg)) == 0)
#define mag_thread_join(th) mag_assert2(pthread_join((th), NULL) == 0)

typedef pthread_mutex_t mag_mutex_t;
#define mag_mutex_create(mtx) mag_assert2(pthread_mutex_init(mtx, NULL) == 0)
#define mag_mutex_destroy(mtx) mag_assert2(pthread_mutex_destroy(mtx) == 0)
#define mag_mutex_lock(mtx) mag_assert2(pthread_mutex_lock(mtx) == 0)
#define mag_mutex_unlock(mtx) mag_assert2(pthread_mutex_unlock(mtx) == 0)

typedef pthread_cond_t mag_condvar_t;
#define mag_cv_create(cv) mag_assert2(pthread_cond_init(cv, NULL) == 0)
#define mag_cv_destroy(cv) mag_assert2(pthread_cond_destroy(cv) == 0)
#define mag_cv_wait(cv, mtx) mag_assert2(pthread_cond_wait(cv, mtx) == 0)
#define mag_cv_signal(cv) mag_assert2(pthread_cond_signal(cv) == 0)
#define mag_cv_broadcast(cv) mag_assert2(pthread_cond_broadcast(cv) == 0)

#endif

#if defined(__amd64__) || defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)
#ifdef _MSC_VER
#define mag_cpu_pause() _mm_pause()
#else
#define mag_cpu_pause() __asm__ __volatile__("pause" ::: "memory")
#endif
#elif defined(__aarch64__)
#define mag_cpu_pause() __asm__ __volatile__("yield" ::: "memory")
#else
#error "Unsupported architecture for mag_cpu_pause()"
#endif

typedef enum mag_thread_prio_t {       /* Thread scheduling priority for CPU compute */
    MAG_THREAD_PRIO_NORMAL = 0,     /* Normal thread priority */
    MAG_THREAD_PRIO_MEDIUM = 1,     /* Medium thread priority */
    MAG_THREAD_PRIO_HIGH = 2,       /* High thread priority */
    MAG_THREAD_PRIO_REALTIME = 3,   /* Real-time thread priority */
} mag_thread_prio_t;

extern MAG_EXPORT void mag_thread_set_prio(mag_thread_prio_t prio); /* Set thread scheduling priority of current thread. */
extern MAG_EXPORT void mag_thread_set_name(const char *name); /* Set thread name. */
extern MAG_EXPORT void mag_thread_yield(void); /* Yield current thread. */
extern MAG_EXPORT int mag_futex_wait(volatile mag_atomic32_t *addr, mag_atomic32_t expect);
extern MAG_EXPORT void mag_futex_wake1(volatile mag_atomic32_t *addr);
extern MAG_EXPORT void mag_futex_wakeall(volatile mag_atomic32_t *addr);

#ifdef __cplusplus
}
#endif

#endif
