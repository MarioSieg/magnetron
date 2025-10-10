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

#include "mag_thread.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <synchapi.h>
#elif defined __APPLE__
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
#include <linux/futex.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#endif

/* Set scheduling priority for current thread. */
void mag_thread_set_prio(mag_thread_prio_t prio) {
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
void mag_thread_set_name(const char *name) {
#if defined(__linux__)
    prctl(PR_SET_NAME, name);
#elif defined(__APPLE__) && defined(__MACH__)
    pthread_setname_np(name);
#endif
}

/* Yield current thread. */
void mag_thread_yield(void) {
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
