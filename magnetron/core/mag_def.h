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

#ifndef MAG_DEF_H
#define MAG_DEF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <magnetron/magnetron.h>

#if defined(__GLIBC__) || defined(__GNU_LIBRARY__) || defined(__ANDROID__)
#include <endian.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <machine/endian.h>
#elif defined(BSD) || defined(_SYSTYPE_BSD)
#if defined(__OpenBSD__)
#include <machine/endian.h>
#else
#include <sys/endian.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(NDEBUG) || !NDEBUG
#define MAG_DEBUG
#endif

#define MAG_MAX_OP_INPUTS 2 /* Maximum number of input tensors for an operation */

#define MAG_SEP ,
#define MAG_GELU_COEFF 0.044715f /* Coefficient for GELU approximation. */

#define MAG_MAX_CPUS 8192
#define MAG_MAX_CPU_TOPO_DEPTH 2
#define MAG_MAX_CPU_CACHE_DEPTH 10
#define MAG_MAX_CPU_NUMA_NODES 64

/* Allows compiling code for host and cuda */
#if defined(__CUDA_ARCH__) && !defined(MAG_CUDA_DEVICE)
#define MAG_CUDA_DEVICE __device__
#else
#define MAG_CUDA_DEVICE
#endif

#ifdef __cplusplus
#define restrict __restrict__
#endif

/* Compiler specific macros and utils for GCC and CLang. */
#if defined(__GNUC__) || defined(__clang__)

#define MAG_NORET __attribute__((noreturn))
#define mag_alignas(x) __attribute__((aligned(x)))
#define MAG_AINLINE inline __attribute__((always_inline))
#define MAG_NOINLINE __attribute__((noinline))
#define MAG_HOTPROC __attribute__((hot))
#define MAG_COLDPROC __attribute__((cold))
#define MAG_PACKED __attribute__((packed))
#define MAG_FALLTHROUGH __attribute__((fallthrough))
#define MAG_UNUSED __attribute__((unused))
#define MAG_THREAD_LOCAL __thread
#define mag_likely(x) __builtin_expect(!!(x), 1)
#define mag_unlikely(x) __builtin_expect(!!(x), 0)
#define mag_ffs(x) ((uint32_t)__builtin_ctz(x))
#define mag_fls(x) ((uint32_t)(__builtin_clz(x)^31))
#define mag_ffs64(x) ((uint32_t)__builtin_ctzll(x))
#define mag_fls64(x) ((uint32_t)(__builtin_clzll(x)^63))
#define mag_printf_fmt(str, idx) __attribute__((format(printf, str, idx)))

/* Memory order for atomic operations. */
typedef enum mag_memory_order_t {
    MAG_MO_RELAXED = __ATOMIC_RELAXED,
    MAG_MO_CONSUME = __ATOMIC_CONSUME,
    MAG_MO_ACQUIRE = __ATOMIC_ACQUIRE,
    MAG_MO_RELEASE = __ATOMIC_RELEASE,
    MAG_MO_ACQ_REL = __ATOMIC_ACQ_REL,
    MAG_MO_SEQ_CST = __ATOMIC_SEQ_CST
} mag_memory_order_t;

typedef int64_t mag_atomic64_t;
static MAG_AINLINE void mag_atomic64_store(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    __atomic_store_n(o, x, order);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_load(volatile mag_atomic64_t *o, mag_memory_order_t order) {
    return __atomic_load_n(o, order);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_add(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    return __atomic_fetch_add(o, x, order);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_sub(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    return __atomic_fetch_sub(o, x, order);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_and(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    return __atomic_fetch_and(o, x, order);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_or(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    return __atomic_fetch_or(o, x, order);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_xor(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    return __atomic_fetch_xor(o, x, order);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_exchange(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    return __atomic_exchange_n(o, x, order);
}
static MAG_AINLINE bool mag_atomic64_compare_exchange_weak(volatile mag_atomic64_t *o, mag_atomic64_t *exp, mag_atomic64_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, true, order_succ, order_fail);
}
static MAG_AINLINE bool mag_atomic64_compare_exchange_strong(volatile mag_atomic64_t *o, mag_atomic64_t *exp, mag_atomic64_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, false, order_succ, order_fail);
}

typedef int32_t mag_atomic32_t;
static MAG_AINLINE void mag_atomic32_store(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    __atomic_store_n(o, x, order);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_load(volatile mag_atomic32_t *o, mag_memory_order_t order) {
    return __atomic_load_n(o, order);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_add(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    return __atomic_fetch_add(o, x, order);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_sub(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    return __atomic_fetch_sub(o, x, order);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_and(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    return __atomic_fetch_and(o, x, order);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_or(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    return __atomic_fetch_or(o, x, order);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_xor(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    return __atomic_fetch_xor(o, x, order);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_exchange(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    return __atomic_exchange_n(o, x, order);
}
static MAG_AINLINE bool mag_atomic32_compare_exchange_weak(volatile mag_atomic32_t *o, mag_atomic32_t *exp, mag_atomic32_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, true, order_succ, order_fail);
}
static MAG_AINLINE bool mag_atomic32_compare_exchange_strong(volatile mag_atomic32_t *o, mag_atomic32_t *exp, mag_atomic32_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, false, order_succ, order_fail);
}

/* Compiler specific macros and utils for MSVC. */
#elif defined(_MSC_VER)

unsigned char _BitScanForward64(unsigned long *, unsigned __int64);
unsigned char _BitScanReverse64(unsigned long *, unsigned __int64);
#pragma intrinsic(_BitScanForward64)
#pragma intrinsic(_BitScanReverse64)
#define MAG_NORET __declspec(noreturn)
#define mag_alignas(x) __declspec(align(x))
#define MAG_AINLINE inline __forceinline
#define MAG_NOINLINE __declspec(noinline)
#define MAG_HOTPROC
#define MAG_COLDPROC
#define MAG_PACKED __declspec(align(1))
#define MAG_FALLTHROUGH
#define MAG_UNUSED
#define MAG_THREAD_LOCAL __declspec(thread)
#define mag_likely(x) (x)
#define mag_unlikely(x) (x)
static MAG_AINLINE uint32_t mag_ffs(uint32_t x) {
    unsigned long r;
    _BitScanForward(&r, x);
    return (uint32_t)r;
}
static MAG_AINLINE uint32_t mag_fls(uint32_t x) {
    unsigned long r;
    _BitScanReverse(&r, x);
    return (uint32_t)r;
}
static MAG_AINLINE uint32_t mag_ffs64(uint64_t x) {
    unsigned long r;
    _BitScanForward64(&r, x);
    return (uint32_t)r;
}
static MAG_AINLINE uint32_t mag_fls64(uint64_t x) {
    unsigned long r;
    _BitScanReverse64(&r, x);
    return (uint32_t)r;
}
#define mag_printf_fmt(str, idx)

typedef enum mag_memory_order_t { /* Atomic memory order. Has no effect with MSVC for now, all operations are sequencial consistent. */
    MAG_MO_RELAXED,
    MAG_MO_CONSUME,
    MAG_MO_ACQUIRE,
    MAG_MO_RELEASE,
    MAG_MO_ACQ_REL,
    MAG_MO_SEQ_CST
} mag_memory_order_t;

typedef long long mag_atomic64_t;
static MAG_AINLINE void mag_atomic64_store(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    _InterlockedExchange64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_load(volatile mag_atomic64_t *o, mag_memory_order_t order) {
    (void)order;
    mag_atomic64_t r;
    _InterlockedExchange64(&r, *o);
    return r;
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_add(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchangeAdd64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_sub(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchangeAdd64(o, -x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_and(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedAnd64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_or(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedOr64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_xor(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedXor64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_exchange(volatile mag_atomic64_t *o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchange64(o, x);
}
static MAG_AINLINE bool mag_atomic64_compare_exchange_weak(volatile mag_atomic64_t *o, mag_atomic64_t *exp, mag_atomic64_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    (void)order_succ;
    (void)order_fail;
    mag_atomic64_t old = _InterlockedCompareExchange64(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else {
        *exp = old;
        return false;
    }
}
static MAG_AINLINE bool mag_atomic64_compare_exchange_strong(volatile mag_atomic64_t *o, mag_atomic64_t *exp, mag_atomic64_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    (void)order_succ;
    (void)order_fail;
    mag_atomic64_t old = _InterlockedCompareExchange64(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else {
        *exp = old;
        return false;
    }
}

#define restrict __restrict

typedef long mag_atomic32_t;
static MAG_AINLINE void mag_atomic32_store(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    _InterlockedExchange(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_load(volatile mag_atomic32_t *o, mag_memory_order_t order) {
    (void)order;
    mag_atomic32_t r;
    _InterlockedExchange(&r, *o);
    return r;
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_add(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchangeAdd(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_sub(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchangeAdd(o, -x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_and(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedAnd(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_or(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedOr(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_xor(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedXor(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_exchange(volatile mag_atomic32_t *o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchange(o, x);
}
static MAG_AINLINE bool mag_atomic32_compare_exchange_weak(volatile mag_atomic32_t *o, mag_atomic32_t *exp, mag_atomic32_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    (void)order_succ;
    (void)order_fail;
    mag_atomic32_t old = _InterlockedCompareExchange(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else {
        *exp = old;
        return false;
    }
}
static MAG_AINLINE bool mag_atomic32_compare_exchange_strong(volatile mag_atomic32_t *o, mag_atomic32_t *exp, mag_atomic32_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    (void)order_succ;
    (void)order_fail;
    mag_atomic32_t old = _InterlockedCompareExchange(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else {
        *exp = old;
        return false;
    }
}

#endif

mag_static_assert(sizeof(0u) == 4);     /* u literal suffix must infer to uint32. */
mag_static_assert(sizeof(0ull) == 8);   /* ull literal suffix must infer to uint64. */

/* Endianness detection. */
#ifdef __BYTE_ORDER
#if defined(__BIG_ENDIAN) && (__BYTE_ORDER == __BIG_ENDIAN)
#define MAG_BE
#elif defined(__LITTLE_ENDIAN) && (__BYTE_ORDER == __LITTLE_ENDIAN)
#define MAG_LE
#endif
#elif defined(_BYTE_ORDER)
#if defined(_BIG_ENDIAN) && (_BYTE_ORDER == _BIG_ENDIAN)
#define MAG_BE
#elif defined(_LITTLE_ENDIAN) && (_BYTE_ORDER == _LITTLE_ENDIAN)
#define MAG_LE
#endif
#elif defined(__BIG_ENDIAN__)
#define MAG_BE
#elif defined(__LITTLE_ENDIAN__)
#define MAG_LE
#else
#if defined(__ARMEL__) || defined(__THUMBEL__) || defined(__AARCH64EL__) || \
defined(_MIPSEL) || defined(__MIPSEL) || defined(__MIPSEL__) || \
defined(__ia64__) || defined(_IA64) || defined(__IA64__) || defined(__ia64) || \
defined(_M_IA64) || defined(__itanium__) || defined(i386) || defined(__i386__) || \
defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__i386) || \
defined(_M_IX86) || defined(_X86_) || defined(__THW_INTEL__) || defined(__I86__) || \
defined(__INTEL__) || defined(__x86_64) || defined(__x86_64__) || \
defined(__amd64__) || defined(__amd64) || defined(_M_X64) || \
defined(__bfin__) || defined(__BFIN__) || defined(bfin) || defined(BFIN)
#define MAG_LE
#elif defined(__m68k__) || defined(M68000) || defined(__hppa__) || defined(__hppa) || defined(__HPPA__) || \
defined(__sparc__) || defined(__sparc) || defined(__370__) || defined(__THW_370__) || \
defined(__s390__) || defined(__s390x__) || defined(__SYSC_ZARCH__)
#define MAG_BE
#elif defined(__arm__) || defined(__arm64) || defined(__thumb__) || \
defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || defined(__ARM_ARCH) || \
defined(_M_ARM) || defined(_M_ARM64)
#if defined(_WIN32) || defined(_WIN64) || \
defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__)
#define MAG_LE
#else
#error "Unknown endianness"
#endif
#endif
#endif

#ifdef __cpp_lib_hardware_interference_size
/* Cache line size. Used for alignment to avoid destructive interference (false sharing). */
#define MAG_DESTRUCTIVE_INTERFERENCE_SIZE std::hardware_destructive_interference_size
#else
/* Cache line size. Used for alignment to avoid destructive interference (false sharing). */
#define MAG_DESTRUCTIVE_INTERFERENCE_SIZE 64
#endif

#define MAG_PAGE_SIZE_4K 0x1000     /* 4 KiB page size */
#define MAG_PAGE_SIZE_2M 0x200000   /* 2 MiB page size */

#define MAG_CPU_BUF_ALIGN 64

mag_static_assert(MAG_CPU_BUF_ALIGN >= 8 && !(MAG_CPU_BUF_ALIGN&(MAG_CPU_BUF_ALIGN-1)));

static uint16_t MAG_AINLINE mag_bswap16(uint16_t x) {
#if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
    x = __builtin_bswap16(x);
#else
    x = (x&0xff00)>>8 | x&0xff<<8;
#endif
    return x;
}
static uint32_t MAG_AINLINE mag_bswap32(uint32_t x) {
#if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
    x = __builtin_bswap32(x);
#else
    x = (x&0xff000000)>>24 |
        (x&0xff0000)>>8 |
        (x&0xff00)<<8 |
        (x&0xff)<<24;
#endif
    return x;
}
static uint64_t MAG_AINLINE mag_bswap64(uint64_t x) {
#if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
    x = __builtin_bswap64(x);
#else
    x = (x&0xff00000000000000)>>56 |
        (x&0xff000000000000)>>40 |
        (x&0xff0000000000)>>24 |
        (x&0xff00000000)>>8 |
        (x&0xff000000)<<8 |
        (x&0xff0000)<<24 |
        (x&0xff00)<<40 |
        (x&0xff)<<56;
#endif
    return x;
}

static MAG_CUDA_DEVICE MAG_AINLINE uint32_t mag_mulhilo32(uint32_t x, uint32_t y, uint32_t *hi) {
    #ifdef __CUDA_ARCH__
        *hi = __umulhi(x, y);
        return x*y;
    #else
        uint64_t p = (uint64_t)x*(uint64_t)y;
        *hi = p>>32;
        return (uint32_t)p;
    #endif
}

static MAG_AINLINE uint64_t mag_mulhilo64(uint64_t x, uint64_t y) {
#if defined(_MSC_VER) && (!defined(__clang__) || _MSC_VER > 1930) && (defined(_M_X64) || defined(_M_ARM64))
    return __umulh(x, y);
#elif defined(__SIZEOF_INT128__)
    unsigned __int128 xl = x, yl = y;
    unsigned __int128 rl = xl*yl;
    return (uint64_t)(rl>>64);
#else
    /* Fastdivmod is slow with this (IPC raise a lot), because we have NO fast compiler intrinsics */
    uint32_t x0 = (uint32_t)(x&~0u);
    uint32_t x1 = (uint32_t)(x>>32);
    uint32_t y0 = (uint32_t)(y&~0u);
    uint32_t y1 = (uint32_t)(y>>32);
    uint32_t x0y0_hi;
    mag_mulhilo32(x0, y0, &x0y0_hi);
    uint64_t x0y1 = x0*(uint64_t)y1;
    uint64_t x1y0 = x1*(uint64_t)y0;
    uint64_t x1y1 = x1*(uint64_t)y1;
    uint64_t temp = x1y0 + x0y0_hi;
    uint64_t tlo = temp&~0u;
    uint64_t thi = temp>>32;
    return x1y1 + thi + ((tlo + x0y1)>>32);
#endif
}

/* Logging and panic utilities. */

extern MAG_EXPORT MAG_NORET MAG_COLDPROC void mag_panic(const char *fmt, ...) mag_printf_fmt(1, 2); /* Print error message and abort. */
extern MAG_EXPORT void mag_log_fmt(mag_log_level_t level, const char *fmt, ...) mag_printf_fmt(2, 3); /* Log message to stdout if below log level. */

/* Logging and debugging macros. */
#define MAG_CC_RED "\x1b[31m"
#define MAG_CC_GREEN "\x1b[32m"
#define MAG_CC_YELLOW "\x1b[33m"
#define MAG_CC_BLUE "\x1b[34m"
#define MAG_CC_MAGENTA "\x1b[35m"
#define MAG_CC_CYAN "\x1b[36m"
#define MAG_CC_RESET "\x1b[0m"
#define MAG_STRINGIZE2(x) #x
#define MAG_STRINGIZE(x) MAG_STRINGIZE2(x)
#ifdef __FILE_NAME__
#   define MAG_SRC_NAME __FILE_NAME__ ":" MAG_STRINGIZE(__LINE__)
#else
#   define MAG_SRC_NAME __FILE__ ":" MAG_STRINGIZE(__LINE__)
#endif
#define mag_log_info(msg, ...) do { mag_log_fmt(MAG_LOG_LEVEL_INFO, MAG_SRC_NAME " " msg, ## __VA_ARGS__); } while (0)
#define mag_log_warn(msg, ...) do { mag_log_fmt(MAG_LOG_LEVEL_WARN, MAG_SRC_NAME " " msg, ## __VA_ARGS__);  } while (0)
#define mag_log_error(msg, ...) do { mag_log_fmt(MAG_LOG_LEVEL_ERROR, MAG_SRC_NAME " " msg, ## __VA_ARGS__); } while (0)

/* Panic and print 'msg' if 'expr' is false. */
#define mag_assert(expr, msg, ...) \
    if (mag_unlikely(!(expr))) { \
        mag_panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }

/* Panic if 'expr' is false. */
#define mag_assert2(expr) mag_assert(expr, "")

#if defined(MAG_DEBUG)
/* Panics if ptr ∉ [base, base+N). */
#define mag_bnd_chk(ptr, base, N) \
    mag_assert((char*)(ptr) >= (char*)(base) && (char*)(ptr) < (char*)(base)+(N), \
        "\nBound check failed: %p not in [%p, %p), base+0x%x, end+0x%x", \
        (void *)(ptr), \
        (void *)(base), \
        (void *)((char *)(base)+(N)), \
        abs((int)((intptr_t)(ptr)-(intptr_t)(base))), /* Allow +-2G delta */ \
        abs((int)(((intptr_t)(base)+(N))-(intptr_t)(ptr))) \
    )

/* Same as mag_assert but only activated in debug builds. */
#define mag_dassert mag_assert

/* Same as mag_assert2 but only activated in debug builds. */
#define mag_dassert2 mag_assert2
#else
#define mag_bnd_chk(ptr, base, nb_src)
#define mag_dassert(...)
#define mag_dassert2(...)
#endif

#define mag_iserr(stat) (mag_unlikely((stat) != MAG_STATUS_OK))
#define mag_isok(stat) (!mag_iserr((stat)))

/*
'* If 'expr' is false, set the last error in the given context to 'status' and
'* return 'status'. Optionally perform 'cleanup' code before returning.
'* The 'msg' is a printf-style format string with optional arguments.
*/
#define mag_contract(ctx, status, cleanup, expr, msg, ...) \
    if (mag_unlikely(!(expr))) { \
        mag_error_t err = { \
            .code = MAG_STATUS_##status, \
            .message = "", \
            .file = __FILE__, \
            .line = __LINE__, \
            .func = __FUNCTION__, \
        }; \
        snprintf(err.message, sizeof(err.message), msg, ## __VA_ARGS__); \
        mag_ctx_set_last_error((ctx), &err); \
        cleanup \
        return MAG_STATUS_##status; \
    }

extern void MAG_COLDPROC mag_print_separator(FILE *f); /* Print a separator line. */

/* Humanize memory size. Format and convert a memory size to the appropriate unit. For example. 1024 => 1 KiB */
extern void mag_humanize_memory_size(size_t n, double *out, const char **unit);
extern uintptr_t mag_thread_id(void); /* Get current native thread ID. */
extern FILE *mag_fopen(const char *file, const char *mode);
extern uint64_t mag_hpc_clock_ns(void); /* Get high precision clock in nanoseconds. */
extern uint64_t mag_hpc_clock_elapsed_ns(uint64_t start);
extern double mag_hpc_clock_elapsed_ms(uint64_t start);
extern uint64_t mag_cycles(void); /* Get current CPU cycles. */

#define mag_swap(T, a, b) do { T tmp = (a); (a) = (b); (b) = tmp; } while (0)
#define mag_xmax(x, y) (((x) > (y)) ? (x) : (y))
#define mag_xmin(x, y) (((x) < (y)) ? (x) : (y))
#define mag_rd_down(x,m) ((x)/(m) * (m))
#define mag_clamp(v, lo, hi) ((v) < (lo) ? (lo) : (v) > (hi) ? (hi) : (v))

#define MAG_TAU 6.283185307179586476925286766559005768394338798f /* τ=2π */
#define MAG_INVSQRT2 0.707106781186547524400844362104849039284835937f /* 1/√2 */

/* Increment pointer or size with correct type alignment. */
static inline void *mag_pincr(void **p, size_t sz, size_t align) {
    void *pp = (void *)(((uintptr_t)*p+align-1)&-align);
    *p = (void *)((uint8_t *)pp+sz);
    return pp;
}

/* Performs c = ab with overflow checking. Returns true on overflow, else false. */
static bool MAG_AINLINE mag_mulov64(int64_t a, int64_t b, int64_t *c) {
#ifdef _MSC_VER
#ifdef _M_ARM64
uint64_t high = __umulh(a, b);
*c = a*b;
return high != (*c>>63);
#else
int64_t high;
int64_t low = _mul128(a, b, &high);
int64_t sign = low>>63;
*c = low;
return high != sign;
#endif
#else
#if __SIZEOF_LONG_LONG__ == 8 && __SIZEOF_LONG__ == 8
return __builtin_smulll_overflow(a, b, (long long *)c);
#else
return __builtin_smull_overflow(a, b, c);
#endif
#endif
}

extern MAG_EXPORT uint64_t mag_hash(const void *key, size_t len, uint32_t seed); /* Compute murmur3_64 hash */
extern uint32_t mag_crc32c(const void *buffer, size_t size); /* Compute CRC32 checksum with CRC32c polynomial. */
extern bool mag_utf8_validate(const char *str, size_t len);
extern char *mag_strdup(const char *s);
extern void mag_path_split_dir_inplace(char *path, char **out_dir, char **out_file);

#ifdef __cplusplus
}
#endif

#endif
