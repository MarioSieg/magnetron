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

#ifndef MAGNETRON_INTERNAL_H
#define MAGNETRON_INTERNAL_H

#include <magnetron/magnetron.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#endif
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#include <pthread.h>
#ifdef __FreeBSD__
#include <pthread_np.h>
#endif
#endif

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

#define MAG_SEP ,

#define MAG_GELU_COEFF 0.044715f /* Coefficient for GELU approximation. */

#define MAG_MAX_CPUS 8192
#define MAG_MAX_CPU_TOPO_DEPTH 2
#define MAG_MAX_CPU_CACHE_DEPTH 10
#define MAG_MAX_CPU_NUMA_NODES 64

/* Compiler specific macros and utils for GCC, Clang and ICC. */
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)

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
static MAG_AINLINE void mag_atomic64_store(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) { __atomic_store_n(o, x, order); }
static MAG_AINLINE mag_atomic64_t mag_atomic64_load(volatile mag_atomic64_t* o, mag_memory_order_t order) { return __atomic_load_n(o, order); }
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_add(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) { return __atomic_fetch_add(o, x, order); }
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_sub(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) { return __atomic_fetch_sub(o, x, order); }
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_and(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) { return __atomic_fetch_and(o, x, order); }
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_or(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) { return __atomic_fetch_or(o, x, order); }
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_xor(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) { return __atomic_fetch_xor(o, x, order); }
static MAG_AINLINE mag_atomic64_t mag_atomic64_exchange(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) { return __atomic_exchange_n(o, x, order); }
static MAG_AINLINE bool mag_atomic64_compare_exchange_weak(volatile mag_atomic64_t* o, mag_atomic64_t* exp, mag_atomic64_t* des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, true, order_succ, order_fail);
}
static MAG_AINLINE bool mag_atomic64_compare_exchange_strong(volatile mag_atomic64_t* o, mag_atomic64_t* exp, mag_atomic64_t* des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, false, order_succ, order_fail);
}

typedef int32_t mag_atomic32_t;
static MAG_AINLINE void mag_atomic32_store(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) { __atomic_store_n(o, x, order); }
static MAG_AINLINE mag_atomic32_t mag_atomic32_load(volatile mag_atomic32_t* o, mag_memory_order_t order) { return __atomic_load_n(o, order); }
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_add(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) { return __atomic_fetch_add(o, x, order); }
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_sub(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) { return __atomic_fetch_sub(o, x, order); }
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_and(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) { return __atomic_fetch_and(o, x, order); }
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_or(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) { return __atomic_fetch_or(o, x, order); }
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_xor(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) { return __atomic_fetch_xor(o, x, order); }
static MAG_AINLINE mag_atomic32_t mag_atomic32_exchange(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) { return __atomic_exchange_n(o, x, order); }
static MAG_AINLINE bool mag_atomic32_compare_exchange_weak(volatile mag_atomic32_t* o, mag_atomic32_t* exp, mag_atomic32_t* des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, true, order_succ, order_fail);
}
static MAG_AINLINE bool mag_atomic32_compare_exchange_strong(volatile mag_atomic32_t* o, mag_atomic32_t* exp, mag_atomic32_t* des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, false, order_succ, order_fail);
}

/* Compiler specific macros and utils for MSVC. */
#elif defined(_MSC_VER)

unsigned char _BitScanForward64(unsigned long*, unsigned __int64);
unsigned char _BitScanReverse64(unsigned long*, unsigned __int64);
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
static MAG_AINLINE uint32_t mag_ffs(uint32_t x) { unsigned long r; _BitScanForward(&r, x); return (uint32_t)r; }
static MAG_AINLINE uint32_t mag_fls(uint32_t x) { unsigned long r; _BitScanReverse(&r, x); return (uint32_t)r; }
static MAG_AINLINE uint32_t mag_ffs64(uint64_t x) { unsigned long r; _BitScanForward64(&r, x); return (uint32_t)r; }
static MAG_AINLINE uint32_t mag_fls64(uint64_t x) { unsigned long r; _BitScanReverse64(&r, x); return (uint32_t)r; }

typedef enum mag_memory_order_t { /* Atomic memory order. Has no effect with MSVC for now, all operations are sequencial consistent. */
    MAG_MO_RELAXED,
    MAG_MO_CONSUME,
    MAG_MO_ACQUIRE,
    MAG_MO_RELEASE,
    MAG_MO_ACQ_REL,
    MAG_MO_SEQ_CST
} mag_memory_order_t;

typedef long long mag_atomic64_t;
static MAG_AINLINE void mag_atomic64_store(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order; _InterlockedExchange64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_load(volatile mag_atomic64_t* o, mag_memory_order_t order) {
    (void)order;
    mag_atomic64_t r;
    _InterlockedExchange64(&r, *o);
    return r;
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_add(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchangeAdd64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_sub(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchangeAdd64(o, -x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_and(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedAnd64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_or(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedOr64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_fetch_xor(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedXor64(o, x);
}
static MAG_AINLINE mag_atomic64_t mag_atomic64_exchange(volatile mag_atomic64_t* o, mag_atomic64_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchange64(o, x);
}
static MAG_AINLINE bool mag_atomic64_compare_exchange_weak(volatile mag_atomic64_t* o, mag_atomic64_t *exp, mag_atomic64_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    (void)order_succ; (void)order_fail;
    mag_atomic64_t old = _InterlockedCompareExchange64(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else { *exp = old; return false; }
}
static MAG_AINLINE bool mag_atomic64_compare_exchange_strong(volatile mag_atomic64_t* o, mag_atomic64_t *exp, mag_atomic64_t *des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    (void)order_succ; (void)order_fail;
    mag_atomic64_t old = _InterlockedCompareExchange64(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else { *exp = old; return false; }
}

#define restrict __restrict

typedef long mag_atomic32_t;
static MAG_AINLINE void mag_atomic32_store(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order; _InterlockedExchange(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_load(volatile mag_atomic32_t* o, mag_memory_order_t order) {
    (void)order;
    mag_atomic32_t r;
    _InterlockedExchange(&r, *o);
    return r;
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_add(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchangeAdd(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_sub(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchangeAdd(o, -x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_and(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedAnd(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_or(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedOr(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_fetch_xor(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedXor(o, x);
}
static MAG_AINLINE mag_atomic32_t mag_atomic32_exchange(volatile mag_atomic32_t* o, mag_atomic32_t x, mag_memory_order_t order) {
    (void)order;
    return _InterlockedExchange(o, x);
}
static MAG_AINLINE bool mag_atomic32_compare_exchange_weak(volatile mag_atomic32_t* o, mag_atomic32_t* exp, mag_atomic32_t* des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    (void)order_succ; (void)order_fail;
    mag_atomic32_t old = _InterlockedCompareExchange(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else { *exp = old; return false; }
}
static MAG_AINLINE bool mag_atomic32_compare_exchange_strong(volatile mag_atomic32_t* o, mag_atomic32_t* exp, mag_atomic32_t* des, mag_memory_order_t order_succ, mag_memory_order_t order_fail) {
    (void)order_succ; (void)order_fail;
    mag_atomic32_t old = _InterlockedCompareExchange(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else { *exp = old; return false; }
}

#endif

mag_static_assert(sizeof(0u) == 4);     /* u literal suffix must infer to uint32. */
mag_static_assert(sizeof(0ull) == 8);   /* ull literal suffix must infer to uint64. */

/*
** Because multiple floating-point formats with the same bit width exist (e.g. f16, bf16), a bit-width specific naming scheme is used:
** All floating-point types are described by their exponent (E) and mantissa (M) bits. The sign bit is always present and not added.
**
** For example:
**      float is an IEEE 754 32-bit float with 8 exponent bits and 23 mantissa bits. So float = e8m23.
**          8 exponent bits + 23 mantissa bits + 1 sign bit = 32 bits.
**      half/f16 is an IEEE 754 16-bit float with 5 exponent bits and 10 mantissa bits. So half = e5m10.
**          5 exponent bits + 10 mantissa bits + 1 sign bit = 16 bits.
**
** The builtin float keyword is only used for the public API in magnetron.h.
** The internal code uses the types below.
*/

/* Floating-point types. */

typedef double mag_e11m52_t;  /* f32 = IEEE 754 64-bit double precision float. */
typedef float mag_e8m23_t;    /* f64 = IEEE 754 32-bit single precision float. */
typedef struct mag_e5m10_t { uint16_t bits; } mag_e5m10_t; /* f16 = IEEE 754 16-bit half precision float. */


#define mag_u64x(hi, lo) (((uint64_t)0x##hi<<32)+(uint64_t)0x##lo)
#define mag_e11m52_sgn(x) (((x)>>63)&0x1ull)
#define mag_e11m52_exp(x) (((x)>>52)&0x7ffull)
#define mag_e11m52_mant(x) ((x)&0xfffffffffffffull)
#define mag_e11m52_pack(S, E, M) (((uint64_t)((S)&0x1ull)<<63)|((uint64_t)((E)&0x7ffull)<<52)|((uint64_t)((M)&0xfffffffffffffull)) )
#define mag_e8m23_sgn(x) (((x)>>31)&0x1u)
#define mag_e8m23_exp(x) (((x)>>23)&0xffu)
#define mag_e8m23_mant(x) ((x)&0x7fffffu)
#define mag_e8m23_pack(S, E, M) (((uint32_t)((S)&0x1u)<<31)|((uint32_t)((E)&0xffu)<<23)|((uint32_t)((M)&0x7fffffu)) )
#define mag_e5m10_sgn(x) (((x)>>15)&0x1u)
#define mag_e5m10_exp(x) (((x)>>10)&0x1fu)
#define mag_e5m10_mant(x) ((x)&0x3ffu)
#define mag_e5m10_pack(S, E, M) ((uint16_t)(((uint16_t)((S)&0x1u)<<15)|((uint16_t)((E)&0x1fu)<<10)|((uint16_t)((M)&0x3ffu))))
#define mag_e5m10_pack2(x) (mag_e5m10_t){(x)&0xffffu}
    
/* Some fp16 constants. */
#define MAG_E5M10_E mag_e5m10_pack2(0x4170)
#define MAG_E5M10_EPS mag_e5m10_pack2(0x1400)
#define MAG_E5M10_INF mag_e5m10_pack2(0x7c00)
#define MAG_E5M10_LN10 mag_e5m10_pack2(0x409b)
#define MAG_E5M10_LN2 mag_e5m10_pack2(0x398c)
#define MAG_E5M10_LOG10_2 mag_e5m10_pack2(0x34d1)
#define MAG_E5M10_LOG10_E mag_e5m10_pack2(0x36f3)
#define MAG_E5M10_LOG2_10 mag_e5m10_pack2(0x42a5)
#define MAG_E5M10_LOG2_E mag_e5m10_pack2(0x3dc5)
#define MAG_E5M10_MAX mag_e5m10_pack2(0x7bff)
#define MAG_E5M10_MAX_SUBNORMAL mag_e5m10_pack2(0x03ff)
#define MAG_E5M10_MIN mag_e5m10_pack2(0xfbff)
#define MAG_E5M10_MIN_POS mag_e5m10_pack2(0x0400)
#define MAG_E5M10_MIN_POS_SUBNORMAL mag_e5m10_pack2(0x0001)
#define MAG_E5M10_NAN mag_e5m10_pack2(0x7e00)
#define MAG_E5M10_NEG_INF mag_e5m10_pack2(0xfc00)
#define MAG_E5M10_NEG_ONE mag_e5m10_pack2(0xbc00)
#define MAG_E5M10_NEG_ZERO mag_e5m10_pack2(0x8000)
#define MAG_E5M10_ONE mag_e5m10_pack2(0x3c00)
#define MAG_E5M10_PI mag_e5m10_pack2(0x4248)
#define MAG_E5M10_SQRT2 mag_e5m10_pack2(0x3da8)
#define MAG_E5M10_ZERO mag_e5m10_pack2(0x0000)

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
#define MAG_DESTRUCTIVE_INTERFERENCE_SIZE hardware_destructive_interference_size
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

#define MAG_FMT_DIM_BUF_SIZE ((21+4)*MAG_MAX_DIMS)

extern MAG_EXPORT MAG_NORET MAG_COLDPROC void mag_panic(const char* msg, ...); /* Print error message and abort. */
extern MAG_EXPORT bool mag_log_enabled; /* Enable/disable logging to stdout/stderr. */

extern void MAG_COLDPROC mag_print_separator(FILE* f); /* Print a separator line. */
extern void mag_fmt_shape(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank);

/*
** Allocator function. Can be set to custom allocator.
**   ! Never returns NULL, if re/allocation fails, it will abort the program by calling mag_panic().
**   ! Never zero initializes, use manual memset if zeroing is required.
**   ! Set alignment value to 0 for system default alignment, else aligned adress is returned.
**     If alignment value > 0, it must be the same when using the realloc and dealloc mode.
**
** This single function is essentially a realloc and is used for allocating, reallocating and deallocating the following way:
** (*mag_alloc)(NULL, size, 0) <=> malloc(size)            <- Passing NULL as reallocation base and size != 0 => allocation.
** (*mag_alloc)(ptr, size, 0) <=> realloc(ptr, size)       <- Passing non-NULL pointer as reallocation base and size != 0 => reallocation.
** (*mag_alloc)(ptr, 0, 0) <=> free(ptr)                   <- Passing NULL as reallocation base and size == 0 => free.
*/
extern MAG_EXPORT void* (*mag_alloc)(void* blk, size_t size, size_t align);

/* Humanize memory size. Format and convert a memory size to the appropriate unit. For example. 1024 => 1 KiB */
extern void mag_humanize_memory_size(size_t n, mag_e11m52_t* out, const char** unit);
extern uintptr_t mag_thread_id(void); /* Get current native thread ID. */
extern uint64_t mag_hpc_clock_ns(void); /* Get high precision clock in nanoseconds. */
extern uint64_t mag_hpc_clock_elapsed_ns(uint64_t start);
extern mag_e11m52_t mag_hpc_clock_elapsed_ms(uint64_t start);
extern uint64_t mag_cycles(void); /* Get current CPU cycles. */

#define mag_swap(T, a, b) do { T tmp = (a); (a) = (b); (b) = tmp; } while (0)
#define mag_xmax(x, y) (((x) > (y)) ? (x) : (y))
#define mag_xmin(x, y) (((x) < (y)) ? (x) : (y))
#define mag_rd_down(x,m) ((x)/(m) * (m))
#define mag_clamp(v, lo, hi) ((v) < (lo) ? (lo) : (v) > (hi) ? (hi) : (v))

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
#define mag_log_info(msg, ...) do { if (mag_unlikely(mag_log_enabled)) fprintf(stdout,   MAG_CC_CYAN "[magnetron] " MAG_CC_RESET msg "\n", ## __VA_ARGS__); } while (0)
#define mag_log_info_force(msg, ...) do { fprintf(stdout,   MAG_CC_CYAN "[magnetron] " MAG_CC_RESET msg "\n", ## __VA_ARGS__); } while (0)
#define mag_log_warn(msg, ...) do { fprintf(stdout,  MAG_CC_CYAN "[magnetron] " MAG_CC_RESET MAG_SRC_NAME " " MAG_CC_YELLOW msg MAG_CC_RESET "\n", ## __VA_ARGS__); fflush(stdout); } while (0)
#define mag_log_error(msg, ...) do { fprintf(stdout,  MAG_CC_CYAN "[magnetron] " MAG_CC_RESET MAG_SRC_NAME " " MAG_CC_RED msg MAG_CC_RESET "\n", ## __VA_ARGS__); fflush(stdout); } while (0)

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
        (void*)(ptr), \
        (void*)(base), \
        (void*)((char*)(base)+(N)), \
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

/* Increment pointer or size with correct type alignment. */
static inline void* mag_pincr(void** p, size_t sz, size_t align) {
    void* pp = (void*)(((uintptr_t)*p+align-1)&-align);
    *p = (void*)((uint8_t*)pp+sz);
    return pp;
}

/*
**  Fast u32 division and remainder. Paper:
**  Torbjörn Granlund and Peter L. Montgomery, "Division by Invariant Integers Using Multiplication",
**  ACM SIGPLAN Notices, Issue 6, Vol 29, 61-72, June 1994.
**	http://gmplib.org/~tege/divcnst-pldi94.pdf
*/
typedef struct mag_div_state_t {
    uint32_t s1;
    uint32_t s2;
    uint32_t d;
} mag_div_state_t;

#ifdef _MSC_VER
static inline DWORD mag_clz(DWORD x) {
    DWORD z = 0;
    return _BitScanForward(&z, x) ? z : 32;
}
#else
#define mag_clz(x) __builtin_clz(x)
#endif

static inline mag_div_state_t mag_ivdiv_mkdi(uint32_t d) { /* Create packed division info from devisor. */
    uint32_t l = (d-1) ? 32 - mag_clz((d-1)) : 0;
    uint32_t s1 = l > 1 ? 1 : l&0xff;
    uint32_t s2 = !l ? 0 : (l-1)&0xff;
    mag_div_state_t r;
    r.s1 = s1;
    r.s2 = s2;
    r.d = (uint32_t)(((1ull<<l) - d)*0x100000000ull)/d + 1;
    return r;
}

/* Performs c = ab with overflow checking. Returns true on overflow, else false. */
static bool MAG_AINLINE mag_mulov64(int64_t a, int64_t b, int64_t* c) {
    #ifdef _MSC_VER
    #ifdef _M_ARM64
        uint64_t high = __umulh(a, b);
        *c = a*b;
        return high != (*c>>63);
    #else
        int64_t high;
        int64_t low = _mul128(a, b, &high);
        int64_t sign = low >> 63;
        *c = low;
        return high != sign;
    #endif
    #else
    #if __SIZEOF_LONG_LONG__ == 8 && __SIZEOF_LONG__ == 8
        return __builtin_smulll_overflow(a, b, (long long*)c);
    #else
        return __builtin_smull_overflow(a, b, c);
    #endif
    #endif
}

/*
** r = x / y.
** Fast division using invariant multiplication.
** Up to 40x times faster on my Threadripper 3970x. Faster on my M3 Pro too.
*/
static inline uint32_t mag_ivdiv32(uint32_t x, uint32_t y, mag_div_state_t di) {
    (void)y;
    uint32_t t = (uint64_t)x*(di.d)>>32;
    return (t + ((x - t)>>((di.s1)&0xff)))>>(di.s2);
}
/* r = x % y. Fast remainder using invariant multiplication */
static inline uint32_t mag_ivrem32(uint32_t x, uint32_t y, mag_div_state_t ctx) {
    return x - y*mag_ivdiv32(x, y, ctx);
}

#ifdef _WIN32 /* WIN32 specific threading and synchronization. */

typedef DWORD mag_thread_ret_t;
#define MAG_THREAD_RET_NONE 0

typedef HANDLE mag_thread_t;

static void mag_thread_create(mag_thread_t* out, mag_thread_ret_t (*f)(void*), void* arg) { /* WIN32 -> pthread style wrapper. */
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

typedef void* mag_thread_ret_t;
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

extern void mag_thread_set_prio(mag_thread_prio_t prio); /* Set thread scheduling priority of current thread. */
extern void mag_thread_set_name(const char* name); /* Set thread name. */
extern void mag_thread_yield(void); /* Yield current thread. */
extern int mag_futex_wait(volatile mag_atomic32_t* addr, mag_atomic32_t expect);
extern void mag_futex_wake1(volatile mag_atomic32_t* addr);
extern void mag_futex_wakeall(volatile mag_atomic32_t* addr);

/* Dynamic zero-terminated string buffer. */
typedef struct mag_sstream_t {
    char* buf;
    size_t len;
    size_t cap;
} mag_sstream_t;

extern void mag_sstream_init(mag_sstream_t* ss);
extern void mag_sstream_free(mag_sstream_t* ss);
extern void mag_sstream_reserve_more(mag_sstream_t* ss, size_t extra);
extern void mag_sstream_vappend(mag_sstream_t* ss, const char* fmt, va_list ap);
extern void mag_sstream_append(mag_sstream_t* ss, const char* fmt, ...);
extern void mag_sstream_append_strn(mag_sstream_t* ss, const char* str, size_t len);
extern void mag_sstream_putc(mag_sstream_t* ss, char c);
extern void mag_sstream_flush(mag_sstream_t* ss, FILE* f);

/* Operation parameter */

/* Operation parameter type tag. */
typedef enum mag_opparam_type_t {
    MAG_OPP_NONE  = 0,
    MAG_OPP_E8M23 = 1,    /* fp32 */
    MAG_OPP_I64 = 2,    /* 64-bit signed integer. */

    MAG_OPP__NUM
} mag_opparam_type_t;
mag_static_assert(((MAG_OPP__NUM-1) & ~3) == 0); /* Must fit in 2 bits */

extern const char* const mag_op_param_type_names[MAG_OPP__NUM]; /* Operation parameter type names. */

/*
** The opp (Operation Parameter) is used to pass additional data to the operation. For example:
**      The FILL init op uses the opp to pass the value to fill the tensor buffer with.
**      The permute op receives all permutation axes in the operation parameters.
** The opp is a tagged union (variant).
*/
typedef struct mag_opparam_t {
    uint8_t type : 2;
    int64_t i62 : 62;
} mag_opparam_t;

static MAG_AINLINE mag_opparam_t mag_op_param_none() {
    mag_opparam_t p;
    p.type = MAG_OPP_NONE;
    p.i62 = 0;
    return p;
}

static MAG_AINLINE mag_opparam_t mag_op_param_wrap_e8m23(mag_e8m23_t x) {
    uint32_t u32;
    memcpy(&u32, &x, sizeof(u32));
    mag_opparam_t p;
    p.type = MAG_OPP_E8M23;
    p.i62 = u32;
    return p;
}

static MAG_AINLINE mag_opparam_t mag_op_param_wrap_i64(int64_t x) {
    mag_assert(!((x < 0 ? ~x+1 : x) & ~((1ULL<<62)-1)), "op param int64 value must fit in 62 bit, but is: %" PRIi64, x);
    mag_opparam_t p;
    p.type = MAG_OPP_I64;
    p.i62 = x;
    return p;
}

/* Unpack value from packed opp. Panics if the type is not the expected type. */
static MAG_AINLINE mag_e8m23_t mag_op_param_unpack_e8m23_or_panic(mag_opparam_t pa) {
    mag_assert(pa.type == MAG_OPP_E8M23, "invalid op param type: %d", pa.type);
    mag_e8m23_t e8m23 = 0.f;
    uint32_t u32 = (uint32_t)pa.i62;
    memcpy(&e8m23, &u32, sizeof(e8m23));
    return e8m23;
}

static MAG_AINLINE int64_t mag_op_param_unpack_i64_or_panic(mag_opparam_t pa) {
    mag_assert(pa.type == MAG_OPP_I64, "invalid op param type: %d", pa.type);
    return pa.i62;
}

static MAG_AINLINE mag_e8m23_t mag_op_param_unpack_e8m23_or(mag_opparam_t pa, mag_e8m23_t fallback) {
    return pa.type == MAG_OPP_E8M23 ? mag_op_param_unpack_e8m23_or_panic(pa) : fallback;
}

static MAG_AINLINE int64_t mag_op_param_unpack_i64_or(mag_opparam_t pa, int64_t fallback) {
    return pa.type == MAG_OPP_I64 ? mag_op_param_unpack_i64_or_panic(pa) : fallback;
}

/* Helper for filling the operation parameters array and validating the amount. */
typedef struct mag_opparam_layout_t {
    mag_opparam_t slots[MAG_MAX_OP_PARAMS];
    uint32_t count;
} mag_op_param_layout_t;

static inline void mag_op_param_layout_init(mag_op_param_layout_t* set) {
    set->count = 0;
    for (int i=0; i < MAG_MAX_OP_PARAMS; ++i)
        set->slots[i] = mag_op_param_none();
}

static inline size_t mag_op_param_layout_insert(mag_op_param_layout_t* set, mag_opparam_t param) {
    mag_assert(set->count < MAG_MAX_OP_PARAMS, "Too many operation parameters");
    set->slots[set->count] = param;
    return set->count++;
}

static inline void mag_op_param_layout_store(mag_op_param_layout_t* set, size_t idx, mag_opparam_t param) {
    mag_assert(idx < set->count, "Invalid operation parameter index");
    mag_assert(set->slots[idx].type == MAG_OPP_NONE, "Operation parameter already set");
    set->slots[idx] = param;
}

static inline void mag_op_param_layout_transfer(const mag_op_param_layout_t* set, mag_opparam_t (*out)[MAG_MAX_OP_PARAMS]) {
    memcpy(*out, set->slots, set->count*sizeof(*set->slots));
    for (size_t i=set->count; i < MAG_MAX_OP_PARAMS; ++i)
        (*out)[i] = mag_op_param_none();
}

typedef enum mag_opflags_t {
    MAG_OP_FLAG_NONE = 0,
    MAG_OP_FLAG_SUPPORTS_INPLACE = 1<<0,                /* Allows to be executed inplace on the input tensor. */
    MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING = 1<<1,      /* Supports multithreading on CPU. */
} mag_opflags_t;

#define MAG_OP_FLAGS_COMMON (MAG_OP_FLAG_SUPPORTS_INPLACE+MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING)
#define mag_params(...) { __VA_ARGS__ }

/* Enumerator, Input Count, Output Count, DType Mask, Op Param Layout, Flags, Backward Function, cpu growth, cpu tresh */
#define mag_opdef(_, __)\
    _(NOP, 0, 0, NONE, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(FILL, 1, 0, ALL, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(MASKED_FILL, 1, 0, ALL, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(RAND_UNIFORM, 1, 0, NUMERIC, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(RAND_NORMAL, 1, 0, FP, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(RAND_BERNOULLI, 1, 0, BOOL, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(ARANGE, 1, 0, NUMERIC, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(CLONE, 1, 1, ALL, {}, MAG_OP_FLAG_NONE, clone)__\
    _(VIEW, 1, 1, ALL, {}, MAG_OP_FLAG_NONE, view)__\
    _(TRANSPOSE, 1, 1, ALL, {}, MAG_OP_FLAG_NONE, transpose)__\
    _(PERMUTE, 1, 1, ALL, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(MEAN, 1, 1, FP, {}, MAG_OP_FLAG_NONE, mean)__\
    _(MIN, 1, 1, FP, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(MAX, 1, 1, FP, {}, MAG_OP_FLAG_NONE, NULL)__\
    _(SUM, 1, 1, FP, {}, MAG_OP_FLAG_NONE, sum)__\
    _(ABS, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, abs)__\
    _(SGN, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(NEG, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, neg)__\
    _(LOG, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, log)__\
    _(SQR, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, sqr)__\
    _(SQRT, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, sqrt)__\
    _(SIN, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, sin)__\
    _(COS, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, cos)__\
    _(STEP, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(EXP, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, exp)__\
    _(FLOOR, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(CEIL, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(ROUND, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(SOFTMAX, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, softmax)__\
    _(SOFTMAX_DV, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(SIGMOID, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, sigmoid)__\
    _(SIGMOID_DV, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(HARD_SIGMOID, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(SILU, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, silu)__\
    _(SILU_DV, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(TANH, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, tanh)__\
    _(TANH_DV, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(RELU, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, relu)__\
    _(RELU_DV, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(GELU, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, gelu)__\
    _(GELU_APPROX, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, gelu)__\
    _(GELU_DV, 1, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(TRIL, 1, 1, ALL, mag_params(MAG_OPP_I64), MAG_OP_FLAG_NONE, NULL)__\
    _(TRIU, 1, 1, ALL, mag_params(MAG_OPP_I64), MAG_OP_FLAG_NONE, NULL)__\
    _(MULTINOMIAL, 1, 1, FP, mag_params(MAG_OPP_I64, MAG_OPP_I64), MAG_OP_FLAG_NONE, NULL)__\
    _(ADD, 2, 1, NUMERIC, {}, MAG_OP_FLAGS_COMMON, add)__\
    _(SUB, 2, 1, NUMERIC, {}, MAG_OP_FLAGS_COMMON, sub)__\
    _(MUL, 2, 1, NUMERIC, {}, MAG_OP_FLAGS_COMMON, mul)__\
    _(DIV, 2, 1, NUMERIC, {}, MAG_OP_FLAGS_COMMON, div)__\
    _(MATMUL, 2, 1, FP, {}, MAG_OP_FLAGS_COMMON, matmul)__\
    _(REPEAT_BACK, 2, 1, FP, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(GATHER, 2, 1, ALL, mag_params(MAG_OPP_I64), MAG_OP_FLAG_NONE, NULL)__\
    _(AND, 2, 1, INTEGRAL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(OR, 2, 1, INTEGRAL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(XOR, 2, 1, INTEGRAL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(NOT, 1, 1, INTEGRAL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(SHL, 2, 1, INTEGRAL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(SHR, 2, 1, INTEGRAL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(EQ, 2, 1, ALL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(NE, 2, 1, ALL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(LE, 2, 1, ALL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(GE, 2, 1, ALL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(LT, 2, 1, ALL, {}, MAG_OP_FLAGS_COMMON, NULL)__\
    _(GT, 2, 1, ALL, {}, MAG_OP_FLAGS_COMMON, NULL)__\

/* Standard opcodes, not including initialization operators. */
typedef enum mag_opcode_t {
    #define _(enu, in, out, dtm, opp, flags, diff) MAG_OP_##enu
        mag_opdef(_, MAG_SEP)
    #undef _
    MAG_OP__NUM
} mag_opcode_t;
mag_static_assert(MAG_OP_NOP == 0);
mag_static_assert(MAG_OP_GT+1 == MAG_OP__NUM);
mag_static_assert(MAG_OP__NUM <= 0xff); /* Must fit in one byte */

typedef uint8_t mag_dtype_mask_t; /* Bitmask of supported dtypes, 1 bit per dtype. */
mag_static_assert(MAG_DTYPE__NUM <= 8); /* Must fit in 8 bits, if this fails increase the type of dtpe_mask. */
#define mag_dtype_bit(x) (((mag_dtype_mask_t)1)<<((x)&7))
#define mag_dtype_mask(enume) mag_dtype_bit(MAG_DTYPE_##enume)
#define MAG_DTYPE_MASK_NONE 0
#define MAG_DTYPE_MASK_ALL (mag_dtype_mask(E8M23)|mag_dtype_mask(E5M10)|mag_dtype_mask(BOOL)|mag_dtype_mask(I32)) /* All data types */
#define MAG_DTYPE_MASK_FP (mag_dtype_mask(E8M23)|mag_dtype_mask(E5M10))         /* Floating-point data types */
#define MAG_DTYPE_MASK_INTEGRAL (mag_dtype_mask(BOOL)|mag_dtype_mask(I32))      /* Integral data types with boolean */
#define MAG_DTYPE_MASK_INTEGER (MAG_DTYPE_MASK_INTEGRAL&~mag_dtype_mask(BOOL))  /* Integral (integer) data types without boolean */
#define MAG_DTYPE_MASK_NUMERIC (MAG_DTYPE_MASK_ALL&~mag_dtype_mask(BOOL))       /* Numeric data types (all except boolean) */
#define MAG_DTYPE_MASK_BOOL (mag_dtype_mask(BOOL))                              /* Boolean data type */

#define MAG_OP_INOUT_DYN SIZE_MAX /* Dynamic input/output count. Used for operations that can have arbitrary number of inputs/outputs. */

typedef struct mag_au_state_t mag_au_state_t;

/* Stores operator metadata such as operation type, number of inputs and parameters, and the types of the parameters. */
typedef struct mag_opmeta_t {
    const char* const mnemonic;
    const uint32_t in;
    const uint32_t out;
    const mag_dtype_mask_t dtype_mask;
    const mag_opparam_type_t op_param_layout[MAG_MAX_OP_PARAMS];
    const mag_opflags_t flags;
    void (*const backward)(mag_au_state_t*, mag_tensor_t**);
} mag_opmeta_t;

extern const mag_opmeta_t* mag_op_meta_of(mag_opcode_t opc); /* Get operation metadata for a specific opcode. */

typedef uint64_t mag_rcint_t;
#define MAG_RCINTEGRAL_MAX UINT64_MAX
#define MAG_RCINTEGRAL_PRI PRIu64

/* Header for all objects that are reference counted. */
typedef struct mag_rccontrol_t {
    mag_rcint_t rc;                      /* Strong reference count. Object is deallocated if this reaches zero. */
    void* self;                    /* Pointer to the self. */
    void (*dtor)(void*);  /* Destructor function (required). */
} mag_rccontrol_t;

/* Initialize reference count header for a new object. Self-reference and destructor functon must be provided. */
static MAG_AINLINE mag_rccontrol_t mag_rc_control_init(void* self, void (*dtor)(void*)) {
    mag_assert2(self && dtor); /* Self and destructor must be set. */
    mag_rccontrol_t control;
    control.rc = 1;
    control.self = self;
    control.dtor = dtor;
    return control;
}

static MAG_AINLINE void mag_rc_control_incref(mag_rccontrol_t* rcb) { /* Increment reference count (retain). */
    mag_assert(++rcb->rc < MAG_RCINTEGRAL_MAX, "reference count overflow, max RC: %" MAG_RCINTEGRAL_PRI, MAG_RCINTEGRAL_MAX);
}
static MAG_AINLINE bool mag_rc_control_decref(mag_rccontrol_t* rcb) { /* Decrement reference count (release). */
    mag_assert(rcb->rc, "reference count underflow (double free)");
    if (!--rcb->rc) { /* Decref and invoke destructor. */
        (*rcb->dtor)(rcb->self);
        return true; /* Object was destroyed. */
    }
    return false;
}

/* Memory chunk for intrusive memory pool. */
typedef struct mag_pool_chunk_t mag_pool_chunk_t;
struct mag_pool_chunk_t {
    uint8_t* bot;          /* Bottom (base) of chunk */
    uint8_t* top;          /* Top of chunk, grows downwards towards bottom */
    mag_pool_chunk_t* next;   /* Link to next chunk */
};

/* Fast memory allocator for memory blocks of same size. Obtains a memory pool and freelist for fast de/allocation. */
typedef struct mag_fixed_pool_t {
    size_t block_size;                   /* Size of each allocated block */
    size_t block_align;                  /* Alignment requirements of each block. */
    size_t blocks_per_chunk;             /* How many blocks fit in each chunk */
    mag_pool_chunk_t* chunks;      /* Linked list of all chunks */
    mag_pool_chunk_t* chunk_head;  /* Last chunk */
    void* free_list;            /* Intrusive single linked list of free chunks */
    uint64_t num_freelist_hits;          /* Number of cache (free-list) hits */
    uint64_t num_pool_hits;              /* Number of cache (pool) hits */
    uint64_t num_chunks;                 /* Number of used chunks */
    uint64_t num_allocs;                 /* Number of total allocations */
} mag_fixed_pool_t;

extern MAG_EXPORT void mag_fixed_pool_init(mag_fixed_pool_t* pool, size_t block_size, size_t block_align, size_t blocks_per_chunk);
extern MAG_EXPORT void* mag_fixed_pool_alloc_block(mag_fixed_pool_t* pool);
extern MAG_EXPORT void mag_fixed_pool_free_block(mag_fixed_pool_t* pool, void* blk);
extern MAG_EXPORT void mag_fixed_pool_destroy(mag_fixed_pool_t* pool);
extern MAG_EXPORT void mag_fixed_pool_print_info(mag_fixed_pool_t* pool, const char* name);

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
typedef struct mag_idevice_t mag_idevice_t;

typedef enum mag_transfer_dir_t {
    MAG_TRANSFER_DIR_H2D,   /* Host -> Device. */
    MAG_TRANSFER_DIR_D2H,   /* Device -> Host. */
} mag_transfer_dir_t;

/* Buffer interface on a compute device */
typedef struct mag_istorage_t mag_istorage_t;
struct mag_istorage_t {
    mag_context_t* ctx;
    mag_rccontrol_t rc_control;      /* Reference count control block. */
    uintptr_t base;                     /* Pointer to buffer on device. Might point to GPU or any other device memory. */
    size_t size;                        /* Size of buffer in bytes. */
    size_t alignment;                   /* Alignment of buffer. */
    size_t granularity;                 /* Element size granularity. */
    mag_dtype_t dtype;                    /* Data type of buffer. */
    mag_idevice_t* host;  /* Host device. */
    void (*broadcast)(mag_istorage_t* sto, size_t offs, const void* src, size_t stride);
    void (*transfer)(mag_istorage_t* sto, mag_transfer_dir_t dir, size_t offs, void* inout, size_t size);
    void (*convert)(mag_istorage_t* sto, mag_transfer_dir_t dir, size_t offs, void* inout, size_t size, mag_dtype_t inout_type);
};

typedef struct mag_command_t {
    mag_opcode_t op;
    mag_tensor_t** in;
    mag_tensor_t** out;
    uint32_t num_in;
    uint32_t num_out;
    mag_opparam_t params[MAG_MAX_OP_PARAMS];
} mag_command_t;

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
struct mag_idevice_t {
    mag_context_t* ctx;
    char name[128];                 /* Device name. */
    void* impl;            /* Device specific implementation, if applicable. */
    bool is_async;                  /* If device is async. */
    mag_device_type_t type;         /* Device type enum. */
    void (*submit)(mag_idevice_t* dvc, const mag_command_t* cmd);                                       /* Execute a single op. */
    void (*alloc_storage)(mag_idevice_t* dvc, mag_istorage_t** out, size_t size, mag_dtype_t dtype);    /* Allocate storage buffer in device memory */
};

/* Device creation and destruction. */
typedef struct mag_idevice_factory_t {
    mag_idevice_t* (*init)(mag_context_t* ctx, const mag_device_desc_t* desc);      /* Initialize device. */
    void (*destroy)(mag_idevice_t* dvc);         /* Destroy device. */
} mag_idevice_factory_t;

/* Global device factories. Implemented in magnetron_device_registry.c */
extern mag_idevice_t* mag_init_dynamic_device(mag_context_t* ctx, const mag_device_desc_t* desc);
extern void mag_destroy_dynamic_device(mag_idevice_t* dvc);

#if defined(__x86_64__) || defined(_M_X64)

#define mag_amd64_capdef(_, __)\
    _(NONE)__\
    _(INTEL)__\
    _(AMD)__\
    \
    _(SSE)__\
    _(SSE2)__\
    _(SSE3)__\
    _(SSSE3)__\
    _(SSE41)__\
    _(SSE42)__\
    _(SSE4A)__\
    \
    _(AVX)__\
    _(FMA)__\
    _(AVX2)__\
    _(F16C)__\
    _(AVX_VNNI)__\
    _(AVX_VNNI_INT8)__\
    _(AVX_NE_CONVERT)__\
    _(AVX_IFMA)__\
    _(AVX_VNNI_INT16)__\
    _(AVX10)__\
    \
    _(AVX512_F)__\
    _(AVX512_DQ)__\
    _(AVX512_IFMA)__\
    _(AVX512_PF)__\
    _(AVX512_ER)__\
    _(AVX512_CD)__\
    _(AVX512_BW)__\
    _(AVX512_VL)__\
    _(AVX512_VBMI)__\
    _(AVX512_4VNNIW)__\
    _(AVX512_4FMAPS)__\
    _(AVX512_VBMI2)__\
    _(AVX512_VNNI)__\
    _(AVX512_BITALG)__\
    _(AVX512_VPOPCNTDQ)__\
    _(AVX512_BF16)__\
    _(AVX512_VP2INTERSECT)__\
    _(AVX512_FP16)__\
    \
    _(AMX_TILE)__\
    _(AMX_INT8)__\
    _(AMX_BF16)__\
    _(AMX_FP16)__\
    _(AMX_TRANSPOSE)__\
    _(AMX_TF32)__\
    _(AMX_AVX512)__\
    _(AMX_MOVRS)__\
    _(AMX_FP8)__\
    \
    _(BMI1)__\
    _(BMI2)__\
    \
    _(OSXSAVE)__\
    _(GFNI)__\
    _(APX_F)__

typedef enum mag_amd64_cap_t {
#define _(name) MAG_AMD64_CAP_##name
    mag_amd64_capdef(_, MAG_SEP)
    MAG_AMD64_CAP__NUM
#undef _
} mag_amd64_cap_t;
typedef uint64_t mag_amd64_cap_bitset_t;
mag_static_assert(MAG_AMD64_CAP__NUM <= sizeof(mag_amd64_cap_bitset_t)<<3); /* Must fit in 64 bits. */
#define mag_amd64_cap_bit(x) (((mag_amd64_cap_bitset_t)1)<<((x)&63))
#define mag_amd64_cap(name) mag_amd64_cap_bit(MAG_AMD64_CAP_##name)

extern const char* const mag_amd64_cpu_cap_names[MAG_AMD64_CAP__NUM]; /* Names of x86-64 CPU capabilities. */

#elif defined(__aarch64__) || defined(_M_ARM64) /* ARM 64 specific CPU features. */

#define mag_armd64_capdef(_, __) /* Enumerator */\
    _(NONE)__\
    _(NEON)__\
    _(DOTPROD)__\
    _(I8MM)__\
    _(F16SCA)__\
    _(F16VEC)__\
    _(BF16)__\
    _(SVE)__\
    _(SVE2)__

typedef enum mag_arm64_cap_t {
    #define _(ident) MAG_ARM64_CAP_##ident
    mag_armd64_capdef(_, MAG_SEP)
    MAG_ARM64_CAP__NUM
    #undef _
} mag_arm64_cap_t;
typedef uint64_t mag_arm64_cap_bitset_t;
mag_static_assert(MAG_ARM64_CAP__NUM <= sizeof(mag_arm64_cap_bitset_t)<<3); /* Must fit in 64 bits. */
#define mag_arm64_cap_bit(x) (((mag_arm64_cap_bitset_t)1)<<((x)&63))
#define mag_arm64_cap(name) mag_arm64_cap_bit(MAG_ARM64_CAP_##name)

extern const char* const mag_arm64_cpu_cap_names[MAG_ARM64_CAP__NUM];

#endif

/* Context specific flags. */
typedef enum mag_context_flags_t {
    MAG_CTX_FLAG_NONE = 0,
    MAG_CTX_FLAG_GRAD_RECORDER = 1<<0,     /* Gradient recording is currently active. */
} mag_context_flags_t;

struct mag_context_t {
    struct {
        char os_name[128];                      /* OS name. */
        char cpu_name[128];                     /* CPU name. */
        uint32_t cpu_virtual_cores;             /* Virtual CPUs. */
        uint32_t cpu_physical_cores;            /* Physical CPU cores. */
        uint32_t cpu_sockets;                   /* CPU sockets. */
        size_t cpu_l1_size;                    /* L1 data cache size in bytes. */
        size_t cpu_l2_size;                     /* L2 cache size in bytes. */
        size_t cpu_l3_size;                     /* L3 cache size in bytes. */
        size_t phys_mem_total;                  /* Total physical memory in bytes. */
        size_t phys_mem_free;                   /* Free physical memory in bytes. */
#if defined(__x86_64__) || defined(_M_X64)
        mag_amd64_cap_bitset_t amd64_cpu_caps;  /* x86-64 CPU capability bits. */
        uint32_t amd64_avx10_ver;               /* x86-64 AVX10 version. */
#elif defined (__aarch64__) || defined(_M_ARM64)
        mag_arm64_cap_bitset_t arm64_cpu_caps;  /* ARM64 CPU features. */
        int64_t arm64_cpu_sve_width;            /* ARM64 SVE vector register width. */
#endif
    } machine;
    size_t num_tensors;                         /* Total tensor instances allocated. */
    size_t num_storages;                        /* Total storage buffers allocated. */
    mag_fixed_pool_t tensor_pool;               /* Tensor header memory pool. */
    mag_fixed_pool_t storage_pool;              /* Storage header memory pool. */
    mag_fixed_pool_t view_meta_pool;            /* View metadata header memory pool. */
    mag_fixed_pool_t au_state_pool;             /* Autodiff state memory pool. */
    mag_context_flags_t flags;                  /* Context flags. */
    mag_prng_algo_t prng_algo;                   /* Active PRNG algorithm. */
    uintptr_t tr_id;                            /* Context thread ID. */
    size_t sh_len;                              /* Number of shutdown hooks. */
    size_t sh_cap;                              /* Maximum number of shutdown hooks. */
    mag_device_type_t device_type;              /* Active compute device type. */
    mag_idevice_t* device;             /* Active compute device interface. */
    void* ud;                         /* User data. */
#ifdef MAG_DEBUG
    mag_tensor_t* alive_head;           /* List of alive tensors used for leak detection. */
#endif
};

/* Tensor specific flags. */
typedef enum mag_tensor_flags_t {
    MAG_TFLAG_NONE = 0,
    MAG_TFLAG_IS_VIEW = 1<<0,           /* Tensor is a view. */
    MAG_TFLAG_IS_GRAD = 1<<1,           /* Tensor is a gradient. */
    MAG_TFLAG_REQUIRES_GRAD = 1<<2,     /* Tensor requires gradient. */

    MAG_TFLAG_LEN = 4                   /* Number of flags. */
} mag_tensor_flags_t;
mag_static_assert(MAG_TFLAG_LEN <= 8); /* Must fit in one byte */

/* Metadata for view tensors */
typedef struct mag_view_meta_t {
    mag_rccontrol_t rc;
    mag_tensor_t* base;
    uint32_t version_snapshot;
} mag_view_meta_t;

extern mag_view_meta_t* mag_view_meta_alloc(mag_tensor_t* base);

/* Autodiff state for parameters */
struct mag_au_state_t {
    mag_context_t* ctx;
    mag_rccontrol_t rc;
    mag_opcode_t op;
    mag_tensor_t* op_inputs[MAG_MAX_OP_INPUTS];
    mag_opparam_t op_params[MAG_MAX_OP_PARAMS];
    mag_tensor_t* grad;
};
extern mag_au_state_t* mag_au_state_lazy_alloc(mag_au_state_t** au_state, mag_context_t* ctx);

/*
** Reference counted tensor header. Stores shape, strides, gradient and other metadata.
** The actual data buffer is compute-device specific and can be only accessed via the storage buffer.
** A tensor can be a view, which references the storage buffer of another tensor, but views have their own header too.
*/
struct mag_tensor_t {
    mag_context_t*  ctx;                            /* Host context. */
    mag_rccontrol_t rc_control;                     /* Reference counting control block. */
    int64_t rank;                                   /* Number of active dimensions. [1, MAX_DIMS] */
    int64_t shape[MAG_MAX_DIMS];                    /* Shape of the tensor. */
    int64_t strides[MAG_MAX_DIMS];                  /* Strides of the tensor. We store the strides in element counts and NOT in bytes. */
    mag_dtype_t dtype : 8;                          /* Data type of the tensor. */
    mag_tensor_flags_t flags : 8;                   /* Tensor flags. */
    mag_istorage_t* storage;                        /* Storage buffer. */
    int64_t numel;                                  /* Number of elements in the tensor. */
    int64_t storage_offset;                         /* Offset in elements in the storage buffer for views. */
    mag_view_meta_t* view_meta;                     /* View metadata, if this is a view. */
    mag_au_state_t* au_state;                       /* Autodiff state, if gradient recording is active. */
    uint64_t version;                               /* Version of the tensor. Used for views to detect changes in the base tensor. */
#ifdef MAG_DEBUG
    mag_tensor_t* alive_next;                       /* Next alive tensor used for leak detection. */
#endif
};

/* PRNG state for random number generation. */
typedef struct mag_prng_state_t {
    union { /* PRNG state of active algo. */
        struct {
            uint64_t state;
            uint64_t inc;
        } pcg;
        struct {
            uint32_t remaining;
            uint32_t next;
            uint32_t state[624];
        } mersenne;
    };
    mag_prng_algo_t algo; /* PRNG algorithm. */
} mag_prng_state_t;

/* Initialize and seed PRNG with specific algorithm. */
void mag_prng_seed(mag_prng_state_t* prng, mag_prng_algo_t algo, uint64_t seed);

typedef struct mag_matmul_block_tune_info_t {
    int64_t nthreads;
    int64_t elsize;
    int64_t vecreg_width;
    int64_t M;
    int64_t N;
    int64_t K;
    int64_t l1_size;
    int64_t l2_size;
    mag_e11m52_t l1_load_factor;
    mag_e11m52_t l2_load_factor;
    int64_t min_tile_flops;
    mag_e11m52_t split_a;
    int64_t min_n_factor;
    int64_t min_m_factor;
} mag_matmul_block_tune_info_t;

typedef struct mag_matmul_block_params_t {
    int64_t MR;
    int64_t NR;
    int64_t MC;
    int64_t KC;
    int64_t NC;
} mag_matmul_block_params_t;

extern void mag_matmul_tune_block_params(const mag_matmul_block_tune_info_t* info, mag_matmul_block_params_t* params);

/* CPU Compute kernel payload passed to each CPU thread. */
typedef struct mag_kernel_payload_t {
    const mag_command_t* cmd;
    int64_t thread_num;
    int64_t thread_idx;
    mag_prng_state_t* local_prng;
    volatile mag_atomic64_t* mm_next_tile;
    mag_matmul_block_params_t mm_params;
} mag_kernel_payload_t;

/*
** Stores function-pointer lookup table for all compute kernels.
** The lookup table is used to dispatch the correct kernel for each operation by indexing with the opcode.
** The CPU runtime dynamically fills these arrays with the best fitting kernel depending on the detected CPU.
** See magnetron_cpu.c for details.
*/
typedef struct mag_kernel_registry_t {
    void (*init)(void);
    void (*deinit)(void);
    void (*operators[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_kernel_payload_t*);       /* Eval operator kernels. */
    void (*vector_cast)(size_t nb, const void* src, mag_dtype_t src_t, void* dst, mag_dtype_t dst_t); /* Vector cast (dtype conversion) kernel. */
    size_t (*vreg_width)(void);
} mag_kernel_registry_t;

/* Combine two hash values. */
static inline void mag_hash_combine(uint32_t* seed, uint32_t value) {
    *seed ^= value + 0x9e3779b9 + (*seed<<6) + (*seed>>2);
}

extern uint64_t mag_hash(const void* key, size_t len, uint32_t seed); /* Compute murmur3_64 hash */
extern uint32_t mag_crc32c(const void* buffer, size_t size); /* Compute CRC32 checksum with CRC32c polynomial. */
extern bool mag_utf8_validate(const char* str, size_t len);
extern char* mag_strdup(const char* s);
extern bool mag_solve_view_strides(int64_t (*out)[MAG_MAX_DIMS], const int64_t* osz, const int64_t* ost, int64_t ork, const int64_t* nsz, int64_t nrk);
extern void mag_infer_missing_dim(int64_t (*out)[MAG_MAX_DIMS], const int64_t* dims, int64_t rank, int64_t numel);
extern bool mag_compute_broadcast_shape(const mag_tensor_t* a, const mag_tensor_t* b, int64_t* dims, int64_t* rank);

#ifdef __cplusplus
}
#endif

#endif
