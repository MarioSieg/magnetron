/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
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

#define mag_assert_name2(name, line) name ## line
#define mag_assert_name(line) mag_assert_name2(_assert_, line)
#define mag_static_assert(expr) extern void mag_assert_name(__LINE__)(bool STATIC_ASSERTION_FAILED[_Nonnull((expr)?1:-1)])
#define MAG_SEP ,

#define MAG_GELU_COEFF 0.044715f /* Coefficient for GELU approximation. */

/* Compute execution stage. */
typedef enum mag_ExecStage {
    MAG_STAGE_EVAL,     /* Eval op. */
    MAG_STAGE_INIT      /* Execute init op. */
} mag_ExecStage;

#define MAG_MAX_CPUS 8192               /* Maximum number of virtual CPUs supported. */
#define MAG_MAX_NUMA_NODES 64           /* Maximum number of NUMA nodes supported. */

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

typedef int64_t mag_Atomic64;       /* Atomic integer type */
typedef enum mag_MemoryOrder {             /* Atomic memory order */
    MAG_MO_RELAXED = __ATOMIC_RELAXED,
    MAG_MO_CONSUME = __ATOMIC_CONSUME,
    MAG_MO_ACQUIRE = __ATOMIC_ACQUIRE,
    MAG_MO_RELEASE = __ATOMIC_RELEASE,
    MAG_MO_ACQ_REL = __ATOMIC_ACQ_REL,
    MAG_MO_SEQ_CST = __ATOMIC_SEQ_CST
} mag_MemoryOrder;

static MAG_AINLINE void mag_atomic_store(volatile mag_Atomic64* _Nonnull o, mag_Atomic64 x, mag_MemoryOrder order) { __atomic_store_n(o, x, order); }
static MAG_AINLINE mag_Atomic64 mag_atomic_load(volatile mag_Atomic64* _Nonnull o, mag_MemoryOrder order) { return __atomic_load_n(o, order); }
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_add(volatile mag_Atomic64* _Nonnull o, mag_Atomic64 x, mag_MemoryOrder order) { return __atomic_fetch_add(o, x, order); }
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_sub(volatile mag_Atomic64* _Nonnull o, mag_Atomic64 x, mag_MemoryOrder order) { return __atomic_fetch_sub(o, x, order); }
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_and(volatile mag_Atomic64* _Nonnull o, mag_Atomic64 x, mag_MemoryOrder order) { return __atomic_fetch_and(o, x, order); }
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_or(volatile mag_Atomic64* _Nonnull o, mag_Atomic64 x, mag_MemoryOrder order) { return __atomic_fetch_or(o, x, order); }
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_xor(volatile mag_Atomic64* _Nonnull o, mag_Atomic64 x, mag_MemoryOrder order) { return __atomic_fetch_xor(o, x, order); }
static MAG_AINLINE mag_Atomic64 mag_atomic_exchange(volatile mag_Atomic64* _Nonnull o, mag_Atomic64 x, mag_MemoryOrder order) { return __atomic_exchange_n(o, x, order); }
static MAG_AINLINE bool mag_atomic_compare_exchange_weak(volatile mag_Atomic64* _Nonnull o, mag_Atomic64* _Nonnull exp, mag_Atomic64* _Nonnull des, mag_MemoryOrder order_succ, mag_MemoryOrder order_fail) { return __atomic_compare_exchange(o, exp, des, true, order_succ, order_fail); }
static MAG_AINLINE bool mag_atomic_compare_exchange_strong(volatile mag_Atomic64* _Nonnull o, mag_Atomic64* _Nonnull exp, mag_Atomic64* _Nonnull des, mag_MemoryOrder order_succ, mag_MemoryOrder order_fail) { return __atomic_compare_exchange(o, exp, des, false, order_succ, order_fail); }

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

typedef __int64 mag_Atomic64;       /* Atomic integer type */
typedef enum mag_MemoryOrder {             /* Atomic memory order. Has no effect with MSVC for now, all operations are sequencial consistent. */
    MAG_MO_RELAXED,
    MAG_MO_CONSUME,
    MAG_MO_ACQUIRE,
    MAG_MO_RELEASE,
    MAG_MO_ACQ_REL,
    MAG_MO_SEQ_CST
} mag_MemoryOrder;

static MAG_AINLINE void mag_atomic_store(volatile mag_Atomic64* o, mag_Atomic64 x, mag_MemoryOrder order) {
    (void)order; _InterlockedExchange64(o, x);
}
static MAG_AINLINE mag_Atomic64 mag_atomic_load(volatile mag_Atomic64* o, mag_MemoryOrder order) {
    (void)order;
    mag_Atomic64 r;
    _InterlockedExchange64(&r, *o);
    return r;
}
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_add(volatile mag_Atomic64* o, mag_Atomic64 x, mag_MemoryOrder order) {
    (void)order;
    return _InterlockedExchangeAdd64(o, x);
}
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_sub(volatile mag_Atomic64* o, mag_Atomic64 x, mag_MemoryOrder order) {
    (void)order;
    return _InterlockedExchangeAdd64(o, -x);
}
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_and(volatile mag_Atomic64* o, mag_Atomic64 x, mag_MemoryOrder order) {
    (void)order;
    return _InterlockedAnd64(o, x);
}
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_or(volatile mag_Atomic64* o, mag_Atomic64 x, mag_MemoryOrder order) {
    (void)order;
    return _InterlockedOr64(o, x);
}
static MAG_AINLINE mag_Atomic64 mag_atomic_fetch_xor(volatile mag_Atomic64* o, mag_Atomic64 x, mag_MemoryOrder order) {
    (void)order;
    return _InterlockedXor64(o, x);
}
static MAG_AINLINE mag_Atomic64 mag_atomic_exchange(volatile mag_Atomic64* o, mag_Atomic64 x, mag_MemoryOrder order) {
    (void)order;
    return _InterlockedExchange64(o, x);
}
static MAG_AINLINE bool mag_atomic_compare_exchange_weak(volatile mag_Atomic64* o, mag_Atomic64 *exp, mag_Atomic64 *des, mag_MemoryOrder order_succ, mag_MemoryOrder order_fail) {
    (void)order_succ; (void)order_fail;
    mag_Atomic64 old = _InterlockedCompareExchange64(o, *des, *exp);
    if (old == *exp) return true; /* Emulate GCC's weak compare exchange. */
    else { *exp = old; return false; }
}
static MAG_AINLINE bool mag_atomic_compare_exchange_strong(volatile mag_Atomic64* o, mag_Atomic64 *exp, mag_Atomic64 *des, mag_MemoryOrder order_succ, mag_MemoryOrder order_fail) {
    (void)order_succ; (void)order_fail;
    mag_Atomic64 old = _InterlockedCompareExchange64(o, *des, *exp);
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

typedef double mag_E11M52;  /* f32 = IEEE 754 64-bit double precision float. */
typedef float mag_E8M23;    /* f64 = IEEE 754 32-bit single precision float. */
typedef struct mag_E5M10 { uint16_t bits; } mag_E5M10; /* f16 = IEEE 754 16-bit half precision float. */

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
#define mag_e5m10_pack2(x) (mag_E5M10){(x)&0xffffu}
    
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

/* Swap bytes of integer, ONLY if the host system is big endian. If the host is little endian, this is a no-op. */
static uint16_t MAG_AINLINE mag_bswap16(uint16_t x) {
    #ifdef MAG_BE
    #if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
        x = (uint32_t)__builtin_bswap16((int32_t)x);
    #else
        x = (x&0xff00)>>8 | x&0xff<<8;
    #endif
    #endif
    return x;
}

/* Swap bytes of integer, ONLY if the host system is big endian. If the host is little endian, this is a no-op. */
static uint32_t MAG_AINLINE mag_bswap32(uint32_t x) {
    #ifdef MAG_BE
        #if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
            x = (uint32_t)__builtin_bswap32((int32_t)x);
        #else
            x = (x&0xff000000)>>24 |
            (x&0xff0000)>>8 |
            (x&0xff00)<<8 |
            (x&0xff)<<24;
        #endif
    #endif
    return x;
}

/* Swap bytes of integer, ONLY if the host system is big endian. If the host is little endian, this is a no-op. */
static uint64_t MAG_AINLINE mag_bswap64(uint64_t x) {
    #ifdef MAG_BE
        #if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
            x = (uint64_t)__builtin_bswap64((int64_t)x);
        #else
            x = (x & 0xff00000000000000) >> 56 |
            (x & 0xff000000000000) >> 40 |
            (x & 0xff0000000000) >> 24 |
            (x & 0xff00000000) >> 8 |
            (x & 0xff000000) << 8 |
            (x & 0xff0000) << 24 |
            (x & 0xff00) << 40 |
            (x & 0xff) << 56;
        #endif
    #endif
    return x;
}

#define MAG_FMT_DIM_BUF_SIZE ((21+4)*MAG_MAX_DIMS)

extern MAG_EXPORT MAG_NORET MAG_COLDPROC void mag_panic(const char* _Nonnull msg, ...); /* Print error message and abort. */
extern MAG_EXPORT bool mag_log_enabled; /* Enable/disable logging to stdout/stderr. */

extern void MAG_COLDPROC mag_print_separator(FILE* _Nonnull f); /* Print a separator line. */
extern void mag_fmt_shape(char (*_Nonnull buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*_Nonnull dims)[MAG_MAX_DIMS], int64_t rank);

/*
** Allocator function. Can be set to custom allocator.
**   ! Never returns NULL, if re/allocation fails, it will abort the program by calling mag_panic().
**   ! Never zero initializes, use manual memset if zeroing is required.
**
** This single function is essentially a realloc and is used for allocating, reallocating and deallocating the following way:
** (*mag_alloc)(NULL, size) <=> malloc(size)            <- Passing NULL as reallocation base and size != 0 => allocation.
** (*mag_alloc)(ptr, size) <=> realloc(ptr, size)       <- Passing non-NULL pointer as reallocation base and size != 0 => reallocation.
** (*mag_alloc)(ptr, 0) <=> free(ptr)                   <- Passing NULL as reallocation base and size == 0 => free.
*/
extern MAG_EXPORT void* _Nonnull (*_Nonnull mag_alloc)(void* _Nullable blk, size_t size);

extern void* _Nonnull mag_alloc_aligned(size_t size, size_t align); /* Aligned allocator function. */
extern void mag_free_aligned(void* _Nonnull blk); /* Free aligned memory. */

/* Humanize memory size. Format and convert a memory size to the appropriate unit. For example. 1024 => 1 KiB */
extern void mag_humanize_memory_size(size_t n, mag_E11M52* _Nonnull out, const char* _Nonnull* _Nonnull unit);
extern uintptr_t mag_thread_id(void); /* Get current native thread ID. */

#define mag_swap(T, a, b) do { T tmp = (a); (a) = (b); (b) = tmp; } while (0)
#define mag_xmax(x, y) (((x) > (y)) ? (x) : (y))
#define mag_xmin(x, y) (((x) < (y)) ? (x) : (y))

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
static inline void* _Nonnull mag_pincr(void* _Nonnull* _Nonnull p, size_t sz, size_t align) {
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
typedef struct mag_DivisionContext {
    uint32_t s1;
    uint32_t s2;
    uint32_t d;
} mag_DivisionContext;

#ifdef _MSC_VER
static inline DWORD mag_clz(DWORD x) {
    DWORD z = 0;
    return _BitScanForward(&z, x) ? z : 32;
}
#else
#define mag_clz(x) __builtin_clz(x)
#endif

static inline mag_DivisionContext mag_ivdiv_mkdi(uint32_t d) { /* Create packed division info from devisor. */
    uint32_t l = (d-1) ? 32 - mag_clz((d-1)) : 0;
    uint32_t s1 = l > 1 ? 1 : l&0xff;
    uint32_t s2 = !l ? 0 : (l-1)&0xff;
    mag_DivisionContext r;
    r.s1 = s1;
    r.s2 = s2;
    r.d = (uint32_t)(((1ull<<l) - d)*0x100000000ull)/d + 1;
    return r;
}

/*
** r = x / y.
** Fast division using invariant multiplication.
** Up to 40x times faster on my Threadripper 3970x. Faster on my M3 Pro too.
*/
static inline uint32_t mag_ivdiv32(uint32_t x, uint32_t y, mag_DivisionContext di) {
    (void)y;
    uint32_t t = (uint64_t)x*(di.d)>>32;
    return (t + ((x - t)>>((di.s1)&0xff)))>>(di.s2);
}
/* r = x % y. Fast remainder using invariant multiplication */
static inline uint32_t mag_ivrem32(uint32_t x, uint32_t y, mag_DivisionContext ctx) {
    return x - y*mag_ivdiv32(x, y, ctx);
}

#ifdef _WIN32 /* WIN32 specific threading and synchronization. */

typedef DWORD mag_ThreadRet;
#define MAG_THREAD_RET_NONE 0

typedef HANDLE mag_Thread;

static void mag_thread_create(mag_Thread* out, mag_ThreadRet (*f)(void*), void* arg) { /* WIN32 -> pthread style wrapper. */
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)f, arg, 0, NULL);
    mag_assert2(handle != 0);
    *out = handle;
}

static void mag_thread_join(mag_Thread th) { /* WIN32 -> pthread style wrapper. */
    int ret = (int)WaitForSingleObject(th, INFINITE);
    CloseHandle(th);
    mag_assert2(ret == 0);
}

typedef SRWLOCK mag_Mutex;
#define mag_mutex_create(mtx) InitializeSRWLock(mtx)
#define mag_mutex_destroy(mtx)
#define mag_mutex_lock(mtx) AcquireSRWLockExclusive(mtx)
#define mag_mutex_unlock(mtx) ReleaseSRWLockExclusive(mtx)

typedef CONDITION_VARIABLE mag_CondVar;
#define mag_cv_create(cv) InitializeConditionVariable(cv)
#define mag_cv_destroy(cv)
#define mag_cv_wait(cv, mtx) SleepConditionVariableSRW(cv, mtx, INFINITE, 0)
#define mag_cv_signal(cv) WakeConditionVariable(cv)
#define mag_cv_broadcast(cv) WakeAllConditionVariable(cv)

#else /* POSIX threading and synchronization. */

typedef void* mag_ThreadRet;
#define MAG_THREAD_RET_NONE NULL

typedef pthread_t mag_Thread;
#define mag_thread_create(out, fn, arg) mag_assert2(pthread_create((out), NULL, (fn), (arg)) == 0)
#define mag_thread_join(th) mag_assert2(pthread_join((th), NULL) == 0)

typedef pthread_mutex_t mag_Mutex;
#define mag_mutex_create(mtx) mag_assert2(pthread_mutex_init(mtx, NULL) == 0)
#define mag_mutex_destroy(mtx) mag_assert2(pthread_mutex_destroy(mtx) == 0)
#define mag_mutex_lock(mtx) mag_assert2(pthread_mutex_lock(mtx) == 0)
#define mag_mutex_unlock(mtx) mag_assert2(pthread_mutex_unlock(mtx) == 0)

typedef pthread_cond_t mag_CondVar;
#define mag_cv_create(cv) mag_assert2(pthread_cond_init(cv, NULL) == 0)
#define mag_cv_destroy(cv) mag_assert2(pthread_cond_destroy(cv) == 0)
#define mag_cv_wait(cv, mtx) mag_assert2(pthread_cond_wait(cv, mtx) == 0)
#define mag_cv_signal(cv) mag_assert2(pthread_cond_signal(cv) == 0)
#define mag_cv_broadcast(cv) mag_assert2(pthread_cond_broadcast(cv) == 0)

#endif

extern void mag_thread_set_prio(mag_ThreadPrio prio); /* Set thread scheduling priority of current thread. */
extern void mag_thread_set_name(const char* _Nonnull name); /* Set thread name. */
extern void mag_thread_yield(void); /* Yield current thread. */

/* Dynamic zero-terminated string buffer. */
typedef struct mag_StrStream {
    char* _Nonnull buf;
    size_t len;
    size_t cap;
} mag_StrStream;

extern void mag_strstream_init(mag_StrStream* _Nonnull ss);
extern void mag_strstream_free(mag_StrStream* _Nonnull ss);
extern void mag_strstream_reserve_more(mag_StrStream* _Nonnull ss, size_t extra);
extern void mag_strstream_vappend(mag_StrStream* _Nonnull ss, const char* _Nonnull fmt, va_list ap);
extern void mag_strstream_append(mag_StrStream* _Nonnull ss, const char* _Nonnull fmt, ...);
extern void mag_strstream_append_strn(mag_StrStream* _Nonnull ss, const char* _Nonnull str, size_t len);
extern void mag_strstream_putc(mag_StrStream* _Nonnull ss, char c);
extern void mag_strstream_flush(mag_StrStream* _Nonnull ss, FILE* _Nonnull f);

/* Operation parameter */

/* Operation parameter type tag. */
typedef enum mag_OPParamType {
    MAG_OPP_NONE  = 0,
    MAG_OPP_E8M23 = 1,    /* fp32 */
    MAG_OPP_I64   = 2,    /* 64-bit signed integer. */
    MAG_OPP_U64   = 3,    /* 64-bit unsigned integer. */

    MAG_OPP__NUM
} mag_OPParamType;

extern const char* _Nonnull const mag_op_param_type_names[MAG_OPP__NUM]; /* Operation parameter type names. */

/*
** The opp (Operation Parameter) is used to pass additional data to the operation. For example:
**      The FILL init op uses the opp to pass the value to fill the tensor buffer with.
**      The permute op receives all permutation axes in the operation parameters.
** The opp is a tagged union (variant).
*/
typedef struct mag_OPParam {
    mag_OPParamType type;
    union { /* Overlapping values, one is active. */
        mag_E8M23 e8m23;
        int64_t i64;
        uint64_t u64;
    };
} mag_OPParam;

static MAG_AINLINE mag_OPParam mag_op_param_none() {
    mag_OPParam p;
    p.type = MAG_OPP_NONE;
    p.u64 = 0;
    return p;
}

static MAG_AINLINE mag_OPParam mag_op_param_wrap_e8m23(mag_E8M23 x) {
    uint32_t u32;
    memcpy(&u32, &x, sizeof(u32));
    mag_OPParam p;
    p.type = MAG_OPP_E8M23;
    p.u64 = u32;
    return p;
}

static MAG_AINLINE mag_OPParam mag_op_param_wrap_i64(int64_t x) {
    mag_OPParam p;
    p.type = MAG_OPP_I64;
    p.i64 = x;
    return p;
}

static MAG_AINLINE mag_OPParam mag_op_param_wrap_u64(uint64_t x) {
    mag_OPParam p;
    p.type = MAG_OPP_U64;
    p.u64 = x;
    return p;
}

/* Unpack value from packed opp. Panics if the type is not the expected type. */
static MAG_AINLINE mag_E8M23 mag_op_param_unpack_e8m23_or_panic(mag_OPParam pa) {
    mag_assert(pa.type == MAG_OPP_E8M23, "invalid op param type: %d", pa.type);
    mag_E8M23 e8m23 = 0.f;
    uint32_t u32 = (uint32_t)pa.u64;
    memcpy(&e8m23, &u32, sizeof(e8m23));
    return e8m23;
}

static MAG_AINLINE int64_t mag_op_param_unpack_i64_or_panic(mag_OPParam pa) {
    mag_assert(pa.type == MAG_OPP_I64, "invalid op param type: %d", pa.type);
    return pa.i64;
}

static MAG_AINLINE uint64_t mag_op_param_unpack_u64_or_panic(mag_OPParam pa) {
    mag_assert(pa.type == MAG_OPP_U64, "invalid op param type: %d", pa.type);
    return pa.u64;
}

static MAG_AINLINE mag_E8M23 mag_op_param_unpack_e8m23_or(mag_OPParam pa, mag_E8M23 fallback) {
    return pa.type == MAG_OPP_E8M23 ? mag_op_param_unpack_e8m23_or_panic(pa) : fallback;
}

static MAG_AINLINE int64_t mag_op_param_unpack_i64_or(mag_OPParam pa, int64_t fallback) {
    return pa.type == MAG_OPP_I64 ? mag_op_param_unpack_i64_or_panic(pa) : fallback;
}

static MAG_AINLINE uint64_t mag_op_param_unpack_u64_or(mag_OPParam pa, uint64_t fallback) {
    return pa.type == MAG_OPP_U64 ? mag_op_param_unpack_u64_or_panic(pa) : fallback;
}

/* Helper for filling the operation parameters array and validating the amount. */
typedef struct mag_OPParamLayout {
    mag_OPParam slots[MAG_MAX_OP_PARAMS];
    size_t count;
} mag_OPParamLayout;

static inline void mag_op_param_layout_init(mag_OPParamLayout* _Nonnull set) {
    set->count = 0;
    for (int i=0; i < MAG_MAX_OP_PARAMS; ++i)
        set->slots[i] = mag_op_param_none();
}

static inline size_t mag_op_param_layout_insert(mag_OPParamLayout* _Nonnull set, mag_OPParam param) {
    mag_assert(set->count < MAG_MAX_OP_PARAMS, "Too many operation parameters");
    set->slots[set->count] = param;
    return set->count++;
}

static inline void mag_op_param_layout_store(mag_OPParamLayout* _Nonnull set, size_t idx, mag_OPParam param) {
    mag_assert(idx < set->count, "Invalid operation parameter index");
    mag_assert(set->slots[idx].type == MAG_OPP_NONE, "Operation parameter already set");
    set->slots[idx] = param;
}

static inline void mag_op_param_layout_transfer(const mag_OPParamLayout* _Nonnull set, mag_OPParam (*_Nonnull out)[MAG_MAX_OP_PARAMS]) {
    memcpy(*out, set->slots, set->count*sizeof(*set->slots));
    for (size_t i=set->count; i < MAG_MAX_OP_PARAMS; ++i)
        (*out)[i] = mag_op_param_none();
}

/* Standard opcodes, not including initialization operators. */
typedef enum mag_Operator {
    /* Pseudo */
    MAG_OP_NOP,
    MAG_OP_CLONE,
    MAG_OP_VIEW,
    MAG_OP_TRANSPOSE,
    MAG_OP_PERMUTE,

    /* Reductions */
    MAG_OP_MEAN,
    MAG_OP_MIN,
    MAG_OP_MAX,
    MAG_OP_SUM,

    /* Unary */
    MAG_OP_ABS,
    MAG_OP_SGN,
    MAG_OP_NEG,
    MAG_OP_LOG,
    MAG_OP_SQR,
    MAG_OP_SQRT,
    MAG_OP_SIN,
    MAG_OP_COS,
    MAG_OP_STEP,
    MAG_OP_EXP,
    MAG_OP_FLOOR,
    MAG_OP_CEIL,
    MAG_OP_ROUND,
    MAG_OP_SOFTMAX,
    MAG_OP_SOFTMAX_DV,
    MAG_OP_SIGMOID,
    MAG_OP_SIGMOID_DV,
    MAG_OP_HARD_SIGMOID,
    MAG_OP_SILU,
    MAG_OP_SILU_DV,
    MAG_OP_TANH,
    MAG_OP_TANH_DV,
    MAG_OP_RELU,
    MAG_OP_RELU_DV,
    MAG_OP_GELU,
    MAG_OP_GELU_DV,

    /* Binary */
    MAG_OP_ADD,
    MAG_OP_SUB,
    MAG_OP_MUL,
    MAG_OP_DIV,
    MAG_OP_ADDS,
    MAG_OP_SUBS,
    MAG_OP_MULS,
    MAG_OP_DIVS,
    MAG_OP_POWS,
    MAG_OP_MATMUL,
    MAG_OP_REPEAT_BACK,
    MAG_OP_AND,
    MAG_OP_OR,
    MAG_OP_XOR,
    MAG_OP_NOT,
    MAG_OP_SHL,
    MAG_OP_SHR,

    MAG_OP_EQ,
    MAG_OP_NE,
    MAG_OP_LE,
    MAG_OP_GE,
    MAG_OP_LT,
    MAG_OP_GT,

    MAG_OP__NUM
} mag_Operator;
mag_static_assert(MAG_OP_NOP == 0);
mag_static_assert(MAG_OP_GT+1 == MAG_OP__NUM);
mag_static_assert(MAG_OP__NUM <= 0xff);

/* Initialization opcodes. */
typedef enum mag_InitOperator {
    MAG_IOP_NOP,
    MAG_IOP_BROADCAST,
    MAG_IOP_RAND_UNIFORM,
    MAG_IOP_RAND_NORMAL,
    MAG_IOP_RAND_BERNOULLI,
    MAG_IOP__NUM
} mag_InitOperator;

typedef enum mag_OPFlags {
    MAG_OP_FLAG_NONE = 0,
    MAG_OP_FLAG_SUPPORTS_INPLACE = 1<<0,                /* Allows to be executed inplace on the input tensor. */
    MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING = 1<<1,      /* Supports multithreading on CPU. */
} mag_OPFlags;

typedef struct mag_OPParamSlot {
    mag_OPParamType type;
    bool is_required;
} mag_OPParamSlot;

typedef struct mag_DTypeSlot {
    mag_DType type;
    bool is_used;
} mag_DTypeSlot;

/* Stores operator metadata such as operation type, number of inputs and parameters, and the types of the parameters. */
typedef struct mag_OPMetadata {
    const char* const _Nonnull mnemonic;                                    /* Operation mnemonic */
    const char* const _Nonnull desc;                                        /* Operation mnemonic */
    const uint8_t input_count;                                              /* Number of inputs */
    const mag_DTypeSlot input_dtypes[MAG_MAX_OP_INPUTS][MAG_DTYPE__NUM];    /* Input data types */
    const mag_OPParamSlot op_param_layout[MAG_MAX_OP_PARAMS];               /* Parameter types */
    const mag_OPFlags flags;                                                /* Operation flags */

    /* Backward pass function or NULL if op is not differentiable */
    void (*_Nullable const backward)(
        mag_Tensor* _Nonnull,
        mag_Tensor* _Nonnull* _Nonnull
    );

    /* Result allocator function or NULL. */
    mag_Tensor* _Nonnull (*_Nullable const r_alloc)(
        mag_Tensor* _Nonnull* _Nonnull,
        const mag_OPParam* _Nullable
    );

    /* Validator function or NULL. */
    bool (*_Nullable const validator)(
        mag_StrStream* _Nullable* _Nonnull,
        bool,
        mag_Tensor* _Nonnull,
        mag_Tensor* _Nonnull* _Nonnull,
        const mag_OPParam* _Nullable
    );

    struct {
        double thread_growth;
        int64_t thread_treshold;
    } cpu; /* CPU specific metadata. */
} mag_OPMetadata;

extern const mag_OPMetadata* _Nonnull mag_op_meta_of(mag_Operator opc); /* Get operation metadata for a specific opcode. */

/* Header for all objects that are reference counted. */
typedef struct mag_RCControlBlock {
    uint64_t rc;                            /* Strong reference count. Object is deallocated if this reaches zero. */
    void* _Nonnull self;                    /* Pointer to the self. */
    void (*_Nonnull dtor)(void* _Nonnull);  /* Destructor function (required). */
} mag_RCControlBlock;

/* Initialize reference count header for a new object. Self-reference and destructor functon must be provided. */
static MAG_AINLINE mag_RCControlBlock mag_rc_control_init(void* _Nonnull self, void (*_Nonnull dtor)(void* _Nonnull)) {
    mag_assert2(self && dtor); /* Self and destructor must be set. */
    mag_RCControlBlock control;
    control.rc = 1;
    control.self = self;
    control.dtor = dtor;
    return control;
}

static MAG_AINLINE void mag_rc_control_incref(mag_RCControlBlock* _Nonnull rcb) { /* Increment reference count (retain). */
    mag_assert(++rcb->rc < 0xffffffffu, "Reference count overflow");
}
static MAG_AINLINE bool mag_rc_control_decref(mag_RCControlBlock* _Nonnull rcb) { /* Decrement reference count (release). */
    mag_assert(rcb->rc, "Reference count underflow (double free)");
    if (--rcb->rc == 0) { /* Call destructor. */
        (*rcb->dtor)(rcb->self);
        return true; /* Object was destroyed. */
    }
    return false;
}

/* Memory chunk for intrusive memory pool. */
typedef struct mag_PoolChunk mag_PoolChunk;
struct mag_PoolChunk {
    uint8_t* _Nonnull bot;          /* Bottom (base) of chunk */
    uint8_t* _Nonnull top;          /* Top of chunk, grows downwards towards bottom */
    mag_PoolChunk* _Nonnull next;   /* Link to next chunk */
};

/* Fast memory allocator for memory blocks of same size. Obtains a memory pool and freelist for fast de/allocation. */
typedef struct mag_Pool {
    size_t block_size;                   /* Size of each allocated block */
    size_t block_align;                  /* Alignment requirements of each block. */
    size_t blocks_per_chunk;             /* How many blocks fit in each chunk */
    mag_PoolChunk* _Nonnull chunks;      /* Linked list of all chunks */
    mag_PoolChunk* _Nonnull chunk_head;  /* Last chunk */
    void* _Nonnull free_list;            /* Intrusive single linked list of free chunks */
    uint64_t num_freelist_hits;          /* Number of cache (free-list) hits */
    uint64_t num_pool_hits;              /* Number of cache (pool) hits */
    uint64_t num_chunks;                 /* Number of used chunks */
    uint64_t num_allocs;                 /* Number of total allocations */
} mag_Pool;

extern MAG_EXPORT void mag_fixed_intrusive_pool_init(mag_Pool* _Nonnull pool, size_t block_size, size_t block_align, size_t blocks_per_chunk);
extern MAG_EXPORT void* _Nonnull mag_fixed_intrusive_pool_malloc(mag_Pool* _Nonnull pool);
extern MAG_EXPORT void mag_fixed_intrusive_pool_free(mag_Pool* _Nonnull pool, void* _Nonnull blk);
extern MAG_EXPORT void mag_fixed_intrusive_pool_destroy(mag_Pool* _Nonnull pool);
extern MAG_EXPORT void mag_fixed_intrusive_pool_print_info(mag_Pool* _Nonnull pool, const char* _Nonnull name);

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
typedef struct mag_IComputeDevice mag_IComputeDevice;

typedef enum mag_TransferDir {
    MAG_TRANSFER_DIR_H2D,   /* Host to device (Host -> Device). */
    MAG_TRANSFER_DIR_D2H    /* Device to host (Device -> Host). */
} mag_TransferDir;

typedef enum mag_TransferOP {
    MAG_TRANSFER_OP_COPY,           /* Copy data bytewise to output. (memcpy) */
    MAG_TRANSFER_OP_CONVERT_E8M23   /* Convert data to f32 when writing to output. (cast) */
} mag_TransferOP;

/* Buffer interface on a compute device */
typedef struct mag_IStorageBuffer mag_IStorageBuffer;
struct mag_IStorageBuffer {
    mag_Context* _Nonnull ctx;
    mag_RCControlBlock rc_control;      /* Reference count control block. */
    uintptr_t base;                     /* Pointer to buffer on device. Might point to GPU or any other device memory. */
    size_t size;                        /* Size of buffer in bytes. */
    size_t alignment;                   /* Alignment of buffer. */
    size_t granularity;                 /* Element size granularity. */
    mag_DType dtype;                    /* Data type of buffer. */
    mag_IComputeDevice* _Nonnull host;  /* Host device. */

    /* Broadcast (fill) buffer with x. */
    void (*_Nonnull broadcast)(
        mag_IStorageBuffer* _Nonnull sto,
        size_t offs,
        const void* _Nonnull src,
        size_t stride
    );

    /* Transfer data between host and device. */
    void (*_Nonnull transfer)(
        mag_IStorageBuffer* _Nonnull sto,
        mag_TransferDir dir,
        mag_TransferOP op,
        size_t offs,
        void* _Nonnull inout,        /* Source or destination buffer. Must point to nb bytes. */
        size_t inout_size   /* Size of input/output buffer. */
    );
};

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
struct mag_IComputeDevice {
    mag_Context* _Nonnull ctx;
    char name[128];                                                                                             /* Device name. */
    void* _Nonnull impl;                                                                                        /* Device specific implementation, if applicable. */
    bool is_async;                                                                                              /* If device is async. */
    mag_ComputeDeviceType type;                                                                                 /* Device type enum. */
    void (*_Nonnull eager_exec_init)(mag_IComputeDevice* _Nonnull dvc, mag_Tensor* _Nonnull root);                                         /* Execute a single init op. */
    void (*_Nonnull eager_exec_fwd)(mag_IComputeDevice* _Nonnull dvc, mag_Tensor* _Nonnull root);                                          /* Execute a single op forward. */
    void (*_Nonnull alloc_storage)(mag_IComputeDevice* _Nonnull dvc, mag_IStorageBuffer* _Nonnull* _Nonnull out, size_t size, mag_DType dtype);   /* Allocate storage buffer in device memory */
};

/* Device creation and destruction. */
typedef struct mag_IDeviceFactory {
    mag_IComputeDevice* _Nonnull (*_Nonnull init)(mag_Context* _Nonnull ctx, const mag_ComputeDeviceDesc* _Nonnull desc);      /* Initialize device. */
    void (*_Nonnull destroy)(mag_IComputeDevice* _Nonnull dvc);         /* Destroy device. */
} mag_IDeviceFactory;

/* Global device factories. Implemented in magnetron_device_registry.c */
extern mag_IComputeDevice* _Nonnull mag_init_dynamic_device(mag_Context* _Nonnull ctx, const mag_ComputeDeviceDesc* _Nonnull desc);
extern void mag_destroy_dynamic_device(mag_IComputeDevice* _Nonnull dvc);

#if defined(__x86_64__) || defined(_M_X64) /* x86_64 or AMD64 specific CPU features. */
#define mag_x86_64_feature_def(_, __) /* Enumerator | CPUDID Leaf | Register | Shift */\
    _(NONE                 , 0,         EAX,   0)__\
    _(AVX                  , H1,        ECX,  28)__\
    _(AVX2                 , H7,        EBX,   5)__\
    _(AVXVNNI              , H7_1H,     EAX,   4)__\
    _(AVXVNNIINT8          , H7_1H,     EDX,   4)__\
    _(AVXVNNIINT16         , H7_1H,     EDX,  10)__\
    _(AVX512BW             , H7,        EBX,  30)__\
    _(AVX512CD             , H7,        EBX,  28)__\
    _(AVX512DQ             , H7,        EBX,  17)__\
    _(AVX512ER             , H7,        EBX,  27)__\
    _(AVX512F              , H7,        EBX,  16)__\
    _(AVX512IFMA           , H7,        EBX,  21)__\
    _(AVX512PF             , H7,        EBX,  26)__\
    _(AVX512VBMI           , H7,        ECX,   1)__\
    _(AVX512VL             , H7,        EBX,  31)__\
    _(AVX512_4FMAPS        , H7,        EDX,   3)__\
    _(AVX512_4VNNIW        , H7,        EDX,   2)__\
    _(AVX512_FP16          , H7,        EDX,  23)__\
    _(AVX512_BF16          , H7_1H,     EAX,   5)__\
    _(AVX512_BITALG        , H7,        ECX,  12)__\
    _(AVX512_VBMI2         , H7,        ECX,   6)__\
    _(AVX512_VNNI          , H7,        ECX,  11)__\
    _(AVX512_VP2INTERSECT  , H7,        EDX,   8)__\
    _(AVX512_VPOPCNTDQ     , H7,        ECX,  14)__\
    _(BMI                  , H7,        EBX,   3)__\
    _(BMI2                 , H7,        EBX,   8)__\
    _(F16C                 , H1,        ECX,  29)__\
    _(FMA                  , H1,        ECX,  12)__\
    _(FPU                  , H1,        EDX,   0)__\
    _(GFNI                 , H7,        ECX,   8)__\
    _(IA64                 , H1,        EDX,  30)__\
    _(MMX                  , H1,        EDX,  23)__\
    _(OSXSAVE              , H1,        ECX,  27)__\
    _(PCLMUL               , H1,        ECX,   1)__\
    _(RDRND                , H1,        ECX,  30)__\
    _(RDSEED               , H7,        EBX,  18)__\
    _(RDTSCP               , H80000001, EDX,  27)__\
    _(SHA                  , H7,        EBX,  29)__\
    _(SSE                  , H1,        EDX,  25)__\
    _(SSE2                 , H1,        EDX,  26)__\
    _(SSE3                 , H1,        ECX,   0)__\
    _(SSE4_1               , H1,        ECX,  19)__\
    _(SSE4_2               , H1,        ECX,  20)__\
    _(SSSE3                , H1,        ECX,   9)__\
    _(VAES                 , H7,        ECX,   9)__\
    _(VME                  , H1,        EDX,   1)__\
    _(VMX                  , H1,        ECX,   5)__\
    _(VPCLMULQDQ           , H7,        ECX,  10)__\
    _(XSAVE                , H1,        ECX,  26)__\
    _(HYBRID_CPU           , H7,        EDX,  15)__

#define _(ident, leaf, reg, bit) MAG_AMD64_CAP_##ident
typedef enum mag_amd64_cap_t {
    mag_x86_64_feature_def(_, MAG_SEP)
    MAG_AMD64_CAP__NUM
} mag_amd64_cap_t;
#undef _

extern const char* _Nullable const mag_amd64_cap_names[MAG_AMD64_CAP__NUM];

#elif defined(__aarch64__) || defined(_M_ARM64) /* ARM 64 specific CPU features. */

#define mag_arm64_feature_def(_, __) /* Enumerator */\
    _(NONE)__\
    _(NEON)__\
    _(DOTPROD)__\
    _(I8MM)__\
    _(F16SCA)__\
    _(F16VEC)__\
    _(BF16)__\
    _(SVE)__\
    _(SVE2)__

#define _(ident) MAG_ARM64_CAP_##ident
typedef enum mag_ARM64CPUCaps {
    mag_arm64_feature_def(_, MAG_SEP)
    MAG_ARM64_CAP__NUM
} mag_ARM64CPUCaps;
#undef _
extern const char* _Nonnull const mag_arm64_cpu_caps_names[MAG_ARM64_CAP__NUM];

#endif

/* Context specific flags. */
typedef enum mag_ContextFlags {
    MAG_CTX_FLAG_NONE = 0,
    MAG_CTX_FLAG_GRAD_RECORDER = 1<<0,     /* Gradient recording is currently active. */
} mag_ContextFlags;

struct mag_Context {
    struct {
        char os_name[128];                        /* OS name. */
        char cpu_name[128];                       /* CPU name. */
        uint32_t cpu_virtual_cores;               /* Virtual CPUs. */
        uint32_t cpu_physical_cores;              /* Physical CPU cores. */
        uint32_t cpu_sockets;                     /* CPU sockets. */
        uint64_t phys_mem_total;                  /* Total physical memory in bytes. */
        uint64_t phys_mem_free;                   /* Free physical memory in bytes. */
#if defined(__x86_64__) || defined(_M_X64)
        uint64_t amd64_cpu_caps;                  /* x86-64 CPU features. Bitset of 1ull<<MAG_AMD64_CAP_* */
        bool is_amd;                              /* Is AMD CPU? */
#elif defined (__aarch64__) || defined(_M_ARM64)
        uint64_t arm64_cpu_caps;                  /* ARM64 CPU features. */
        int64_t arm64_cpu_sve_width;              /* ARM64 SVE vector register width. */
#endif
    } machine;
    size_t num_tensors;                         /* Total tensor instances allocated. */
    size_t num_storages;                        /* Total storage buffers allocated. */
    mag_Pool tensor_pool;                       /* Tensor struct memory pool. */
    mag_Pool storage_pool;                      /* Storage struct memory pool. */
    mag_ContextFlags flags;                     /* Context flags. */
    mag_PRNGAlgo prng_algo;                     /* Active PRNG algorithm. */
    uintptr_t tr_id;                            /* Context thread ID. */
    size_t sh_len;                              /* Number of shutdown hooks. */
    size_t sh_cap;                              /* Maximum number of shutdown hooks. */
    mag_ComputeDeviceType device_type;          /* Active compute device type. */
    mag_IComputeDevice* _Nonnull device;        /* Active compute device interface. */
    void* _Nullable ud;                         /* User data. */
#ifdef MAG_DEBUG
    mag_Tensor* _Nullable alive_head;           /* List of alive tensors used for leak detection. */
#endif
};

/* Tensor specific flags. */
typedef enum mag_TensorFlags {
    MAG_TFLAG_NONE = 0,
    MAG_TFLAG_IS_VIEW = 1<<0,           /* Tensor is a view. */
    MAG_TFLAG_IS_GRAD = 1<<1,           /* Tensor is a gradient. */
    MAG_TFLAG_REQUIRES_GRAD = 1<<2,     /* Tensor requires gradient. */

    MAG_TFLAG_LEN = 4                   /* Number of flags. */
} mag_TensorFlags;
mag_static_assert(MAG_TFLAG_LEN <= 0xff);

/*
** Reference counted tensor header. Stores shape, strides, gradient and other metadata.
** The actual data buffer is compute-device specific and can be only accessed via the storage buffer.
** A tensor can be a view, which references the storage buffer of another tensor, but views have their own header too.
*/
struct mag_Tensor {
    mag_Context* _Nonnull  ctx;                             /* Host context. */
    mag_RCControlBlock rc_control;                          /* Reference counting control block. */
    int64_t rank;                                           /* Number of active dimensions. [1, MAX_DIMS] */
    int64_t shape[MAG_MAX_DIMS];                            /* Shape of the tensor. */
    int64_t strides[MAG_MAX_DIMS];                          /* Strides of the tensor. We store the strides in element counts and NOT in bytes. */
    mag_DType dtype;                                        /* Data type of the tensor. */
    mag_IStorageBuffer* _Nonnull storage;                   /* Storage buffer. */
    int64_t numel;                                          /* Number of elements in the tensor. */
    mag_TensorFlags flags;                                  /* Tensor flags. */
    mag_Operator op;                                        /* Opcode for operators. */
    mag_Tensor* _Nullable op_inputs[MAG_MAX_OP_INPUTS];     /* Input tensors for operators. */
    mag_OPParam op_params[MAG_MAX_OP_PARAMS];               /* Operator parameters. */
    mag_InitOperator init_op;                               /* Initialization op */
    mag_OPParam init_op_params[MAG_MAX_OP_PARAMS];          /* Init operator parameters */
    mag_Tensor* _Nullable view_uplink;                      /* View base tensor. */
    size_t view_offs;                                       /* Offset in view tensor. */
    mag_Tensor* _Nullable grad;                             /* ∇f - Gradient tensor. */
    uint8_t name[MAG_MAX_TENSOR_NAME_LEN];                  /* Tensor debug name. */
    void* _Nullable ud;                                     /* User data. */
#ifdef MAG_DEBUG
    mag_Tensor* _Nullable alive_next;                       /* Next alive tensor used for leak detection. */
#endif
};

/* PRNG state for random number generation. */
typedef struct mag_PRNGState {
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
    mag_PRNGAlgo algo; /* PRNG algorithm. */
} mag_PRNGState;

/* Initialize and seed PRNG with specific algorithm. */
void mag_prng_seed(mag_PRNGState* _Nonnull prng, mag_PRNGAlgo algo, uint64_t seed);

/* CPU Compute kernel payload passed to each CPU thread. */
typedef struct mag_CPUKernelPayload {
    int64_t thread_num;                     /* Total number of threads involved. */
    int64_t thread_idx;                     /* Current thread index used to compute thread-local partition. */
    mag_Tensor* _Nullable node;             /* Result tensor. Stores input tensors and all other op-specific data. */
    mag_ExecStage stage;                    /* Graph evaluation type. */
    mag_PRNGState* _Nonnull local_prng;     /* Thread-local CPU PRNG state. */
} mag_CPUKernelPayload;

/*
** Stores function-pointer lookup table for all compute kernels.
** The lookup table is used to dispatch the correct kernel for each operation by indexing with the opcode.
** The CPU runtime dynamically fills these arrays with the best fitting kernel depending on the detected CPU.
** See magnetron_cpu.c for details.
*/
typedef struct mag_CPUKernelRegistry {
    void (*_Nonnull init[MAG_IOP__NUM][MAG_DTYPE__NUM])(const mag_CPUKernelPayload* _Nonnull);      /* Initialization operator kernels. */
    void (*_Nonnull eval[MAG_OP__NUM][MAG_DTYPE__NUM])(const mag_CPUKernelPayload* _Nonnull);       /* Eval operator kernels. */
    void (*_Nonnull vector_cast)(size_t nb, const void* _Nonnull src, mag_DType src_t, void* _Nonnull dst, mag_DType dst_t); /* Vector cast (dtype conversion) kernel. */
} mag_CPUKernelRegistry;

/* Combine two hash values. */
static inline void mag_hash_combine(uint32_t* _Nonnull seed, uint32_t value) {
    *seed ^= value + 0x9e3779b9 + (*seed<<6) + (*seed>>2);
}

extern uint64_t mag_hash(const void* _Nonnull key, size_t len, uint32_t seed); /* Compute murmur3_64 hash */
extern uint32_t mag_crc32c(const void* _Nonnull buffer, size_t size); /* Compute CRC32 checksum with CRC32c polynomial. */

#define MAG_DEF_MAP_GROW_FACTOR 0.6 /* 60% - Default grow factor. */
#define MAG_DEF_MAP_SHRINK_FACTOR 0.1 /* 10% - Default shrink factor. */
#define MAG_DEF_MAP_LOAD_FACTOR MAG_DEF_MAP_GROW_FACTOR /* 60% - Default load factor. */

/*
** Generic hashmap for storing key-value pairs.
** Uses open addressing with robin-hood hashing.
*/
typedef struct mag_HashMap mag_HashMap;

/* Create a new hashmap. */
extern mag_HashMap* _Nonnull mag_hashmap_create(
    size_t elsize,
    size_t cap,
    uint32_t seed,
    uint64_t (*_Nonnull hash)(const void* _Nonnull item, uint32_t seed),
    bool (*_Nonnull cmp)(const void* _Nonnull a, const void* _Nonnull b, void* _Nullable ud),
    void (*_Nullable elfree)(void* _Nonnull item),
    void* _Nullable ud,
    double grow_fac,
    double shrink_fac,
    double load_fac
);

extern void mag_hashmap_destroy(mag_HashMap* _Nonnull map);
extern void mag_hashmap_clear(mag_HashMap* _Nonnull map, bool update_cap);
extern size_t mag_hashmap_count(mag_HashMap* _Nonnull map);
extern bool mag_hashmap_is_oom(mag_HashMap* _Nonnull map);
extern const void* _Nullable mag_hashmap_lookup(mag_HashMap* _Nonnull map, const void* _Nonnull item);
extern const void* _Nullable mag_hashmap_insert(mag_HashMap* _Nonnull map, const void* _Nonnull item);
extern const void* _Nullable mag_hashmap_delete(mag_HashMap* _Nonnull map, const void* _Nonnull item);
extern const void* _Nullable mag_hashmap_probe(mag_HashMap* _Nonnull map, uint64_t position);
extern bool mag_hashmap_scan(mag_HashMap* _Nonnull map, bool (*_Nonnull iter)(const void* _Nonnull item, void* _Nullable ud), void* _Nullable ud);
extern bool mag_hashmap_iter(mag_HashMap* _Nonnull map, size_t* _Nonnull i, void* _Nonnull* _Nonnull item);
extern const void* _Nullable mag_hashmap_get_with_hash(mag_HashMap* _Nonnull map, const void* _Nonnull key, uint64_t hash);
extern const void* _Nullable mag_hashmap_delete_with_hash(mag_HashMap* _Nonnull map, const void* _Nonnull key, uint64_t hash);
extern const void* _Nullable mag_hashmap_set_with_hash(mag_HashMap* _Nonnull map, const void* _Nonnull item, uint64_t hash);
extern void mag_hashmap_set_grow_by_power(mag_HashMap* _Nonnull map, size_t pow);
extern void mag_hashmap_set_load_factor(mag_HashMap* _Nonnull map, double load_factor);

#ifdef __cplusplus
}
#endif

#endif
