/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef MAGNETRON_INTERNAL_H
#define MAGNETRON_INTERNAL_H

#include "magnetron.h"

#include <string.h>
#include <stdio.h>

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

#define MAG_GELU_COEFF 0.044715f
#define MAG_GRA_FWD MAG_GRAPH_EVAL_ORDER_FORWARD
#define MAG_GRA_BWD MAG_GRAPH_EVAL_ORDER_REVERSE
#define MAG_GRA_LEN 2
#define MAG_MAX_CPUS 8192
#define MAG_MAX_NUMA_NODES 64
#define MAG_STORAGE_EXT ".magnetron"

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
#define mag_likely(x) __builtin_expect(!!(x), 1)
#define mag_unlikely(x) __builtin_expect(!!(x), 0)
#define mag_ffs(x) ((uint32_t)__builtin_ctz(x))
#define mag_fls(x) ((uint32_t)(__builtin_clz(x)^31))
#define mag_ffs64(x) ((uint32_t)__builtin_ctzll(x))
#define mag_fls64(x) ((uint32_t)(__builtin_clzll(x)^63))

typedef int64_t mag_atomic_t;       /* Atomic integer type */
typedef enum mag_mo_t {             /* Atomic memory order */
    MAG_MO_RELAXED = __ATOMIC_RELAXED,
    MAG_MO_CONSUME = __ATOMIC_CONSUME,
    MAG_MO_ACQUIRE = __ATOMIC_ACQUIRE,
    MAG_MO_RELEASE = __ATOMIC_RELEASE,
    MAG_MO_ACQ_REL = __ATOMIC_ACQ_REL,
    MAG_MO_SEQ_CST = __ATOMIC_SEQ_CST
} mag_mo_t;

static MAG_AINLINE void mag_atomic_store(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    __atomic_store_n(o, x, order);
}
static MAG_AINLINE mag_atomic_t mag_atomic_load(volatile mag_atomic_t* o, mag_mo_t order) {
    return __atomic_load_n(o, order);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_add(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    return __atomic_fetch_add(o, x, order);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_sub(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    return __atomic_fetch_sub(o, x, order);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_and(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    return __atomic_fetch_and(o, x, order);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_or(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    return __atomic_fetch_or(o, x, order);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_xor(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    return __atomic_fetch_xor(o, x, order);
}
static MAG_AINLINE mag_atomic_t mag_atomic_exchange(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    return __atomic_exchange_n(o, x, order);
}
static MAG_AINLINE bool mag_atomic_compare_exchange_weak(volatile mag_atomic_t* o, mag_atomic_t *exp, mag_atomic_t *des, mag_mo_t order_succ, mag_mo_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, true, order_succ, order_fail);
}
static MAG_AINLINE bool mag_atomic_compare_exchange_strong(volatile mag_atomic_t* o, mag_atomic_t *exp, mag_atomic_t *des, mag_mo_t order_succ, mag_mo_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, false, order_succ, order_fail);
}

#else

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
#define mag_likely(x) (x)
#define mag_unlikely(x) (x)
static MAG_AINLINE uint32_t mag_ffs(const uint32_t x) {
    unsigned long r; _BitScanForward(&r, x); return (uint32_t)r;
}
static MAG_AINLINE uint32_t mag_fls(const uint32_t x) {
    unsigned long r; _BitScanReverse(&r, x); return (uint32_t)r;
}
static MAG_AINLINE uint32_t mag_ffs64(const uint64_t x) {
  unsigned long r; _BitScanForward64(&r, x); return (uint32_t)r;
}
static MAG_AINLINE uint32_t mag_fls64(const uint64_t x) {
  unsigned long r; _BitScanReverse64(&r, x); return (uint32_t)r;
}

typedef __int64 mag_atomic_t;       /* Atomic integer type */
typedef enum mag_mo_t {             /* Atomic memory order */
    MAG_MO_RELAXED,
    MAG_MO_CONSUME,
    MAG_MO_ACQUIRE,
    MAG_MO_RELEASE,
    MAG_MO_ACQ_REL,
    MAG_MO_SEQ_CST
} mag_mo_t;

static MAG_AINLINE void mag_atomic_store(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    (void)order; _InterlockedExchange64(o, x);
}
static MAG_AINLINE mag_atomic_t mag_atomic_load(volatile mag_atomic_t* o, mag_mo_t order) {
    (void)order;
    mag_atomic_t r;
    _InterlockedExchange64(&r, *o);
    return r;
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_add(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    (void)order;
    return _InterlockedExchangeAdd64(o, x);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_sub(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    (void)order;
    return _InterlockedExchangeAdd64(o, -x);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_and(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    (void)order;
    return _InterlockedAnd64(o, x);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_or(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    (void)order;
    return _InterlockedOr64(o, x);
}
static MAG_AINLINE mag_atomic_t mag_atomic_fetch_xor(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    (void)order;
    return _InterlockedXor64(o, x);
}
static MAG_AINLINE mag_atomic_t mag_atomic_exchange(volatile mag_atomic_t* o, mag_atomic_t x, mag_mo_t order) {
    (void)order;
    return _InterlockedExchange64(o, x);
}
static MAG_AINLINE bool mag_atomic_compare_exchange_weak(volatile mag_atomic_t* o, mag_atomic_t *exp, mag_atomic_t *des, mag_mo_t order_succ, mag_mo_t order_fail) {
    (void)order_succ; (void)order_fail;
    mag_atomic_t old = _InterlockedCompareExchange64(o, *des, *exp);
    if (old == *exp) return true;
    else { *exp = old; return false; }
}
static MAG_AINLINE bool mag_atomic_compare_exchange_strong(volatile mag_atomic_t* o, mag_atomic_t *exp, mag_atomic_t *des, mag_mo_t order_succ, mag_mo_t order_fail) {
    (void)order_succ; (void)order_fail;
    mag_atomic_t old = _InterlockedCompareExchange64(o, *des, *exp);
    if (old == *exp) return true;
    else { *exp = old; return false; }
}

#endif

mag_static_assert(sizeof(0u) == 4);
mag_static_assert(sizeof(0ull) == 8);

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
/* Hardware destructive interference size. */
#define MAG_HDI hardware_destructive_interference_size
#else
/* Hardware destructive interference size. */
#define MAG_HDI 64
#endif

static uint32_t MAG_AINLINE mag_bswap32(uint32_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    #ifdef MAG_BE
        #if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
            x = (uint32_t)__builtin_bswap32((int32_t)x);
        #else
            x = (x & 0xff000000) >> 24 |
            (x & 0xff0000) >> 8 |
            (x & 0xff00) << 8 |
            (x & 0xff) << 24;
        #endif
    #endif
    return x;
}

static uint64_t MAG_AINLINE mag_bswap64(uint64_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
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

extern MAG_NORET MAG_COLDPROC MAG_EXPORT void mag_panic(const char* msg, ...);
extern MAG_EXPORT bool mag_log_enabled;
extern MAG_EXPORT void* (*mag_alloc)(void* blk, size_t size);
extern MAG_EXPORT void* mag_alloc_aligned(size_t size, size_t align);
extern MAG_EXPORT void mag_free_aligned(void* blk);
extern MAG_EXPORT void mag_humanize_memory_size(size_t n, double* out, const char** unit);
extern MAG_EXPORT uintptr_t mag_thread_id(void);

#define mag_swap(T, a, b) do { T tmp = (a); (a) = (b); (b) = tmp; } while (0)
#define mag_xmax(x, y) (((x) > (y)) ? (x) : (y))
#define mag_xmin(x, y) (((x) < (y)) ? (x) : (y))
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

#define mag_assert(expr, msg, ...) \
    if (mag_unlikely(!(expr))) { \
        mag_panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }
#define mag_assert2(expr) mag_assert(expr, "")

#ifdef MAG_DEBUG
#define mag_bnd_chk(ptr, base, n) \
    mag_assert((uintptr_t)(ptr) >= (uintptr_t)(base) && (uintptr_t)(ptr) < (uintptr_t)(base) + (n), \
        "\nBound check failed: %p not in [%p, %p), base+0x%x, end+0x%x", \
        (void*)(ptr), \
        (void*)(base), \
        (void*)((uintptr_t)(base)+(n)), \
        (int)llabs((long long)((int64_t)(ptr)-(int64_t)(base))), \
        (int)llabs((long long)(((int64_t)(base)+(n))-(int64_t)(ptr))) \
    )
#else
#define mag_bnd_chk(ptr, base, n)
#endif

/* Increment pointer or size with correct type alignment. */
static MAG_AINLINE void* mag_pincr(void** p, size_t sz, size_t align) {
    void* pp = (void*)(((uintptr_t)*p+align-1)&-align);
    *p = (void*)((uint8_t*)pp+sz);
    return pp;
}

#ifdef _WIN32

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

typedef CONDITION_VARIABLE mag_cond_var_t;
#define mag_cv_create(cv) InitializeConditionVariable(cv)
#define mag_cv_destroy(cv)
#define mag_cv_wait(cv, mtx) SleepConditionVariableSRW(cv, mtx, INFINITE, 0)
#define mag_cv_signal(cv) WakeConditionVariable(cv)
#define mag_cv_broadcast(cv) WakeAllConditionVariable(cv)

#else

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

typedef pthread_cond_t mag_cond_var_t;
#define mag_cv_create(cv) mag_assert2(pthread_cond_init(cv, NULL) == 0)
#define mag_cv_destroy(cv) mag_assert2(pthread_cond_destroy(cv) == 0)
#define mag_cv_wait(cv, mtx) mag_assert2(pthread_cond_wait(cv, mtx) == 0)
#define mag_cv_signal(cv) mag_assert2(pthread_cond_signal(cv) == 0)
#define mag_cv_broadcast(cv) mag_assert2(pthread_cond_broadcast(cv) == 0)

#endif

typedef enum mag_thread_sched_prio_t {  /* Thread scheduling priority for CPU compute */
    MAG_THREAD_SCHED_PRIO_NORMAL = 0,   /* Normal thread priority */
    MAG_THREAD_SCHED_PRIO_MEDIUM = 1,   /* Medium thread priority */
    MAG_THREAD_SCHED_PRIO_HIGH = 2,     /* High thread priority */
    MAG_THREAD_SCHED_PRIO_REALTIME = 3, /* Real-time thread priority */
} mag_thread_sched_prio_t;

extern MAG_EXPORT void mag_thread_set_prio(mag_thread_sched_prio_t prio); /* Set thread scheduling priority of current thread. */
extern MAG_EXPORT void mag_thread_set_name(const char* name); /* Set thread name. */
extern MAG_EXPORT void mag_thread_yield(void); /* Yield current thread. */

typedef enum mag_op_t {
    MAG_OP_NOP,
    MAG_OP_CLONE,
    MAG_OP_VIEW,
    MAG_OP_TRANSPOSE,
    MAG_OP_PERMUTE,
    MAG_OP_MEAN,
    MAG_OP_MIN,
    MAG_OP_MAX,
    MAG_OP_SUM,
    MAG_OP_ABS,
    MAG_OP_NEG,
    MAG_OP_LOG,
    MAG_OP_SQR,
    MAG_OP_SQRT,
    MAG_OP_SIN,
    MAG_OP_COS,
    MAG_OP_STEP,
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
    MAG_OP_ADD,
    MAG_OP_SUB,
    MAG_OP_MUL,
    MAG_OP_DIV,
    MAG_OP_ADDS,
    MAG_OP_SUBS,
    MAG_OP_MULS,
    MAG_OP_DIVS,
    MAG_OP_MATMUL,
    MAG_OP__NUM
} mag_op_t;
mag_static_assert(MAG_OP_NOP == 0);
mag_static_assert(MAG_OP_MATMUL+1 == MAG_OP__NUM);
mag_static_assert(MAG_OP__NUM <= 0xff);

typedef enum mag_init_op {
    MAG_INIT_OP_NOP,
    MAG_INIT_OP_ZERO,
    MAG_INIT_OP_FILL,
    MAG_INIT_OP_RAND_UNIFORM,
    MAG_INIT_OP_RAND_NORMAL,

    MAG_INIT_OP__NUM
} mag_init_op;

typedef enum mag_op_param_type_t {
    MAG_OP_TPARAM_NONE  = 0,
    MAG_OP_TPARAM_F32   = 1,
    MAG_OP_TPARAM_I32   = 2,
    MAG_OP_TPARAM_U32   = 3,

    MAG_OP_TPARAM__NUM
} mag_op_param_type_t;

typedef struct mag_op_param_t {
    mag_op_param_type_t type : 8; /* Parameter type */
    union {
        float f32;
        int32_t i32;
        uint32_t u32;
    } x;
} mag_op_param_t;

typedef struct mag_op_meta_t {
    const char* mnemonic;                                   /* Operation mnemonic */
    uint8_t argcount;                                       /* Number of arguments */
    uint8_t paramcount;                                     /* Number of parameters */
    mag_op_param_type_t param_types[MAG_MAX_OP_PARAMS];     /* Parameter types */
    bool inplace;                                           /* Supports inplace execution */
    mag_tensor_t* (*r_alloc)(mag_tensor_t**, const mag_op_param_t*);
    bool (*validator)(mag_op_t, mag_tensor_t*, mag_tensor_t**, const mag_op_param_t*);
} mag_op_meta_t;
extern MAG_EXPORT const mag_op_meta_t* mag_op_meta_of(mag_op_t type);

typedef struct mag_intrusive_chunk mag_intrusive_chunk;
struct mag_intrusive_chunk {
    uint8_t* bot;                       /* Bottom (base) of chunk */
    uint8_t* top;                       /* Top of chunk, grows downwards towards bottom */
    mag_intrusive_chunk* next;          /* Link to next chunk */
};

/* Fast memory allocator for memory blocks of same size. Obtains a memory pool and freelist for fast de/allocation. */
typedef struct mag_fixed_intrusive_pool {
    size_t block_size;                  /* Size of each allocated block */
    size_t block_align;                 /* Alignment requirements of each block. */
    size_t blocks_per_chunk;            /* How many blocks fit in each chunk */
    mag_intrusive_chunk* chunks;        /* Linked list of all chunks */
    mag_intrusive_chunk* chunk_head;    /* Last chunk */
    void* free_list;                    /* Intrusive single linked list of free chunks */
    uint64_t num_freelist_hits;         /* Number of cache (free-list) hits */
    uint64_t num_pool_hits;             /* Number of cache (pool) hits */
    uint64_t num_chunks;                /* Number of used chunks */
    uint64_t num_allocs;                /* Number of total allocations */
} mag_fixed_intrusive_pool;

extern MAG_EXPORT void mag_fixed_intrusive_pool_init(mag_fixed_intrusive_pool* pool, size_t block_size, size_t block_align, size_t blocks_per_chunk);
extern MAG_EXPORT void* mag_fixed_intrusive_pool_malloc(mag_fixed_intrusive_pool* pool);
extern MAG_EXPORT void mag_fixed_intrusive_pool_free(mag_fixed_intrusive_pool* pool, void* blk);
extern MAG_EXPORT void mag_fixed_intrusive_pool_destroy(mag_fixed_intrusive_pool* pool);
extern MAG_EXPORT void mag_fixed_intrusive_pool_print_info(mag_fixed_intrusive_pool* pool, const char* name);

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
typedef struct mag_compute_device_t mag_compute_device_t;

/* Buffer interface on a compute device */
typedef struct mag_storage_buffer_t mag_storage_buffer_t;
struct mag_storage_buffer_t {
    uintptr_t base;                                                                                 /* Pointer to buffer on device. Might point to GPU or any other device memory. */
    size_t size;                                                                                    /* Size of buffer in bytes. */
    size_t alignment;                                                                               /* Alignment of buffer. */
    mag_compute_device_t* host;                                                                     /* Host device. */
    void (*set)(mag_storage_buffer_t* sto, size_t offs, uint8_t x);                                 /* Memset buffer. */
    void (*cpy_host_device)(mag_storage_buffer_t* sto, size_t offs, const void* src, size_t n);     /* Copy data from host to device. */
    void (*cpy_device_host)(mag_storage_buffer_t* sto, size_t offs, void* dst, size_t n);           /* Copy data from device to host. */
};

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
struct mag_compute_device_t {
    char name[128];                                                             /* Device name. */
    void* impl;                                                                 /* Device specific implementation, if applicable. */
    bool is_async;                                                              /* If device is async. */
    mag_compute_device_type_t type;                                             /* Device type enum. */
    void (*eager_exec_fwd)(mag_compute_device_t* dvc, mag_tensor_t* root);      /* Execute a single op forward. */
    void (*eager_exec_bwd)(mag_compute_device_t* dvc, mag_tensor_t* root);      /* Execute a single op backwards. */
    void (*alloc_storage)(mag_compute_device_t* dvc, mag_storage_buffer_t* out, size_t size);
    void (*free_storage)(mag_compute_device_t* dvc, mag_storage_buffer_t* buf);
};

/* Device creation and destruction. */
typedef struct mag_device_factory_t {
    mag_compute_device_t* (*init)(mag_ctx_t* ctx, const mag_device_descriptor_t* desc);      /* Initialize device. */
    void (*destroy)(mag_compute_device_t* dvc);         /* Destroy device. */
} mag_device_factory_t;

/* Global device factories. Implemented in magnetron_device_registry.c */
extern mag_compute_device_t* mag_init_dynamic_device(mag_ctx_t* ctx, const mag_device_descriptor_t* desc);
extern void mag_destroy_dynamic_device(mag_compute_device_t* dvc);

/* Profiling performance monitor per op. */
typedef struct mag_perf_mon_t {
    uint64_t elapsed_ns;
    uint64_t elapsed_ns_acc;
    uint64_t n_execs;
} mag_perf_mon_t;

/* Performance monitor for profiler session. */
typedef struct mag_op_perf_info_t {
    uint64_t elapsed_ns_acc;
    uint64_t n_execs;
} mag_op_perf_info_t;

#ifdef MAG_DEBUG
typedef struct mag_tensor_node_t mag_tensor_node_t;
struct mag_tensor_node_t {
    mag_tensor_t* tensor;
    mag_tensor_node_t* next;
};
#endif


/*
** Context contains all isolated state and data.
** Lifetimes of tensors and compute graphs are bound to the context - the context is the owner.
** Context itself is not thread-safe, use a thread-local context or synchronize access. (Multiple contexts can be used.)
*/
struct mag_ctx_t {
    struct {
        char os_name[128];                          /* OS name. */
        char cpu_name[128];                         /* CPU name. */
        uint32_t cpu_virtual_cores;                 /* Virtual CPUs. */
        uint32_t cpu_physical_cores;                /* Physical CPU cores. */
        uint32_t cpu_sockets;                       /* CPU sockets. */
        uint64_t phys_mem_total;                    /* Total physical memory in bytes. */
        uint64_t phys_mem_free;                     /* Free physical memory in bytes. */
#if defined(__x86_64__) || defined(_M_X64)
        uint32_t x86_64_cpu_features[8][4];         /* x86-64 CPU features. */
#elif defined (__aarch64__)
        long cpu_arm64_hwcap;                           /* ARM 64 hardware capabilities */
#endif
    } sys;
#ifdef MAG_DEBUG
    mag_tensor_node_t* rc_tracked;                      /* Linked list of RC tensors for sanitize. */
#endif
    mag_prng_algorithm_t prng_algo;                     /* PRNG algorithm. */
    mag_fixed_intrusive_pool tensor_pool;               /* Fixed-size memory pool for tensors. */
    mag_exec_mode_t exec_mode;
    bool profiler_enabled;
    mag_op_perf_info_t op_perf_mons_total[MAG_OP__NUM];
    uintptr_t tr_id;                                    /* Host thread ID. */
    size_t sh_len;                                      /* Number of shutdown hooks. */
    size_t sh_cap;                                      /* Maximum number of shutdown hooks. */
    mag_compute_device_type_t device_type;              /* Active compute device. */
    mag_compute_device_t* device;                       /* Active compute device. */
    uint8_t* (*image_load_fn)(const char*, uint32_t(*)[3], mag_color_channels_t);    /* Image loader. stb_image by default, you can plug-in your own. */
    void (*image_load_free_fn)(uint8_t*);                                           /* Image loader free function.  stb_image by default, you can plug-in your own. */
    bool (*image_save_fn)(const char*, const uint8_t*, const uint32_t(*)[3]);       /* Image saver. stb_image by default, you can plug-in your own. */
    void* ud; /* User data. */
};

typedef enum mag_tensor_flags_t {
    MAG_TFLAG_NONE = 0,
    MAG_TFLAG_OWNER = 1<<0,         /* Tensor is the owner of the buffer. */
    MAG_TFLAG_VIEW = 1<<1,          /* Tensor is a view. */
    MAG_FLAG_GRAD = 1<<2,           /* Tensor is a gradient. */
    MAG_TFLAG_EXEC_EAGER = 1<<3,    /* Tensor is executed eagerly. */

    MAG_TFLAG_LEN = 4
} mag_tensor_flags_t;
mag_static_assert(MAG_TFLAG_LEN <= 0xff);

/*
** Tensor with up to 6 Dimensions.
*/
struct mag_tensor_t {
    struct {
        uint32_t rc_strong;                                 /* Strong reference count. */
        uint32_t rc_weak;                                   /* Weak reference count. */
#ifdef MAG_DEBUG
        void (*dtor)(mag_tensor_t*);                        /* Debug destructor. */
#endif
    } rcb;                                                  /* Reference count control block. */
    mag_ctx_t* ctx;                                         /* Host context. */
    int64_t rank;                                           /* Number of active dimensions. [1, MAX_DIMS] */
    int64_t shape[MAG_MAX_DIMS];                            /* Shape of the tensor. */
    int64_t strides[MAG_MAX_DIMS];                          /* Strides of the tensor. We store the strides in element counts and NOT in bytes. */
    mag_dtype_t dtype;                                      /* Data type of the tensor. */
    mag_storage_buffer_t storage;                           /* Storage buffer. */
    int64_t numel;                                          /* Number of elements in the tensor. */
    mag_tensor_flags_t flags;                               /* Tensor flags. */
    mag_op_t op;                                            /* Opcode for operators. */
    mag_tensor_t* op_inputs[MAG_MAX_INPUT_TENSORS];         /* Input tensors for operators. */
    mag_op_param_t op_params[MAG_MAX_OP_PARAMS];            /* Operator parameters. */
    mag_init_op init_op;                                    /* Initialization operation. */
    mag_op_param_t init_op_params[MAG_MAX_INIT_OP_PARAMS];  /* Operator parameters. */
    mag_tensor_t* view_uplink;                              /* View base tensor. */
    size_t view_offs;                                       /* Offset in view tensor. */
    mag_tensor_t* grad;                                     /* ∇f - Gradient tensor. */
    mag_perf_mon_t pmon;                                    /* Performance monitor. */
    char name[MAG_MAX_TENSOR_NAME_LEN];                     /* Tensor debug name. */
    void* ud;                                               /* User data. */
};

#define mag_load_local_storage_group_arr(arr, prefix) \
    const int64_t prefix##0 = (arr)[0]; \
    const int64_t prefix##1 = (arr)[1]; \
    const int64_t prefix##2 = (arr)[2]; \
    const int64_t prefix##3 = (arr)[3]; \
    const int64_t prefix##4 = (arr)[4]; \
    const int64_t prefix##5 = (arr)[5]; \
    (void)prefix##0; \
    (void)prefix##1; \
    (void)prefix##2; \
    (void)prefix##3; \
    (void)prefix##4; \
    (void)prefix##5

typedef struct mag_prng_state_t {
    union {
        struct {
            uint64_t state;
            uint64_t inc;
        } pcg;
        struct {
            uint32_t remaining;
            uint32_t next;
            uint32_t state[624];
        } mersenne;
    } state;
    mag_prng_algorithm_t algo;
} mag_prng_state_t;
extern void mag_prng_init(mag_prng_state_t* rng, mag_prng_algorithm_t algo, uint64_t seed);

typedef struct mag_compute_payload_t {
    int64_t thread_num;
    int64_t thread_idx;
    mag_prng_state_t local_prng;
    mag_tensor_t* node;
} mag_compute_payload_t;

typedef struct mag_kernel_registry_t {
    void (*init[MAG_INIT_OP__NUM])(const mag_compute_payload_t*);
    void (*fwd[MAG_OP__NUM])(const mag_compute_payload_t*);
    void (*bwd[MAG_OP__NUM])(const mag_compute_payload_t*);
} mag_kernel_registry_t;

#define mag_load_local_storage_group(xk, prefix, var) mag_load_local_storage_group_arr((xk)->var, prefix)

#if defined(__x86_64__) || defined(_M_X64)
#define MAG_X86_64_CPUID_0H 0
#define MAG_X86_64_CPUID_1H 1
#define MAG_X86_64_CPUID_2H 2
#define MAG_X86_64_CPUID_7H 3
#define MAG_X86_64_CPUID_80000001H 4
#define MAG_X86_64_CPUID_80000007H 5
#define MAG_X86_64_CPUID_16H 6
#define MAG_X86_64_CPUID_7H_1H 7
#define MAG_X86_64_CPUID_EAX 0
#define MAG_X86_64_CPUID_EBX 1
#define MAG_X86_64_CPUID_ECX 2
#define MAG_X86_64_CPUID_EDX 3

#define mag_x86_64_feature_def(_, __) /* Enumerator | CPUDID Leaf | Register | Bit Index */\
    _(AVX                  ,    1H,        ECX,     28)__\
    _(AVX2                 ,    7H,        EBX,      5)__\
    _(AVXVNNI              ,    7H_1H,     EAX,      4)__\
    _(AVXVNNIINT8          ,    7H_1H,     EDX,      4)__\
    _(AVXVNNIINT16         ,    7H_1H,     EDX,     10)__\
    _(AVX512BW             ,    7H,        EBX,     30)__\
    _(AVX512CD             ,    7H,        EBX,     28)__\
    _(AVX512DQ             ,    7H,        EBX,     17)__\
    _(AVX512ER             ,    7H,        EBX,     27)__\
    _(AVX512F              ,    7H,        EBX,     16)__\
    _(AVX512IFMA           ,    7H,        EBX,     21)__\
    _(AVX512PF             ,    7H,        EBX,     26)__\
    _(AVX512VBMI           ,    7H,        ECX,      1)__\
    _(AVX512VL             ,    7H,        EBX,     31)__\
    _(AVX512_4FMAPS        ,    7H,        EDX,      3)__\
    _(AVX512_4VNNIW        ,    7H,        EDX,      2)__\
    _(AVX512_FP16          ,    7H,        EDX,     23)__\
    _(AVX512_BF16          ,    7H_1H,     EAX,      5)__\
    _(AVX512_BITALG        ,    7H,        ECX,     12)__\
    _(AVX512_VBMI2         ,    7H,        ECX,      6)__\
    _(AVX512_VNNI          ,    7H,        ECX,     11)__\
    _(AVX512_VP2INTERSECT  ,    7H,        EDX,      8)__\
    _(AVX512_VPOPCNTDQ     ,    7H,        ECX,     14)__\
    _(BMI                  ,    7H,        EBX,      3)__\
    _(BMI2                 ,    7H,        EBX,      8)__\
    _(F16C                 ,    1H,        ECX,     29)__\
    _(FMA                  ,    1H,        ECX,     12)__\
    _(FPU                  ,    1H,        EDX,      0)__\
    _(GFNI                 ,    7H,        ECX,      8)__\
    _(IA64                 ,    1H,        EDX,     30)__\
    _(MMX                  ,    1H,        EDX,     23)__\
    _(OSXSAVE              ,    1H,        ECX,     27)__\
    _(PCLMUL               ,    1H,        ECX,      1)__\
    _(RDRND                ,    1H,        ECX,     30)__\
    _(RDSEED               ,    7H,        EBX,     18)__\
    _(RDTSCP               ,    80000001H, EDX,     27)__\
    _(SHA                  ,    7H,        EBX,     29)__\
    _(SSE                  ,    1H,        EDX,     25)__\
    _(SSE2                 ,    1H,        EDX,     26)__\
    _(SSE3                 ,    1H,        ECX,      0)__\
    _(SSE4_1               ,    1H,        ECX,     19)__\
    _(SSE4_2               ,    1H,        ECX,     20)__\
    _(SSSE3                ,    1H,        ECX,      9)__\
    _(VAES                 ,    7H,        ECX,      9)__\
    _(VME                  ,    1H,        EDX,      1)__\
    _(VMX                  ,    1H,        ECX,      5)__\
    _(VPCLMULQDQ           ,    7H,        ECX,     10)__\
    _(XSAVE                ,    1H,        ECX,     26)__\
    _(HYBRID_CPU           ,    7H,        EDX,     15)__

#define _(enumerator, leaf, reg, bit) MAG_X86_64_FEATURE_##enumerator
typedef enum mag_x86_64_feature_t {
    mag_x86_64_feature_def(_, MAG_SEP)
    MAG_X86_64_FEATURE__NUM
} mag_x86_64_feature_t;
#undef _
extern const char* const mag_x86_64_feature_names[MAG_X86_64_FEATURE__NUM];
extern const uint8_t mag_x86_64_feature_leaves[MAG_X86_64_FEATURE__NUM];
extern const uint8_t mag_x86_64_feature_regs[MAG_X86_64_FEATURE__NUM];
extern const uint32_t mag_x86_64_feature_masks[MAG_X86_64_FEATURE__NUM];

static inline bool mag_ctx_x86_64_cpu_has_feature(const mag_ctx_t* ctx, mag_x86_64_feature_t feature) {
    const uint8_t (*leafs)[49] = &mag_x86_64_feature_leaves;
    const uint8_t (*regs)[49] = &mag_x86_64_feature_regs;
    const uint32_t (*masks)[49] = &mag_x86_64_feature_masks;
    const uint32_t (*features)[8][4] = &ctx->sys.x86_64_cpu_features;
    return (*features)[(*leafs)[feature]][(*regs)[feature]] & (*masks)[feature];
}
#endif

#ifdef __cplusplus
}
#endif

#endif
