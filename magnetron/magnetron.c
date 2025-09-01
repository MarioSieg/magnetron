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

#include <magnetron/magnetron.h>
#include "magnetron_internal.h"

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <time.h>
#include <float.h>
#include <ctype.h>
#include <errno.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <synchapi.h>
#include <io.h>
#include <fcntl.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>
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
__attribute__((weak_import)) extern int __ulock_wait(uint32_t op, void* addr, uint64_t value, uint32_t timeout);
__attribute__((weak_import)) extern int __ulock_wake(uint32_t op, void* addr, uint64_t value);
#else
#include <unistd.h>
#ifdef __linux__
#include <linux/prctl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/futex.h>
#ifdef __aarch64__
#include <sys/auxv.h>
#endif
#endif
#endif

#ifdef MAGNETRON_USE_MIMALLOC
#include <mimalloc.h>
#endif

#ifdef MAG_DEBUG
#define MAG_LOG_DEFAULT_ENABLE 1
#else
#define MAG_LOG_DEFAULT_ENABLE 0
#endif
bool mag_log_enabled = MAG_LOG_DEFAULT_ENABLE; /* Read from multiple threads, allowed to be written from main thread once at start. */
#undef MAG_LOG_DEFAULT_ENABLE

void mag_set_log_mode(bool enabled) {
    mag_log_enabled = enabled;
}

#if defined(__linux__) && defined(__GLIBC__)
#include <sys/wait.h>
#include <execinfo.h>
static void mag_dump_backtrace(void) { /* Try to print backtrace using gdb or lldb. */
    char proc[64];
    snprintf(proc, sizeof(proc), "attach %d", getpid());
    int pid = fork();
    if (pid == 0) {
        execlp("gdb", "gdb", "--batch", "-ex", "set style enabled on", "-ex", proc, "-ex", "bt -frame-info source-and-location", "-ex", "detach", "-ex", "quit", NULL);
        execlp("lldb", "lldb", "--batch", "-o", "bt", "-o", "quit", "-p", proc, NULL);
        exit(EXIT_FAILURE);
    }
    int stat;
    waitpid(pid, &stat, 0);
    if (WIFEXITED(stat) && WEXITSTATUS(stat) == EXIT_FAILURE) {
        void* trace[0xff];
        backtrace_symbols_fd(trace, backtrace(trace, sizeof(trace)/sizeof(*trace)), STDERR_FILENO);
    }
}
#else
static void mag_dump_backtrace(void) { }
#endif

static void MAG_COLDPROC mag_panic_dump(FILE* f, bool cc, const char* msg, va_list args) {
    if (cc) fprintf(f, "%s", MAG_CC_RED);
    vfprintf(f, msg, args);
    if (cc) fprintf(f, "%s", MAG_CC_RESET);
    fputc('\n', f);
    fflush(f);
}

MAG_NORET MAG_COLDPROC void mag_panic(const char* msg, ...) { /* Panic and exit the program. If available print backtrace. */
    va_list args;
    va_start(args, msg);
    #if 0
        FILE* f = fopen("magnetron_panic.log", "w");
        if (f) {
            mag_panic_dump(f, false, msg, args);
            fclose(f), f = NULL;
        }
    #endif
    fflush(stdout);
    mag_panic_dump(stderr, true, msg, args);
    va_end(args);
    #ifdef NDEBUG
        mag_dump_backtrace();
    #endif
    abort();
}

#ifdef MAGNETRON_USE_MIMALLOC

static void* mag_alloc_stub(void* blk, size_t size, size_t align) { /* Allocator stub for mimalloc. */
    if (mag_unlikely(align <= sizeof(void*))) align = 0;
    mag_assert2(!align || !(align & (align-1)));
    if (!size) {
        mi_free(blk);
        return NULL;
    }
    if (!blk) {
        void* p = align ? mi_malloc_aligned(size, align) : mi_malloc(size);
        if (mag_unlikely(!p)) mag_panic("Failed to allocate %zu bytes", size);
        return p;
    }
    void* p = align ? mi_realloc_aligned(blk, size, align) : mi_realloc(blk, size);
    if (mag_unlikely(!p)) mag_panic("Failed to reallocate %zu bytes", size);
    return p;
}

#else

#if defined(__GLIBC__) || defined(__linux__)
#include <malloc.h>
#define mag_msize malloc_usable_size
#elif defined(__FreeBSD__)
#include <malloc_np.h>
#define mag_msize malloc_usable_size
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#define mag_msize malloc_size
#elif defined(_WIN32)
#include <malloc.h>
#define mag_msize _msize
#else
#error "Unknown platform"
#endif

static void* mag_alloc_stub(void* blk, size_t size, size_t align) {
  if (mag_unlikely(align <= sizeof(void*))) align = 0;
  if (!size) {
      if (!blk) return NULL;
      free(align ? ((void**)blk)[-1] : blk);
      return NULL;
  }
  if (!blk) goto alloc;
  if (!align) {
    void* new_blk = realloc(blk, size);
    if (!new_blk) mag_panic("Failed to reallocate %zu bytes", size);
    return new_blk;
  } else {
    void* old_base = ((void**)blk)[-1];
    size_t old_size = mag_msize(old_base)-((uintptr_t)blk-(uintptr_t)old_base);
    alloc:
    if (!align) {
        void* p = malloc(size);
        if (!p) mag_panic("Failed to allocate %zu bytes", size);
        return p;
    }
    if (align & (align-1) || align < sizeof(void*)) mag_panic("Alignment %zu is not a power of two ≥ sizeof(void*)", align);
    if (size > SIZE_MAX-align-sizeof(void*)) mag_panic("Size/align overflow");
    void* raw = malloc(size+align+sizeof(void*));
    if (!raw) mag_panic("Failed to allocate %zu bytes", size);
    uintptr_t aligned_addr = ((uintptr_t)raw+sizeof(void*)+align-1)&~(uintptr_t)(align-1);
    void* user = (void*)aligned_addr;
    ((void**)user)[-1] = raw;
    if (blk) {
      memcpy(user, blk, old_size < size ? old_size : size);
      free(old_base);
    }
    return user;
  }
}

/* Allocate aligned memory by overallocating. Alignment must be a power of two. */
void* mag_alloc_aligned(size_t size, size_t align) {
    mag_assert(align && !(align&(align-1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    void* p = (*mag_alloc)(NULL, size+sizeof(void*)+align-1, 0);
    uintptr_t pp = ((uintptr_t)p+sizeof(void*)+align-1)&-align;
    ((void**)pp)[-1] = p;
    return (void*)pp;
}

void mag_free_aligned(void* blk) {
    (*mag_alloc)(((void**)blk)[-1], 0, 0);
}

#endif

void* (*mag_alloc)(void*, size_t, size_t) = &mag_alloc_stub;

void* (*mag_get_alloc_fn(void))(void*, size_t, size_t) { return mag_alloc; } /* Get global allocator. */
void mag_set_alloc_fn(void* (*alloc)(void*, size_t, size_t)) { mag_assert2(alloc); mag_alloc = alloc; } /* Set global allocator. */

/* Humanize memory size. Format and convert a memory size to the appropriate unit. For example. 1024 => 1 KiB */
void mag_humanize_memory_size(size_t n, mag_e11m52_t* out, const char** unit) {
    if (n < (1<<10)) {
        *out = (mag_e11m52_t)n;
        *unit = "B";
    } else if (n < (1<<20)) {
        *out = (mag_e11m52_t)n/(mag_e11m52_t)(1<<10);
        *unit = "KiB";
    } else if (n < (1<<30)) {
        *out = (mag_e11m52_t)n/(mag_e11m52_t)(1<<20);
        *unit = "MiB";
    } else {
        *out = (mag_e11m52_t)n/(mag_e11m52_t)(1<<30);
        *unit = "GiB";
    }
}

void MAG_COLDPROC mag_print_separator(FILE* f) { /* Print a separator line. */
    f = f ? f : stdout;
    char sep[100+1];
    for (size_t i=0; i < (sizeof(sep)/sizeof(*sep))-1; ++i) sep[i] = '-';
    sep[sizeof(sep)/sizeof(*sep)-1] = '\0';
    fprintf(f, "%s\n", sep);
}

#define MAG_FMT_DIM_BUF_SIZE ((21+4)*MAG_MAX_DIMS)

/* Format a dimension tuple into a Python-like string. e.g. (4, 12). */
void mag_fmt_shape(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    memset(*buf, 0, sizeof(*buf));
    char* p = *buf;
    mag_assert2(p+rank*21+3 < *buf+MAG_FMT_DIM_BUF_SIZE);
    *p++ = '(';
    for (int64_t i=0; i < rank; ++i) {
        p += snprintf(p, 21, "%" PRIi64, (*dims)[i]);
        if (i < rank-1) {
            *p++ = ',';
            *p++ = ' ';
        }
    }
    *p++ = ')';
    *p = '\0';
}

#ifdef _WIN32
#include <wchar.h>
extern __declspec(dllimport) int __stdcall MultiByteToWideChar(
    unsigned int cp,
    unsigned long flags,
    const char* str,
    int cbmb,
    wchar_t* widestr,
    int cchwide
);
extern __declspec(dllimport) int __stdcall WideCharToMultiByte(
    unsigned int cp,
    unsigned long flags,
    const wchar_t* widestr,
    int cchwide,
    char* str,
    int cbmb,
    const char* defchar,
    int* used_default
);
#endif

/* Open file. Basically fopen but with UTF-8 support on Windows. */
static FILE* mag_fopen(const char* file, const char* mode) {
    mag_assert(file && *file && mode && *mode, "Invalid file name or mode");
    FILE* f = NULL;
    #ifdef _WIN32
        wchar_t w_mode[64];
        wchar_t w_file[1024];
        if (MultiByteToWideChar(65001 /* UTF8 */, 0, file, -1, w_file, sizeof(w_file)/sizeof(*w_file)) == 0) return NULL;
        if (MultiByteToWideChar(65001 /* UTF8 */, 0, mode, -1, w_mode, sizeof(w_mode)/sizeof(*w_mode)) == 0) return NULL;
        #if defined(_MSC_VER) && _MSC_VER >= 1400
           if (_wfopen_s(&f, w_file, w_mode) != 0)
               return NULL;
        #else
           f = _wfopen(w_file, w_mode);
        #endif
    #elif defined(_MSC_VER) && _MSC_VER >= 1400
        if (fopen_s(&f, filename, mode) != 0) return NULL;
    #else
        f = fopen(file, mode);
    #endif
    return f;
}

uintptr_t mag_thread_id(void) { /* Get the current thread ID. */
    uintptr_t tid;
    #if defined(_MSC_VER) && defined(_M_X64)
        tid = __readgsqword(48);
    #elif defined(_MSC_VER) && defined(_M_IX86)
        tid = __readfsdword(24);
    #elif defined(_MSC_VER) && defined(_M_ARM64)
        tid = __getReg(18);
    #elif defined(__i386__)
        __asm__ __volatile__("movl %%gs:0, %0" : "=r" (tid));  /* x86-32 WIN32 uses %GS */
    #elif defined(__MACH__) && defined(__x86_64__)
        __asm__ __volatile__("movq %%gs:0, %0" : "=r" (tid));  /* x86.64 OSX uses %GS */
    #elif defined(__x86_64__)
        __asm__ __volatile__("movq %%fs:0, %0" : "=r" (tid));  /* x86-64 Linux and BSD uses %FS */
    #elif defined(__arm__)
        __asm__ __volatile__("mrc p15, 0, %0, c13, c0, 3\nbic %0, %0, #3" : "=r" (tid));
    #elif defined(__aarch64__) && defined(__APPLE__)
        __asm__ __volatile__("mrs %0, tpidrro_el0" : "=r" (tid));
    #elif defined(__aarch64__)
        __asm__ __volatile__("mrs %0, tpidr_el0" : "=r" (tid));
    #elif defined(__powerpc64__)
    #ifdef __clang__
        tid = (uintptr_t)__builtin_thread_pointer();
    #else
        register uintptr_t tp __asm__ ("r13");
        __asm__ __volatile__("" : "=r" (tp));
        tid = tp;
    #endif
    #elif defined(__powerpc__)
    #ifdef __clang__
        tid = (uintptr_t)__builtin_thread_pointer();
    #else
        register uintptr_t tp __asm__ ("r2");
        __asm__ __volatile__("" : "=r" (tp));
        tid = tp;
    #endif
    #elif defined(__s390__) && defined(__GNUC__)
        tid = (uintptr_t)__builtin_thread_pointer();
    #elif defined(__riscv)
    #ifdef __clang__
        tid = (uintptr_t)__builtin_thread_pointer();
    #else
        __asm__ ("mv %0, tp" : "=r" (tid));
    #endif
    #else
    #error "Unsupported magnetron platform"
    #endif
    return tid;
}

#if defined(__x86_64__) || defined(_M_X64)
const char* const mag_amd64_cpu_cap_names[MAG_AMD64_CAP__NUM] = {
    #define _(name) #name
    mag_amd64_capdef(_, MAG_SEP)
    #undef _
};
#elif defined(__aarch64__)
#define _(ident) #ident
const char* const mag_arm64_cpu_cap_names[MAG_ARM64_CAP__NUM] = {
    mag_armd64_capdef(_, MAG_SEP)
};
#endif

uint64_t mag_hpc_clock_ns(void) { /* High precision clock in nanoseconds. */
    #ifdef _WIN32
        static LONGLONG t_freq;
        static LONGLONG t_boot;
        static bool t_init = false;
        if (!t_init) { /* Reduce chance of integer overflow when uptime is high. */
            LARGE_INTEGER li;
            QueryPerformanceFrequency(&li);
            t_freq = li.QuadPart;
            QueryPerformanceCounter(&li);
            t_boot = li.QuadPart;
            t_init = true;
        }
        LARGE_INTEGER li;
        QueryPerformanceCounter(&li);
        return ((li.QuadPart - t_boot)*1000000000) / t_freq;
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (uint64_t)ts.tv_sec*1000000000 + (uint64_t)ts.tv_nsec;
    #endif
}
uint64_t mag_hpc_clock_elapsed_ns(uint64_t start) { /* High precision clock elapsed time in microseconds. */
    return (uint64_t)llabs((long long)mag_hpc_clock_ns() - (long long)start);
}
mag_e11m52_t mag_hpc_clock_elapsed_ms(uint64_t start) { /* High precision clock elapsed time in milliseconds. */
    return (mag_e11m52_t)mag_hpc_clock_elapsed_ns(start) / 1e6;
}

#ifdef _MSC_VER
extern uint64_t __rdtsc();
#pragma intrinsic(__rdtsc)
#endif

uint64_t mag_cycles(void) {
    #ifdef __APPLE__
        return mach_absolute_time();
    #elif defined(_MSC_VER)
        return __rdtsc();
    #elif defined(__x86_64__) || defined(__amd64__)
        uint64_t lo, hi;
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return (hi<<32) | lo;
    #elif defined(__aarch64__)
        uint32_t cntrl;
        uint32_t rwx;
        uint32_t fset;
        __asm__ __volatile__("mrc p15, 0, %0, c9, c14, 0" : "=r"(rwx)); /* Read perfmon access control register */
        if (rwx & 1) {
            __asm__ __volatile__("mrc p15, 0, %0, c9, c12, 1" : "=r"(fset));
            if (fset & 0x80000000U) {
                __asm__ __volatile__("mrc p15, 0, %0, c9, c13, 0" : "=r"(cntrl));
                return (uint64_t)(cntrl)<<6;
            }
        }
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (uint64_t)(tv.tv_sec)*1000000 + tv.tv_usec;
    #else
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (uint64_t)(tv.tv_sec)*1000000 + tv.tv_usec;
    #endif
}

/* Bitset for 32-bit integers. */
typedef uint32_t mag_bitset_t;
mag_static_assert(sizeof(mag_bitset_t) == 4);
#define mag_bitset_size(n) (((n)+((4<<3)-1))>>5)
#define mag_bitset_get(sets, i) (!!(sets[(i)>>5]&(1u<<((i)&((4<<3)-1)))))
#define mag_bitset_set(sets, i) (sets[(i)>>5]|=(1u<<((i)&((4<<3)-1))))
#define mag_bitset_clear(sets, i) (sets[(i)>>5]&=~(1u<<((i)&((4<<3)-1))))
#define mag_bitset_toggle(sets, i) (sets[(i)>>5]^=(1u<<((i)&((4<<3)-1))))

/* Tensor hashset with linear probing. */
typedef struct mag_hashset_t {
    size_t len;
    mag_bitset_t* used;
    const mag_tensor_t** keys;
} mag_hashset_t;
#define MAG_HASHSET_FULL ((size_t)-1)
#define MAG_HASHSET_DUPLICATE ((size_t)-2)
#define MAG_HASHSET_MAX ((size_t)-3) /* Must be last. */
#define mag_hashset_hash_fn(ptr) ((size_t)(uintptr_t)(ptr)>>3)

/* Find optimal hash size for lim sz. */
static size_t mag_hashset_compute_hash_size(size_t sz) {
    mag_assert2(sz > 0 && sz < MAG_HASHSET_MAX);
    static const size_t prime_lut[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    size_t l = 0;
    size_t r = sizeof(prime_lut)/sizeof(*prime_lut);
    while (l < r) { /* Binary search for the smallest prime > sz. */
        size_t mid = (l+r)>>1;
        if (prime_lut[mid] < sz) l = mid+1;
        else r = mid;
    }
    return l < sizeof(prime_lut)/sizeof(*prime_lut) ? prime_lut[l] : sz|1;
}

/* Create a new hashset. */
static mag_hashset_t mag_hashset_init(size_t size) {
    size = mag_hashset_compute_hash_size(size);
    mag_hashset_t set = {
        .len = size,
        .used = (*mag_alloc)(NULL, mag_bitset_size(size)*sizeof(*set.used), 0),
        .keys = (*mag_alloc)(NULL, size*sizeof(*set.keys), 0),
    };
    memset(set.used, 0, mag_bitset_size(size)*sizeof(*set.used));
    return set;
}

/* Lookup a key in the hashset. Returns index or MAG_HASHSET_FULL if full. */
static size_t mag_hashset_lookup(mag_hashset_t* set, const mag_tensor_t* key) {
    size_t k = mag_hashset_hash_fn(key) % set->len, i = k;
    while (mag_bitset_get(set->used, i) && set->keys[i] != key) { /* Simple linear probe. */
        i = (i+1) % set->len;
        if (i == k) return MAG_HASHSET_FULL; /* Full */
    }
    return i;
}

/* Check if a key exists in the hashset. */
static bool mag_hashset_contains_key(mag_hashset_t* set, const mag_tensor_t* key) {
    size_t i = mag_hashset_lookup(set, key);
    return mag_bitset_get(set->used, i) && i != MAG_HASHSET_FULL;
}

/* Insert a key into the hashset. Returns index or MAG_HASHSET_DUPLICATE if already exists. */
static size_t mag_hashset_insert(mag_hashset_t* set, const mag_tensor_t* key) {
    size_t k = mag_hashset_hash_fn(key) % set->len, i = k;
    do { /* Simple linear probing */
        if (!mag_bitset_get(set->used, i)) { /* Insert key. */
            mag_bitset_set(set->used, i);
            set->keys[i] = key;
            return i;
        }
        if (set->keys[i] == key) return MAG_HASHSET_DUPLICATE; /* Key already exists. */
        i = (i+1) % set->len;
    } while (i != k);
    return MAG_HASHSET_FULL; /* Full */
}

/* Reset the hashset. */
static void mag_hashset_reset(mag_hashset_t* set) {
    memset(set->used, 0, mag_bitset_size(set->len)*sizeof(*set->used));
}

/* Clear the hashset. */
static void mag_hashset_free(mag_hashset_t* set) {
    (*mag_alloc)(set->used, 0, 0);
    (*mag_alloc)(set->keys, 0, 0);
}

/* Eval Chebyshev coeffs steps for some x. f(x) : [a, b] -> ℝ. */
static mag_e11m52_t mag_chebyshev_eval(mag_e11m52_t x, mag_e11m52_t a, mag_e11m52_t b, const mag_e11m52_t* coeffs, uint32_t steps) {
    mag_e11m52_t scale = 4.0/(b - a);
    mag_e11m52_t rls = -2.0 + (x - a)*scale;
    mag_e11m52_t k1 = 0.0, k2 = 0.0;
    for (uint32_t j = steps-1; j; --j) {
        mag_e11m52_t tmp = k1;
        k1 = rls*k1 - k2 + coeffs[j];
        k2 = tmp;
    }
    return 0.5*rls*k1 - k2 + 0.5**coeffs;
}

/* Generate Chebyshev coeffs for f(x) : [a, b] -> ℝ. */
static mag_e11m52_t* mag_chebyshev_setup(mag_e11m52_t (*f)(mag_e11m52_t), mag_e11m52_t a, mag_e11m52_t b, uint32_t steps, bool linear_l, bool linear_r) {
    mag_assert2(steps);
    mag_e11m52_t* r = (*mag_alloc)(NULL, sizeof(*r)*steps, 0);
    memset(r, 0, sizeof(*r)*steps);
    mag_e11m52_t dsteps = (mag_e11m52_t)steps;
    for (uint32_t i=0; i < steps; ++i) {
        for (uint32_t j=0; j < steps; ++j) {
            mag_e11m52_t wav = 0.5*(1.0 + cos(M_PI*(j + 0.5)/dsteps));
            mag_e11m52_t x = a + (b - a)*wav, y = (*f)(x);
            mag_e11m52_t weight = cos(M_PI*(mag_e11m52_t)i*(j + 0.5)/dsteps);
            r[i] += 2.0*y*weight/dsteps;
        }
    }
    mag_e11m52_t xmi=0.0, xma = 0.0;
    if (linear_l) xmi = (*f)(a) - mag_chebyshev_eval(a, a, b, r, steps);
    if (linear_r) xma = (*f)(b) - mag_chebyshev_eval(b, a, b, r, steps);
    r[0] += 2.0*(xma + xmi)*0.5;
    r[1] += (xma - xmi)*0.5;
    return r;
}

mag_device_desc_t mag_compute_device_desc_cpu(uint32_t thread_count) {
    return (mag_device_desc_t){
        .type = MAG_DEVICE_TYPE_CPU,
        .cpu_thread_count = thread_count
    };
}

mag_device_desc_t mag_compute_device_desc_cuda(uint32_t cuda_device_id) {
    return (mag_device_desc_t){
        .type = MAG_DEVICE_TYPE_GPU_CUDA,
        .cpu_thread_count = cuda_device_id
    };
}

static void mag_view_meta_dtor(void* p) {
    mag_view_meta_t* vm = p;
    mag_context_t* ctx = vm->base->ctx;
    if (vm->base->view_meta == vm)
        vm->base->view_meta = NULL;
    mag_rc_control_decref(&vm->base->rc_control);
    mag_fixed_pool_free_block(&ctx->view_meta_pool, vm);
}

mag_view_meta_t* mag_view_meta_alloc(mag_tensor_t* base){
    mag_view_meta_t* vm = mag_fixed_pool_alloc_block(&base->ctx->view_meta_pool);
    vm->rc = mag_rc_control_init(vm, &mag_view_meta_dtor);
    vm->base = base;
    mag_rc_control_incref(&base->rc_control);
    vm->version_snapshot = base->version;
    return vm;
}

static void mag_au_state_dtor(void* p) {
    mag_au_state_t* au = p;
    if (au->grad) {
        mag_tensor_decref(au->grad);
        au->grad = NULL;
    }
    for (size_t i=0; i < sizeof(au->op_inputs)/sizeof(*au->op_inputs); ++i)
        if (au->op_inputs[i]) mag_tensor_decref(au->op_inputs[i]);
    mag_fixed_pool_free_block(&au->ctx->view_meta_pool, au);
}

mag_au_state_t* mag_au_state_lazy_alloc(mag_au_state_t** au_state, mag_context_t* ctx) {
    if (*au_state) return *au_state;
    *au_state = mag_fixed_pool_alloc_block(&ctx->au_state_pool);
    **au_state = (mag_au_state_t){
        .ctx = ctx,
        .rc = mag_rc_control_init(*au_state, &mag_au_state_dtor),
        .op = MAG_OP_NOP,
        .op_inputs = {},
        .op_params = {},
        .grad = NULL,
    };
    return *au_state;
}

/* Initialize and seed PRNG state. */
void mag_prng_seed(mag_prng_state_t* prng, mag_prng_algo_t algo, uint64_t seed) {
    seed = seed ? seed : 0x853c49e6748fea9bull;
    switch ((prng->algo = algo)) {
        case MAG_PRNG_MERSENNE_TWISTER: { /* Mersenne Twister */
            uint32_t* state = prng->mersenne.state;
            *state = (uint32_t)seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1]^(state[i-1]>>30))*1812433253 + i)&~0u;
            prng->mersenne.next = 0;
            prng->mersenne.remaining = 1;
        } break;
        case MAG_PRNG_PCG: { /* PCG-XSH-RR */
            prng->pcg.state = seed^0x853c49e6748fea9bull;
            prng->pcg.inc = 0xda3e39cb94b95bdbull;
        } break;
        default:
            mag_panic("invalid PRNG algorithm: %d", prng->algo);
    }
}

static void mag_machine_probe(mag_context_t* ctx); /* Query host system information. */

/* Print host system and machine information. */
static void mag_system_host_info_dump(mag_context_t* ctx) {
    mag_log_info("OS/Kernel: %s", ctx->machine.os_name);
    const char* cpu_arch = "?";
    #if defined(__x86_64__) || defined(_M_X64)
        cpu_arch = "x86-64";
    #elif defined(__aarch64__) || defined(_M_ARM64)
        cpu_arch = "aarch64";
    #else
    #error "Unknwon CPU arch"
    #endif
    mag_log_info(
        "CPU: %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u, L1D: %.01f KiB, L2: %.01f KiB, L3: %.01f MiB",
        ctx->machine.cpu_name,
        ctx->machine.cpu_virtual_cores,
        ctx->machine.cpu_physical_cores,
        ctx->machine.cpu_sockets,
        (mag_e11m52_t)ctx->machine.cpu_l1_size/1024.0,
        (mag_e11m52_t) ctx->machine.cpu_l2_size/1024.0,
        (mag_e11m52_t)ctx->machine.cpu_l3_size/1024.0/1024.0
    );
    #if defined(__x86_64__) || defined(_M_X64) /* Print detected CPU features for x86-64 platforms. */
        if (mag_log_enabled) {
            printf(MAG_CC_CYAN "[magnetron] " MAG_CC_RESET "%s caps: ", cpu_arch);
            for (unsigned i=0; i < MAG_AMD64_CAP__NUM; ++i) {
                if (i == MAG_AMD64_CAP_AMD || i == MAG_AMD64_CAP_INTEL) continue; /* Skip vendor caps */
                if (ctx->machine.amd64_cpu_caps & mag_amd64_cap_bit(i))
                    printf("%s ", mag_amd64_cpu_cap_names[i]);
            }
            putchar('\n');
        }
    #elif defined(__aarch64__) /* Print detected CPU features for ARM64 platforms. */
        if (mag_log_enabled) {
            printf(MAG_CC_CYAN "[magnetron] " MAG_CC_RESET "%s caps: ", cpu_arch);
            for (uint32_t i=0; i < MAG_ARM64_CAP__NUM; ++i)
                if (ctx->machine.arm64_cpu_caps & (1ull<<i))
                    printf("%s ", mag_arm64_cpu_cap_names[i]);
            putchar('\n');
        }
    #endif
    /* Now print memory information. */
    mag_e11m52_t mem_total, mem_free, mem_used;
    const char* mem_unit_total, *mem_unit_free, *mem_unit_used;
    mag_humanize_memory_size(ctx->machine.phys_mem_total, &mem_total, &mem_unit_total);
    mag_humanize_memory_size(ctx->machine.phys_mem_free, &mem_free, &mem_unit_free);
    mag_humanize_memory_size((size_t)llabs((int64_t)ctx->machine.phys_mem_total-(int64_t)ctx->machine.phys_mem_free), &mem_used, &mem_unit_used);
    mag_e11m52_t mem_used_percent = fabs((mag_e11m52_t)(ctx->machine.phys_mem_total-ctx->machine.phys_mem_free))/(mag_e11m52_t)ctx->machine.phys_mem_total*100.0;
    mag_log_info("Physical Machine Memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);
}

/* Print compiler information such as name, version and build time. */
static MAG_COLDPROC void mag_ctx_dump_compiler_info(void) {
    const char* compiler_name = "Unknown";
    int compiler_version_major = 0, compiler_version_minor = 0;
    #ifdef __clang__
        compiler_name = "Clang";
        compiler_version_major = __clang_major__;
        compiler_version_minor = __clang_minor__;
    #elif defined(__GNUC__)
        compiler_name = "GCC";
        compiler_version_major = __GNUC__;
        compiler_version_minor = __GNUC_MINOR__;
    #elif defined(_MSC_VER)
        compiler_name = "MSVC";
        compiler_version_major = _MSC_VER / 100;
        compiler_version_minor = _MSC_VER % 100;
    #endif
    mag_log_info("magnetron v.%d.%d - " __DATE__ " " __TIME__ " - %s %d.%d", mag_version_major(MAG_VERSION), mag_version_minor(MAG_VERSION), compiler_name, compiler_version_major, compiler_version_minor);
}

#ifdef MAG_DEBUG
/* Leak detection helpers */

static void mag_leak_detector_enqueue(mag_tensor_t* t) {
    mag_context_t* ctx = t->ctx;
    t->alive_next = ctx->alive_head;
    ctx->alive_head = t;
}

static void mag_leak_detector_dequeue(mag_tensor_t* t) {
    mag_context_t* ctx = t->ctx;
    for (mag_tensor_t** p = &ctx->alive_head; *p; p = &(*p)->alive_next) {
        if (*p == t) {
            *p = t->alive_next;
            break;
        }
    }
}

static MAG_COLDPROC void mag_leak_detector_dump_results(mag_context_t* ctx) {
    for (mag_tensor_t* leaked = ctx->alive_head; leaked; leaked = leaked->alive_next) {
        char shape[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&shape, &leaked->shape, leaked->rank);
        fprintf(
            stderr,
            MAG_CC_RED "[magnetron] " MAG_CC_RESET "Leaked tensor: %p, Shape: %s\n",
            leaked,
            shape
        );
    }
    fflush(stderr);
}

#endif

/* Create a magnetron context with the selected compute device. */
mag_context_t* mag_ctx_create(mag_device_type_t device) {
    const mag_device_desc_t info = {device};
    return mag_ctx_create2(&info);
}

/* Create context with compute device descriptor. */
mag_context_t* mag_ctx_create2(const mag_device_desc_t* device_info) {
    mag_log_info("Creating magnetron context...");

    uint64_t time_stamp_start = mag_hpc_clock_ns();
    mag_ctx_dump_compiler_info(); /* Dump compiler info. */

    /* Initialize context with default values or from context info. */
    mag_context_t* ctx = (*mag_alloc)(NULL, sizeof(*ctx), 0); /* Allocate context. */
    memset(ctx, 0, sizeof(*ctx));

    /* Init memory pools */
    mag_fixed_pool_init(&ctx->tensor_pool, sizeof(mag_tensor_t), __alignof(mag_tensor_t), 0x1000);
    mag_fixed_pool_init(&ctx->storage_pool, sizeof(mag_istorage_t), __alignof(mag_istorage_t), 0x1000);
    mag_fixed_pool_init(&ctx->view_meta_pool, sizeof(mag_view_meta_t), __alignof(mag_view_meta_t), 0x1000);
    mag_fixed_pool_init(&ctx->au_state_pool, sizeof(mag_au_state_t), __alignof(mag_au_state_t), 0x1000);

    ctx->tr_id = mag_thread_id(); /* Get thread ID. */
    ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER; /* Enable gradient recording by default. */
    ctx->prng_algo = MAG_PRNG_MERSENNE_TWISTER;

    /* Query and print host system information. */
    mag_machine_probe(ctx);
    mag_system_host_info_dump(ctx);

    /* Create selected compute device. */
    ctx->device_type = device_info->type;
    ctx->device = mag_init_dynamic_device(ctx, device_info);
    mag_log_info("Compute device: %s", ctx->device->name);

    /* Print context initialization time. */
    mag_log_info("magnetron context initialized in %.05f ms", mag_hpc_clock_elapsed_ms(time_stamp_start));
    return ctx;
}

void mag_ctx_destroy(mag_context_t* ctx, bool suppress_leak_detection) { /* Destroy magnetron context. */
    #ifdef MAG_DEBUG
        mag_leak_detector_dump_results(ctx);  /* Provide detailed leak check info */
    #endif
    bool leaks_detected = ctx->num_tensors || ctx->num_storages;
    if (mag_unlikely(leaks_detected)) {
        char msg[256] = {0};
        snprintf(msg, sizeof(msg), "magnetron context destroyed with %zu leaked tensors and %zu leaked storages", ctx->num_tensors, ctx->num_storages);
        if (suppress_leak_detection) mag_log_warn("%s", msg);
        else mag_panic("%s", msg);
    }
    mag_fixed_pool_destroy(&ctx->au_state_pool);
    mag_fixed_pool_destroy(&ctx->view_meta_pool);
    mag_fixed_pool_destroy(&ctx->tensor_pool);
    mag_fixed_pool_destroy(&ctx->storage_pool);
    mag_destroy_dynamic_device(ctx->device); ctx->device = NULL; /* Shutdown compute device. */
    memset(ctx, 255, sizeof(*ctx)); /* Poison context memory range. */
    (*mag_alloc)(ctx, 0, 0); /* Free ctx. */
    ctx = NULL;
    mag_log_info("magnetron context destroyed");
}

mag_prng_algo_t mag_ctx_get_prng_algorithm(const mag_context_t* ctx) {
    return ctx->prng_algo;
}

void mag_ctx_set_prng_algorithm(mag_context_t* ctx, mag_prng_algo_t algorithm, uint64_t seed) {
    mag_log_warn("Setting the PRNG algorithm is not implemented at the moment");
}

mag_device_type_t mag_ctx_get_compute_device_type(const mag_context_t* ctx) { return ctx->device_type; }
const char* mag_ctx_get_compute_device_name(const mag_context_t* ctx) { return ctx->device->name; }
const char* mag_ctx_get_os_name(const mag_context_t* ctx) { return ctx->machine.os_name; }
const char* mag_ctx_get_cpu_name(const mag_context_t* ctx) { return ctx->machine.cpu_name; }
uint32_t mag_ctx_get_cpu_virtual_cores(const mag_context_t* ctx) { return ctx->machine.cpu_virtual_cores; }
uint32_t mag_ctx_get_cpu_physical_cores(const mag_context_t* ctx) { return ctx->machine.cpu_physical_cores; }
uint32_t mag_ctx_get_cpu_sockets(const mag_context_t* ctx) { return ctx->machine.cpu_sockets; }
uint64_t mag_ctx_get_physical_memory_total(const mag_context_t* ctx) { return ctx->machine.phys_mem_total; }
uint64_t mag_ctx_get_physical_memory_free(const mag_context_t* ctx) { return ctx->machine.phys_mem_free; }
bool mag_ctx_is_numa_system(const mag_context_t* ctx) { return false; /* TODO */ }
size_t mag_ctx_get_total_tensors_created(const mag_context_t* ctx) { return 0; /* TODO */ }

/* Set scheduling priority for current thread. */
void mag_thread_set_prio(mag_thread_prio_t prio) {
#ifdef _WIN32
    DWORD policy = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case MAG_THREAD_PRIO_NORMAL: policy = THREAD_PRIORITY_NORMAL; break;
        case MAG_THREAD_PRIO_MEDIUM: policy = THREAD_PRIORITY_ABOVE_NORMAL; break;
        case MAG_THREAD_PRIO_HIGH: policy = THREAD_PRIORITY_HIGHEST; break;
        case MAG_THREAD_PRIO_REALTIME: policy = THREAD_PRIORITY_TIME_CRITICAL; break;
    }
    if (mag_unlikely(!SetThreadPriority(GetCurrentThread(), policy))) {
        mag_log_warn("Failed to set thread scheduling priority: %d", prio);
    }
#else
    int32_t policy = SCHED_OTHER;
    struct sched_param p;
    switch (prio) {
        case MAG_THREAD_PRIO_NORMAL: p.sched_priority = 0;  policy = SCHED_OTHER; break;
        case MAG_THREAD_PRIO_MEDIUM: p.sched_priority = 40; policy = SCHED_FIFO; break;
        case MAG_THREAD_PRIO_HIGH: p.sched_priority = 80; policy = SCHED_FIFO; break;
        case MAG_THREAD_PRIO_REALTIME: p.sched_priority = 90; policy = SCHED_FIFO; break;
    }
    int status = pthread_setschedparam(pthread_self(), policy, &p);
    if (mag_unlikely(status)) {
        mag_log_warn("Failed to set thread scheduling priority: %d, error: %x", prio, status);
    }
    #endif
}

/* Set thread name for current thread. */
void mag_thread_set_name(const char* name) {
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

int mag_futex_wait(volatile mag_atomic32_t* addr, mag_atomic32_t expect) {
    #ifdef __linux__
        return syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expect, NULL, NULL, 0);
    #elif defined(__APPLE__)
        mag_assert2(__ulock_wait);
        return __ulock_wait(UL_COMPARE_AND_WAIT, (void*)addr, expect, 0);
    #elif defined(_WIN32)
        BOOL ok = WaitOnAddress((volatile VOID*)addr, &expect, sizeof(expect), INFINITE);
        if (mag_likely(ok)) return 0;
        errno = GetLastError() == ERROR_TIMEOUT ? ETIMEDOUT : EAGAIN;
        return -1;
    #else
    #error "Not implemented for this platform"
    #endif
}

void mag_futex_wake1(volatile mag_atomic32_t* addr) {
    #ifdef __linux__
        syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
    #elif defined(__APPLE__)
        mag_assert2(__ulock_wake);
        __ulock_wake(UL_COMPARE_AND_WAIT, (void*)addr, 0);
    #elif defined(_WIN32)
        WakeByAddressSingle((PVOID)addr);
    #else
    #error "Not implemented for this platform"
    #endif
}

void mag_futex_wakeall(volatile mag_atomic32_t* addr) {
    #ifdef __linux__
        syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, 0x7fffffff, NULL, NULL, 0);
    #elif defined(__APPLE__)
        mag_assert2(__ulock_wake);
        __ulock_wake(UL_COMPARE_AND_WAIT|ULF_WAKE_ALL, (void*)addr, 0);
    #elif defined(_WIN32)
        WakeByAddressAll((PVOID)addr);
    #else
    #error "Not implemented for this platform"
    #endif
}

void mag_sstream_init(mag_sstream_t* ss) {
    memset(ss, 0, sizeof(*ss));
    ss->cap = 0x200;
    ss->len = 0;
    ss->buf = (*mag_alloc)(NULL, ss->cap, 0);
    *ss->buf = '\0';
}

void mag_sstream_free(mag_sstream_t* ss) {
    (*mag_alloc)(ss->buf, 0, 0);
    memset(ss, 0, sizeof(*ss));
}

void mag_sstream_reserve_more(mag_sstream_t* ss, size_t extra) {
    size_t want = ss->len+extra+1; /* +1 for terminator */
    if (want <= ss->cap) return;
    while (ss->cap < want) ss->cap <<= 1; /* geometric growth */
    ss->buf = (*mag_alloc)(ss->buf, ss->cap, 0);
}

void mag_sstream_vappend(mag_sstream_t* ss, const char* fmt, va_list ap0) {
    va_list ap;
    va_copy(ap, ap0);
    int need = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (mag_unlikely(need < 0)) return;
    size_t want = ss->len + (size_t)need+1; /* +1 for terminator */
    if (want > ss->cap) {
        while (ss->cap < want) ss->cap <<= 1; /* geometric growth */
        ss->buf = (*mag_alloc)(ss->buf, ss->cap, 0);
    }
    va_copy(ap, ap0);
    vsnprintf(ss->buf + ss->len, ss->cap - ss->len, fmt, ap);
    va_end(ap);
    ss->len += (size_t)need;
}

void mag_sstream_append(mag_sstream_t* ss, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    mag_sstream_vappend(ss, fmt, ap);
    va_end(ap);
}

void mag_sstream_append_strn(mag_sstream_t* ss, const char* str, size_t len) {
    if (mag_unlikely(!len)) return;
    mag_sstream_reserve_more(ss, len);
    memcpy(ss->buf + ss->len, str, len);
    ss->len += len;
    ss->buf[ss->len] = '\0';
}

void mag_sstream_putc(mag_sstream_t* ss, char c){
    mag_sstream_reserve_more(ss, 1);
    ss->buf[ss->len++] = c;
    ss->buf[ss->len] = '\0';
}

void mag_sstream_flush(mag_sstream_t* ss, FILE* f) {
   fputs(ss->buf, f);
}

/* Allocate a new linear chunk for a fixed pool. */
static mag_pool_chunk_t* mag_fixed_pool_chunk_new(size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    size_t cap = blocks_per_chunk*block_size;
    uintptr_t size = 0;
    mag_pincr((void**)&size, sizeof(mag_pool_chunk_t), __alignof(mag_pool_chunk_t));
    mag_pincr((void**)&size, cap, block_align);
    void* base = (*mag_alloc)(NULL, size, 0), *pos = base;
    mag_pool_chunk_t* chunk = mag_pincr(&pos, sizeof(mag_pool_chunk_t), __alignof(mag_pool_chunk_t));
    uint8_t* bot = mag_pincr(&pos, cap, block_align);
    *chunk = (mag_pool_chunk_t) {
        .bot = bot,
        .top = bot+cap,
        .next = NULL
    };
    return chunk;
}

/* Initialize fixed intrusive pool and allocate start chunk. */
void mag_fixed_pool_init(mag_fixed_pool_t* pool, size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    mag_assert2(blocks_per_chunk);
    block_size = mag_xmax(sizeof(void*), block_size); /* Ensure block size is at least sizeof(void*) to store intrusive free list. */
    mag_pool_chunk_t* chunk = mag_fixed_pool_chunk_new(block_size, block_align, blocks_per_chunk);
    *pool = (mag_fixed_pool_t) {
        .block_size = block_size,
        .block_align = block_align,
        .blocks_per_chunk = blocks_per_chunk,
        .chunks = chunk,
        .chunk_head = chunk,
        .free_list = NULL,
        .num_freelist_hits = 0,
        .num_pool_hits = 0,
        .num_chunks = 1,
        .num_allocs = 0
    };
}

/* Allocate a new fixed block from the pool. Memory is uninitialized. */
void* mag_fixed_pool_alloc_block(mag_fixed_pool_t* pool) {
    ++pool->num_allocs;
    if (mag_likely(pool->free_list)) { /* 1. Try to pop from free_list (fastest path) */
        ++pool->num_freelist_hits;
        void* blk = pool->free_list;
        pool->free_list = *(void**)blk; /* Next free block is stored at block [0..sizeof(void*)-1] */
        return blk;
    }
    mag_pool_chunk_t* chunk = pool->chunk_head;
    mag_assert2(chunk);
    uint8_t* top = chunk->top-pool->block_size;
    if (mag_likely(top >= chunk->bot)) {  /* 2. Allocate from the last pool if possible (fast path) */
        ++pool->num_pool_hits;
        chunk->top = top;
        return top;
    }
    /* 3. Current chunk is exhausted, allocate new (slow path) */
    mag_pool_chunk_t* new_chunk = mag_fixed_pool_chunk_new(pool->block_size, pool->block_align, pool->blocks_per_chunk);
    chunk->next = new_chunk;
    pool->chunk_head = new_chunk;
    new_chunk->top -= pool->block_size;
    ++pool->num_chunks;
    return new_chunk->top;
}

/* Free a fixed block back to the pool. This effectively pushes it into the freelist. */
void mag_fixed_pool_free_block(mag_fixed_pool_t* pool, void* blk) {
    *(void**)blk = pool->free_list;
    pool->free_list = blk;
}

/* Destroy fixed intrusive pool and free all allocated memory. */
void mag_fixed_pool_destroy(mag_fixed_pool_t* pool) {
    mag_pool_chunk_t* chunk = pool->chunks;
    while (chunk) {
        mag_pool_chunk_t* next = chunk->next;
        (*mag_alloc)(chunk, 0, 0);
        chunk = next;
    }
    memset(pool, 0, sizeof(*pool));
}

/* Print pool information and allocation stats. */
MAG_COLDPROC void mag_fixed_pool_print_info(mag_fixed_pool_t* pool, const char* name) {
    mag_log_info("Fixed Intrusive Pool: %s", name);
    mag_log_info(
        "\tBlock Size: %zu B, Block Align: %zu B, Blocks Per Chunk: %zu B",
        pool->block_size,
        pool->block_align,
        pool->blocks_per_chunk
    );
    mag_log_info(
        "\tChunks: %zu, Allocs: %zu, Freelist Hits: %zu, Num Pool Hits: %zu",
        (size_t)pool->num_chunks,
        (size_t)pool->num_allocs,
        (size_t)pool->num_freelist_hits,
        (size_t)pool->num_pool_hits
    );
    mag_e11m52_t mem_alloced, pool_mem;
    const char* mem_unit_alloced, *mem_unit_pool;
    mag_humanize_memory_size(pool->num_chunks*pool->blocks_per_chunk*pool->block_size, &mem_alloced, &mem_unit_alloced);
    mag_humanize_memory_size(pool->num_allocs*pool->block_size, &pool_mem, &mem_unit_pool);
    mag_log_info("\t Real Mem Allocated: %.03f %s, Total Pool Mem %.03f %s", mem_alloced, mem_unit_alloced, pool_mem, mem_unit_pool);
}

/* Pack rgb8 into a 32-bit color. Alpha channel unused. */
uint32_t mag_pack_color_u8(uint8_t r, uint8_t g, uint8_t b) { return ((uint32_t)r<<16)|((uint32_t)g<<8)|(uint32_t)b; }

/* Pack rgb8 into a 32-bit color and normalize. Alpha channel unused. */
uint32_t mag_pack_color_f32(mag_e8m23_t r, mag_e8m23_t g, mag_e8m23_t b) {
    return mag_pack_color_u8((uint8_t)(r*255.f), (uint8_t)(g*255.f), (uint8_t)(b*255.f));
}

void mag_ctx_grad_recorder_start(mag_context_t* ctx) { ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER; }
void mag_ctx_grad_recorder_stop(mag_context_t* ctx) { ctx->flags &= ~MAG_CTX_FLAG_GRAD_RECORDER; }
bool mag_ctx_grad_recorder_is_running(const mag_context_t* ctx) { return ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER; }

const char* mag_device_type_get_name(mag_device_type_t op) {
    static const char* const names[MAG_DEVICE_TYPE__NUM] = {
        [MAG_DEVICE_TYPE_CPU] = "CPU",
        [MAG_DEVICE_TYPE_GPU_CUDA] = "GPU (CUDA)",
    };
    return names[op];
}

const mag_dtype_meta_t* mag_dtype_meta_of(mag_dtype_t type) {
    static const mag_dtype_meta_t infos[MAG_DTYPE__NUM] = {
        [MAG_DTYPE_E8M23] = {
            .name="e8m23",
            .size=sizeof(mag_e8m23_t),
            .align=__alignof(mag_e8m23_t),
        },
        [MAG_DTYPE_E5M10] = {
            .name="e5m10",
            .size=sizeof(mag_e5m10_t),
            .align=__alignof(mag_e5m10_t),
        },
        [MAG_DTYPE_BOOL] = {
            .name="bool",
            .size=sizeof(uint8_t),
            .align=__alignof(uint8_t),
        },
        [MAG_DTYPE_I32] = {
            .name="i32",
            .size=sizeof(int32_t),
            .align=__alignof(int32_t),
        },
    };
    return &infos[type];
}

static void mag_tensor_dtor(void* self); /* Destructor forward declaration. */

static mag_tensor_t* mag_tensor_init_header(mag_context_t* ctx, mag_dtype_t type, int64_t rank, int64_t numel) {
    mag_tensor_t* hdr = mag_fixed_pool_alloc_block(&ctx->tensor_pool); /* Allocate tensor header. */
    memset(hdr, 0, sizeof(*hdr));
    *hdr = (mag_tensor_t) { /* Initialize tensor header. */
        .ctx = ctx,
        .rc_control = mag_rc_control_init(hdr, &mag_tensor_dtor), /* Initialize reference counter. */
        .rank = rank,
        .shape = {0},
        .strides = {0},
        .dtype = type,
        .storage = NULL,
        .numel = numel,
        .flags = MAG_TFLAG_NONE,
        .storage_offset = 0,
        .view_meta = NULL,
        .au_state = NULL,
        .version = 0,
    };
#ifdef MAG_DEBUG
    hdr->alive_next = NULL;
    mag_leak_detector_enqueue(hdr);
#endif
    ++ctx->num_tensors; /* Increase tensor count in context. */
    return hdr;
}

/* Create a new tensor. The must be created on the same thread as the context. */
mag_tensor_t* mag_tensor_new(mag_context_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape) {
    uintptr_t tr_id = mag_thread_id();
    mag_assert(ctx && tr_id == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", tr_id, ctx->tr_id);  /* Ensure that the tensor is created on the same thread as the context. */
    mag_assert(shape && rank > 0 && rank <= MAG_MAX_DIMS, "Rank must be within (0, %d]", MAG_MAX_DIMS); /* Check rank */
    int64_t dts = (int64_t)mag_dtype_meta_of(type)->size;
    int64_t numel = 1;
    for (int64_t i=0; i < rank; ++i)
        mag_assert2(shape[i] > 0 && !mag_mulov64(shape[i], numel, &numel));
    int64_t numbytes;
    mag_assert2(!mag_mulov64(numel, dts, &numbytes)); /* Compute number of bytes */
    mag_tensor_t* tensor = mag_tensor_init_header(ctx, type, rank, numel); /* Alloc tensor header. */
    mag_idevice_t* dvc = ctx->device;
    void (*allocator)(mag_idevice_t*, mag_istorage_t**, size_t, mag_dtype_t) = dvc->alloc_storage;
    (*allocator)(dvc, &tensor->storage, numbytes, type);
    for (int i=0; i < MAG_MAX_DIMS; ++i)  {
        tensor->shape[i] = i < rank ? shape[i] : 1;
        tensor->strides[i] = 1;
    }
    /* Compute contiguous row-major strides and check for overflow. */
    tensor->strides[rank-1] = 1;
    for (int64_t i=rank-2; i >= 0; --i)
        mag_assert2(!mag_mulov64(tensor->strides[i+1], tensor->shape[i+1], tensor->strides+i));
    return tensor;
}

mag_tensor_t* mag_tensor_as_strided(mag_context_t* ctx, mag_tensor_t* base, int64_t rank, const int64_t* shape, const int64_t* strides, int64_t offset) {
    uintptr_t tr_id = mag_thread_id();
    mag_assert(ctx && tr_id == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", tr_id, ctx->tr_id);  /* Ensure that the tensor is created on the same thread as the context. */
    mag_assert(base && shape && strides && rank > 0 && rank <= MAG_MAX_DIMS, "Rank must be within (0, %d]", MAG_MAX_DIMS);
    mag_assert(offset >= 0, "negative storage offset: %" PRIi64, offset);
    int64_t last = offset;
    int64_t numel = 1;
    for (int64_t i=0; i < rank; ++i) {
        mag_assert2(shape[i] > 0 && (shape[i] == 1 ? strides[i] >= 0 : strides[i] > 0));
        int64_t span;
        mag_assert2(!mag_mulov64(shape[i]-1, strides[i], &span));
        mag_assert2(!mag_mulov64(shape[i], numel, &numel));
        last += span;
    }
    int64_t numel_end = (int64_t)base->storage->size/base->storage->granularity;
    mag_assert(last < numel_end, "view exceeds backing storage size: %" PRIi64 " >= %" PRIi64, last, numel_end);
    mag_tensor_t* tensor = mag_tensor_init_header(ctx, base->dtype, rank, numel); /* Alloc tensor header. */
    for (int i=0; i < MAG_MAX_DIMS; ++i) {
        tensor->shape[i] = i < rank ? shape[i] : 1;
        tensor->strides[i] = i < rank ? strides[i] : 1;
    }
    tensor->storage = base->storage;
    mag_rc_control_incref(&base->storage->rc_control); /* Retain base storage */
    tensor->storage_offset = offset;
    tensor->version = base->version;
    if (!(base->flags & MAG_TFLAG_IS_VIEW)) /* first view */
        tensor->view_meta = mag_view_meta_alloc(base);
    else {
        tensor->view_meta = base->view_meta;
        mag_rc_control_incref(&tensor->view_meta->rc); /* Retain view meta */
    }
    tensor->flags = base->flags | MAG_TFLAG_IS_VIEW; /* Set view flag */
    return tensor;
}

static void mag_tensor_dtor(void* self) {
    mag_tensor_t* t = self;
    mag_context_t* ctx = t->ctx;
    mag_assert(ctx->num_tensors > 0, "double freed tensor");
    --ctx->num_tensors;
    if (t->view_meta) {
        mag_rc_control_decref(&t->view_meta->rc);
        t->view_meta = NULL;
    }
    if (t->au_state) {
        mag_rc_control_decref(&t->au_state->rc);
        t->au_state = NULL;
    }
    mag_rc_control_decref(&t->storage->rc_control);
#ifdef MAG_DEBUG
    mag_leak_detector_dequeue(t); /* Pop from alive list */
    memset(t, 0, sizeof(*t));
#endif
    mag_fixed_pool_free_block(&ctx->tensor_pool, t);
}

mag_tensor_t* mag_tensor_empty(mag_context_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape) {
    return mag_tensor_new(ctx, type, rank, shape);
}

mag_tensor_t* mag_tensor_empty_like(mag_tensor_t* isomorph) {
    return mag_tensor_new(isomorph->ctx, isomorph->dtype, isomorph->rank, isomorph->shape);
}

mag_tensor_t* mag_tensor_empty_scalar(mag_context_t* ctx, mag_dtype_t type) {
    return mag_tensor_empty(ctx, type, 1, (int64_t[1]){1});
}

mag_tensor_t* mag_tensor_scalar(mag_context_t* ctx, mag_dtype_t type, mag_e8m23_t value) {
    mag_tensor_t* tensor = mag_tensor_empty_scalar(ctx, type);
    mag_tensor_fill_float(tensor, value);
    return tensor;
}

mag_tensor_t* mag_tensor_full(mag_context_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape, mag_e8m23_t value) {
    mag_tensor_t* tensor = mag_tensor_empty(ctx, type, rank, shape);
    mag_tensor_fill_float(tensor, value);
    return tensor;
}

mag_tensor_t* mag_tensor_full_like(mag_tensor_t* isomorph, mag_e8m23_t value) {
    mag_tensor_t* tensor = mag_tensor_empty_like(isomorph);
    mag_tensor_fill_float(tensor, value);
    return tensor;
}

mag_tensor_t* mag_contiguous(mag_tensor_t* x) {
    if (!x->storage_offset && mag_tensor_is_contiguous(x)) {
        mag_tensor_incref(x); /* If already contiguous, just incref */
        return x;
    }
    return mag_clone(x);
}

int64_t mag_tensor_get_data_size(const mag_tensor_t* t) { return t->storage->size; }
int64_t mag_tensor_get_numel(const mag_tensor_t* t) { return t->numel; }

void mag_tensor_incref(mag_tensor_t* t) { /* Increase reference count of the tensor. */
    mag_rc_control_incref(&t->rc_control);
}

bool mag_tensor_decref(mag_tensor_t* t) { /* Decrease reference count of the tensor. */
    return mag_rc_control_decref(&t->rc_control);
}

void mag_tensor_detach_inplace(mag_tensor_t* target) {
    target->flags &= ~MAG_TFLAG_REQUIRES_GRAD; /* Detach from gradient recording */
    if (target->au_state) {
        target->au_state->op = MAG_OP_NOP; /* Detach from operations */
        memset(target->au_state->op_inputs, 0, sizeof(target->au_state->op_inputs)); /* Clear op inputs */
        memset(target->au_state->op_params, 0, sizeof(target->au_state->op_params));
    }
}

mag_tensor_t* mag_tensor_detach(mag_tensor_t* t) {
   mag_tensor_detach_inplace(t);
    return t;
}

uint64_t mag_tensor_get_refcount(const mag_tensor_t* t) { return t->rc_control.rc; }
uint64_t mag_tensor_get_storage_refcount(const mag_tensor_t* t) { return t->storage->rc_control.rc; }
size_t mag_tensor_get_memory_usage(const mag_tensor_t* t) {
    return sizeof(*t) + mag_tensor_get_data_size(t);
}

int64_t mag_tensor_get_rank(const mag_tensor_t* t) { return t->rank; }
const int64_t* mag_tensor_get_shape(const mag_tensor_t* t) { return t->shape; }
const int64_t* mag_tensor_get_strides(const mag_tensor_t* t) { return t->strides; }
mag_dtype_t mag_tensor_get_dtype(const mag_tensor_t* t) { return t->dtype; }
size_t mag_tensor_get_data_offset(const mag_tensor_t *t) {
    return (size_t)t->storage_offset*t->storage->granularity; /* Return offset in bytes */
}
void* mag_tensor_get_data_ptr(const mag_tensor_t* t) {
    return (void*)(t->storage->base + mag_tensor_get_data_offset(t));
}
void* mag_tensor_get_storage_base_ptr(const mag_tensor_t* t) { return (void*)t->storage->base; }

void* mag_tensor_get_raw_data_as_bytes(mag_tensor_t* t) {
    t = mag_contiguous(t); /* Ensure tensor is contiguous */
    size_t size = mag_tensor_get_data_size(t);
    mag_assert2(size);
    void* dst = (*mag_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
    mag_istorage_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t), dst, size);
    mag_tensor_decref(t);
    return dst;
}

void mag_tensor_get_raw_data_as_bytes_free(void* ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

mag_e8m23_t* mag_tensor_get_data_as_floats(mag_tensor_t* t) {
    t = mag_contiguous(t); /* Ensure tensor is contiguous */
    mag_assert(mag_tensor_is_floating_point_typed(t), "Tensor must be a floating point tensor, but has dtype: %s", mag_dtype_meta_of(t->dtype)->name);
    size_t size = t->numel*sizeof(mag_e8m23_t);
    mag_assert2(size);
    mag_e8m23_t* dst = (*mag_alloc)(NULL, size, 0); /* TODO: Use dynamic scratch buffer */
    mag_istorage_t* sto = t->storage;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t), dst, size, MAG_DTYPE_E8M23);
    mag_tensor_decref(t);
    return dst;
}

void mag_tensor_get_data_as_floats_free(mag_e8m23_t* ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

mag_e8m23_t mag_tensor_get_item_float(const mag_tensor_t* t) {
    mag_istorage_t* sto = t->storage;
    mag_e8m23_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t), &val, sizeof(val), MAG_DTYPE_E8M23);
    return val;
}

int32_t mag_tensor_get_item_int(const mag_tensor_t* t) {
    mag_istorage_t* sto = t->storage;
    int32_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t), &val, sizeof(val), MAG_DTYPE_I32);
    return val;
}

bool mag_tensor_get_item_bool(const mag_tensor_t* t) {
    mag_istorage_t* sto = t->storage;
    uint8_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t), &val, sizeof(val), MAG_DTYPE_BOOL);
    return !!val;
}

bool mag_tensor_is_shape_eq(const mag_tensor_t* x, const mag_tensor_t* y) {
    return memcmp(x->shape, y->shape, sizeof(x->shape)) == 0;
}

bool mag_tensor_are_strides_eq(const mag_tensor_t* x, const mag_tensor_t* y) {
    return memcmp(x->strides, y->strides, sizeof(x->strides)) == 0;
}

bool mag_tensor_can_broadcast(const mag_tensor_t* small, const mag_tensor_t* big) {
    int64_t mr = mag_xmax(small->rank, big->rank);
    for (int64_t d=0; d < mr; ++d) {
        int64_t asz = d < small->rank ? small->shape[small->rank-1-d] : 1;
        int64_t bsz = d < big->rank ? big->shape[big->rank-1-d] : 1;
        if (asz != bsz && asz != 1 && bsz != 1)
            return false;
    }
    return true;
}

bool mag_tensor_is_transposed(const mag_tensor_t* t) { return t->strides[0] > t->strides[1]; }

bool mag_tensor_is_permuted(const mag_tensor_t* t) {
    for (int i=0; i < MAG_MAX_DIMS-1; ++i)
        if (t->strides[i] > t->strides[i+1])
            return true;
    return false;
}

bool mag_tensor_is_contiguous(const mag_tensor_t* t) {
    int64_t str = 1;
    for (int64_t d=t->rank-1; d >= 0; --d) {
        int64_t size_d = t->shape[d];
        if (size_d == 1) continue;
        if (t->strides[d] != str) return false;
        str *= size_d;
    }
    return true;
}

bool mag_tensor_can_view(const mag_tensor_t *t, const int64_t* dims, int64_t rank) {
    int64_t tmp[MAG_MAX_DIMS];
    return mag_solve_view_strides(&tmp, t->shape, t->strides, t->rank, dims, rank);
}

mag_tensor_t* mag_tensor_get_grad(const mag_tensor_t* t) {
    mag_assert2(t->flags & MAG_TFLAG_REQUIRES_GRAD && t->au_state);
    if (t->au_state->grad) mag_tensor_incref(t->au_state->grad);
    return t->au_state->grad;
}

bool mag_tensor_requires_grad(const mag_tensor_t* t) {
    return t->flags & MAG_TFLAG_REQUIRES_GRAD;
}

void mag_tensor_set_requires_grad(mag_tensor_t* t, bool requires_grad) {
    if (requires_grad) {
        mag_assert(mag_tensor_is_floating_point_typed(t), "Gradient tracking tensors must be floating-point typed, but tensor has dtype: %s", mag_dtype_meta_of(t->dtype)->name);
        t->flags |= MAG_TFLAG_REQUIRES_GRAD;
        mag_au_state_lazy_alloc(&t->au_state, t->ctx);
    } else t->flags &= ~MAG_TFLAG_REQUIRES_GRAD;
}

typedef struct mag_topo_record_t {
    mag_tensor_t* tensor;
    uint32_t next_child_idx;
} mag_topo_record_t;

typedef struct mag_tensor_set_t {
    mag_tensor_t** data;
    size_t size;
    size_t capacity;
} mag_tensor_set_t;

static void mag_tensor_array_init(mag_tensor_set_t* arr) {
    arr->data = NULL;
    arr->size = 0;
    arr->capacity = 0;
}

static void mag_tensor_array_free(mag_tensor_set_t* arr) {
    (*mag_alloc)(arr->data, 0, 0);
    arr->size = 0;
    arr->capacity = 0;
}

static void mag_tensor_array_push(mag_tensor_set_t* arr, mag_tensor_t* t) {
    if (arr->size == arr->capacity) {
        size_t cap = !arr->capacity ? 16 : arr->capacity<<1;
        arr->data = (*mag_alloc)(arr->data, cap*sizeof(*arr->data), 0);
        arr->capacity = cap;
    }
    arr->data[arr->size++] = t;
}

static void mag_collect_topo_iterative(mag_tensor_t* root, mag_tensor_set_t* out_array) {
    size_t sta_len = 0, sta_cap = 0;
    mag_topo_record_t* stack = NULL;

    #define mag_sta_push(_t) do { \
        if (sta_len == sta_cap) { \
        size_t old_cap = sta_cap; \
        size_t nc = (old_cap == 0) ? 16 : (old_cap * 2); \
        stack = (*mag_alloc)(stack, nc*sizeof(*stack), 0); \
        sta_cap = nc; \
        } \
        stack[sta_len].tensor = (_t); \
        stack[sta_len].next_child_idx = 0; \
        sta_len++; \
    } while(0)
    #define mag_sta_pop() (stack[--sta_len])

    if (!(root->flags & MAG_TFLAG_REQUIRES_GRAD)) return;
    mag_hashset_t visited = mag_hashset_init(8192); // todo dynamic
    mag_sta_push(root);
    while (sta_len) { /* Iterative DFS */
        mag_topo_record_t* top = &stack[sta_len - 1];
        mag_tensor_t* cur_tensor = top->tensor;
        mag_assert(cur_tensor->au_state, "Autodiff state not allocated for tensor that requires gradient");
        if (top->next_child_idx < mag_op_meta_of(cur_tensor->au_state->op)->in) {
            mag_tensor_t* child = cur_tensor->au_state->op_inputs[top->next_child_idx++];
            if (child && (child->flags & MAG_TFLAG_REQUIRES_GRAD)) {
                if (!mag_hashset_contains_key(&visited, child)) {
                    mag_hashset_insert(&visited, child);
                    mag_sta_push(child);
                }
            }
        } else {
            (void)mag_sta_pop();
            mag_tensor_array_push(out_array, cur_tensor);
        }
    }

    #undef mag_sta_push
    #undef mag_sta_pop

    (*mag_alloc)(stack, 0, 0);
    mag_hashset_free(&visited);
}

static void mag_tensor_patch_grad(mag_tensor_t* dst, mag_tensor_t* grad) {
    if (dst->au_state->grad)
        mag_tensor_decref(dst->au_state->grad);
    grad->flags = (grad->flags|MAG_TFLAG_IS_GRAD)&~MAG_TFLAG_REQUIRES_GRAD;
    dst->au_state->grad = grad;
}

void mag_tensor_backward(mag_tensor_t* root) {
    mag_assert(root->flags & MAG_TFLAG_REQUIRES_GRAD, "Tensor must require grad to back-propagate");
    mag_assert(root->rank == 1 && root->numel == 1, "Tensor must be a scalar to back-propagate");
    mag_ctx_grad_recorder_stop(root->ctx);
    mag_tensor_set_t post_order;
    mag_tensor_array_init(&post_order);
    mag_collect_topo_iterative(root, &post_order);
    if (mag_unlikely(!post_order.size)) goto end;
    for (size_t i=0, j = post_order.size-1; i < j; ++i, --j)
        mag_swap(mag_tensor_t*, post_order.data[i], post_order.data[j]);
    for (size_t id=0; id < post_order.size; ++id) {
        mag_tensor_t* child = post_order.data[id];
        mag_assert(child && child->au_state, "Autodiff state not allocated for tensor that requires gradient");
        const mag_opmeta_t* meta = mag_op_meta_of(child->au_state->op);
        if (!child->au_state->grad) {
            mag_tensor_t* grad = mag_tensor_full_like(child, 1.0f);
            mag_tensor_patch_grad(child, grad);
        }
        if (mag_unlikely(child->au_state->op == MAG_OP_NOP)) continue;
        mag_tensor_t* grads[MAG_MAX_OP_INPUTS] = {0};
        void (*backward)(mag_au_state_t*, mag_tensor_t**) = meta->backward;
        mag_assert2(backward);
        (*backward)(child->au_state, grads);
        uint32_t numin = meta->in;
        mag_assert2(numin <= MAG_MAX_OP_INPUTS);
        for (uint32_t i=0; i < numin; ++i) {
            mag_tensor_t* input = child->au_state->op_inputs[i];
            mag_assert2(input);
            if (!(input->flags & MAG_TFLAG_REQUIRES_GRAD)) continue;
            mag_tensor_t* gri = grads[i];
            mag_assert(gri, "Gradient for op %s, input #%d is not computed", meta->mnemonic, i);
            if (!input->au_state->grad) {
                mag_tensor_patch_grad(input, gri);
            } else {
                mag_tensor_t* acc = mag_add(gri, input->au_state->grad);
                mag_tensor_patch_grad(input, acc);
                mag_tensor_decref(gri);
            }
        }
    }
    mag_tensor_array_free(&post_order);
    end:
    mag_ctx_grad_recorder_start(root->ctx);
}

void mag_tensor_zero_grad(mag_tensor_t* t) {
    if (t->flags & MAG_TFLAG_REQUIRES_GRAD && t->au_state && t->au_state->grad)
        mag_tensor_fill_float(t->au_state->grad, 0.0f);
}

/*
** Load all 6 elements of a 6-element array into local storage.
** Used for compute kernels to help the compiler to hold shape and stride values inside registers.
*/
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

#define mag_load_local_storage_group(xk, prefix, var) mag_load_local_storage_group_arr((xk)->var, prefix)

/* Compute dot product of 6 integers. Used to compute offsets in 6-dimensional index space. */
#define mag_address_dotprod6(x,y) ((x##0*y##0)+(x##1*y##1)+(x##2*y##2)+(x##3*y##3)+(x##4*y##4)+(x##5*y##5))

mag_e8m23_t mag_tensor_subscript_get_multi(mag_tensor_t* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    mag_istorage_t* sto = t->storage;
    mag_e8m23_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t) + sto->granularity*mag_address_dotprod6(i, s), &val, sizeof(val), MAG_DTYPE_E8M23);
    return val;
}

void mag_tensor_subscript_set_multi(mag_tensor_t* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5, mag_e8m23_t val) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    mag_istorage_t* sto = t->storage;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_H2D, mag_tensor_get_data_offset(t) + sto->granularity*mag_address_dotprod6(i, s), &val, sizeof(val), MAG_DTYPE_E8M23);
}

static MAG_AINLINE void mag_tensor_unravel_index(const mag_tensor_t* t, int64_t v_idx, int64_t(*p_idx)[MAG_MAX_DIMS]) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, d, shape);
    (*p_idx)[5] = v_idx / (d4*d3*d2*d1*d0);
    (*p_idx)[4] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0) / (d3*d2*d1*d0);
    (*p_idx)[3] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0) / (d2*d1*d0);
    (*p_idx)[2] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0) / (d1*d0);
    (*p_idx)[1] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0) / d0;
    (*p_idx)[0] =  v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0 - (*p_idx)[1]*d0;
}

mag_e8m23_t mag_tensor_subscript_get_flattened(mag_tensor_t* t, int64_t idx) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_unravel_index(t, idx, &pidx);
        return mag_tensor_subscript_get_multi(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5]);
    }
    mag_istorage_t* sto = t->storage;
    mag_e8m23_t val;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_D2H, mag_tensor_get_data_offset(t) + sto->granularity*(size_t)idx, &val, sizeof(val), MAG_DTYPE_E8M23);
    return val;
}

void mag_tensor_subscript_set_flattened(mag_tensor_t* t, int64_t idx, mag_e8m23_t val) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_unravel_index(t, idx, &pidx);
        mag_tensor_subscript_set_multi(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5], val);
        return;
    }
    mag_istorage_t* sto = t->storage;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_H2D, mag_tensor_get_data_offset(t) + sto->granularity*(size_t)idx, &val, sizeof(val), MAG_DTYPE_E8M23);
}

static void mag_fmt_single_elem(mag_sstream_t* ss, const void* buf, size_t i, mag_dtype_t dtype) {
    switch (dtype) {
        case MAG_DTYPE_E8M23:
        case MAG_DTYPE_E5M10:
            mag_sstream_append(ss, "%g", (mag_e11m52_t)((const mag_e8m23_t*)buf)[i]);
        return;
        case MAG_DTYPE_BOOL:
            mag_sstream_append(ss, "%s", ((const uint8_t*)buf)[i] ? "True" : "False");
        return;
        case MAG_DTYPE_I32:
            mag_sstream_append(ss, "%" PRIi32, ((const int32_t*)buf)[i]);
        return;
        default:
            mag_panic("DType formatting not implemented: %d", dtype);
    }
}

static void mag_tensor_fmt_recursive(
    mag_sstream_t* ss,
    const void* buf,
    mag_dtype_t dtype,
    const int64_t* shape,
    const int64_t* strides,
    int64_t rank,
    int depth,
    int64_t moff
) {
    if (depth == rank) /* scalar leaf */ {
        mag_fmt_single_elem(ss, buf, moff, dtype);
        return;
    }
    mag_sstream_putc(ss, '[');
    for (int64_t i=0; i < shape[depth]; ++i) {
        mag_tensor_fmt_recursive(ss, buf, dtype, shape, strides, rank, depth+1, moff + i*strides[depth]); /* Recurse down */
        if (i != shape[depth]-1) { /* separator */
            mag_sstream_putc(ss, ',');
            if (rank-depth > 1) { /* newline + indent for outer dims */
                mag_sstream_putc(ss, '\n');
                for (int j=0; j <= depth; ++j)
                    mag_sstream_putc(ss, ' ');
            } else { /* simple space for last dim */
                mag_sstream_putc(ss, ' ');
            }
        }
    }
    mag_sstream_putc(ss, ']');
}

char* mag_tensor_to_string(mag_tensor_t* t, bool with_header, size_t from_start_count, size_t from_end_count) {
    if (!from_end_count) from_end_count = UINT64_MAX;
    void* buf = NULL;
    if (mag_tensor_is_floating_point_typed(t)) /* For all float types we want a (maybe converted) fp32 buffer for easy formatting. */
        buf = mag_tensor_get_data_as_floats(t);
    else /* Integral types can be formated easily */
        buf = mag_tensor_get_raw_data_as_bytes(t);
    mag_sstream_t ss;
    mag_sstream_init(&ss);
    mag_tensor_fmt_recursive(&ss, buf, t->dtype, t->shape, t->strides, t->rank, 0, 0); /* Recursive format */
    /* Free allocated buffer */
    if (mag_tensor_is_floating_point_typed(t)) mag_tensor_get_data_as_floats_free(buf);
    else mag_tensor_get_raw_data_as_bytes_free(buf);
    return ss.buf; /* Return the string, must be freed with mag_tensor_to_string_free_data. */
}

void mag_tensor_to_string_free_data(char* ret_val) {
    (*mag_alloc)(ret_val, 0, 0);
}

mag_context_t* mag_tensor_get_ctx(const mag_tensor_t* t) { return t->ctx; }
int64_t mag_tensor_get_width(const mag_tensor_t* t) { return t->shape[2]; }
int64_t mag_tensor_get_height(const mag_tensor_t* t) { return t->shape[1]; }
int64_t mag_tensor_get_channels(const mag_tensor_t* t) { return t->shape[0]; }
bool mag_tensor_is_view(const mag_tensor_t* t) { return t->flags & MAG_TFLAG_IS_VIEW; }
bool mag_tensor_is_floating_point_typed(const mag_tensor_t* t) { return mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_FP; }
bool mag_tensor_is_integral_typed(const mag_tensor_t* t) { return mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_INTEGRAL; }
bool mag_tensor_is_integer_typed(const mag_tensor_t* t) { return mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_INTEGER; }
bool mag_tensor_is_numeric_typed(const mag_tensor_t* t) { return mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_NUMERIC; }

#ifdef __APPLE__
    static bool mag_sysctl_mib01(uint8_t (*out)[256], size_t* o_len, int mib0, int mib1) { /* Get sysctl data */
        memset(out, 0, sizeof(*out));
        *o_len = 0;
        int name[2] = {mib0, mib1};
        size_t len = 0;
        if (mag_unlikely(sysctl(name, sizeof(name) / sizeof(*name), NULL, &len, NULL, 0))) return false; /* Get length */
        if (mag_unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
        if (mag_unlikely(sysctl(name, sizeof(name) / sizeof(*name), *out, &len, NULL, 0))) return false; /* Get data */
        *o_len = len;
        return true;
    }
    static bool mag_sysctl_key(uint8_t (*out)[256], size_t* o_len, const char* key) { /* Get sysctl data */
        memset(out, 0, sizeof(*out));
        *o_len = 0;
        size_t len = 0;
        if (mag_unlikely(sysctlbyname(key, NULL, &len, NULL, 0))) return false; /* Get length */
        if (mag_unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
        if (mag_unlikely(sysctlbyname(key, *out, &len, NULL, 0))) return false; /* Get data */
        *o_len = len;
        return true;
    }
    static uint64_t mag_sysctl_unpack_int(const uint8_t (*in)[256], size_t len) { /* Unpack sysctl data */
        switch (len) {
            case sizeof(uint16_t): { uint16_t r; memcpy(&r, *in, sizeof(r)); return r; }
            case sizeof(uint32_t): { uint32_t r; memcpy(&r, *in, sizeof(r)); return r; }
            case sizeof(uint64_t): { uint64_t r; memcpy(&r, *in, sizeof(r)); return r; }
            default: return 0;
        }
    }
#else
    static bool mag_cpuinfo_parse_value(const char* key, char (*out)[128]) {
        FILE* cpuinfo = mag_fopen("/proc/cpuinfo", "rt");
        if (mag_unlikely(!cpuinfo)) return false;
        size_t key_len = strlen(key);
        char line[128];
        while (fgets(line, sizeof(line), cpuinfo)) {
            size_t line_len = strlen(line);
            if (line_len > 0 && line[line_len-1] == '\n') line[line_len-1] = '\0';
            if (strncmp(line, key, key_len) == 0 && (isspace((unsigned char)line[key_len]) || line[key_len] == ':')) {
                char* colon = strchr(line, ':');
                if (!colon) continue;
                char* value = colon+1;
                while (isspace((unsigned char)*value)) ++value;
                char* end = value + strlen(value);
                for (; end > value && isspace((unsigned char)*(end-1)); --end);
                *end = '\0';
                size_t value_len = llabs(end-value);
                if (mag_unlikely(!value_len || value_len >= sizeof(*out))) {
                    fclose(cpuinfo);
                    return false;
                }
                snprintf(*out, sizeof(*out), "%s", value);
                fclose(cpuinfo);
                return true;
            }
        }
        fclose(cpuinfo);
        return false;
    }
    static uint64_t mag_parse_meminfo_value(const char* line) {
        const char *p = strchr(line, ':');
        if (mag_unlikely(!p)) return 0;
        ++p;
        p += strspn(p, " \t");
        errno = 0;
        char* end;
        uint64_t value = strtoull(p, &end, 10);
        if (mag_unlikely(errno != 0 || p == end)) return 0;
        return value<<10;
    }
#endif

#ifdef __linux__
static void mag_trim_quotes(char* in) {
    if (in == NULL || *in == '\0') return;
    size_t len = strlen(in);
    if (in[len - 1] == '"') {
        in[len - 1] = '\0';
        len--;
    }
    if (in[0] == '"') {
        memmove(in, in + 1, len);
    }
}
#endif

static void mag_machine_probe_os_name(char (*out_os_name)[128]) { /* Get OS name */
    #ifdef _WIN32

    #elif defined(__APPLE__)
        size_t len;
        uint8_t tmp[256];
        if (mag_likely(mag_sysctl_mib01(&tmp, &len, CTL_KERN, KERN_VERSION) && len && *tmp)) {
            char* colon = strchr((const char*)tmp, ':');
            if (colon) *colon = '\0';
            snprintf(*out_os_name, sizeof(*out_os_name), "%s", (const char*)tmp);
        }
    #elif defined (__linux__)
        FILE* f = mag_fopen("/etc/os-release", "r");
        if (!f) {
            f = mag_fopen("/usr/lib/os-release", "r");
            if (!f) {
                f = mag_fopen("/etc/lsb-release", "r");
                if (mag_unlikely(!f)) return;
                char line[256];
                while (fgets(line, sizeof(line), f) != NULL) {
                    size_t len = strlen(line);
                    if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
                    if (strncmp(line, "DISTRIB_ID", sizeof("DISTRIB_ID")-1) == 0) {
                        char* equals_sign = strchr(line, '=');
                        if (equals_sign && *(equals_sign+1)) {
                            strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                            (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                        }
                    } else if (strncmp(line, "DISTRIB_DESCRIPTION", sizeof("DISTRIB_DESCRIPTION")-1) == 0) {
                        char* equals_sign = strchr(line, '=');
                        if (equals_sign && *(equals_sign+1)) {
                            char* start_quote = strchr(equals_sign+1, '"');
                            if (start_quote) {
                                char* end_quote = strchr(start_quote+1, '"');
                                if (end_quote) {
                                    size_t desc_len = end_quote-start_quote-1;
                                    if (desc_len >= sizeof(*out_os_name)) desc_len = sizeof(*out_os_name)-1;
                                    strncpy(*out_os_name, start_quote+1, desc_len);
                                    (*out_os_name)[desc_len] = '\0';
                                } else {
                                    strncpy(*out_os_name, start_quote+1, sizeof(*out_os_name)-1);
                                    (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                                }
                            } else {
                                strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
                            }
                        }
                    }
                }
                fclose(f);
                return;
            }
        }
    char line[256];
    while (fgets(line, sizeof(line), f) != NULL) {
        size_t len = strlen(line);
        if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
        if (strncmp(line, "NAME", sizeof("NAME")-1) == 0) {
            char* equals_sign = strchr(line, '=');
            if (equals_sign && *(equals_sign+1)) {
                strncpy(*out_os_name, equals_sign + 1, sizeof(*out_os_name)-1);
                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
            }
        } else if (strncmp(line, "PRETTY_NAME", sizeof("PRETTY_NAME")-1) == 0) {
            char* equals_sign = strchr(line, '=');
            if (equals_sign && *(equals_sign+1)) {
                strncpy(*out_os_name, equals_sign+1, sizeof(*out_os_name)-1);
                (*out_os_name)[sizeof(*out_os_name)-1] = '\0';
            }
        }
    }
    fclose(f);
    mag_trim_quotes(*out_os_name);
    #endif
}

static void mag_machine_probe_cpu_name(char (*out_cpu_name)[128]) { /* Get CPU name */
    #ifdef _WIN32
        HKEY key;
        if (mag_unlikely(RegOpenKeyExA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key))) return;
        char tmp[64+1] = {0};
        DWORD len = sizeof(tmp);
        if (mag_unlikely(RegQueryValueExA(key, "ProcessorNameString", NULL, NULL, (LPBYTE)tmp, &len))) return;
        if (mag_likely(strlen(tmp))) tmp[strlen(tmp)-1] = '\0';
        snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", tmp);
    #elif defined(__APPLE__)
        size_t len;
        uint8_t tmp[256];
        if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.brand_string") && len && *tmp))
            snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", (const char*)tmp);
    #else
        char cpu_name[128];
        if (mag_likely((mag_cpuinfo_parse_value("model name", &cpu_name) && *cpu_name) || (mag_cpuinfo_parse_value("Model", &cpu_name) && *cpu_name)))
            snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", cpu_name);
    #endif
}

static void mag_machine_probe_cpu_cores(uint32_t* out_virtual, uint32_t* out_physical, uint32_t* out_sockets) { /* Get CPU virtual (logical) cores. */
    #ifdef _WIN32
        DWORD size = 0;
        GetLogicalProcessorInformation(NULL, &size);
        if (mag_unlikely(!size)) return;
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION* info = (*mag_alloc)(NULL, size, 0);
        if (mag_unlikely(!GetLogicalProcessorInformation(info, &size))) goto end;
        for (DWORD i=0; i < size/sizeof(*info); ++i) {
            switch (info[i].Relationship) {
                default: continue;
                case RelationProcessorPackage: ++*out_sockets; continue;
                case RelationProcessorCore: {
                    ++*out_physical;
                    uintptr_t m = (uintptr_t)info[i].ProcessorMask;
                    m = m - ((m>>1) & 0x5555555555555555);
                    m = (m & 0x3333333333333333) + ((m>>2) & 0x3333333333333333);
                    *out_virtual += (((m + (m>>4)) & 0xf0f0f0f0f0f0f0f) * 0x101010101010101)>>56;
                } continue;
            }
        }
        end: (*mag_alloc)(info, 0, 0);
    #elif defined(__APPLE__)
        uint8_t tmp[256];
        size_t len;
        if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.thread_count") && len))
            *out_virtual = mag_sysctl_unpack_int(&tmp, len);
        if (mag_likely(mag_sysctl_key(&tmp, &len, "machdep.cpu.core_count") && len))
            *out_physical = mag_sysctl_unpack_int(&tmp, len);
        if (mag_likely(mag_sysctl_key(&tmp, &len, "hw.packages") && len))
            *out_sockets = mag_sysctl_unpack_int(&tmp, len);
    #else
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        FILE* cpuinfo = mag_fopen("/proc/cpuinfo", "r");
        if (mag_unlikely(!cpuinfo)) return;
        uint32_t physical_ids[MAG_MAX_CPUS];
        uint32_t core_ids[MAG_MAX_CPUS];
        uint32_t package_ids[MAG_MAX_CPUS];
        uint32_t cpu_count = 0;
        uint32_t package_count = 0;
        uint32_t current_physical_id = 0;
        uint32_t current_core_id = 0;
        bool got_physical_id = false;
        bool got_core_id = false;
        char line[256];
        while (fgets(line, sizeof(line), cpuinfo) != NULL) {
            if (strncmp(line, "physical id", sizeof("physical id")-1) == 0) {
                char* ptr = strchr(line, ':');
                if (ptr) {
                    ++ptr;
                    for (; *ptr && !isdigit((unsigned char)*ptr); ++ptr);
                    if (*ptr) { current_physical_id = (uint32_t)strtoul(ptr, NULL, 10); got_physical_id = true; }
                }
            } else if (strncmp(line, "core id", sizeof("core id")-1) == 0) {
                char* ptr = strchr(line, ':');
                if (ptr) {
                    ++ptr;
                    for (; *ptr && !isdigit((unsigned char)*ptr); ++ptr);
                    if (*ptr) { current_core_id = (uint32_t)strtoul(ptr, NULL, 10); got_core_id = true; }
                }
            } else if (*line == '\n') {
                if (got_physical_id && got_core_id) {
                    bool is_unique = true;
                    for (int32_t i=0; i < cpu_count; ++i) if (physical_ids[i] == current_physical_id && core_ids[i] == current_core_id) { is_unique = false; break; }
                    if (is_unique) {
                        if (cpu_count < MAG_MAX_CPUS) {
                            physical_ids[cpu_count] = current_physical_id;
                            core_ids[cpu_count] = current_core_id;
                            ++cpu_count;
                        } else break;
                    }
                    is_unique = true;
                    for (int32_t i=0; i < package_count; ++i) if (package_ids[i] == current_physical_id) { is_unique = false; break; }
                    if (is_unique) {
                        if (package_count < MAG_MAX_CPUS) package_ids[package_count++] = current_physical_id;
                        else break;
                    }
                }
                got_physical_id = false;
                got_core_id = false;
            }
        }
        fclose(cpuinfo);
        *out_virtual = nprocs > 0 ? (uint32_t)nprocs : 0;
        if (!cpu_count && *out_virtual) cpu_count = *out_virtual;
        *out_physical = mag_xmax(1, cpu_count);
        *out_virtual = nprocs > 0 ? (uint32_t)nprocs : *out_physical;
        *out_sockets = mag_xmax(1, package_count);
    #endif
}

static void mag_machine_probe_memory(size_t* out_phys_mem_total, size_t* out_phys_mem_free) { /* Get physical memory */
    #ifdef _WIN32
        MEMORYSTATUSEX mem;
        mem.dwLength = sizeof(mem);
        if (mag_likely(GlobalMemoryStatusEx(&mem))) {
            *out_phys_mem_total = mem.ullTotalPhys;
            *out_phys_mem_free = mem.ullAvailPhys;
        }
    #elif defined(__APPLE__)
        uint8_t tmp[256];
        size_t len;
        if (mag_likely(mag_sysctl_mib01(&tmp, &len, CTL_HW, HW_MEMSIZE) && len))
            *out_phys_mem_total = mag_sysctl_unpack_int(&tmp, len);
        struct vm_statistics64 stats;
        natural_t count = HOST_VM_INFO64_COUNT;
        if (mag_likely(host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)(&stats), &count) == KERN_SUCCESS))
            *out_phys_mem_free = stats.free_count * getpagesize();
    #else
        FILE* meminfo = mag_fopen("/proc/meminfo", "r");
        if (mag_unlikely(!meminfo)) return;
        char line[256];
        while (fgets(line, sizeof(line), meminfo)) {
            if (strncmp(line, "MemTotal:", sizeof("MemTotal:")-1) == 0)
                *out_phys_mem_total = mag_parse_meminfo_value(line);
            else if (strncmp(line, "MemAvailable:", sizeof("MemAvailable:")-1) == 0)
                *out_phys_mem_free = mag_parse_meminfo_value(line);
        }
        fclose(meminfo);
    #endif
}

#if defined(__x86_64__) || defined(_M_X64)
    static void mag_cpuid_ex(uint32_t (*o)[4], uint32_t eax, uint32_t ecx) {
        #ifdef _WIN32
            __cpuidex((int*)(*o), eaxIn, ecxIn);
        #else
            __cpuid_count(eax, ecx, (*o)[0], (*o)[1], (*o)[2], (*o)[3]);
        #endif
    }
    static void mag_cpuid(uint32_t (*o)[4], uint32_t eax) {
        mag_cpuid_ex(o, eax, 0);
    }
    static bool mag_cpuid_streq(uint32_t ebx, uint32_t ecx, uint32_t edx, const char str[static 12]) {
        #define mag_strbe(x) ((x)[0] | ((x)[1]<<8) | ((x)[2]<<16) | ((x)[3]<<24))
        return ebx == mag_strbe(str) && edx == mag_strbe(str+4) && ecx == mag_strbe(str+8);
        #undef mag_strbe
    }
    static uint64_t MAG_AINLINE mag_xgetbv(void) { /* Query extended control register value. */
        #ifdef _MSC_VER
            return _xgetbv(0);
        #else
            uint32_t eax, edx;
            __asm__ volatile(".byte 0x0f,0x01,0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
            return (uint64_t)edx<<32 | eax;
        #endif
    }
    static void mag_probe_cpu_amd64(mag_amd64_cap_bitset_t* o, uint32_t* avx10ver) {
        memset(o, 0, sizeof(*o));
        uint32_t id[4] = {0};
        const uint32_t* eax = id+0, *ebx = id+1, *ecx = id+2, *edx = id+3;
        mag_cpuid(&id, 0);
        uint32_t max = *eax;
        if (mag_cpuid_streq(*ebx, *ecx, *edx, "AuthenticAMD")) *o|=mag_amd64_cap(AMD);
        else if (mag_cpuid_streq(*ebx, *ecx, *edx, "GenuineIntel")) *o|=mag_amd64_cap(INTEL);
        mag_cpuid(&id, 0x80000000);
        #define mag_captest(reg, bit, cp) if (*(reg)&(1u<<((bit)&31))) *o|=mag_amd64_cap(cp)
        uint32_t max_ex = *eax;
        if (max_ex >= 0x80000001) {
            mag_cpuid(&id, 0x80000001);
            mag_captest(ecx, 0, SSE4A);
        }
        mag_cpuid(&id, 1);
        mag_captest(ecx, 0, SSE3);
        mag_captest(ecx, 9, SSSE3);
        mag_captest(ecx, 19, SSE41);
        mag_captest(ecx, 20, SSE42);
        mag_captest(ecx, 27, OSXSAVE);
        mag_captest(ecx, 29, F16C);
        mag_captest(edx, 25, SSE);
        mag_captest(edx, 26, SSE2);
        if (*o & mag_amd64_cap(OSXSAVE)) {
            uint64_t cr = mag_xgetbv();
            if ((cr&6) == 6) {
                mag_captest(ecx, 12, FMA);
                mag_captest(ecx, 28, AVX);
                if (((cr>>5)&7) == 7) {
                    mag_cpuid_ex(&id, 7, 0);
                    mag_captest(ebx, 16, AVX512_F);
                    if (*o & mag_amd64_cap(AVX512_F)) {
                        mag_captest(ebx, 17, AVX512_DQ);
                        mag_captest(ebx, 21, AVX512_IFMA);
                        mag_captest(ebx, 26, AVX512_PF);
                        mag_captest(ebx, 27, AVX512_ER);
                        mag_captest(ebx, 28, AVX512_CD);
                        mag_captest(ebx, 30, AVX512_BW);
                        mag_captest(ebx, 31, AVX512_VL);
                        mag_captest(ecx, 1, AVX512_VBMI);
                        mag_captest(ecx, 6, AVX512_VBMI2);
                        mag_captest(ecx, 11, AVX512_VNNI);
                        mag_captest(ecx, 12, AVX512_BITALG);
                        mag_captest(ecx, 14, AVX512_VPOPCNTDQ);
                        mag_captest(edx, 2, AVX512_4VNNIW);
                        mag_captest(edx, 3, AVX512_4FMAPS);
                        mag_captest(edx, 8, AVX512_VP2INTERSECT);
                        if (*o & mag_amd64_cap(AVX512_BW))
                            mag_captest(edx, 23, AVX512_FP16);
                    }
                }
            }
        }
        if (max >= 7) {
            mag_cpuid_ex(&id, 7, 0);
            uint32_t max_sub = *eax;
            if (*o & mag_amd64_cap(AVX) && (*ebx & 1u<<5))
                *o |= mag_amd64_cap(AVX2);
            mag_captest(ebx, 3, BMI1);
            mag_captest(ebx, 8, BMI2);
            mag_captest(ecx, 8, GFNI);
            mag_captest(edx, 22, AMX_BF16);
            mag_captest(edx, 24, AMX_TILE);
            mag_captest(edx, 25, AMX_INT8);
            if (max_sub >= 1) {
                mag_cpuid_ex(&id, 7, 1);
                mag_captest(eax, 4, AVX_VNNI);
                if (*o & mag_amd64_cap(AVX512_F))
                    mag_captest(eax, 5, AVX512_BF16);
                mag_captest(edx, 22, AMX_FP16);
                mag_captest(edx, 4, AVX_VNNI_INT8);
                mag_captest(edx, 5, AVX_NE_CONVERT);
                mag_captest(edx, 10, AVX_VNNI_INT16);
                mag_captest(edx, 19, AVX10);
                mag_captest(edx, 21, APX_F);
                mag_cpuid_ex(&id, 0x1e, 1);
                mag_captest(eax, 4, AMX_FP8);
                mag_captest(eax, 5, AMX_TRANSPOSE);
                mag_captest(eax, 6, AMX_TF32);
                mag_captest(eax, 7, AMX_AVX512);
                mag_captest(eax, 8, AMX_MOVRS);
            }
        }
        #undef mag_captest
        if (*o & mag_amd64_cap(AVX10)) {
            mag_cpuid_ex(&id, 0x24, 0);
            *avx10ver = *ebx & 127;
        }
    }

    #define mag_bextract(x, b, e) (((x)>>(b))&((1u<<((e)+1-(b)))-1))

    typedef enum mag_cpu_topology_level {
        MAG_CPU_TOPO_STMT = 1,
        MAG_CPU_TOPO_CORE = 2
    } mag_cpu_topology_level;

    static void mag_probe_cpu_core_topology(mag_amd64_cap_bitset_t caps, uint32_t (*num_cores)[MAG_MAX_CPU_TOPO_DEPTH]) {
		uint32_t id[4] = {0};
		mag_cpuid(&id, 0x0);
		if (*id >= 0xB) {
			mag_cpuid_ex(&id, 0xb, 0);
			if (*id || id[1]) {
				for (uint32_t i=0; i < MAG_MAX_CPU_TOPO_DEPTH; ++i) {
					mag_cpuid_ex(&id, 0xb, i);
					mag_cpu_topology_level level = (mag_cpu_topology_level)mag_bextract(id[2], 8, 15);
					if (level == MAG_CPU_TOPO_STMT || level == MAG_CPU_TOPO_CORE)
						(*num_cores)[level-1] = mag_bextract(id[1], 0, 15);
				}
				(*num_cores)[MAG_CPU_TOPO_STMT-1] = mag_xmax(1u, (*num_cores)[MAG_CPU_TOPO_STMT-1]);
				(*num_cores)[MAG_CPU_TOPO_CORE-1] = mag_xmax((*num_cores)[MAG_CPU_TOPO_STMT-1], (*num_cores)[MAG_CPU_TOPO_CORE-1]);
				return;
			}
		}
		if (caps & mag_amd64_cap(AMD)) {
			int32_t ptc = 0;
			mag_cpuid(&id, 0x1);
			int32_t ltc = mag_bextract(id[1], 16, 23);
			int32_t htn = mag_bextract(id[3], 28, 28);
			mag_cpuid(&id, 0x80000000);
			uint32_t max_leaf = *id;
			if (max_leaf >= 0x80000008) {
				mag_cpuid(&id, 0x80000008);
				ptc = mag_bextract(id[2], 0, 7) + 1;
			}
			if (!htn) {
				(*num_cores)[MAG_CPU_TOPO_STMT-1] = 1;
				(*num_cores)[MAG_CPU_TOPO_CORE-1] = 1;
			} else if (ptc > 1) {
			    mag_cpuid(&id, 1);
			    int32_t fam_ext = mag_bextract(*id, 20, 27);
			    int32_t fam = mag_bextract(*id, 8, 11);
			    int32_t dis_fam = fam;
			    if (dis_fam == 0x0f) dis_fam += fam_ext;
				if (dis_fam >= 0x17 && max_leaf >= 0x8000001e) {
					mag_cpuid(&id, 0x8000001e);
					ptc /= mag_bextract(id[1], 8, 15)+1;
				}
				(*num_cores)[MAG_CPU_TOPO_STMT-1] = ltc/ptc;
				(*num_cores)[MAG_CPU_TOPO_CORE-1] = ltc;
			} else {
				(*num_cores)[MAG_CPU_TOPO_STMT-1] = 1;
				(*num_cores)[MAG_CPU_TOPO_CORE-1] = ltc > 1 ? ltc : 2;
			}
		} else if (caps & mag_amd64_cap(INTEL)) {
			int32_t ptc = 0;
			mag_cpuid(&id, 0x1);
			int32_t lpc = mag_bextract(id[1], 16, 23);
			int32_t htt = mag_bextract(id[3], 28, 28);
			mag_cpuid(&id, 0);
			if (*id >= 0x4) {
				mag_cpuid(&id, 0x4);
				ptc = mag_bextract(id[0], 26, 31)+1;
			}
			if (!htt) {
				(*num_cores)[MAG_CPU_TOPO_STMT-1] = 1;
				(*num_cores)[MAG_CPU_TOPO_CORE-1] = 1;
			} else if (ptc > 1) {
				(*num_cores)[MAG_CPU_TOPO_STMT-1] = lpc/ptc;
				(*num_cores)[MAG_CPU_TOPO_CORE-1] = lpc;
			} else {
				(*num_cores)[MAG_CPU_TOPO_STMT-1] = 1;
				(*num_cores)[MAG_CPU_TOPO_CORE-1] = lpc > 0 ? lpc : 1;
			}
		}
    }

    static void mag_probe_cpu_cache_topology(mag_amd64_cap_bitset_t caps, size_t* ol1, size_t* ol2, size_t* ol3) {
        uint32_t levels = 0;
        uint32_t data_cache[MAG_MAX_CPU_CACHE_DEPTH] = {0};
        uint32_t shared_cache[MAG_MAX_CPU_CACHE_DEPTH] = {0};
        uint32_t num_cores[MAG_MAX_CPU_TOPO_DEPTH] = {0};
        mag_probe_cpu_core_topology(caps, &num_cores);
        uint32_t id[4] = {0};
        if (caps & mag_amd64_cap(AMD)) {
            mag_cpuid(&id, 0x80000000);
            if (*id >= 0x8000001d) {
                levels = 0;
                for (uint32_t leaf=0; levels < MAG_MAX_CPU_CACHE_DEPTH; ++leaf) {
                    mag_cpuid_ex(&id, 0x8000001d, leaf);
                    int32_t type = mag_bextract(*id, 0, 4);
                    if (!type) break;
                    if (type == 0x2) continue;
                    int32_t assoc = mag_bextract(*id, 9, 9);
                    int32_t sharing = mag_bextract(*id, 14, 25)+1;
                    int32_t ways = mag_bextract(id[1], 22, 31)+1;
                    int32_t partitions = mag_bextract(id[1], 12, 21)+1;
                    int32_t line = mag_bextract(id[1], 0, 11)+1;
                    int32_t sets = id[2]+1;
                    data_cache[levels] = line*partitions*ways;
                    if (!assoc) data_cache[levels] *= sets;
                    if (leaf > 0) {
                        sharing = mag_xmin(sharing, num_cores[1]);
                        sharing /= mag_xmax(1u, *shared_cache);
                    }
                    shared_cache[levels] = sharing;
                    ++levels;
                }
                *shared_cache = mag_xmin(1u, *shared_cache);
            } else if (*id >= 0x80000006) {
                levels = 1;
                mag_cpuid(&id, 0x80000005);
                int32_t l1dc = mag_bextract(id[2], 24, 31);
                *data_cache = l1dc<<10;
                *shared_cache = 1;
                mag_cpuid(&id, 0x80000006);
                int32_t l2 = mag_bextract(id[2], 12, 15);
                if (l2 > 0) {
                    levels = 2;
                    int32_t l2s = mag_bextract(id[2], 16, 31);
                    data_cache[1] = l2s<<10;
                    shared_cache[1] = 1;
                }
                int32_t l3 = mag_bextract(id[3], 12, 15);
                if (l3 > 0) {
                    levels = 3;
                    int32_t l3s = mag_bextract(id[3], 18, 31);
                    data_cache[2] = l3s<<19;
                    shared_cache[2] = num_cores[1];
                }
            }
        } else if (caps & mag_amd64_cap(INTEL)) {
            uint32_t smt_width = *num_cores;
            uint32_t logical_cores = num_cores[1];
            for (uint32_t i=0; levels < MAG_MAX_CPU_CACHE_DEPTH; ++i) {
                mag_cpuid_ex(&id, 0x4, i);
                uint32_t type = mag_bextract(*id, 0, 4);
                if (!type) break;
                if (type == 1 || type == 3) {
                    uint32_t actual_logical_cores = mag_bextract(*id, 14, 25)+1;
                    if (logical_cores != 0) actual_logical_cores = mag_xmin(actual_logical_cores, logical_cores);
                    mag_assert2(actual_logical_cores);
                    data_cache[levels] =
                        (mag_bextract(id[1], 22, 31)+1)
                        * (mag_bextract(id[1], 12, 21)+1)
                        * (mag_bextract(id[1], 0, 11)+1)
                        * (id[2]+1);
                    if (type == 1 && smt_width == 0) smt_width = actual_logical_cores;
                    mag_assert2(smt_width != 0);
                    shared_cache[levels] = mag_xmax(actual_logical_cores / smt_width, 1u);
                    ++levels;
                }
            }
        }
        if (levels) {
            *ol1 = data_cache[0]/shared_cache[0];
            *ol2 = data_cache[1]/shared_cache[1];
            *ol3 = data_cache[2]/shared_cache[2];
        } else {
            *ol1 = 32ull<<10;
            *ol2 = 512ull<<10;
            *ol3 = 1024ull<<10;
        }
    }

    #undef mag_bextract

#elif defined(__aarch64__) || defined(_M_ARM64)

    static void mag_probe_cpu_arm64(mag_arm64_cap_bitset_t* o, int64_t* sve_width) {
        *o = MAG_ARM64_CAP_NONE;
        #ifdef __linux__
            unsigned long hwcap = getauxval(AT_HWCAP);
            unsigned long hwcap2 = getauxval(AT_HWCAP2);
            (void)hwcap2;
            *o|=mag_arm64_cap(NEON); /* NEON is always required by build */
            #ifdef HWCAP_ASIMD
                if (hwcap & HWCAP_ASIMD) *o|=mag_arm64_cap(NEON);
            #endif
            #ifdef HWCAP_ASIMDDP
                if (hwcap & HWCAP_ASIMDDP) *o|=mag_arm64_cap(DOTPROD);
            #endif
            #ifdef HWCAP2_I8MM
                if (hwcap2 & HWCAP2_I8MM) *o|=mag_arm64_cap(I8MM);
            #endif
            #ifdef HWCAP_FPHP
                if (hwcap & HWCAP_FPHP) *o|=mag_arm64_cap(F16SCA);
            #endif
            #ifdef HWCAP_ASIMDHP
                if (hwcap & HWCAP_ASIMDHP) *o|=mag_arm64_cap(F16VEC);
            #endif
            #ifdef HWCAP2_BF16
                if (hwcap2 & HWCAP2_BF16) *o|=mag_arm64_cap(BF16);
            #endif
            #ifdef HWCAP_SVE
                if (hwcap & HWCAP_SVE) *o|=mag_arm64_cap(SVE);
            #endif
            #ifdef HWCAP2_SVE2
                if (hwcap2 & HWCAP2_SVE2) *o|=mag_arm64_cap(SVE2);
            #endif
            *sve_width = 0; /* NYI */
        #elif defined(_WIN32)
            if (IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE)) *o|=mag_arm64_cap(NEON);
            if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)) *o|=mag_arm64_cap(DOTPROD);
            /* Other features not supported by IsProcessorFeaturePresent*/
            *sve_width = 0; /* NYI */
        #elif defined(__APPLE__)
            *o|=mag_arm64_cap(NEON); /* NEON is always required by build */
            int sx = 0;
            size_t size = sizeof(sx);
            if (sysctlbyname("hw.optional.AdvSIMD", &sx, &size, NULL, 0) != 0) sx = 0;
            if (sx) *o|=mag_arm64_cap(NEON);
            if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &sx, &size, NULL, 0) != 0) sx = 0;
            if (sx) *o|=mag_arm64_cap(DOTPROD);
            if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &sx, &size, NULL, 0) != 0) sx = 0;
            if (sx) *o|=mag_arm64_cap(I8MM);
            if (sysctlbyname("hw.optional.arm.FEAT_FP16", &sx, &size, NULL, 0) != 0) sx = 0;
            if (sx) *o|=mag_arm64_cap(F16SCA);
            if (sysctlbyname("hw.optional.AdvSIMD_HPFPCvt", &sx, &size, NULL, 0) != 0) sx = 0;
            if (sx) *o|=mag_arm64_cap(F16VEC);
            if (sysctlbyname("hw.optional.arm.FEAT_BF16", &sx, &size, NULL, 0) != 0) sx = 0;
            if (sx) *o|=mag_arm64_cap(BF16);
            if (sysctlbyname("hw.optional.arm.FEAT_SVE", &sx, &size, NULL, 0) != 0) sx = 0;
            if (sx) *o|=mag_arm64_cap(SVE);
            *sve_width = 0; /* NYI */
        #endif
    }

    static void mag_probe_cpu_cache_topology(mag_arm64_cap_bitset_t caps, size_t* ol1, size_t* ol2, size_t* ol3) {
        #ifdef __APPLE__
            size_t sz;
            uint64_t v = 0;
            uint32_t pcores = 0;
            sz = sizeof(pcores);
            if (sysctlbyname("hw.perflevel0.physicalcpu", &pcores, &sz, NULL, 0) != 0) {
                sz = sizeof(pcores);
                if (sysctlbyname("hw.physicalcpu", &pcores, &sz, NULL, 0) != 0) pcores = 1;
            }
            sz = sizeof(v);
            if (sysctlbyname("hw.perflevel0.l1dcachesize", &v, &sz, NULL, 0) == 0) *ol1 = v;
            else if (sysctlbyname("hw.l1dcachesize", &v, &sz, NULL, 0) == 0) *ol1 = v;
            else *ol1 = 128ull<<10;
            sz = sizeof(v);
            if (sysctlbyname("hw.perflevel0.l2cachesize", &v, &sz, NULL, 0) == 0) *ol2 = v / (pcores*2);
            else if (sysctlbyname("hw.l2cachesize", &v, &sz, NULL, 0) == 0) *ol2 = v / (pcores*2);
            else *ol2 = 3ull<<20;
            sz = sizeof(v);
            if (sysctlbyname("hw.perflevel0.l3cachesize", &v, &sz, NULL, 0) == 0) *ol3 = v;
            else if (sysctlbyname("hw.l3cachesize", &v, &sz, NULL, 0) == 0) *ol3 = v;
            else *ol3 = 64ull<<20;
        #else
            *ol1 = 32ull<<10;
            *ol2 = 512ull<<10;
            *ol3 = 1024ull<<10;
        #endif
    }

#endif

static void mag_machine_probe(mag_context_t* ctx) {
    mag_machine_probe_os_name(&ctx->machine.os_name);
    mag_machine_probe_cpu_name(&ctx->machine.cpu_name);
    mag_machine_probe_cpu_cores(&ctx->machine.cpu_virtual_cores, &ctx->machine.cpu_physical_cores, &ctx->machine.cpu_sockets);
    mag_machine_probe_memory(&ctx->machine.phys_mem_total, &ctx->machine.phys_mem_free);
    uint64_t caps = 0;
    #if defined(__x86_64__) || defined(_M_X64)
        mag_probe_cpu_amd64(&ctx->machine.amd64_cpu_caps, &ctx->machine.amd64_avx10_ver);
        caps = ctx->machine.amd64_cpu_caps;
    #elif defined(__aarch64__)
        mag_probe_cpu_arm64(&ctx->machine.arm64_cpu_caps, &ctx->machine.arm64_cpu_sve_width);
        caps = ctx->machine.arm64_cpu_caps;
    #endif
    mag_probe_cpu_cache_topology(caps, &ctx->machine.cpu_l1_size, &ctx->machine.cpu_l2_size, &ctx->machine.cpu_l3_size);
    if (mag_unlikely(!*ctx->machine.os_name)) snprintf(ctx->machine.os_name, sizeof(ctx->machine.os_name), "Unknown");
    if (mag_unlikely(!*ctx->machine.cpu_name)) snprintf(ctx->machine.cpu_name, sizeof(ctx->machine.cpu_name), "Unknown");
}

static MAG_COLDPROC void mag_graphviz_dump(const mag_tensor_t* node, FILE *fp, mag_hashset_t* visited) {
    if (!node->au_state) return;
    if (mag_hashset_contains_key(visited, node)) return;
    mag_hashset_insert(visited, node);
    bool is_input = true;
    for (unsigned i=0; i < MAG_MAX_OP_INPUTS; ++i) {
        if (node->au_state->op_inputs[i] != NULL) {
            is_input = false;
            break;
        }
    }
    const char* fillcolor = is_input ? "palegreen" : "skyblue2";
    char dim_buf[150];
    mag_fmt_shape(&dim_buf, &node->shape, node->rank);
    bool gra = node->flags & MAG_TFLAG_REQUIRES_GRAD;
    fprintf(
        fp,
        "  \"%p\" [label=\"⊕ %s|∇ %s|%s|0x%x\", shape=record, style=\"rounded,filled\", fillcolor=%s];\n",
        (void*)node,
        mag_op_meta_of(node->au_state->op)->mnemonic,\
        gra ? "✓" : "🗙",
        dim_buf,
        node->flags,
        fillcolor
    );
    for (unsigned i=0; i < MAG_MAX_OP_INPUTS; ++i) {
        mag_tensor_t* input = node->au_state->op_inputs[i];
        if (!input) continue;
        char name[128];
        snprintf(name, sizeof(name), " in %u", i);
        fprintf(fp, "  \"%p\" -> \"%p\" [label=\"%s\"];\n", (void*)input, (void*)node, name);
        mag_graphviz_dump(input, fp, visited);
    }
}

MAG_COLDPROC void mag_tensor_export_forward_graph_graphviz(mag_tensor_t* t, const char* file) {
    mag_assert2(t && file && *file);
    FILE* f = mag_fopen(file, "w");
    fprintf(f, "digraph computation_graph {\n");
    fprintf(f, "  rankdir=TD;\n");
    fprintf(f, "  node [fontname=\"Helvetica\", shape=box];\n");
    fprintf(f, "  edge [fontname=\"Helvetica\"];\n");
    mag_hashset_t visited = mag_hashset_init(0xffff);
    mag_graphviz_dump(t, f, &visited);
    mag_hashset_free(&visited);
    fprintf(f, "}\n");
    fclose(f);
}

MAG_COLDPROC void mag_tensor_export_backward_graph_graphviz(mag_tensor_t* t, const char* file) {
    mag_tensor_set_t post_order;
    mag_tensor_array_init(&post_order);
    mag_collect_topo_iterative(t, &post_order);
    for (size_t i=0, j=post_order.size - 1; i < j; ++i, --j) {
        mag_swap(mag_tensor_t*, post_order.data[i], post_order.data[j]);
    }
    FILE* fp = mag_fopen(file, "wt");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing the graphviz output.\n");
        return;
    }
    fprintf(fp, "digraph backward_graph {\n");
    fprintf(fp, "    rankdir=TD;\n");
    fprintf(fp, "    node [shape=record, style=\"rounded,filled\", fontname=\"Helvetica\"];\n");
    for (size_t i=0; i < post_order.size; ++i) {
        mag_tensor_t* node = post_order.data[i];
        if (!node->au_state) continue;
        const mag_opmeta_t* meta = mag_op_meta_of(node->au_state->op);
        fprintf(fp, "    \"%p\" [label=\"%s\\nShape: (", node, meta->mnemonic);
        for (int64_t r=0; r < node->rank; ++r) {
            fprintf(fp, "%zu", (size_t)node->shape[r]);
            if (r < node->rank - 1)
                fprintf(fp, ", ");
        }
        fprintf(fp, ")\\nGrad: %s\"];\n", node->au_state->grad ? "set" : "none");
    }
    for (size_t i=0; i < post_order.size; ++i) {
        mag_tensor_t* node = post_order.data[i];
        const mag_opmeta_t* meta = mag_op_meta_of(node->au_state->op);
        for (uint32_t j = 0; j < meta->in; ++j) {
            mag_tensor_t* input = node->au_state->op_inputs[j];
            if (input) {
                fprintf(fp, "    \"%p\" -> \"%p\" [label=\"input %u\"];\n", node, input, j);
            }
        }
    }
    fprintf(fp, "}\n");
    fclose(fp);
    mag_tensor_array_free(&post_order);
}

uint64_t mag_hash(const void* key, size_t len, uint32_t seed) {
    #define	mag_rol32(x, r) (((x)<<(r))|((x)>>(32-(r))))
    #define mag_mix32(h) h^=h>>16; h*=0x85ebca6b; h^=h>>13; h*=0xc2b2ae35; h^=h>>16;
    const uint8_t* p = key;
    int64_t nblocks = (int64_t)len>>4;
    uint32_t h1 = seed;
    uint32_t h2 = seed;
    uint32_t h3 = seed;
    uint32_t h4 = seed;
    uint32_t c1 = 0x239b961b;
    uint32_t c2 = 0xab0e9789;
    uint32_t c3 = 0x38b34ae5;
    uint32_t c4 = 0xa1e38b93;
    const uint32_t * blocks = (const uint32_t *)(p + nblocks*16);
    for (int64_t i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i*4+0];
        uint32_t k2 = blocks[i*4+1];
        uint32_t k3 = blocks[i*4+2];
        uint32_t k4 = blocks[i*4+3];
        k1 *= c1; k1  = mag_rol32(k1,15); k1 *= c2; h1 ^= k1;
        h1 = mag_rol32(h1,19); h1 += h2; h1 = h1*5+0x561ccd1b;
        k2 *= c2; k2  = mag_rol32(k2,16); k2 *= c3; h2 ^= k2;
        h2 = mag_rol32(h2,17); h2 += h3; h2 = h2*5+0x0bcaa747;
        k3 *= c3; k3  = mag_rol32(k3,17); k3 *= c4; h3 ^= k3;
        h3 = mag_rol32(h3,15); h3 += h4; h3 = h3*5+0x96cd1c35;
        k4 *= c4; k4  = mag_rol32(k4,18); k4 *= c1; h4 ^= k4;
        h4 = mag_rol32(h4,13); h4 += h1; h4 = h4*5+0x32ac3b17;
    }
    const uint8_t * tail = (const uint8_t*)(p + nblocks*16);
    uint32_t k1 = 0;
    uint32_t k2 = 0;
    uint32_t k3 = 0;
    uint32_t k4 = 0;
    switch(len&15) {
        case 15: k4 ^= tail[14] << 16;
        case 14: k4 ^= tail[13] << 8;
        case 13: k4 ^= tail[12] << 0;
            k4 *= c4;
            k4 = mag_rol32(k4,18);
            k4 *= c1;
            h4 ^= k4;
        case 12: k3 ^= tail[11] << 24;
        case 11: k3 ^= tail[10] << 16;
        case 10: k3 ^= tail[9] << 8;
        case 9: k3 ^= tail[8] << 0;
            k3 *= c3;
            k3 = mag_rol32(k3,17);
            k3 *= c4;
            h3 ^= k3;
        case 8: k2 ^= tail[7] << 24;
        case 7: k2 ^= tail[6] << 16;
        case 6: k2 ^= tail[5] << 8;
        case 5: k2 ^= tail[4] << 0;
            k2 *= c2;
            k2 = mag_rol32(k2,16);
            k2 *= c3;
            h2 ^= k2;
        case 4: k1 ^= tail[3] << 24;
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0] << 0;
            k1 *= c1;
            k1 = mag_rol32(k1,15);
            k1 *= c2;
            h1 ^= k1;
    };
    h1 ^= len; h2 ^= len; h3 ^= len; h4 ^= len;
    h1 += h2; h1 += h3; h1 += h4;
    h2 += h1; h3 += h1; h4 += h1;
    mag_mix32(h1); mag_mix32(h2); mag_mix32(h3); mag_mix32(h4);
    h1 += h2; h1 += h3; h1 += h4;
    h2 += h1; h3 += h1; h4 += h1;
    return (((uint64_t)h2)<<32)|h1;
    #undef mag_rol32
    #undef mag_mix32
}

uint32_t mag_crc32c(const void* buffer, size_t size) {
    if (mag_unlikely(!buffer || !size)) return 0;
    const uint8_t* buf = buffer;
    static const uint32_t crc_lut[256] = {
        0x00000000, 0xf26b8303, 0xe13b70f7, 0x1350f3f4, 0xc79a971f, 0x35f1141c,
        0x26a1e7e8, 0xd4ca64eb, 0x8ad958cf, 0x78b2dbcc, 0x6be22838, 0x9989ab3b,
        0x4d43cfd0, 0xbf284cd3, 0xac78bf27, 0x5e133c24, 0x105ec76f, 0xe235446c,
        0xf165b798, 0x030e349b, 0xd7c45070, 0x25afd373, 0x36ff2087, 0xc494a384,
        0x9a879fa0, 0x68ec1ca3, 0x7bbcef57, 0x89d76c54, 0x5d1d08bf, 0xaf768bbc,
        0xbc267848, 0x4e4dfb4b, 0x20bd8ede, 0xd2d60ddd, 0xc186fe29, 0x33ed7d2a,
        0xe72719c1, 0x154c9ac2, 0x061c6936, 0xf477ea35, 0xaa64d611, 0x580f5512,
        0x4b5fa6e6, 0xb93425e5, 0x6dfe410e, 0x9f95c20d, 0x8cc531f9, 0x7eaeb2fa,
        0x30e349b1, 0xc288cab2, 0xd1d83946, 0x23b3ba45, 0xf779deae, 0x05125dad,
        0x1642ae59, 0xe4292d5a, 0xba3a117e, 0x4851927d, 0x5b016189, 0xa96ae28a,
        0x7da08661, 0x8fcb0562, 0x9c9bf696, 0x6ef07595, 0x417b1dbc, 0xb3109ebf,
        0xa0406d4b, 0x522bee48, 0x86e18aa3, 0x748a09a0, 0x67dafa54, 0x95b17957,
        0xcba24573, 0x39c9c670, 0x2a993584, 0xd8f2b687, 0x0c38d26c, 0xfe53516f,
        0xed03a29b, 0x1f682198, 0x5125dad3, 0xa34e59d0, 0xb01eaa24, 0x42752927,
        0x96bf4dcc, 0x64d4cecf, 0x77843d3b, 0x85efbe38, 0xdbfc821c, 0x2997011f,
        0x3ac7f2eb, 0xc8ac71e8, 0x1c661503, 0xee0d9600, 0xfd5d65f4, 0x0f36e6f7,
        0x61c69362, 0x93ad1061, 0x80fde395, 0x72966096, 0xa65c047d, 0x5437877e,
        0x4767748a, 0xb50cf789, 0xeb1fcbad, 0x197448ae, 0x0a24bb5a, 0xf84f3859,
        0x2c855cb2, 0xdeeedfb1, 0xcdbe2c45, 0x3fd5af46, 0x7198540d, 0x83f3d70e,
        0x90a324fa, 0x62c8a7f9, 0xb602c312, 0x44694011, 0x5739b3e5, 0xa55230e6,
        0xfb410cc2, 0x092a8fc1, 0x1a7a7c35, 0xe811ff36, 0x3cdb9bdd, 0xceb018de,
        0xdde0eb2a, 0x2f8b6829, 0x82f63b78, 0x709db87b, 0x63cd4b8f, 0x91a6c88c,
        0x456cac67, 0xb7072f64, 0xa457dc90, 0x563c5f93, 0x082f63b7, 0xfa44e0b4,
        0xe9141340, 0x1b7f9043, 0xcfb5f4a8, 0x3dde77ab, 0x2e8e845f, 0xdce5075c,
        0x92a8fc17, 0x60c37f14, 0x73938ce0, 0x81f80fe3, 0x55326b08, 0xa759e80b,
        0xb4091bff, 0x466298fc, 0x1871a4d8, 0xea1a27db, 0xf94ad42f, 0x0b21572c,
        0xdfeb33c7, 0x2d80b0c4, 0x3ed04330, 0xccbbc033, 0xa24bb5a6, 0x502036a5,
        0x4370c551, 0xb11b4652, 0x65d122b9, 0x97baa1ba, 0x84ea524e, 0x7681d14d,
        0x2892ed69, 0xdaf96e6a, 0xc9a99d9e, 0x3bc21e9d, 0xef087a76, 0x1d63f975,
        0x0e330a81, 0xfc588982, 0xb21572c9, 0x407ef1ca, 0x532e023e, 0xa145813d,
        0x758fe5d6, 0x87e466d5, 0x94b49521, 0x66df1622, 0x38cc2a06, 0xcaa7a905,
        0xd9f75af1, 0x2b9cd9f2, 0xff56bd19, 0x0d3d3e1a, 0x1e6dcdee, 0xec064eed,
        0xc38d26c4, 0x31e6a5c7, 0x22b65633, 0xd0ddd530, 0x0417b1db, 0xf67c32d8,
        0xe52cc12c, 0x1747422f, 0x49547e0b, 0xbb3ffd08, 0xa86f0efc, 0x5a048dff,
        0x8ecee914, 0x7ca56a17, 0x6ff599e3, 0x9d9e1ae0, 0xd3d3e1ab, 0x21b862a8,
        0x32e8915c, 0xc083125f, 0x144976b4, 0xe622f5b7, 0xf5720643, 0x07198540,
        0x590ab964, 0xab613a67, 0xb831c993, 0x4a5a4a90, 0x9e902e7b, 0x6cfbad78,
        0x7fab5e8c, 0x8dc0dd8f, 0xe330a81a, 0x115b2b19, 0x020bd8ed, 0xf0605bee,
        0x24aa3f05, 0xd6c1bc06, 0xc5914ff2, 0x37faccf1, 0x69e9f0d5, 0x9b8273d6,
        0x88d28022, 0x7ab90321, 0xae7367ca, 0x5c18e4c9, 0x4f48173d, 0xbd23943e,
        0xf36e6f75, 0x0105ec76, 0x12551f82, 0xe03e9c81, 0x34f4f86a, 0xc69f7b69,
        0xd5cf889d, 0x27a40b9e, 0x79b737ba, 0x8bdcb4b9, 0x988c474d, 0x6ae7c44e,
        0xbe2da0a5, 0x4c4623a6, 0x5f16d052, 0xad7d5351
    };
    uint32_t crc = ~0u;
    for (size_t i=0; i < size; ++i)
        crc = (crc>>8) ^ crc_lut[buf[i] ^ (crc&0xff)];
    return ~crc;
}

bool mag_utf8_validate(const char* str, size_t len) {
    const uint8_t* data = (const uint8_t*)str;
    size_t pos = 0;
    uint32_t cp = 0;
    while (pos < len) {
        uint64_t next_pos = pos+16;
        if (next_pos <= len) {
            uint64_t v1, v2;
            memcpy(&v1, data+pos, sizeof(v1));
            memcpy(&v2, data+pos+sizeof(v1), sizeof(v2));
            if (!((v1 | v2) & 0x8080808080808080)) {
                pos = next_pos;
                continue;
            }
        }
        uint8_t byte = data[pos];
        while (byte < 0x80) {
            if (++pos == len) return true;
            byte = data[pos];
        }
        if ((byte & 0xe0) == 0xc0) {
            next_pos = pos+2;
            if (next_pos > len) return false;
            if ((data[pos+1] & 0xc0) != 0x80) return false;
            cp = (byte & 0x1f)<<6 | (data[pos+1] & 0x3f);
            if ((cp < 0x80) || (0x7ff < cp)) return false;
        } else if ((byte & 0xf0) == 0xe0) {
            next_pos = pos+3;
            if (next_pos > len) return false;
            if ((data[pos+1] & 0xc0) != 0x80) return false;
            if ((data[pos+2] & 0xc0) != 0x80) return false;
            cp = (byte & 0xf)<<12 | (data[pos+1] & 0x3f)<<6 | (data[pos+2] & 0x3f);
            if ((cp < 0x800) || (0xffff < cp) || (0xd7ff < cp && cp < 0xe000)) return false;
        } else if ((byte & 0xf8) == 0xf0) {
          next_pos = pos + 4;
          if (next_pos > len) return false;
          if ((data[pos+1] & 0xc0) != 0x80) return false;
          if ((data[pos+2] & 0xc0) != 0x80) return false;
          if ((data[pos+3] & 0xc0) != 0x80) return false;
          cp = (byte & 0x7)<<18 | (data[pos+1] & 0x3f)<<12 | (data[pos+2] & 0x3f)<<6 | (data[pos+3] & 0x3f);
          if (cp <= 0xffff || 0x10ffff < cp) return false;
        } else return false;
        pos = next_pos;
    }
    return true;
}

char* mag_strdup(const char* s) {
    if (mag_unlikely(!s)) return NULL;
    size_t len = strlen(s);
    char* clone = (*mag_alloc)(NULL, len+1, 0);
    memcpy(clone, s, len);
    clone[len] = '\0';
    return clone;
}

bool mag_solve_view_strides(int64_t (*out)[MAG_MAX_DIMS], const int64_t* osz, const int64_t* ost, int64_t ork, const int64_t* nsz, int64_t nrk) {
    int64_t numel = 1;
    for (int64_t i=0; i < ork; ++i)
        mag_assert2(!mag_mulov64(numel, osz[i], &numel));
    if (!numel) {
        if (!nrk) return false;
        (*out)[nrk-1] = 1;
        for (int64_t d=nrk-2; d >= 0; --d)
            mag_assert2(!mag_mulov64((*out)[d+1], nsz[d+1], &(*out)[d]));
        return true;
    }
    int64_t oi = ork-1;
    int64_t ni = nrk-1;
    while (oi >= 0 && ni >= 0) {
        if (nsz[ni] == 1) { (*out)[ni] = 0; --ni; continue; }
        for (; oi >= 0 && osz[oi] == 1; --oi);
        if (oi < 0) return false;
        if (nsz[ni] == osz[oi]) {
            (*out)[ni] = ost[oi];
            --ni; --oi;
            continue;
        }
        int64_t nc = nsz[ni];
        int64_t oc = osz[oi];
        int64_t cs = ost[oi];
        int64_t nkf = ni;
        while (nc != oc) {
            if (nc < oc) {
                --ni;
                if (ni < 0) return false;
                nc *= nsz[ni];
            } else {
                --oi;
                for (; oi >= 0 && osz[oi] == 1; --oi);
                if (oi < 0) return false;
                if (ost[oi] != osz[oi+1]*ost[oi+1])
                    return false;
                oc *= osz[oi];
            }
        }
        int64_t stride = cs;
        for (int64_t k=ni; k <= nkf; ++k) {
            (*out)[k] = stride;
            mag_assert2(!mag_mulov64(stride, nsz[k], &stride));
        }
        --ni; --oi;
    }
    while (ni >= 0) { (*out)[ni] = 0; --ni; }
    for (; oi >= 0 && osz[oi] == 1; --oi);
    return oi < 0;
}

void mag_infer_missing_dim(int64_t(*out)[MAG_MAX_DIMS], const int64_t* dims, int64_t rank, int64_t numel) {
    int64_t prod = 1, infer = -1;
    for (int64_t i=0; i < rank; ++i) {
        int64_t ax = dims[i];
        if (ax == -1) {
            mag_assert(infer == -1, "only one dimension can be -1");
            infer = i;
            (*out)[i] = 1;
        } else {
            mag_assert(ax > 0, "dimension must be > 0 or -1");
            (*out)[i] = ax;
            mag_assert2(!mag_mulov64(prod, ax, &prod));
        }
    }
    if (infer >= 0) {
        mag_assert(numel % prod == 0, "cannot infer dimension size from %" PRIi64 " and known product %" PRIi64, numel, prod);
        (*out)[infer] = numel / prod;
    } else {
        mag_assert(prod == numel, "total shape size mismatch: expected %" PRIi64 ", got %" PRIi64, numel, prod);
    }
}

bool mag_compute_broadcast_shape(const mag_tensor_t* a, const mag_tensor_t* b, int64_t* dims, int64_t* rank) {
    int64_t ar = a->rank, br = b->rank;
    int64_t r = *rank = ar > br ? ar : br;
    for (int64_t i=0; i < r; ++i) {
        int64_t ra = ar-1-i >= 0 ? a->shape[ar-1-i] : 1;
        int64_t rb = br-1-i >= 0 ? b->shape[br-1-i] : 1;
        if (mag_unlikely(!(ra == rb || ra == 1 || rb == 1))) return false;
        dims[r-1-i] = ra == 1 ? rb : ra;
    }
    return true;
}

void mag_matmul_tune_block_params(const mag_matmul_block_tune_info_t* info, mag_matmul_block_params_t* params) {
    if (!info->l1_size || !info->l2_size || !info->elsize) {
        *params = (mag_matmul_block_params_t){
            .MR = 8,
            .NR = 16,
            .MC = 256,
            .KC = 256,
            .NC = 128
        };
        return;
    }
    int64_t nt = info->nthreads;
    int64_t M = info->M;
    int64_t N = info->N;
    int64_t K = info->K;
    int64_t MR;
    int64_t NR;
    int64_t KC;
    int64_t VW = info->vecreg_width;
    int64_t W = VW >= 64 ? 64 : VW >= 32 ? 32 : 16;
    MR = VW / info->elsize;
    int64_t NR_cap = W == 64 ? 32 : W == 32 ? 32 : 16;
    NR = mag_clamp((MR)<<1, MR, NR_cap);
    if (W == 64) MR = 16, NR = 32;
    mag_e11m52_t aL1 = info->l1_load_factor ? info->l1_load_factor : W == 64 ? 0.55 : W == 32 ? 0.60 : 0.65;
    mag_e11m52_t aL2 = info->l2_load_factor ? info->l2_load_factor : W == 64 ? 0.40 : W == 32 ? 0.45 : 0.50;
    mag_e11m52_t L1e = aL1 * (mag_e11m52_t)info->l1_size;
    mag_e11m52_t L2e = aL2 * (mag_e11m52_t)info->l2_size;
    if (nt >= 2) {
        L1e *= 0.85;
        L2e *= 0.85;
    }
    mag_e11m52_t nb = (mag_e11m52_t)info->elsize;
    int64_t kc = (int64_t)(L1e / (nb*(mag_e11m52_t)(MR + NR)));
    kc = mag_rd_down(kc, 8);
    int64_t KC_lo = W == 64 ? 384 : W == 32 ? 256 : 192;
    int64_t KC_hi = W == 64 ? 1024 : W == 32 ? 768 : 512;
    kc = mag_clamp(kc, KC_lo, KC_hi);
    if (K >= 2048) kc = mag_clamp(kc + 128, KC_lo, KC_hi);
    KC = kc;
    int64_t MC = (int64_t)(info->split_a*L2e / (nb*(mag_e11m52_t)KC));
    int64_t NC = (int64_t)((1.0-info->split_a)*L2e / (nb*(mag_e11m52_t)KC));
    MC = mag_rd_down(MC, MR);
    NC = mag_rd_down(NC, NR);
    MR = mag_xmax(8, MR);
    NR = mag_xmax(8, NR);
    if (MC < MR) MC = MR;
    if (NC < NR) NC = NR;
    int64_t NC_cap = W == 64 ? 256 : 128;
    if (N < 8192) NC_cap = 128;
    if (NC > NC_cap) NC = mag_rd_down(NC_cap, NR);
    int64_t tic = (M + MC - 1)/MC;
    int64_t tjc = (N + NC - 1)/NC;
    int64_t tiles = tic * tjc;
    int64_t flops_call = (M*N*K)<<1;
    int64_t min_tiles_core = flops_call >= 0x10000000ll ? 1 : flops_call >= 0x2000000ll ? 2 : 4;
    int64_t tiles_needed = min_tiles_core * nt;
    if (tiles_needed < (nt<<1)+nt) tiles_needed = (nt<<1)+nt;
    while (tiles < tiles_needed && (MC > MR<<4 || NC > NR<<4)) {
        bool changed = false;
        int64_t nMC = MC>>1;
        if (!changed && nMC >= MR && (nMC*NC*KC)<<1 >= info->min_tile_flops) {
            MC = mag_rd_down(nMC, MR);
            changed = true;
        }
        int64_t nNC = NC>>1;
        if (!changed && nNC >= NR && (MC*nNC*KC)<<1 >= info->min_tile_flops) {
            NC = mag_rd_down(nNC, NR);
            changed = true;
        }
        if (!changed) break;
        tic = (M + MC - 1)/MC;
        tjc = (N + NC - 1)/NC;
        tiles = tic * tjc;
    }
    if (N >= 512 && NC < NR<<1) NC = NR<<1;
    *params = (mag_matmul_block_params_t){
        .MR = MR,
        .NR = NR,
        .MC = MC,
        .KC = KC,
        .NC = NC
    };
}

#undef mag_cdiv

/* The file format is implemented after the spec in ../docs/mag-file-format.md */

typedef enum {
    MAG_MAP_READ,        // open existing file, read-only
    MAG_MAP_WRITE,       // create/truncate to `size`, read-write
    MAG_MAP_READWRITE    // open existing file, read-write (size unchanged unless caller requests change)
} mag_map_mode_t;

typedef struct mag_mapped_file_t {
    uint8_t* map;
    size_t fs;
    bool writable;
    #if defined(_WIN32)
        HANDLE hfile;
        HANDLE hmap;
    #else
        int fd;
    #endif
} mag_mapped_file_t;

static bool mag_mmap_preallocate_grow(
    #if defined(_WIN32)
        HANDLE hfile,
    #else
        int fd,
    #endif
    size_t size
) {
    #ifdef _WIN32
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG)size;
        if (!SetFilePointerEx(hfile, li, NULL, FILE_BEGIN)) return false;
        if (!SetEndOfFile(hfile)) return false;
        return true;
    #elif defined(__APPLE__)
        fstore_t fst = {0};
        fst.fst_flags = F_ALLOCATECONTIG;
        fst.fst_posmode = F_PEOFPOSMODE;
        fst.fst_offset = 0;
        fst.fst_length = (off_t)size;
        if (fcntl(fd, F_PREALLOCATE, &fst) == -1) {
            fst.fst_flags = F_ALLOCATEALL;
            if (fcntl(fd, F_PREALLOCATE, &fst) == -1) {
                if (ftruncate(fd, (off_t)size) == -1) return false;
                return true;
            }
        }
        return ftruncate(fd, (off_t)size) == 0;
    #else
        #if (_XOPEN_SOURCE >= 600) || (_POSIX_C_SOURCE >= 200112L)
          #ifdef __linux__
            int r = posix_fallocate(fd, 0, (off_t)size);
            if (mag_unlikely(r != 0)) { errno = r; return false; }
            return true;
          #else
            return ftruncate(fd, (off_t)size) == 0;
          #endif
        #else
            return ftruncate(fd, (off_t)size) == 0;
        #endif
    #endif
}

static bool mag_mmap_strong_fsync(
#if defined(_WIN32)
    HANDLE hfile
#else
    int fd
#endif
) {
    #if defined(_WIN32)
        return FlushFileBuffers(hfile) != 0;
    #elif defined(__APPLE__)
        if (fcntl(fd, F_FULLFSYNC) == 0) return true;
        return fsync(fd) == 0;
    #else
        #ifdef __linux__
          if (fdatasync(fd) == 0)
              return true;
        #endif
        return fsync(fd) == 0;
    #endif
}

bool mag_map_file(mag_mapped_file_t* o, const char* filename, size_t size, mag_map_mode_t mode) {
    if (!o || !filename || !*filename) return false;
    memset(o, 0, sizeof(*o));
    #ifdef _WIN32
        DWORD access = 0;
        DWORD share = FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE;
        DWORD dispo = 0;
        DWORD attrs = FILE_ATTRIBUTE_NORMAL;
        BOOL inherit = FALSE;
        switch (mode) {
            case MAG_MAP_READ:
                access = GENERIC_READ;
                dispo = OPEN_EXISTING;
                o->writable = false;
                break;
            case MAG_MAP_WRITE:
                access = GENERIC_READ|GENERIC_WRITE;
                dispo = CREATE_ALWAYS;
                o->writable = true;
                break;
            case MAG_MAP_READWRITE:
                access = GENERIC_READ|GENERIC_WRITE;
                dispo = OPEN_EXISTING;
                o->writable = true;
                break;
            default: return false;
        }
        HANDLE hfile = CreateFileA(filename, access, share, NULL, dispo, attrs, NULL);
        if (mag_unlikely(hfile == INVALID_HANDLE_VALUE)) return false;
        LARGE_INTEGER fsz = {0};
        if (mode == MAG_MAP_WRITE) {
            if (mag_unlikely(!size)) {
                CloseHandle(hfile);
                return false;
            }
            if (mag_unlikely(!mag_mmap_preallocate_grow(hfile, size))) {
                CloseHandle(hfile);
                return false;
            }
            fsz.QuadPart = (LONGLONG)size;
        } else {
            if (mag_unlikely(!GetFileSizeEx(hfile, &fsz))) {
                CloseHandle(hfile);
                return false;
            }
            if (mode == MAG_MAP_READWRITE && size > 0 && (uint64_t)fsz.QuadPart != size) {
                if (mag_unlikely(!mag_mmap_preallocate_grow(hfile, size))) {
                    CloseHandle(hfile);
                    return false;
                }
                fsz.QuadPart = (LONGLONG)size;
            }
        }
        size_t fs = (size_t)fsz.QuadPart;
        HANDLE hmap = NULL;
        uint8_t* view = NULL;
        if (fs > 0) {
            DWORD pageProt = o->writable ? PAGE_READWRITE : PAGE_READONLY;
            hmap = CreateFileMappingA(hfile, NULL, pageProt, (DWORD)((uint64_t)fs>>32), (DWORD)((uint64_t)fs&0xffffffff), NULL);
            if (mag_unlikely(!hmap)) {
                CloseHandle(hfile);
                return false;
            }
            DWORD mapAcc = o->writable ? FILE_MAP_WRITE|FILE_MAP_READ : FILE_MAP_READ;
            view = (uint8_t*)MapViewOfFile(hmap, mapAcc, 0, 0, 0);
            if (mag_unlikely(!view)) {
                CloseHandle(hmap);
                CloseHandle(hfile);
                return false;
            }
        }
        o->hfile = hfile;
        o->hmap = hmap;
        o->map = view;
        o->fs = fs;
        return true;
    #else
        int oflags = 0;
        int prot = 0;
        int mflags = 0;
        switch (mode) {
            case MAG_MAP_READ:
                oflags = O_RDONLY;
                prot = PROT_READ;
                mflags = MAP_PRIVATE;
                o->writable = false;
                break;
            case MAG_MAP_WRITE:
                oflags = O_RDWR|O_CREAT|O_TRUNC;
                prot = PROT_READ|PROT_WRITE;
                mflags = MAP_SHARED;
                o->writable = true;
                break;
            case MAG_MAP_READWRITE:
                oflags = O_RDWR;
                prot = PROT_READ|PROT_WRITE;
                mflags = MAP_SHARED;
                o->writable = true;
                break;
            default: return false;
        }
        #ifdef O_CLOEXEC
          oflags |= O_CLOEXEC;
        #endif
        int fd = open(filename, oflags, 0666);
        if (mag_unlikely(fd == -1)) return false;
        size_t fs = 0;
        if (mode == MAG_MAP_WRITE) {
            if (mag_unlikely(!size)) { close(fd); return false; }
            if (mag_unlikely(!mag_mmap_preallocate_grow(fd, size))) { close(fd); return false; }
            fs = size;
        } else {
            struct stat st;
            if (mag_unlikely(fstat(fd, &st) == -1)) { close(fd); return false; }
            if (mag_unlikely(!S_ISREG(st.st_mode)))  { close(fd); return false; }
            fs = (size_t)st.st_size;
            if (mode == MAG_MAP_READWRITE && size > 0 && size != fs) {
                if (mag_unlikely(!mag_mmap_preallocate_grow(fd, size))) { close(fd); return false; }
                fs = size;
            }
        }
        uint8_t* view = NULL;
        if (fs > 0) {
            void* map = mmap(NULL, fs, prot, mflags, fd, 0);
            if (mag_unlikely(map == MAP_FAILED)) { close(fd); return false; }
            view = (uint8_t*)map;
        }
        o->fd = fd;
        o->map = view;
        o->fs = fs;
        return true;
    #endif
}

bool mag_unmap_file(mag_mapped_file_t* f) {
    if (mag_unlikely(!f)) return false;
    bool ok = true;
    #if defined(_WIN32)
        if (f->map && f->fs) {
            if (f->writable) {
                if (!FlushViewOfFile(f->map, 0)) ok = false;
                if (!mag_mmap_strong_fsync((HANDLE)f->hfile)) ok = false;
            }
            if (!UnmapViewOfFile(f->map)) ok = false;
        }
        if (f->hmap) { if (!CloseHandle((HANDLE)f->hmap)) ok = false; }
        if (f->hfile) { if (!CloseHandle((HANDLE)f->hfile)) ok = false; }
        memset(f, 0, sizeof(*f));
        return ok;
    #else
        if (f->map && f->fs) {
            if (f->writable) {
                if (mag_unlikely(msync(f->map, f->fs, MS_SYNC) == -1)) ok = false;
                if (mag_unlikely(!mag_mmap_strong_fsync(f->fd))) ok = false;
            }
            if (mag_unlikely(munmap(f->map, f->fs) == -1)) ok = false;
        }
        if (mag_unlikely(close(f->fd) == -1)) ok = false;
        memset(f, 0, sizeof(*f));
        return ok;
    #endif
}

typedef union mag_record_payload_t {
    mag_tensor_t* tensor;
    int64_t i64;
    mag_e11m52_t f64;
} mag_record_payload_t;
mag_static_assert(sizeof(mag_record_payload_t) == 8);

typedef struct mag_record_tagged_payload_t {
    mag_record_type_t type;
    mag_record_payload_t payload;
} mag_record_tagged_payload_t;
mag_static_assert(sizeof(mag_record_tagged_payload_t) == 16);

typedef struct mag_storage_record_t {
    const char* key;
    size_t key_len;
    mag_record_tagged_payload_t payload;
} mag_storage_record_t;

typedef struct mag_record_map_t {
    mag_storage_record_t* arr;
    size_t len;
    size_t cap;
} mag_record_map_t;

static void mag_record_map_rehash(mag_record_map_t* map, size_t nc) {
    mag_storage_record_t* pv = map->arr;
    size_t oc = map->cap;
    mag_storage_record_t* fresh = (*mag_alloc)(NULL, nc*sizeof(*fresh), 0);
    memset(fresh, 0, nc*sizeof(*fresh));
    size_t nl = 0;
    size_t mask = nc-1;
    for (size_t k=0; k < oc; ++k) {
        mag_storage_record_t rec = pv[k];
        if (!rec.key) continue;
        size_t h = mag_hash(rec.key, rec.key_len, 0)&mask;
        for (size_t j=h;;j=(j+1)&mask) {
            if (!fresh[j].key) {
                fresh[j] = rec;
                ++nl;
                break;
            }
        }
    }
    (*mag_alloc)(pv, 0, 0);
    map->arr = fresh;
    map->cap = nc;
    map->len = nl;
}

static void mag_record_map_init(mag_record_map_t* map) {
    size_t cap = 32;
    *map = (mag_record_map_t){
        .arr = (*mag_alloc)(NULL, cap*sizeof(*map->arr), 0),
        .len = 0,
        .cap = cap
    };
    memset(map->arr, 0, cap*sizeof(*map->arr));
}

static size_t mag_record_required_size(const mag_storage_record_t* records, size_t num_records) {
    size_t size = 0;
    for (size_t i=0; i < num_records; ++i) {
        size += records[i].key_len;
    }
    return size;
}

static void mag_record_map_free(mag_record_map_t* map) {
    for (size_t i=0; i < map->cap; ++i) {
        if (!map->arr[i].key) continue;
        (*mag_alloc)((void*)map->arr[i].key, 0, 0);
        if (map->arr[i].payload.type == MAG_RECORD_TYPE_TENSOR && map->arr[i].payload.payload.tensor)
            mag_tensor_decref(map->arr[i].payload.payload.tensor);
    }
    (*mag_alloc)(map->arr, 0, 0);
    memset(map, 0, sizeof(*map));
}

static bool mag_record_map_insert(mag_record_map_t* map, const char* key, mag_record_tagged_payload_t val) {
    if (mag_unlikely(!key)) return false;
    size_t kl = strlen(key);
    if (mag_unlikely(!kl)) return false;
    if (mag_unlikely(!mag_utf8_validate(key, kl))) return false;
    if ((map->len+1)*10 > map->cap*7)
        mag_record_map_rehash(map, map->cap<<1);
    size_t h = mag_hash(key, kl, 0);
    size_t mask = map->cap-1;
    size_t i;
    for (i=h&mask;;i=(i+1)&mask) {
        mag_storage_record_t* slot = &map->arr[i];
        if (!slot->key) break;
        if (mag_unlikely(slot->key_len == kl && !memcmp(slot->key, key, kl))) return false;
    }
    char* cloned_key = mag_strdup(key);
    if (val.type == MAG_RECORD_TYPE_TENSOR) {
        if (mag_unlikely(!val.payload.tensor)) {
            (*mag_alloc)(cloned_key, 0, 0);
            return false;
        }
        mag_tensor_incref(val.payload.tensor);
    }
    map->arr[i] = (mag_storage_record_t){
        .key = cloned_key,
        .key_len = kl,
        .payload = val
    };
    ++map->len;
    return true;
}

static mag_record_tagged_payload_t* mag_record_map_find(mag_record_map_t* map, const char* key) {
    if (mag_unlikely(!key)) return false;
    size_t kl = strlen(key);
    if (mag_unlikely(!kl || !mag_utf8_validate(key, kl))) return NULL;
    size_t h = mag_hash(key, kl, 0);
    size_t mask = map->cap-1;
    for (size_t i=h&mask;;i=(i+1)&mask) {
        mag_storage_record_t* slot = &map->arr[i];
        if (!slot->key) return NULL;
        if (slot->key_len == kl && !memcmp(slot->key, key, kl))
            return &slot->payload;
    }
}

#define mag_sto_san(expr) do { if (mag_unlikely(!(expr))) return false; } while (0)
#define mag_sto_san_do(expr, then) do { if (mag_unlikely(!(expr))) { then; } } while (0)
#define mag_sto_magic4(a,b,c,d) ((((d)&255)<<24) + (((c)&255)<<16) + (((b)&255)<<8) + ((a)&255))
#define MAG_STO_FILE_MAGIC mag_sto_magic4('M', 'A', 'G', '!')
#define MAG_STO_MAX_STR_LEN 65535
#define MAG_STO_FILE_HEADER_SIZE (4*1 + 4 + 4 + 4 + 4 + 4)
#define MAG_STO_META_HEADER_SIZE (4 + 8) /* aux + payload (key+key_len excluded) */
#define MAG_STO_TENSOR_HEADER_SIZE (4 + 8 + 8) /* aux + numel + abs_offset */
#define mag_sto_pack_aux(a,b,c,d) ((((uint8_t)(d)&255)<<24) + (((uint8_t)(c)&255)<<16) + (((uint8_t)(b)&255)<<8) + ((uint8_t)(a)&255))
#define mag_sto_unpack_aux(v, a, b, c, d) do { \
    *(a) = (v)&255; \
    *(b) = ((v)>>8)&255; \
    *(c) = ((v)>>16)&255; \
    *(d) = ((v)>>24)&255; \
} while (0)

static size_t mag_sto_aligned_buf_size(const mag_tensor_t* t) {
    mag_assert2(t->storage->host->type == MAG_DEVICE_TYPE_CPU);
    size_t sz = mag_tensor_get_data_size(t);
    mag_assert2(t->storage->alignment == MAG_CPU_BUF_ALIGN);
    mag_sto_san(sz <= SIZE_MAX-MAG_CPU_BUF_ALIGN-1);
    return (sz+(MAG_CPU_BUF_ALIGN-1))&~(MAG_CPU_BUF_ALIGN-1);
}

static inline bool mag_sto_wu32le(uint8_t** p, uint8_t* e, uint32_t v) {
    mag_sto_san((size_t)(e - *p) >= sizeof(v));
    #ifdef MAG_BE
        v = mag_bswap32(v);
    #endif
    memcpy(*p, &v, sizeof(v));
    *p += sizeof(v);
    return true;
}
static inline bool mag_sto_wu64le(uint8_t** p, uint8_t* e, uint64_t v) {
    mag_sto_san((size_t)(e - *p) >= sizeof(v));
    #ifdef MAG_BE
        v = mag_bswap64(v);
    #endif
    memcpy(*p, &v, sizeof(v));
    *p += sizeof(v);
    return true;
}
static inline bool mag_sto_wstr(uint8_t** p, uint8_t* e, const char* str) {
    size_t len = strlen(str);
    mag_sto_san(len <= MAG_STO_MAX_STR_LEN);
    mag_sto_san(mag_utf8_validate(str, len));
    mag_sto_san(mag_sto_wu32le(p, e, len));
    mag_sto_san((size_t)(e - *p) >= len);
    memcpy(*p, str, len);
    *p += len;
    return true;
}

static inline bool mag_sto_ru32le(const uint8_t** p, const uint8_t* e, uint32_t* v) {
    mag_sto_san((size_t)(e - *p) >= sizeof(*v));
    memcpy(v, *p, sizeof(*v));
    #ifdef MAG_BE
        *v = mag_bswap32(*v);
    #endif
    *p += sizeof(*v);
    return true;
}
static inline bool mag_sto_ru64le(const uint8_t** p, const uint8_t* e, uint64_t* v) {
    mag_sto_san((size_t)(e - *p) >= sizeof(*v));
    memcpy(v, *p, sizeof(*v));
    #ifdef MAG_BE
        *v = mag_bswap64(*v);
    #endif
    *p += sizeof(*v);
    return true;
}
static inline bool mag_sto_rstr(const uint8_t** p, const uint8_t* e, char** str) {
    uint32_t len;
    mag_sto_san(mag_sto_ru32le(p, e, &len));
    mag_sto_san(len <= MAG_STO_MAX_STR_LEN);
    mag_sto_san((size_t)(e - *p) >= len);
    *str = (*mag_alloc)(NULL, len+1, 0);
    memcpy(*str, *p, len);
    (*str)[len] = '\0';
    mag_sto_san_do(mag_utf8_validate(*str, len), (*mag_alloc)(*str, 0, 0); return false);
    *p += len;
    return true;
}

static bool mag_sto_patch_checksum_window_field(uint8_t* checksum_needle, uint32_t checksum) {
    #ifdef MAG_BE
        checksum = mag_bswap32(checksum);
    #endif
    memcpy(checksum_needle, &checksum, sizeof(checksum));
    return true;
}

static bool mag_sto_file_hdr_ser(uint8_t** p, uint8_t* e, uint32_t ver, uint32_t num_tensors, uint32_t num_meta_kv, uint8_t** checksum_needle) {
    uint8_t* b = *p;
    mag_sto_san(mag_sto_wu32le(p, e, MAG_STO_FILE_MAGIC));
    mag_sto_san(mag_sto_wu32le(p, e, ver));
    *checksum_needle = *p; /* Save the pointer to the checksum field. */
    mag_sto_san(mag_sto_wu32le(p, e, 0)); /* Checksum is written later. */
    mag_sto_san(mag_sto_wu32le(p, e, num_tensors));
    mag_sto_san(mag_sto_wu32le(p, e, num_meta_kv));
    mag_sto_san(mag_sto_wu32le(p, e, 0));
    mag_assert2(*p-b == MAG_STO_FILE_HEADER_SIZE);
    return true;
}

static bool mag_sto_file_hdr_deser(const uint8_t** p, const uint8_t* e, uint32_t* ver, uint32_t* checksum, uint32_t* num_tensors, uint32_t* num_meta_kv) {
    const uint8_t* b = *p;
    uint32_t magic;
    mag_sto_san(mag_sto_ru32le(p, e, &magic));
    mag_sto_san(magic == MAG_STO_FILE_MAGIC);
    mag_sto_san(mag_sto_ru32le(p, e, ver));
    mag_sto_san(*ver >= 1 && *ver <= MAG_STORAGE_VERSION);
    mag_sto_san(mag_sto_ru32le(p, e, checksum));
    mag_sto_san(mag_sto_ru32le(p, e, num_tensors));
    mag_sto_san(mag_sto_ru32le(p, e, num_meta_kv));
    uint32_t aux;
    mag_sto_san(mag_sto_ru32le(p, e, &aux)); /* Reserved field, must be zero. */
    mag_sto_san(aux == 0);
    mag_assert2(*p-b == MAG_STO_FILE_HEADER_SIZE);
    return true;
}

static bool mag_sto_meta_hdr_ser(uint8_t** p, uint8_t* e, const char* key, mag_record_tagged_payload_t val) {
    const uint8_t* b = *p;
    mag_sto_san(mag_sto_wu32le(p, e, mag_sto_pack_aux(val.type, 0, 0, 0)));
    switch (val.type) {
        case MAG_RECORD_TYPE_I64: {
            uint64_t u;
            memcpy(&u, &val.payload.i64, sizeof(u));
            mag_sto_san(mag_sto_wu64le(p, e, u));
        } break;
        case MAG_RECORD_TYPE_F64: {
            uint64_t u;
            memcpy(&u, &val.payload.f64, sizeof(u));
            mag_sto_san(mag_sto_wu64le(p, e, u));
        } break;
        case MAG_RECORD_TYPE_TENSOR:
        default: mag_sto_san(false);
    }
    mag_assert2(*p-b == MAG_STO_META_HEADER_SIZE);
    mag_sto_san(mag_sto_wstr(p, e, key));
    mag_assert2(*p-b == MAG_STO_META_HEADER_SIZE+4+strlen(key));
    return true;
}

static bool mag_sto_meta_hdr_deser(const uint8_t** p, const uint8_t* e, char** key, mag_record_tagged_payload_t* val) {
    const uint8_t* b = *p;
    uint32_t aux;
    mag_sto_san(mag_sto_ru32le(p, e, &aux));
    uint8_t t, u1, u2, u3;
    mag_sto_unpack_aux(aux, &t, &u1, &u2, &u3);
    mag_sto_san(t < MAG_RECORD_TYPE__COUNT && (aux>>8) == 0);
    val->type = (mag_record_type_t)t;
    switch (val->type) {
        case MAG_RECORD_TYPE_I64: {
            uint64_t v;
            mag_sto_san(mag_sto_ru64le(p, e, &v));
            memcpy(&val->payload.i64, &v, sizeof(v));
        } break;
        case MAG_RECORD_TYPE_F64: {
            uint64_t v;
            mag_sto_san(mag_sto_ru64le(p, e, &v));
            memcpy(&val->payload.f64, &v, sizeof(v));
        } break;
        case MAG_RECORD_TYPE_TENSOR:
        default: mag_sto_san(false);
    }
    mag_assert2(*p-b == MAG_STO_META_HEADER_SIZE);
    mag_sto_san(mag_sto_rstr(p, e, key));
    mag_assert2(*p-b == MAG_STO_META_HEADER_SIZE+4+strlen(*key));
    return true;
}

static bool mag_sto_tensor_hdr_ser(uint8_t** p, uint8_t* e, const char* key, const mag_tensor_t* t, size_t data_base, size_t data_offs) {
    const uint8_t* b = *p;
    mag_sto_san(t->storage->host->type == MAG_DEVICE_TYPE_CPU);
    mag_sto_san(mag_tensor_is_contiguous(t));
    mag_sto_san(t->rank > 0 && t->rank <= MAG_MAX_DIMS);
    mag_sto_san(t->dtype < MAG_DTYPE__NUM);
    mag_sto_san(mag_sto_wu32le(p, e, mag_sto_pack_aux(t->dtype, (uint8_t)t->rank, 0, 0)));
    mag_sto_san(mag_sto_wu64le(p, e, t->numel));
    mag_sto_san(mag_sto_wu64le(p, e, data_base+data_offs));
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE);
    for (int64_t i=0; i<t->rank; ++i) {
        mag_sto_san(t->shape[i] > 0);
        mag_sto_san(mag_sto_wu64le(p, e, (uint64_t)t->shape[i]));
    }
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE + t->rank*8);
    mag_sto_san(mag_sto_wstr(p, e, key));
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE + t->rank*8 + 4+strlen(key));
    return true;
}

static bool mag_sto_tensor_hdr_deser(const uint8_t** p, const uint8_t* e, char** key, mag_dtype_t* odt, int64_t* ork, int64_t(*oshape)[MAG_MAX_DIMS], uint64_t* sz, uint64_t* abs_offs) {
    const uint8_t* b = *p;
    uint32_t aux;
    mag_sto_san(mag_sto_ru32le(p, e, &aux));
    uint8_t dt, rank, u2, u3;
    mag_sto_unpack_aux(aux, &dt, &rank, &u2, &u3);
    mag_sto_san(dt < MAG_DTYPE__NUM && rank > 0 && rank <= MAG_MAX_DIMS && (aux>>16) == 0);
    mag_sto_san(mag_sto_ru64le(p, e, sz));
    mag_sto_san(mag_sto_ru64le(p, e, abs_offs));
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE);
    for (int64_t i=0; i<rank; ++i) {
        uint64_t d;
        mag_sto_san(mag_sto_ru64le(p, e, &d));
        mag_sto_san(d > 0 && d <= INT64_MAX);
        (*oshape)[i] = (int64_t)d;
    }
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE + rank*8);
    mag_sto_san(mag_sto_rstr(p, e, key));
    mag_assert2(*p-b == MAG_STO_TENSOR_HEADER_SIZE + rank*8 + 4+strlen(*key));
    *odt = (mag_dtype_t)dt;
    *ork = (int64_t)rank;
    return true;
}

static bool mag_sto_tensor_buf_ser(uint8_t** p, uint8_t* e, const mag_tensor_t* t) {
    mag_sto_san(t->storage->host->type == MAG_DEVICE_TYPE_CPU);
    mag_sto_san(mag_tensor_is_contiguous(t));
    size_t al = MAG_CPU_BUF_ALIGN;
    size_t payload = mag_tensor_get_data_size(t);
    size_t mis = (uintptr_t)*p&(al-1);
    size_t lead = mis ? al-mis : 0;
    size_t rem = payload&(al-1);
    size_t tail = rem ? al-rem : 0;
    mag_sto_san((size_t)(e-*p) >= lead+payload+tail);
    if (lead) memset(*p, 0xff, lead), *p += lead;
    const void* src = mag_tensor_get_data_ptr(t);
    memcpy(*p, src, payload);
    if (tail) memset(*p+payload, 0xff, tail);
    *p += payload+tail;
    mag_assert2(mag_sto_aligned_buf_size(t) == payload+tail);
    return true;
}

static bool mag_sto_tensor_buf_deser(const uint8_t** p, const uint8_t* file_begin, const uint8_t* e, size_t abs_offs, size_t logical_nb, size_t aligned_nb, void* dst) {
    mag_sto_san(logical_nb <= aligned_nb);
    const uint8_t* src = file_begin + abs_offs;
    const uint8_t* region_end = src + aligned_nb;
    mag_sto_san(src >= file_begin && region_end <= e);
    if (*p != src) {
        mag_sto_san(*p <= src);
        for (const uint8_t* q=*p; q < src; ++q)
            mag_sto_san(*q == 0xff);
        *p = src;
    }
    mag_sto_san(dst != NULL);
    mag_sto_san((size_t)(e - *p) >= logical_nb);
    memcpy(dst, *p, logical_nb);
    const uint8_t* pad = *p + logical_nb;
    const uint8_t* pad_end = src + aligned_nb;
    for (const uint8_t* q=pad; q < pad_end; ++q)
        mag_sto_san(*q == 0xff);
    *p = region_end;
    return true;
}

typedef enum mag_storage_mode_t {
    MAG_STORAGE_MODE_READ = 1<<0,
    MAG_STORAGE_MODE_WRITE = 1<<1,
} mag_storage_mode_t;

struct mag_storage_archive_t {
    mag_context_t* ctx;
    char* path;
    mag_storage_mode_t mode;
    mag_record_map_t tensors;
    mag_record_map_t metadata;
};

static size_t mag_sto_estimate_file_size(const mag_record_map_t* tensors, const mag_record_map_t* metadata) {
    size_t size = MAG_STO_FILE_HEADER_SIZE;
    for (size_t i=0; i < metadata->cap; ++i) {
        if (!metadata->arr[i].key) continue;
        size += MAG_STO_META_HEADER_SIZE;
        size += 4+metadata->arr[i].key_len;
        if (metadata->arr[i].payload.type != MAG_RECORD_TYPE_TENSOR)
            size += 8;
    }
    for (size_t i=0; i < tensors->cap; ++i) {
        if (!tensors->arr[i].key) continue;
        size += MAG_STO_TENSOR_HEADER_SIZE;
        size += 8*tensors->arr[i].payload.payload.tensor->rank; /* shape */
        size += 4+tensors->arr[i].key_len; /* key length + key */
        size = (size+(MAG_CPU_BUF_ALIGN-1))&~(MAG_CPU_BUF_ALIGN-1);
        size += mag_sto_aligned_buf_size(tensors->arr[i].payload.payload.tensor);
    }
    return size;
}

static bool mag_storage_read_from_buffer(mag_storage_archive_t* archive, const uint8_t* buf, size_t size) {
    const uint8_t* p = buf;
    const uint8_t* e = p+size;
    uint32_t ver, checksum, num_tensors, num_meta_kv;
    mag_sto_san(mag_sto_file_hdr_deser(&p, e, &ver, &checksum, &num_tensors, &num_meta_kv));
    for (uint32_t i=0; i < num_meta_kv; ++i) { /* Read the metadata key-value pairs */
        char* key = NULL;
        mag_record_tagged_payload_t val = {0};
        mag_sto_san(mag_sto_meta_hdr_deser(&p, e, &key, &val));
        mag_sto_san_do(mag_record_map_insert(&archive->metadata, key, val), (*mag_alloc)(key, 0, 0); return false);
        (*mag_alloc)(key, 0, 0);
    }
    return true;
}

static bool mag_storage_read_from_disk(mag_storage_archive_t* archive, const char* filename) {
    if (mag_unlikely(!archive || !filename || !*filename)) return false;
    mag_mapped_file_t map;
    if (mag_unlikely(!mag_map_file(&map, filename, 0, MAG_MAP_READ)))
        return false;
    if (mag_unlikely(map.fs < MAG_STO_FILE_HEADER_SIZE || !mag_storage_read_from_buffer(archive, map.map, map.fs))) {
        mag_unmap_file(&map);
        return false;
    }
    return mag_unmap_file(&map);
}

static bool mag_storage_write_to_disk(mag_storage_archive_t* archive, const char* filename) {
    if (mag_unlikely(!archive || !filename || !*filename)) return false;
    if (mag_unlikely(archive->tensors.len > UINT32_MAX || archive->metadata.len > UINT32_MAX)) return false;
    size_t fs = mag_sto_estimate_file_size(&archive->tensors, &archive->metadata);
    mag_mapped_file_t map;
    if (mag_unlikely(!mag_map_file(&map, filename, fs, MAG_MAP_WRITE)))
        return false;
    uint8_t* b = map.map;
    uint8_t* p = b;
    uint8_t* e = p+fs;
    uint8_t* checksum_needle = NULL;
    /* Write the file header */
    mag_sto_san_do(mag_sto_file_hdr_ser(&p, e, MAG_STORAGE_VERSION, (uint32_t)archive->tensors.len, (uint32_t)archive->metadata.len, &checksum_needle), goto error);
    for (size_t i=0; i < archive->metadata.cap; ++i) { /* Write the metadata key-value pairs */
        if (!archive->metadata.arr[i].key) continue;
        mag_sto_san_do(mag_sto_meta_hdr_ser(&p, e, archive->metadata.arr[i].key, archive->metadata.arr[i].payload), goto error);
    }
    uint8_t* dp = p;
    for (size_t i=0; i < archive->tensors.cap; ++i) { /* Compute the base offset for tensor data */
        if (!archive->tensors.arr[i].key) continue;
        mag_tensor_t* t = archive->tensors.arr[i].payload.payload.tensor;
        dp += MAG_STO_TENSOR_HEADER_SIZE;
        dp += 8*t->rank;
        dp += 4+archive->tensors.arr[i].key_len;
    }
    size_t nbh = (size_t)(dp-b);
    size_t al = MAG_CPU_BUF_ALIGN;
    size_t data_base = (nbh+(al-1))&~(al-1);
    for (size_t i=0,data_offs=0; i < archive->tensors.cap; ++i) {     /* Write the tensor records */
        if (!archive->tensors.arr[i].key) continue;
        const char* key = archive->tensors.arr[i].key;
        mag_tensor_t* tensor = archive->tensors.arr[i].payload.payload.tensor;
        mag_sto_san_do(mag_sto_tensor_hdr_ser(&p, e, key, tensor, data_base, data_offs), goto error);
        data_offs += mag_sto_aligned_buf_size(tensor);
    }
    uint32_t crc = mag_crc32c(4+checksum_needle, (size_t)(p-(4+checksum_needle)));     /* Compute checksum and patch the file header */
    mag_sto_san_do(mag_sto_patch_checksum_window_field(checksum_needle, crc), goto error);
    for (size_t i=0; i < archive->tensors.cap; ++i) { /* Write the tensor data */
        if (!archive->tensors.arr[i].key) continue;
        const mag_tensor_t* t = archive->tensors.arr[i].payload.payload.tensor;
        mag_sto_san_do(mag_sto_tensor_buf_ser(&p, e, t), goto error);
    }
    return mag_unmap_file(&map);
    error:
        mag_unmap_file(&map);
        return false;
}

mag_storage_archive_t* mag_storage_open(mag_context_t* ctx, const char* filename, char mode) {
    if (mag_unlikely(!ctx || !filename || !*filename)) return NULL;
    if (mag_unlikely(mode != 'r' && mode != 'w')) return NULL;
    mag_storage_archive_t* archive = (*mag_alloc)(NULL, sizeof(*archive), 0);
    archive->ctx = ctx;
    archive->path = mag_strdup(filename);
    archive->mode = mode == 'r' ? MAG_STORAGE_MODE_READ : MAG_STORAGE_MODE_WRITE;
    mag_record_map_init(&archive->tensors);
    mag_record_map_init(&archive->metadata);
    if (archive->mode & MAG_STORAGE_MODE_READ && mag_unlikely(!mag_storage_read_from_disk(archive, archive->path))) {
        (*mag_alloc)(archive, 0, 0);
        return NULL;
    }
    return archive;
}

bool mag_storage_close(mag_storage_archive_t* archive) {
    if (mag_unlikely(!archive)) return false;
    bool sync_ok = true;
    if (archive->mode & MAG_STORAGE_MODE_WRITE)
        sync_ok &= mag_storage_write_to_disk(archive, archive->path);
    (*mag_alloc)(archive->path, 0, 0);
    mag_record_map_free(&archive->tensors);
    mag_record_map_free(&archive->metadata);
    (*mag_alloc)(archive, 0, 0);
    return sync_ok;
}

const char** mag_storage_get_all_tensor_keys(mag_storage_archive_t* archive, size_t* out_len) {
    char** mem = (*mag_alloc)(NULL, archive->tensors.len*sizeof(char*), 0);
    for (size_t i=0, j=0; i < archive->tensors.cap; ++i) {
        if (!archive->tensors.arr[i].key) continue;
        mem[j++] = mag_strdup(archive->tensors.arr[i].key);
    }
    *out_len = archive->tensors.len;
    return (const char**)mem;
}

const char** mag_storage_get_all_metadata_keys(mag_storage_archive_t* archive, size_t* out_len) {
    char** mem = (*mag_alloc)(NULL, archive->metadata.len*sizeof(char*), 0);
    for (size_t i=0, j=0; i < archive->metadata.cap; ++i) {
        if (!archive->metadata.arr[i].key) continue;
        mem[j++] = mag_strdup(archive->metadata.arr[i].key);
    }
    *out_len = archive->metadata.len;
    return (const char**)mem;
}

void mag_storage_get_all_keys_free(const char** keys, size_t len) {
    for (size_t i=0; i < len; ++i)
        (*mag_alloc)((void*)keys[i], 0, 0);
    (*mag_alloc)((void*)keys, 0, 0);
}

bool mag_storage_set_tensor(mag_storage_archive_t* archive, const char* key, mag_tensor_t* tensor){
    if (mag_unlikely(!archive || !key || !*key || !tensor)) return false;
    mag_record_tagged_payload_t val = {
        .type = MAG_RECORD_TYPE_TENSOR,
        .payload = { .tensor = tensor }
    };
    return mag_record_map_insert(&archive->tensors, key, val);
}

mag_tensor_t* mag_storage_get_tensor(mag_storage_archive_t* archive, const char* key){
    if (mag_unlikely(!archive || !key || !*key)) return NULL;
    mag_record_tagged_payload_t* val = mag_record_map_find(&archive->tensors, key);
    if (mag_unlikely(!val || val->type != MAG_RECORD_TYPE_TENSOR || !val->payload.tensor)) return NULL;
    return val->payload.tensor;
}

bool mag_storage_set_metadata_i64(mag_storage_archive_t* archive, const char* key, int64_t value) {
    if (mag_unlikely(!archive || !key || !*key)) return false;
    mag_record_tagged_payload_t val = {
        .type = MAG_RECORD_TYPE_I64,
        .payload = {.i64 = value}
    };
    return mag_record_map_insert(&archive->metadata, key, val);
}

bool mag_storage_get_metadata_i64(mag_storage_archive_t* archive, const char* key, int64_t* value) {
    if (mag_unlikely(!archive || !key || !*key || !value)) return false;
    mag_record_tagged_payload_t* val = mag_record_map_find(&archive->metadata, key);
    if (mag_unlikely(!val || val->type != MAG_RECORD_TYPE_I64)) {
        *value = 0;
        return false;
    }
    *value = val->payload.i64;
    return true;
}

bool mag_storage_set_metadata_f64(mag_storage_archive_t* archive, const char* key, mag_e11m52_t value) {
    if (mag_unlikely(!archive || !key || !*key)) return false;
    mag_record_tagged_payload_t val = {
        .type = MAG_RECORD_TYPE_F64,
        .payload = {.f64 = value}
    };
    return mag_record_map_insert(&archive->metadata, key, val);
}

bool mag_storage_get_metadata_f64(mag_storage_archive_t* archive, const char* key, mag_e11m52_t* value) {
    if (mag_unlikely(!archive || !key || !*key || !value)) return false;
    mag_record_tagged_payload_t* val = mag_record_map_find(&archive->metadata, key);
    if (mag_unlikely(!val || val->type != MAG_RECORD_TYPE_F64)) {
        *value = 0;
        return false;
    }
    *value = val->payload.f64;
    return true;
}

mag_record_type_t mag_storage_get_metadata_type(mag_storage_archive_t* archive, const char* key) {
    if (mag_unlikely(!archive || !key || !*key)) return MAG_RECORD_TYPE__COUNT;
    mag_record_tagged_payload_t* val = mag_record_map_find(&archive->metadata, key);
    return val ? val->type : MAG_RECORD_TYPE__COUNT;
}

