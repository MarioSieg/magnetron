/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

/*
**
**
** ### To add a new operation:
** 1. Add the operation to the mag_op_def macro, which defines all operations, with all information needed.
** 2. Write a validation routine, or use an existing one (e.g. 'mag_validate_op_binary').
** 3. Add the validation routine to the 'routines' table in 'mag_op_get_validator_routine', at the op index.
** 4. Write a result tensor constructor routine or use an existing one (e.g. 'mag_result_constructor_routine_isomorph').
** 5. Add the result tensor constructor routine to the 'routines' table in 'mag_op_get_result_constructor_routine', at the op index.
** 6. Write a BLAS computation routine.
** 7. Add the BLAS computation routine to the 'dispatch_lut' table in 'mag_blas_compute_dispatch_table_default', at the op index.
*/

/*
** Here ⊕ denotes a binary or unary operator.
** Note that an operators can or cannot support any of those forms. This must be specified in mag_op_def.
**
** Operators can have two forms:
**
** 1. R = A ⊕ B
**  Result is a new tensor of shape of A and contains element-wise result of A ⊕ B, where A and B are tensors.
**
** 2. R = A ⊕= B
**  Result is a view tensor of shape of A and contains element-wise result of A ⊕= B, where A and B are tensors. (Safes 1 allocation)
*/

#include "magnetron.h"
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
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <unistd.h>
#ifdef __linux__
#include <linux/prctl.h>
#include <sys/prctl.h>
#ifdef __aarch64__
#include <libc/sysv/consts/hwcap.h>
#include <sys/auxv.h
#endif
#endif
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

MAG_NORET MAG_COLDPROC MAG_EXPORT void mag_panic(const char* msg, ...) {
    fprintf(stdout, "%s", MAG_CC_RED);
    va_list args;
    va_start(args, msg);
    vfprintf(stdout, msg, args);
    va_end(args);
    fprintf(stdout, "%s", MAG_CC_RESET);
    fputc('\n', stdout);
    fflush(stdout);
    abort();
}

static void* mag_os_alloc_stub(void* blk, size_t size) {
    if (!size) {
        free(blk);
        return NULL;
    }
    if(!blk) {
        blk = malloc(size);
        if (mag_unlikely(!blk)) {
            double mem = 0.0;
            const char* unit = "";
            mag_humanize_memory_size(size, &mem, &unit);
            mag_panic("Failed to allocate %.01f %s memory", mem, unit);
        }
        return blk;
    }
    void* block = realloc(blk, size);
    if (mag_unlikely(!block)) {
        double mem = 0.0;
        const char* unit = "";
        mag_humanize_memory_size(size, &mem, &unit);
        mag_panic("Failed to reallocate %.01f %s memory", mem, unit);
    }
    return block;
}

void* (*mag_alloc)(void* blk, size_t size) = &mag_os_alloc_stub;

void* mag_alloc_aligned(size_t size, size_t align) {
    mag_assert(align && !(align&(align-1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    void* p = (*mag_alloc)(NULL, size+sizeof(void*)+align-1);
    uintptr_t pp = ((uintptr_t)p+sizeof(void*)+align-1)&-align;
    ((void**)pp)[-1] = p;
    return (void*)pp;
}

void mag_free_aligned(void* blk) {
    (*mag_alloc)(((void**)blk)[-1], 0);
}

/* Include STB libraries and override their allocator with ours. */
#define STBI_STATIC
#define STBI_MALLOC(sz) ((*mag_alloc)(NULL, (sz)))
#define STBI_FREE(ptr) ((*mag_alloc)((ptr), 0))
#define STBI_REALLOC(ptr, sz) ((*mag_alloc)((ptr), (sz)))
#define STBIW_MALLOC(sz) ((*mag_alloc)(NULL, (sz)))
#define STBIW_FREE(ptr) ((*mag_alloc)((ptr), 0))
#define STBIW_REALLOC(ptr, sz) ((*mag_alloc)((ptr), (sz)))
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#if defined(__x86_64__) || defined(_M_X64)
#define _(enumerator, leaf, reg, bit) #enumerator
const char* const mag_x86_64_feature_names[MAG_X86_64_FEATURE__NUM] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#define _(enumerator, leaf, reg, bit) (0xff&MAG_X86_64_CPUID_##leaf)
const uint8_t mag_x86_64_feature_leaves[MAG_X86_64_FEATURE__NUM] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#define _(enumerator, leaf, reg, bit) (0xff&MAG_X86_64_CPUID_##reg)
const uint8_t mag_x86_64_feature_regs[MAG_X86_64_FEATURE__NUM] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#define _(enumerator, leaf, reg, bit) (1u<<(bit))
const uint32_t mag_x86_64_feature_masks[MAG_X86_64_FEATURE__NUM] = {
    mag_x86_64_feature_def(_, MAG_SEP)
};
#undef _
#undef mag_x86_64_feature_def
#endif

void* (*mag_get_alloc_fn(void))(void* blk, size_t size) {
    return mag_alloc;
}

void mag_set_alloc_fn(void* (*alloc)(void* blk, size_t size)) {
    mag_assert2(alloc);
    mag_alloc = alloc;
}

void mag_humanize_memory_size(size_t n, double* out, const char** unit) {
    if (n < (1<<10)) {
        *out = (double)n;
        *unit = "B";
    } else if (n < (1<<20)) {
        *out = (double)n/(double)(1<<10);
        *unit = "KiB";
    } else if (n < (1<<30)) {
        *out = (double)n/(double)(1<<20);
        *unit = "MiB";
    } else {
        *out = (double)n/(double)(1<<30);
        *unit = "GiB";
    }
}

static void MAG_COLDPROC mag_print_separator(FILE* f) {
    f = f ? f : stdout;
    char sep[100+1];
    for (size_t i=0; i < (sizeof(sep)/sizeof(*sep))-1; ++i) sep[i] = '-';
    sep[sizeof(sep)/sizeof(*sep)-1] = '\0';
    fprintf(f, "%s\n", sep);
}

#define MAG_FMT_DIM_BUF_SIZE ((21+4)*MAG_MAX_DIMS)
static void mag_fmt_dims(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank) {
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

uintptr_t mag_thread_id(void) {
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

static uint64_t mag_hpc_clock_ns(void) { /* High precision clock in nanoseconds. */
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
static uint64_t mag_hpc_clock_elapsed_ns(uint64_t start) { /* High precision clock elapsed time in microseconds. */
    return (uint64_t)llabs((int64_t)mag_hpc_clock_ns() - (int64_t)start);
}
static double mag_hpc_clock_elapsed_ms(uint64_t start) { /* High precision clock elapsed time in milliseconds. */
    return (double)mag_hpc_clock_elapsed_ns(start) / 1e6;
}
#define mag_clock_cycles() ((uint64_t)clock())
#define mag_cycles_per_ms() ((uint64_t)CLOCKS_PER_SEC/1000)

typedef uint32_t mag_bitset_t;
mag_static_assert(sizeof(mag_bitset_t) == 4);
#define mag_bitset_size(n) (((n)+((4<<3)-1))>>5)
#define mag_bitset_get(sets, i) (!!(sets[(i)>>5]&(1u<<((i)&((4<<3)-1)))))
#define mag_bitset_set(sets, i) (sets[(i)>>5]|=(1u<<((i)&((4<<3)-1))))
#define mag_bitset_clear(sets, i) (sets[(i)>>5]&=~(1u<<((i)&((4<<3)-1))))
#define mag_bitset_toggle(sets, i) (sets[(i)>>5]^=(1u<<((i)&((4<<3)-1))))

/* Eval Chebyshev coeffs steps for some x. f(x) : [a, b] -> ℝ. */
static double mag_chebyshev_eval(double x, double a, double b, const double* coeffs, uint32_t steps) {
    double scale = 4.0/(b - a);
    double rls = -2.0 + (x - a)*scale;
    double k1 = 0.0, k2 = 0.0;
    for (uint32_t j = steps-1; j; --j) {
        double tmp = k1;
        k1 = rls*k1 - k2 + coeffs[j];
        k2 = tmp;
    }
    return 0.5*rls*k1 - k2 + 0.5**coeffs;
}

/* Generate Chebyshev coeffs for f(x) : [a, b] -> ℝ. */
static double* mag_chebyshev_setup(double (*f)(double), double a, double b, uint32_t steps, bool linear_l, bool linear_r) {
    mag_assert2(steps);
    double* r = (*mag_alloc)(NULL, sizeof(*r)*steps);
    memset(r, 0, sizeof(*r)*steps);
    double dsteps = (double)steps;
    for (uint32_t i=0; i < steps; ++i) {
        for (uint32_t j=0; j < steps; ++j) {
            double wav = 0.5*(1.0 + cos(M_PI*(j + 0.5)/dsteps));
            double x = a + (b - a)*wav, y = (*f)(x);
            double weight = cos(M_PI*(double)i*(j + 0.5)/dsteps);
            r[i] += 2.0*y*weight/dsteps;
        }
    }
    double xmi = 0.0, xma = 0.0;
    if (linear_l) xmi = (*f)(a) - mag_chebyshev_eval(a, a, b, r, steps);
    if (linear_r) xma = (*f)(b) - mag_chebyshev_eval(b, a, b, r, steps);
    r[0] += 2.0*(xma + xmi)*0.5;
    r[1] += (xma - xmi)*0.5;
    return r;
}

static bool MAG_AINLINE mag_imull64_ov(int64_t a, int64_t b, int64_t* out) { /* Performs c = a*b with overflow checking. Returns true on overflow, else false. */
    #ifdef _MSC_VER
    #ifdef _M_ARM64
        uint64_t high = __umulh(a, b);
        *out = a*b;
        return high != (*out>>63);
    #else
        int64_t high;
        int64_t low = _mul128(a, b, &high);
        int64_t sign = low >> 63;
        *out = low;
        return high != sign;
    #endif
    #else
    #if __SIZEOF_LONG_LONG__ == 8 && __SIZEOF_LONG__ == 8
        return __builtin_smulll_overflow(a, b, (long long*)out);
    #else
        return __builtin_smull_overflow(a, b, out);
    #endif
    #endif
}

static void mag_system_host_info_query(mag_ctx_t* ctx); /* Query host system information. */
static void mag_system_host_info_dump(mag_ctx_t* ctx) {
    mag_log_info("OS/Kernel: %s", ctx->sys.os_name);
    const char* cpu_arch = "?";
    #if defined(__x86_64__) || defined(_M_X64)
        cpu_arch = "x86-64";
    #elif defined(__aarch64__) || defined(_M_ARM64)
        cpu_arch = "aarch64";
    #else
    #error "Unknwon CPU arch"
    #endif
    mag_log_info("CPU (%s): %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u", cpu_arch, ctx->sys.cpu_name, ctx->sys.cpu_virtual_cores, ctx->sys.cpu_physical_cores, ctx->sys.cpu_sockets);
    #if defined(__x86_64__) || defined(_M_X64) /* Print CPU features for x86-64 platforms. */
        if (mag_log_enabled) {
            printf("CPU Features:");
            for (uint32_t i=0, k=0; i < MAG_X86_64_FEATURE__NUM; ++i) {
                if (mag_ctx_x86_64_cpu_has_feature(ctx, i)) {
                    if ((k++ & 7) == 0) printf("\n\t");
                    printf("%s ", mag_x86_64_feature_names[i]);
                }
            }
            putchar('\n');
        }
    #endif
    double mem_total, mem_free, mem_used;
    const char* mem_unit_total, *mem_unit_free, *mem_unit_used;
    mag_humanize_memory_size(ctx->sys.phys_mem_total, &mem_total, &mem_unit_total);
    mag_humanize_memory_size(ctx->sys.phys_mem_free, &mem_free, &mem_unit_free);
    mag_humanize_memory_size((size_t)llabs((int64_t)ctx->sys.phys_mem_total-(int64_t)ctx->sys.phys_mem_free), &mem_used, &mem_unit_used);
    double mem_used_percent = fabs((double)(ctx->sys.phys_mem_total-ctx->sys.phys_mem_free))/(double)ctx->sys.phys_mem_total*100.0;
    mag_log_info("Physical Machine Memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);
}

/* Default image loader/saver implementation. */
static uint8_t* mag_default_image_load_impl(const char*, uint32_t(*)[3], mag_color_channels_t);
static void mag_default_image_load_free_fn_impl(uint8_t*);
static bool mag_default_image_save_impl(const char*, const uint8_t*, const uint32_t(*)[3]);

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

mag_ctx_t* mag_ctx_create(mag_compute_device_type_t device) {
    const mag_device_descriptor_t info = {device};
    return mag_ctx_create2(&info);
}

mag_ctx_t* mag_ctx_create2(const mag_device_descriptor_t* device_info) {
    mag_log_info("Creating magnetron context...");

    uint64_t time_stamp_start = mag_hpc_clock_ns();
    mag_ctx_dump_compiler_info(); /* Dump compiler info. */

    /* Initialize context with default values or from context info. */
    mag_ctx_t* ctx = (*mag_alloc)(NULL, sizeof(*ctx)); /* Allocate context. */
    memset(ctx, 0, sizeof(*ctx));

    /* Init allocators */
    mag_fixed_intrusive_pool_init(&ctx->tensor_pool, sizeof(mag_tensor_t), __alignof(mag_tensor_t), 4096);

    ctx->tr_id = mag_thread_id(); /* Get thread ID. */

    /* Query and print host system information. */
    mag_system_host_info_query(ctx);
    mag_system_host_info_dump(ctx);

    /* Configure configureable media processors */
    ctx->image_load_fn = &mag_default_image_load_impl;
    ctx->image_load_free_fn = &mag_default_image_load_free_fn_impl;
    ctx->image_save_fn = &mag_default_image_save_impl;

    /* Create selected compute device. */
    ctx->exec_mode = MAG_EXEC_MODE_EAGER;
    ctx->device_type = device_info->type;
    ctx->device = mag_init_dynamic_device(ctx, device_info);
    mag_log_info("Compute device: %s", ctx->device->name);


    /* Print context initialization time. */
    mag_log_info("magnetron context initialized in %.05f ms", mag_hpc_clock_elapsed_ms(time_stamp_start));
    return ctx;
}

static void mag_tensor_destroy(mag_tensor_t* t);

void mag_ctx_destroy(mag_ctx_t* ctx) {
#ifdef MAG_DEBUG /* Check for leaked tensors in RC tracking list and print them */
    mag_tensor_node_t** head = &ctx->rc_tracked;
    mag_tensor_node_t* curr = *head;
    uint32_t nleaked = 0;
    for (; curr; curr = curr->next, ++nleaked) {
        mag_log_error("Leaked tensor detected: %p, RCS: %u, RCW: %u, CTOR: %s", curr->tensor, curr->tensor->rcb.rc_strong, curr->tensor->rcb.rc_weak, curr->tensor->rcb.dtor ? "Y" : "N");
        mag_tensor_print(curr->tensor, true, false);
    }
    if (nleaked) mag_log_error("Leaked tensors detected: %u", nleaked);
    else mag_log_info("No leaked tensors detected.");
    curr = *head;
    while (curr) { /* Free tracking list */
        mag_tensor_node_t* tmp = curr;
        curr = curr->next;
        mag_tensor_destroy(tmp->tensor);
        (*mag_alloc)(tmp, 0);
    }
    *head = NULL;
#endif
    mag_fixed_intrusive_pool_print_info(&ctx->tensor_pool, "Tensor Hull Pool");
    mag_fixed_intrusive_pool_destroy(&ctx->tensor_pool);
    mag_destroy_dynamic_device(ctx->device); ctx->device = NULL;
    memset(ctx, 0, sizeof(*ctx));
    (*mag_alloc)(ctx, 0);
    ctx = NULL;
    mag_log_info("magnetron context destroyed.");
}

mag_exec_mode_t mag_ctx_get_exec_mode(const mag_ctx_t* ctx) { return ctx->exec_mode; }

void mag_ctx_set_exec_mode(mag_ctx_t* ctx, mag_exec_mode_t mode) {
    ctx->exec_mode = mode;
    mag_log_info("Execution mode set to: %s", mode == MAG_EXEC_MODE_EAGER ? "Eager" : "Deferred");
}

mag_prng_algorithm_t mag_ctx_get_prng_algorithm(const mag_ctx_t* ctx) { return ctx->prng_algo; }

void mag_ctx_set_prng_algorithm(mag_ctx_t* ctx, mag_prng_algorithm_t algorithm, uint64_t seed) {
    ctx->prng_algo = algorithm;
}

mag_compute_device_type_t mag_ctx_get_compute_device_type(const mag_ctx_t* ctx) { return ctx->device_type; }
const char* mag_ctx_get_compute_device_name(const mag_ctx_t* ctx) { return ctx->device->name; }
const char* mag_ctx_get_os_name(const mag_ctx_t* ctx) { return ctx->sys.os_name; }
const char* mag_ctx_get_cpu_name(const mag_ctx_t* ctx) { return ctx->sys.cpu_name; }
uint32_t mag_ctx_get_cpu_virtual_cores(const mag_ctx_t* ctx) { return ctx->sys.cpu_virtual_cores; }
uint32_t mag_ctx_get_cpu_physical_cores(const mag_ctx_t* ctx) { return ctx->sys.cpu_physical_cores; }
uint32_t mag_ctx_get_cpu_sockets(const mag_ctx_t* ctx) { return ctx->sys.cpu_sockets; }
uint64_t mag_ctx_get_physical_memory_total(const mag_ctx_t* ctx) { return ctx->sys.phys_mem_total; }
uint64_t mag_ctx_get_physical_memory_free(const mag_ctx_t* ctx) { return ctx->sys.phys_mem_free; }
bool mag_ctx_is_numa_system(const mag_ctx_t* ctx) { return false; /* TODO */ }
size_t mag_ctx_get_total_tensors_created(const mag_ctx_t* ctx) { return 0; /* TODO */ }

void mag_thread_set_prio(mag_thread_sched_prio_t prio) {
#ifdef _WIN32
    DWORD policy = THREAD_PRIORITY_NORMAL;
    switch (prio) {
        case MAG_THREAD_SCHED_PRIO_NORMAL: policy = THREAD_PRIORITY_NORMAL; break;
        case MAG_THREAD_SCHED_PRIO_MEDIUM: policy = THREAD_PRIORITY_ABOVE_NORMAL; break;
        case MAG_THREAD_SCHED_PRIO_HIGH: policy = THREAD_PRIORITY_HIGHEST; break;
        case MAG_THREAD_SCHED_PRIO_REALTIME: policy = THREAD_PRIORITY_TIME_CRITICAL; break;
    }
    if (mag_unlikely(!SetThreadPriority(GetCurrentThread(), policy))) {
        mag_log_warn("Failed to set thread scheduling priority: %d", prio);
    }
#else
    int32_t policy = SCHED_OTHER;
    struct sched_param p;
    switch (prio) {
        case MAG_THREAD_SCHED_PRIO_NORMAL: p.sched_priority = 0;  policy = SCHED_OTHER; break;
        case MAG_THREAD_SCHED_PRIO_MEDIUM: p.sched_priority = 40; policy = SCHED_FIFO; break;
        case MAG_THREAD_SCHED_PRIO_HIGH: p.sched_priority = 80; policy = SCHED_FIFO; break;
        case MAG_THREAD_SCHED_PRIO_REALTIME: p.sched_priority = 90; policy = SCHED_FIFO; break;
    }
    int status = pthread_setschedparam(pthread_self(), policy, &p);
    if (mag_unlikely(status)) {
        mag_log_warn("Failed to set thread scheduling priority: %d, error: %x", prio, status);
    }
    #endif
}

void mag_thread_set_name(const char* name) {
    #if defined(__linux__)
        prctl(PR_SET_NAME, name);
    #elif defined(__APPLE__) && defined(__MACH__)
        pthread_setname_np(name);
    #endif
}

void mag_thread_yield(void) {
    #if defined(_WIN32)
        YieldProcessor();
    #else
        sched_yield();
    #endif
}

static mag_intrusive_chunk* mag_fixed_pool_chunk_new(size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    size_t cap = blocks_per_chunk*block_size;
    uintptr_t size = 0;
    mag_pincr((void**)&size, sizeof(mag_intrusive_chunk), __alignof(mag_intrusive_chunk));
    mag_pincr((void**)&size, cap, block_align);
    void* base = (*mag_alloc)(NULL, size), *pos = base;
    mag_intrusive_chunk* chunk = mag_pincr(&pos, sizeof(mag_intrusive_chunk), __alignof(mag_intrusive_chunk));
    uint8_t* bot = mag_pincr(&pos, cap, block_align);
    *chunk = (mag_intrusive_chunk) {
        .bot = bot,
        .top = bot+cap,
        .next = NULL
    };
    return chunk;
}

void mag_fixed_intrusive_pool_init(mag_fixed_intrusive_pool* pool, size_t block_size, size_t block_align, size_t blocks_per_chunk) {
    mag_assert2(blocks_per_chunk);
    block_size = mag_xmax(sizeof(void*), block_size); /* Ensure block size is at least sizeof(void*) to store intrusive free list. */
    mag_intrusive_chunk* chunk = mag_fixed_pool_chunk_new(block_size, block_align, blocks_per_chunk);
    *pool = (mag_fixed_intrusive_pool) {
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

void* mag_fixed_intrusive_pool_malloc(mag_fixed_intrusive_pool* pool) {
    ++pool->num_allocs;
    if (mag_likely(pool->free_list)) { /* 1. Try to pop from free_list (fastest path) */
        ++pool->num_freelist_hits;
        void* blk = pool->free_list;
        pool->free_list = *(void**)blk; /* Next free block is stored at block [0..sizeof(void*)-1] */
        return blk;
    }
    mag_intrusive_chunk* chunk = pool->chunk_head;
    mag_assert2(chunk);
    uint8_t* top = chunk->top-pool->block_size;
    if (mag_likely(top >= chunk->bot)) {  /* 2. Allocate from the last pool if possible (fast path) */
        ++pool->num_pool_hits;
        chunk->top = top;
        return top;
    }
    mag_intrusive_chunk* new_chunk = mag_fixed_pool_chunk_new(pool->block_size, pool->block_align, pool->blocks_per_chunk);     /* 3. Current chunk is exhausted, allocate new (slow path) */
    chunk->next = new_chunk;
    pool->chunk_head = new_chunk;
    new_chunk->top -= pool->block_size;
    ++pool->num_chunks;
    return new_chunk->top;
}

void mag_fixed_intrusive_pool_free(mag_fixed_intrusive_pool* pool, void* blk) { /* Push chunk into free list */
    *(void**)blk = pool->free_list;
    pool->free_list = blk;
}

void mag_fixed_intrusive_pool_destroy(mag_fixed_intrusive_pool* pool) {
    mag_intrusive_chunk* chunk = pool->chunks;
    while (chunk) {
        mag_intrusive_chunk* next = chunk->next;
        (*mag_alloc)(chunk, 0);
        chunk = next;
    }
    memset(pool, 0, sizeof(*pool));
}

MAG_COLDPROC void mag_fixed_intrusive_pool_print_info(mag_fixed_intrusive_pool* pool, const char* name) {
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
    double mem_alloced, pool_mem;
    const char* mem_unit_alloced, *mem_unit_pool;
    mag_humanize_memory_size(pool->num_chunks*pool->blocks_per_chunk*pool->block_size, &mem_alloced, &mem_unit_alloced);
    mag_humanize_memory_size(pool->num_allocs*pool->block_size, &pool_mem, &mem_unit_pool);
    mag_log_info("\t Real Mem Allocated: %.03f %s, Total Pool Mem %.03f %s", mem_alloced, mem_unit_alloced, pool_mem, mem_unit_pool);
}

void mag_prng_init(mag_prng_state_t* rng, mag_prng_algorithm_t algo, uint64_t seed) {
    seed = seed ? seed : 0x853c49e6748fea9bull; /* Default seed. */
    rng->algo = algo;
    switch (algo) {
    case MAG_PRNG_MERSENNE_TWISTER: {
            uint32_t* state = rng->state.mersenne.state;
            *state = (uint32_t)seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1] ^ (state[i-1] >> 30))*1812433253 + i) & ~0u;
            rng->state.mersenne.next = 0;
            rng->state.mersenne.remaining = 1;
    } break;
    case MAG_PRNG_PCG: {
            rng->state.pcg.state = seed ^ 0x853c49e6748fea9bull;
            rng->state.pcg.inc = 0xda3e39cb94b95bdbull;
    } break;
    default:
        mag_panic("Unknown PRNG algorithm: %d", rng->algo);
    }
}

void mag_ctx_profile_start_recording(mag_ctx_t* ctx) {
    if (ctx->profiler_enabled) return;
    memset(ctx->op_perf_mons_total, 0, sizeof(ctx->op_perf_mons_total));
    ctx->profiler_enabled = true;
}

typedef struct mag_op_perf_record_t {
    mag_op_perf_info_t perf;
    mag_op_t op;
} mag_op_perf_record_t;

static int mag_cmp_perf_info(const void* x, const void* y) {
    const mag_op_perf_record_t* op1 = (const mag_op_perf_record_t *)x;
    const mag_op_perf_record_t* op2 = (const mag_op_perf_record_t *)y;
    if (op1->perf.elapsed_ns_acc < op2->perf.elapsed_ns_acc) return 1;
    if (op1->perf.elapsed_ns_acc > op2->perf.elapsed_ns_acc) return -1;
    return 0;
}

void mag_ctx_profile_stop_recording(mag_ctx_t* ctx, const char* export_csv_file) {
    mag_assert(ctx->profiler_enabled, "Profiler must be enabled to generate report");
    ctx->profiler_enabled = false;
    bool csv = export_csv_file && *export_csv_file;
    if (!csv) {
        mag_print_separator(stdout);
        printf("OS/Kernel: %s\n", ctx->sys.os_name);
        printf("CPU: %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u\n", ctx->sys.cpu_name, ctx->sys.cpu_virtual_cores, ctx->sys.cpu_physical_cores, ctx->sys.cpu_sockets);
        #if defined(__x86_64__) || defined(_M_X64) /* Print CPU features for x86-64 platforms. */
        printf("CPU Features:");
        for (unsigned i=0, k=0; i < MAG_X86_64_FEATURE__NUM; ++i) {
            if (mag_ctx_x86_64_cpu_has_feature(ctx, i)) {
                if (k++ % 8 == 0) printf("\n\t");
                printf("%s ", mag_x86_64_feature_names[i]);
            }
        }
        putchar('\n');
        #endif
        double mem_total, mem_free, mem_used;
        const char* mem_unit_total, *mem_unit_free, *mem_unit_used;
        mag_humanize_memory_size(ctx->sys.phys_mem_total, &mem_total, &mem_unit_total);
        mag_humanize_memory_size(ctx->sys.phys_mem_free, &mem_free, &mem_unit_free);
        mag_humanize_memory_size((size_t)llabs((int64_t)ctx->sys.phys_mem_total-(int64_t)ctx->sys.phys_mem_free), &mem_used, &mem_unit_used);
        double mem_used_percent = fabs((double)(ctx->sys.phys_mem_total-ctx->sys.phys_mem_free))/(double)ctx->sys.phys_mem_total*100.0;
        printf("Physical memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)\n", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);
        mag_print_separator(stdout);
        printf("%16s %16s %16s %16s %16s\n", "Operation", "Executions", "Usage (%)", "AVG Time (μs)", "Total Time (μs)");
    }
    mag_op_perf_record_t sorted[MAG_OP__NUM];
    uint64_t exec_total = 0;
    for (mag_op_t op=MAG_OP_NOP; op < MAG_OP__NUM; ++op) { /* Convert to sortable record. */
        sorted[op].op = op;
        sorted[op].perf = ctx->op_perf_mons_total[op];
        exec_total += sorted[op].perf.n_execs;
    }
    if (mag_unlikely(!exec_total) && !csv) {
        printf("\n! No operations profiled. Enable profiler and execute any operation to see results.\n");
        mag_print_separator(stdout);
        return;
    }
    qsort(sorted, MAG_OP__NUM, sizeof(*sorted), &mag_cmp_perf_info); /* Quicksort by time descending. */
    FILE* f = NULL;
    if (csv) {
        f = mag_fopen(export_csv_file, "wt");
        mag_assert(f, "Failed to open CSV file: %s", export_csv_file);
        fprintf(f, "Operation,Executions,Usage,AVG Time,Total Time\n"); /* CSV Header */
    }
    for (mag_op_t i=MAG_OP_NOP; i < MAG_OP__NUM; ++i) { /* Format sorted performance data */
        const mag_op_perf_record_t* info = sorted+i;
        const mag_op_perf_info_t* perf = &info->perf;
        if (!perf->n_execs) continue; /* Op never executed. */
        const char* op_name = mag_op_meta_of(info->op)->mnemonic;
        double perc_exec = (double)perf->n_execs/(double)exec_total * 100.0;
        char perc_exec_str[64];
        snprintf(perc_exec_str, sizeof(perc_exec_str), "%.1f", perc_exec);
        double avg_time = (double)perf->elapsed_ns_acc/1e3/(double)perf->n_execs;
        char avg_time_str[64];
        snprintf(avg_time_str, sizeof(avg_time_str), "%f", avg_time);
        double tot_time = (double)perf->elapsed_ns_acc/1e3;
        char tot_time_str[64];
        snprintf(tot_time_str, sizeof(tot_time_str), "%f", tot_time);
        if (csv) {
            fprintf(f, "%s,%" PRIu64 ",%s,%s,%s\n", op_name, perf->n_execs, perc_exec_str, avg_time_str, tot_time_str);
        } else {
            printf("%16s %16" PRIu64 " %16s%16s%16s\n", op_name, perf->n_execs, perc_exec_str, avg_time_str, tot_time_str);
        }
    }
    if (csv) fclose(f);
    else {
        putchar('\n');
        printf("Total operations profiled: %" PRIu64 "\n", exec_total);
        mag_print_separator(stdout);
    }
}

uint32_t mag_pack_color_u8(uint8_t r, uint8_t g, uint8_t b) { return ((uint32_t)r<<16)|((uint32_t)g<<8)|(uint32_t)b; }
uint32_t mag_pack_color_f32(float r, float g, float b) {
    return (((uint32_t)(r*255.0f)&255)<<16)|(((uint32_t)(g*255.0f)&255)<<8)|((uint32_t)(b*255.0f)&255);
}

const char* mag_device_type_get_name(mag_compute_device_type_t op) {
    static const char* const names[MAG_COMPUTE_DEVICE_TYPE__NUM] = {
        [MAG_COMPUTE_DEVICE_TYPE_CPU] = "CPU",
        [MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA] = "CUDA GPU",
    };
    return names[op];
}

const mag_dtype_meta_t* mag_dtype_meta_of(mag_dtype_t type) {
    static const mag_dtype_meta_t infos[MAG_DTYPE__NUM] = {
        [MAG_DTYPE_F32] = {
            sizeof(float),
            "f32"
        },
    };
    return &infos[type];
}

/*
**  validation error print template
**
**
printf("SHORT ERROR DESCRIPTION"
        "ERROR: Failed to execute operation: %s.\n"
        "    - Input Tensor 1 '%s': MISMATCHES\n"
        "    - Input Tensor 2 '%s': MISMATCHES\n"
        "    Hint: ANY HINT FOR USR."
);
*/

static bool mag_validate_inputs(mag_op_t op, mag_tensor_t** inputs, uint32_t numin) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_unlikely(meta->argcount != numin || numin > MAG_MAX_INPUT_TENSORS)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires %u input tensors, but %u were provided.\n"
            "    Hint: Ensure the correct number of input tensors are provided.\n",
            meta->mnemonic, meta->argcount, numin
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    for (uint32_t i=0; i < meta->argcount; ++i) {
        if (mag_unlikely(!inputs[i])) {
            mag_print_separator(stderr);
            fprintf(stderr,
                "Failed to execute operation: %s.\n"
                "ERROR: Input tensor %u is NULL.\n"
                "    Hint: Ensure all input tensors are valid and non-NULL.\n",
                meta->mnemonic, i
            );
            mag_print_separator(stderr);
            fputc('\n', stderr);
            fflush(stderr);
            return false;
        }
    }
    return true;
}

static bool mag_validate_op_params(mag_op_t op, const mag_op_param_t* params, uint32_t numparams) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (!meta->paramcount) return true; /* No parameters to validate. */
    if (mag_unlikely(meta->paramcount != numparams || numparams > MAG_MAX_OP_PARAMS)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires at most %u parameters, but %u were provided.\n"
            "    Hint: Ensure the correct number of operation parameters are provided.\n",
            meta->mnemonic, MAG_MAX_OP_PARAMS, meta->paramcount
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    if (mag_unlikely(!params)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation parameters are NULL.\n"
            "    Hint: Ensure all operation parameters are valid and non-NULL.\n",
            meta->mnemonic
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    for (uint32_t i=0; i < meta->paramcount; ++i) {
        if (mag_unlikely(params[i].type != meta->param_types[i])) {
            mag_print_separator(stderr);
            fprintf(stderr,
                "Failed to execute operation: %s.\n"
                "ERROR: Operation parameter %u type mismatch.\n"
                "    - Expected type id: %d\n"
                "    - Provided type id: %d\n"
                "    Hint: Ensure the correct parameter types are provided.\n",
                meta->mnemonic, i, meta->param_types[i], params[i].type
            );
        }
    }
    return true;
}

static bool mag_validate_shape_eq(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_is_shape_eq(a, b))) return true;
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_dims(&shape_1, &a->shape, a->rank);
    mag_fmt_dims(&shape_2, &b->shape, b->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes must be equal.\n"
        "    - Input Tensor 1 '%s' Shape: %s\n"
        "    - Input Tensor 2 '%s' Shape: %s\n"
        "    Hint: Adjust tensor shapes using transposition or permutation.\n",
        meta->mnemonic,
        a->name, shape_1,
        b->name, shape_2
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

static bool mag_validate_shape_broadcastable(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) { /* Check if tensor shapes are broadcast-able. (b into a) */
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_can_broadcast(b, a))) return true;
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_dims(&shape_1, &a->shape, a->rank);
    mag_fmt_dims(&shape_2, &b->shape, b->rank);
    char broadcast_able_str[MAG_MAX_DIMS*2+4+1] = {0};
    char* p = broadcast_able_str;
    *p++ = '[';
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i) {
        *p++ = a->shape[i] % b->shape[i] == 0 ? 'Y' : 'N';
        *p++ = i < MAG_MAX_DIMS-1 ? ',' : ']';
    }
    *p = '\0';
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes must be broadcast-able.\n"
        "    - Input Tensor 1 '%s' Shape: %s\n"
        "    - Input Tensor 2 '%s' Shape: %s\n"
        "    Broadcast-able: %s\n"
        "    Hint: Adjust tensor shapes using transposition or permutation.\n",
        meta->mnemonic,
        a->name, shape_1,
        b->name, shape_2,
        broadcast_able_str
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

#define mag_validate_expr_gen(expr, message, ...) \
    if (mag_unlikely(!(expr))) { \
        if (1) { \
           mag_log_error(message, ## __VA_ARGS__); \
        } \
        return false; \
    }

static bool mag_validate_op_unary(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    if (mag_unlikely(!mag_validate_shape_eq(op, result, inputs[0]))) return false;
    return true;
}

static bool mag_validate_op_binary(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    if (mag_unlikely(!mag_validate_shape_eq(op, result, inputs[0]))) return false;
    if (mag_unlikely(!mag_validate_shape_broadcastable(op, inputs[0], inputs[1]))) return false;
    mag_validate_expr_gen(mag_tensor_is_contiguous(result), "Result must be contiguous.");
    mag_validate_expr_gen(mag_tensor_is_contiguous(inputs[0]), "First tensor must be contiguous.");
    return true;
}

static bool mag_validate_op_transpose(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return true;
}

static bool mag_validate_op_scalar(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    mag_validate_expr_gen(mag_tensor_is_contiguous(inputs[0]), "Mean"); /* TODO */
    mag_validate_expr_gen(result->shape[0] == 1, "Mean");
    mag_validate_expr_gen(result->shape[1] == inputs[0]->shape[1], "Mean");
    mag_validate_expr_gen(result->shape[2] == inputs[0]->shape[2], "Mean");
    mag_validate_expr_gen(result->shape[3] == inputs[0]->shape[3], "Mean");
    return true;
}

static bool mag_validate_op_matmul(mag_op_t op, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    mag_validate_expr_gen(inputs[0]->shape[1] == inputs[1]->shape[0], "Input tensor shapes must be compatible for matrix multiplication.");
    mag_validate_expr_gen(mag_tensor_is_contiguous(inputs[0]), "First tensor must be contiguous.");
    mag_validate_expr_gen(mag_tensor_is_contiguous(inputs[1]), "Second tensor must be contiguous.");
    mag_validate_expr_gen(inputs[1]->shape[2] % inputs[0]->shape[2] == 0, "Result tensor shape mismatch.");
    mag_validate_expr_gen(inputs[1]->shape[3] % inputs[0]->shape[3] == 0, "Result tensor shape mismatch.");
    return true;
}

static mag_tensor_t* mag_tensor_create(mag_ctx_t* ctx, mag_dtype_t type, const int64_t* dims, int64_t rank, mag_tensor_t* view, size_t view_offs);

static mag_tensor_t* mag_result_constructor_routine_isomorph(mag_tensor_t** inputs, const mag_op_param_t* params) {
    (void)params;
    return mag_tensor_create(inputs[0]->ctx, inputs[0]->dtype, inputs[0]->shape, inputs[0]->rank, NULL, 0);
}

static mag_tensor_t* mag_result_constructor_routine_view(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    (void)params;
    return mag_tensor_create(inputs[0]->ctx, inputs[0]->dtype, inputs[0]->shape, MAG_MAX_DIMS, inputs[0], 0);
}

static mag_tensor_t* mag_result_constructor_routine_scalar(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    int64_t shape[MAG_MAX_DIMS];
    *shape = 1;
    #pragma GCC unroll 5
    for (uint32_t i=1; i < MAG_MAX_DIMS; ++i)
        shape[i] = inputs[0]->shape[i];
    return mag_tensor_create(inputs[0]->ctx, inputs[0]->dtype, shape, MAG_MAX_DIMS, NULL, 0);
}

static mag_tensor_t* mag_result_constructor_routine_transposed(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_tensor_t* transposed = mag_result_constructor_routine_view(inputs, params);
    mag_swap(int64_t, transposed->shape[0], transposed->shape[1]);
    mag_swap(int64_t, transposed->strides[0], transposed->strides[1]);
    return transposed;
}

static mag_tensor_t* mag_result_constructor_routine_permuted(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_assert2(params != NULL); /* TODO */
    mag_tensor_t* permuted = mag_result_constructor_routine_view(inputs, params);
    uint32_t axes[MAG_MAX_DIMS];
    for (uint32_t i = 0; i < MAG_MAX_DIMS; ++i) /* Unpack axes */
        axes[i] = params[i].x.u32;
    for (uint32_t i = 0; i < MAG_MAX_DIMS; ++i) { /* Check that all axes are unique */
        for (uint32_t j = i+1; j < MAG_MAX_DIMS; ++j)
            mag_assert(axes[i] != axes[j], "Axes must be unique: %zu != %zu", axes[i], axes[j]);
    }
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i) { /* Permute shape and strides */
        mag_assert2(axes[i] >= 0 && axes[i] < MAG_MAX_DIMS);
        permuted->shape[axes[i]] = inputs[0]->shape[i];
        permuted->strides[axes[i]] = inputs[0]->strides[i];
    }
    return permuted;
}

static mag_tensor_t* mag_result_constructor_routine_matmul(mag_tensor_t** inputs,  const mag_op_param_t* params) { /* MxR = MxN * NxR */
    (void)params;
    int64_t shape[MAG_MAX_DIMS];
    shape[0] = inputs[0]->shape[0]; /* M */
    shape[1] = inputs[1]->shape[1]; /* R */
    return mag_tensor_create(inputs[0]->ctx, MAG_DTYPE_F32, shape, 2, NULL, 0);
}

const mag_op_meta_t* mag_op_meta_of(mag_op_t type) {
    static const mag_op_meta_t infos[MAG_OP__NUM] = {
        [MAG_OP_NOP] = {
            .mnemonic = "nop",
            .argcount = 0,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = NULL,
            .validator = NULL
        },
        [MAG_OP_CLONE] = {
            .mnemonic = "clone",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_VIEW] = {
            .mnemonic = "view",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_view,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_TRANSPOSE] = {
            .mnemonic = "transpose",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_transposed,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_PERMUTE] = {
            .mnemonic = "permute",
            .argcount = 1,
            .paramcount = MAG_MAX_DIMS,
            .param_types = {
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
                MAG_OP_TPARAM_U32,
            },
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_permuted,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_MEAN] = {
            .mnemonic = "mean",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_MIN] = {
            .mnemonic = "min",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_MAX] = {
            .mnemonic = "max",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_SUM] = {
            .mnemonic = "sum",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = false,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_ABS] = {
            .mnemonic = "abs",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_NEG] = {
            .mnemonic = "neg",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_LOG] = {
            .mnemonic = "log",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SQR] = {
            .mnemonic = "sqr",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SQRT] = {
            .mnemonic = "sqrt",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SIN] = {
            .mnemonic = "sin",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_COS] = {
            .mnemonic = "cos",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_STEP] = {
            .mnemonic = "step",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SOFTMAX] = {
            .mnemonic = "softmax",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SOFTMAX_DV] = {
            .mnemonic = "softmax_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SIGMOID] = {
            .mnemonic = "sigmoid",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SIGMOID_DV] = {
            .mnemonic = "sigmoid_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_HARD_SIGMOID] = {
            .mnemonic = "hard_sigmoid",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SILU] = {
            .mnemonic = "silu",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SILU_DV] = {
            .mnemonic = "silu_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_TANH] = {
            .mnemonic = "tanh",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_TANH_DV] = {
            .mnemonic = "tanh_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_RELU] = {
            .mnemonic = "relu",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_RELU_DV] = {
            .mnemonic = "relu_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_GELU] = {
            .mnemonic = "gelu",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_GELU_DV] = {
            .mnemonic = "gelu_dv",
            .argcount = 1,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_ADD] = {
            .mnemonic = "add",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary
        },
        [MAG_OP_SUB] = {
            .mnemonic = "sub",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary
        },
        [MAG_OP_MUL] = {
            .mnemonic = "mul",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary
        },
        [MAG_OP_DIV] = {
            .mnemonic = "div",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {MAG_OP_TPARAM_NONE},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary
        },
        [MAG_OP_ADDS] = {
            .mnemonic = "adds",
            .argcount = 1,
            .paramcount = 1,
            .param_types = {MAG_OP_TPARAM_F32},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_SUBS] = {
            .mnemonic = "subs",
            .argcount = 1,
            .paramcount = 1,
            .param_types = {MAG_OP_TPARAM_F32},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_MULS] = {
            .mnemonic = "muls",
            .argcount = 1,
            .paramcount = 1,
            .param_types = {MAG_OP_TPARAM_F32},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_DIVS] = {
            .mnemonic = "divs",
            .argcount = 1,
            .paramcount = 1,
            .param_types = {MAG_OP_TPARAM_F32},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_MATMUL] = {
            .mnemonic = "matmul",
            .argcount = 2,
            .paramcount = 0,
            .param_types = {},
            .inplace = true,
            .r_alloc = &mag_result_constructor_routine_matmul,
            .validator = &mag_validate_op_matmul
        }
    };
    return infos+type;
}

#undef mag_validate_inputs

int64_t mag_tensor_data_size(const mag_tensor_t* t) { return t->numel*mag_dtype_meta_of(t->dtype)->size; }
int64_t mag_tensor_numel(const mag_tensor_t* t) { return t->numel; }
int64_t mag_tensor_num_rows(const mag_tensor_t* t) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, d, shape);
    return d1*d2*d3*d4*d5;
}
int64_t mag_tensor_num_cols(const mag_tensor_t* t) { return *t->shape; }

#ifdef MAG_DEBUG
    static void mag_tensor_sanitize_dtor(mag_tensor_t* t) {
        (void)t;
    }
#endif

static mag_tensor_t* mag_tensor_create(mag_ctx_t* ctx, mag_dtype_t type, const int64_t* dims, int64_t rank, mag_tensor_t* view, size_t view_offs) {
    uintptr_t tr_id = mag_thread_id();
    mag_assert(tr_id == ctx->tr_id, "%" PRIx64 " != %" PRIx64 " Tensor must be created on the same thread as the context.", tr_id, ctx->tr_id);
    mag_assert(dims != NULL && rank >= 0 && rank <= MAG_MAX_DIMS, "Rank must be within (0, %d]", MAG_MAX_DIMS);
    mag_assert2(view_offs == 0); /* NYI. TODO */
    if (view) {
        if (view->view_uplink) { /* Traverse view chain and accumulate offset */
            view_offs += view->view_offs;
            view = view->view_uplink;
        }
        mag_tensor_incref(view); /* Increment view tensor strong RC */
    }
    int64_t dts = mag_dtype_meta_of(type)->size;
    int64_t numel = 1;
    for (int64_t i=0; i < rank; ++i) /* Calculate buffer size and check for overflow. */
        mag_assert2(dims[i] > 0 && !mag_imull64_ov(dims[i], numel, &numel)); /* Overflow in buffer size. Max: INT64_MAX. Reduce dimensions. */
    int64_t numbytes = numel*dts;
    mag_assert2(!view || !numbytes || numbytes + view_offs <= mag_tensor_data_size(view)); /* Slice must be within viewed tensor data range. *//* Allocate memory for tensor struct on CPU RAM. */
    mag_tensor_t* t = mag_fixed_intrusive_pool_malloc(&ctx->tensor_pool);
    memset(t, 0, sizeof(*t));
    *t = (mag_tensor_t) {
        .rcb = {
            .rc_strong = 0,
            .rc_weak = 0,
            #ifdef MAG_DEBUG
                .dtor = &mag_tensor_sanitize_dtor
            #endif
        },
        .ctx = ctx,
        .rank = rank,
        .shape = {0},
        .strides = {0},
        .dtype = type,
        .storage = {0},
        .numel = numel,
        .flags = view ? MAG_TFLAG_VIEW : MAG_TFLAG_OWNER,
        .op = MAG_OP_NOP,
        .op_inputs = {0},
        .op_params = {{0}},
        .view_uplink = view,
        .view_offs = view_offs,
        .grad = NULL,
        .pmon = {0},
        .name = "",
        .ud = NULL
    };
    mag_tensor_incref(t); /* First strong RC=1 */
    /* Allocate device memory */
    mag_compute_device_t* dvc = ctx->device;
    void (*allocator)(mag_compute_device_t*, mag_storage_buffer_t*, size_t) = dvc->alloc_storage;
    if (view) t->storage = view->storage; /* Reference memory from view */
    else (*allocator)(dvc, &t->storage, numbytes); /* Allocate new device memory */
    #pragma GCC unroll 6
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i)    /* Copy dimensions and set unused to identity. */
        t->shape[i] = i < rank ? dims[i] : 1;
    *t->strides = 1;
    #pragma GCC unroll 5
    for (uint32_t i=1; i < MAG_MAX_DIMS; ++i)    /* Calculate strides and check for overflow. */
        mag_assert2(!mag_imull64_ov(t->strides[i-1], t->shape[i-1], t->strides+i));
#ifdef MAG_DEBUG /* If tensor RC sanitize is enabled, insert into tracking list */
    mag_tensor_node_t** head = &ctx->rc_tracked;
    mag_tensor_node_t* node = (*mag_alloc)(NULL, sizeof(*node));
    *node = (mag_tensor_node_t) {
        .tensor = t,
        .next = *head
    };
    *head = node;
#endif
    return t;
}

static void mag_tensor_destroy(mag_tensor_t* t) {
    mag_ctx_t* ctx = t->ctx;
#ifdef MAG_DEBUG  /* If tensor RC sanitize is enabled, invoke destructor and erase from tracking list */
    void (*dtor)(mag_tensor_t*) = t->rcb.dtor;  /* Invoke Debug destructor. */
    if (dtor) (*dtor)(t);
    mag_tensor_node_t** head = &ctx->rc_tracked;
    if (*head) {
        mag_tensor_node_t* curr = *head, *prev = NULL;
        if (curr->tensor == t) { /* Head itself holds key */
            *head = curr->next;
            (*mag_alloc)(curr, 0);
        } else {
            while (curr && curr->tensor != t) { /* Find node */
                prev = curr;
                curr = curr->next;
            }
            if (curr) { /* Found node */
                prev->next = curr->next;
                (*mag_alloc)(curr, 0); /* Free node */
            }
        }
    }
#endif
    if (t->flags & MAG_TFLAG_OWNER) { /* Free device memory if tensor owns it. */
        mag_compute_device_t* dvc = t->ctx->device;
        void (*dtor)(mag_compute_device_t*, mag_storage_buffer_t*) = dvc->free_storage;
        (*dtor)(dvc, &t->storage);
    }
    mag_fixed_intrusive_pool_free(&ctx->tensor_pool, t);
}

void mag_tensor_incref(mag_tensor_t* t) {
    mag_assert2(t->rcb.rc_strong < UINT32_MAX);
    ++t->rcb.rc_strong;
}

bool mag_tensor_decref(mag_tensor_t* t) {
    if (t->view_uplink) { /* If tensor is a view, decrement base RC and free tensor chain */
        mag_tensor_decref(t->view_uplink);
    }
    if (!--t->rcb.rc_strong) { /* Strong RC reaches zero, destroy. */
        mag_tensor_destroy(t);
        return true;
    }
    return false;
}

mag_tensor_t* mag_tensor_create_1d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1}, 1, NULL, 0);
}

mag_tensor_t* mag_tensor_create_2d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2}, 2, NULL, 0);
}

mag_tensor_t* mag_tensor_create_3d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2, d3}, 3, NULL, 0);
}

mag_tensor_t* mag_tensor_create_4d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4}, 4, NULL, 0);
}

mag_tensor_t* mag_tensor_create_5d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4, d5}, 5, NULL, 0);
}

mag_tensor_t* mag_tensor_create_6d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6) {
    return mag_tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4, d5, d6}, 6, NULL, 0);
}

static void MAG_HOTPROC mag_op_exec(mag_tensor_t* R, mag_compute_device_t* dvc, mag_graph_eval_order_t ord) {
    mag_perf_mon_t* pmon = &R->pmon;
    mag_op_perf_info_t (*pmon_ops)[MAG_OP__NUM] = &R->ctx->op_perf_mons_total;
    mag_op_perf_info_t* pmon_op = (*pmon_ops)+R->op;
    uint64_t start = R->ctx->profiler_enabled ? mag_hpc_clock_ns() : 0;    /* Profiling monitoring */
    void (*exec)(mag_compute_device_t*, mag_tensor_t*) = ord == MAG_GRAPH_EVAL_ORDER_FORWARD ? dvc->eager_exec_fwd : dvc->eager_exec_bwd;
    (*exec)(dvc, R); /* Dispatch to backend. */
    if (!R->ctx->profiler_enabled) return; /* Profiling disabled. */
    pmon->elapsed_ns = mag_hpc_clock_elapsed_ns(start);
    pmon->elapsed_ns_acc += pmon->elapsed_ns;
    pmon_op->elapsed_ns_acc += pmon->elapsed_ns;
    ++pmon->n_execs;
    ++pmon_op->n_execs;
}

static mag_tensor_t* MAG_HOTPROC mag_tensor_operator(
    mag_ctx_t* ctx,
    mag_op_t op,
    bool inplace,
    mag_tensor_t** inputs,
    uint32_t numin,
    const mag_op_param_t* params,
    uint32_t numparams
) {
    /* Validate inputs and params first */
    mag_assert2(op != MAG_OP_NOP);
    mag_assert(inputs && mag_validate_inputs(op, inputs, numin), "Invalid input tensors for operation %s.", mag_op_meta_of(op)->mnemonic);
    mag_assert(mag_validate_op_params(op, params, numparams), "Invalid parameters for operation %s.", mag_op_meta_of(op)->mnemonic);

    const mag_op_meta_t* meta = mag_op_meta_of(op);
    mag_graph_eval_order_t gra = MAG_GRA_FWD; /* TODO */
    mag_tensor_t* (*r_alloc)(mag_tensor_t**, const mag_op_param_t*) = meta->r_alloc;
    bool (*validate_op)(mag_op_t, mag_tensor_t*, mag_tensor_t**, const mag_op_param_t*) = meta->validator;
    mag_tensor_t* R = (inplace && numin && meta->inplace)                                                   /* Inplace requested? */
        ? mag_tensor_create(ctx, (*inputs)->dtype, (*inputs)->shape, (*inputs)->rank, *inputs, 0)  /* View R <- X for inplace aliasing op. */
        : (*r_alloc)(inputs, params);                                                                       /* Construct new result tensor. */
    if (mag_unlikely(!(*validate_op)(op, R, inputs, params))) return NULL;                                  /* Validation failed. */
    mag_tensor_t* grad = NULL;                                                                              /* ∇ᵦL = ∂L/∂B - Upper gradient tensor. */  /* TODO */
    if (gra == MAG_GRA_BWD && grad) {
        R->grad = R->grad                       /* ∇ₐL = ∑ᵢ (∂L/∂Bᵢ) ⋅ (∂Bᵢ/∂A) - Chain rule accumulate. */
            ? mag_add(R->grad, grad)      /* ∇ₐL <- ∇ₐL + ∇ᵦL */
            : mag_clone(grad);                  /* ∇ₐL <- ∂L/∂B */
    }
    mag_assert2(R->op == MAG_OP_NOP);
    R->op = op;                                                         /* Set operation for deferred execution mode. */
    for (uint32_t i=0; i < numin; ++i) {                             /* Set input tensors and flags. */
        R->op_inputs[i] = inputs[i];
    }
    if (params) memcpy(R->op_params, params, sizeof(*params));   /* Copy operation parameters */
    if (ctx->exec_mode == MAG_EXEC_MODE_EAGER) {                    /* In eager execution mode, we execute immediately. */
        mag_op_exec(R, ctx->device, gra);                           /* Execute the operation immediately. */
    }
    return R;
}

mag_tensor_t* mag_clone(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CLONE, false, &x, 1, NULL, 0);
}

mag_tensor_t* mag_view(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_VIEW, false, &x, 1, NULL, 0);
}

mag_tensor_t* mag_transpose(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TRANSPOSE, false, &x, 1, NULL, 0);
}

mag_tensor_t* mag_permute(mag_tensor_t* x, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t d4, uint32_t d5) {
    mag_op_param_t params[MAG_MAX_OP_PARAMS] = {
        {.type=MAG_OP_TPARAM_U32, .x.u32=d0},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d1},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d2},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d3},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d4},
        {.type=MAG_OP_TPARAM_U32, .x.u32=d5}
    };
    return mag_tensor_operator(x->ctx, MAG_OP_PERMUTE, false, &x, 1, params, sizeof(params)/sizeof(*params));
}

mag_tensor_t* mag_mean(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_MEAN, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_min(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_max(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sum(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SUM, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_abs(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_ABS, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_abs_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_ABS, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_neg(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_NEG, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_neg_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_NEG, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_log(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_LOG, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_log_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_LOG, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sqr(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SQR, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sqr_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SQR, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sqrt(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SQRT, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sqrt_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SQRT, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sin(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIN, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sin_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIN, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_cos(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_COS, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_cos_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_COS, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_step(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_STEP, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_step_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_STEP, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_softmax(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_softmax_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_softmax_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_softmax_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sigmoid(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sigmoid_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_sigmoid_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_sigmoid_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_hard_sigmoid(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_hard_sigmoid_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_silu(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SILU, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_silu_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SILU, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_silu_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_silu_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_tanh(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_TANH, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_tanh_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_TANH, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_tanh_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_tanh_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_relu(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_RELU, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_relu_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_RELU, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_relu_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_relu_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_gelu(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_GELU, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_gelu_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_GELU, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_gelu_dv(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, false, &x, 1, NULL, 0); }
mag_tensor_t* mag_gelu_dv_(mag_tensor_t* x) { return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, true, &x, 1, NULL, 0); }
mag_tensor_t* mag_add(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_ADD, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_add_(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_ADD, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_sub(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_SUB, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_sub_(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_SUB, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_mul(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_MUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_mul_(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_MUL, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_div(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_DIV, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }
mag_tensor_t* mag_div_(mag_tensor_t* x, mag_tensor_t* y) { return mag_tensor_operator(x->ctx, MAG_OP_DIV, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0); }

mag_tensor_t* mag_adds(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_ADDS, false, &x, 1, &param, 1);
}

mag_tensor_t* mag_adds_(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_ADDS, true, &x, 1, &param, 1);
}

mag_tensor_t* mag_subs(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_SUBS, false, &x, 1, &param, 1);
}

mag_tensor_t* mag_subs_(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_SUBS, true, &x, 1, &param, 1);
}

mag_tensor_t* mag_muls(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_MULS, false, &x, 1, &param, 1);
}

mag_tensor_t* mag_muls_(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_MULS, true, &x, 1, &param, 1);
}

mag_tensor_t* mag_divs(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_DIVS, false, &x, 1, &param, 1);
}

mag_tensor_t* mag_divs_(mag_tensor_t* x, float xi) {
    mag_op_param_t param = {.type=MAG_OP_TPARAM_F32, .x.f32=xi};
    return mag_tensor_operator(x->ctx, MAG_OP_DIVS, true, &x, 1, &param, 1);
}

mag_tensor_t* mag_matmul(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MATMUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0);
}

static MAG_AINLINE void mag_tensor_virtual_to_physical_index(const mag_tensor_t* t, int64_t v_idx, int64_t(*p_idx)[MAG_MAX_DIMS]) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, d, shape);
    (*p_idx)[5] = v_idx / (d4*d3*d2*d1*d0);
    (*p_idx)[4] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0) / (d3*d2*d1*d0);
    (*p_idx)[3] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0) / (d2*d1*d0);
    (*p_idx)[2] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0) / (d1*d0);
    (*p_idx)[1] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0) / d0;
    (*p_idx)[0] =  v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0 - (*p_idx)[1]*d0;
}

static MAG_AINLINE int64_t mag_tensor_physical_to_virtual_index(const mag_tensor_t* t, const int64_t (*p_idx)[MAG_MAX_DIMS]) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    mag_load_local_storage_group_arr(*p_idx, i);
    return s0*i0 + s1*i1 + s2*i2 + s3*i3 + s4*i4 + s5*i5;
}

mag_tensor_t* mag_tensor_get_arg(const mag_tensor_t* t, size_t slot) {
    mag_assert(slot < MAG_MAX_INPUT_TENSORS, "Slot must be within [0, %d)", MAG_MAX_INPUT_TENSORS);
    return t->op_inputs[slot];
}

void mag_tensor_set_arg(mag_tensor_t* t, size_t slot, mag_tensor_t* arg) {
    mag_assert(slot < MAG_MAX_INPUT_TENSORS, "Slot must be within [0, %d)", MAG_MAX_INPUT_TENSORS);
    mag_assert(t->op_inputs[slot] == NULL, "Argument at slot #%zu already set", slot);
    t->op_inputs[slot] = arg;
}

void mag_tensor_copy_buffer_from(mag_tensor_t* t, const void* data, size_t size) {
    mag_assert(size == (size_t) mag_tensor_data_size(t), "Buffer size mismatch: %zu != %lld", size, mag_tensor_data_size(t));
    mag_storage_buffer_t* sto = &t->storage;
    (*sto->cpy_host_device)(sto, 0, data, size);
}

void mag_tensor_fill(mag_tensor_t* t, float x) {
    mag_panic("NYI");
}

void mag_tensor_fill_random_uniform(mag_tensor_t* t, float min, float max) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_panic("NYI");
}

void mag_tensor_fill_random_normal(mag_tensor_t* t, float mean, float stddev) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    mag_panic("NYI");
}

uint64_t mag_tensor_get_packed_refcounts(const mag_tensor_t* t) {
    return (uint64_t)t->rcb.rc_strong|((uint64_t)t->rcb.rc_weak << 32);
}

void mag_tensor_retain(mag_tensor_t* t) {
    ++t->rcb.rc_strong;
}

size_t mag_tensor_get_memory_usage(const mag_tensor_t* t) {
    return sizeof(*t) + mag_tensor_data_size(t);
}

static void mag_print_tensor_recursive(FILE* f, const mag_tensor_t* t, int64_t (*idx)[MAG_MAX_DIMS], const int64_t (*stri)[MAG_MAX_DIMS], int64_t curr_dim, int64_t total_dims, int indent) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    int64_t dim_size = t->shape[curr_dim];
    mag_load_local_storage_group_arr(*stri, s);
    if (curr_dim == total_dims - 1) {
        fprintf(f, "%*s[", indent, "");
        for (int64_t i = 0; i < dim_size; ++i) {
            (*idx)[curr_dim] = i;
            mag_load_local_storage_group_arr(*idx, i);
            float val = *((const float*)t->storage.base + i0*s0 + i1*s1 + i2*s2 + i3*s3 + i4*s4 + i5*s5);
            fprintf(f, "%g", val);
            if (i < dim_size - 1) {
                fprintf(f, " ");
            }
        }
        fprintf(f, "]");
    } else {
        fprintf(f, "%*s[\n", indent, "");
        for (int64_t i = 0; i < dim_size; ++i) {
            (*idx)[curr_dim] = i;
            mag_print_tensor_recursive(f, t, idx, stri, curr_dim + 1, total_dims, indent + 1);
            if (i < dim_size - 1) {
                fprintf(f, ",\n");
            } else {
                fprintf(f, "\n");
            }
        }
        fprintf(f, "%*s]\n", indent, "");
    }
}


void mag_tensor_print(const mag_tensor_t* t, bool with_header, bool with_data) {
    mag_assert(t->dtype == MAG_DTYPE_F32, "Tensor must be F32");
    mag_assert2(with_header || with_data);
    mag_load_local_storage_group(t, x_d, shape);
    mag_load_local_storage_group(t, x_s, strides);
    FILE* f = stdout;
    if (with_header) {
        double buf_size_cvt = 0.0;
        const char* buf_size_unit = NULL;
        mag_humanize_memory_size(mag_tensor_get_memory_usage(t), &buf_size_cvt, &buf_size_unit);
        char shape[MAG_FMT_DIM_BUF_SIZE];
        char strides[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_dims(&shape, &t->shape, t->rank);
        mag_fmt_dims(&strides, &t->strides, MAG_MAX_DIMS);
        static const char* flag_abbrs = "OVGE";
        mag_assert2(strlen(flag_abbrs) == MAG_TFLAG_LEN);
        char flags[MAG_TFLAG_LEN+1] = {0};
        for (uint32_t i=0, k=0; i < MAG_TFLAG_LEN; ++i)
            if (t->flags & (1 << i))
                flags[k++] = flag_abbrs[i];
        flags[MAG_TFLAG_LEN] = '\0';
        fprintf(f, "Tensor '%s', DType: %s, Rank: %" PRIi64 ", Elements: %" PRIi64 ", Shape: %s, Strides: %s, Mem: %.03f %s, Flags: %s (%x)\n",
            t->name,
            mag_dtype_meta_of(t->dtype)->name,
            t->rank,
            mag_tensor_numel(t),
            shape,
            strides,
            buf_size_cvt,
            buf_size_unit,
            flags,
            t->flags
        );
    }
    if (with_data) {
        int64_t strides[MAG_MAX_DIMS];
        strides[MAG_MAX_DIMS-1] = 1;
        for (int32_t i = MAG_MAX_DIMS-2; i >= 0; --i)    // TODO: Fix this
            strides[i] = strides[i+1] * t->shape[i+1];
        int64_t idx[MAG_MAX_DIMS] = {0};
        mag_print_tensor_recursive(f, t, &idx, &strides, 0, t->rank, 0);
    }
}

void mag_tensor_set_name(mag_tensor_t* t, const char* name) {
    strncpy(t->name, name, MAG_MAX_TENSOR_NAME_LEN);
    t->name[MAG_MAX_TENSOR_NAME_LEN-1] = '\0';
}

void mag_tensor_fmt_name(mag_tensor_t* t, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(t->name, sizeof(t->name), fmt, args);
    va_end(args);
}

const char* mag_tensor_get_name(const mag_tensor_t* t) {
    return t->name;
}

int64_t mag_tensor_rank(const mag_tensor_t* t) { return t->rank; }
const int64_t* mag_tensor_shape(const mag_tensor_t* t) { return t->shape; }
const int64_t* mag_tensor_strides(const mag_tensor_t* t) { return t->strides; }
mag_dtype_t mag_tensor_dtype(const mag_tensor_t* t) { return t->dtype; }

void* mag_tensor_data_ptr(const mag_tensor_t* t) {
    return (void*)t->storage.base;
}

bool mag_tensor_is_scalar(const mag_tensor_t* t) {
    #pragma GCC unroll 6
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool mag_tensor_is_vector(const mag_tensor_t* t) {
    #pragma GCC unroll 5
    for (uint32_t i=1; i < MAG_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool mag_tensor_is_matrix(const mag_tensor_t* t) {
    #pragma GCC unroll 4
    for (uint32_t i=2; i < MAG_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool mag_tensor_is_volume(const mag_tensor_t* t) {
    #pragma GCC unroll 3
    for (uint32_t i=3; i < MAG_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool mag_tensor_is_shape_eq(const mag_tensor_t* a, const mag_tensor_t* b) {
    return memcmp(a->shape, b->shape, sizeof(a->shape)) == 0;
}

bool mag_tensor_are_strides_eq(const mag_tensor_t* a, const mag_tensor_t* b) {
    return memcmp(a->strides, b->strides, sizeof(a->strides)) == 0;
}

bool mag_tensor_can_broadcast(const mag_tensor_t* a, const mag_tensor_t* b) {
    #pragma GCC unroll 6
    for (uint32_t i=0; i < MAG_MAX_DIMS; ++i)
        if ((b->shape[i] % a->shape[i]) != 0)
            return false;
    return true;
}

bool mag_tensor_is_transposed(const mag_tensor_t* t) { return t->strides[0] > t->strides[1]; }

bool mag_tensor_is_permuted(const mag_tensor_t* t) {
    #pragma GCC unroll 5
    for (uint32_t i=0; i < MAG_MAX_DIMS-1; ++i)
        if (t->strides[i] > t->strides[i+1])
            return true;
    return false;
}

bool mag_tensor_is_contiguous(const mag_tensor_t* t) {
    return *t->strides == 1;
}

float mag_tensor_get_scalar_physical_index(mag_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            float r;
            mag_storage_buffer_t* sto = &t->storage;
            (*sto->cpy_device_host)(sto, sizeof(r)*(d0*s0 + d1*s1 + d2*s2 + d3*s3 + d4*s4 + d5*s5), &r, sizeof(r));
            return r;
        }
        default: mag_panic("Unsupported data type: %s", mag_dtype_meta_of(t->dtype)->name);
    }
}

void mag_tensor_set_scalar_physical_index(mag_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, float x) {
    mag_static_assert(MAG_MAX_DIMS == 6);
    mag_load_local_storage_group(t, s, strides);
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            mag_storage_buffer_t* sto = &t->storage;
            (*sto->cpy_host_device)(sto, sizeof(x)*(d0*s0 + d1*s1 + d2*s2 + d3*s3 + d4*s4 + d5*s5), &x, sizeof(x));
        } break;
        default: mag_panic("Unsupported data type: %s", mag_dtype_meta_of(t->dtype)->name);
    }
}

float mag_tensor_get_scalar_virtual_index(mag_tensor_t* t, int64_t v_idx) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_virtual_to_physical_index(t, v_idx, &pidx);
        return mag_tensor_get_scalar_physical_index(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5]);
    }
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            float r;
            mag_storage_buffer_t* sto = &t->storage;
            (*sto->cpy_device_host)(sto, sizeof(r)*v_idx, &r, sizeof(r));
            return r;
        }
        default:
            mag_panic("Unsupported data type: %s", mag_dtype_meta_of(t->dtype)->name);
    }
}

void mag_tensor_set_scalar_virtual_index(mag_tensor_t* t, int64_t v_idx, float x) {
    if (!mag_tensor_is_contiguous(t)) {
        int64_t pidx[MAG_MAX_DIMS];
        mag_tensor_virtual_to_physical_index(t, v_idx, &pidx);
        mag_tensor_set_scalar_physical_index(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5], x);
        return;
    }
    switch (t->dtype) {
        case MAG_DTYPE_F32: {
            mag_storage_buffer_t* sto = &t->storage;
            (*sto->cpy_host_device)(sto, sizeof(x)*v_idx, &x, sizeof(x));
        } break;
        default:
            mag_panic("Unsupported data type: %s", mag_dtype_meta_of(t->dtype)->name);
    }
}

bool mag_tensor_eq(const mag_tensor_t* a, const mag_tensor_t* b) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->numel != b->numel) return false;
    /*int64_t n = mag_tensor_num_elements(a); TODO
    switch (a->dtype) {
        case MAG_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i) {
                if (buf_a[i] != buf_b[i]) {
                    return false;
                }
            }
        } break;
        default: mag_panic("Unsupported data type: %s", mag_dtype_info_of(a->dtype)->name);
    }*/
    return false;
}

bool mag_tensor_is_close(const mag_tensor_t* a, const mag_tensor_t* b, float eps, double* percent_eq) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->numel != b->numel) return false;
    /*eps = eps < 0.0f ? FLT_EPSILON : eps; TODO
    int64_t n = mag_tensor_num_elements(a);
    int64_t n_eq = 0;
    switch (a->dtype) {
        case MAG_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i)   |x - y| <= ε     ∀ x, y ∈ A, B
                if (fabsf(buf_a[i] - buf_b[i]) <= eps) ++n_eq;
        } break;
        default: mag_panic("Unsupported data type: %s", mag_dtype_info_of(a->dtype)->name);
    }
    if (percent_eq) *percent_eq = (double)n_eq / (double)n * 100.0;
    return n_eq == n;*/
    return false;
}

void mag_tensor_img_draw_box(mag_tensor_t* t, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t wi, uint32_t rgb) {
    mag_assert(t->rank == 3, "Tensor must be 3D image tensor");
    mag_assert2(x2 > x1 && y2 > y1 && x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0);
    float* buf = mag_tensor_data_ptr(t);
    int32_t w = (int32_t)mag_tensor_image_width(t);
    int32_t h = (int32_t)mag_tensor_image_height(t);
    int32_t c = (int32_t)mag_tensor_image_channels(t);
    mag_assert2(w && h && c == 3);
    float r = (float)((rgb>>16)&0xff) / 255.0f;
    float g = (float)((rgb>>8)&0xff) / 255.0f;
    float b = (float)(rgb&0xff) / 255.0f;
    wi = mag_xmax(1, wi);
    for (int32_t i=0; i < wi; ++i) {
        int32_t xx1 = x1+i;
        int32_t yy1 = y1+i;
        int32_t xx2 = x2-i;
        int32_t yy2 = y2-i;
        if (mag_unlikely(xx1 >= w)) xx1 = w-1;
        if (mag_unlikely(xx2 >= w)) xx2 = w-1;
        if (mag_unlikely(yy1 >= h)) yy1 = h-1;
        if (mag_unlikely(yy2 >= h)) yy2 = h-1;
        for (int32_t j=xx1; j <= xx2; ++j) {
            float* r1 = buf + j + yy1*w + 0*w*h;
            float* r2 = buf + j + yy2*w + 0*w*h;
            float* g1 = buf + j + yy1*w + 1*w*h;
            float* g2 = buf + j + yy2*w + 1*w*h;
            float* b1 = buf + j + yy1*w + 2*w*h;
            float* b2 = buf + j + yy2*w + 2*w*h;
            mag_bnd_chk(r1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(r2, buf, mag_tensor_data_size(t));
            mag_bnd_chk(g1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(g2, buf, mag_tensor_data_size(t));
            mag_bnd_chk(b1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(b2, buf, mag_tensor_data_size(t));
            *r1 = *r2 = r;
            *g1 = *g2 = g;
            *b1 = *b2 = b;
        }
        for (int32_t j = yy1; j <= yy2; ++j) {
            float* r1 = buf + xx1 + j*w + 0*w*h;
            float* r2 = buf + xx2 + j*w + 0*w*h;
            float* g1 = buf + xx1 + j*w + 1*w*h;
            float* g2 = buf + xx2 + j*w + 1*w*h;
            float* b1 = buf + xx1 + j*w + 2*w*h;
            float* b2 = buf + xx2 + j*w + 2*w*h;
            mag_bnd_chk(r1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(r2, buf, mag_tensor_data_size(t));
            mag_bnd_chk(g1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(g2, buf, mag_tensor_data_size(t));
            mag_bnd_chk(b1, buf, mag_tensor_data_size(t));
            mag_bnd_chk(b2, buf, mag_tensor_data_size(t));
            *r1 = *r2 = r;
            *g1 = *g2 = g;
            *b1 = *b2 = b;
        }
    }
}

static bool mag_glyph(uint32_t c, uint32_t x, uint32_t y) {
    c -= 33, --x;
    if (mag_unlikely(c > 93 || x > 6 || y > 13)) return false;
    uint32_t i = 98*c + 7*y + x;
    return (("0@P01248@00120000P49B0000000000000:DXlW2UoDX@10008@h;IR4n@R<Y?48000PYDF"
             "PP011J:U1000<T8QQQDAR4a50000@P012000000000000222448@P024@010028P0148@PP011100000"
             "ABELDU410000000048@l7124000000000000000H`01100000000n10000000000000000006<0000@P"
             "P011224488@00000`CXHY:=:D8?0000004<DT01248@000000l4:444444h700000`C8@Ph02D8?0000"
             "008HX89b?8@P000000n58`7@P05b300000`CP0O25:D8?00000POPP0112248000000l4:D8?Q25b300"
             "000`CX@Ql1244700000000H`0000<H00000000`P1000H`0110000044444@014@0000000n100PO000"
             "0000004@014@@@@@0000h948@@@@00120000`G`l5;F\\Lf0n100000l4:DXOQ25:400000hCX@Qn4:D"
             "X?000000?Q248@P0Ql000000N49DX@Q25i100000hGP01N48@PO00000PO124hAP012000000l4:@PLQ"
             "25b3000008DX@Qn5:DX@000000748@P0124L00000001248@P25b3000008DT456D8AT@00000P01248"
             "@P01n10000017G=IbP1364000008dXAU:U:E\\H000000?Q25:DX@Ql000000n4:DX?1248000000`CX"
             "@Q2U:E4GP0000P?Q25jCR8Q2100000l4:@0?P05b300000l71248@P01200000P@Q25:DX@Ql0000002"
             "5:D89BT`P1000004<HbT9[:BT800000P@QT8QQ49Q210000013:B4548@P000000h7888888@PO00000"
             "7248@P01248`10P0148P0148P0148000h01248@P0124>000015A000000000000000000000000h?00"
             "04@010000000000000000l0bGX@aL10000124XcX@Q25j300000000?Q248@8?000008@Pl5:DX@aL10"
             "000000`CX@o24`70000`AP01N48@P0100000000l5:DX@aL12T70124XcX@Q25:40000@P0P348@P01>"
             "00000240HP01248@P0a101248@T47B4940000HP01248@P01L00000000oBV<IbT910000000hCX@Q25"
             ":400000000?Q25:D8?00000000j<:DX@Qn48@00000`GX@Q25c58@P0000P>S248@P000000000l48P7"
             "@Pn0000048@`31248@030000000P@Q25:D<G0000000025:T49<H000000004<HbTE5920000000P@QT"
             "`@BX@0000000025:DX@aL12T70000h744444h70000PS01248>P0124`1001248@P01248@P0007@P01"
             "24`@P01R30000000S9S10000000"[i/6]-'0')>>(i%6))&1;
}

void mag_tensor_img_draw_text(mag_tensor_t* t, int32_t x, int32_t y, int32_t size, uint32_t rgb, const char* txt) { /* TODO: Implement font scaling, size is ignored currently */
    mag_assert(t->rank == 3, "Tensor must be a 3D image tensor");
    mag_assert2(x >= 0 && y >= 0 && size >= 8 && txt && *txt);
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    float* buf = (float*)t->storage.base;
    int32_t w = (int32_t)mag_tensor_image_width(t);
    int32_t h = (int32_t)mag_tensor_image_height(t);
    int32_t c = (int32_t)mag_tensor_image_channels(t);
    mag_assert2(w && h && c == 3);
    float* pr = buf;
    float* pg = buf + w*h;
    float* pb = buf + w*h*2;
    float r = (float)((rgb>>16)&0xff) / 255.0f;
    float g = (float)((rgb>>8)&0xff) / 255.0f;
    float b = (float)(rgb&0xff) / 255.0f;
    int32_t ly = y;
    for (int32_t lx = x; *txt; lx = (*txt == '\n' ? x : lx+8), ly = (*txt == '\n' ? ly+14 : ly), txt++) {
        if (mag_unlikely(!isprint(*txt))) continue;
        for (int32_t yy = 0; yy < 14; ++yy) {
            for (int32_t xx = 0; xx < 8; ++xx) {
                if (!mag_glyph(*txt, xx, yy)) continue;
                int32_t px = lx + xx;
                int32_t py = ly + yy;
                if (mag_unlikely(px >= w || py >= h)) continue;
                int32_t ii = py*w + px;
                pr[ii] = r;
                pg[ii] = g;
                pb[ii] = b;
            }
        }
    }
}

mag_ctx_t* mag_tensor_get_ctx(const mag_tensor_t* t) { return t->ctx; }
void* mag_tensor_get_user_data(const mag_tensor_t* t) { return t->ud; }
void mag_tensor_set_user_data(mag_tensor_t* t, void* ud) { t->ud = ud; }

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

static void MAG_COLDPROC mag_system_host_info_query_os_name(char (*out_os_name)[128]) { /* Get OS name */
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

static void MAG_COLDPROC mag_system_host_info_query_cpu_name(char (*out_cpu_name)[128]) { /* Get CPU name */
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

static void MAG_COLDPROC mag_system_host_info_query_cpu_cores(uint32_t* out_virtual, uint32_t* out_physical, uint32_t* out_sockets) { /* Get CPU virtual (logical) cores. */
    #ifdef _WIN32
        DWORD size = 0;
        GetLogicalProcessorInformation(NULL, &size);
        if (mag_unlikely(!size)) return;
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION* info = (*mag_alloc)(NULL, size);
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
        end: (*mag_alloc)(info, 0);
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
                    for (int32_t i = 0; i < cpu_count; ++i) if (physical_ids[i] == current_physical_id && core_ids[i] == current_core_id) { is_unique = false; break; }
                    if (is_unique) {
                        if (cpu_count < MAG_MAX_CPUS) {
                            physical_ids[cpu_count] = current_physical_id;
                            core_ids[cpu_count] = current_core_id;
                            ++cpu_count;
                        } else break;
                    }
                    is_unique = true;
                    for (int32_t i = 0; i < package_count; ++i) if (package_ids[i] == current_physical_id) { is_unique = false; break; }
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

static void MAG_COLDPROC mag_system_host_info_query_memory(uint64_t* out_phys_mem_total, uint64_t* out_phys_mem_free) { /* Get physical memory */
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
    static void mag_cpuid(uint32_t leaf, int32_t sub, uint32_t* oeax, uint32_t* oebx, uint32_t* oecx, uint32_t* oedx) {
        #ifdef _MSC_VER
            int regs[4];
            if (sub != -1) __cpuidex(regs, leaf, sub);
            else __cpuid(regs, leaf);
            *oeax = regs[0], *oebx = regs[1], *oecx = regs[2], *oedx = regs[3];
        #else
            uint32_t eax, ebx, ecx, edx;
            if (sub != -1) __cpuid_count(leaf, sub, eax, ebx, ecx, edx);
            else __cpuid(leaf, eax, ebx, ecx, edx);
            *oeax = eax, *oebx = ebx, *oecx = ecx, *oedx = edx;
        #endif
    }
    static uint64_t MAG_AINLINE mag_xgetbv(void) { /* Query extended control register value. */
        #ifdef _MSC_VER
            return _xgetbv(0);
        #else
            uint32_t lo, hi;
            __asm__ __volatile__("xgetbv\n\t" : "=a" (lo), "=d" (hi) : "c" (0));
            return (uint64_t)lo | ((uint64_t)hi << 32);
        #endif
    }
    #define mag_cpy_regs(id) \
        (*features)[MAG_X86_64_CPUID_##id][MAG_X86_64_CPUID_EAX] = eax; \
        (*features)[MAG_X86_64_CPUID_##id][MAG_X86_64_CPUID_EBX] = ebx; \
        (*features)[MAG_X86_64_CPUID_##id][MAG_X86_64_CPUID_ECX] = ecx; \
        (*features)[MAG_X86_64_CPUID_##id][MAG_X86_64_CPUID_EDX] = edx
    static void MAG_COLDPROC mag_system_info_query_x86_64_cpu_features(uint32_t (*features)[8][4]) {
        uint32_t eax=0, ebx=0, ecx=0, edx=0;
        uint32_t max_basic_leaf, max_extended_leaf;
        mag_cpuid(0, -1, &eax, &ebx, &ecx, &edx);
        mag_cpy_regs(0H);
        max_basic_leaf = eax;
        mag_cpuid(0x80000000u, -1, &eax, &ebx, &ecx, &edx);
        max_extended_leaf = eax;
        if (max_basic_leaf >= 1u) {
            mag_cpuid(1, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(1H);
        }
        if (max_basic_leaf >= 2u) {
            mag_cpuid(2u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(2H);
        }
        if (max_basic_leaf >= 7u) {
            mag_cpuid(7u, 0, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(7H);
        }
        if (max_basic_leaf >= 7u) {
            mag_cpuid(7u, 1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(7H_1H);
        }
        if (max_basic_leaf >= 0x16u) {
            mag_cpuid(0x16u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(16H);
        }
        if (max_extended_leaf >= 0x80000001u) {
            mag_cpuid(0x80000001u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(80000001H);
        }
        if (max_extended_leaf >= 0x80000007u) {
            mag_cpuid(0x80000007u, -1, &eax, &ebx, &ecx, &edx);
            mag_cpy_regs(80000007H);
        }
        bool cpu_avx_support = ((*features)[MAG_X86_64_CPUID_1H][MAG_X86_64_CPUID_ECX] & 0x10000000u) != 0;
        bool cpu_osxsave_support = ((*features)[MAG_X86_64_CPUID_1H][MAG_X86_64_CPUID_ECX] & 0x8000000u) != 0;
        if (cpu_avx_support && cpu_osxsave_support) {
            uint64_t xcr0 = mag_xgetbv();
            if ((xcr0 & 0x6) != 0x6u) {
                (*features)[MAG_X86_64_CPUID_1H][MAG_X86_64_CPUID_ECX] &= ~0x10000000u; /* Clear AVX */
                (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EBX] &= ~0x20u; /* Clear AVX2 */
            }
            if ((xcr0 & 0xe0) != 0xe0u) { /* OS does not support AVX-512, clear AVX512 */
                (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EBX] &= ~0xdc230000u;
                (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_ECX] &= ~0x5842u;
                (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EDX] &= ~0x10cu;
                (*features)[MAG_X86_64_CPUID_7H_1H][MAG_X86_64_CPUID_EAX] &= ~0x20u;
            }
        } else {
            (*features)[MAG_X86_64_CPUID_1H][MAG_X86_64_CPUID_ECX] &= ~0x10000000u; /* Clear AVX */
            (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EBX] &= ~0x20u; /* Clear AVX2 */
            (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EBX] &= ~0xdc230000u; /* Clear AVX512 */
            (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_ECX] &= ~0x5842u; /* Clear AVX512 */
            (*features)[MAG_X86_64_CPUID_7H][MAG_X86_64_CPUID_EDX] &= ~0x10cu; /* Clear AVX512 */
            (*features)[MAG_X86_64_CPUID_7H_1H][MAG_X86_64_CPUID_EAX] &= ~0x20u; /* Clear AVX512 */
        }
    }
    #undef mag_cpy_regs

#elif defined(__aarch64__) && defined(__linux__)
static void MAG_COLDPROC mag_system_info_query_arm64_cpu_features(long* hwcap) {
    *hwcap = getauxval(AT_HWCAP);
}
#endif

static void MAG_COLDPROC mag_system_host_info_query(mag_ctx_t* ctx) {
    mag_system_host_info_query_os_name(&ctx->sys.os_name);
    mag_system_host_info_query_cpu_name(&ctx->sys.cpu_name);
    mag_system_host_info_query_cpu_cores(&ctx->sys.cpu_virtual_cores, &ctx->sys.cpu_physical_cores, &ctx->sys.cpu_sockets);
    mag_system_host_info_query_memory(&ctx->sys.phys_mem_total, &ctx->sys.phys_mem_free);
    #if defined(__x86_64__) || defined(_M_X64)
        mag_system_info_query_x86_64_cpu_features(&ctx->sys.x86_64_cpu_features);
    #elif defined(__aarch64__) && defined(__linux__)
        mag_system_info_query_arm64_cpu_features(&ctx->sys.cpu_arm64_hwcap);
    #endif
    if (mag_unlikely(!*ctx->sys.os_name)) snprintf(ctx->sys.os_name, sizeof(ctx->sys.os_name), "Unknown");
    if (mag_unlikely(!*ctx->sys.cpu_name)) snprintf(ctx->sys.cpu_name, sizeof(ctx->sys.cpu_name), "Unknown");
}

static MAG_AINLINE void mag_sto_write_u32_le(uint8_t** p, uint32_t x) {
    x = mag_bswap32(x);
    memcpy(*p, &x, sizeof(x));
    *p += sizeof(x);
}

static MAG_AINLINE void mag_sto_write_u64_le(uint8_t** p, uint64_t x) {
    x = mag_bswap64(x);
    memcpy(*p, &x, sizeof(x));
    *p += sizeof(x);
}

static MAG_AINLINE uint32_t mag_sto_read_u32_le(const uint8_t** p) {
    uint32_t x;
    memcpy(&x, *p, sizeof(x));
    x = mag_bswap32(x);
    *p += sizeof(x);
    return x;
}

static MAG_AINLINE uint64_t mag_sto_read_u64_le(const uint8_t** p) {
    uint64_t x;
    memcpy(&x, *p, sizeof(x));
    x = mag_bswap64(x);
    *p += sizeof(x);
    return x;
}

#define MAG_STO_MAGIC "magtron!"
mag_static_assert(sizeof(MAG_STO_MAGIC)-1 == sizeof(uint64_t));
#define MAG_STO_FILE_HEADER_SIZE ((sizeof(MAG_STO_MAGIC)-1) + sizeof(uint32_t)*3)
#define MAG_STO_TENSOR_HEADER_SIZE (MAG_MAX_TENSOR_NAME_LEN + sizeof(uint32_t) + sizeof(int64_t)*MAG_MAX_DIMS)
mag_static_assert(MAG_MAX_TENSOR_NAME_LEN % 8 == 0);
mag_static_assert(MAG_DTYPE__NUM <= 0xff);
mag_static_assert(MAG_MAX_DIMS <= 0xff);
#define mag_sto_sanitize(exp, ret) do { if (mag_unlikely(!(exp))) { mag_log_error("magnetron storage sanitize error: " #exp); return (ret); } } while (0)

static bool mag_sto_write_file_header( /* Write file header -  file header must be same in every version. */
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    uint32_t num_tensors,
    uint32_t ud
) {
    mag_sto_sanitize(*p + MAG_STO_FILE_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    uint64_t mag_magic;
    memcpy(&mag_magic, MAG_STO_MAGIC, sizeof(mag_magic));
    mag_sto_write_u64_le(p, mag_magic);
    mag_sto_write_u32_le(p, version);
    mag_sto_write_u32_le(p, num_tensors);
    mag_sto_write_u32_le(p, ud);
    return mag_likely(*p - start == MAG_STO_FILE_HEADER_SIZE);
}

static bool mag_sto_read_file_header(  /* Read file header - file header must be same in every version. */
    const uint8_t** p,
    const uint8_t* end,
    uint32_t* version,
    uint32_t* num_tensors,
    uint32_t* ud
) {
    mag_sto_sanitize(*p + MAG_STO_FILE_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    uint64_t mag_magic = mag_sto_read_u64_le(p);
    mag_sto_sanitize(memcmp(&mag_magic, MAG_STO_MAGIC, sizeof(mag_magic)) == 0, false);
    *version = mag_sto_read_u32_le(p);
    *num_tensors = mag_sto_read_u32_le(p);
    *ud = mag_sto_read_u32_le(p);
    return mag_likely(*p - start == MAG_STO_FILE_HEADER_SIZE);
}

static bool mag_sto_write_tensor_header(
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    const char (*name)[MAG_MAX_TENSOR_NAME_LEN],
    mag_tensor_flags_t flags,
    mag_dtype_t dtype,
    int64_t rank,
    const int64_t (*shape)[MAG_MAX_DIMS]
) {
    mag_sto_sanitize(*p + MAG_STO_TENSOR_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            uint64_t name_u64[sizeof(*name)/sizeof(uint64_t)];
            memcpy(name_u64, *name, sizeof(*name));
            for (size_t i=0; i < sizeof(name_u64)/sizeof(*name_u64); ++i)   /* Write name as multiple u64 */
                mag_sto_write_u64_le(p, name_u64[i]);
            uint32_t aux = 0;   /* Pack small fields into aux field */
            aux |= (flags & 0xff) << 16;
            aux |= (dtype & 0xff) << 8;
            aux |= (rank & 0xff);
            mag_sto_write_u32_le(p, aux);     /* Write aux field */
            for (size_t i=0; i < MAG_MAX_DIMS; ++i) {      /* Write shape */
                mag_sto_sanitize((*shape)[i] >= 1 && (*shape)[i] < INT64_MAX, false);
                mag_sto_write_u64_le(p, (uint64_t)(*shape)[i]);
            }
        } break;
        default: return false;
    }
    return mag_likely(*p - start == MAG_STO_TENSOR_HEADER_SIZE);
}

static bool mag_sto_read_tensor_header(
    const uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    char (*name)[MAG_MAX_TENSOR_NAME_LEN],
    mag_tensor_flags_t* flags,
    mag_dtype_t* dtype,
    int64_t* rank,
    int64_t (*shape)[MAG_MAX_DIMS]
) {
    mag_sto_sanitize(*p + MAG_STO_TENSOR_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            uint64_t name_u64[sizeof(*name)/sizeof(uint64_t)];
            for (size_t i=0; i < sizeof(name_u64)/sizeof(*name_u64); ++i) /* Read name as multiple u64 */
                name_u64[i] = mag_sto_read_u64_le(p);
            memcpy(name, name_u64, sizeof(*name));
            (*name)[sizeof(*name)-1] = '\0';
            uint32_t aux = mag_sto_read_u32_le(p); /* Read aux field */
            *flags = (mag_tensor_flags_t)((aux >> 16) & 0xff);
            *dtype = (mag_dtype_t)((aux >> 8) & 0xff);
            *rank = (int64_t)(aux & 0xff);
            mag_sto_sanitize((*flags & ~((1u<<MAG_TFLAG_LEN)-1)) == 0, false); /* Check fields */
            mag_sto_sanitize(*dtype >= 0 && *dtype < MAG_DTYPE__NUM, false);
            mag_sto_sanitize(*rank >= 1 && *rank <= MAG_MAX_DIMS, false);
            for (size_t i=0; i < MAG_MAX_DIMS; ++i) {  /* Read shape */
                uint64_t u64 = mag_sto_read_u64_le(p);
                mag_sto_sanitize(u64>= 1 && u64 <= (uint64_t)INT64_MAX, false);
                (*shape)[i] = (int64_t)u64;
            }
            mag_sto_sanitize(INT64_MAX/(*shape)[1] > (*shape)[0], false); /* Check for shape overflow */
            mag_sto_sanitize(INT64_MAX/(*shape)[2] > (*shape)[0]*(*shape)[1], false);
            mag_sto_sanitize(INT64_MAX/(*shape)[3] > (*shape)[0]*(*shape)[1]*(*shape)[2], false);
            mag_sto_sanitize(INT64_MAX/(*shape)[4] > (*shape)[0]*(*shape)[1]*(*shape)[2]*(*shape)[3], false);
            mag_sto_sanitize(INT64_MAX/(*shape)[5] > (*shape)[0]*(*shape)[1]*(*shape)[2]*(*shape)[3]*(*shape)[4], false);
        } break;
        default: return false;
    }
    return mag_likely(*p - start == MAG_STO_TENSOR_HEADER_SIZE);
}

static bool mag_sto_write_tensor_data(
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    mag_dtype_t dtype,
    const void* data,
    int64_t size
) {
    mag_sto_sanitize(size > 0, false);
    mag_sto_sanitize(*p + size <= end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            memcpy(*p, data, size);
            *p += size;
        } break;
        default: return false;
    }
    return mag_likely(*p - start == size);
}

static bool mag_sto_read_tensor_data(
    const uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    mag_dtype_t dtype,
    void* data,
    int64_t size
) {
    mag_sto_sanitize(size > 0, false);
    mag_sto_sanitize(*p + size <= end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            memcpy(data, *p, size); /* TODO: endianess conversion */
            *p += size;
        } break;
        default: return false;
    }
    return mag_likely(*p - start == size);
}

static size_t mag_accumulate_data_size(mag_dtype_t dtype, const int64_t (*shape)[MAG_MAX_DIMS]) {
    size_t size = mag_dtype_meta_of(dtype)->size;
    #pragma GCC unroll 6
    for (size_t i=0; i < MAG_MAX_DIMS; ++i) size *= (size_t)mag_xmax(1, (*shape)[i]);
    return size;
}

static size_t mag_sto_total_size(const mag_tensor_t** tensors, size_t n) {
    size_t total = MAG_STO_FILE_HEADER_SIZE;
    for (size_t i=0; i < n; ++i) {
        total += MAG_STO_TENSOR_HEADER_SIZE;
        total += mag_accumulate_data_size(tensors[i]->dtype, &tensors[i]->shape);
    }
    mag_assert((total & 3) == 0, "Unaligned storage size: %zu", total);
    return total;
}

static uint8_t* mag_sto_write_buffered(const mag_tensor_t** tensors, size_t n_tensors, size_t* out_size, uint32_t version) {
    if (mag_unlikely(!tensors || !n_tensors || n_tensors > UINT32_MAX || !out_size || !version || version > MAG_STORAGE_VERSION)) return NULL;  /* Check input */
    *out_size = mag_sto_total_size(tensors, n_tensors);
    uint8_t* base = (uint8_t*)(*mag_alloc)(NULL, *out_size );     /* Allocate buffer */
    uint8_t* needle = base;
    const uint8_t* end = base + *out_size ;
    if (mag_unlikely(!mag_sto_write_file_header(&needle, end, version, (uint32_t)n_tensors, 0))) goto error;     /* Write file header */
    for (size_t i=0; i < n_tensors; ++i) {   /* Write tensor headers */
        const mag_tensor_t* t = tensors[i];
        mag_assert2(t != NULL);
        if (mag_unlikely(!mag_sto_write_tensor_header(
            &needle,
            end,
            version,
            &t->name,
            t->flags,
            t->dtype,
            t->rank,
            &t->shape
        ))) goto error;
    }
    mag_assert2(needle - base == MAG_STO_FILE_HEADER_SIZE + n_tensors*MAG_STO_TENSOR_HEADER_SIZE);    /* Check written data size */
    for (size_t i=0; i < n_tensors; ++i) {  /* Write tensor data */
        const mag_tensor_t* t = tensors[i];
        mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
        if (mag_unlikely(!mag_sto_write_tensor_data(&needle, end, version, t->dtype, (const void*)t->storage.base, mag_tensor_data_size(t)))) goto error;     /* Write data */
    }
    return base;
    error: /* Error handling */
        (*mag_alloc)(base, 0);
        return NULL;
}

MAG_EXPORT mag_tensor_t** mag_sto_read_buffered(mag_ctx_t* ctx, const uint8_t* buf, size_t size, uint32_t* out_n_tensors, uint32_t* out_version) { /* Load stored tensors from buffer. Function is exported for fuzzing test. */
    if (mag_unlikely(!ctx || !buf || !out_n_tensors || !out_version || size <= MAG_STO_FILE_HEADER_SIZE + MAG_STO_TENSOR_HEADER_SIZE + 1)) return NULL;    /* Check input */
    const uint8_t* needle = buf;
    const uint8_t* end = buf + size;
    uint32_t n_tensors;
    uint32_t ud;
    if (mag_unlikely(!mag_sto_read_file_header(&needle, end, out_version, &n_tensors, &ud))) return NULL;   /* Read file header */
    if (mag_unlikely(!*out_version || *out_version > MAG_VERSION)) return NULL;
    if (mag_unlikely(!n_tensors)) return NULL;
    mag_tensor_t** tensors = (mag_tensor_t**)(*mag_alloc)(NULL, n_tensors*sizeof(*tensors));   /* Allocate return tensor array */
    for (size_t i=0; i < n_tensors; ++i) {  /* Read tensor headers */
        char name[MAG_MAX_TENSOR_NAME_LEN] = {0};
        mag_tensor_flags_t flags = 0;
        mag_dtype_t dtype = 0;
        int64_t rank = 0;
        int64_t shape[MAG_MAX_DIMS] = {0};
        if (mag_unlikely(!mag_sto_read_tensor_header(&needle, end, *out_version, &name, &flags, &dtype, &rank, &shape))) goto error;   /* Read tensor header */
        mag_tensor_t* t = mag_tensor_create(ctx, dtype, shape, rank, NULL, 0);   /* Create placeholder tensor */
        mag_tensor_fmt_name(t, "%s", name);
        t->flags = flags;
        tensors[i] = t;
    }
    for (size_t i=0; i < n_tensors; ++i) {  /* Read tensor data */
        mag_tensor_t* t = tensors[i];
        mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
        size_t data_size = mag_accumulate_data_size(t->dtype, &t->shape);
        mag_assert2(needle + data_size <= end && data_size == mag_tensor_data_size(t));
        if (mag_unlikely(!mag_sto_read_tensor_data(&needle, end, *out_version, t->dtype, (void*)t->storage.base, data_size))) goto error;  /* Read data into tensor's buffer */
    }
    *out_n_tensors = n_tensors;
    return tensors;
    error:
        (*mag_alloc)(tensors, 0);
        return NULL;
}

static bool mag_sto_has_mag_ext(const char* file) { /* Check if file path has magnetron extension. */
    if (mag_unlikely(!file || strlen(file) < sizeof(MAG_STORAGE_EXT))) return false;
    const char* dot = strrchr(file, '.');
    return dot && !strcmp(dot, MAG_STORAGE_EXT);
}

void mag_tensor_save(const mag_tensor_t* t, const char* file) {
    mag_assert(mag_sto_has_mag_ext(file), "Invalid file extension: %s", file);
    FILE* f = mag_fopen(file, "wb");  /* Open file */
    mag_assert(f, "Failed to open file stream: %s", file);
    uint32_t version = MAG_STORAGE_VERSION;
    size_t n_tensors = 1;
    size_t n_bytes = 0;
    uint8_t* ser = mag_sto_write_buffered(&t, n_tensors, &n_bytes, version);   /* Serialize tensor */
    mag_assert(ser && n_bytes, "Failed to serialize tensor to file: %s", file);   /* Check serialization */
    mag_assert(fwrite(ser, 1, n_bytes, f) == n_bytes, "Failed to write %zu bytes to file: %s", n_bytes, file);    /* Write to file */
    (*mag_alloc)(ser, 0);     /* Free buffer */
    fflush(f);
    fclose(f);
    double mem;
    const char* unit;
    mag_humanize_memory_size(n_bytes, &mem, &unit);
    mag_log_info("Saved %zu tensor%s to file: %s, %.03f %s written, storage v.%u", n_tensors, n_tensors > 1 ? "s" : "", file, mem, unit, version);
}

mag_tensor_t* mag_tensor_load(mag_ctx_t* ctx, const char* file) {
    mag_assert(mag_sto_has_mag_ext(file), "Invalid file extension: %s", file);
    FILE* f = mag_fopen(file, "rb");  /* Open file */
    mag_assert(f, "Failed to open file stream: %s", file);
    mag_assert2(fseek(f, 0, SEEK_END) == 0);  /* Seek to end */
    long n_bytes = ftell(f);    /* Get file size */
    mag_assert(n_bytes > MAG_STO_FILE_HEADER_SIZE + MAG_STO_TENSOR_HEADER_SIZE + 1, "Malformed file size");   /* Check file size */
    mag_assert2(fseek(f, 0, SEEK_SET) == 0); /* Seek to start */
    uint8_t* buf = (uint8_t*)(*mag_alloc)(NULL, n_bytes);  /* Allocate buffer */
    mag_assert(fread(buf, 1, n_bytes, f) == n_bytes, "Failed to read %zu bytes from file: %s", n_bytes, file);    /* Read while file into buffer */
    fclose(f), f = NULL;    /* Close file */
    uint32_t n_tensors = 0, version = 0;
    mag_tensor_t** tensors = mag_sto_read_buffered(ctx, buf, n_bytes, &n_tensors, &version);   /* Deserialize tensors */
    mag_assert(version > 0 && version <= MAG_VERSION, "Unsupported storage version: %u", version);   /* Check version */
    mag_assert(tensors && n_tensors > 0, "Failed to load tensor from file: %s", file);
    mag_tensor_t* target = *tensors;
    (*mag_alloc)(buf, 0);     /* Free buffer */
    (*mag_alloc)(tensors, 0);     /* Free tensor array */
    double mem;
    const char* unit;
    mag_humanize_memory_size(n_bytes, &mem, &unit);
    mag_log_info("Loaded %u tensor%s from file: %s, %.03f %s read, storage v.%u", n_tensors, n_tensors > 1 ? "s" : "", file, mem, unit, version);
    return target;
}

mag_tensor_t* mag_tensor_load_image(mag_ctx_t* ctx, const char* file, mag_color_channels_t channels, uint32_t resize_w, uint32_t resize_h) {
    uint8_t* (*loader)(const char*, uint32_t(*)[3], mag_color_channels_t) = ctx->image_load_fn;
    void (*load_free)(uint8_t*) = ctx->image_load_free_fn;
    mag_assert(loader && load_free, "Image loader not set");
    uint32_t whc[3] = {0};
    uint8_t* src = (*loader)(file, &whc, channels);
    mag_assert(src, "Failed to load tensor from image: '%s'", file);
    if (resize_w && resize_h) { /* Resize requested. */
        float* ori = (*mag_alloc)(NULL, whc[2]*whc[1]*whc[0]*sizeof(*ori));
        for (int64_t k=0; k < whc[2]; ++k) { /* Convert from interleaved to planar representation. */
            for (int64_t j=0; j < whc[1]; ++j) {
                for (int64_t i=0; i < whc[0]; ++i) {
                    ori[i + whc[0]*j + whc[0]*whc[1]*k] = (float)src[k + whc[2]*i + whc[2]*whc[0]*j] / 255.0f;  /* Normalize pixel values to [0, 1] */
                }
            }
        }
        mag_tensor_t* t = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, whc[2], resize_h, resize_w);
        float* dst = mag_tensor_data_ptr(t);
        float* part = (*mag_alloc)(NULL, whc[2] * whc[1] * resize_w * sizeof(*part));
        float ws = (float)(whc[0] - 1)/(float)(resize_w - 1);
        float hs = (float)(whc[1] - 1)/(float)(resize_h - 1);
        for (uint32_t k = 0; k < whc[2]; ++k){
            for (uint32_t r = 0; r < whc[1]; ++r) {
                for (uint32_t c = 0; c < resize_w; ++c) {
                    float val = 0;
                    if (c == resize_w - 1 || whc[0] == 1) {
                        val = ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + (whc[0] - 1)];
                    } else {
                        float sx = (float)c*ws;
                        uint32_t ix = (uint32_t)sx;
                        float dx = sx - (float)ix;
                        val = (1-dx) * (ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + ix]) + dx*(ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + (ix + 1)]);
                    }
                    part[k * resize_w * (whc[1]) + r * resize_w + c] = val;
                }
            }
        }
        for (uint32_t k = 0; k < whc[2]; ++k) {
            for (uint32_t r = 0; r < resize_h; ++r) {
                float sy = (float)r*hs;
                uint32_t iy = (uint32_t)sy;
                float dy = sy - (float)iy;
                for (uint32_t c = 0; c < resize_w; ++c) {
                    float val = (1-dy)*(part[k * resize_w * whc[1] + iy * resize_w + c]);
                    dst[k * resize_w * resize_h + r * resize_w + c] = val;
                }
                if (r == resize_h - 1 || whc[1] == 1) continue;
                for (uint32_t c = 0; c < resize_w; ++c) {
                    float val = dy*(part[k * resize_w * (whc[1]) + (iy + 1) * resize_w + c]);
                    dst[k * resize_w * resize_h + r * resize_w + c] += val;
                }
            }
        }
        (*mag_alloc)(ori, 0);
        (*mag_alloc)(part, 0);
        mag_assert(resize_w * resize_h * whc[2] == mag_tensor_numel(t), "Buffer size mismatch: %zu != %zu", resize_w * resize_h * whc[2], (size_t)mag_tensor_numel(t));
        (*load_free)(src);
        mag_log_info("Loaded and resized tensor from image: %s, %u x %u x %u", file, resize_w, resize_h, whc[2]);
        return t;
    } else {
        mag_tensor_t* t = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, whc[2], whc[1], whc[0]);
        float* dst = mag_tensor_data_ptr(t);
        for (int64_t k = 0; k < whc[2]; ++k) { /* Convert from interleaved to planar representation. */
            for (int64_t j = 0; j < whc[1]; ++j) {
                for (int64_t i = 0; i < whc[0]; ++i) {
                    dst[i + whc[0]*j + whc[0]*whc[1]*k] = (float)src[k + whc[2]*i + whc[2]*whc[0]*j] / 255.0f;  /* Normalize pixel values to [0, 1] */
                }
            }
        }
        mag_assert(whc[0]*whc[1]*whc[2] == mag_tensor_numel(t), "Buffer size mismatch: %zu != %zu", whc[0]*whc[1]*whc[2], (size_t)mag_tensor_numel(t));
        (*load_free)(src);
        mag_log_info("Loaded tensor from image: %s, %u x %u x %u", file, whc[0], whc[1], whc[2]);
        return t;
    }
}

void mag_tensor_save_image(const mag_tensor_t* t, const char* file) {
    bool (*saver)(const char*, const uint8_t*, const uint32_t(*)[3]) = t->ctx->image_save_fn;
    mag_assert(saver, "Image saver not set");
    int64_t rank = mag_tensor_rank(t);
    mag_assert(rank == 3, "Tensor rank must be 3, but is: %" PRIi64, (size_t)rank);
    int64_t w = mag_tensor_image_width(t);
    int64_t h = mag_tensor_image_height(t);
    int64_t c = mag_tensor_image_channels(t);
    mag_assert(c == 1 || c == 3 || c == 4, "Invalid number of channels: %zu", (size_t)c);
    mag_assert(w*h*c == mag_tensor_numel(t), "Buffer size mismatch: %zu != %zu", w*h*c, (size_t)mag_tensor_numel(t));
    uint8_t* dst = (*mag_alloc)(NULL, w*h*c); /* Allocate memory for image data */
    const float* src = mag_tensor_data_ptr(t);
    for (int64_t k = 0; k < c; ++k) /* Convert from planar to interleaved format. */
        for (int64_t i = 0; i < w*h; ++i)
            dst[i*c + k] = (uint8_t)(src[i + k*w*h]*255.0f);
    const uint32_t whc[3] = {(uint32_t)w,(uint32_t)h,(uint32_t)c};
    mag_assert((*saver)(file, dst, &whc), "Failed to save tensor to image: %s", file);
    (*mag_alloc)(dst, 0); /* Free image data */
    mag_log_info("Saved tensor to image: %s, width: %d, height: %d, channels: %d", file, (int)w, (int)h, (int)c);
}

static uint8_t* mag_default_image_load_impl(const char* file, uint32_t(*whc)[3], mag_color_channels_t channels) {
    mag_assert2(file && *file && whc);
    int w, h, c, dc;
    switch (channels) {
        default: dc = STBI_default; break;
        case MAG_COLOR_CHANNELS_GRAY: dc = STBI_grey; break;
        case MAG_COLOR_CHANNELS_GRAY_A: dc = STBI_grey_alpha; break;
        case MAG_COLOR_CHANNELS_RGB: dc = STBI_rgb; break;
        case MAG_COLOR_CHANNELS_RGBA: dc = STBI_rgb_alpha; break;
    }
    uint8_t* buf = stbi_load(file, &w, &h, &c, dc);
    if (mag_unlikely(!buf || !w || !h || !c || (c != 1 && c != 3 && c != 4))) return NULL;
    (*whc)[0] = (uint32_t)w;
    (*whc)[1] = (uint32_t)h;
    (*whc)[2] = (uint32_t)c;
    return buf;
}

static void mag_default_image_load_free_fn_impl(uint8_t* p) {
    stbi_image_free(p);
}

static bool mag_default_image_save_impl(const char* file, const uint8_t* buf, const uint32_t(*whc)[3]) {
    mag_assert2(file && *file && buf && whc);
    return stbi_write_jpg(file, (int)(*whc)[0], (int)(*whc)[1], (int)(*whc)[2], buf, 100) != 0;
}
