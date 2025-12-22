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

#include "mag_def.h"
#include "mag_alloc.h"

#include <ctype.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif
#include <unistd.h>
#include <time.h>
#endif

static mag_log_level_t mag_log_level_var = MAG_LOG_LEVEL_ERROR;
void mag_set_log_level(mag_log_level_t level) { mag_log_level_var = level; }
mag_log_level_t mag_log_level(void) { return mag_log_level_var; }

const char *mag_status_get_name(mag_status_t op){
    static const char *names[] = {
        "MAG_STATUS_OK",
        "MAG_STATUS_ERR_PENDING",
        "MAG_STATUS_ERR_THREAD_MISMATCH",
        "MAG_STATUS_ERR_INVALID_RANK",
        "MAG_STATUS_ERR_INVALID_DIM",
        "MAG_STATUS_ERR_DIM_OVERFLOW",
        "MAG_STATUS_ERR_INVALID_INDEX",
        "MAG_STATUS_ERR_OUT_OF_BOUNDS",
        "MAG_STATUS_ERR_INVALID_PARAM",
        "MAG_STATUS_ERR_STRIDE_SOLVER_FAILED",
        "MAG_STATUS_ERR_BROADCAST_IMPOSSIBLE",
        "MAG_STATUS_ERR_OPERATOR_IMPOSSIBLE",
        "MAG_STATUS_ERR_INVALID_STATE",
        "MAG_STATUS_ERR_IMAGE_ERROR",
        "MAG_STATUS_ERR_UNKNOWN",
    };
    mag_static_assert(sizeof(names)/sizeof(*names)-1 == MAG_STATUS_ERR_UNKNOWN);
    return names[op];
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
    void *trace[0xff];
    backtrace_symbols_fd(trace, backtrace(trace, sizeof(trace)/sizeof(*trace)), STDERR_FILENO);
  }
}
#else
static void mag_dump_backtrace(void) { }
#endif

static void MAG_COLDPROC mag_panic_dump(FILE *f, bool cc, const char *msg, va_list args) {
  if (cc) fprintf(f, "%s", MAG_CC_RED);
  vfprintf(f, msg, args);
  if (cc) fprintf(f, "%s", MAG_CC_RESET);
  fputc('\n', f);
  fflush(f);
}

MAG_NORET MAG_COLDPROC void mag_panic(const char *fmt, ...) { /* Panic and exit the program. If available print backtrace. */
  va_list args;
  va_start(args, fmt);
#if 0
  FILE *f = fopen("magnetron_panic.log", "w");
  if (f) {
    mag_panic_dump(f, false, fmt, args);
    fclose(f), f = NULL;
  }
#endif
  fflush(stdout);
  mag_panic_dump(stderr, true, fmt, args);
  va_end(args);
#ifdef NDEBUG
  mag_dump_backtrace();
#endif
  abort();
}

void mag_log_fmt(mag_log_level_t level, const char *fmt, ...) {
    if (level > mag_log_level_var) return;
    FILE *f = stdout;
    const char *color = NULL;
    switch (level) {
        case MAG_LOG_LEVEL_ERROR: color = MAG_CC_RED; break;
        case MAG_LOG_LEVEL_WARN: color = MAG_CC_YELLOW; break;
        case MAG_LOG_LEVEL_INFO: color = NULL; break;
        case MAG_LOG_LEVEL_DEBUG: color = MAG_CC_MAGENTA; break;
        default:;
    }
    fprintf(f, MAG_CC_CYAN "[magnetron] " MAG_CC_RESET "%s", color ? color : "");
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fprintf(f, "%s\n", color ? MAG_CC_RESET : "");
    if (level == MAG_LOG_LEVEL_ERROR) fflush(f);
}

void MAG_COLDPROC mag_print_separator(FILE *f) {
  f = f ? f : stdout;
  char sep[100+1];
  for (size_t i=0; i < (sizeof(sep)/sizeof(*sep))-1; ++i) sep[i] = '-';
  sep[sizeof(sep)/sizeof(*sep)-1] = '\0';
  fprintf(f, "%s\n", sep);
}

void mag_humanize_memory_size(size_t n, double *out, const char **unit) {
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

#ifdef _WIN32
#include <wchar.h>
extern __declspec(dllimport) int __stdcall MultiByteToWideChar(
    unsigned int cp,
    unsigned long flags,
    const char *str,
    int cbmb,
    wchar_t *widestr,
    int cchwide
);
extern __declspec(dllimport) int __stdcall WideCharToMultiByte(
    unsigned int cp,
    unsigned long flags,
    const wchar_t *widestr,
    int cchwide,
    char *str,
    int cbmb,
    const char *defchar,
    int *used_default
);
#endif

/* Open file. Basically fopen but with UTF-8 support on Windows. */
FILE *mag_fopen(const char *file, const char *mode) {
    mag_assert(file && *file && mode && *mode, "Invalid file name or mode");
    FILE *f = NULL;
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
double mag_hpc_clock_elapsed_ms(uint64_t start) { /* High precision clock elapsed time in milliseconds. */
    return (double)mag_hpc_clock_elapsed_ns(start) / 1e6;
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
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)(tv.tv_sec)*1000000 + tv.tv_usec;
#endif
}

uint64_t mag_hash(const void *key, size_t len, uint32_t seed) {
#define	mag_rol32(x, r) (((x)<<(r))|((x)>>(32-(r))))
#define mag_mix32(h) h^=h>>16; h*=0x85ebca6b; h^=h>>13; h*=0xc2b2ae35; h^=h>>16;
    const uint8_t *p = key;
    int64_t nblocks = (int64_t)len>>4;
    uint32_t h1 = seed;
    uint32_t h2 = seed;
    uint32_t h3 = seed;
    uint32_t h4 = seed;
    uint32_t c1 = 0x239b961b;
    uint32_t c2 = 0xab0e9789;
    uint32_t c3 = 0x38b34ae5;
    uint32_t c4 = 0xa1e38b93;
    const uint32_t *blocks = (const uint32_t *)(p + nblocks*16);
    for (int64_t i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i*4+0];
        uint32_t k2 = blocks[i*4+1];
        uint32_t k3 = blocks[i*4+2];
        uint32_t k4 = blocks[i*4+3];
        k1 *= c1;
        k1  = mag_rol32(k1,15);
        k1 *= c2;
        h1 ^= k1;
        h1 = mag_rol32(h1,19);
        h1 += h2;
        h1 = h1*5+0x561ccd1b;
        k2 *= c2;
        k2  = mag_rol32(k2,16);
        k2 *= c3;
        h2 ^= k2;
        h2 = mag_rol32(h2,17);
        h2 += h3;
        h2 = h2*5+0x0bcaa747;
        k3 *= c3;
        k3  = mag_rol32(k3,17);
        k3 *= c4;
        h3 ^= k3;
        h3 = mag_rol32(h3,15);
        h3 += h4;
        h3 = h3*5+0x96cd1c35;
        k4 *= c4;
        k4  = mag_rol32(k4,18);
        k4 *= c1;
        h4 ^= k4;
        h4 = mag_rol32(h4,13);
        h4 += h1;
        h4 = h4*5+0x32ac3b17;
    }
    const uint8_t *tail = (const uint8_t *)(p + nblocks*16);
    uint32_t k1 = 0;
    uint32_t k2 = 0;
    uint32_t k3 = 0;
    uint32_t k4 = 0;
    switch(len&15) {
    case 15:
        k4 ^= tail[14] << 16;
    case 14:
        k4 ^= tail[13] << 8;
    case 13:
        k4 ^= tail[12] << 0;
        k4 *= c4;
        k4 = mag_rol32(k4,18);
        k4 *= c1;
        h4 ^= k4;
    case 12:
        k3 ^= tail[11] << 24;
    case 11:
        k3 ^= tail[10] << 16;
    case 10:
        k3 ^= tail[9] << 8;
    case 9:
        k3 ^= tail[8] << 0;
        k3 *= c3;
        k3 = mag_rol32(k3,17);
        k3 *= c4;
        h3 ^= k3;
    case 8:
        k2 ^= tail[7] << 24;
    case 7:
        k2 ^= tail[6] << 16;
    case 6:
        k2 ^= tail[5] << 8;
    case 5:
        k2 ^= tail[4] << 0;
        k2 *= c2;
        k2 = mag_rol32(k2,16);
        k2 *= c3;
        h2 ^= k2;
    case 4:
        k1 ^= tail[3] << 24;
    case 3:
        k1 ^= tail[2] << 16;
    case 2:
        k1 ^= tail[1] << 8;
    case 1:
        k1 ^= tail[0] << 0;
        k1 *= c1;
        k1 = mag_rol32(k1,15);
        k1 *= c2;
        h1 ^= k1;
    };
    h1 ^= len;
    h2 ^= len;
    h3 ^= len;
    h4 ^= len;
    h1 += h2;
    h1 += h3;
    h1 += h4;
    h2 += h1;
    h3 += h1;
    h4 += h1;
    mag_mix32(h1);
    mag_mix32(h2);
    mag_mix32(h3);
    mag_mix32(h4);
    h1 += h2;
    h1 += h3;
    h1 += h4;
    h2 += h1;
    h3 += h1;
    h4 += h1;
    return (((uint64_t)h2)<<32)|h1;
#undef mag_rol32
#undef mag_mix32
}

bool mag_utf8_validate(const char *str, size_t len) {
    const uint8_t *data = (const uint8_t *)str;
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

char *mag_strdup(const char *s) {
    if (mag_unlikely(!s)) return NULL;
    size_t len = strlen(s);
    char *clone = (*mag_alloc)(NULL, len+1, 0);
    memcpy(clone, s, len);
    clone[len] = '\0';
    return clone;
}

void mag_path_split_dir_inplace(char *path, char **out_dir, char **out_file) {
#ifdef _WIN32
    char *sep = strrchr(path, '\\');
    if (!sep) sep = strrchr(path, '/');
#else
    char *sep = strrchr(path, '/');
#endif
    if (sep) {
        *sep = '\0';
        *out_dir = path;
        *out_file = sep+1;
    } else {
        *out_dir = path;
        *out_file = path;
    }
}

int mag_casecmp(const char *a, const char *b) {
    for (; *a && *b; a++, b++)
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
            return 0;
    return *a == 0 && *b == 0;
}
