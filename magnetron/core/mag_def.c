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
        case MAG_LOG_LEVEL_WARN: color = MAG_CC_YELLOW; break;
        case MAG_LOG_LEVEL_ERROR: color = MAG_CC_RED; break;
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

uint32_t mag_crc32c(const void *buffer, size_t size) {
    if (mag_unlikely(!buffer || !size)) return 0;
    const uint8_t *buf = buffer;
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
