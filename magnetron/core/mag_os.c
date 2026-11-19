/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "mag_os.h"
#include "mag_hash.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/syscall.h>
#ifdef __APPLE__
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 101200
extern int getentropy(void *buf, size_t len);
#endif
#endif
#endif

char *mag_os_chmod_path(void) {
#if defined(_WIN32)
  HMODULE hModule = NULL;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)&mag_os_chmod_path, &hModule))
    return NULL;
  char buf[MAX_PATH];
  DWORD len = GetModuleFileNameA(hModule, buf, MAX_PATH);
  if (!len || len == MAX_PATH) return NULL;
  return mag_strdup(buf);
#elif defined(__linux__) || defined(__APPLE__)
  Dl_info info;
  volatile void *sym = &mag_os_chmod_path;
  if (dladdr((void *)sym, &info) && info.dli_fname) {
    char *real = realpath(info.dli_fname, NULL);
    char *ret = real ? mag_strdup(real) : mag_strdup(info.dli_fname);
    if (real) free(real);
    return ret;
  }
  return NULL;
#else
#error "Not implemented for this platform"
#endif
}

#ifdef _WIN32
typedef BOOLEAN (WINAPI *mag_win32_prgr_fn_t)(void *buf, ULONG len);
static mag_win32_prgr_fn_t mag_win32_prgr_fn;
#endif

bool mag_os_sample_crypto_entropy(void *buf, size_t len) {
#if defined(__linux__) && SYS_getrandom
  return syscall(SYS_getrandom, buf, len, 0) == len;
#elif defined(__APPLE__) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101200
  return getentropy(buf, len) == 0;
#elif defined(_WIN32)
  if (!mag_win32_prgr_fn) {
    HMODULE lib = LoadLibraryExA("advapi32.dll", NULL, 0);
    if (mag_unlikely(!lib)) return false;
    mag_win32_prgr_fn = (mag_win32_prgr_fn_t)GetProcAddress(lib, "SystemFunction036");
    if (mag_unlikely(!mag_win32_prgr_fn)) return false;
  }
  return (*mag_win32_prgr_fn)(buf, (ULONG)len);
#else
  int fd = open("/dev/urandom", O_RDONLY|O_CLOEXEC);
  if (mag_unlikely(fd == -1)) return false;
  ssize_t n = read(fd, buf, len);
  (void)close(fd);
  return n == (ssize_t)len;
#endif
}

uint64_t mag_os_boot_uid(void) {
  char buf[512] = {0};
  size_t n=0;
  if (gethostname(buf, 255) == 0) { /* TODO: win32 */
    buf[255] = '\0';
    n=strlen(buf);
  }
  buf[n++] = '/';
  #ifdef __linux__   /* distinguishes containers or VMs that share a hostname. */
    FILE *f = fopen("/proc/sys/kernel/random/boot_id", "r");
    if (f) {
      n += fread(buf+n, 1, sizeof(buf)-n-1, f);
      fclose(f);
    }
  #elif defined(__APPLE__)
    size_t blen = sizeof(ident)-len-1;
    if (sysctlbyname("kern.bootsessionuuid", ident+len, &blen, NULL, 0) == 0)
      len += strnlen(ident+len, blen);
  #endif
  return mag_murmur3_128_red64(buf, n, 0x686f7374u);
}
