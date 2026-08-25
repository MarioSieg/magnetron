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

#include "mag_mmap.h"

#ifdef _WIN32

#else
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#endif

static bool mag_mmap_preallocate_grow(
  #ifdef _WIN32
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
    if (mag_unlikely(r != 0)) {
      errno = r;
      return false;
    }
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
  #ifdef _WIN32
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

bool mag_mmap_file(mag_mapped_file_t *mf, const char *path, size_t size, mag_map_mode_t mode) {
  if (!mf || !path || !*path) return false;
  memset(mf, 0, sizeof(*mf));
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
      mf->writable = false;
      break;
    case MAG_MAP_WRITE:
      access = GENERIC_READ|GENERIC_WRITE;
      dispo = CREATE_ALWAYS;
      mf->writable = true;
      break;
    case MAG_MAP_READWRITE:
      access = GENERIC_READ|GENERIC_WRITE;
      dispo = OPEN_EXISTING;
      mf->writable = true;
      break;
    default:
      return false;
    }
    HANDLE hfile = CreateFileA(path, access, share, NULL, dispo, attrs, NULL);
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
    uint8_t *view = NULL;
    if (fs > 0) {
      DWORD pageProt = mf->writable ? PAGE_READWRITE : PAGE_READONLY;
      hmap = CreateFileMappingA(hfile, NULL, pageProt, (DWORD)((uint64_t)fs>>32), (DWORD)((uint64_t)fs&0xffffffff), NULL);
      if (mag_unlikely(!hmap)) {
        CloseHandle(hfile);
        return false;
      }
      DWORD mapAcc = mf->writable ? FILE_MAP_WRITE|FILE_MAP_READ : FILE_MAP_READ;
      view = (uint8_t *)MapViewOfFile(hmap, mapAcc, 0, 0, 0);
      if (mag_unlikely(!view)) {
        CloseHandle(hmap);
        CloseHandle(hfile);
        return false;
      }
    }
    mf->hfile = hfile;
    mf->hmap = hmap;
    mf->map = view;
    mf->fs = fs;
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
      mf->writable = false;
      break;
    case MAG_MAP_WRITE:
      oflags = O_RDWR|O_CREAT|O_TRUNC;
      prot = PROT_READ|PROT_WRITE;
      mflags = MAP_SHARED;
      mf->writable = true;
      break;
    case MAG_MAP_READWRITE:
      oflags = O_RDWR;
      prot = PROT_READ|PROT_WRITE;
      mflags = MAP_SHARED;
      mf->writable = true;
      break;
    default:
      return false;
    }
  #ifdef O_CLOEXEC
    oflags |= O_CLOEXEC;
  #endif
    int fd = open(path, oflags, 0666);
    if (mag_unlikely(fd == -1)) return false;
    size_t fs = 0;
    if (mode == MAG_MAP_WRITE) {
      if (mag_unlikely(!size)) {
        close(fd);
        return false;
      }
      if (mag_unlikely(!mag_mmap_preallocate_grow(fd, size))) {
        close(fd);
        return false;
      }
      fs = size;
    } else {
      struct stat st;
      if (mag_unlikely(fstat(fd, &st) == -1)) {
        close(fd);
        return false;
      }
      if (mag_unlikely(!S_ISREG(st.st_mode)))  {
        close(fd);
        return false;
      }
      fs = (size_t)st.st_size;
      if (mode == MAG_MAP_READWRITE && size > 0 && size != fs) {
        if (mag_unlikely(!mag_mmap_preallocate_grow(fd, size))) {
          close(fd);
          return false;
        }
        fs = size;
      }
    }
    uint8_t *view = NULL;
    if (fs > 0) {
      void *map = mmap(NULL, fs, prot, mflags, fd, 0);
      if (mag_unlikely(map == MAP_FAILED)) {
        close(fd);
        return false;
      }
      view = (uint8_t *)map;
    }
    mf->fd = fd;
    mf->map = view;
    mf->fs = fs;
    return true;
  #endif
}

bool mag_mflush_file(mag_mapped_file_t* mf) {
  if (mag_unlikely(!(mf && mf->map && mf->fs))) return false;
  if (!mf->writable) return true;
  #ifdef _WIN32
    if (!FlushViewOfFile(mf->map, 0)) return false;
    return mag_mmap_strong_fsync((HANDLE)mf->hfile);
  #else
    if (mag_unlikely(msync(mf->map, mf->fs, MS_SYNC) == -1)) return false;
    return mag_mmap_strong_fsync(mf->fd);
  #endif
}

bool mag_munmap_file(mag_mapped_file_t *mf) {
  if (mag_unlikely(!mf)) return false;
  bool ok = true;
  #if defined(_WIN32)
    if (mf->map && mf->fs) {
      if (mf->writable) {
        if (!FlushViewOfFile(mf->map, 0)) ok = false;
        if (!mag_mmap_strong_fsync((HANDLE)mf->hfile)) ok = false;
      }
      if (!UnmapViewOfFile(mf->map)) ok = false;
    }
    if (mf->hmap) {
      if (!CloseHandle((HANDLE)mf->hmap)) ok = false;
    }
    if (mf->hfile) {
      if (!CloseHandle((HANDLE)mf->hfile)) ok = false;
    }
    memset(mf, 0, sizeof(*mf));
    return ok;
  #else
    if (mf->map && mf->fs) {
      if (mf->writable) {
        if (mag_unlikely(msync(mf->map, mf->fs, MS_SYNC) == -1)) ok = false;
        if (mag_unlikely(!mag_mmap_strong_fsync(mf->fd))) ok = false;
      }
      if (mag_unlikely(munmap(mf->map, mf->fs) == -1)) ok = false;
    }
    if (mag_unlikely(close(mf->fd) == -1)) ok = false;
    memset(mf, 0, sizeof(*mf));
    return ok;
  #endif
}
