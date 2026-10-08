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

#include "mag_mmap_ra.h"
#include "mag_alloc.h"
#include "mag_envcfg.h"
#include "mag_threadlib.h"

#ifndef _WIN32

#include <unistd.h>
#include <sys/mman.h>

typedef struct mag_ra_page_t {
  uint64_t off;
  uint64_t len;
} mag_ra_page_t;

typedef struct mag_ra_engine_t {
  int fd;
  const uint8_t *map;
  uint64_t fs;
  uint64_t window;
  uint64_t page;
  bool release;
  mag_thread_t *thread;
  mag_mutex_t *mtx;
  mag_condvar_t *cv;
  mag_ra_page_t *log;
  uint64_t *hash_key;
  uint32_t *hash_idx;
  uint32_t log_len;
  uint32_t next_issue;
  uint64_t seq_cursor;
  mag_ra_page_t *queue;
  uint32_t qhead;
  uint32_t qtail;
  uint8_t *scratch;
  bool stop;
} mag_ra_engine_t;

static mag_lock_t mag_ra_registry_lock = MAG_LOCK_INIT;
static mag_ra_engine_t *mag_ra_registry[MAG_RA_MAX_ENGINES];
static mag_atomic32_t mag_ra_active = 0;

static void mag_ra_free(mag_ra_engine_t *ra) {
  mag_error_t err = {0};
  if (ra->cv) mag_condvar_destroy(&err, ra->cv);
  if (ra->mtx) mag_mutex_destroy(&err, ra->mtx);
  if (ra->log) (*mag_alloc)(ra->log, 0, 0);
  if (ra->hash_key) (*mag_alloc)(ra->hash_key, 0, 0);
  if (ra->hash_idx) (*mag_alloc)(ra->hash_idx, 0, 0);
  if (ra->queue) (*mag_alloc)(ra->queue, 0, 0);
  if (ra->scratch) (*mag_alloc)(ra->scratch, 0, 0);
  (*mag_alloc)(ra, 0, 0);
}

void mag_mmap_ra_register(const mag_mapped_file_t *mf) {
  int mode = mag_envcfg_readahead();
  uint64_t window_mb = mag_envcfg_readahead_mb(MAG_RA_DEFAULT_WINDOW_MB);
  if (!mf->map || !mf->fs) return;
  if (!mode || !window_mb) {
    mag_log_info("mmap readahead (prefetch) disabled for %.2f GiB snapshot: %s", (double)mf->fs/(1<<30), !mode ? MAG_ENV_READAHEAD "=0" : MAG_ENV_READAHEAD_MB "=0");
    return;
  }
  long pg = sysconf(_SC_PAGESIZE);
  uint64_t page = pg > 0 ? (uint64_t)pg : 4096;
  long phys_pages = sysconf(_SC_PHYS_PAGES);
  uint64_t phys = phys_pages > 0 ? (uint64_t)phys_pages*page : 0;
  bool exceeds_ram = phys && mf->fs > phys/4*3;
  if (mode < 0 && !exceeds_ram) {
    mag_log_info("mmap readahead (prefetch) disabled for %.2f GiB snapshot: fits in %.2f GiB RAM (" MAG_ENV_READAHEAD "=auto)", (double)mf->fs/(1<<30), (double)phys/(1<<30));
    return;
  }
  mag_ra_engine_t *ra = (*mag_try_alloc)(NULL, sizeof(*ra), 0);
  if (!ra) return;
  memset(ra, 0, sizeof(*ra));
  ra->fd = mf->fd;
  ra->map = mf->map;
  ra->fs = mf->fs;
  ra->window = window_mb<<20;
  ra->page = page;
  int rel = mag_envcfg_readahead_release();
  bool release = rel < 0 ? exceeds_ram : rel > 0;
  ra->release = release;
  ra->log = (*mag_try_alloc)(NULL, sizeof(*ra->log)*MAG_RA_LOG_CAP, 0);
  ra->hash_key = (*mag_try_alloc)(NULL, sizeof(*ra->hash_key)*MAG_RA_HASH_CAP, 0);
  ra->hash_idx = (*mag_try_alloc)(NULL, sizeof(*ra->hash_idx)*MAG_RA_HASH_CAP, 0);
  ra->queue = (*mag_try_alloc)(NULL, sizeof(*ra->queue)*MAG_RA_QUEUE_CAP, 0);
  ra->scratch = (*mag_try_alloc)(NULL, MAG_RA_CHUNK, 0);
  mag_error_t err = {0};
  if (mag_unlikely(!ra->log || !ra->hash_key
    || !ra->hash_idx || !ra->queue || !ra->scratch
    || mag_iserr(mag_mutex_create(&err, &ra->mtx))
    || mag_iserr(mag_condvar_create(&err, &ra->cv)))) {
    mag_log_warn("mmap readahead (prefetch) disabled: engine allocation failed");
    mag_ra_free(ra);
    return;
  }
  memset(ra->hash_key, 0, sizeof(*ra->hash_key)*MAG_RA_HASH_CAP);
  mag_lock_acquire(&mag_ra_registry_lock);
  for (int i=0; i < MAG_RA_MAX_ENGINES; ++i) {
    if (!mag_ra_registry[i]) {
      mag_ra_registry[i] = ra;
      mag_atomic32_fetch_add(&mag_ra_active, 1, MAG_MO_RELAXED);
      ra = NULL;
      break;
    }
  }
  mag_lock_release(&mag_ra_registry_lock);
  if (ra) {
    mag_log_warn("mmap readahead (prefetch) disabled for %.2f GiB snapshot: all %u engine slots in use", (double)mf->fs/(1<<30), MAG_RA_MAX_ENGINES);
    mag_ra_free(ra);
    return;
  }
  mag_log_info(
    "mmap readahead (prefetch) enabled for %.2f GiB snapshot: %s, window %llu MiB, release-behind %s",
    (double)mf->fs/(1<<30),
    mode > 0 ? "forced by " MAG_ENV_READAHEAD "=1" : "exceeds 3/4 of RAM",
    (unsigned long long)window_mb, release ? "on" : "off"
  );
}

static void mag_ra_worker(void *arg) {
  mag_ra_engine_t *ra = arg;
  mag_error_t err = {0};
  for (;;) {
    mag_mutex_lock(&err, ra->mtx);
    while (!ra->stop && ra->qhead == ra->qtail) mag_condvar_wait(&err, ra->cv, ra->mtx);
    if (ra->stop) {
      mag_mutex_unlock(&err, ra->mtx);
      return;
    }
    mag_ra_page_t r = ra->queue[ra->qhead++%MAG_RA_QUEUE_CAP];
    mag_mutex_unlock(&err, ra->mtx);
    uint64_t end = r.off+r.len;
    for (uint64_t o=r.off; o < end && !ra->stop; o += MAG_RA_CHUNK) {
      size_t n = (size_t)(end-o < MAG_RA_CHUNK ? end-o : MAG_RA_CHUNK);
      if (pread(ra->fd, ra->scratch, n, (off_t)o) <= 0) break;
    }
  }
}

void mag_mmap_ra_unregister(const mag_mapped_file_t *mf) {
  mag_ra_engine_t *ra = NULL;
  mag_lock_acquire(&mag_ra_registry_lock);
  for (int i=0; i < MAG_RA_MAX_ENGINES; ++i) {
    if (mag_ra_registry[i] && mag_ra_registry[i]->map == mf->map) {
      ra = mag_ra_registry[i];
      mag_ra_registry[i] = NULL;
      mag_atomic32_fetch_sub(&mag_ra_active, 1, MAG_MO_RELAXED);
      break;
    }
  }
  mag_lock_release(&mag_ra_registry_lock);
  if (!ra) return;
  mag_error_t err = {0};
  mag_mutex_lock(&err, ra->mtx);
  ra->stop = true;
  mag_condvar_broadcast(&err, ra->cv);
  mag_mutex_unlock(&err, ra->mtx);
  if (ra->thread) mag_thread_join(&err, ra->thread);
  mag_ra_free(ra);
}

static uint32_t mag_ra_hash_slot(uint64_t key) {
  uint64_t h = (key>>4)*0x9e3779b97f4a7c15ull;
  return (uint32_t)(h>>40) & (MAG_RA_HASH_CAP-1);
}

static bool mag_ra_hash_find(const mag_ra_engine_t *ra, uint64_t key, uint32_t *idx) {
  uint32_t slot = mag_ra_hash_slot(key);
  for (uint32_t i=0; i < MAG_RA_HASH_CAP; ++i) {
    uint64_t k = ra->hash_key[slot+i & (MAG_RA_HASH_CAP-1)];
    if (!k) return false;
    if (k == key) {
      *idx = ra->hash_idx[slot+i & (MAG_RA_HASH_CAP-1)];
      return true;
    }
  }
  return false;
}

static void mag_ra_hash_set(mag_ra_engine_t *ra, uint64_t key, uint32_t idx) {
  uint32_t slot = mag_ra_hash_slot(key);
  for (uint32_t i=0; i < MAG_RA_HASH_CAP; ++i) {
    uint32_t s = (slot+i) & (MAG_RA_HASH_CAP-1);
    if (!ra->hash_key[s] || ra->hash_key[s] == key) {
      ra->hash_key[s] = key;
      ra->hash_idx[s] = idx;
      return;
    }
  }
}

static void mag_ra_enqueue(mag_ra_engine_t *ra, uint64_t off, uint64_t len) {
  if (ra->qtail-ra->qhead >= MAG_RA_QUEUE_CAP) return;
  ra->queue[ra->qtail++%MAG_RA_QUEUE_CAP] = (mag_ra_page_t){off, len};
}

static void mag_ra_hint(mag_ra_engine_t *ra, uint64_t off, uint64_t len) {
  mag_error_t err = {0};
  uint64_t end = off+len;
  off &= ~(ra->page-1);
  end = (end+ra->page-1) & ~(ra->page-1);
  if (end > ra->fs) end = ra->fs;
  len = end-off;
  uint64_t key = off+1;
  mag_mutex_lock(&err, ra->mtx);
  if (mag_unlikely(!ra->thread && mag_iserr(mag_thread_create(&err, &ra->thread, &mag_ra_worker, MAG_THREAD_PRIO_NORMAL, "mag_readahead", ra)))) {
    ra->thread = NULL;
    mag_mutex_unlock(&err, ra->mtx);
    return;
  }
  uint32_t cur = ra->log_len;
  uint32_t prev = 0;
  bool replay = mag_ra_hash_find(ra, key, &prev) && cur-prev < MAG_RA_LOG_CAP && ra->log[prev%MAG_RA_LOG_CAP].off == off;
  ra->log[cur%MAG_RA_LOG_CAP] = (mag_ra_page_t){off, len};
  mag_ra_hash_set(ra, key, cur);
  ra->log_len = cur+1;
  if (replay) {
    uint32_t start = ra->next_issue > prev+1 ? ra->next_issue : prev+1;
    uint64_t ahead = 0;
    for (uint32_t j=prev+1; j < start; ++j) ahead += ra->log[j%MAG_RA_LOG_CAP].len;
    uint32_t j = start;
    for (; j < cur && ahead < ra->window; ++j) {
      mag_ra_page_t r = ra->log[j%MAG_RA_LOG_CAP];
      mag_ra_enqueue(ra, r.off, r.len);
      ahead += r.len;
    }
    ra->next_issue = j;
  } else {
    uint64_t s = end;
    if (s < ra->seq_cursor && ra->seq_cursor-s <= ra->window) s = ra->seq_cursor;
    uint64_t f = end+ra->window < ra->fs ? end+ra->window : ra->fs;
    if (s < f) {
      mag_ra_enqueue(ra, s, f-s);
      ra->seq_cursor = f;
    }
  }
  mag_condvar_signal(&err, ra->cv);
  mag_mutex_unlock(&err, ra->mtx);
}

static mag_ra_engine_t *mag_ra_find(const void *ptr) {
  mag_ra_engine_t *ra = NULL;
  mag_lock_acquire(&mag_ra_registry_lock);
  for (int i=0; i < MAG_RA_MAX_ENGINES; ++i) {
    mag_ra_engine_t *c = mag_ra_registry[i];
    if (c && (const uint8_t *)ptr >= c->map && (const uint8_t *)ptr < c->map+c->fs) {
      ra = c;
      break;
    }
  }
  mag_lock_release(&mag_ra_registry_lock);
  return ra;
}

void mag_mmap_readahead_hint(const void *ptr, size_t len) {
  if (len < MAG_RA_MIN_HINT_BYTES || !mag_atomic32_load(&mag_ra_active, MAG_MO_RELAXED)) return;
  mag_ra_engine_t *ra = mag_ra_find(ptr);
  if (!ra) return;
  mag_ra_hint(ra, (uint64_t)((const uint8_t *)ptr-ra->map), (uint64_t)len);
}

void mag_mmap_release_hint(const void *ptr, size_t len) {
  if (len < MAG_RA_MIN_HINT_BYTES || !mag_atomic32_load(&mag_ra_active, MAG_MO_RELAXED)) return;
  mag_ra_engine_t *ra = mag_ra_find(ptr);
  if (!ra || !ra->release) return;
  uint64_t off = (uint64_t)((const uint8_t *)ptr-ra->map);
  uint64_t end = off+len;
  off=(off+ra->page-1)&-ra->page;
  end&=-ra->page;
  if (end > ra->fs) end = ra->fs;
  if (end > off) madvise((void *)(ra->map+off), (size_t)(end-off), MADV_DONTNEED);
}

#else

void mag_mmap_ra_register(const mag_mapped_file_t *mf) {
  (void)mf;
}

void mag_mmap_ra_unregister(const mag_mapped_file_t *mf) {
  (void)mf;
}

void mag_mmap_readahead_hint(const void *ptr, size_t len) {
  (void)ptr; (void)len;
}

void mag_mmap_release_hint(const void *ptr, size_t len) {
  (void)ptr; (void)len;
}

#endif
