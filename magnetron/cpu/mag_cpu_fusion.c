#include "mag_cpu_fusion.h"

#include <core/mag_alloc.h>
#include <core/mag_dylib.h>
#include <core/mag_envcfg.h>
#include <core/mag_hash.h>
#include <core/mag_operator.h>

#include <errno.h>
#include <stdarg.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__unix__) || defined(__APPLE__)
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#define MAG_CPU_FUSE_CAN_COMPILE 1
#else
#define MAG_CPU_FUSE_CAN_COMPILE 0
#endif



/*
** A minimal append-only text buffer.
**
** Core has one of these, but it is an internal helper rather than something backends are meant to
** link against, and widening what core exports to save twenty lines here would be the wrong trade in
** a change whose whole point is keeping the two apart.
*/
typedef struct mag_cpu_fuse_buf_t {
  char *data;
  size_t len, cap;
  bool oom;
} mag_cpu_fuse_buf_t;

static void mag_cpu_fuse_buf_grow(mag_cpu_fuse_buf_t *b, size_t extra) {
  if (b->oom || b->len+extra+1 <= b->cap) return;
  size_t cap = b->cap ? b->cap : 1024;
  while (cap < b->len+extra+1) cap *= 2;
  char *grown = (*mag_alloc)(b->data, cap, 1); /* Contracted never to fail; the flag is belt and braces. */
  if (!grown) { b->oom = true; return; }
  b->data = grown;
  b->cap = cap;
}

static void mag_cpu_fuse_bputs(mag_cpu_fuse_buf_t *b, const char *s) {
  size_t n = strlen(s);
  mag_cpu_fuse_buf_grow(b, n);
  if (b->oom) return;
  memcpy(b->data+b->len, s, n);
  b->len += n;
  b->data[b->len] = 0;
}

static void mag_cpu_fuse_bprintf(mag_cpu_fuse_buf_t *b, const char *fmt, ...) {
  char tmp[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tmp, sizeof(tmp), fmt, ap);
  va_end(ap);
  mag_cpu_fuse_bputs(b, tmp);
}

/*
** C expression for each instruction.
**
** $0..$2 name the operands, already in named locals, so an operand appearing twice is evaluated
** once. Each expression is the arithmetic the eager kernel performs, written the same way round:
** the whole point of compiling a chain is to get the same bits faster, and an expression that
** reassociates is not the same expression.
*/
static const char *mag_cpu_fuse_form(uint8_t op) {
  switch (op) {
    case MAG_OP_ADD:   return "($0 + $1)";
    case MAG_OP_SUB:   return "($0 - $1)";
    case MAG_OP_MUL:   return "($0 * $1)";
    case MAG_OP_DIV:   return "($0 / $1)";
    case MAG_OP_MIN:   return "fminf($0, $1)";
    case MAG_OP_MAX:   return "fmaxf($0, $1)";
    case MAG_OP_NEG:   return "(-$0)";
    case MAG_OP_ABS:   return "fabsf($0)";
    case MAG_OP_SGN:   return "(float)(($0 > 0.0f) - ($0 < 0.0f))";
    case MAG_OP_SQR:   return "($0 * $0)";
    case MAG_OP_SQRT:  return "sqrtf($0)";
    case MAG_OP_FLOOR: return "floorf($0)";
    case MAG_OP_CEIL:  return "ceilf($0)";
    case MAG_OP_ROUND: return "roundf($0)";
    case MAG_OP_TRUNC: return "truncf($0)";
    case MAG_OP_STEP:  return "($0 > 0.0f ? 1.0f : 0.0f)";
    case MAG_OP_RELU:  return "fmaxf($0, 0.0f)";
    case MAG_OP_CLAMP: return "fminf(fmaxf($0, $1), $2)";
    default: return NULL;
  }
}

static void mag_cpu_fuse_operand_name(const mag_fuse_operand_t *o, char *buf, size_t cap) {
  switch (o->kind) {
    case MAG_FUSE_REG: snprintf(buf, cap, "r%u", o->idx); break;
    case MAG_FUSE_IMM: snprintf(buf, cap, "s%u", o->idx); break;
    case MAG_FUSE_SCL: snprintf(buf, cap, "b%u[0]", o->idx); break;
    default:           snprintf(buf, cap, "b%u[i]", o->idx); break;
  }
}

char *mag_cpu_fuse_codegen(const mag_fuse_graph_t *g) {
  if (g->dtype != MAG_DTYPE_FLOAT32) return NULL; /* Narrow floats need rounding at each step; not yet. */
  for (uint32_t i=0; i < g->num_ins; ++i)
    if (g->ins[i].op != MAG_FUSE_OP_LOAD && !mag_cpu_fuse_form(g->ins[i].op)) return NULL;

  bool written[MAG_FUSE_MAX_BUF];
  memset(written, 0, sizeof(written));
  for (uint8_t s=0; s < g->num_stores; ++s) written[g->stores[s].buf] = true;

  mag_cpu_fuse_buf_t ss = {0};
  mag_cpu_fuse_bprintf(&ss, "#include <math.h>\n\n");
  mag_cpu_fuse_bprintf(&ss, "void mag_fused(void *const *bufs, const double *imms, long long begin, long long end) {\n");
  for (uint8_t b=0; b < g->num_bufs; ++b) {
    /* A written buffer may alias one that is read, so only read-only operands get restrict. Telling
       the compiler otherwise would licence it to reorder a load across the store that feeds it. */
    if (written[b]) mag_cpu_fuse_bprintf(&ss, "  float *b%u = (float *)bufs[%u];\n", b, b);
    else            mag_cpu_fuse_bprintf(&ss, "  const float *restrict b%u = (const float *)bufs[%u];\n", b, b);
  }
  for (uint8_t k=0; k < g->num_imms; ++k)
    mag_cpu_fuse_bprintf(&ss, "  const float s%u = (float)imms[%u];\n", k, k);
  mag_cpu_fuse_bprintf(&ss, "  for (long long i=begin; i < end; ++i) {\n");
  for (uint32_t i=0; i < g->num_ins; ++i) {
    const mag_fuse_ins_t *ins = g->ins+i;
    char name[32];
    if (ins->op == MAG_FUSE_OP_LOAD) {
      mag_cpu_fuse_operand_name(ins->in+0, name, sizeof(name));
      mag_cpu_fuse_bprintf(&ss, "    const float r%u = %s;\n", i, name);
      continue;
    }
    const char *form = mag_cpu_fuse_form(ins->op);
    mag_cpu_fuse_bprintf(&ss, "    const float r%u = ", i);
    for (const char *p=form; *p; ++p) {
      if (*p == '$' && p[1] >= '0' && p[1] <= '2') {
        mag_cpu_fuse_operand_name(ins->in + (p[1]-'0'), name, sizeof(name));
        mag_cpu_fuse_bputs(&ss, name);
        ++p;
      } else {
        { char one[2] = {*p, 0}; mag_cpu_fuse_bputs(&ss, one); }
      }
    }
    mag_cpu_fuse_bprintf(&ss, ";\n");
  }
  for (uint8_t s=0; s < g->num_stores; ++s)
    mag_cpu_fuse_bprintf(&ss, "    b%u[i] = r%u;\n", g->stores[s].buf, g->stores[s].reg);
  mag_cpu_fuse_bprintf(&ss, "  }\n}\n");
  if (ss.oom) { mag_cpu_fuse_source_free(ss.data); return NULL; }
  return ss.data; /* Ownership passes to the caller. */
}

void mag_cpu_fuse_source_free(char *src) {
  if (src) (*mag_alloc)(src, 0, 0);
}

/* ---- compiling and caching ---- */

typedef struct mag_cpu_fuse_entry_t {
  uint64_t key;
  mag_dylib_t *lib;
  mag_cpu_fused_fn_t fn;
} mag_cpu_fuse_entry_t;

struct mag_cpu_fuse_cache_t {
  mag_cpu_fuse_entry_t *entries;
  size_t len, cap;
  uint64_t hits, compiles;
  bool unavailable; /* Latched once the toolchain has been shown to be missing. */
};

#if MAG_CPU_FUSE_CAN_COMPILE

/*
** Where compiled chains live between runs.
**
** Under the user's own directory rather than a shared temporary one. A world-writable location with
** a predictable name is somewhere another process can leave a library of its choosing and have this
** one load it, which is not a trade worth making to save a compile.
*/
static bool mag_cpu_fuse_cache_dir(char *out, size_t cap) {
  const char *dir = mag_envcfg_fuse_cache_dir();
  int len;
  if (dir && *dir) {
    len = snprintf(out, cap, "%s", dir);
  } else {
    const char *home = getenv("HOME");
    if (!home || !*home) return false;
    len = snprintf(out, cap, "%s/.cache/magnetron/fused", home);
  }
  /* A truncated path names a different directory than the one asked for, which is not somewhere to
     write a library this process will then load. */
  if (len < 0 || (size_t)len >= cap) return false;
  /* mkdir -p, one component at a time. Failures other than "already there" are fatal to caching
     but not to running: the caller falls back to interpreting. */
  for (char *p=out+1; *p; ++p) {
    if (*p != '/') continue;
    *p = '\0';
    if (mkdir(out, 0700) && errno != EEXIST) { *p = '/'; return false; }
    *p = '/';
  }
  return !mkdir(out, 0700) || errno == EEXIST;
}

/* Run the compiler with an argument vector. No shell: the compiler name comes from the environment
   and passing it through a shell would let it carry anything else along with it. */
static bool mag_cpu_fuse_invoke_cc(const char *cc, const char *csrc, const char *out) {
  char *argv[] = {
    (char *)cc,
    "-O3",
    "-ffp-contract=off", /* Load bearing: contraction into FMA would change the result. */
    "-fPIC",
    "-shared",
    "-o", (char *)out,
    (char *)csrc,
    NULL
  };
  pid_t pid = 0;
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
  int rc = posix_spawnp(&pid, cc, &fa, NULL, argv, environ);
  posix_spawn_file_actions_destroy(&fa);
  if (rc) return false;
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return false;
  return WIFEXITED(status) && !WEXITSTATUS(status);
}

static mag_cpu_fused_fn_t mag_cpu_fuse_build(const mag_fuse_graph_t *g, mag_dylib_t **out_lib) {
  char dir[512];
  if (!mag_cpu_fuse_cache_dir(dir, sizeof(dir))) return NULL;
  char *src = mag_cpu_fuse_codegen(g);
  if (!src) return NULL;

  /* Keyed on the generated text, so a change to codegen cannot pick up an object built by an older
     version of it. */
  uint64_t key = mag_murmur3_128_reduced_64(src, strlen(src), 0x5bf03635u);
  char lib_path[640];
  snprintf(lib_path, sizeof(lib_path), "%s/mag_fused_%016llx.%s", dir, (unsigned long long)key, MAG_DYLIB_EXT);

  mag_error_t err = {0};
  if (mag_iserr(mag_dylib_open(&err, out_lib, lib_path))) { /* Not built yet, or built by an older run. */
    char c_path[640];
    snprintf(c_path, sizeof(c_path), "%s/mag_fused_%016llx.c", dir, (unsigned long long)key);
    /* O_EXCL so a half-written source from a concurrent process is never handed to the compiler. */
    int fd = open(c_path, O_WRONLY|O_CREAT|O_EXCL|O_TRUNC, 0600);
    if (fd < 0) {
      if (errno != EEXIST) { mag_cpu_fuse_source_free(src); return NULL; }
      fd = open(c_path, O_WRONLY|O_TRUNC, 0600); /* Left behind by a run that died mid-compile. */
      if (fd < 0) { mag_cpu_fuse_source_free(src); return NULL; }
    }
    size_t len = strlen(src);
    bool wrote = write(fd, src, len) == (ssize_t)len;
    close(fd);
    bool built = wrote && mag_cpu_fuse_invoke_cc(mag_envcfg_fuse_cc(), c_path, lib_path);
    remove(c_path);
    if (!built) {
      /* The compiler's own complaint went to /dev/null, because failing here is not an error - the
         chain still runs, interpreted. Say that it happened, or the only symptom is that everything
         is quietly slower than it should be. */
      mag_log_warn("cpu: could not compile a fused chain with '%s'; running it interpreted instead.", mag_envcfg_fuse_cc());
      mag_cpu_fuse_source_free(src);
      return NULL;
    }
    memset(&err, 0, sizeof(err));
    if (mag_iserr(mag_dylib_open(&err, out_lib, lib_path))) { mag_cpu_fuse_source_free(src); return NULL; }
  }
  mag_cpu_fuse_source_free(src);
  mag_cpu_fused_fn_t fn = (mag_cpu_fused_fn_t)mag_dylib_sym(*out_lib, "mag_fused");
  if (!fn) { mag_dylib_close(*out_lib); *out_lib = NULL; return NULL; }
  return fn;
}

#else

static mag_cpu_fused_fn_t mag_cpu_fuse_build(const mag_fuse_graph_t *g, mag_dylib_t **out_lib) {
  (void)g; (void)out_lib;
  return NULL; /* No spawn available here; chains are interpreted. */
}

#endif

mag_cpu_fused_fn_t mag_cpu_fuse_resolve(mag_cpu_fuse_cache_t **slot, const mag_fuse_graph_t *g) {
  if (!mag_envcfg_fuse_compile_enabled()) return NULL;
  mag_cpu_fuse_cache_t *cache = *slot;
  if (!cache) {
    cache = (*mag_alloc)(NULL, sizeof(*cache), __alignof(mag_cpu_fuse_cache_t));
    if (!cache) return NULL;
    memset(cache, 0, sizeof(*cache));
    *slot = cache;
  }
  if (cache->unavailable) return NULL;

  /* Keyed on structure, not on the immediate values, so changing a learning rate is a hit. */
  uint64_t key = mag_fuse_graph_hash(g);
  for (size_t i=0; i < cache->len; ++i) {
    if (cache->entries[i].key != key) continue;
    ++cache->hits;
    return cache->entries[i].fn;
  }

  mag_dylib_t *lib = NULL;
  mag_cpu_fused_fn_t fn = mag_cpu_fuse_build(g, &lib);
  if (!fn) {
    /* Distinguish "this graph cannot be written as C" from "there is no compiler here". Only the
       latter is worth latching, and codegen answering NULL is the signal for the former. */
    char *probe = mag_cpu_fuse_codegen(g);
    if (probe) { mag_cpu_fuse_source_free(probe); cache->unavailable = true; }
    return NULL;
  }
  if (cache->len == cache->cap) {
    size_t cap = cache->cap ? cache->cap*2 : 16;
    mag_cpu_fuse_entry_t *grown = (*mag_alloc)(cache->entries, cap*sizeof(*grown), __alignof(mag_cpu_fuse_entry_t));
    if (!grown) { mag_dylib_close(lib); return NULL; }
    cache->entries = grown;
    cache->cap = cap;
  }
  cache->entries[cache->len++] = (mag_cpu_fuse_entry_t){.key = key, .lib = lib, .fn = fn};
  ++cache->compiles;
  return fn;
}

void mag_cpu_fuse_cache_stats(const mag_cpu_fuse_cache_t *cache, uint64_t *out_hits, uint64_t *out_compiles) {
  if (out_hits) *out_hits = cache ? cache->hits : 0;
  if (out_compiles) *out_compiles = cache ? cache->compiles : 0;
}

void mag_cpu_fuse_cache_destroy(mag_cpu_fuse_cache_t *cache) {
  if (!cache) return;
  for (size_t i=0; i < cache->len; ++i)
    if (cache->entries[i].lib) mag_dylib_close(cache->entries[i].lib);
  if (cache->entries) (*mag_alloc)(cache->entries, 0, 0);
  (*mag_alloc)(cache, 0, 0);
}
