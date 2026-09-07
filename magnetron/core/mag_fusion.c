/*
** Pointwise fusion JIT. See mag_fusion.h for why this generates code instead of interpreting.
*/

#include "mag_fusion.h"
#include "mag_context.h"
#include "mag_dylib.h"
#include "mag_alloc.h"
#include "mag_hash.h"
#include "mag_envcfg.h"
#include "mag_sstream.h"
#include "mag_tensor.h"

#include <stdio.h>
#include <stdlib.h>

/*
** C expression for each fusible opcode.
**
** Only operations whose CPU kernels are exact IEEE arithmetic appear here. The transcendentals are
** deliberately absent: magnetron's vector kernels approximate them (tanh is 2/(1+exp(-2x))-1 built
** on a reciprocal estimate plus Newton steps, see mag_cpu_simd_functions.h), so libm in generated
** code would not reproduce them and fusing would silently change results. They end a chain instead.
** Adding one means emitting magnetron's own approximation, not calling libm. $0..$2 are the operands, already materialized into named
** locals, so an operand appearing twice costs nothing and has no double-evaluation hazard.
** An opcode absent from this table is not fusible and ends a chain.
*/
typedef struct mag_fuse_op_form_t {
  const char *f32;    /* Expression for float and, via promotion, the narrower float types. */
  uint8_t arity;
} mag_fuse_op_form_t;

static const mag_fuse_op_form_t *mag_fuse_form(mag_opcode_t op) {
  static const mag_fuse_op_form_t forms[MAG_OP__NUM] = {
    [MAG_OP_ADD]          = {"($0 + $1)", 2},
    [MAG_OP_SUB]          = {"($0 - $1)", 2},
    [MAG_OP_MUL]          = {"($0 * $1)", 2},
    [MAG_OP_DIV]          = {"($0 / $1)", 2},
    [MAG_OP_MIN]          = {"fminf($0, $1)", 2},
    [MAG_OP_MAX]          = {"fmaxf($0, $1)", 2},
    [MAG_OP_NEG]          = {"(-$0)", 1},
    [MAG_OP_ABS]          = {"fabsf($0)", 1},
    [MAG_OP_SGN]          = {"(($0 > 0.0f) - ($0 < 0.0f))", 1},
    [MAG_OP_SQR]          = {"($0 * $0)", 1},
    [MAG_OP_SQRT]         = {"sqrtf($0)", 1},
    [MAG_OP_FLOOR]        = {"floorf($0)", 1},
    [MAG_OP_CEIL]         = {"ceilf($0)", 1},
    [MAG_OP_ROUND]        = {"roundf($0)", 1},
    [MAG_OP_TRUNC]        = {"truncf($0)", 1},
    [MAG_OP_STEP]         = {"($0 > 0.0f ? 1.0f : 0.0f)", 1},
    [MAG_OP_RELU]         = {"fmaxf($0, 0.0f)", 1},
    [MAG_OP_CLAMP]        = {"fminf(fmaxf($0, $1), $2)", 3},
  };
  const mag_fuse_op_form_t *f = forms+op;
  return f->f32 ? f : NULL;
}

bool mag_fuse_op_is_pointwise(mag_opcode_t op) {
  return op < MAG_OP__NUM && mag_fuse_form(op) != NULL;
}

void mag_fuse_plan_init(mag_fuse_plan_t *plan, mag_dtype_t dtype) {
  memset(plan, 0, sizeof(*plan));
  plan->dtype = (uint8_t)dtype;
}

int32_t mag_fuse_load(mag_fuse_plan_t *plan, uint8_t buf) {
  if (plan->num_ins >= MAG_FUSE_MAX_INS || plan->num_ins >= MAG_FUSE_MAX_REG) return -1;
  if (buf >= MAG_FUSE_MAX_BUF) return -1;
  if (buf >= plan->num_bufs) plan->num_bufs = buf+1;
  mag_fuse_ins_t *ins = plan->ins + plan->num_ins;
  ins->op = MAG_OP_NOP;             /* NOP means a plain load of buffer element i. */
  ins->dst = (uint8_t)plan->num_ins;
  ins->num_in = 1;
  ins->in[0] = (mag_fuse_operand_t){.kind = MAG_FUSE_BUF, .idx = buf};
  return (int32_t)plan->num_ins++;
}

int32_t mag_fuse_load_scalar(mag_fuse_plan_t *plan, uint8_t buf) {
  if (plan->num_ins >= MAG_FUSE_MAX_INS || plan->num_ins >= MAG_FUSE_MAX_REG) return -1;
  if (buf >= MAG_FUSE_MAX_BUF) return -1;
  if (buf >= plan->num_bufs) plan->num_bufs = buf+1;
  mag_fuse_ins_t *ins = plan->ins + plan->num_ins;
  ins->op = MAG_OP_NOP;
  ins->dst = (uint8_t)plan->num_ins;
  ins->num_in = 1;
  ins->in[0] = (mag_fuse_operand_t){.kind = MAG_FUSE_SCL, .idx = buf};
  return (int32_t)plan->num_ins++;
}

int32_t mag_fuse_emit(mag_fuse_plan_t *plan, mag_opcode_t op, const mag_fuse_operand_t *in, uint8_t num_in) {
  const mag_fuse_op_form_t *form = mag_fuse_form(op);
  if (!form || form->arity != num_in) return -1;
  if (plan->num_ins >= MAG_FUSE_MAX_INS || plan->num_ins >= MAG_FUSE_MAX_REG) return -1;
  for (uint8_t i=0; i < num_in; ++i) {
    if (in[i].kind == MAG_FUSE_IMM && in[i].idx >= MAG_FUSE_MAX_IMM) return -1;
    if (in[i].kind == MAG_FUSE_BUF && in[i].idx >= MAG_FUSE_MAX_BUF) return -1;
    if (in[i].kind == MAG_FUSE_REG && (uint32_t)in[i].idx >= plan->num_ins) return -1;
    if (in[i].kind == MAG_FUSE_IMM && in[i].idx >= plan->num_imms) plan->num_imms = in[i].idx+1;
    if (in[i].kind == MAG_FUSE_BUF && in[i].idx >= plan->num_bufs) plan->num_bufs = in[i].idx+1;
  }
  mag_fuse_ins_t *ins = plan->ins + plan->num_ins;
  ins->op = (uint8_t)op;
  ins->dst = (uint8_t)plan->num_ins;
  ins->num_in = num_in;
  memcpy(ins->in, in, num_in*sizeof(*in));
  return (int32_t)plan->num_ins++;
}

bool mag_fuse_store(mag_fuse_plan_t *plan, uint8_t buf, int32_t reg) {
  if (reg < 0 || (uint32_t)reg >= plan->num_ins) return false;
  if (plan->num_stores >= MAG_FUSE_MAX_BUF || buf >= MAG_FUSE_MAX_BUF) return false;
  if (buf >= plan->num_bufs) plan->num_bufs = buf+1;
  plan->stores[plan->num_stores++] = (mag_fuse_store_t){.buf = buf, .reg = (uint8_t)reg};
  return true;
}

/* ---- code generation ---- */


/*
** Per-dtype prologue for the generated kernel.
**
** Narrow types are computed in float, exactly as the eager kernels do, but eager materializes a
** tensor after every operator and therefore rounds every intermediate back to the storage type.
** The fused kernel has to round at the same points or the results drift, so mag_rt() is applied to
** each instruction's result. That keeps bit-identity, and it is only worth doing when the
** conversion is a hardware instruction; see mag_fuse_elem_ctype. For float32 all three helpers are
** the identity.
*/
static const char *mag_fuse_elem_ctype(mag_dtype_t dt) {
  switch (dt) {
    case MAG_DTYPE_FLOAT32: return "float";
    case MAG_DTYPE_FLOAT16: return "uint16_t";
    /* bfloat16 is absent on purpose. Its conversion is soft-fp, and mag_rt runs on every
       instruction result, so a 20-operator chain pays 20 software round-trips per element while
       the eager kernels convert a whole vector at a time. Measured on an Adam step it is 1.1x
       slower fused than eager at 64K elements and above, so it stays eager. */
    default: return NULL;
  }
}

static void mag_fuse_emit_prologue(mag_sstream_t *ss, mag_dtype_t dt) {
  switch (dt) {
    case MAG_DTYPE_FLOAT32:
      mag_sstream_append(ss,
        "static inline float mag_ld(float x) { return x; }\n"
        "static inline float mag_st(float x) { return x; }\n"
        "#define mag_rt(x) (x)\n\n");
      break;
    case MAG_DTYPE_FLOAT16: /* mirror mag_cpu_dispatch.h; #error where we cannot guarantee a match */
      mag_sstream_append(ss,
        "#if defined(__ARM_NEON) && !defined(_MSC_VER)\n"
        "static inline float mag_ld(uint16_t b) { union { __fp16 f; uint16_t u; } c = {.u=b}; return c.f; }\n"
        "static inline uint16_t mag_st(float x) { union { __fp16 f; uint16_t u; } c = {.f=(__fp16)x}; return c.u; }\n"
        "#elif defined(__F16C__) && !defined(_MSC_VER)\n"
        "#include <immintrin.h>\n"
        "static inline float mag_ld(uint16_t b) { return _cvtsh_ss(b); }\n"
        "static inline uint16_t mag_st(float x) { return _cvtss_sh(x, 0); }\n"
        "#else\n"
        "#error \"no float16 conversion here matches the eager path; fall back to eager\"\n"
        "#endif\n"
        "#define mag_rt(x) mag_ld(mag_st(x))\n\n");
      break;
    default: break;
  }
}

static void mag_fuse_operand_name(const mag_fuse_operand_t *o, char (*out)[32]) {
  switch (o->kind) {
    case MAG_FUSE_REG: snprintf(*out, sizeof(*out), "r%u", o->idx); break;
    case MAG_FUSE_BUF: snprintf(*out, sizeof(*out), "b%u[i]", o->idx); break;
    case MAG_FUSE_SCL: snprintf(*out, sizeof(*out), "b%u[0]", o->idx); break;
    default:           snprintf(*out, sizeof(*out), "s%u", o->idx); break;
  }
}

static void mag_fuse_codegen_ss(const mag_fuse_plan_t *plan, mag_sstream_t *ss) {
  mag_sstream_append(ss,
    "/* generated by magnetron pointwise fusion, do not edit */\n"
    "#include <math.h>\n"
    "#include <stdint.h>\n\n");
  mag_fuse_emit_prologue(ss, (mag_dtype_t)plan->dtype);
  mag_sstream_append(ss, "void mag_fused(void *const *bufs, const double *imms, int64_t begin, int64_t end) {\n");
  const char *ct = mag_fuse_elem_ctype((mag_dtype_t)plan->dtype);
  for (uint8_t b=0; b < plan->num_bufs; ++b) {
    bool written = false;
    for (uint8_t k=0; k < plan->num_stores; ++k) if (plan->stores[k].buf == b) written = true;
    /* A written buffer may alias one that is read, and the loop reads element i before storing it,
       so only read-only buffers are marked restrict. */
    if (written) mag_sstream_append(ss, "  %s *b%u = (%s *)bufs[%u];\n", ct, b, ct, b);
    else mag_sstream_append(ss, "  const %s *restrict b%u = (const %s *)bufs[%u];\n", ct, b, ct, b);
  }
  for (uint8_t k=0; k < plan->num_imms; ++k)
    /* Eager converts a scalar operand to the tensor's dtype before using it
       (mag_scalar_to_bfloat16 and friends), so a narrow chain must round immediates too. */
    mag_sstream_append(ss, "  const float s%u = mag_rt((float)imms[%u]);\n", k, k);
  mag_sstream_append(ss, "  for (int64_t i=begin; i < end; ++i) {\n");
  for (uint32_t k=0; k < plan->num_ins; ++k) {
    const mag_fuse_ins_t *ins = plan->ins + k;
    char names[3][32];
    for (uint8_t j=0; j < ins->num_in; ++j) mag_fuse_operand_name(ins->in+j, &names[j]);
    if (ins->op == MAG_OP_NOP) { /* a plain load of buffer element i */
      mag_sstream_append(ss, "    const float r%u = mag_ld(%s);\n", ins->dst, names[0]);
      continue;
    }
    const mag_fuse_op_form_t *form = mag_fuse_form((mag_opcode_t)ins->op);
    mag_sstream_append(ss, "    const float r%u = mag_rt(", ins->dst);
    for (const char *p = form->f32; *p; ++p) { /* expand $0..$2 into the operand names */
      if (*p == '$' && p[1] >= '0' && p[1] <= '2') {
        const char *nm = names[p[1]-'0'];
        mag_sstream_append_strn(ss, nm, strlen(nm));
        ++p;
      } else {
        mag_sstream_putc(ss, *p);
      }
    }
    mag_sstream_append_strn(ss, ");\n", 3);
  }
  for (uint8_t k=0; k < plan->num_stores; ++k)
    mag_sstream_append(ss, "    b%u[i] = mag_st(r%u);\n", plan->stores[k].buf, plan->stores[k].reg);
  mag_sstream_append(ss, "  }\n}\n");
}

char *mag_fuse_codegen(const mag_fuse_plan_t *plan) {
  mag_sstream_t ss;
  mag_sstream_init(&ss);
  mag_fuse_codegen_ss(plan, &ss);
  if (ss.oom) { mag_sstream_free(&ss); return NULL; }
  return ss.buf; /* ownership moves to the caller */
}

void mag_fuse_source_free(char *src) {
  if (src) (*mag_alloc)(src, 0, 0);
}

/* ---- compile and cache ---- */

typedef struct mag_fuse_cache_entry_t {
  uint64_t key;
  mag_dylib_t *lib;
  mag_fused_fn_t fn;
} mag_fuse_cache_entry_t;

struct mag_fuse_cache_t {
  mag_fuse_cache_entry_t *entries;
  uint32_t len, cap;
  uint64_t hits, compiles;
  bool compiler_unavailable;   /* Latched after a failed compile so we stop retrying every call. */
};

/*
** Two keys, because they answer different questions.
**
** The in-process cache is keyed on the plan structure, which is cheap enough to compute on every
** call: codegen cannot change while the process runs, so identical structure means identical code.
** Generating the source just to key this lookup would put a few hundred microseconds of string
** building in front of a kernel that runs in tens, which is what a captured chain pays per entry.
**
** The on-disk name is keyed on the source itself, computed only when we are about to compile. That
** is what makes a codegen change invalidate objects left by an earlier build instead of silently
** loading a stale one.
*/
static uint64_t mag_fuse_plan_hash(const mag_fuse_plan_t *plan) {
  size_t used = offsetof(mag_fuse_plan_t, ins) + plan->num_ins*sizeof(plan->ins[0]);
  uint64_t h = mag_murmur3_128_reduced_64(plan, used, 0x9e3779b9u);
  return h ^ mag_murmur3_128_reduced_64(plan->stores, plan->num_stores*sizeof(plan->stores[0]), (uint32_t)h);
}

static uint64_t mag_fuse_source_hash(const char *src) {
  return mag_murmur3_128_reduced_64(src, strlen(src), 0x9e3779b9u);
}

static const char *mag_fuse_compiler(void) {
  const char *cc = mag_envcfg_raw(MAG_ENV_JIT_CC);
  return cc ? cc : MAG_JIT_DEFAULT_CC;
}

static mag_status_t mag_fuse_build(mag_error_t *err, uint64_t key, mag_sstream_t *src, mag_dylib_t **out_lib, mag_fused_fn_t *out_fn) {
  const char *dir = mag_envcfg_raw(MAG_ENV_JIT_CACHE_DIR);
  char base[512];
  snprintf(base, sizeof(base), "%s/mag_fused_%016llx", dir ? dir : MAG_JIT_DEFAULT_CACHE_DIR, (unsigned long long)key);
  char cpath[600], lpath[600], cmd[1600];
  snprintf(cpath, sizeof(cpath), "%s.c", base);
  snprintf(lpath, sizeof(lpath), "%s.%s", base, MAG_DYLIB_EXT);

  /* An object left on disk by an earlier run is reused, so a repeated session pays no compile. */
  if (mag_iserr(mag_dylib_open(NULL, out_lib, lpath))) {
    if (!mag_sstream_flush(src, cpath))
      return mag_set_error(err, MAG_ERR_IO, "fusion: cannot write generated source to '%s'.", cpath);
    snprintf(cmd, sizeof(cmd), "%s -O3 -ffp-contract=off -fPIC -shared -o '%s' '%s' 2>/dev/null",
      mag_fuse_compiler(), lpath, cpath);
    int rc = system(cmd);
    remove(cpath);
    if (rc != 0)
      return mag_set_error(err, MAG_ERR_BACKEND, "fusion: host compiler '%s' failed (exit %d); set " MAG_ENV_JIT_CC " or disable the JIT.", mag_fuse_compiler(), rc);
    mag_log_info("JIT: compiled fused kernel %016llx with '%s'", (unsigned long long)key, mag_fuse_compiler());
    mag_status_t st = mag_dylib_open(err, out_lib, lpath);
    if (mag_iserr(st)) return st;
  }
  *out_fn = (mag_fused_fn_t)mag_dylib_sym(*out_lib, "mag_fused");
  if (!*out_fn) {
    mag_dylib_close(*out_lib);
    *out_lib = NULL;
    return mag_set_error(err, MAG_ERR_BACKEND, "fusion: compiled object '%s' has no mag_fused symbol.", lpath);
  }
  return MAG_OK;
}

mag_status_t mag_fuse_compile(mag_error_t *err, mag_context_t *ctx, const mag_fuse_plan_t *plan, mag_fused_fn_t *out_fn) {
  *out_fn = NULL;
  if (mag_unlikely(!mag_fuse_elem_ctype((mag_dtype_t)plan->dtype)))
    return mag_set_error(err, MAG_ERR_OP, "fusion: dtype '%s' has no fusible representation.",
      mag_type_trait((mag_dtype_t)plan->dtype)->name);
  struct mag_fuse_cache_t *cache = ctx->fuse_cache;
  if (!cache) {
    cache = (*mag_try_alloc)(NULL, sizeof(*cache), 0);
    if (!cache) return mag_set_error(err, MAG_ERR_OOM, "fusion: cannot allocate kernel cache.");
    memset(cache, 0, sizeof(*cache));
    ctx->fuse_cache = cache;
  }
  static int jit_enabled = -1;
  if (mag_unlikely(jit_enabled < 0)) jit_enabled = mag_envcfg_jit_enabled() ? 1 : 0;
  if (!jit_enabled)
    return mag_set_error(err, MAG_ERR_BACKEND, "fusion: disabled by " MAG_ENV_JIT "=off.");
  if (cache->compiler_unavailable)
    return mag_set_error(err, MAG_ERR_BACKEND, "fusion: no usable host compiler, falling back to eager.");
  uint64_t key = mag_fuse_plan_hash(plan);
  for (uint32_t i=0; i < cache->len; ++i) {
    if (cache->entries[i].key == key) {
      ++cache->hits;
      *out_fn = cache->entries[i].fn;
      return MAG_OK;
    }
  }
  mag_sstream_t src;
  mag_sstream_init(&src);
  mag_fuse_codegen_ss(plan, &src);
  if (src.oom) { mag_sstream_free(&src); return mag_set_error(err, MAG_ERR_OOM, "fusion: cannot build kernel source."); }
  mag_dylib_t *lib = NULL;
  mag_fused_fn_t fn = NULL;
  mag_status_t st = mag_fuse_build(err, mag_fuse_source_hash(src.buf), &src, &lib, &fn);
  mag_sstream_free(&src);
  if (mag_iserr(st)) {
    cache->compiler_unavailable = true; /* Do not pay a failed compile on every call. */
    return st;
  }
  if (cache->len == cache->cap) {
    uint32_t cap = cache->cap ? cache->cap<<1 : 16;
    void *grown = (*mag_try_alloc)(cache->entries, cap*sizeof(*cache->entries), 0); /* (block, size, alignment) */
    if (!grown) { mag_dylib_close(lib); return mag_set_error(err, MAG_ERR_OOM, "fusion: cannot grow kernel cache."); }
    cache->entries = grown;
    cache->cap = cap;
  }
  cache->entries[cache->len++] = (mag_fuse_cache_entry_t){.key = key, .lib = lib, .fn = fn};
  ++cache->compiles;
  *out_fn = fn;
  return MAG_OK;
}

void mag_fuse_cache_stats(mag_context_t *ctx, uint64_t *out_hits, uint64_t *out_compiles) {
  struct mag_fuse_cache_t *cache = ctx->fuse_cache;
  if (out_hits) *out_hits = cache ? cache->hits : 0;
  if (out_compiles) *out_compiles = cache ? cache->compiles : 0;
}

void mag_fuse_cache_shutdown(mag_context_t *ctx) {
  struct mag_fuse_cache_t *cache = ctx->fuse_cache;
  if (!cache) return;
  for (uint32_t i=0; i < cache->len; ++i)
    if (cache->entries[i].lib) mag_dylib_close(cache->entries[i].lib);
  if (cache->entries) (*mag_alloc)(cache->entries, 0, 0);
  (*mag_alloc)(cache, 0, 0);
  ctx->fuse_cache = NULL;
}

/* ---- fused Adam ----
**
** The optimizer is the clearest fusion target in a training step: a chain of about twenty
** elementwise ops over every parameter, run under no_grad, with no control flow. Expressed eagerly
** each op is a separate pass; expressed as one plan it is a single pass with everything in
** registers. This builds that plan, compiles it once and reuses it for every parameter tensor.
*/

/* Buffer slots and immediate slots of the Adam plan. */
enum { ADAM_BUF_P = 0, ADAM_BUF_G = 1, ADAM_BUF_M = 2, ADAM_BUF_V = 3 };
/* One minus beta is passed in rather than computed in the kernel: the eager path evaluates
   (1.0 - beta) in Python double and only then rounds to float, and 1.0f - (float)beta gives a
   different last bit. Matching it exactly is what keeps fused and eager training identical. */
enum { ADAM_IMM_B1 = 0, ADAM_IMM_B2 = 1, ADAM_IMM_LR = 2, ADAM_IMM_EPS = 3,
       ADAM_IMM_C1 = 4, ADAM_IMM_C2 = 5, ADAM_IMM_OMB1 = 6, ADAM_IMM_OMB2 = 7, ADAM_IMM__NUM = 8 };

static bool mag_fuse_build_adam(mag_fuse_plan_t *plan, mag_dtype_t dtype) {
  mag_fuse_plan_init(plan, dtype);
  #define R(x) ((mag_fuse_operand_t){MAG_FUSE_REG, (uint8_t)(x)})
  #define I(x) ((mag_fuse_operand_t){MAG_FUSE_IMM, (uint8_t)(x)})
  int32_t p = mag_fuse_load(plan, ADAM_BUF_P);
  int32_t g = mag_fuse_load(plan, ADAM_BUF_G);
  int32_t m = mag_fuse_load(plan, ADAM_BUF_M);
  int32_t v = mag_fuse_load(plan, ADAM_BUF_V);
  /* Mirrors optim.Adam.step operation for operation, including evaluation order:
       m = beta1*m + (1-beta1)*grad
       v = beta2*v + (1-beta2)*grad^2
       p = p - (lr * (m/c1)) / (sqrt(v/c2) + eps)                                       */
  int32_t mn = mag_fuse_emit(plan, MAG_OP_ADD, (mag_fuse_operand_t[]){
    R(mag_fuse_emit(plan, MAG_OP_MUL, (mag_fuse_operand_t[]){R(m), I(ADAM_IMM_B1)}, 2)),
    R(mag_fuse_emit(plan, MAG_OP_MUL, (mag_fuse_operand_t[]){R(g), I(ADAM_IMM_OMB1)}, 2))}, 2);
  int32_t g2 = mag_fuse_emit(plan, MAG_OP_SQR, (mag_fuse_operand_t[]){R(g)}, 1);
  int32_t vn = mag_fuse_emit(plan, MAG_OP_ADD, (mag_fuse_operand_t[]){
    R(mag_fuse_emit(plan, MAG_OP_MUL, (mag_fuse_operand_t[]){R(v), I(ADAM_IMM_B2)}, 2)),
    R(mag_fuse_emit(plan, MAG_OP_MUL, (mag_fuse_operand_t[]){R(g2), I(ADAM_IMM_OMB2)}, 2))}, 2);
  int32_t mh = mag_fuse_emit(plan, MAG_OP_DIV, (mag_fuse_operand_t[]){R(mn), I(ADAM_IMM_C1)}, 2);
  int32_t vh = mag_fuse_emit(plan, MAG_OP_DIV, (mag_fuse_operand_t[]){R(vn), I(ADAM_IMM_C2)}, 2);
  int32_t den = mag_fuse_emit(plan, MAG_OP_ADD, (mag_fuse_operand_t[]){
    R(mag_fuse_emit(plan, MAG_OP_SQRT, (mag_fuse_operand_t[]){R(vh)}, 1)), I(ADAM_IMM_EPS)}, 2);
  int32_t upd = mag_fuse_emit(plan, MAG_OP_DIV, (mag_fuse_operand_t[]){
    R(mag_fuse_emit(plan, MAG_OP_MUL, (mag_fuse_operand_t[]){I(ADAM_IMM_LR), R(mh)}, 2)), R(den)}, 2);
  int32_t pn = mag_fuse_emit(plan, MAG_OP_SUB, (mag_fuse_operand_t[]){R(p), R(upd)}, 2);
  #undef R
  #undef I
  if (pn < 0) return false; /* any emit failure poisons the chain */
  return mag_fuse_store(plan, ADAM_BUF_P, pn)
      && mag_fuse_store(plan, ADAM_BUF_M, mn)
      && mag_fuse_store(plan, ADAM_BUF_V, vn);
}

bool mag_fused_adam_supported(const mag_tensor_t *p, const mag_tensor_t *g, const mag_tensor_t *m, const mag_tensor_t *v) {
  if (!p || !mag_fuse_elem_ctype(p->meta.dtype)) return false;
  const mag_tensor_t *all[4] = {p, g, m, v};
  for (int i=0; i < 4; ++i) {
    if (!all[i]) return false;
    if (all[i]->meta.dtype != p->meta.dtype) return false;
    if (all[i]->meta.device->id.type != MAG_BACKEND_TYPE_CPU) return false;
    if (!mag_tensor_is_contiguous(all[i])) return false;
    if (all[i]->meta.numel != p->meta.numel) return false;
    if (all[i]->meta.storage_offset != 0) return false;
  }
  return true;
}

mag_status_t mag_fused_adam_step(
  mag_error_t *err,
  mag_tensor_t *p,
  mag_tensor_t *g,
  mag_tensor_t *m,
  mag_tensor_t *v,
  double lr,
  double beta1,
  double beta2,
  double eps,
  double bias_correction1,
  double bias_correction2
) {
  if (!mag_fused_adam_supported(p, g, m, v))
    return mag_set_error(err, MAG_ERR_OP, "fused_adam: operands must be contiguous float32 CPU tensors of equal length.");
  mag_fuse_plan_t plan;
  if (!mag_fuse_build_adam(&plan, p->meta.dtype))
    return mag_set_error(err, MAG_ERR_OP, "fused_adam: could not build the fusion plan.");
  mag_fused_fn_t fn = NULL;
  mag_status_t st = mag_fuse_compile(err, p->ctx, &plan, &fn);
  if (mag_iserr(st)) return st; /* caller falls back to the eager path */
  double imms[ADAM_IMM__NUM];
  imms[ADAM_IMM_B1] = beta1;
  imms[ADAM_IMM_B2] = beta2;
  imms[ADAM_IMM_LR] = lr;
  imms[ADAM_IMM_EPS] = eps;
  imms[ADAM_IMM_C1] = bias_correction1;
  imms[ADAM_IMM_C2] = bias_correction2;
  imms[ADAM_IMM_OMB1] = 1.0 - beta1;  /* in double, exactly as the eager path does */
  imms[ADAM_IMM_OMB2] = 1.0 - beta2;
  void *bufs[4];
  bufs[ADAM_BUF_P] = (void *)p->storage->base;
  bufs[ADAM_BUF_G] = (void *)g->storage->base;
  bufs[ADAM_BUF_M] = (void *)m->storage->base;
  bufs[ADAM_BUF_V] = (void *)v->storage->base;
  (*fn)(bufs, imms, 0, p->meta.numel);
  return MAG_OK;
}

/* ---- automatic capture ---- */

#define MAG_FUSE_TAPE_MAX 48 /* Chain length before a flush is forced. */

typedef struct mag_fuse_tape_node_t {
  uint8_t op;
  uint8_t num_in;
  mag_tensor_t *in[3];
  mag_tensor_t *out;
} mag_fuse_tape_node_t;

struct mag_fuse_tape_t {
  mag_fuse_tape_node_t nodes[MAG_FUSE_TAPE_MAX];
  uint32_t len;
  uint32_t depth;      /* Region nesting; only the outermost exit flushes. */
  bool flushing;       /* Guards the data-pointer hook against re-entering during a flush. */
  uint64_t chains;
  uint64_t ops_fused;
};

static struct mag_fuse_tape_t *mag_fuse_tape(mag_context_t *ctx) {
  if (!ctx->fuse_tape) {
    ctx->fuse_tape = (*mag_try_alloc)(NULL, sizeof(*ctx->fuse_tape), 0);
    if (ctx->fuse_tape) memset(ctx->fuse_tape, 0, sizeof(*ctx->fuse_tape));
  }
  return ctx->fuse_tape;
}

void mag_fuse_region_begin(mag_context_t *ctx) {
  struct mag_fuse_tape_t *tape = mag_fuse_tape(ctx);
  if (!tape) return; /* Out of memory: stay eager rather than fail the user's call. */
  ++tape->depth;
  ctx->flags |= MAG_CTX_FLAG_FUSING;
}

mag_status_t mag_fuse_region_end(mag_error_t *err, mag_context_t *ctx) {
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (!tape || !tape->depth) return MAG_OK;
  if (--tape->depth) return MAG_OK;
  ctx->flags &= ~MAG_CTX_FLAG_FUSING;
  return mag_fuse_flush(err, ctx);
}

bool mag_fuse_region_active(const mag_context_t *ctx) {
  return (ctx->flags & MAG_CTX_FLAG_FUSING) != 0;
}

void mag_fuse_region_stats(mag_context_t *ctx, uint64_t *out_chains, uint64_t *out_ops_fused) {
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (out_chains) *out_chains = tape ? tape->chains : 0;
  if (out_ops_fused) *out_ops_fused = tape ? tape->ops_fused : 0;
}

/*
** A value needs storing only if something outside the chain can still read it.
**
** The tape holds references of its own - one because the tensor is a node's output, and one more
** for every later node that consumes it - so comparing the refcount against 1 would classify every
** intermediate as escaping and store the whole chain, which is exactly the traffic fusion exists to
** remove. Subtract what the tape itself holds and test the remainder.
*/
static bool mag_fuse_value_escapes(const struct mag_fuse_tape_t *tape, const mag_tensor_t *t) {
  int32_t held_by_tape = 0;
  for (uint32_t i=0; i < tape->len; ++i) {
    if (tape->nodes[i].out == t) ++held_by_tape;
    for (uint32_t j=0; j < tape->nodes[i].num_in; ++j)
      if (tape->nodes[i].in[j] == t) ++held_by_tape;
  }
  const mag_rc_control_block_t *rc = (const mag_rc_control_block_t *)t;
  int32_t total = mag_atomic32_load((mag_atomic32_t *)&rc->rc_strong, MAG_MO_RELAXED);
  return total - held_by_tape > 0;
}

static bool mag_fuse_operand_ok(const mag_tensor_t *t, const mag_tensor_t *out) {
  if (t->meta.dtype != out->meta.dtype) return false;
  if (mag_tensor_device_id(t).type != MAG_BACKEND_TYPE_CPU) return false;
  if (!mag_tensor_is_contiguous(t) || t->meta.storage_offset) return false;
  return t->meta.numel == out->meta.numel || t->meta.numel == 1; /* elementwise, or a broadcast scalar */
}

mag_status_t mag_fuse_capture(
  mag_error_t *err,
  mag_context_t *ctx,
  mag_opcode_t op,
  bool inplace,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out,
  bool *captured
) {
  *captured = false;
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (!tape || tape->flushing) return MAG_OK;
  bool ok = !inplace && num_out == 1 && num_in >= 1 && num_in <= 3
         && mag_fuse_op_is_pointwise(op)
         && mag_fuse_elem_ctype(out[0]->meta.dtype)
         && mag_fuse_operand_ok(out[0], out[0])
         && out[0]->meta.numel > 1;             /* a chain over one element is not worth a kernel */
  for (uint32_t i=0; ok && i < num_in; ++i) ok = mag_fuse_operand_ok(in[i], out[0]);
  if (ok && tape->len) /* every node in a chain walks the same number of elements */
    ok = tape->nodes[0].out->meta.numel == out[0]->meta.numel
      && tape->nodes[0].out->meta.dtype == out[0]->meta.dtype;
  if (!ok) {
    /* Only flush when this op actually touches the chain. Creating a scalar operand, for instance,
       dispatches a FILL that has nothing to do with the pending values and must not break it. */
    bool touches = false;
    for (uint32_t i=0; i < num_in && !touches; ++i) touches = (in[i]->meta.flags & MAG_TFLAG_PENDING) != 0;
    for (uint32_t i=0; i < num_out && !touches; ++i) touches = (out[i]->meta.flags & MAG_TFLAG_PENDING) != 0;
    return touches ? mag_fuse_flush(err, ctx) : MAG_OK;
  }
  if (tape->len == MAG_FUSE_TAPE_MAX) {
    mag_status_t st = mag_fuse_flush(err, ctx);
    if (mag_iserr(st)) return st;
  }
  mag_fuse_tape_node_t *n = tape->nodes + tape->len++;
  n->op = (uint8_t)op;
  n->num_in = (uint8_t)num_in;
  n->out = out[0];
  mag_rc_incref(out[0]);
  for (uint32_t i=0; i < num_in; ++i) {
    n->in[i] = in[i];
    mag_rc_incref(in[i]);
  }
  out[0]->meta.flags |= MAG_TFLAG_PENDING;
  *captured = true;
  return MAG_OK;
}

static void mag_fuse_tape_release(struct mag_fuse_tape_t *tape) {
  for (uint32_t i=0; i < tape->len; ++i) {
    mag_fuse_tape_node_t *n = tape->nodes + i;
    n->out->meta.flags &= ~MAG_TFLAG_PENDING;
    mag_rc_decref(n->out);
    for (uint32_t j=0; j < n->num_in; ++j) mag_rc_decref(n->in[j]);
  }
  tape->len = 0;
}

/* Fallback when the chain cannot be compiled: submit each recorded node in order. The results are
   identical, the fusion is simply not applied. */
static mag_status_t mag_fuse_replay_eager(mag_error_t *err, struct mag_fuse_tape_t *tape) {
  for (uint32_t i=0; i < tape->len; ++i) {
    mag_fuse_tape_node_t *n = tape->nodes + i;
    n->out->meta.flags &= ~MAG_TFLAG_PENDING; /* the kernel is about to read and write it */
    mag_device_t *dvc = n->out->meta.device;
    mag_command_t cmd = {.op = (mag_opcode_t)n->op, .in = n->in, .num_in = n->num_in, .out = &n->out, .num_out = 1, .params = NULL};
    mag_status_t st = (*dvc->submit)(err, dvc, &cmd);
    if (mag_iserr(st)) return st;
  }
  return MAG_OK;
}

mag_status_t mag_fuse_flush(mag_error_t *err, mag_context_t *ctx) {
  struct mag_fuse_tape_t *tape = ctx->fuse_tape;
  if (!tape || !tape->len || tape->flushing) return MAG_OK;
  tape->flushing = true;
  mag_status_t status = MAG_OK;
  mag_tensor_t *bufs[MAG_FUSE_MAX_BUF] = {0};
  uint8_t num_bufs = 0;
  int32_t regs[MAG_FUSE_TAPE_MAX];       /* register holding each node's result */
  mag_fuse_plan_t plan;
  mag_fuse_plan_init(&plan, tape->nodes[0].out->meta.dtype);

  /* Bind a tensor to a buffer slot, reusing the slot if it is already bound. */
  #define mag_bind_buf(T, OUT_SLOT) do { \
    int mag__found = -1; /* distinct name: OUT_SLOT is usually called 'slot' at the call site */ \
    for (uint8_t mag__b=0; mag__b < num_bufs; ++mag__b) if (bufs[mag__b] == (T)) { mag__found = mag__b; break; } \
    if (mag__found < 0) { \
      if (num_bufs == MAG_FUSE_MAX_BUF) { status = MAG_ERR_OP; goto fallback; } \
      mag__found = num_bufs; bufs[num_bufs++] = (T); \
    } \
    (OUT_SLOT) = (uint8_t)mag__found; \
  } while (0)

  for (uint32_t i=0; i < tape->len; ++i) {
    mag_fuse_tape_node_t *n = tape->nodes + i;
    mag_fuse_operand_t ops[3];
    for (uint32_t j=0; j < n->num_in; ++j) {
      int32_t produced = -1; /* was this operand produced earlier in the chain? */
      for (uint32_t k=0; k < i; ++k) if (tape->nodes[k].out == n->in[j]) produced = regs[k];
      if (produced >= 0) {
        ops[j] = (mag_fuse_operand_t){MAG_FUSE_REG, (uint8_t)produced};
        continue;
      }
      uint8_t slot;
      mag_bind_buf(n->in[j], slot);
      int32_t r = n->in[j]->meta.numel == 1 ? mag_fuse_load_scalar(&plan, slot) : mag_fuse_load(&plan, slot);
      if (r < 0) { status = MAG_ERR_OP; goto fallback; }
      ops[j] = (mag_fuse_operand_t){MAG_FUSE_REG, (uint8_t)r};
    }
    regs[i] = mag_fuse_emit(&plan, (mag_opcode_t)n->op, ops, n->num_in);
    if (regs[i] < 0) { status = MAG_ERR_OP; goto fallback; }
  }
  for (uint32_t i=0; i < tape->len; ++i) { /* store only the values something outside can still read */
    if (!mag_fuse_value_escapes(tape, tape->nodes[i].out)) continue;
    uint8_t slot;
    mag_bind_buf(tape->nodes[i].out, slot);
    if (!mag_fuse_store(&plan, slot, regs[i])) { status = MAG_ERR_OP; goto fallback; }
  }
  #undef mag_bind_buf
  if (!plan.num_stores) { status = MAG_OK; goto done; } /* the whole chain was dead */
  {
    mag_fused_fn_t fn = NULL;
    mag_error_t local = {0};
    if (mag_iserr(mag_fuse_compile(&local, ctx, &plan, &fn))) goto fallback;
    void *ptrs[MAG_FUSE_MAX_BUF];
    for (uint8_t b=0; b < num_bufs; ++b) ptrs[b] = (void *)bufs[b]->storage->base;
    (*fn)(ptrs, NULL, 0, tape->nodes[0].out->meta.numel);
    ++tape->chains;
    tape->ops_fused += tape->len;
    goto done;
  }
fallback:
  status = mag_fuse_replay_eager(err, tape);
done:
  mag_fuse_tape_release(tape);
  tape->flushing = false;
  return status;
}

void mag_fuse_tape_shutdown(mag_context_t *ctx) {
  if (!ctx->fuse_tape) return;
  mag_fuse_tape_release(ctx->fuse_tape);
  (*mag_alloc)(ctx->fuse_tape, 0, 0);
  ctx->fuse_tape = NULL;
}
