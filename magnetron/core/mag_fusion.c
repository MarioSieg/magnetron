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
** C expression for each fusible opcode. $0..$2 are the operands, already materialized into named
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
    [MAG_OP_POW]          = {"powf($0, $1)", 2},
    [MAG_OP_MIN]          = {"fminf($0, $1)", 2},
    [MAG_OP_MAX]          = {"fmaxf($0, $1)", 2},
    [MAG_OP_NEG]          = {"(-$0)", 1},
    [MAG_OP_ABS]          = {"fabsf($0)", 1},
    [MAG_OP_SGN]          = {"(($0 > 0.0f) - ($0 < 0.0f))", 1},
    [MAG_OP_SQR]          = {"($0 * $0)", 1},
    [MAG_OP_SQRT]         = {"sqrtf($0)", 1},
    [MAG_OP_RSQRT]        = {"(1.0f / sqrtf($0))", 1},
    [MAG_OP_RCP]          = {"(1.0f / $0)", 1},
    [MAG_OP_EXP]          = {"expf($0)", 1},
    [MAG_OP_EXP2]         = {"exp2f($0)", 1},
    [MAG_OP_EXPM1]        = {"expm1f($0)", 1},
    [MAG_OP_LOG]          = {"logf($0)", 1},
    [MAG_OP_LOG2]         = {"log2f($0)", 1},
    [MAG_OP_LOG10]        = {"log10f($0)", 1},
    [MAG_OP_LOG1P]        = {"log1pf($0)", 1},
    [MAG_OP_SIN]          = {"sinf($0)", 1},
    [MAG_OP_COS]          = {"cosf($0)", 1},
    [MAG_OP_TAN]          = {"tanf($0)", 1},
    [MAG_OP_TANH]         = {"tanhf($0)", 1},
    [MAG_OP_SINH]         = {"sinhf($0)", 1},
    [MAG_OP_COSH]         = {"coshf($0)", 1},
    [MAG_OP_ERF]          = {"erff($0)", 1},
    [MAG_OP_ERFC]         = {"erfcf($0)", 1},
    [MAG_OP_FLOOR]        = {"floorf($0)", 1},
    [MAG_OP_CEIL]         = {"ceilf($0)", 1},
    [MAG_OP_ROUND]        = {"roundf($0)", 1},
    [MAG_OP_TRUNC]        = {"truncf($0)", 1},
    [MAG_OP_STEP]         = {"($0 > 0.0f ? 1.0f : 0.0f)", 1},
    [MAG_OP_RELU]         = {"fmaxf($0, 0.0f)", 1},
    [MAG_OP_SIGMOID]      = {"(1.0f / (1.0f + expf(-$0)))", 1},
    [MAG_OP_HARD_SIGMOID] = {"fminf(1.0f, fmaxf(0.0f, $0 * (1.0f/6.0f) + 0.5f))", 1},
    [MAG_OP_SILU]         = {"($0 / (1.0f + expf(-$0)))", 1},
    [MAG_OP_GELU]         = {"(0.5f * $0 * (1.0f + erff($0 * 0.70710678118654752f)))", 1},
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
** Keyed on the generated source, not on the plan structure. Structure alone is not enough: any
** change to codegen (a new rounding rule, a different expression for an op) produces different
** code from an identical plan, and a structure key would silently serve the stale object still
** sitting in the on-disk cache. Generating the source first costs a few microseconds of string
** building, against a compile that costs tens of milliseconds.
*/
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
  mag_sstream_t src;
  mag_sstream_init(&src);
  mag_fuse_codegen_ss(plan, &src);
  if (src.oom) { mag_sstream_free(&src); return mag_set_error(err, MAG_ERR_OOM, "fusion: cannot build kernel source."); }
  uint64_t key = mag_fuse_source_hash(src.buf);
  for (uint32_t i=0; i < cache->len; ++i) {
    if (cache->entries[i].key == key) {
      ++cache->hits;
      *out_fn = cache->entries[i].fn;
      mag_sstream_free(&src);
      return MAG_OK;
    }
  }
  mag_dylib_t *lib = NULL;
  mag_fused_fn_t fn = NULL;
  mag_status_t st = mag_fuse_build(err, key, &src, &lib, &fn);
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
