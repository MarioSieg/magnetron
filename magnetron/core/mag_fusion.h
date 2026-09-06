/*
** Pointwise fusion JIT.
**
** A chain of elementwise operators is described as a small SSA program, lowered to C, compiled by
** the host toolchain and loaded back as a shared object. The compiled kernel evaluates the whole
** chain per element with intermediates in registers, which is where the win comes from: eager
** execution performs one load and one store per element for every operator in the chain.
**
** Interpreting the chain instead was measured and does not work. Running each operator over a
** cache-resident tile still performs those loads and stores and recovers only about 1.2x of an
** available 6-7x, because the cost is instruction count and register pressure rather than DRAM
** traffic. Generating code is the only approach that removes them.
*/

#ifndef MAG_FUSION_H
#define MAG_FUSION_H

#include "mag_def.h"
#include "mag_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_FUSE_MAX_INS 96   /* Instructions in one fused chain. */
#define MAG_FUSE_MAX_REG 64   /* SSA values live at once. */
#define MAG_FUSE_MAX_BUF 16   /* Tensor operands, inputs and outputs together. */
#define MAG_FUSE_MAX_IMM 16   /* Runtime scalars. */

/* Where an instruction's operand comes from. */
typedef enum mag_fuse_operand_kind_t {
  MAG_FUSE_REG = 0,   /* A value produced earlier in the chain. */
  MAG_FUSE_BUF = 1,   /* Element i of a tensor operand. */
  MAG_FUSE_IMM = 2    /* A scalar passed in at call time, not baked into the code. */
} mag_fuse_operand_kind_t;

typedef struct mag_fuse_operand_t {
  uint8_t kind;       /* mag_fuse_operand_kind_t */
  uint8_t idx;        /* Register, buffer or immediate slot. */
} mag_fuse_operand_t;

typedef struct mag_fuse_ins_t {
  uint8_t op;                     /* mag_opcode_t, must be pointwise. */
  uint8_t dst;                    /* Register written. */
  uint8_t num_in;
  uint8_t _pad;
  mag_fuse_operand_t in[3];
} mag_fuse_ins_t;

typedef struct mag_fuse_store_t {
  uint8_t buf;                    /* Tensor operand written. */
  uint8_t reg;                    /* Register holding the value. */
} mag_fuse_store_t;

/*
** A fused chain. Immediates are referenced by slot rather than by value so that changing a learning
** rate reuses the compiled kernel instead of triggering a recompile; only the structure is hashed.
*/
typedef struct mag_fuse_plan_t {
  uint8_t dtype;                              /* mag_dtype_t of every operand. */
  uint8_t num_bufs;
  uint8_t num_stores;
  uint8_t num_imms;
  uint32_t num_ins;
  mag_fuse_ins_t ins[MAG_FUSE_MAX_INS];
  mag_fuse_store_t stores[MAG_FUSE_MAX_BUF];
} mag_fuse_plan_t;

/* Signature of every generated kernel. */
typedef void (*mag_fused_fn_t)(void *const *bufs, const double *imms, int64_t begin, int64_t end);

/* Build a plan incrementally. Returns the register holding each result, or a negative value on
   overflow, so a caller that outgrows the limits can fall back to eager execution. */
extern MAG_EXPORT void mag_fuse_plan_init(mag_fuse_plan_t *plan, mag_dtype_t dtype);
extern MAG_EXPORT int32_t mag_fuse_load(mag_fuse_plan_t *plan, uint8_t buf);
extern MAG_EXPORT int32_t mag_fuse_emit(mag_fuse_plan_t *plan, mag_opcode_t op, const mag_fuse_operand_t *in, uint8_t num_in);
extern MAG_EXPORT bool mag_fuse_store(mag_fuse_plan_t *plan, uint8_t buf, int32_t reg);

/* True when an opcode can appear in a fused chain, i.e. it reads and writes element i only. */
extern MAG_EXPORT bool mag_fuse_op_is_pointwise(mag_opcode_t op);

/* Emit the C source for a plan. Returns a heap string the caller releases with
   mag_fuse_source_free, so callers outside core need no string-builder type. */
extern MAG_EXPORT char *mag_fuse_codegen(const mag_fuse_plan_t *plan);
extern MAG_EXPORT void mag_fuse_source_free(char *src);

/* Compile (or fetch from cache) the kernel for a plan. The returned pointer stays valid for the
   lifetime of the context. Fails cleanly when no host compiler is available, so callers can fall
   back to eager execution rather than aborting. */
extern MAG_EXPORT mag_status_t mag_fuse_compile(mag_error_t *err, mag_context_t *ctx, const mag_fuse_plan_t *plan, mag_fused_fn_t *out_fn);

extern MAG_COLDPROC void mag_fuse_cache_shutdown(mag_context_t *ctx); /* Unload every compiled kernel. */
extern MAG_EXPORT void mag_fuse_cache_stats(mag_context_t *ctx, uint64_t *out_hits, uint64_t *out_compiles);

/* Fused Adam. The whole update runs as one pass per parameter instead of about twenty. Returns an
   error when the operands are unsupported or no compiler is available, so the caller falls back. */
extern MAG_EXPORT bool mag_fused_adam_supported(const mag_tensor_t *p, const mag_tensor_t *g, const mag_tensor_t *m, const mag_tensor_t *v);
extern MAG_EXPORT mag_status_t mag_fused_adam_step(mag_error_t *err, mag_tensor_t *p, mag_tensor_t *g, mag_tensor_t *m, mag_tensor_t *v,
  double lr, double beta1, double beta2, double eps, double bias_correction1, double bias_correction2);

#ifdef __cplusplus
}
#endif

#endif
