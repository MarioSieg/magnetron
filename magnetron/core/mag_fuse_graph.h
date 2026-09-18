/*
** The representation a fused chain is handed to a backend in.
**
** A chain of pointwise operators is one loop over one index, so the graph describes the body of that
** loop and nothing else: a straight-line SSA program over scalar values, where every value is read
** from a buffer at element i, produced by an earlier instruction, or passed in as a scalar at call
** time. Shape does not appear, because every tensor in a chain has the same element count and the
** loop bound is the only extent that matters. Neither does anything about how the program is turned
** into code, which is the backend's business.
**
** What a backend does with this is its own affair. Generating C and compiling it is one answer;
** emitting PTX or a Metal library, or walking the instructions in an interpreter, are others. Core
** never learns which was chosen.
*/

#ifndef MAG_FUSE_GRAPH_H
#define MAG_FUSE_GRAPH_H

#include "mag_def.h"
#include "mag_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
** Limits.
**
** These are sized so a graph is a plain value that can live on the stack and be hashed as bytes,
** rather than a heap structure with its own lifetime. A chain that outgrows any of them is not an
** error: the builder reports the overflow, and the caller splits the chain or falls back to eager
** execution. They are deliberately consistent with each other - every instruction can produce a
** register and every instruction can read a distinct buffer - so that only one of them can ever be
** the binding constraint in practice.
*/
#define MAG_FUSE_MAX_INS 96     /* Instructions in one chain. */
#define MAG_FUSE_MAX_REG 96     /* SSA values. One per instruction, so this can never bind first. */
#define MAG_FUSE_MAX_BUF 16     /* Distinct tensors read and written, together. */
#define MAG_FUSE_MAX_IMM 16     /* Scalars supplied at call time. */

/* Where an instruction's operand comes from. */
typedef enum mag_fuse_operand_kind_t {
  MAG_FUSE_REG = 0,   /* A value produced earlier in the chain. */
  MAG_FUSE_BUF = 1,   /* Element i of a tensor operand. */
  MAG_FUSE_IMM = 2,   /* A scalar supplied at call time, not baked into the program. */
  MAG_FUSE_SCL = 3,   /* Element 0 of a tensor operand, broadcast across the loop. */
} mag_fuse_operand_kind_t;

typedef struct mag_fuse_operand_t {
  uint8_t kind;   /* mag_fuse_operand_kind_t */
  uint8_t idx;    /* Register, buffer or immediate slot, according to kind. */
} mag_fuse_operand_t;

/*
** One instruction.
**
** MAG_FUSE_OP_LOAD is not an opcode from the operator table: it names the one instruction that has no
** operator behind it, moving an element of a buffer into a register so the rest of the program can
** speak only in registers. Everything else is a real opcode carrying MAG_OP_FLAG_FUSIBLE.
*/
#define MAG_FUSE_OP_LOAD 0xff   /* Distinct from every opcode: MAG_OP__NUM is asserted to fit in a byte. */

typedef struct mag_fuse_ins_t {
  uint8_t op;                   /* An opcode with MAG_OP_FLAG_FUSIBLE, or MAG_FUSE_OP_LOAD. */
  uint8_t dst;                  /* Register written. Always the instruction's own index. */
  uint8_t num_in;
  uint8_t _pad;
  mag_fuse_operand_t in[3];
} mag_fuse_ins_t;

/* A value written back to a tensor operand once the chain has finished computing it. */
typedef struct mag_fuse_store_t {
  uint8_t buf;    /* Tensor operand written. */
  uint8_t reg;    /* Register holding the value. */
} mag_fuse_store_t;

/*
** A fused chain.
**
** Immediates are referenced by slot rather than by value, so two chains that differ only in a
** learning rate are the same graph and a backend that caches what it lowers gets a hit rather than
** recompiling. That is why the structure hash below covers the instructions and stores but not the
** immediate values.
*/
typedef struct mag_fuse_graph_t {
  uint8_t dtype;                            /* mag_dtype_t. Every operand in the chain shares it. */
  uint8_t num_bufs;
  uint8_t num_stores;
  uint8_t num_imms;
  uint32_t num_ins;
  mag_fuse_ins_t ins[MAG_FUSE_MAX_INS];
  mag_fuse_store_t stores[MAG_FUSE_MAX_BUF];
} mag_fuse_graph_t;

/*
** Building a graph.
**
** Every call that can overflow says so in its return rather than aborting, because overflow is an
** ordinary outcome: a chain is built by watching operators go past, and the point at which one
** becomes too long is not known until it happens. A caller that gets a negative register or false
** stops extending the chain and runs what it has.
*/
extern MAG_EXPORT void mag_fuse_graph_init(mag_fuse_graph_t *g, mag_dtype_t dtype);
extern MAG_EXPORT int32_t mag_fuse_graph_load(mag_fuse_graph_t *g, uint8_t buf);          /* Element i of buf. */
extern MAG_EXPORT int32_t mag_fuse_graph_load_scalar(mag_fuse_graph_t *g, uint8_t buf);   /* Element 0 of buf, broadcast. */
extern MAG_EXPORT int32_t mag_fuse_graph_emit(mag_fuse_graph_t *g, mag_opcode_t op, const mag_fuse_operand_t *in, uint8_t num_in);
extern MAG_EXPORT bool mag_fuse_graph_store(mag_fuse_graph_t *g, uint8_t buf, int32_t reg);
extern MAG_EXPORT int32_t mag_fuse_graph_imm(mag_fuse_graph_t *g);                        /* Reserve an immediate slot. */

/* True when an opcode may appear in a chain. Reads MAG_OP_FLAG_FUSIBLE from the operator table. */
extern MAG_EXPORT bool mag_fuse_op_is_fusible(mag_opcode_t op);

/*
** Drop instructions whose results nothing stores and nothing else reads.
**
** A chain is captured as it is executed, so it accumulates values that turned out to be needed only
** by later links. Once the stores are decided - which happens at the end, when it is known which
** results anything outside the chain can still observe - whatever remains unreachable from them is
** dead. Returns the number of instructions removed.
*/
extern MAG_EXPORT uint32_t mag_fuse_graph_prune(mag_fuse_graph_t *g);

/*
** The structure hash, covering everything a lowering depends on and nothing else.
**
** Two graphs with the same hash lower to the same program, so a backend can use this as the key for
** whatever it caches. Immediate values are excluded deliberately, since they arrive at call time.
*/
extern MAG_EXPORT uint64_t mag_fuse_graph_hash(const mag_fuse_graph_t *g);

#ifdef __cplusplus
}
#endif

#endif
