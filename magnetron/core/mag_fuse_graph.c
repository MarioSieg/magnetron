#include "mag_fuse_graph.h"
#include "mag_hash.h"

#include <string.h>

bool mag_fuse_op_is_fusible(mag_opcode_t op) {
  if (op >= MAG_OP__NUM) return false;
  return !!(mag_op_trait(op)->flags & MAG_OP_FLAG_FUSIBLE);
}

void mag_fuse_graph_init(mag_fuse_graph_t *g, mag_dtype_t dtype) {
  memset(g, 0, sizeof(*g)); /* Zeroes the padding too, so the structure hash reads no stale bytes. */
  g->dtype = (uint8_t)dtype;
}

/* Every instruction writes the register named by its own index, which is what makes the form SSA. */
static mag_fuse_ins_t *mag_fuse_graph_push(mag_fuse_graph_t *g, uint8_t op, uint8_t num_in) {
  if (g->num_ins >= MAG_FUSE_MAX_INS || g->num_ins >= MAG_FUSE_MAX_REG) return NULL;
  mag_fuse_ins_t *ins = g->ins + g->num_ins;
  memset(ins, 0, sizeof(*ins));
  ins->op = op;
  ins->dst = (uint8_t)g->num_ins;
  ins->num_in = num_in;
  return ins;
}

static int32_t mag_fuse_graph_load_any(mag_fuse_graph_t *g, uint8_t buf, mag_fuse_operand_kind_t kind) {
  if (buf >= MAG_FUSE_MAX_BUF) return -1;
  mag_fuse_ins_t *ins = mag_fuse_graph_push(g, MAG_FUSE_OP_LOAD, 1);
  if (!ins) return -1;
  ins->in[0] = (mag_fuse_operand_t){.kind = (uint8_t)kind, .idx = buf};
  if (buf >= g->num_bufs) g->num_bufs = buf+1;
  return (int32_t)g->num_ins++;
}

int32_t mag_fuse_graph_load(mag_fuse_graph_t *g, uint8_t buf) {
  return mag_fuse_graph_load_any(g, buf, MAG_FUSE_BUF);
}

int32_t mag_fuse_graph_load_scalar(mag_fuse_graph_t *g, uint8_t buf) {
  return mag_fuse_graph_load_any(g, buf, MAG_FUSE_SCL);
}

int32_t mag_fuse_graph_imm(mag_fuse_graph_t *g) {
  if (g->num_imms >= MAG_FUSE_MAX_IMM) return -1;
  return (int32_t)g->num_imms++;
}

static const mag_fuse_ins_t ins_operand_probe; /* Only to size the operand array below. */

int32_t mag_fuse_graph_emit(mag_fuse_graph_t *g, mag_opcode_t op, const mag_fuse_operand_t *in, uint8_t num_in) {
  if (!mag_fuse_op_is_fusible(op)) return -1;
  /* Arity comes from the operator table rather than from anything fusion-specific, so a chain can
     never disagree with the operator it is standing in for. */
  if (mag_op_trait(op)->in != num_in) return -1;
  if (num_in > sizeof(ins_operand_probe.in)/sizeof(*ins_operand_probe.in)) return -1;
  for (uint8_t i=0; i < num_in; ++i) {
    switch (in[i].kind) {
      case MAG_FUSE_REG: if ((uint32_t)in[i].idx >= g->num_ins) return -1; break; /* Forward reference. */
      case MAG_FUSE_BUF:
      case MAG_FUSE_SCL: if (in[i].idx >= MAG_FUSE_MAX_BUF) return -1; break;
      case MAG_FUSE_IMM: if (in[i].idx >= MAG_FUSE_MAX_IMM) return -1; break;
      default: return -1;
    }
  }
  mag_fuse_ins_t *ins = mag_fuse_graph_push(g, (uint8_t)op, num_in);
  if (!ins) return -1;
  memcpy(ins->in, in, num_in*sizeof(*in));
  for (uint8_t i=0; i < num_in; ++i) { /* Widen the operand counts only once the push has succeeded. */
    if (in[i].kind == MAG_FUSE_BUF || in[i].kind == MAG_FUSE_SCL) {
      if (in[i].idx >= g->num_bufs) g->num_bufs = in[i].idx+1;
    } else if (in[i].kind == MAG_FUSE_IMM) {
      if (in[i].idx >= g->num_imms) g->num_imms = in[i].idx+1;
    }
  }
  return (int32_t)g->num_ins++;
}

bool mag_fuse_graph_store(mag_fuse_graph_t *g, uint8_t buf, int32_t reg) {
  if (reg < 0 || (uint32_t)reg >= g->num_ins) return false;
  if (buf >= MAG_FUSE_MAX_BUF || g->num_stores >= MAG_FUSE_MAX_BUF) return false;
  g->stores[g->num_stores++] = (mag_fuse_store_t){.buf = buf, .reg = (uint8_t)reg};
  if (buf >= g->num_bufs) g->num_bufs = buf+1;
  return true;
}

uint32_t mag_fuse_graph_prune(mag_fuse_graph_t *g) {
  if (!g->num_ins) return 0;
  bool live[MAG_FUSE_MAX_INS];
  memset(live, 0, sizeof(live));
  for (uint8_t s=0; s < g->num_stores; ++s) live[g->stores[s].reg] = true;
  /* One backward sweep suffices: an operand can only name a lower-numbered register, so by the time
     instruction i is examined every consumer of it has already been seen. */
  for (int32_t i=(int32_t)g->num_ins-1; i >= 0; --i) {
    if (!live[i]) continue;
    const mag_fuse_ins_t *ins = g->ins+i;
    for (uint8_t k=0; k < ins->num_in; ++k)
      if (ins->in[k].kind == MAG_FUSE_REG) live[ins->in[k].idx] = true;
  }
  uint8_t remap[MAG_FUSE_MAX_INS];
  memset(remap, 0, sizeof(remap));
  uint32_t kept = 0;
  for (uint32_t i=0; i < g->num_ins; ++i) {
    if (!live[i]) continue;
    mag_fuse_ins_t *ins = g->ins+kept;
    *ins = g->ins[i];
    for (uint8_t k=0; k < ins->num_in; ++k)
      if (ins->in[k].kind == MAG_FUSE_REG) ins->in[k].idx = remap[ins->in[k].idx];
    ins->dst = (uint8_t)kept;
    remap[i] = (uint8_t)kept;
    ++kept;
  }
  for (uint8_t s=0; s < g->num_stores; ++s) g->stores[s].reg = remap[g->stores[s].reg];
  /* Instructions beyond the compacted end are zeroed so the structure hash cannot see what was
     pruned: two graphs that prune to the same program must hash the same. */
  uint32_t removed = g->num_ins-kept;
  if (removed) memset(g->ins+kept, 0, removed*sizeof(*g->ins));
  g->num_ins = kept;
  /* num_bufs and num_imms are left alone on purpose. They index the caller's operand arrays, and
     renumbering them here would silently invalidate the mapping the caller still holds. */
  return removed;
}

uint64_t mag_fuse_graph_hash(const mag_fuse_graph_t *g) {
  /* Only the bytes a lowering reads: the header, the live instructions and the stores. Trailing
     instruction slots are excluded so a short chain does not hash the whole fixed-size array. */
  struct {
    uint8_t dtype, num_bufs, num_stores, num_imms;
    uint32_t num_ins;
  } head = {g->dtype, g->num_bufs, g->num_stores, g->num_imms, g->num_ins};
  uint64_t h = mag_murmur3_128_reduced_64(&head, sizeof(head), 0x9e3779b9u);
  h ^= mag_murmur3_128_reduced_64(g->ins, g->num_ins*sizeof(*g->ins), (uint32_t)h);
  h ^= mag_murmur3_128_reduced_64(g->stores, g->num_stores*sizeof(*g->stores), (uint32_t)h);
  return h;
}
