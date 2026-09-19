/*
** Running a fused chain on the CPU.
**
** Core hands this backend a graph and asks for the whole chain in one pass. What that pass is made
** of is entirely this backend's business: core has no opinion, and a different backend answering
** the same question would produce something else altogether.
**
** The answer here walks the instructions over a tile of elements at a time, keeping every
** intermediate in a scratch block small enough to sit in L1. That is what the chain buys. Eager
** execution writes each intermediate to memory and reads it back once per operator; a tile of a few
** hundred elements never leaves cache between one instruction and the next.
**
** It is an interpreter, so it pays a dispatch per instruction per tile. A chain lowered to compiled
** code would not, and that is worth doing - but an interpreter needs no toolchain at runtime, works
** on every platform this library builds for, and is the thing a compiled path has to be measured
** against before anyone can say the compiler earned its place.
**
** This lives inside the dispatch translation unit so it is compiled once per instruction set like
** every other kernel, which is what lets the tile loops widen to whatever the host supports.
*/

/*
** Elements held in scratch between one instruction and the next.
**
** Sized so a chain with a realistic number of live values stays inside L1: at 256 elements a
** twenty-instruction chain needs 20 KiB. Larger tiles amortize the per-instruction dispatch better
** but start spilling to L2, which is the traffic the chain exists to avoid.
*/
#define MAG_CPU_FUSE_TILE 256

/*
** One instruction over one tile.
**
** A switch over a whole tile rather than a switch per element, so dispatch is paid once per tile
** and the inner loop is left as a plain expression the compiler can vectorize. Every arm performs
** exactly the arithmetic the eager kernel performs, in the same order: a chain that rounds
** differently from the operators it replaces is not a chain anyone can safely turn on.
*/
static void MAG_HOTPROC mag_cpu_fuse_step_f32(
  uint8_t op,
  float *restrict dst,
  const float *restrict a,
  const float *restrict b,
  const float *restrict c,
  int64_t n
) {
  switch (op) {
    case MAG_OP_ADD:   for (int64_t i=0; i < n; ++i) dst[i] = a[i]+b[i]; break;
    case MAG_OP_SUB:   for (int64_t i=0; i < n; ++i) dst[i] = a[i]-b[i]; break;
    case MAG_OP_MUL:   for (int64_t i=0; i < n; ++i) dst[i] = a[i]*b[i]; break;
    case MAG_OP_DIV:   for (int64_t i=0; i < n; ++i) dst[i] = a[i]/b[i]; break;
    case MAG_OP_MIN:   for (int64_t i=0; i < n; ++i) dst[i] = fminf(a[i], b[i]); break;
    case MAG_OP_MAX:   for (int64_t i=0; i < n; ++i) dst[i] = fmaxf(a[i], b[i]); break;
    case MAG_OP_NEG:   for (int64_t i=0; i < n; ++i) dst[i] = -a[i]; break;
    case MAG_OP_ABS:   for (int64_t i=0; i < n; ++i) dst[i] = fabsf(a[i]); break;
    case MAG_OP_SGN:   for (int64_t i=0; i < n; ++i) dst[i] = (float)((a[i] > 0.0f)-(a[i] < 0.0f)); break;
    case MAG_OP_SQR:   for (int64_t i=0; i < n; ++i) dst[i] = a[i]*a[i]; break;
    case MAG_OP_SQRT:  for (int64_t i=0; i < n; ++i) dst[i] = sqrtf(a[i]); break;
    case MAG_OP_FLOOR: for (int64_t i=0; i < n; ++i) dst[i] = floorf(a[i]); break;
    case MAG_OP_CEIL:  for (int64_t i=0; i < n; ++i) dst[i] = ceilf(a[i]); break;
    case MAG_OP_ROUND: for (int64_t i=0; i < n; ++i) dst[i] = roundf(a[i]); break;
    case MAG_OP_TRUNC: for (int64_t i=0; i < n; ++i) dst[i] = truncf(a[i]); break;
    case MAG_OP_STEP:  for (int64_t i=0; i < n; ++i) dst[i] = a[i] > 0.0f ? 1.0f : 0.0f; break;
    case MAG_OP_RELU:  for (int64_t i=0; i < n; ++i) dst[i] = fmaxf(a[i], 0.0f); break;
    case MAG_OP_CLAMP: for (int64_t i=0; i < n; ++i) dst[i] = fminf(fmaxf(a[i], b[i]), c[i]); break;
    default: mag_panic("cpu: operator %u reached the fused interpreter without a case.", op);
  }
}

/* Whether this backend can run the chain. Anything it declines, core replays one operator at a time. */
static bool mag_cpu_fuse_supported(const mag_fuse_graph_t *g) {
  /* float32 only so far. The narrow floats compute in float and round back to storage after every
     operator, so a chain over them has to round at exactly the same points as the eager kernels or
     the numbers drift. That is worth doing and is not done here yet. */
  if (g->dtype != MAG_DTYPE_FLOAT32) return false;
  if (!g->num_ins || !g->num_stores) return false;
  for (uint32_t i=0; i < g->num_ins; ++i)
    if (g->ins[i].op != MAG_FUSE_OP_LOAD && !mag_fuse_op_is_fusible((mag_opcode_t)g->ins[i].op))
      return false;
  return true;
}

static mag_status_t MAG_HOTPROC mag_cpu_kernel_fused_f32(mag_error_t *err, const mag_kernel_payload_t *payload) {
  const mag_command_t *cmd = payload->cmd;
  if (mag_unlikely(!cmd->params))
    return mag_set_error(err, MAG_ERR_KERNEL, "cpu: fused command carries no graph.");
  const mag_fuse_graph_t *g = cmd->params->fused.graph;
  if (mag_unlikely(!g || !mag_cpu_fuse_supported(g)))
    return mag_set_error(err, MAG_ERR_KERNEL, "cpu: no lowering for this fused chain.");

  if (mag_unlikely(cmd->num_in > MAG_FUSE_MAX_BUF))
    return mag_set_error(err, MAG_ERR_KERNEL, "cpu: fused chain binds %u buffers, more than the graph allows.", cmd->num_in);

  /*
  ** The loop extent. Every operand is either the full chain length or a single broadcast element,
  ** so the longest one is the length. Shape never enters the graph; this is the only extent.
  **
  ** Only the buffers the chain writes are asked for write access. An operand can be a tensor
  ** borrowing memory it is not allowed to write - mag_tensor_borrow_cpu_buffer makes those - and
  ** asking for a mutable pointer to one of those aborts rather than fails.
  */
  bool written[MAG_FUSE_MAX_BUF];
  memset(written, 0, sizeof(written));
  for (uint8_t st=0; st < g->num_stores; ++st) written[g->stores[st].buf] = true;

  int64_t total = 0;
  void *bufs[MAG_FUSE_MAX_BUF];
  for (uint32_t b=0; b < cmd->num_in; ++b) {
    bufs[b] = (void *)(written[b] ? mag_tensor_data_ptr_mut(cmd->in[b]) : mag_tensor_data_ptr(cmd->in[b]));
    if (cmd->in[b]->meta.numel > total) total = cmd->in[b]->meta.numel;
  }

  int64_t tc = payload->thread_num;
  int64_t ti = payload->thread_idx;
  int64_t chunk = (total+tc-1)/tc;
  int64_t begin = ti*chunk;
  int64_t end = mag_xmin(begin+chunk, total);
  if (mag_unlikely(begin >= end)) return MAG_OK;

  /*
  ** Scratch holds intermediates and nothing else.
  **
  ** A load does not copy: the register simply points at the operand's own memory for this tile.
  ** A value that gets written back is computed straight into its destination buffer. What is left
  ** in scratch is exactly the values that exist only inside the chain - which is the traffic fusion
  ** is supposed to remove, and copying operands in and results out would have added more of it than
  ** eager execution moves in the first place.
  **
  ** Broadcast scalars are the one thing that must be materialized, and they are constant for the
  ** whole run, so they are splatted once rather than per tile.
  */
  size_t mark = mag_scratch_arena_mark(&mag_tls_arena);
  size_t num_tiles = g->num_ins + g->num_imms;
  float *scratch = mag_scratch_arena_alloc(&mag_tls_arena, num_tiles*MAG_CPU_FUSE_TILE*sizeof(float));
  if (mag_unlikely(!scratch))
    return mag_set_error(err, MAG_ERR_OOM, "cpu: no scratch for a fused chain of %u instructions.", g->num_ins);
  float *imm_tiles = scratch + (size_t)g->num_ins*MAG_CPU_FUSE_TILE;

  const double *imms = NULL; /* Immediates arrive at call time; no chain uses them yet. */
  for (uint8_t k=0; k < g->num_imms; ++k) {
    float v = imms ? (float)imms[k] : 0.0f;
    float *t = imm_tiles + (size_t)k*MAG_CPU_FUSE_TILE;
    for (int64_t i=0; i < MAG_CPU_FUSE_TILE; ++i) t[i] = v;
  }

  /*
  ** Where each value lives, decided once for the whole run.
  **
  ** reg_buf names the buffer a register is an alias of, which is what makes writing a result
  ** straight into its destination safe to check: a result may only go there when no operand of the
  ** same instruction reads that buffer, or the write would clobber a value still being read.
  */
  int32_t reg_buf[MAG_FUSE_MAX_INS];
  int32_t dst_buf[MAG_FUSE_MAX_INS];
  for (uint32_t i=0; i < g->num_ins; ++i) { reg_buf[i] = -1; dst_buf[i] = -1; }
  for (uint32_t i=0; i < g->num_ins; ++i) {
    const mag_fuse_ins_t *ins = g->ins+i;
    if (ins->op == MAG_FUSE_OP_LOAD) {
      if (ins->in[0].kind == MAG_FUSE_BUF) reg_buf[i] = ins->in[0].idx; /* Aliases the operand. */
      continue;
    }
    for (uint8_t sidx=0; sidx < g->num_stores; ++sidx) {
      if (g->stores[sidx].reg != i) continue;
      int32_t target = g->stores[sidx].buf;
      bool clobbers = false;
      for (uint8_t j=0; j < ins->num_in; ++j)
        if (ins->in[j].kind == MAG_FUSE_REG && reg_buf[ins->in[j].idx] == target) clobbers = true;
      if (!clobbers) { dst_buf[i] = target; reg_buf[i] = target; }
      break;
    }
  }

  for (int64_t base=begin; base < end; base += MAG_CPU_FUSE_TILE) {
    int64_t n = mag_xmin(end-base, MAG_CPU_FUSE_TILE);
    const float *reg[MAG_FUSE_MAX_INS];
    for (uint32_t i=0; i < g->num_ins; ++i) {
      const mag_fuse_ins_t *ins = g->ins+i;
      if (ins->op == MAG_FUSE_OP_LOAD) {
        const float *buf = bufs[ins->in[0].idx];
        if (ins->in[0].kind == MAG_FUSE_SCL) { /* One element, broadcast over the loop. */
          float *t = scratch + (size_t)i*MAG_CPU_FUSE_TILE;
          float v = buf[0];
          for (int64_t k=0; k < n; ++k) t[k] = v;
          reg[i] = t;
        } else {
          reg[i] = buf+base; /* No copy: the value is already exactly where it needs to be. */
        }
        continue;
      }
      float *dst = dst_buf[i] >= 0
        ? (float *)bufs[dst_buf[i]]+base            /* Straight into the tensor it is written to. */
        : scratch + (size_t)i*MAG_CPU_FUSE_TILE;    /* Lives and dies inside the chain. */
      const float *src[3] = {NULL, NULL, NULL};
      for (uint8_t j=0; j < ins->num_in; ++j)
        src[j] = ins->in[j].kind == MAG_FUSE_REG
          ? reg[ins->in[j].idx]
          : imm_tiles + (size_t)ins->in[j].idx*MAG_CPU_FUSE_TILE;
      mag_cpu_fuse_step_f32(ins->op, dst, src[0], src[1], src[2], n);
      reg[i] = dst;
    }
    /* Whatever could not be written in place still has to reach its tensor. */
    for (uint8_t sidx=0; sidx < g->num_stores; ++sidx) {
      const mag_fuse_store_t *st = g->stores+sidx;
      if (dst_buf[st->reg] == st->buf) continue;
      memcpy((float *)bufs[st->buf]+base, reg[st->reg], (size_t)n*sizeof(float));
    }
  }
  mag_scratch_arena_reset(&mag_tls_arena, mark);
  return MAG_OK;
}
