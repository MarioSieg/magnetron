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

#include "mag_op_dispatch.h"
#include "mag_backend.h"
#include "mag_context.h"
#include "mag_autodiff.h"

static void MAG_COLDPROC mag_dbg_trace_op_ir(
  mag_opcode_t op,
  bool inplace,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out
);

static void mag_assert_correct_op_data(
  mag_opcode_t op,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out
) {
  mag_assert(op != MAG_OP_NOP, "op_validate: invalid opcode %d.", op);
  const mag_op_traits_t *meta = mag_op_trait(op);
  if (meta->in) mag_assert(in != NULL, "op_validate: input tensors for operator '%s' are NULL.", meta->mnemonic);
  if (meta->out) mag_assert(out != NULL, "op_validate: output tensors for operator '%s' are NULL.", meta->mnemonic);
  if (meta->in != MAG_OP_INOUT_DYN) {
    mag_assert(meta->in == num_in, "op_validate: operator '%s' expected %u input tensors but got %u.", meta->mnemonic, meta->in, num_in);
    mag_assert(meta->out == num_out, "op_validate: operator '%s' expected %u output tensors but got %u.", meta->mnemonic, meta->out, num_out);
  }
  for (uint32_t i=0; i < num_in; ++i)
    mag_assert(in[i] != NULL, "op_validate: input tensor %u for operator '%s' is NULL.", i, meta->mnemonic);
  for (uint32_t i=0; i < num_out; ++i)
    mag_assert(out[i] != NULL, "op_validate: output tensor %u for operator '%s' is NULL.", i, meta->mnemonic);
}

static void mag_bump_version(mag_tensor_t *t) {
  if (t->flags & MAG_TFLAG_IS_VIEW) /* If this is a view, bump the version of the base tensor */
    t = t->view_meta->base;
  ++t->version;
}

mag_status_t MAG_HOTPROC mag_dispatch(
  mag_error_t *err,
  mag_opcode_t op,
  bool inplace,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out,
  const mag_op_params_t *params
) {
  const mag_op_traits_t *meta = mag_op_trait(op);
  mag_assert2((in && num_in) || (out && num_out));
  mag_assert2(op != MAG_OP_NOP);
#if 0 /* Debug: print dispatched ops */
  mag_dbg_trace_op_ir(op, inplace, in, num_in, out, num_out);
#endif
  mag_context_t *ctx = in ? (*in)->ctx : (*out)->ctx;
  mag_device_t *device = in ? (*in)->storage->device : (*out)->storage->device;
  mag_assert_correct_op_data(op, in, num_in, out, num_out);
  mag_status_t status;
  if (!!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && meta->backward) {
    for (uint32_t i=0; i < num_out; ++i) {
      mag_tensor_t *r = out[i];
      mag_au_state_t *au = mag_au_state_lazy_alloc(&r->au_state, r->ctx);
      if (mag_unlikely(!au))
        return mag_set_error(err, MAG_ERR_OOM, "dispatch: failed to allocate autodiff state for gradient recording.");
      au->op = op;
      if (mag_unlikely(!mag_au_state_reserve_more_input_cap(au, num_in)))
        return mag_set_error(err, MAG_ERR_OOM, "dispatch: failed to reserve autodiff state input array.");
      for (uint32_t j=0; j < num_in; ++j) {
        mag_tensor_t *input = in[j];
        if (mag_unlikely(!input))
          return mag_set_error(err, MAG_ERR_OP, "dispatch: input tensor %u is NULL.", j);
        if (input->flags & MAG_TFLAG_REQUIRES_GRAD && !(r->flags & MAG_TFLAG_REQUIRES_GRAD)) {
          status = mag_tensor_set_requires_grad(err, r, true);
          if (mag_iserr(status)) return status;
        }
        if (mag_unlikely(!mag_au_state_append_input(au, input)))
          return mag_set_error(err, MAG_ERR_OOM, "dispatch: failed to push input into autodiff input array.");
      }
      if (params) au->params = *params;
    }
  }
  mag_command_t cmd = {
    .op = op,
    .in = in,
    .out = out,
    .num_in = num_in,
    .num_out = num_out,
    .params = params
  };
  mag_status_t (*submit)(mag_error_t *, mag_device_t *, const mag_command_t *) = device->submit;
  mag_status_t stat = (*submit)(err, device, &cmd);
  if (inplace)
    for (uint32_t i=0; i < num_out; ++i)
      mag_bump_version(out[i]);
  ++ctx->telemetry.ops_dispatched;
  return stat;
}
