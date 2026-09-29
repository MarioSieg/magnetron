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
#include "mag_op_grads.h"

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

static void mag_bump_version(mag_tensor_t *tensor) {
  if (tensor->meta.flags & MAG_TFLAG_IS_VIEW) /* If this is a view, bump the version of the base tensor */
    tensor = tensor->view_meta->base;
  mag_atomic64_fetch_add(&tensor->version, 1, MAG_MO_RELAXED);
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
  mag_device_t *device = in ? (*in)->meta.device : (*out)->meta.device;
  mag_assert_correct_op_data(op, in, num_in, out, num_out);
  bool record = false;
  if (!mag_tls_state.no_grad && meta->backward)
    for (uint32_t j=0; j < num_in; ++j)
      if (in[j] && in[j]->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
        record = true;
        break;
      }
  if (record && inplace)
    for (uint32_t i=0; i < num_out; ++i)
      if (mag_unlikely(out[i]->meta.flags & MAG_TFLAG_REQUIRES_GRAD && mag_tensor_is_leaf(out[i])))
        return mag_set_error(err, MAG_ERR_PARAM, "autograd: a leaf tensor that requires grad is being used in an in-place operation.\n\tHint: use the out-of-place variant, detach the tensor, or disable gradient tracking.");
  mag_tensor_t *rec_in_intrusive[MAG_AU_STATE_INTRUSIVE_STORAGE_NUM];
  mag_tensor_t **rec_in = in;
  mag_tensor_t *prev = NULL;
  mag_status_t rec_status = MAG_OK;
  if (record) {
    for (uint32_t i=0; i < num_out; ++i) {
      mag_tensor_t *r = out[i];
      bool aliased = false;
      for (uint32_t j=0; j < num_in; ++j) aliased|=in[j] == r;
      if (!aliased && !(inplace && r->au_state && r->au_state->op != MAG_OP_NOP)) continue;
      if (mag_unlikely(prev))
        return mag_set_error(err, MAG_ERR_PARAM, "dispatch: in-place operator '%s' has more than one aliased output.", meta->mnemonic);
      if (mag_unlikely(r->meta.flags&MAG_TFLAG_REQUIRES_GRAD && mag_tensor_is_leaf(r)))
        return mag_set_error(err, MAG_ERR_PARAM, "autograd: a leaf tensor that requires grad is being used in an in-place operation.\n\tHint: use the out-of-place variant, detach the tensor, or disable gradient tracking.");
      if (mag_unlikely(r->meta.flags&MAG_TFLAG_IS_VIEW && r->view_meta->base->meta.flags&MAG_TFLAG_REQUIRES_GRAD))
        return mag_set_error(err, MAG_ERR_PARAM, "autograd: in-place operations on a view of a tensor that requires grad are not supported.\n\tHint: apply the operation to the base tensor, or build the result out-of-place.");
      bool clone = false;
      for (uint32_t j=0; j < num_in; ++j)
        if (in[j] == r && mag_op_backward_reads_input(op, in, num_in, j)) clone = true;
      bool grad_on = mag_ctx_grad_recorder_is_running(ctx);
      if (grad_on) mag_ctx_grad_recorder_stop(ctx);
      if (clone) rec_status = mag_clone(err, &prev, r); /* Inplace grad might require clone */
      else rec_status = mag_strided_view(err, &prev, ctx, r, r->meta.coords.rank, r->meta.coords.shape, r->meta.coords.strides, r->meta.storage_offset);
      if (grad_on) mag_ctx_grad_recorder_start(ctx);
      if (mag_iserr(rec_status)) return rec_status;
      if (prev->au_state) { mag_rc_decref(prev->au_state); prev->au_state = NULL; }
      prev->au_state = r->au_state;
      r->au_state = NULL;
      if (prev->au_state) prev->au_state->owner = prev;
      prev->meta.flags = (prev->meta.flags&~MAG_TFLAG_REQUIRES_GRAD)|(r->meta.flags&MAG_TFLAG_REQUIRES_GRAD);
      if (num_in > MAG_AU_STATE_INTRUSIVE_STORAGE_NUM) {
        mag_rc_decref(prev);
        return mag_set_error(err, MAG_ERR_PARAM, "dispatch: in-place operator '%s' has too many inputs for gradient recording.", meta->mnemonic);
      }
      for (uint32_t j=0; j < num_in; ++j) rec_in_intrusive[j] = in[j] == r ? prev : in[j];
      rec_in = rec_in_intrusive;
    }
    for (uint32_t i=0; i < num_out; ++i) {
      mag_tensor_t *r = out[i];
      if (!mag_tensor_is_floating_point_typed(r)) continue;
      mag_au_state_t *au = mag_au_state_lazy_alloc(&r->au_state, r->ctx);
      if (mag_unlikely(!au)) {
        if (prev) mag_rc_decref(prev);
        return mag_set_error(err, MAG_ERR_OOM, "dispatch: failed to allocate autodiff state for gradient recording.");
      }
      mag_au_state_clear_inputs(au);
      au->op = op;
      au->owner = r;
      au->retain_grad = false;
      if (mag_unlikely(!mag_au_state_reserve_more_input_cap(au, num_in))) {
        if (prev) mag_rc_decref(prev);
        return mag_set_error(err, MAG_ERR_OOM, "dispatch: failed to reserve autodiff state input array.");
      }
      for (uint32_t j=0; j < num_in; ++j) {
        mag_tensor_t *input = rec_in[j];
        if (mag_unlikely(!input))
          return mag_set_error(err, MAG_ERR_OP, "dispatch: input tensor %u is NULL.", j);
        if (input->meta.flags&MAG_TFLAG_REQUIRES_GRAD && !(r->meta.flags&MAG_TFLAG_REQUIRES_GRAD)) {
          mag_status_t status = mag_tensor_set_requires_grad(err, r, true);
          if (mag_iserr(status)) return status;
        }
        if (mag_unlikely(!mag_au_state_set_input(au, input))) {
          if (prev) mag_rc_decref(prev);
          return mag_set_error(err, MAG_ERR_OOM, "dispatch: failed to push input into autodiff input array.");
        }
      }
      if (params) mag_au_state_set_op_params(au, params);
    }
    if (prev) mag_rc_decref(prev);
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
  if (record)
    for (uint32_t i=0; i < num_out; ++i)
      if (out[i]->au_state && out[i]->au_state->owner == out[i])
        out[i]->au_state->owner_version = mag_tensor_current_version(out[i]);
  mag_atomic64_fetch_add(&ctx->telemetry.ops_dispatched, 1, MAG_MO_RELAXED);
  return stat;
}
