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

#include "mag_autodiff.h"
#include "mag_slab.h"
#include "mag_context.h"
#include "mag_alloc.h"
#include "mag_hashset.h"
#include "mag_toposort.h"

static mag_status_t mag_au_state_dtor(void *p) {
  mag_au_state_t *au = p;
  if (au->grad) {
    mag_rc_decref(au->grad);
    au->grad = NULL;
  }
  for (size_t i=0; i < sizeof(au->op_inputs)/sizeof(*au->op_inputs); ++i)
    if (au->op_inputs[i]) mag_rc_decref(au->op_inputs[i]);
  mag_slab_free(&au->ctx->au_state_slab, au);
  return MAG_STATUS_OK;
}

mag_au_state_t *mag_au_state_lazy_alloc(mag_au_state_t **au_state, mag_context_t *ctx) {
  if (*au_state) return *au_state;
  *au_state = mag_slab_alloc(&ctx->au_state_slab);
  if (mag_unlikely(!*au_state)) return NULL;
  **au_state = (mag_au_state_t) {
    .ctx = ctx,
    .op = MAG_OP_NOP,
    .op_inputs = {},
    .op_attrs = {},
    .grad = NULL,
  };
  mag_rc_init_object(*au_state, &mag_au_state_dtor);
  return *au_state;
}

mag_status_t mag_tensor_grad(mag_error_t *err, const mag_tensor_t *tensor, mag_tensor_t **out_grad) {
  if (mag_unlikely(!(tensor->flags & MAG_TFLAG_REQUIRES_GRAD)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "autograd: tensor does not require gradient; enable requires_grad to access its gradient.");
  if (mag_unlikely(!tensor->au_state))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_STATE, "autograd: autodiff state is missing for this tensor.");
  if (tensor->au_state->grad) mag_rc_incref(tensor->au_state->grad);
  *out_grad = tensor->au_state->grad;
  return MAG_STATUS_OK;
}

bool mag_tensor_requires_grad(const mag_tensor_t *tensor) {
  return tensor->flags & MAG_TFLAG_REQUIRES_GRAD;
}

mag_status_t mag_tensor_set_requires_grad(mag_error_t *err, mag_tensor_t *tensor, bool requires_grad) {
  if (requires_grad) {
    if (mag_unlikely(!mag_tensor_is_floating_point_typed(tensor)))
      return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "autograd: gradient tracking requires a floating-point dtype, but tensor has dtype %s.", mag_type_trait(tensor->dtype)->name);
    tensor->flags |= MAG_TFLAG_REQUIRES_GRAD;
    if (mag_unlikely(!mag_au_state_lazy_alloc(&tensor->au_state, tensor->ctx))) {
      tensor->flags &= ~MAG_TFLAG_REQUIRES_GRAD;
      return mag_set_error(err, MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "autograd: failed to allocate autodiff state.");
    }
    return MAG_STATUS_OK;
  }
  tensor->flags &= ~MAG_TFLAG_REQUIRES_GRAD;
  return MAG_STATUS_OK;
}

static void mag_tensor_patch_grad(mag_tensor_t *dst, mag_tensor_t *grad) {
  if (dst->au_state->grad)
    mag_rc_decref(dst->au_state->grad);
  grad->flags = (grad->flags|MAG_TFLAG_IS_GRAD)&~MAG_TFLAG_REQUIRES_GRAD;
  dst->au_state->grad = grad;
}

mag_status_t mag_tensor_backward(mag_error_t *err, mag_tensor_t *root) {
  mag_status_t stat = MAG_STATUS_OK;
  mag_topo_set_t post_order;
  bool topo_init = false;
  if (mag_unlikely(!(root->flags & MAG_TFLAG_REQUIRES_GRAD)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "autograd: cannot backpropagate from a tensor that does not require gradients; create it with requires_grad=True.");
  if (mag_unlikely(!(root->coords.rank == 0 && root->numel == 1)))
    return mag_set_error(err, MAG_STATUS_ERR_INVALID_PARAM, "autograd: backpropagation requires a scalar root tensor (rank 0, numel 1).");
  mag_ctx_grad_recorder_stop(root->ctx);
  mag_topo_set_init(&post_order);
  topo_init = true;
  stat = mag_topo_sort(err, root, &post_order);
  if (mag_unlikely(mag_iserr(stat)))
    goto cleanup;
  if (mag_unlikely(!post_order.size))
    goto cleanup;
  for (size_t i=0, j=post_order.size-1; i < j; ++i, --j)
    mag_swap(mag_tensor_t *, post_order.data[i], post_order.data[j]);
  for (size_t id=0; id < post_order.size; ++id) {
    mag_tensor_t *child = post_order.data[id];
    if (mag_unlikely(!(child && child->au_state))) {
      stat = mag_set_error(err, MAG_STATUS_ERR_INVALID_STATE, "autograd: autodiff state is missing for a tensor in the computation graph.");
      goto cleanup;
    }
    const mag_op_traits_t *meta = mag_op_trait(child->au_state->op);
    if (!child->au_state->grad) {
      mag_tensor_t *grad = NULL;
      stat = mag_full_like(err, &grad, child, mag_scalar_from_f64(1.0));
      if (mag_unlikely(stat != MAG_STATUS_OK))
        goto cleanup;
      mag_tensor_patch_grad(child, grad);
    }
    if (mag_unlikely(child->au_state->op == MAG_OP_NOP))
      continue;
    mag_tensor_t *grads[MAG_MAX_OP_INPUTS] = {0};
    mag_status_t (*backward)(mag_error_t *, mag_au_state_t *, mag_tensor_t **) = meta->backward;
    if (mag_unlikely(backward == NULL)) {
      stat = mag_set_error(err, MAG_STATUS_ERR_INVALID_STATE, "autograd: operator '%s' has no backward implementation.", meta->mnemonic);
      goto cleanup;
    }
    stat = backward(err, child->au_state, grads);
    if (mag_unlikely(stat != MAG_STATUS_OK))
      goto cleanup;
    uint32_t numin = meta->in;
    mag_assert(numin <= MAG_MAX_OP_INPUTS, "autograd: operator '%s' has too many inputs (%u > %d).", meta->mnemonic, numin, MAG_MAX_OP_INPUTS);
    for (uint32_t i=0; i < numin; ++i) {
      mag_tensor_t *input = child->au_state->op_inputs[i];
      mag_assert2(input);
      if (!(input->flags & MAG_TFLAG_REQUIRES_GRAD))
        continue;
      mag_tensor_t *gri = grads[i];
      mag_assert(gri, "autograd: backward of operator '%s' did not produce a gradient for input %d.", meta->mnemonic, i);
      if (!input->au_state->grad) {
        mag_tensor_patch_grad(input, gri);
      } else {
        mag_tensor_t *acc = NULL;
        stat = mag_add(err, &acc, gri, input->au_state->grad);
        if (mag_unlikely(stat != MAG_STATUS_OK)) {
          mag_rc_decref(gri);
          goto cleanup;
        }
        mag_tensor_patch_grad(input, acc);
        mag_rc_decref(gri);
      }
    }
  }
cleanup:
  if (topo_init)
    mag_topo_set_free(&post_order);
  mag_ctx_grad_recorder_start(root->ctx);
  return stat;
}

mag_status_t mag_tensor_zero_grad(mag_error_t *err,mag_tensor_t *tensor) {
  if (tensor->flags & MAG_TFLAG_REQUIRES_GRAD && tensor->au_state && tensor->au_state->grad)
    return mag_zeros_(err, tensor->au_state->grad);
  return MAG_STATUS_OK;
}
