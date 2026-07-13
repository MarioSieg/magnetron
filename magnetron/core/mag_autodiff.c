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
  for (uint32_t i=0; i < au->num_in; ++i) {
    if (au->in[i])
      mag_rc_decref(au->in[i]);
  }
  if (au->in != au->in_intrusive) { /* AU state stored inputs on the heap */
    (*mag_alloc)(au->in, 0, 0);
    au->in = NULL;
  }
  if (au->params) {
    mag_slab_free(&au->ctx->au_state_op_params_slab, au->params);
    au->params = NULL;
  }
  mag_slab_free(&au->ctx->au_state_slab, au);
  return MAG_OK;
}

mag_au_state_t *mag_au_state_lazy_alloc(mag_au_state_t **au, mag_context_t *ctx) {
  if (*au) return *au;
  mag_au_state_t *state = mag_slab_alloc(&ctx->au_state_slab);
  if (mag_unlikely(!state)) return NULL;
  *state = (mag_au_state_t) {
    .ctx = ctx,
    .op = MAG_OP_NOP,
    .in_intrusive = {NULL},
    .in = state->in_intrusive,
    .num_in = 0,
    .cap_in = MAG_AU_STATE_INTRUSIVE_STORAGE_NUM,
    .grad = NULL,
  };
  mag_rc_init_object(state, &mag_au_state_dtor);
  *au = state;
  return state;
}

bool mag_au_state_reserve_more_input_cap(mag_au_state_t *au,uint32_t extra) {
  if (mag_unlikely(extra > UINT32_MAX-au->num_in)) return false;
  uint32_t want = au->num_in + extra;
  if (want <= au->cap_in) return true;
  uint32_t new_cap = au->cap_in;
  while (new_cap < want) {
    if (new_cap > (UINT32_MAX>>1)) {
      new_cap = want;
      break;
    }
    new_cap<<=1;
  }
  mag_tensor_t **realloced;
  if (au->in == au->in_intrusive) { /* Transition from inline storage to heap */
    realloced = (*mag_try_alloc)(NULL, sizeof(*realloced)*new_cap, 0);
    if (mag_unlikely(!realloced)) return false;
    memcpy(realloced, au->in_intrusive, sizeof(*realloced) * au->num_in);
  } else {
    realloced = (*mag_try_alloc)(au->in, sizeof(*realloced) * new_cap, 0);
    if (mag_unlikely(!realloced)) return false;
  }
  au->in = realloced;
  au->cap_in = new_cap;
  return true;
}

bool mag_au_state_set_op_params(mag_au_state_t *au, const mag_op_params_t *params) {
  if (!au->params) {
    au->params = mag_slab_alloc(&au->ctx->au_state_op_params_slab);
    if (mag_unlikely(!au->params)) return false;
  }
  *au->params = *params;
  return true;
}

bool mag_au_state_set_input(mag_au_state_t *au, mag_tensor_t *x) {
  if (mag_unlikely(!x)) return false;
  if (mag_unlikely(!mag_au_state_reserve_more_input_cap(au, 1))) return false;
  mag_rc_incref(x);
  au->in[au->num_in++] = x;
  return true;
}

mag_tensor_t *mag_tensor_grad(const mag_tensor_t *tensor) {
  if (!(tensor->flags & MAG_TFLAG_REQUIRES_GRAD)) return NULL;
  if (!tensor->au_state) return NULL;
  mag_tensor_t *gra = tensor->au_state->grad;
  if (gra) mag_rc_incref(gra);
  return gra;
}

mag_status_t mag_tensor_set_grad(mag_error_t *err, mag_tensor_t *tensor, mag_tensor_t *grad) {
  if (!grad) {
    if (tensor->au_state && tensor->au_state->grad) {
      mag_rc_decref(tensor->au_state->grad);
      tensor->au_state->grad = NULL;
    }
    return MAG_OK;
  }
  if (!(tensor->flags & MAG_TFLAG_REQUIRES_GRAD)) {
    mag_status_t status = mag_tensor_set_requires_grad(err, tensor, true);
    if (mag_iserr(status)) return status;
  }
  if (!tensor->au_state) {
    if (!mag_au_state_lazy_alloc(&tensor->au_state, tensor->ctx))
      return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate autodiff state for grad assignment.");
  }
  if (tensor->au_state->grad)
    mag_rc_decref(tensor->au_state->grad);
  mag_rc_incref(grad);
  grad->flags = (grad->flags|MAG_TFLAG_IS_GRAD)&~MAG_TFLAG_REQUIRES_GRAD;
  tensor->au_state->grad = grad;
  return MAG_OK;
}

bool mag_tensor_requires_grad(const mag_tensor_t *tensor) { return tensor->flags & MAG_TFLAG_REQUIRES_GRAD; }

mag_status_t mag_tensor_set_requires_grad(mag_error_t *err, mag_tensor_t *tensor, bool requires_grad) {
  if (requires_grad) {
    if (mag_unlikely(!mag_tensor_is_floating_point_typed(tensor)))
      return mag_set_error(err, MAG_ERR_PARAM, "autograd: gradient tracking requires a floating-point dtype, but tensor has dtype %s.", mag_type_trait(tensor->dtype)->name);
    tensor->flags |= MAG_TFLAG_REQUIRES_GRAD;
    if (mag_unlikely(!mag_au_state_lazy_alloc(&tensor->au_state, tensor->ctx))) {
      tensor->flags &= ~MAG_TFLAG_REQUIRES_GRAD;
      return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate autodiff state.");
    }
    return MAG_OK;
  }
  tensor->flags &= ~MAG_TFLAG_REQUIRES_GRAD;
  return MAG_OK;
}

static void mag_tensor_patch_grad(mag_tensor_t *dst, mag_tensor_t *grad) {
  if (dst->au_state->grad)
    mag_rc_decref(dst->au_state->grad);
  grad->flags = (grad->flags|MAG_TFLAG_IS_GRAD)&~MAG_TFLAG_REQUIRES_GRAD;
  dst->au_state->grad = grad;
}

mag_status_t mag_tensor_backward(mag_error_t *err, mag_tensor_t *root) {
  mag_status_t status = MAG_OK;
  if (mag_unlikely(!(root->flags & MAG_TFLAG_REQUIRES_GRAD)))
    return mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: missing backward info for tensor - it does not require gradients.");
  if (mag_unlikely(!(root->coords.rank == 0 && root->numel == 1)))
    return mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: backpropagation requires a scalar root tensor.");
  mag_ctx_grad_recorder_stop(root->ctx);
  mag_tensor_t *root_grad=NULL; /* Seed root gradient */
  if (mag_iserr(mag_ones_like(err, &root_grad, root))) {
    mag_ctx_grad_recorder_start(root->ctx);
    return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate root gradient.");
  }
  mag_tensor_patch_grad(root, root_grad);
  mag_topo_set_t post_order;
  mag_topo_set_init(&post_order);
  status = mag_topo_sort(err, root, &post_order);
  mag_tensor_t *grads_intrusive[MAG_AU_STATE_INTRUSIVE_STORAGE_NUM];
  mag_tensor_t **grads_dyn = NULL;
  size_t grads_cap = 0;
  if (mag_unlikely(mag_iserr(status))) goto cleanup;
  if (mag_unlikely(!post_order.size)) goto cleanup;
  for (size_t i=post_order.size; i --> 0;) {
    mag_tensor_t *child = post_order.data[i];
    if (mag_unlikely(!(child && child->au_state))) {
      status = mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: autodiff state is missing for a tensor in the computation graph.");
      goto cleanup;
    }
    if (mag_unlikely(!child->au_state->grad || child->au_state->op == MAG_OP_NOP))
      continue;
    const mag_op_traits_t *meta = mag_op_trait(child->au_state->op);
    mag_status_t (*backward)(mag_error_t *, mag_au_state_t *, mag_tensor_t **) = meta->backward;
    if (mag_unlikely(backward == NULL)) {
      status = mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: operator '%s' has no backward implementation.", meta->mnemonic);
      goto cleanup;
    }
    mag_tensor_t **grads;
    uint32_t num_in;
    if (meta->in == MAG_OP_INOUT_DYN) {
      num_in = child->au_state->num_in;
    } else {
      if (mag_unlikely(child->au_state->num_in != meta->in)) {
        status = mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: operator '%s' input count is invalid, required: %u, got: %u", meta->mnemonic, meta->in, child->au_state->num_in);
        goto cleanup;
      }
      num_in = meta->in;
    }
    if (num_in <= MAG_AU_STATE_INTRUSIVE_STORAGE_NUM) {
      grads = grads_intrusive;
    } else {
      if (num_in > grads_cap) {
        size_t nc = grads_cap ? grads_cap : MAG_AU_STATE_INTRUSIVE_STORAGE_NUM;
        while (nc < num_in)
          nc <<= 1;
        void *realloced = (*mag_try_alloc)(grads_dyn, nc * sizeof(*grads_dyn), grads_cap * sizeof(*grads_dyn));
        if (mag_unlikely(!realloced)) {
          status = mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate backward gradients.");
          goto cleanup;
        }
        grads_dyn = realloced;
        grads_cap = nc;
      }
      grads = grads_dyn;
    }
    memset(grads, 0, num_in*sizeof(*grads)); /* Reset only activate range */
    status = (*backward)(err, child->au_state, grads);
    if (mag_iserr(status))
      goto cleanup;
    for (uint32_t j=0; j < num_in; ++j) {
      mag_tensor_t *input = child->au_state->in[j];
      if (mag_unlikely(!input) || !(input->flags & MAG_TFLAG_REQUIRES_GRAD))
        continue;
      mag_tensor_t *gri = grads[j];
      if (mag_unlikely(!gri)) {
        status = mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: backward of operator '%s' did not produce a valid gradient for input %u.", meta->mnemonic, j);
        goto cleanup;
      }
      if (!input->au_state->grad) {
        mag_tensor_patch_grad(input, gri);
      } else {
        status = mag_add_(err, &gri, gri, input->au_state->grad);
        if (mag_iserr(status)) goto cleanup;
        mag_tensor_patch_grad(input, gri);
        mag_rc_decref(gri);
      }
    }
  }
cleanup:
  if (grads_dyn)
    (*mag_alloc)(grads_dyn, 0, 0);
  mag_topo_set_free(&post_order);
  mag_ctx_grad_recorder_start(root->ctx);
  return status;
}

mag_status_t mag_tensor_zero_grad(mag_error_t *err,mag_tensor_t *tensor) {
  if (tensor->flags & MAG_TFLAG_REQUIRES_GRAD && tensor->au_state && tensor->au_state->grad)
    return mag_zeros_(err, tensor->au_state->grad);
  return MAG_OK;
}
