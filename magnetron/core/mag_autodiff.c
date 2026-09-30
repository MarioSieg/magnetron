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
#include "mag_op_grads.h"

static mag_status_t mag_au_state_dtor(void *p) {
  mag_au_state_t *au = p;
  if (au->grad) {
    mag_rc_decref(au->grad);
    au->grad = NULL;
  }
  for (uint32_t i=0; i < au->num_in; ++i) {
    if (au->in[i])
      mag_rc_decref(au->in[i]);
    if (au->in_nodes[i])
      mag_rc_decref(au->in_nodes[i]);
  }
  if (au->in != au->in_intrusive) { /* AU state stored inputs on the heap */
    (*mag_alloc)(au->in, 0, 0);
    au->in = NULL;
  }
  if (au->in_nodes != au->in_nodes_intrusive) {
    (*mag_alloc)(au->in_nodes, 0, 0);
    au->in_nodes = NULL;
  }
  if (au->in_versions != au->in_versions_intrusive) {
    (*mag_alloc)(au->in_versions, 0, 0);
    au->in_versions = NULL;
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
    .owner = NULL,
    .in_nodes = state->in_nodes_intrusive,
    .in_versions = state->in_versions_intrusive,
    .owner_version = 0,
    .retain_grad = false,
  };
  mag_rc_init_object(state, &mag_au_state_dtor);
  *au = state;
  return state;
}

bool mag_au_state_reserve_more_input_cap(mag_au_state_t *au,uint32_t extra) {
  if (mag_unlikely(extra > UINT32_MAX-au->num_in)) return false;
  uint32_t len = au->num_in + extra;
  if (len <= au->cap_in) return true;
  uint32_t cap = au->cap_in;
  for (; cap < len; cap<<=1) {
    if (cap > (UINT32_MAX>>1)) {
      cap = len;
      break;
    }
  }
  mag_tensor_t **re_in;
  mag_au_state_t **re_au;
  uint64_t *re_v;
  if (au->in == au->in_intrusive) { /* Transition from inline storage to heap */
    re_in = (*mag_try_alloc)(NULL, sizeof(*re_in)*cap, 0);
    if (mag_unlikely(!re_in)) return false;
    memcpy(re_in, au->in_intrusive, sizeof(*re_in)*au->num_in);
    re_au = (*mag_try_alloc)(NULL, sizeof(*re_au)*cap, 0);
    if (mag_unlikely(!re_au)) { (*mag_alloc)(re_in, 0, 0); return false; }
    memcpy(re_au, au->in_nodes_intrusive, sizeof(*re_au)*au->num_in);
    re_v = (*mag_try_alloc)(NULL, sizeof(*re_v)*cap, 0);
    if (mag_unlikely(!re_v)) { (*mag_alloc)(re_in, 0, 0); (*mag_alloc)(re_au, 0, 0); return false; }
    memcpy(re_v, au->in_versions_intrusive, sizeof(*re_v)*au->num_in);
  } else {
    re_in = (*mag_try_alloc)(au->in, sizeof(*re_in)*cap, 0);
    if (mag_unlikely(!re_in)) return false;
    au->in = re_in;
    re_au = (*mag_try_alloc)(au->in_nodes, sizeof(*re_au)*cap, 0);
    if (mag_unlikely(!re_au)) return false;
    au->in_nodes = re_au;
    re_v = (*mag_try_alloc)(au->in_versions, sizeof(*re_v)*cap, 0);
    if (mag_unlikely(!re_v)) return false;
  }
  au->in = re_in;
  au->in_nodes = re_au;
  au->in_versions = re_v;
  au->cap_in = cap;
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
  au->in_versions[au->num_in] = mag_tensor_current_version(x);
  au->in_nodes[au->num_in] = x->au_state;
  if (x->au_state) mag_rc_incref(x->au_state);
  au->in[au->num_in++] = x;
  return true;
}

uint64_t mag_tensor_current_version(const mag_tensor_t *t) {
  if (t->meta.flags & MAG_TFLAG_IS_VIEW) t = t->view_meta->base;
  return (uint64_t)mag_atomic64_load((mag_atomic64_t *)&t->version, MAG_MO_RELAXED);
}

bool mag_au_state_output_is_valid(const mag_au_state_t *node) {
  return node->owner && node->owner->au_state == node && mag_tensor_current_version(node->owner) == node->owner_version;
}

bool mag_tensor_is_leaf(const mag_tensor_t *tensor) {
  return !tensor->au_state || tensor->au_state->op == MAG_OP_NOP;
}

mag_status_t mag_tensor_retain_grad(mag_error_t *err, mag_tensor_t *tensor) {
  if (mag_unlikely(!(tensor->meta.flags & MAG_TFLAG_REQUIRES_GRAD)))
    return mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: retain_grad requires a tensor that requires gradients.");
  if (!tensor->au_state && !mag_au_state_lazy_alloc(&tensor->au_state, tensor->ctx))
    return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate autodiff state.");
  if (!tensor->au_state->owner) tensor->au_state->owner = tensor;
  tensor->au_state->retain_grad = true;
  return MAG_OK;
}

void mag_au_state_clear_inputs(mag_au_state_t *au) {
  for (uint32_t i=0; i < au->num_in; ++i) {
    if (au->in[i]) mag_rc_decref(au->in[i]);
    au->in[i] = NULL;
    if (au->in_nodes[i]) mag_rc_decref(au->in_nodes[i]);
    au->in_nodes[i] = NULL;
  }
  au->num_in = 0;
  if (au->params) {
    mag_slab_free(&au->ctx->au_state_op_params_slab, au->params);
    au->params = NULL;
  }
}

mag_tensor_t *mag_tensor_grad(const mag_tensor_t *tensor) {
  if (!(tensor->meta.flags & MAG_TFLAG_REQUIRES_GRAD)) return NULL;
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
  if (!(tensor->meta.flags & MAG_TFLAG_REQUIRES_GRAD)) {
    mag_status_t status = mag_tensor_set_requires_grad(err, tensor, true);
    if (mag_iserr(status)) return status;
  }
  if (!tensor->au_state) {
    if (!mag_au_state_lazy_alloc(&tensor->au_state, tensor->ctx))
      return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate autodiff state for grad assignment.");
  }
  if (!tensor->au_state->owner) tensor->au_state->owner = tensor;
  if (tensor->au_state->grad)
    mag_rc_decref(tensor->au_state->grad);
  mag_rc_incref(grad);
  grad->meta.flags = (grad->meta.flags|MAG_TFLAG_IS_GRAD)&~MAG_TFLAG_REQUIRES_GRAD;
  tensor->au_state->grad = grad;
  return MAG_OK;
}

bool mag_tensor_requires_grad(const mag_tensor_t *tensor) { return tensor->meta.flags & MAG_TFLAG_REQUIRES_GRAD; }

mag_status_t mag_tensor_set_requires_grad(mag_error_t *err, mag_tensor_t *tensor, bool requires_grad) {
  if (requires_grad) {
    if (mag_unlikely(!mag_tensor_is_floating_point_typed(tensor)))
      return mag_set_error(err, MAG_ERR_PARAM, "autograd: gradient tracking requires a floating-point dtype, but tensor has dtype %s.", mag_type_trait(tensor->meta.dtype)->name);
    tensor->meta.flags |= MAG_TFLAG_REQUIRES_GRAD;
    if (mag_unlikely(!mag_au_state_lazy_alloc(&tensor->au_state, tensor->ctx))) {
      tensor->meta.flags &= ~MAG_TFLAG_REQUIRES_GRAD;
      return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate autodiff state.");
    }
    if (!tensor->au_state->owner) tensor->au_state->owner = tensor;
    return MAG_OK;
  }
  tensor->meta.flags &= ~MAG_TFLAG_REQUIRES_GRAD;
  return MAG_OK;
}

static void mag_node_patch_grad(mag_au_state_t *node, mag_tensor_t *grad) {
  if (node->grad)
    mag_rc_decref(node->grad);
  grad->meta.flags = (grad->meta.flags|MAG_TFLAG_IS_GRAD)&~MAG_TFLAG_REQUIRES_GRAD;
  node->grad = grad;
}

static void mag_tensor_patch_grad(mag_tensor_t *dst, mag_tensor_t *grad) {
  mag_node_patch_grad(dst->au_state, grad);
}

mag_status_t mag_tensor_backward(mag_error_t *err, mag_tensor_t *root) {
  mag_status_t status = MAG_OK;
  if (mag_unlikely(!(root->meta.flags & MAG_TFLAG_REQUIRES_GRAD)))
    return mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: missing backward info for tensor - it does not require gradients.");
  if (mag_unlikely(!(root->meta.coords.rank == 0 && root->meta.numel == 1)))
    return mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: backpropagation requires a scalar root tensor.");
  mag_context_t *ctx = root->ctx;
  mag_atomic64_fetch_add(&ctx->telemetry.backward_passes, 1, MAG_MO_RELAXED);
  bool grad_was_on = mag_ctx_grad_recorder_is_running(ctx);
  mag_ctx_grad_recorder_stop(ctx);
  mag_tensor_t *root_grad=NULL; /* Seed root gradient */
  if (mag_iserr(mag_ones_like(err, &root_grad, root))) {
    if (grad_was_on) mag_ctx_grad_recorder_start(ctx);
    return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate root gradient.");
  }
  mag_tensor_patch_grad(root, root_grad);
  mag_topo_stack_t topo_stack = {0};
  mag_topo_set_t topo_set = {0};
  mag_topo_set_t *post_order = &topo_set;
  if (mag_unlikely(!mag_topo_set_init(&topo_set, MAG_TOPOSORT_HASHSET_INIT_CAP))) {
    if (grad_was_on) mag_ctx_grad_recorder_start(ctx);
    return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate traversal set.");
  }
  if (mag_unlikely(!mag_topo_stack_init(&topo_stack, MAG_TOPOSORT_STACK_INIT_CAP))) {
    mag_topo_set_free(&topo_set);
    if (grad_was_on) mag_ctx_grad_recorder_start(ctx);
    return mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate traversal stack.");
  }
  int64_t topo_epoch = 0;
  status = mag_topo_sort(err, root->au_state, &topo_stack, post_order, &topo_epoch);
  mag_tensor_t *grads_intrusive[MAG_AU_STATE_INTRUSIVE_STORAGE_NUM];
  mag_tensor_t **grads_dyn = NULL;
  size_t grads_cap = 0;
  if (mag_unlikely(mag_iserr(status))) goto cleanup;
  if (mag_unlikely(!post_order->len)) goto cleanup;
  mag_atomic64_fetch_add(&ctx->telemetry.backward_nodes_visited, (mag_atomic64_t)post_order->len, MAG_MO_RELAXED);
  for (size_t i=post_order->len; i --> 0;) {
    mag_au_state_t *node = post_order->buf[i];
    if (mag_unlikely(!node->grad || node->op == MAG_OP_NOP))
      continue;
    const mag_op_traits_t *meta = mag_op_trait(node->op);
    mag_status_t (*backward)(mag_error_t *, mag_au_state_t *, mag_tensor_t **) = meta->backward;
    if (mag_unlikely(backward == NULL)) {
      status = mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: operator '%s' has no backward implementation.", meta->mnemonic);
      goto cleanup;
    }
    bool recompute = mag_op_trait(node->op)->flags&MAG_OP_FLAG_GRAD_READS_OUT && !mag_au_state_output_is_valid(node);
    if (mag_op_trait(node->op)->flags&MAG_OP_FLAG_GRAD_READS_IN || recompute) {
      for (uint32_t j=0; j < node->num_in; ++j) {
        mag_tensor_t *input = node->in[j];
        if (!recompute && !mag_op_backward_reads_input(node->op, node->in, node->num_in, j)) continue;
        if (mag_unlikely(input && mag_tensor_current_version(input) != node->in_versions[j])) {
          status = mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: a tensor needed for the gradient of operator '%s' has been modified by an in-place operation.", meta->mnemonic);
          goto cleanup;
        }
      }
    }
    mag_tensor_t **grads;
    uint32_t num_in;
    if (meta->in == MAG_OP_INOUT_DYN) {
      num_in = node->num_in;
    } else {
      if (mag_unlikely(node->num_in != meta->in)) {
        status = mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: operator '%s' input count is invalid, required: %u, got: %u", meta->mnemonic, meta->in, node->num_in);
        goto cleanup;
      }
      num_in = meta->in;
    }
    if (num_in <= MAG_AU_STATE_INTRUSIVE_STORAGE_NUM) {
      grads = grads_intrusive;
    } else {
      if (num_in > grads_cap) {
        size_t cap = grads_cap ? grads_cap : MAG_AU_STATE_INTRUSIVE_STORAGE_NUM;
        for (; cap < num_in; cap <<= 1);
        void *realloced = (*mag_try_alloc)(grads_dyn, cap*sizeof(*grads_dyn), grads_cap*sizeof(*grads_dyn));
        if (mag_unlikely(!realloced)) {
          status = mag_set_error(err, MAG_ERR_OOM, "autograd: failed to allocate backward gradients.");
          goto cleanup;
        }
        grads_dyn = realloced;
        grads_cap = cap;
      }
      grads = grads_dyn;
    }
    memset(grads, 0, num_in*sizeof(*grads)); /* Reset only activate range */
    status = (*backward)(err, node, grads);
    if (mag_iserr(status))
      goto cleanup;
    for (uint32_t j=0; j < num_in; ++j) {
      mag_tensor_t *input = node->in[j];
      mag_au_state_t *inode = node->in_nodes[j];
      if (mag_unlikely(!input || !inode) || !(input->meta.flags & MAG_TFLAG_REQUIRES_GRAD))
        continue;
      mag_tensor_t *gri = grads[j];
      if (mag_unlikely(!gri)) {
        status = mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: backward of operator '%s' did not produce a valid gradient for input %u.", meta->mnemonic, j);
        goto cleanup;
      }
      if (!inode->grad) {
        mag_node_patch_grad(inode, gri);
        mag_atomic64_fetch_add(&ctx->telemetry.grads_materialized, 1, MAG_MO_RELAXED);
      } else {
        status = mag_add_(err, &gri, gri, inode->grad);
        if (mag_iserr(status)) goto cleanup;
        mag_node_patch_grad(inode, gri);
        mag_rc_decref(gri);
      }
    }
    if (!node->retain_grad && node->grad) {
      mag_rc_decref(node->grad);
      node->grad = NULL;
    }
  }
cleanup:
  if (topo_epoch) mag_topo_release(post_order, topo_epoch);
  if (grads_dyn)
    (*mag_alloc)(grads_dyn, 0, 0);
  mag_topo_stack_free(&topo_stack);
  mag_topo_set_free(&topo_set);
  if (grad_was_on) mag_ctx_grad_recorder_start(ctx);
  return status;
}

mag_status_t mag_tensor_zero_grad(mag_error_t *err,mag_tensor_t *tensor) {
  if (tensor->meta.flags&MAG_TFLAG_REQUIRES_GRAD && tensor->au_state && tensor->au_state->grad)
    return mag_zeros_(err, tensor->au_state->grad);
  return MAG_OK;
}
