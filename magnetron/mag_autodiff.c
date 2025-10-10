/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "mag_autodiff.h"
#include "mag_pool.h"
#include "mag_context.h"
#include "mag_alloc.h"
#include "mag_hashset.h"
#include "mag_toposort.h"

static void mag_au_state_dtor(void *p) {
    mag_au_state_t *au = p;
    if (au->grad) {
        mag_tensor_decref(au->grad);
        au->grad = NULL;
    }
    for (size_t i=0; i < sizeof(au->op_inputs)/sizeof(*au->op_inputs); ++i)
        if (au->op_inputs[i]) mag_tensor_decref(au->op_inputs[i]);
    mag_fixed_pool_free_block(&au->ctx->view_meta_pool, au);
}

mag_au_state_t *mag_au_state_lazy_alloc(mag_au_state_t **au_state, mag_context_t *ctx) {
    if (*au_state) return *au_state;
    *au_state = mag_fixed_pool_alloc_block(&ctx->au_state_pool);
    **au_state = (mag_au_state_t) {
        .ctx = ctx,
        .rc = mag_rc_control_init(*au_state, &mag_au_state_dtor),
        .op = MAG_OP_NOP,
        .op_inputs = {},
        .op_params = {},
        .grad = NULL,
    };
    return *au_state;
}

mag_tensor_t *mag_tensor_get_grad(const mag_tensor_t *t) {
    mag_assert2(t->flags & MAG_TFLAG_REQUIRES_GRAD && t->au_state);
    if (t->au_state->grad) mag_tensor_incref(t->au_state->grad);
    return t->au_state->grad;
}

bool mag_tensor_requires_grad(const mag_tensor_t *t) {
    return t->flags & MAG_TFLAG_REQUIRES_GRAD;
}

void mag_tensor_set_requires_grad(mag_tensor_t *t, bool requires_grad) {
    if (requires_grad) {
        mag_assert(mag_tensor_is_floating_point_typed(t), "Gradient tracking tensors must be floating-point typed, but tensor has dtype: %s", mag_dtype_meta_of(t->dtype)->name);
        t->flags |= MAG_TFLAG_REQUIRES_GRAD;
        mag_au_state_lazy_alloc(&t->au_state, t->ctx);
    } else t->flags &= ~MAG_TFLAG_REQUIRES_GRAD;
}

static void mag_tensor_patch_grad(mag_tensor_t *dst, mag_tensor_t *grad) {
    if (dst->au_state->grad)
        mag_tensor_decref(dst->au_state->grad);
    grad->flags = (grad->flags|MAG_TFLAG_IS_GRAD)&~MAG_TFLAG_REQUIRES_GRAD;
    dst->au_state->grad = grad;
}

void mag_tensor_backward(mag_tensor_t *root) {
    mag_assert(root->flags & MAG_TFLAG_REQUIRES_GRAD, "Tensor must require grad to back-propagate");
    mag_assert(root->rank == 1 && root->numel == 1, "Tensor must be a scalar to back-propagate");
    mag_ctx_grad_recorder_stop(root->ctx);
    mag_tensor_array_t post_order;
    mag_tensor_array_init(&post_order);
    mag_toposort(root, &post_order);
    if (mag_unlikely(!post_order.size)) goto end;
    for (size_t i=0, j = post_order.size-1; i < j; ++i, --j)
        mag_swap(mag_tensor_t *, post_order.data[i], post_order.data[j]);
    for (size_t id=0; id < post_order.size; ++id) {
        mag_tensor_t *child = post_order.data[id];
        mag_assert(child && child->au_state, "Autodiff state not allocated for tensor that requires gradient");
        const mag_opmeta_t *meta = mag_op_meta_of(child->au_state->op);
        if (!child->au_state->grad) {
            mag_tensor_t *grad = mag_tensor_full_like(child, 1.0f);
            mag_tensor_patch_grad(child, grad);
        }
        if (mag_unlikely(child->au_state->op == MAG_OP_NOP)) continue;
        mag_tensor_t *grads[MAG_MAX_OP_INPUTS] = {0};
        void (*backward)(mag_au_state_t *, mag_tensor_t **) = meta->backward;
        mag_assert2(backward);
        (*backward)(child->au_state, grads);
        uint32_t numin = meta->in;
        mag_assert2(numin <= MAG_MAX_OP_INPUTS);
        for (uint32_t i=0; i < numin; ++i) {
            mag_tensor_t *input = child->au_state->op_inputs[i];
            mag_assert2(input);
            if (!(input->flags & MAG_TFLAG_REQUIRES_GRAD)) continue;
            mag_tensor_t *gri = grads[i];
            mag_assert(gri, "Gradient for op %s, input #%d is not computed", meta->mnemonic, i);
            if (!input->au_state->grad) {
                mag_tensor_patch_grad(input, gri);
            } else {
                mag_tensor_t *acc = mag_add(gri, input->au_state->grad);
                mag_tensor_patch_grad(input, acc);
                mag_tensor_decref(gri);
            }
        }
    }
    mag_tensor_array_free(&post_order);
end:
    mag_ctx_grad_recorder_start(root->ctx);
}

void mag_tensor_zero_grad(mag_tensor_t *t) {
    if (t->flags & MAG_TFLAG_REQUIRES_GRAD && t->au_state && t->au_state->grad)
        mag_tensor_fill_float(t->au_state->grad, 0.0f);
}
