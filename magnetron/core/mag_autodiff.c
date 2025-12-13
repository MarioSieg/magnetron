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
        mag_rc_decref(au->grad);
        au->grad = NULL;
    }
    for (size_t i=0; i < sizeof(au->op_inputs)/sizeof(*au->op_inputs); ++i)
        if (au->op_inputs[i]) mag_rc_decref(au->op_inputs[i]);
    mag_fixed_pool_free_block(&au->ctx->au_state_pool, au);
}

mag_au_state_t *mag_au_state_lazy_alloc(mag_au_state_t **au_state, mag_context_t *ctx) {
    if (*au_state) return *au_state;
    *au_state = mag_fixed_pool_alloc_block(&ctx->au_state_pool);
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

mag_status_t mag_tensor_grad(const mag_tensor_t *tensor, mag_tensor_t **out_grad) {
    mag_contract(tensor->ctx, ERR_INVALID_PARAM, {}, tensor->flags & MAG_TFLAG_REQUIRES_GRAD, "Tensor does not require gradient");
    mag_contract(tensor->ctx, ERR_INVALID_STATE, {}, tensor->au_state, "Autodiff state missing for tensor");
    if (tensor->au_state->grad) mag_rc_incref(tensor->au_state->grad);
    *out_grad = tensor->au_state->grad;
    return MAG_STATUS_OK;
}

bool mag_tensor_requires_grad(const mag_tensor_t *tensor) {
    return tensor->flags & MAG_TFLAG_REQUIRES_GRAD;
}

mag_status_t mag_tensor_set_requires_grad(mag_tensor_t *tensor, bool requires_grad) {
    if (requires_grad) {
        mag_contract(tensor->ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_floating_point_typed(tensor), "Gradient tracking tensors must be floating-point typed, but tensor has dtype: %s", mag_type_trait(tensor->dtype)->name);
        tensor->flags |= MAG_TFLAG_REQUIRES_GRAD;
        mag_au_state_lazy_alloc(&tensor->au_state, tensor->ctx);
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

mag_status_t mag_tensor_backward(mag_tensor_t *root) {
    mag_context_t *ctx = root->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, root->flags & MAG_TFLAG_REQUIRES_GRAD, "Tensor must require gradient to perform backpropagation. Set requires_grad=True when creating the tensor");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, root->coords.rank == 0 && root->numel == 1, "Can only backpropagate from a scalar (rank 1, numel 1) tensor");
    mag_ctx_grad_recorder_stop(root->ctx);
    mag_topo_set_t post_order;
    mag_topo_set_init(&post_order);
    mag_topo_sort(root, &post_order);
    if (mag_unlikely(!post_order.size)) goto end;
    for (size_t i=0, j = post_order.size-1; i < j; ++i, --j) {
        mag_swap(mag_tensor_t *, post_order.data[i], post_order.data[j]);
    }
    for (size_t id=0; id < post_order.size; ++id) {
        mag_tensor_t *child = post_order.data[id];
        mag_contract(ctx, ERR_INVALID_STATE, { mag_topo_set_free(&post_order); }, child && child->au_state, "Autodiff state missing for tensor");
        const mag_op_traits_t *meta = mag_op_traits(child->au_state->op);
        if (!child->au_state->grad) {
            mag_tensor_t *grad;
            if (mag_iserr(mag_full_like(&grad, child, mag_scalar_float(1.0))))
                continue;
            mag_tensor_patch_grad(child, grad);
        }
        if (mag_unlikely(child->au_state->op == MAG_OP_NOP)) continue;
        mag_tensor_t *grads[MAG_MAX_OP_INPUTS] = {0};
        mag_status_t (*backward)(mag_au_state_t *, mag_tensor_t **) = meta->backward;
        mag_contract(ctx, ERR_INVALID_STATE, { mag_topo_set_free(&post_order); }, backward != NULL, "Backward function not implemented for op %s", meta->mnemonic);
        mag_status_t stat = (*backward)(child->au_state, grads);
        if (mag_unlikely(stat != MAG_STATUS_OK)) {
            mag_topo_set_free(&post_order);
            return stat;
        }
        uint32_t numin = meta->in;
        mag_assert(numin <= MAG_MAX_OP_INPUTS, "Invalid number of op inputs for op %s", meta->mnemonic); /* Should never happen (BUG) */
        for (uint32_t i=0; i < numin; ++i) {
            mag_tensor_t *input = child->au_state->op_inputs[i];
            mag_assert2(input);
            if (!(input->flags & MAG_TFLAG_REQUIRES_GRAD)) continue;
            mag_tensor_t *gri = grads[i];
            mag_assert(gri, "Gradient for op %s, input #%d is not computed", meta->mnemonic, i); /* Should never happen (BUG) */
            if (!input->au_state->grad) {
                mag_tensor_patch_grad(input, gri);
            } else {
                mag_tensor_t *acc;
                if (mag_iserr(mag_add(&acc, gri, input->au_state->grad))) {
                    mag_rc_decref(gri);
                    continue;
                }
                mag_tensor_patch_grad(input, acc);
                mag_rc_decref(gri);
            }
        }
    }
    mag_topo_set_free(&post_order);
end:
    mag_ctx_grad_recorder_start(root->ctx);
    return MAG_STATUS_OK;
}

void mag_tensor_zero_grad(mag_tensor_t *tensor) {
    if (tensor->flags & MAG_TFLAG_REQUIRES_GRAD && tensor->au_state && tensor->au_state->grad)
        mag_zero_(tensor->au_state->grad);
}
