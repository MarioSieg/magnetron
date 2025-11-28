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
#include "mag_context.h"

/*
** ###################################################################################################################
** Operator Validation Helpers
** ###################################################################################################################
*/

static bool mag_op_requires_op_params(mag_opcode_t op) { /* Returns true if the op requires any op params and thus requires validation of them. */
    const mag_opmeta_t *meta = mag_op_meta_of(op);
    for (int i=0; i < MAG_MAX_OP_PARAMS; ++i) {
        if (meta->op_attr_types[i] != MAG_OP_ATTR_TYPE_EMPTY) {
            return true;
        }
    }
    return false;
}

static void mag_assert_correct_op_data(
    mag_opcode_t op,
    mag_tensor_t **in,
    uint32_t num_in,
    mag_tensor_t **out,
    uint32_t num_out,
    const mag_op_attr_t *op_params,
    uint32_t num_op_params
) {
    mag_assert(op != MAG_OP_NOP, "invalid operation: %d", op);
    const mag_opmeta_t *meta = mag_op_meta_of(op);

    /* Check input/output tensors */
    if (meta->in) mag_assert(in != NULL, "input tensors for operation '%s' are NULL", meta->mnemonic);
    if (meta->out) mag_assert(out != NULL, "output tensors for operation '%s' are NULL", meta->mnemonic);
    if (meta->in != MAG_OP_INOUT_DYN) {
        mag_assert(meta->in == num_in, "invalid number of input tensors for operation '%s': %u != %u", meta->mnemonic, num_in, meta->in);
        mag_assert(meta->out == num_out, "invalid number of output tensors for operation '%s': %u != %u", meta->mnemonic, num_out, meta->out);
    }
    for (uint32_t i=0; i < num_in; ++i)
        mag_assert(in[i] != NULL, "input tensor %u for operation '%s' is NULL", i, meta->mnemonic);
    for (uint32_t i=0; i < num_out; ++i)
        mag_assert(out[i] != NULL, "output tensor %u for operation '%s' is NULL", i, meta->mnemonic);

    /* Check op params if required */
    if (mag_op_requires_op_params(op)) {
        mag_assert(op_params != NULL, "operation '%s' requires operation parameters, but none were provided", meta->mnemonic);
        mag_assert(num_op_params <= MAG_MAX_OP_PARAMS, "too many operation parameters for operation '%s': %u > %u", meta->mnemonic, num_op_params, MAG_MAX_OP_PARAMS);
        for (uint32_t i=0; i < num_op_params; ++i) {
            if (meta->op_attr_types[i] != MAG_OP_ATTR_TYPE_EMPTY) { /* Only check for type equality if op param is required */
                mag_assert(op_params[i].tag == meta->op_attr_types[i], "invalid operation parameter type for operation '%s': %d != %d", meta->mnemonic, op_params[i].tag, meta->op_attr_types[i]);
            }
        }
    }
}

extern void mag_tensor_detach_inplace(mag_tensor_t *target);
static void mag_bump_version(mag_tensor_t *t) {
    if (t->flags & MAG_TFLAG_IS_VIEW) /* If this is a view, bump the version of the base tensor */
        t = t->view_meta->base;
    ++t->version;
}

static mag_status_t mag_tensor_strided_view(mag_tensor_t **out, mag_tensor_t *base) {
    return mag_tensor_as_strided(out, base->ctx, base, base->coords.rank, base->coords.shape, base->coords.strides, base->storage_offset);
}

/* Execute an operator on the active compute device and return result tensor. */
static void MAG_HOTPROC mag_dispatch(mag_opcode_t op, bool inplace, const mag_op_attr_registry_t *layout, mag_tensor_t **in, uint32_t num_in, mag_tensor_t **out, uint32_t num_out) {
    const mag_opmeta_t *meta = mag_op_meta_of(op);
    mag_assert2((in && num_in) || (out && num_out));
    mag_assert2(op != MAG_OP_NOP);
    mag_context_t *ctx = in ? (*in)->ctx : (*out)->ctx;
    const mag_op_attr_t *params = layout ? layout->slots : NULL;
    uint32_t num_params = layout ? layout->count : 0;
    mag_assert_correct_op_data(op, in, num_in, out, num_out, params, num_params);
    inplace &= !!(meta->flags & MAG_OP_FLAG_SUPPORTS_INPLACE);
    bool rec_grads = !!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && meta->backward;
    for (uint32_t i=0; i < num_out; ++i) { /* Populate autodiff state and handle gradient tracking */
        mag_tensor_t *r = out[i];
        mag_au_state_t *au = mag_au_state_lazy_alloc(&r->au_state, r->ctx);
        au->op = op;
        for (uint32_t j=0; j < num_in; ++j) {
            mag_tensor_t *input = in[j];
            au->op_inputs[j] = input;
            if (rec_grads) {
                if (input->flags & MAG_TFLAG_REQUIRES_GRAD && !(r->flags & MAG_TFLAG_REQUIRES_GRAD)) /* If any input requires grad, the output must also require grad*/
                    mag_tensor_set_requires_grad(r, true);
                mag_rc_incref(input); /* Keep input alive for the backward pass. */
            }
        }
        if (params) memcpy(au->op_attrs, params, num_params*sizeof(*params));
    }
    mag_command_t cmd = {
        .op = op,
        .in = in,
        .out = out,
        .num_in = num_in,
        .num_out = num_out,
    };
    if (params) memcpy(cmd.attrs, params, num_params*sizeof(*params));
    void (*submit)(mag_device_t *, const mag_command_t *) = ctx->device->submit;
    (*submit)(ctx->device, &cmd);
    for (uint32_t i=0; i < num_out; ++i) {
        if (inplace) mag_bump_version(out[i]);   /* Result aliases the modified storage */
        if (!rec_grads) mag_tensor_detach_inplace(out[i]); /* If gradient are not recorded, detach the tensor's parents (clear parent and opcode). TODO: why are we doing this? */
    }
}

static void mag_assert_dtype_compat(mag_opcode_t op, mag_tensor_t **inputs) {
    const mag_opmeta_t *meta = mag_op_meta_of(op);
    for (uint32_t i=0; i < meta->in; ++i) { /* Check that the input data types are supported by the operator. */
        bool supported = meta->dtype_mask & mag_dtype_bit(inputs[i]->dtype);
        if (mag_unlikely(!supported)) {
            const char *dtype = mag_dtype_meta_of(inputs[i]->dtype)->name;
            mag_panic(
                "Data type '%s' is not supported by operator '%s'.\n"
                "    Hint: Use a different data type or operator.\n",
                dtype, meta->mnemonic
            );
        }
    }
    if (mag_unlikely(meta->in == 2 && inputs[0]->dtype != inputs[1]->dtype)) { /* For binary operators, check that both inputs have the same data type. */
        const char *dtype_x = mag_dtype_meta_of(inputs[0]->dtype)->name;
        const char *dtype_y = mag_dtype_meta_of(inputs[1]->dtype)->name;
        mag_panic(
            "Data types of inputs for operator '%s' must match, but are '%s' and '%s'.\n"
            "    Hint: Use the same data type for both inputs.\n",
            meta->mnemonic, dtype_x, dtype_y
        );
    }
}

static void mag_assert_inplace_and_grad_mode_off(const mag_tensor_t *result) {
    if (mag_unlikely((result->ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && (result->flags & MAG_TFLAG_REQUIRES_GRAD))) {
        mag_panic(
            "Inplace operation on tensor with gradient tracking enabled is not allowed.\n"
            "    Hint: Disable gradient tracking or use a non-inplace operation.\n"
        );
    }
}

mag_status_t mag_tensor_empty(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape) {
    return mag_tensor_new(out, ctx, type, rank, shape);
}

mag_status_t mag_tensor_empty_like(mag_tensor_t **out, mag_tensor_t *isomorph) {
    return mag_tensor_new(out, isomorph->ctx, isomorph->dtype, isomorph->coords.rank, isomorph->coords.shape);
}

mag_status_t mag_tensor_empty_scalar(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type) {
    return mag_tensor_empty(out, ctx, type, 1, (int64_t[1]){1});
}

mag_status_t mag_tensor_scalar(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, mag_e8m23_t value) {
    mag_status_t stat = mag_tensor_empty_scalar(out, ctx, type);
    if (mag_iserr(stat)) return stat;
    mag_tensor_fill_float(*out, value);
    return MAG_STATUS_OK;
}

mag_status_t mag_tensor_full(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_e8m23_t value) {
    mag_status_t stat = mag_tensor_empty(out, ctx, type, rank, shape);
    if (mag_iserr(stat)) return stat;
    mag_tensor_fill_float(*out, value);
    return MAG_STATUS_OK;
}

mag_status_t mag_tensor_full_like(mag_tensor_t **out, mag_tensor_t *isomorph, mag_e8m23_t value) {
    mag_status_t stat = mag_tensor_empty_like(out, isomorph);
    if (mag_iserr(stat)) return stat;
    mag_tensor_fill_float(*out, value);
    return MAG_STATUS_OK;
}

mag_status_t mag_tensor_arange(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, mag_e8m23_t start, mag_e8m23_t end, mag_e8m23_t step) {
    *out = NULL;
    mag_tensor_t *result;
    int64_t numel = (int64_t)((end - start + step - 1)/step);
    mag_status_t stat = mag_tensor_empty(&result, ctx, type, 1, &numel);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(start));
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(step));
    mag_dispatch(MAG_OP_ARANGE, false, &layout, NULL, 0, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_tensor_rand_perm(mag_tensor_t **out, mag_context_t *ctx, mag_dtype_t type, int64_t n) {
    *out = NULL;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_dtype_bit(type) & MAG_DTYPE_MASK_INTEGER, "Data type must be integer");
    mag_tensor_t *result;
    mag_status_t stat = mag_tensor_empty(&result, ctx, type, 1, &n);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_dispatch(MAG_OP_RAND_PERM, false, NULL, NULL, 0, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_clone(mag_tensor_t **out, mag_tensor_t *x) {
    *out = NULL;
    mag_tensor_t *result;
    mag_status_t stat = mag_tensor_empty_like(&result, x);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_dispatch(MAG_OP_CLONE, false, NULL, &x, 1, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_cast(mag_tensor_t **out, mag_tensor_t *x, mag_dtype_t dst_type) {
    if (x->dtype == dst_type) return mag_clone(out, x); /* If dtypes match, we just clone */
    *out = NULL;
    mag_tensor_t *result;
    mag_status_t stat = mag_tensor_empty(&result, x->ctx, dst_type, x->coords.rank, x->coords.shape);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
        mag_dispatch(MAG_OP_CAST, false, NULL, &x, 1, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_view(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "Invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    if (rank <= 0) {
        stat = mag_tensor_as_strided(&result, x->ctx, x, x->coords.rank, x->coords.shape, x->coords.strides, x->storage_offset);
        if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    } else {
        mag_contract(ctx, ERR_INVALID_PARAM, {}, dims != NULL, "Dims cannot be NULL if rank > 0");
        int64_t new_dims[MAG_MAX_DIMS];
        for (int64_t i=0; i < rank; ++i) new_dims[i] = dims[i];
        int64_t shape[MAG_MAX_DIMS];
        mag_contract(ctx, ERR_INVALID_DIM, {}, mag_infer_missing_dim(&shape, new_dims, rank, x->numel), "Cannot infer missing dimension for view");
        int64_t strides[MAG_MAX_DIMS];
        if (rank == x->coords.rank && !memcmp(shape, x->coords.shape, rank*sizeof(*shape))) { /* Stride strategy: same shape as base */
            memcpy(strides, x->coords.strides, rank*sizeof(*shape));
        } else if (rank == x->coords.rank+1 && shape[rank-2]*shape[rank-1] == x->coords.shape[x->coords.rank-1]) { /* Stride strategy: last dim only */
            memcpy(strides, x->coords.strides, (rank-2)*sizeof(*strides));
            strides[rank-2] = x->coords.strides[x->coords.rank-1]*shape[rank-1];
            strides[rank-1] = x->coords.strides[x->coords.rank-1];
        } else if (mag_tensor_is_contiguous(x)) { /* Stride strategy: contiguous row-major */
            strides[rank-1] = 1;
            for (int64_t d = rank-2; d >= 0; --d) {
                mag_contract(ctx, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[d+1], strides[d+1], strides+d), "Dimension overflow when calculating strides for view");
            }
        } else { /* Stride strategy: solve generic strides */
            mag_contract(ctx, ERR_STRIDE_SOLVER_FAILED, {}, mag_solve_view_strides(&strides, x->coords.shape, x->coords.strides, x->coords.rank, shape, rank),
               "Tensor is not contiguous enough to be viewed\n"
               "Consider calling contiguous() or reshape() instead"
            );
        }
        stat = mag_tensor_as_strided(&result, x->ctx, x, rank, shape, strides, x->storage_offset);
        if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    }

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(rank));
    if (dims)
        for (int64_t i=0; i < rank; ++i)
            mag_op_attr_registry_insert(&layout, mag_op_attr_i64(dims[i]));

    mag_dispatch(MAG_OP_VIEW, false, &layout, &x, 1, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_reshape(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    int64_t shape[MAG_MAX_DIMS];
    mag_contract(ctx, ERR_INVALID_DIM, {}, mag_infer_missing_dim(&shape, dims, rank, x->numel), "Cannot infer missing dimension for reshape");
    if (x->coords.rank == rank && !memcmp(x->coords.shape, shape, sizeof(*dims)*rank)) {
        mag_rc_incref(x);
        *out = x;
        return MAG_STATUS_OK;
    }
    if (mag_tensor_is_contiguous(x)) {
        int64_t strides[MAG_MAX_DIMS];
        strides[rank-1] = 1;
        for (int64_t d=rank-2; d >= 0; --d) {
            mag_contract(ctx, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[d+1], strides[d+1], strides+d), "Dimension overflow when calculating strides for reshape")
        }
        stat = mag_tensor_as_strided(&result, x->ctx, x, rank, shape, strides, x->storage_offset);
        if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
        *out = result;
        return MAG_STATUS_OK;
    }
    if (mag_tensor_can_view(x, shape, rank)) {
        stat = mag_view(&result, x, shape, rank);
        if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
        *out = result;
        return MAG_STATUS_OK;
    }
    stat = mag_contiguous(&result, x);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    int64_t strides[MAG_MAX_DIMS];
    strides[rank-1] = 1;
    for (int64_t d = rank-2; d >= 0; --d)
        mag_assert2(!mag_mulov64(shape[d+1], strides[d+1], strides+d));
    mag_tensor_t *reshaped;
    stat = mag_tensor_as_strided(&reshaped, result->ctx, result, rank, shape, strides, result->storage_offset);
    if (mag_unlikely(stat != MAG_STATUS_OK)) {
        mag_rc_decref(result);
        return stat;
    }
    mag_rc_decref(result);
    *out = reshaped;
    return MAG_STATUS_OK;
}

mag_status_t mag_view_slice(mag_tensor_t **out, mag_tensor_t *x, int64_t dim, int64_t start, int64_t len, int64_t step) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_contract(ctx, ERR_INVALID_RANK, {}, dim >= 0 && dim < x->coords.rank, "Dim %" PRIi64 " out of range for rank %" PRIi64, dim, x->coords.rank);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, step != 0, "Slice step must be != 0");
    int64_t sz = x->coords.shape[dim];
    int64_t stop;
    if (step > 0) {
        if (start < 0) start += sz;
        stop = len < 0 ? sz : start + len*step;
        int64_t last = start + (len - 1)*step;
        mag_contract(ctx, ERR_INVALID_PARAM, {}, 0 <= start && start < sz, "Slice start out of bounds for dim %" PRIi64 ": %" PRIi64 " >= %" PRIi64, dim, start, sz);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, stop >= start, "Slice stop < start with %" PRIi64 " < %" PRIi64, stop, start);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, last < sz, "Slice exceeds bounds for dim %" PRIi64 ": last index %" PRIi64 " >= %" PRIi64, dim, last, sz);
    } else {
        step = (int64_t)(~(uint64_t)step+1);
        if (start < 0) start += sz;
        stop = len < 0 ? -1 : start - len*step;
        mag_contract(ctx, ERR_INVALID_PARAM, {}, 0 <= start && start < sz, "Slice start out of bounds");
        mag_contract(ctx, ERR_INVALID_PARAM, {}, stop < start, "Slice stop >= start with negative step");
        mag_contract(ctx, ERR_INVALID_PARAM, {}, stop >= -1, "Slice exceeds bounds (neg)");
    }
    if (len < 0) len = step > 0 ? (stop - start + step - 1)/step : (start - stop + step - 1)/step;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, len > 0, "Slice length is 0");
    int64_t shape[MAG_MAX_DIMS];
    int64_t strides[MAG_MAX_DIMS];
    memcpy(shape, x->coords.shape, sizeof(shape));
    memcpy(strides, x->coords.strides, sizeof(strides));
    shape[dim] = len;
    strides[dim] = x->coords.strides[dim]*step;
    int64_t tmp[MAG_MAX_DIMS];
    if (mag_solve_view_strides(&tmp, shape, strides, x->coords.rank, shape, x->coords.rank))
        memcpy(strides, tmp, sizeof(tmp));
    int64_t offset = x->storage_offset + start*x->coords.strides[dim];
    return mag_tensor_as_strided(out, x->ctx, x, x->coords.rank, shape, strides, offset);
}

mag_status_t mag_transpose(mag_tensor_t **out, mag_tensor_t *x, int64_t dim1, int64_t dim2) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, x->coords.rank >= 2, "Transpose requires rank >= 2, but got: %" PRIi64, x->coords.rank);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, dim1 != dim2, "Transposition axes must be unequal, but: %" PRIi64 " = %" PRIi64, dim1, dim2);
    int64_t ra = x->coords.rank;
    int64_t ax0 = dim1;
    int64_t ax1 = dim2;
    if (ax0 < 0) ax0 += ra;
    if (ax1 < 0) ax1 += ra;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, ax0 >= 0 && ax0 < ra, "Invalid transposition axis: %" PRIi64, dim1);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, ax1 >= 0 && ax1 < ra, "Invalid transposition axis: %" PRIi64, dim2);
    int64_t shape[MAG_MAX_DIMS];
    int64_t stride[MAG_MAX_DIMS];
    memcpy(shape, x->coords.shape, sizeof shape);
    memcpy(stride, x->coords.strides, sizeof stride);
    mag_swap(int64_t, shape[ax0], shape[ax1]);
    mag_swap(int64_t, stride[ax0], stride[ax1]);
    stat = mag_tensor_as_strided(&result, x->ctx, x, x->coords.rank, shape, stride, x->storage_offset);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(ax0));
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(ax1));
    mag_dispatch(MAG_OP_TRANSPOSE, false, &layout, &x, 1, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_permute(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "Invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    int64_t axes[MAG_MAX_DIMS];
    for (int64_t i=0; i < rank; ++i) axes[i] = dims[i];
    for (int64_t i=0; i < rank; ++i) {
        for (int64_t j = i+1; j < rank; ++j) {
            mag_contract(ctx, ERR_INVALID_PARAM, {}, axes[i] != axes[j], "Duplicated permutation axis: %" PRIi64 " == %" PRIi64, axes[i], axes[j]);
        }
    }
    int64_t shape[MAG_MAX_DIMS];
    int64_t stride[MAG_MAX_DIMS];
    for (int64_t i=0; i < rank; ++i) {
        shape[i] = x->coords.shape[axes[i]];
        stride[i] = x->coords.strides[axes[i]];
    }
    stat = mag_tensor_as_strided(&result, x->ctx, x, x->coords.rank, shape, stride, x->storage_offset);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_dispatch(MAG_OP_PERMUTE, false, NULL, &x, 1, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_contiguous(mag_tensor_t **out, mag_tensor_t *x) {
    if (!x->storage_offset && mag_tensor_is_contiguous(x)) {
        mag_rc_incref(x); /* If already contiguous, just incref */
        *out = x;
        return MAG_STATUS_OK;
    }
    return mag_clone(out, x);
}

static int mag_cmp_axis(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a;
    int64_t y = *(const int64_t *)b;
    return (x>y) - (x<y);
}

static mag_status_t mag_op_stub_reduction(mag_tensor_t **out, mag_opcode_t op, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_RANK, {}, dims != NULL || rank == 0, "Either dims must be non-NULL or rank must be 0");
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "Invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, x->coords.rank >= rank, "Cannot reduce over more dimensions than tensor has: rank=%" PRIi64 ", dims=%" PRIi64, x->coords.rank, rank);
    int64_t ax[MAG_MAX_DIMS];
    if (!dims && !rank) {
        rank = x->coords.rank;
        for (int64_t i=0; i < rank; ++i) ax[i] = i;
        dims = ax;
    } else if (dims) {
        for (int64_t i=0; i<rank; ++i) {
            int64_t a = dims[i];
            if (a < 0) a += x->coords.rank;
            mag_contract(ctx, ERR_INVALID_DIM, {}, 0 <= a && a < x->coords.rank, "Axis out of bounds: %" PRIi64 " for rank %" PRIi64, a, x->coords.rank);
            ax[i] = a;
        }
        qsort(ax, (size_t)rank, sizeof(int64_t), &mag_cmp_axis);
        int64_t r=0;
        for (int64_t i=0; i < rank; ++i)
            if (!i || ax[i] != ax[i-1])
                ax[r++] = ax[i];
        rank = r;
        dims = ax;
    }
    int64_t xrank = x->coords.rank;
    int64_t prev = -1;
    for (int64_t i=0; i < rank; ++i) {
        int64_t a = dims[i];
        mag_contract(ctx, ERR_INVALID_DIM, {}, 0 <= a && a < xrank, "Axis out of bounds: %" PRIi64 " for rank %" PRIi64, a, xrank);
        mag_contract(ctx, ERR_INVALID_DIM, {}, a > prev, "Axes must be strictly increasing and unique");
        prev = a;
    }
    int64_t shape[MAG_MAX_DIMS] = {0};
    int64_t j=0, k=0;
    for (int64_t d=0; d < xrank; ++d) {
        if (k < rank && dims[k] == d) {
            if (keepdim) shape[j++] = 1;
            ++k;
        }
        else shape[j++] = x->coords.shape[d];
    }
    int64_t orank = keepdim ? xrank : xrank - rank;
    stat = !keepdim && !orank ? mag_tensor_empty_scalar(&result, x->ctx, x->dtype) : mag_tensor_empty(&result, x->ctx, x->dtype, orank, shape);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(rank));
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(!!keepdim));
    for (int64_t i=0; i<rank; ++i)
        mag_op_attr_registry_insert(&layout, mag_op_attr_i64(dims[i]));
    mag_dispatch(op, false, &layout, &x, 1, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_mean(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out, MAG_OP_MEAN, x, dims, rank, keepdim);
}

mag_status_t mag_min(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out, MAG_OP_MIN, x, dims, rank, keepdim);
}

mag_status_t mag_max(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out, MAG_OP_MAX, x, dims, rank, keepdim);
}

mag_status_t mag_sum(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out, MAG_OP_SUM, x, dims, rank, keepdim);
}

mag_status_t mag_argmin(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    mag_panic("Not implemented yet");
    return mag_tensor_empty_like(out, x);
}

mag_status_t mag_argmax(mag_tensor_t **out, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    mag_panic("Not implemented yet");
    return mag_tensor_empty_like(out, x);
}

static mag_status_t mag_op_stub_unary(mag_tensor_t **out, mag_opcode_t op, mag_tensor_t *x, const mag_op_attr_registry_t *layout, bool inplace) {
    *out = NULL;
    mag_assert_dtype_compat(op, &x);
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    if (inplace) {
        stat = mag_tensor_strided_view(&result, x); /* Use the same storage as x */
        if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
        mag_assert_inplace_and_grad_mode_off(x);
    } else {
        stat = mag_tensor_empty_like(&result, x); /* Allocate a new tensor for the result */
        if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    }
    mag_dispatch(op, inplace, layout, &x, 1, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

#define mag_impl_unary_pair(name, op) \
    mag_status_t mag_##name(mag_tensor_t **out, mag_tensor_t* x) { return mag_op_stub_unary(out, MAG_OP_##op, x, NULL, false); } \
    mag_status_t mag_##name##_(mag_tensor_t **out, mag_tensor_t* x) { return mag_op_stub_unary(out, MAG_OP_##op, x, NULL, true); }

mag_impl_unary_pair(not, NOT)
mag_impl_unary_pair(abs, ABS)
mag_impl_unary_pair(sgn, SGN)
mag_impl_unary_pair(neg, NEG)
mag_impl_unary_pair(log, LOG)
mag_impl_unary_pair(log10, LOG10)
mag_impl_unary_pair(log1p, LOG1P)
mag_impl_unary_pair(log2, LOG2)
mag_impl_unary_pair(sqr, SQR)
mag_impl_unary_pair(sqrt, SQRT)
mag_impl_unary_pair(sin, SIN)
mag_impl_unary_pair(cos, COS)
mag_impl_unary_pair(tan, TAN)
mag_impl_unary_pair(sinh, SINH)
mag_impl_unary_pair(cosh, COSH)
mag_impl_unary_pair(tanh, TANH)
mag_impl_unary_pair(asin, ASIN)
mag_impl_unary_pair(acos, ACOS)
mag_impl_unary_pair(atan, ATAN)
mag_impl_unary_pair(asinh, ASINH)
mag_impl_unary_pair(acosh, ACOSH)
mag_impl_unary_pair(atanh, ATANH)
mag_impl_unary_pair(step, STEP)
mag_impl_unary_pair(erf, ERF)
mag_impl_unary_pair(erfc, ERFC)
mag_impl_unary_pair(exp, EXP)
mag_impl_unary_pair(exp2, EXP2)
mag_impl_unary_pair(expm1, EXPM1)
mag_impl_unary_pair(floor, FLOOR)
mag_impl_unary_pair(ceil, CEIL)
mag_impl_unary_pair(round, ROUND)
mag_impl_unary_pair(trunc, TRUNC)
mag_impl_unary_pair(softmax, SOFTMAX)
mag_impl_unary_pair(softmax_dv, SOFTMAX_DV)
mag_impl_unary_pair(sigmoid, SIGMOID)
mag_impl_unary_pair(sigmoid_dv, SIGMOID_DV)
mag_impl_unary_pair(hard_sigmoid, HARD_SIGMOID)
mag_impl_unary_pair(silu, SILU)
mag_impl_unary_pair(silu_dv, SILU_DV)
mag_impl_unary_pair(tanh_dv, TANH_DV)
mag_impl_unary_pair(relu, RELU)
mag_impl_unary_pair(relu_dv, RELU_DV)
mag_impl_unary_pair(gelu, GELU)
mag_impl_unary_pair(gelu_approx, GELU_APPROX)
mag_impl_unary_pair(gelu_dv, GELU_DV)

#undef mag_impl_unary_pair

mag_status_t mag_tril(mag_tensor_t **out, mag_tensor_t *x, int32_t diag) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, x->coords.rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, x->coords.rank);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(diag));
    return mag_op_stub_unary(out, MAG_OP_TRIL, x, &layout, false);
}

mag_status_t mag_tril_(mag_tensor_t **out, mag_tensor_t *x, int32_t diag) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, x->coords.rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, x->coords.rank);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(diag));
    return mag_op_stub_unary(out, MAG_OP_TRIL, x, &layout, true);
}

mag_status_t mag_triu(mag_tensor_t **out, mag_tensor_t *x, int32_t diag) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, x->coords.rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, x->coords.rank);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(diag));
    return mag_op_stub_unary(out, MAG_OP_TRIU, x, &layout, false);
}

mag_status_t mag_triu_(mag_tensor_t **out, mag_tensor_t *x, int32_t diag) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, x->coords.rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, x->coords.rank);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(diag));
    return mag_op_stub_unary(out, MAG_OP_TRIU, x, &layout, true);
}

mag_status_t mag_multinomial(mag_tensor_t **out, mag_tensor_t *x, int64_t num_samples, bool replacement) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, x->coords.rank == 1 || x->coords.rank == 2, "Multinomial dist requires rank 1 or 2, but got: %" PRIi64, x->coords.rank);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_contiguous(x), "Input tensor must be contiguous row-major");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, num_samples > 0, "Number of samples must be > 0, but got: %" PRIi64, num_samples);
    mag_assert_dtype_compat(MAG_OP_MULTINOMIAL, &x);
    int64_t shape[MAG_MAX_DIMS] = {0};
    if (x->coords.rank > 1) memcpy(shape, x->coords.shape, (x->coords.rank - 1)*sizeof(*shape));
    shape[x->coords.rank-1] = num_samples;
    mag_tensor_t *result;
    mag_status_t stat = mag_tensor_new(&result, x->ctx, MAG_DTYPE_I64, x->coords.rank, shape);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(num_samples));
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(!!replacement));
    mag_dispatch(MAG_OP_MULTINOMIAL, false, &layout, &x, 1, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_cat(mag_tensor_t **out, mag_tensor_t **tensors, size_t count, int64_t dim) {
    *out = NULL;
    mag_assert(tensors && *tensors, "Tensors array cannot be NULL or contain NULL elements"); /* TODO: Use contract */
    mag_context_t *ctx = (*tensors)->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, count > 0, "Tensor count must be > 0");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, dim >= 0 && dim < MAG_MAX_DIMS, "Dim must be in [0, %d), but got: %" PRIi64, MAG_MAX_DIMS, dim);
    mag_tensor_t *t0 = tensors[0];
    mag_contract(ctx, ERR_INVALID_PARAM, {}, t0 != NULL, "First tensor cannot be NULL");
    int64_t rank = t0->coords.rank;
    mag_contract(ctx, ERR_INVALID_DIM, {}, rank > 0 && dim < rank, "Concat dim must be in [0, %" PRIi64 "), but got: %" PRIi64, rank, dim);
    mag_dtype_t dtype = t0->dtype;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_contiguous(t0), "Inputs must be contiguous row-major");
    int64_t shape[MAG_MAX_DIMS];
    for (int64_t d=0; d < rank; ++d)
        shape[d] = t0->coords.shape[d];
    shape[dim] = 0;
    for (size_t i = 0; i < count; ++i) {
        mag_tensor_t *ti = tensors[i];
        mag_contract(ctx, ERR_INVALID_PARAM, {}, ti != NULL, "Tensor %" PRIu64 " cannot be NULL", (uint64_t)i);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, ti->coords.rank == rank, "All tensors must have same rank (%" PRIi64 " != %" PRIi64 ")", ti->coords.rank, rank);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, ti->dtype == dtype, "All tensors must have same dtype (%d != %d)", ti->dtype, dtype);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_contiguous(ti), "All tensors must be contiguous row-major");
        for (int64_t d=0; d < rank; ++d) {
            if (d == dim) continue;
            mag_contract(ctx, ERR_INVALID_PARAM, {}, ti->coords.shape[d] == t0->coords.shape[d], "Shapes must match on non-concat dims (dim=%" PRIi64 " mismatch on axis %" PRIi64 ")", dim, d);
        }
        shape[dim] += ti->coords.shape[dim];
    }
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(dim));
    stat = mag_tensor_new(&result, t0->ctx, dtype, rank, shape);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_dispatch(MAG_OP_CAT, false, &layout, tensors, count, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

enum {
    MAG_BINOP_NONE = 0,
    MAG_BINOP_LOGICAL = 1<<0,
    MAG_BINOP_INPLACE = 1<<1
};

static mag_status_t mag_op_stub_binary(mag_tensor_t **out, mag_opcode_t op, mag_tensor_t *x, mag_tensor_t *y, int flags) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_assert_dtype_compat(op, (mag_tensor_t *[]) {x, y});
    if (flags & MAG_BINOP_INPLACE) {
        mag_assert2(!(flags & MAG_BINOP_LOGICAL));
        mag_assert_inplace_and_grad_mode_off(x);
        stat = mag_tensor_strided_view(&result, x);
        if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    } else {
        int64_t dims[MAG_MAX_DIMS];
        int64_t rank;
        if (mag_unlikely(!mag_coords_broadcast_shape(&x->coords, &y->coords, dims, &rank))) {
            char sx[MAG_FMT_DIM_BUF_SIZE];
            char sy[MAG_FMT_DIM_BUF_SIZE];
            mag_fmt_shape(&sx, &x->coords.shape, x->coords.rank);
            mag_fmt_shape(&sy, &y->coords.shape, y->coords.rank);
            mag_contract(ctx, ERR_BROADCAST_IMPOSSIBLE, {}, 0,
                "Cannot broadcast tensors with shapes %s and %s for operator '%s'.\n"
                "    Hint: Ensure that the shapes are compatible for broadcasting.\n",
                sx, sy, mag_op_meta_of(op)->mnemonic
            );
        }
        mag_dtype_t rtype = flags & MAG_BINOP_LOGICAL ? MAG_DTYPE_BOOL : x->dtype;
        stat = rank ? mag_tensor_empty(&result, x->ctx, rtype, rank, dims) : mag_tensor_empty_scalar(&result, x->ctx, rtype);
        if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    }
    mag_dispatch(op, flags & MAG_BINOP_INPLACE, NULL, (mag_tensor_t *[2]){x, y}, 2, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

#define mag_impl_binary_pair(name, op, logical) \
    mag_status_t mag_##name(mag_tensor_t **out, mag_tensor_t* x, mag_tensor_t* y) { return mag_op_stub_binary(out, MAG_OP_##op, x, y, logical ? MAG_BINOP_LOGICAL : 0); } \
    mag_status_t mag_##name##_(mag_tensor_t **out, mag_tensor_t* x, mag_tensor_t* y) { return mag_op_stub_binary(out, MAG_OP_##op, x, y, (logical ? MAG_BINOP_LOGICAL : 0)+MAG_BINOP_INPLACE); }

mag_impl_binary_pair(add, ADD, false)
mag_impl_binary_pair(sub, SUB, false)
mag_impl_binary_pair(mul, MUL, false)
mag_impl_binary_pair(div, DIV, false)
mag_impl_binary_pair(and, AND, false)
mag_impl_binary_pair(or, OR, false)
mag_impl_binary_pair(xor, XOR, false)
mag_impl_binary_pair(shl, SHL, false)
mag_impl_binary_pair(shr, SHR, false)
mag_impl_binary_pair(eq, EQ, true)
mag_impl_binary_pair(ne, NE, true)
mag_impl_binary_pair(le, LE, true)
mag_impl_binary_pair(ge, GE, true)
mag_impl_binary_pair(lt, LT, true)
mag_impl_binary_pair(gt, GT, true)

#undef mag_impl_binary_pair

mag_status_t mag_matmul(mag_tensor_t **out, mag_tensor_t *x, mag_tensor_t *y) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_assert_dtype_compat(MAG_OP_MATMUL, (mag_tensor_t *[]) {x, y});
    int64_t kx = x->coords.shape[x->coords.rank-1];
    int64_t ky = y->coords.rank == 1 ? *y->coords.shape : y->coords.rank == 2 && x->coords.rank == 1 ? *y->coords.shape : y->coords.shape[y->coords.rank-2];
    if (kx != ky) {
        char fmt_x[MAG_FMT_DIM_BUF_SIZE], fmt_y[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&fmt_x, &x->coords.shape, x->coords.rank);
        mag_fmt_shape(&fmt_y, &y->coords.shape, y->coords.rank);
        mag_contract(
            ctx, ERR_OPERATOR_IMPOSSIBLE, {}, 0,
            "Cannot perform matmul on tensors with shapes %s and %s: "
            "last dimension of first tensor (%" PRIi64 ") does not match second tensor (%" PRIi64 ").\n"
            "    Hint: Ensure that the last dimension of the first tensor matches the second-to-last dimension of the second tensor.\n",
            fmt_x, fmt_y, kx, ky
        );
    }
    /* verify broadcastability of batch dims */
    int64_t x_bd = x->coords.rank > 2 ? x->coords.rank-2 : 0;
    int64_t y_bd = y->coords.rank > 2 ? y->coords.rank-2 : 0;
    int64_t r_bd = x_bd > y_bd ? x_bd : y_bd;
    for (int64_t i=0; i < r_bd; ++i) {
        int64_t x_dim = i < r_bd-x_bd ? 1 : x->coords.shape[i-(r_bd-x_bd)];
        int64_t y_dim = i < r_bd-y_bd ? 1 : y->coords.shape[i-(r_bd-y_bd)];
        if (x_dim != y_dim && x_dim != 1 && y_dim != 1) {
            char fmt_x[MAG_FMT_DIM_BUF_SIZE], fmt_y[MAG_FMT_DIM_BUF_SIZE];
            mag_fmt_shape(&fmt_x, &x->coords.shape, x->coords.rank);
            mag_fmt_shape(&fmt_y, &y->coords.shape, y->coords.rank);
            mag_contract(
                ctx, ERR_OPERATOR_IMPOSSIBLE, {}, 0,
                "Cannot perform matmul on tensors with shapes %s and %s: "
                "batch dimensions at index %" PRIi64 " do not match (%" PRIi64 " != %" PRIi64 ").\n"
                "    Hint: Ensure that the batch dimensions are compatible for broadcasting.\n",
                fmt_x, fmt_y, i, x_dim, y_dim
            );
        }
    }
    if (x->coords.rank == 1 && y->coords.rank == 1) { /* (K)x(K) -> () */
        int64_t shape[1] = {1};
        stat = mag_tensor_new(&result, x->ctx, x->dtype, 1, shape);
    } else if (x->coords.rank == 1 && y->coords.rank == 2) { /* (K)x(K,N) -> (N) */
        int64_t shape[1] = {y->coords.shape[1]};
        stat = mag_tensor_new(&result, x->ctx, x->dtype, 1, shape);
    } else if (x->coords.rank == 2 && y->coords.rank == 1) { /* (M,K)x(K) -> (M) */
        int64_t shape[1] = {x->coords.shape[0]};
        stat = mag_tensor_new(&result, x->ctx, x->dtype, 1, shape);
    } else { /* Batched ND version */
        int64_t a_bd = x->coords.rank-2;
        int64_t b_bd = y->coords.rank-2;
        int64_t shape[MAG_MAX_DIMS] = {0};
        for (int64_t i=0; i < r_bd; ++i) {
            int64_t a_dim = i < r_bd-a_bd ? 1 : x->coords.shape[i-(r_bd-a_bd)];
            int64_t b_dim = i < r_bd-b_bd ? 1 : y->coords.shape[i-(r_bd-b_bd)];
            shape[i] = a_dim > b_dim ? a_dim : b_dim;
        }
        shape[r_bd] = x->coords.shape[x->coords.rank-2];
        shape[r_bd+1] = y->coords.shape[y->coords.rank-1];
        stat = mag_tensor_new(&result,x->ctx, x->dtype, r_bd+2, shape);
    }
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_dispatch(MAG_OP_MATMUL, false, false, (mag_tensor_t *[2]) {x, y}, 2, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_repeat_back(mag_tensor_t **out, mag_tensor_t *x, mag_tensor_t *y) {
    *out = NULL;
    mag_tensor_t *result = NULL;
    mag_assert_dtype_compat(MAG_OP_REPEAT_BACK, (mag_tensor_t *[]) {x, y});
    mag_status_t stat = mag_tensor_new(&result, x->ctx, x->dtype, y->coords.rank, y->coords.shape);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    /* TODO: Check for broadcastability of x and y */
    mag_dispatch(MAG_OP_REPEAT_BACK, false, NULL, (mag_tensor_t *[2]) {x, y}, 2, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_gather(mag_tensor_t **out, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx) {
    *out = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, idx->dtype == MAG_DTYPE_I64, "Index tensor must be of type: i64");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, dim >= 0 && dim < x->coords.rank, "Gather dim must be in [0, %" PRIi64 "), but got: %" PRIi64, x->coords.rank, dim);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, idx->coords.rank <= x->coords.rank, "Index tensor rank must be <= input tensor rank (%" PRIi64 " <= %" PRIi64")", idx->coords.rank, x->coords.rank);
    if (dim < 0) dim += x->coords.rank;
    mag_assert2(dim >= 0 && dim < x->coords.rank);
    int64_t ax[MAG_MAX_DIMS];
    int64_t ork = 0;
    bool full = false;
    if (idx->coords.rank == x->coords.rank) {
        full = true;
        for (int64_t d=0; d < x->coords.rank; ++d) {
            if (d == dim) continue;
            if (idx->coords.shape[d] != x->coords.shape[d]) {
                full = false;
                break;
            }
        }
    }
    if (full) {
        for (int64_t d = 0; d < x->coords.rank; ++d)
            ax[ork++] = idx->coords.shape[d];
    } else if (idx->coords.rank == 1) {
        for (int64_t d=0; d < x->coords.rank; ++d) {
            ax[ork++] = d == dim ? idx->coords.shape[0] : x->coords.shape[d];
        }
    } else {
        for (int64_t d = 0; d < dim; ++d) ax[ork++] = x->coords.shape[d];
        for (int64_t i=0; i < idx->coords.rank; ++i) ax[ork++] = idx->coords.shape[i];
        for (int64_t d=dim+1; d < x->coords.rank; ++d) ax[ork++] = x->coords.shape[d];
    }
    mag_contract(ctx, ERR_INVALID_RANK, {}, ork >= 1 && ork <= MAG_MAX_DIMS, "Gather output rank must be in [1, %d], but got: %" PRIi64, MAG_MAX_DIMS, ork);
    stat = mag_tensor_empty(&result, x->ctx, x->dtype, ork, ax);
    if (mag_unlikely(stat != MAG_STATUS_OK)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(dim)); /* Store dimension in op_params[0] */
    mag_dispatch(MAG_OP_GATHER, false, &layout, (mag_tensor_t *[2]) {x, idx}, 2, &result, 1);
    *out = result;
    return MAG_STATUS_OK;
}

void mag_tensor_fill_from_floats(mag_tensor_t *t, const mag_e8m23_t *data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_storage_buffer_t *sto = t->storage;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_H2D, 0, (void *)data, len*sizeof(*data), MAG_DTYPE_E8M23);
}

void mag_tensor_fill_from_raw_bytes(mag_tensor_t *t, const void *data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_storage_buffer_t *sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, 0, (void *)data, len);
}

void mag_tensor_fill_float(mag_tensor_t *t, mag_e8m23_t x) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(x));

    mag_dispatch(MAG_OP_FILL, false, &layout, NULL, 0, &t, 1);
}

void mag_tensor_fill_int(mag_tensor_t *t, int64_t x) {
    mag_assert2(mag_tensor_is_integral_typed(t));

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(x));

    mag_dispatch(MAG_OP_FILL, false, &layout, NULL, 0, &t, 1);
}

void mag_tensor_masked_fill_float(mag_tensor_t *t, mag_tensor_t *mask, mag_e8m23_t x) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    mag_assert2(mask->dtype == MAG_DTYPE_BOOL);

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(x));
    mag_op_attr_registry_insert(&layout, mag_op_attr_ptr(mask));

    mag_dispatch(MAG_OP_MASKED_FILL, false, &layout, NULL, 0, &t, 1);
}

void mag_tensor_masked_fill_int(mag_tensor_t *t, mag_tensor_t *mask, int64_t x) {
    mag_assert2(mag_tensor_is_integral_typed(t));
    mag_assert2(mask->dtype == MAG_DTYPE_BOOL);

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(x));
    mag_op_attr_registry_insert(&layout, mag_op_attr_ptr(mask));

    mag_dispatch(MAG_OP_MASKED_FILL, false, &layout, NULL, 0, &t, 1);
}

void mag_tensor_fill_random_uniform_float(mag_tensor_t *t, mag_e8m23_t min, mag_e8m23_t max) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(min));
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(max));

    mag_dispatch(MAG_OP_RAND_UNIFORM, false, &layout, NULL, 0, &t, 1);
}

void mag_tensor_fill_random_uniform_int(mag_tensor_t *t, int64_t min, int64_t max) {
    mag_assert2(mag_tensor_is_integral_typed(t));

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(min));
    mag_op_attr_registry_insert(&layout, mag_op_attr_i64(max));

    mag_dispatch(MAG_OP_RAND_UNIFORM, false, &layout, NULL, 0, &t, 1);
}

void mag_tensor_fill_random_normal(mag_tensor_t *t, mag_e8m23_t mean, mag_e8m23_t stddev) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(mean));
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(stddev));

    mag_dispatch(MAG_OP_RAND_NORMAL, false, &layout, NULL, 0, &t, 1);
}

void mag_tensor_fill_random_bernoulli(mag_tensor_t *t, mag_e8m23_t p) {
    mag_assert2(t->dtype == MAG_DTYPE_BOOL);

    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_e8m23(p));

    mag_dispatch(MAG_OP_RAND_BERNOULLI, false, &layout, NULL, 0, &t, 1);
}
