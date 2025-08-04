/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
*/

#include "magnetron_internal.h"

/*
** ###################################################################################################################
** Operator Validation Helpers
** ###################################################################################################################
*/

static bool mag_op_requires_op_params(mag_opcode_t op) { /* Returns true if the op requires any op params and thus requires validation of them. */
    const mag_opmeta_t* meta = mag_op_meta_of(op);
    for (int i=0; i < MAG_MAX_OP_PARAMS; ++i) {
        if (meta->op_param_layout[i] != MAG_OPP_NONE) {
            return true;
        }
    }
    return false;
}

static void mag_assert_correct_op_data(
    mag_opcode_t op,
    mag_tensor_t** inputs,
    uint32_t num_inputs,
    const mag_opparam_t* op_params,
    uint32_t num_op_params
) {
    mag_assert(op != MAG_OP_NOP, "invalid operation: %d", op);
    const mag_opmeta_t* meta = mag_op_meta_of(op);

    /* Check input tensors */
    mag_assert(inputs != NULL, "input tensors for operation '%s' are NULL", meta->mnemonic);
    mag_assert(num_inputs <= MAG_MAX_OP_INPUTS, "too many input tensors for operation '%s': %u > %u", meta->mnemonic, num_inputs, MAG_MAX_OP_INPUTS);
    mag_assert(meta->input_count == num_inputs, "invalid number of input tensors for operation '%s': %u != %u", meta->mnemonic, num_inputs, meta->input_count);
    for (uint32_t i=0; i < meta->input_count; ++i) {
        mag_assert(inputs[i] != NULL, "input tensor %u for operation '%s' is NULL", i, meta->mnemonic);
    }

    /* Check op params if required */
    if (mag_op_requires_op_params(op)) {
        mag_assert(op_params != NULL, "operation '%s' requires operation parameters, but none were provided", meta->mnemonic);
        mag_assert(num_op_params <= MAG_MAX_OP_PARAMS, "too many operation parameters for operation '%s': %u > %u", meta->mnemonic, num_op_params, MAG_MAX_OP_PARAMS);
        for (uint32_t i=0; i < num_op_params; ++i) {
            if (meta->op_param_layout[i] != MAG_OPP_NONE) { /* Only check for type equality if op param is required */
                mag_assert(op_params[i].type == meta->op_param_layout[i],
                    "invalid operation parameter type for operation '%s': %d != %d",
                    meta->mnemonic, op_params[i].type, meta->op_param_layout[i]
                );
            }
        }
    }
}

/* Execute init/normal operator on R. */
static void MAG_HOTPROC mag_op_exec(mag_tensor_t* R, mag_idevice_t* dvc, mag_exec_stage_t stage) {
    void (*exec)(mag_idevice_t*, mag_tensor_t*) = stage == MAG_STAGE_INIT ? dvc->eager_exec_init : dvc->eager_exec_fwd;
    (*exec)(dvc, R); /* Dispatch to backend. */
}

extern void mag_tensor_detach_inplace(mag_tensor_t* target);
static void mag_bump_version(mag_tensor_t* t) {
    if (t->flags & MAG_TFLAG_IS_VIEW) /* If this is a view, bump the version of the base tensor */
        t = t->view_meta->base;
    ++t->version;
}

static mag_tensor_t* mag_tensor_strided_view(mag_tensor_t* base) {
    return mag_tensor_as_strided(base->ctx, base, base->rank, base->shape, base->strides, base->storage_offset);
}

/* Execute an operator on the active compute device and return result tensor. */
static mag_tensor_t* MAG_HOTPROC mag_dispatch(
    mag_tensor_t* r,
    mag_opcode_t op,
    bool inplace,
    mag_tensor_t** inputs,
    uint32_t num_inputs,
    const mag_op_param_layout_t* layout,
    mag_exec_stage_t stage
) {
    const mag_opparam_t* params = layout ? layout->slots : NULL;
    uint32_t num_params = layout ? layout->count : 0;
    mag_assert_correct_op_data(op, inputs, num_inputs, params, num_params);
    const mag_opmeta_t* meta = mag_op_meta_of(op);
    inplace &= !!(meta->flags & MAG_OP_FLAG_SUPPORTS_INPLACE);
    r->op = op; /* Set opcode of result. */
    bool is_recording_grads = !!(r->ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER);
    for (uint32_t i=0; i < num_inputs; ++i) {
        mag_tensor_t* input = inputs[i];
        r->op_inputs[i] = input;
        /* If gradient tracking is enabled, the result tensor inherits the input's gradient rules. */
        if (is_recording_grads) {
            r->flags |= input->flags & MAG_TFLAG_REQUIRES_GRAD; /* Set gradient tracking flag if set in input. */
            mag_tensor_incref(input);                                /* Keep input alive for the backward pass. */
        }
    }
    if (params) /* If available, copy operation parameters to result */
        memcpy(r->op_params, params, num_params*sizeof(*params));
    mag_op_exec(r, r->ctx->device, stage);  /* Execute the operator. */
    if (inplace) mag_bump_version(r);   /* result aliases the modified storage */
    if (!is_recording_grads)
        mag_tensor_detach_inplace(r); /* If gradient are not recorded, detach the tensor's parents (clear parent and opcode). TODO: why are we doing this? */
    return r;
}

static void mag_assert_dtype_compat(mag_opcode_t op, mag_tensor_t** inputs) {
    const mag_opmeta_t* meta = mag_op_meta_of(op);
    for (uint32_t i=0; i < meta->input_count; ++i) { /* Check that the input data types are supported by the operator. */
        bool supported = meta->dtype_mask & mag_dtype_bit(inputs[i]->dtype);
        if (mag_unlikely(!supported)) {
            const char* dtype = mag_dtype_meta_of(inputs[i]->dtype)->name;
            mag_panic(
                "Data type '%s' is not supported by operator '%s'.\n"
                "    Hint: Use a different data type or operator.\n",
                dtype, meta->mnemonic
            );
        }
    }
    if (mag_unlikely(meta->input_count == 2 && inputs[0]->dtype != inputs[1]->dtype)) { /* For binary operators, check that both inputs have the same data type. */
        const char* dtype_x = mag_dtype_meta_of(inputs[0]->dtype)->name;
        const char* dtype_y = mag_dtype_meta_of(inputs[1]->dtype)->name;
        mag_panic(
            "Data types of inputs for operator '%s' must match, but are '%s' and '%s'.\n"
            "    Hint: Use the same data type for both inputs.\n",
            meta->mnemonic, dtype_x, dtype_y
        );
    }
}

static void mag_assert_inplace_and_grad_mode_off(const mag_tensor_t* result) {
    if (mag_unlikely((result->ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && (result->flags & MAG_TFLAG_REQUIRES_GRAD))) {
        mag_panic(
            "Inplace operation on tensor with gradient tracking enabled is not allowed.\n"
            "    Hint: Disable gradient tracking or use a non-inplace operation.\n"
        );
    }
}

mag_tensor_t* mag_clone(mag_tensor_t* x) {
    mag_assert2(x != NULL);
    return mag_dispatch(mag_tensor_empty_like(x), MAG_OP_CLONE, false, &x, 1,NULL, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_view(mag_tensor_t* x, const int64_t* dims, int64_t rank) {
    mag_assert2(x != NULL);
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid view dimensions count, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);

    mag_tensor_t* result = NULL;
    if (rank <= 0) {
        result = mag_tensor_as_strided(x->ctx, x, x->rank, x->shape, x->strides, x->storage_offset);
    } else {
        mag_assert2(dims != NULL);
        int64_t new_dims[MAG_MAX_DIMS];
        for (int64_t i=0; i < rank; ++i)
            new_dims[i] = dims[i];
        int64_t shape[MAG_MAX_DIMS];
        mag_infer_missing_dim(&shape, new_dims, rank, x->numel);
        int64_t strides[MAG_MAX_DIMS];
        if (rank == x->rank && !memcmp(shape, x->shape, rank*sizeof(*shape))) { /* Stride strategy: same shape as base */
            memcpy(strides, x->strides, rank*sizeof(*shape));
        } else if (rank == x->rank+1 && shape[rank-2]*shape[rank-1] == x->shape[x->rank-1]) { /* Stride strategy: last dim only */
            memcpy(strides, x->strides, (rank-2)*sizeof(*strides));
            strides[rank-2] = x->strides[x->rank-1]*shape[rank-1];
            strides[rank-1] = x->strides[x->rank-1];
        } else if (mag_tensor_is_contiguous(x)) { /* Stride strategy: contiguous row-major */
            strides[rank-1] = 1;
            for (int64_t d = rank-2; d >= 0; --d)
                mag_assert2(!mag_mulov64(shape[d+1], strides[d+1], strides+d));
        } else { /* Stride strategy: solve generic strides */
            mag_assert(mag_solve_view_strides(&strides, x->shape, x->strides, x->rank, shape, rank),
                "Tensor is not contiguous enough to be viewed\n"
                "Consider calling contiguous() or reshape() instead"
            );
        }
        result = mag_tensor_as_strided(x->ctx, x, rank, shape, strides, x->storage_offset);
    }

    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank));
    if (dims)
        for (int64_t i=0; i < rank; ++i)
            mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));

    return mag_dispatch(result, MAG_OP_VIEW, false, &x, 1, &layout, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_reshape(mag_tensor_t* x, const int64_t* dims, int64_t rank) {
    mag_assert2(x != NULL);
    mag_assert2(dims != NULL);
    int64_t shape[MAG_MAX_DIMS];
    mag_infer_missing_dim(&shape, dims, rank, x->numel);
    if (x->rank == rank && !memcmp(x->shape, shape, sizeof(*dims)*rank)) {
        mag_tensor_incref(x);
        return x;
    }
    if (mag_tensor_is_contiguous(x)) {
        int64_t strides[MAG_MAX_DIMS];
        strides[rank-1] = 1;
        for (int64_t d=rank-2; d >= 0; --d)
            mag_assert2(!mag_mulov64(shape[d+1], strides[d+1], strides+d));
        return mag_tensor_as_strided(x->ctx, x, rank, shape, strides, x->storage_offset);
    }
    if (mag_tensor_can_view(x, shape, rank))
        return mag_view(x, shape, rank);
    mag_tensor_t* cont = mag_contiguous(x);
    int64_t strides[MAG_MAX_DIMS];
    strides[rank-1] = 1;
    for (int64_t d = rank-2; d >= 0; --d)
        mag_assert2(!mag_mulov64(shape[d+1], strides[d+1], strides+d));
    mag_tensor_t* resh = mag_tensor_as_strided(cont->ctx, cont, rank, shape, strides, cont->storage_offset);
    mag_tensor_decref(cont);
    return resh;
}

mag_tensor_t* mag_view_slice(mag_tensor_t* x, int64_t dim, int64_t start, int64_t len, int64_t step) {
    mag_assert2(x != NULL);
    mag_assert(dim >= 0 && dim < x->rank, "dim %" PRIi64 " out of range for rank %" PRIi64, dim, x->rank);
    mag_assert(step != 0, "slice step cannot be 0");
    int64_t sz = x->shape[dim];
    int64_t stop;
    if (step > 0) {
        if (start < 0) start += sz;
        if (len < 0) stop = sz;
        else stop = start + len * step;
        mag_assert(0 <= start && start < sz, "start out of bounds for dim %" PRIi64 ": %" PRIi64 " >= %" PRIi64, dim, start, sz);
        mag_assert(stop >= start, "slice stop < start with %" PRIi64 " < %" PRIi64, stop, start);
        int64_t last = start + (len - 1)*step;
        mag_assert(last < sz, "slice exceeds bounds for dim %" PRIi64 ": last index %" PRIi64 " >= %" PRIi64, dim, last, sz);
    } else {
        step = (int64_t)(~(uint64_t)step+1);
        if (start < 0) start += sz;
        if (len < 0) stop = -1;
        else stop = start - len*step;
        mag_assert(0 <= start && start < sz, "start out of bounds");
        mag_assert(stop < start, "slice stop >= start with negative step");
        mag_assert(stop >= -1, "slice exceeds bounds (neg)");
    }
    if (len < 0) len = step > 0 ? (stop - start + step - 1)/step : (start - stop + step - 1)/step;
    mag_assert(len > 0, "Slice length is 0");
    int64_t shape [MAG_MAX_DIMS];
    int64_t strides[MAG_MAX_DIMS];
    memcpy(shape, x->shape, sizeof(shape));
    memcpy(strides, x->strides, sizeof(strides));
    shape[dim] = len;
    strides[dim] = x->strides[dim]*step;
    int64_t tmp[MAG_MAX_DIMS];
    if (mag_solve_view_strides(&tmp, shape, strides, x->rank, shape, x->rank))
        memcpy(strides, tmp, sizeof(tmp));
    int64_t offset = x->storage_offset + start*x->strides[dim];
    return mag_tensor_as_strided(x->ctx, x, x->rank, shape, strides, offset);
}

mag_tensor_t* mag_transpose(mag_tensor_t* x, int64_t dim1, int64_t dim2) {
    mag_assert2(x != NULL);
    mag_assert(x->rank >= 2, "Transpose requires rank >= 2, but got: %" PRIi64, x->rank);
    mag_assert(dim1 != dim2, "Transposition axes must be unequal, but: %" PRIi64 " = %" PRIi64, dim1, dim2);
    int64_t ra = x->rank;
    int64_t ax0 = dim1;
    int64_t ax1 = dim2;
    if (ax0 < 0) ax0 += ra;
    if (ax1 < 0) ax1 += ra;
    mag_assert(ax0 >= 0 && ax0 < ra, "Invalid transposition axis: %" PRIi64, dim1);
    mag_assert(ax1 >= 0 && ax1 < ra, "Invalid transposition axis: %" PRIi64, dim2);

    int64_t shape[MAG_MAX_DIMS];
    int64_t stride[MAG_MAX_DIMS];
    memcpy(shape, x->shape, sizeof shape);
    memcpy(stride, x->strides, sizeof stride);
    mag_swap(int64_t, shape[ax0], shape[ax1]);
    mag_swap(int64_t, stride[ax0], stride[ax1]);
    mag_tensor_t* result = mag_tensor_as_strided(x->ctx, x, x->rank, shape, stride, x->storage_offset);

    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(ax0));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(ax1));

    return mag_dispatch(result, MAG_OP_TRANSPOSE, false, &x, 1, &layout, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_permute(mag_tensor_t* x, const int64_t* dims, int64_t rank) {
    mag_assert2(x != NULL);
    mag_assert2(dims != NULL);
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "Invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);

    int64_t axes[MAG_MAX_DIMS];
    for (int64_t i=0; i < rank; ++i)
        axes[i] = dims[i];
    for (int64_t i=0; i < rank; ++i)
        for (int64_t j = i+1; j < rank; ++j)
            mag_assert(axes[i] != axes[j], "Axes must be unique: %" PRIi64 " == %" PRIi64, axes[i], axes[j]);
    int64_t shape[MAG_MAX_DIMS];
    int64_t stride[MAG_MAX_DIMS];
    for (int64_t i=0; i < rank; ++i) {
        shape[i] = x->shape[axes[i]];
        stride[i] = x->strides[axes[i]];
    }
    mag_tensor_t* result = mag_tensor_as_strided(x->ctx, x, x->rank, shape, stride, x->storage_offset);

    return mag_dispatch(result, MAG_OP_PERMUTE, false, &x, 1, NULL, MAG_STAGE_EVAL);
}

static int mag_cmp_axis(const void* a, const void* b) {
    int64_t x = *(const int64_t*)a;
    int64_t y = *(const int64_t*)b;
    return (x>y) - (x<y);
}

static mag_tensor_t* mag_op_stub_reduction(mag_opcode_t op, mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_assert2(x != NULL);
    mag_assert2(dims != NULL || rank == 0);
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "Invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_assert(x->rank >= rank, "Cannot reduce over more dimensions than tensor has: rank=%" PRIi64 ", dims=%" PRIi64, x->rank, rank);

    int64_t ax[MAG_MAX_DIMS];
    if (!dims && !rank) {
        rank = x->rank;
        for (int64_t i=0; i < rank; ++i) ax[i] = i;
        dims = ax;
    } else if (dims) {
        for (int64_t i=0; i<rank; ++i) {
            int64_t a = dims[i];
            if (a < 0) a += x->rank;
            mag_assert(0 <= a && a < x->rank, "Axis out of bounds: %" PRIi64 " for rank %" PRIi64, a, x->rank);
            ax[i] = a;
        }
        qsort(ax, (size_t)rank, sizeof(int64_t), mag_cmp_axis);
        int64_t r = 0;
        for (int64_t i=0; i < rank; ++i)
            if (i == 0 || ax[i] != ax[i-1])
                ax[r++] = ax[i];
        rank = r;
        dims = ax;
    }
    mag_tensor_t* result = NULL;
    int64_t xrank = x->rank;
    int64_t prev = -1;
    for (int64_t i=0; i < rank; ++i) {
        int64_t a = dims[i];
        mag_assert(0 <= a && a < xrank, "Axis out of bounds: %" PRIi64 " for rank %" PRIi64, a, xrank);
        mag_assert(a > prev, "Axes must be strictly increasing and unique");
        prev = a;
    }
    int64_t out_dims[MAG_MAX_DIMS], j=0, k=0;
    for (int64_t d=0; d < xrank; ++d) {
        if (k < rank && dims[k] == d) { if (keepdim) out_dims[j++] = 1; ++k; }
        else { out_dims[j++] = x->shape[d]; }
    }
    int64_t orank = keepdim ? xrank : xrank - rank;
    if (!keepdim && !orank) { result = mag_tensor_empty_scalar(x->ctx, x->dtype); }
    else result = mag_tensor_empty(x->ctx, x->dtype, orank, out_dims);

    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim));
    for (int64_t i=0; i<rank; ++i)
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));

    return mag_dispatch(result, op, false, &x, 1, &layout, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mean(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(MAG_OP_MEAN, x, dims, rank, keepdim);
}

mag_tensor_t* mag_min(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(MAG_OP_MIN, x, dims, rank, keepdim);
}

mag_tensor_t* mag_max(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(MAG_OP_MAX, x, dims, rank, keepdim);
}

mag_tensor_t* mag_sum(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(MAG_OP_SUM, x, dims, rank, keepdim);
}

mag_tensor_t* mag_argmin(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_panic("Not implemented yet");
    return mag_tensor_empty_like(x);
}

mag_tensor_t* mag_argmax(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_panic("Not implemented yet");
    return mag_tensor_empty_like(x);
}

static mag_tensor_t* mag_op_stub_unary(mag_opcode_t op, mag_tensor_t* x, const mag_op_param_layout_t* layout, bool inplace) {
    mag_assert2(x != NULL);
    mag_assert_dtype_compat(op, &x);
    mag_tensor_t* result = NULL;
    if (inplace) {
        result = mag_tensor_strided_view(x); /* Use the same storage as x */
        mag_assert_inplace_and_grad_mode_off(x);
    } else {
        result = mag_tensor_empty_like(x); /* Allocate a new tensor for the result */
    }
    return mag_dispatch(result, op, inplace, &x, 1, layout, MAG_STAGE_EVAL);
}

#define mag_impl_unary_pair(name, op) \
    mag_tensor_t* mag_##name(mag_tensor_t* x) { return mag_op_stub_unary(MAG_OP_##op, x, NULL, false); } \
    mag_tensor_t* mag_##name##_(mag_tensor_t* x) { return mag_op_stub_unary(MAG_OP_##op, x, NULL, true); }

mag_impl_unary_pair(not, NOT)
mag_impl_unary_pair(abs, ABS)
mag_impl_unary_pair(sgn, SGN)
mag_impl_unary_pair(neg, NEG)
mag_impl_unary_pair(log, LOG)
mag_impl_unary_pair(sqr, SQR)
mag_impl_unary_pair(sqrt, SQRT)
mag_impl_unary_pair(sin, SIN)
mag_impl_unary_pair(cos, COS)
mag_impl_unary_pair(step, STEP)
mag_impl_unary_pair(exp, EXP)
mag_impl_unary_pair(floor, FLOOR)
mag_impl_unary_pair(ceil, CEIL)
mag_impl_unary_pair(round, ROUND)
mag_impl_unary_pair(softmax, SOFTMAX)
mag_impl_unary_pair(softmax_dv, SOFTMAX_DV)
mag_impl_unary_pair(sigmoid, SIGMOID)
mag_impl_unary_pair(sigmoid_dv, SIGMOID_DV)
mag_impl_unary_pair(hard_sigmoid, HARD_SIGMOID)
mag_impl_unary_pair(silu, SILU)
mag_impl_unary_pair(silu_dv, SILU_DV)
mag_impl_unary_pair(tanh, TANH)
mag_impl_unary_pair(tanh_dv, TANH_DV)
mag_impl_unary_pair(relu, RELU)
mag_impl_unary_pair(relu_dv, RELU_DV)
mag_impl_unary_pair(gelu, GELU)
mag_impl_unary_pair(gelu_dv, GELU_DV)

#undef mag_impl_unary_pair

mag_tensor_t* _Nonnull mag_tril(mag_tensor_t* _Nonnull x, int32_t diag) {
    mag_assert2(x != NULL);
    mag_assert(x->rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, x->rank);

    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(diag));
    return mag_op_stub_unary(MAG_OP_TRIL, x, &layout, false);
}

mag_tensor_t* _Nonnull mag_tril_(mag_tensor_t* _Nonnull x, int32_t diag) {
    mag_assert2(x != NULL);
    mag_assert(x->rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, x->rank);

    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(diag));
    return mag_op_stub_unary(MAG_OP_TRIL, x, &layout, true);
}

mag_tensor_t* _Nonnull mag_triu(mag_tensor_t* _Nonnull x, int32_t diag) {
    mag_assert2(x != NULL);
    mag_assert(x->rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, x->rank);

    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(diag));
    return mag_op_stub_unary(MAG_OP_TRIU, x, &layout, false);
}

mag_tensor_t* _Nonnull mag_triu_(mag_tensor_t* _Nonnull x, int32_t diag) {
    mag_assert2(x != NULL);
    mag_assert(x->rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, x->rank);

    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(diag));
    return mag_op_stub_unary(MAG_OP_TRIU, x, &layout, true);
}

static mag_tensor_t* mag_op_stub_binary(mag_opcode_t op, mag_tensor_t* x, mag_tensor_t* y, bool boolean_result, bool inplace) {
    mag_assert2(x != NULL);
    mag_assert2(y != NULL);
    mag_assert_dtype_compat(op, (mag_tensor_t*[]){x, y});
    mag_tensor_t* result = NULL;
    if (inplace) {
        mag_assert2(!boolean_result);
        mag_assert_inplace_and_grad_mode_off(x);
        result = mag_tensor_strided_view(x); /* Use the same storage as x */
    } else {
        int64_t dims[MAG_MAX_DIMS];
        int64_t rank;
        if (mag_unlikely(!mag_compute_broadcast_shape(x, y, dims, &rank))) {
            char sx[MAG_FMT_DIM_BUF_SIZE];
            char sy[MAG_FMT_DIM_BUF_SIZE];
            mag_fmt_shape(&sx, &x->shape, x->rank);
            mag_fmt_shape(&sy, &y->shape, y->rank);
            mag_panic(
                "Cannot broadcast tensors with shapes %s and %s for operator '%s'.\n"
                "    Hint: Ensure that the shapes are compatible for broadcasting.\n",
                sx, sy, mag_op_meta_of(op)->mnemonic
            );
        }
        mag_dtype_t rtype = boolean_result ? MAG_DTYPE_BOOL : x->dtype;
        result = rank ? mag_tensor_empty(x->ctx, rtype, rank, dims) : mag_tensor_empty_scalar(x->ctx, rtype);
    }
    return mag_dispatch(result, op, inplace, (mag_tensor_t*[]){x, y}, 2, NULL, MAG_STAGE_EVAL);
}

#define mag_impl_binary_pair(name, op, boolean_result) \
    mag_tensor_t* mag_##name(mag_tensor_t* x, mag_tensor_t* y) { return mag_op_stub_binary(MAG_OP_##op, x, y, boolean_result, false); } \
    mag_tensor_t* mag_##name##_(mag_tensor_t* x, mag_tensor_t* y) { return mag_op_stub_binary(MAG_OP_##op, x, y, boolean_result, true); }

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

mag_tensor_t* mag_matmul(mag_tensor_t* x, mag_tensor_t* y) {
    mag_assert2(x != NULL);
    mag_assert2(y != NULL);
    mag_assert_dtype_compat(MAG_OP_MATMUL, (mag_tensor_t*[]){x, y});

    int64_t k_x = x->shape[x->rank-1];
    int64_t k_y = y->rank == 1 ? y->shape[0] : y->rank == 2 && x->rank == 1 ? y->shape[0] : y->shape[y->rank-2];
    if (k_x != k_y) {
        char fmt_x[MAG_FMT_DIM_BUF_SIZE], fmt_y[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&fmt_x, &x->shape, x->rank);
        mag_fmt_shape(&fmt_y, &y->shape, y->rank);
        mag_panic(
            "Cannot perform matmul on tensors with shapes %s and %s: "
            "last dimension of first tensor (%" PRIi64 ") does not match second tensor (%" PRIi64 ").\n"
            "    Hint: Ensure that the last dimension of the first tensor matches the second-to-last dimension of the second tensor.\n",
            fmt_x, fmt_y, k_x, k_y
        );
    }
    /* verify broadcastability of batch dims */
    int64_t x_bd = x->rank > 2 ? x->rank-2 : 0;
    int64_t y_bd = y->rank > 2 ? y->rank-2 : 0;
    int64_t r_bd = x_bd > y_bd ? x_bd : y_bd;
    for (int64_t i=0; i < r_bd; ++i) {
        int64_t x_dim = i < r_bd-x_bd ? 1 : x->shape[i-(r_bd-x_bd)];
        int64_t y_dim = i < r_bd-y_bd ? 1 : y->shape[i-(r_bd-y_bd)];
        if (x_dim != y_dim && x_dim != 1 && y_dim != 1) {
            char fmt_x[MAG_FMT_DIM_BUF_SIZE], fmt_y[MAG_FMT_DIM_BUF_SIZE];
            mag_fmt_shape(&fmt_x, &x->shape, x->rank);
            mag_fmt_shape(&fmt_y, &y->shape, y->rank);
            mag_panic(
                "Cannot perform matmul on tensors with shapes %s and %s: "
                "batch dimensions at index %" PRIi64 " do not match (%" PRIi64 " != %" PRIi64 ").\n"
                "    Hint: Ensure that the batch dimensions are compatible for broadcasting.\n",
                fmt_x, fmt_y, i, x_dim, y_dim
            );
        }
    }

    mag_tensor_t* result = NULL;
    if (x->rank == 1 && y->rank == 1) { /* (K)x(K) -> () */
        int64_t shape[1] = {1};
        result = mag_tensor_new(x->ctx, x->dtype, 1, shape);
    } else if (x->rank == 1 && y->rank == 2) { /* (K)x(K,N) -> (N) */
        int64_t shape[1] = {y->shape[1]};
        result = mag_tensor_new(x->ctx, x->dtype, 1, shape);
    } else if (x->rank == 2 && y->rank == 1) { /* (M,K)x(K) -> (M) */
        int64_t shape[1] = {x->shape[0]};
        result = mag_tensor_new(x->ctx, x->dtype, 1, shape);
    } else { /* Batched ND version */
        int64_t a_bd = x->rank-2;
        int64_t b_bd = y->rank-2;
        int64_t shape[MAG_MAX_DIMS] = {0};
        for (int64_t i=0; i < r_bd; ++i) {
            int64_t a_dim = i < r_bd-a_bd ? 1 : x->shape[i-(r_bd-a_bd)];
            int64_t b_dim = i < r_bd-b_bd ? 1 : y->shape[i-(r_bd-b_bd)];
            shape[i] = a_dim > b_dim ? a_dim : b_dim;
        }
        shape[r_bd] = x->shape[x->rank-2];
        shape[r_bd+1] = y->shape[y->rank-1];
        result = mag_tensor_new(x->ctx, x->dtype, r_bd+2, shape);
    }
    return mag_dispatch(result, MAG_OP_MATMUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_repeat_back(mag_tensor_t* x, mag_tensor_t* y) {
    mag_assert2(x != NULL);
    mag_assert2(y != NULL);
    mag_assert_dtype_compat(MAG_OP_REPEAT_BACK, (mag_tensor_t*[]){x, y});
    mag_tensor_t* result = mag_tensor_new(x->ctx, x->dtype, y->rank, y->shape);
    /* TODO: Check for broadcastability of x and y */
    return mag_dispatch(result, MAG_OP_REPEAT_BACK, false, (mag_tensor_t*[]){x, y}, 2, NULL, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gather(mag_tensor_t* x, int64_t dim, mag_tensor_t* idx){
    mag_assert2(x != NULL);
    mag_assert2(idx != NULL);
    mag_assert2(mag_tensor_is_integer_typed(idx));
    mag_assert(dim >= 0 && dim < x->rank, "gather dimension %" PRIi64 " out of range for rank %" PRIi64, dim, x->rank);

    if (dim < 0) dim += x->rank;
    mag_assert2(dim >= 0 && dim < x->rank);
    int64_t ax[MAG_MAX_DIMS];
    int64_t ork = 0;
    bool full = false;
    if (idx->rank == x->rank) {
        full = true;
        for (int64_t d=0; d < x->rank; ++d) {
            if (d == dim) continue;
            if (idx->shape[d] != x->shape[d]) {
                full = false;
                break;
            }
        }
    }
    if (full) {
        for (int64_t d = 0; d < x->rank; ++d)
            ax[ork++] = idx->shape[d];
    } else if (idx->rank == 1) {
        for (int64_t d=0; d < x->rank; ++d) {
            ax[ork++] = d == dim ? idx->shape[0] : x->shape[d];
        }
    } else {
        for (int64_t d = 0; d < dim; ++d) ax[ork++] = x->shape[d];
        for (int64_t i=0; i < idx->rank; ++i) ax[ork++] = idx->shape[i];
        for (int64_t d=dim+1; d < x->rank; ++d) ax[ork++] = x->shape[d];
    }
    mag_assert(ork >= 1 && ork <= MAG_MAX_DIMS,
        "Gather output rank must be in [1, %d], but got: %" PRIi64,
        MAG_MAX_DIMS, ork
    );
    mag_tensor_t* result = mag_tensor_empty(x->ctx, x->dtype, ork, ax);

    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dim)); /* Store dimension in op_params[0] */
    return mag_dispatch(result, MAG_OP_GATHER, false, (mag_tensor_t*[]){x, idx}, 2, &layout, MAG_STAGE_EVAL);
}

void mag_tensor_fill_from_floats(mag_tensor_t* t, const mag_e8m23_t* data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_istorage_t* sto = t->storage;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_H2D, 0, (void*)data, len*sizeof(*data), MAG_DTYPE_E8M23);
}

void mag_tensor_fill_from_raw_bytes(mag_tensor_t* t, const void* data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_istorage_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, 0, (void*)data, len);
}

void mag_tensor_fill_float(mag_tensor_t* t, mag_e8m23_t x) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    t->init_op = MAG_IOP_BROADCAST;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(x));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_int(mag_tensor_t* t, int32_t x) {
    mag_assert2(mag_tensor_is_integral_typed(t));
    t->init_op = MAG_IOP_BROADCAST;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(x));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_masked_fill_float(mag_tensor_t* t, mag_tensor_t* mask, mag_e8m23_t x) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    mag_assert2(mask->dtype == MAG_DTYPE_BOOL);
    t->init_op = MAG_IOP_MASKED_BROADCAST;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(x));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64((int64_t)(uintptr_t)mask));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_masked_fill_int(mag_tensor_t* t, mag_tensor_t* mask, int32_t x) {
    mag_assert2(mag_tensor_is_integral_typed(t));
    mag_assert2(mask->dtype == MAG_DTYPE_BOOL);
    t->init_op = MAG_IOP_MASKED_BROADCAST;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(x));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64((int64_t)(uintptr_t)mask));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_uniform_float(mag_tensor_t* t, mag_e8m23_t min, mag_e8m23_t max) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    t->init_op = MAG_IOP_RAND_UNIFORM;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(min));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(max));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_uniform_int(mag_tensor_t* t, int32_t min, int32_t max) {
    mag_assert2(mag_tensor_is_integral_typed(t));
    t->init_op = MAG_IOP_RAND_UNIFORM;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(min));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(max));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_normal(mag_tensor_t* t, mag_e8m23_t mean, mag_e8m23_t stddev) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    t->init_op = MAG_IOP_RAND_NORMAL;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(mean));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(stddev));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_bernoulli(mag_tensor_t* t, mag_e8m23_t p) {
    mag_assert2(t->dtype == MAG_DTYPE_BOOL);
    t->init_op = MAG_IOP_RAND_BERNOULLI;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(p));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_arange(mag_tensor_t* t, float start, float step){
    mag_assert2(t->rank == 1 && (mag_dtype_bit(t->dtype) & MAG_DTYPE_MASK_NUMERIC));
    t->init_op = MAG_IOP_ARANGE;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(start));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(step));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

/*
** ###################################################################################################################
** Operator Backprop Impls
** ###################################################################################################################
*/

static void mag_op_backward_clone(mag_tensor_t* node, mag_tensor_t** grads) {
    *grads = mag_clone(node->grad);
}

static void mag_op_backward_view(mag_tensor_t* node, mag_tensor_t** grads) {
    *grads = mag_clone(node->grad);
}

static void mag_op_backward_transpose(mag_tensor_t* node, mag_tensor_t** grads) {
    int64_t ax0 = mag_op_param_unpack_i64_or_panic(node->op_params[0]);
    int64_t ax1 = mag_op_param_unpack_i64_or_panic(node->op_params[1]);
    *grads = mag_transpose(node->grad, ax0, ax1);
}

static void mag_op_backward_mean(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* scale = mag_tensor_full_like(x, (mag_e8m23_t)(1.0/(mag_e11m52_t)x->numel));
    *grads = mag_mul(scale, node->grad);
    mag_tensor_decref(scale);
}

static void mag_op_backward_sum(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* ones = mag_tensor_full_like(x, 1.0f);
    *grads = mag_mul(ones, node->grad);
    mag_tensor_decref(ones);
}

static void mag_op_backward_abs(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* step = mag_step(x);
    mag_tensor_t* one = mag_tensor_scalar(x->ctx, x->dtype, 1.0f);
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* step2 = mag_mul(step, two);
    mag_tensor_t* sign = mag_sub(step2, one);
    grads[0] = mag_mul(node->grad, sign);
    mag_tensor_decref(two);
    mag_tensor_decref(one);
    mag_tensor_decref(step);
    mag_tensor_decref(step2);
    mag_tensor_decref(sign);
}

static void mag_op_backward_neg(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* m1 = mag_tensor_scalar(node->grad->ctx, node->grad->dtype, -1.f);
    grads[0] = mag_mul(node->grad, m1);
    mag_tensor_decref(m1);
}

static void mag_op_backward_log(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    grads[0] = mag_div(node->grad, x);
}

static void mag_op_backward_sqr(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* two_x = mag_mul(x, two);
    grads[0] = mag_mul(node->grad, two_x);
    mag_tensor_decref(two);
    mag_tensor_decref(two_x);
}

static void mag_op_backward_sqrt(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* sqrt_x = mag_sqrt(x);
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* denom = mag_mul(sqrt_x, two);
    grads[0] = mag_div(node->grad, denom);
    mag_tensor_decref(two);
    mag_tensor_decref(sqrt_x);
    mag_tensor_decref(denom);
}

static void mag_op_backward_sin(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* cos_x = mag_cos(x);
    grads[0] = mag_mul(node->grad, cos_x);
    mag_tensor_decref(cos_x);
}

static void mag_op_backward_cos(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* sinx = mag_sin(x);
    mag_tensor_t* nsinx = mag_neg(sinx);
    grads[0] = mag_mul(node->grad, nsinx);
    mag_tensor_decref(sinx);
    mag_tensor_decref(nsinx);
}

static void mag_op_backward_exp(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* exp_x = mag_exp(x);
    grads[0] = mag_mul(node->grad, exp_x);
    mag_tensor_decref(exp_x);
}

static void mag_op_backward_softmax(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = mag_softmax(x);
    mag_tensor_t* tmp = mag_mul(node->grad, y);
    mag_tensor_t* sum_tmp = mag_sum(tmp, NULL, 0, false);
    mag_tensor_t* diff = mag_sub(node->grad, sum_tmp);
    grads[0] = mag_mul(y, diff);
    mag_tensor_decref(tmp);
    mag_tensor_decref(sum_tmp);
    mag_tensor_decref(diff);
    mag_tensor_decref(y);
}

static void mag_op_backward_sigmoid(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_sigmoid_dv(x);
    grads[0] = mag_mul(dv, node->grad);
    mag_tensor_decref(dv);
}

static void mag_op_backward_silu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_silu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_tanh(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_tanh_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_relu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* mask = mag_step(x);
    grads[0] = mag_mul(node->grad, mask);
    mag_tensor_decref(mask);
}

static void mag_op_backward_gelu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_gelu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_add(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_clone(node->grad);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* grad = node->grad;
        if (!mag_tensor_is_shape_eq(x, y)) {
            grad = mag_repeat_back(grad, y);
        } else {
            grad = mag_clone(grad); /* Output gradients must be a new allocated tensor, so we clone. */
        }
        grads[1] = grad;
    }
}

static void mag_op_backward_sub(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_clone(node->grad);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* mg = mag_neg(node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pmg = mg;
            mg = mag_repeat_back(pmg, y);
            mag_tensor_decref(pmg);
        }
        grads[1] = mg;
    }
}

static void mag_op_backward_mul(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_mul(node->grad, y);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* xg = mag_mul(x, node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pxg = xg;
            xg = mag_repeat_back(pxg, y);
            mag_tensor_decref(pxg);
        }
        grads[1] = xg;
    }
}

static void mag_op_backward_div(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_div(node->grad, y);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* gx = mag_mul(node->grad, x);
        mag_tensor_t* yy = mag_mul(y, y);
        mag_tensor_t* gxyy = mag_div(gx, yy);
        mag_tensor_t* mgxyy = mag_neg(gxyy);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pmgxyy = mgxyy;
            mgxyy = mag_repeat_back(pmgxyy, y);
            mag_tensor_decref(pmgxyy);
        }
        grads[1] = mgxyy;
        mag_tensor_decref(gxyy);
        mag_tensor_decref(yy);
        mag_tensor_decref(gx);
    }
}

static void mag_op_backward_matmul(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* yt = mag_transpose(y, 0, 1);
        grads[0] = mag_matmul(node->grad, yt);
        mag_tensor_decref(yt);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* xt = mag_transpose(x, 0, 1);
        grads[1] = mag_matmul(xt, node->grad);
        mag_tensor_decref(xt);
    }
}

/*
** ###################################################################################################################
** Operator Metadata List
** ###################################################################################################################
*/

const mag_opmeta_t* mag_op_meta_of(mag_opcode_t opc) {
    static const mag_opmeta_t infos[MAG_OP__NUM] = {
        [MAG_OP_NOP] = {
            .mnemonic = "nop",
            .desc = "nop",
            .input_count = 0,
            .dtype_mask = 0,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
        },
        [MAG_OP_CLONE] = {
            .mnemonic = "clone",
            .desc = "clone",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_clone,
        },
        [MAG_OP_VIEW] = {
            .mnemonic = "view",
            .desc = "view",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_view,
        },
        [MAG_OP_TRANSPOSE] = {
            .mnemonic = "transpose",
            .desc = "𝑥ᵀ",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {
                MAG_OPP_I64, /*ax0*/
                MAG_OPP_I64, /*ax1*/
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_transpose,
        },
        [MAG_OP_PERMUTE] = {
            .mnemonic = "permute",
            .desc = "permute",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
        },
        [MAG_OP_MEAN] = {
            .mnemonic = "mean",
            .desc = "(∑𝑥)∕𝑛",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {
                MAG_OPP_I64,  /* reduction dim count : u32 */
                MAG_OPP_I64,  /* keepdim : bool */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_mean,
        },
        [MAG_OP_MIN] = {
            .mnemonic = "min",
            .desc = "min(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {
                MAG_OPP_I64,  /* reduction dim count : u32 */
                MAG_OPP_I64,  /* keepdim : bool */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
        },
        [MAG_OP_MAX] = {
            .mnemonic = "max",
            .desc = "max(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {
                MAG_OPP_I64,  /* reduction dim count : u32 */
                MAG_OPP_I64,  /* keepdim : bool */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
        },
        [MAG_OP_SUM] = {
            .mnemonic = "sum",
            .desc = "∑𝑥",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {
                MAG_OPP_I64,  /* reduction dim count : u32 */
                MAG_OPP_I64,  /* keepdim : bool */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
                MAG_OPP_I64, /* reduction dim : u32 [optional] */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_sum,
        },
        [MAG_OP_ABS] = {
            .mnemonic = "abs",
            .desc = "|𝑥|",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_abs,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SGN] = {
            .mnemonic = "sgn",
            .desc = "𝑥⁄",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_NEG] = {
            .mnemonic = "neg",
            .desc = "−𝑥",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_neg,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_LOG] = {
            .mnemonic = "log",
            .desc = "log₁₀(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_log,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SQR] = {
            .mnemonic = "sqr",
            .desc = "𝑥²",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sqr,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SQRT] = {
            .mnemonic = "sqrt",
            .desc = "√𝑥",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags =  MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sqrt,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIN] = {
            .mnemonic = "sin",
            .desc = "sin(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sin,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_COS] = {
            .mnemonic = "cos",
            .desc = "cos(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_cos,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_STEP] = {
            .mnemonic = "step",
            .desc = "𝐻(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_EXP] = {
            .mnemonic = "exp",
            .desc = "𝑒ˣ",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_exp,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_FLOOR] = {
            .mnemonic = "floor",
            .desc = "⌊𝑥⌋",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_CEIL] = {
            .mnemonic = "ceil",
            .desc = "⌈𝑥⌉",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_ROUND] = {
            .mnemonic = "round",
            .desc = "⟦𝑥⟧",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SOFTMAX] = {
            .mnemonic = "softmax",
            .desc = "𝑒ˣⁱ∕∑𝑒ˣʲ",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_softmax,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SOFTMAX_DV] = {
            .mnemonic = "softmax_dv",
            .desc = "𝑑⁄𝑑𝑥 softmax(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIGMOID] = {
            .mnemonic = "sigmoid",
            .desc = "1∕(1 + 𝑒⁻ˣ)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = mag_op_backward_sigmoid,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIGMOID_DV] = {
            .mnemonic = "sigmoid_dv",
            .desc = "𝑑⁄𝑑𝑥 sigmoid(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_HARD_SIGMOID] = {
            .mnemonic = "hard_sigmoid",
            .desc = "max(0,min(1,0.2×𝑥+0.5))",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SILU] = {
            .mnemonic = "silu",
            .desc = "𝑥∕(1+𝑒⁻ˣ)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_silu,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SILU_DV] = {
            .mnemonic = "silu_dv",
            .desc = "𝑑⁄𝑑𝑥 silu(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_TANH] = {
            .mnemonic = "tanh",
            .desc = "tanh(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_tanh,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_TANH_DV] = {
            .mnemonic = "tanh_dv",
            .desc = "𝑑⁄𝑑𝑥 tanh(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_RELU] = {
            .mnemonic = "relu",
            .desc = "max(0, 𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_relu,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_RELU_DV] = {
            .mnemonic = "relu_dv",
            .desc = "𝑑⁄𝑑𝑥 relu(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_GELU] = {
            .mnemonic = "gelu",
            .desc = "0.5×𝑥×(1+erf(𝑥∕√2))",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_gelu,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_GELU_DV] = {
            .mnemonic = "gelu_dv",
            .desc = "𝑑⁄𝑑𝑥 gelu(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_TRIL] = {
            .mnemonic = "tril",
            .desc = "tril(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {
                MAG_OPP_I64
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_TRIU] = {
            .mnemonic = "triu",
            .desc = "triu(𝑥)",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {
                MAG_OPP_I64
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_ADD] = {
            .mnemonic = "add",
            .desc = "𝑥 + 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_NUMERIC,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_add,
            .cpu = {
                .thread_growth = 3.5,
                .thread_treshold = 10000
            }
        },
        [MAG_OP_SUB] = {
            .mnemonic = "sub",
            .desc = "𝑥 − 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_NUMERIC,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sub,
            .cpu = {
                .thread_growth = 3.5,
                 .thread_treshold = 10000
            }
        },
        [MAG_OP_MUL] = {
            .mnemonic = "mul",
            .desc = "𝑥 ⊙ 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_NUMERIC,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_mul,
            .cpu = {
                .thread_growth = 3.5,
                .thread_treshold = 10000
            }
        },
        [MAG_OP_DIV] = {
            .mnemonic = "div",
            .desc = "𝑥 ∕ 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_NUMERIC,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_div,
            .cpu = {
                .thread_growth = 3.5,
                .thread_treshold = 10000
            }
        },
        [MAG_OP_MATMUL] = {
            .mnemonic = "matmul",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_matmul,
            .cpu = {
                .thread_growth = 4.5,
                .thread_treshold = 1000
            }
        },
        [MAG_OP_REPEAT_BACK] = {
            .mnemonic = "repeat_back",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE,
            .backward = NULL,
        },
        [MAG_OP_GATHER] = {
            .mnemonic = "gather",
            .desc = "gather",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {
                MAG_OPP_I64, /* dim : i64 */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
        },
        [MAG_OP_AND] = {
            .mnemonic = "and",
            .desc = "𝑥 ∧ 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_INTEGRAL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_OR] = {
            .mnemonic = "or",
            .desc = "𝑥 ∨ 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_INTEGRAL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_XOR] = {
            .mnemonic = "xor",
            .desc = "𝑥 ⊕ 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_INTEGRAL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_NOT] = {
            .mnemonic = "not",
            .desc = "¬𝑥",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_INTEGRAL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SHL] = {
            .mnemonic = "shl",
            .desc = "𝑥 ≪ 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_INTEGER,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SHR] = {
            .mnemonic = "shr",
            .desc = "𝑥 ≫ 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_INTEGER,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_EQ] = {
            .mnemonic = "eq",
            .desc = "𝑥 == 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_NE] = {
            .mnemonic = "ne",
            .desc = "𝑥 != 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_LE] = {
            .mnemonic = "le",
            .desc = "𝑥 <= 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_GE] = {
            .mnemonic = "ge",
            .desc = "𝑥 >= 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_GT] = {
            .mnemonic = "gt",
            .desc = "𝑥 < 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_LT] = {
            .mnemonic = "lt",
            .desc = "𝑥 > 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
    };
    return infos+opc;
}
