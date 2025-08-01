/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
*/

#include "magnetron_internal.h"

#include <stdarg.h>

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

static void mag_push_verification_error(mag_sstream_t** ss, const char* fmt, ...) {
    if (!*ss) {  /* Lazy init error stream if needed */
        *ss = (*mag_alloc)(NULL, sizeof(**ss), 0);
        mag_sstream_init(*ss);
    }
    va_list ap;
    va_start(ap, fmt);
    mag_sstream_vappend(*ss, fmt, ap);
    va_end(ap);
    mag_sstream_putc(*ss, '\n');
}

static bool mag_verify_is_shape_eq(mag_sstream_t** ss, const mag_tensor_t* x, const mag_tensor_t* y) {
    if (mag_unlikely(!mag_tensor_is_shape_eq(x, y))) {
        char fmt_shape_x[MAG_FMT_DIM_BUF_SIZE];
        char fmt_shape_y[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&fmt_shape_x, &x->shape, x->rank);
        mag_fmt_shape(&fmt_shape_y, &y->shape, y->rank);
        mag_push_verification_error(ss,
            "Shape mismatch: %s != %s\n"
            "    Hint: Tensors must have the same shape and rank.\n",
            fmt_shape_x, fmt_shape_y
        );
        return false;
    }
    return true;
}
static bool mag_verify_is_min_rank(mag_sstream_t** ss, const mag_tensor_t* x, int64_t min_rank) {
    if (mag_unlikely(x->rank < min_rank)) {
        mag_push_verification_error(ss,
            "Rank >= %" PRIi64 " required, but is: %" PRIi64
            "    Hint: Extend tensor dimensionality.\n",
            min_rank, x->rank
        );
        return false;
    }
    return true;
}
static bool mag_verify_can_broadcast(mag_sstream_t** ss, const mag_tensor_t* x, const mag_tensor_t* y) {
    if (mag_unlikely(!mag_tensor_can_broadcast(y, x))) {
        char fmt_shape_x[MAG_FMT_DIM_BUF_SIZE];
        char fmt_shape_y[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&fmt_shape_x, &x->shape, x->rank);
        mag_fmt_shape(&fmt_shape_y, &y->shape, y->rank);
        mag_push_verification_error(ss,
            "Shape mismatch: %s cannot be broadcasted to %s\n"
            "    Hint: Tensors must have compatible shapes for broadcasting.\n",
            fmt_shape_y, fmt_shape_x
        );
        return false;
    }
    return true;
}
static bool mag_verify_dtype_compat(mag_sstream_t** ss, mag_opcode_t op, mag_tensor_t** inputs) {
    const mag_opmeta_t* meta = mag_op_meta_of(op);
    for (uint32_t i=0; i < meta->input_count; ++i) { /* Check that the input data types are supported by the operator. */
        bool supported = meta->dtype_mask & mag_dtype_bit(inputs[i]->dtype);
        if (mag_unlikely(!supported)) {
            const char* dtype = mag_dtype_meta_of(inputs[i]->dtype)->name;
            mag_push_verification_error(ss,
                "Unsupported data type '%s' for operator: %s\n"
                "    Hint: Cast the tensor.\n",
                dtype, meta->mnemonic
            );
            return false;
        }
    }
    if (mag_unlikely(meta->input_count == 2 && inputs[0]->dtype != inputs[1]->dtype)) { /* For binary operators, check that both inputs have the same data type. */
        const char* dtype_x = mag_dtype_meta_of(inputs[0]->dtype)->name;
        const char* dtype_y = mag_dtype_meta_of(inputs[1]->dtype)->name;
        mag_push_verification_error(ss,
            "Data type mismatch: %s != %s\n"
            "    Hint: Tensors must have the same data type for this operation.\n",
            dtype_x, dtype_y
        );
        return false;
    }
    return true;
}
static bool mag_verify_can_matmul(mag_sstream_t** ss, const mag_tensor_t* x, const mag_tensor_t* y) {
    int64_t k_x = x->shape[x->rank - 1]; /* Find contracted dims */
    int64_t k_y = y->rank == 1 ? y->shape[0] : y->rank == 2 && x->rank == 1 ? y->shape[0] : y->shape[y->rank-2];
    if (k_x != k_y) {
        char fmt_x[MAG_FMT_DIM_BUF_SIZE], fmt_y[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&fmt_x, &x->shape, x->rank);
        mag_fmt_shape(&fmt_y, &y->shape, y->rank);
        mag_push_verification_error(
            ss,
            "Shape mismatch: %s cannot be matrix-multiplied with %s "
            "(contracted dims %" PRIi64 " != %" PRIi64 ")\n",
            fmt_x, fmt_y, k_x, k_y);
        return false;
    }
    /* verify broadcastability of batch dims */
    int64_t x_bd = x->rank > 2 ? x->rank-2 : 0;
    int64_t y_bd = y->rank > 2 ? y->rank-2 : 0;
    int64_t r_bd = x_bd > y_bd ? x_bd : y_bd;
    for (int64_t i = 0; i < r_bd; ++i) {
        int64_t x_dim = i < r_bd-x_bd ? 1 : x->shape[i-(r_bd-x_bd)];
        int64_t y_dim = i < r_bd-y_bd ? 1 : y->shape[i-(r_bd-y_bd)];
        if (x_dim != y_dim && x_dim != 1 && y_dim != 1) {
            char fmt_x[MAG_FMT_DIM_BUF_SIZE], fmt_y[MAG_FMT_DIM_BUF_SIZE];
            mag_fmt_shape(&fmt_x, &x->shape, x->rank);
            mag_fmt_shape(&fmt_y, &y->shape, y->rank);
            mag_push_verification_error(
                ss,
                "Cannot broadcast batch dims of %s and %s for matmul\n",
                fmt_x, fmt_y);
            return false;
        }
    }
    return true;

    return true;
}
static bool mag_verify_is_inplace_and_grad_mode_off(mag_sstream_t** ss, const mag_tensor_t* result, bool is_inplace) {
    if (mag_unlikely(is_inplace && (result->ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && (result->flags & MAG_TFLAG_REQUIRES_GRAD))) {
        mag_push_verification_error(ss,
            "Inplace operation is not allowed when recording gradients: %s\n"
            "    Hint: Disable gradient recording or use a non-inplace operation.\n",
            result->name
        );
        return false;
    }
    return true;
}

static bool mag_validate_op_unary(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    bool ok = true;
    ok = ok && mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace);     /* Check if inplace operation is allowed */
    ok = ok && mag_verify_dtype_compat(ss, result->op, inputs);                     /* Check if the operator is defined between the given dtypes */
    ok = ok && mag_verify_is_shape_eq(ss, result, inputs[0]);                   /* Check if result shape matches input */
    return ok;
}
static bool mag_validate_op_view(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    bool ok = true;
    ok = ok && mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace);     /* Check if inplace operation is allowed */
    ok = ok && mag_verify_dtype_compat(ss, result->op, inputs);                     /* Check if the operator is defined between the given dtypes */
    return ok;
}
static bool mag_validate_op_unary_matrix(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    bool ok = mag_validate_op_unary(ss, is_inplace, result, inputs, params);
    ok = ok && mag_verify_is_min_rank(ss, result, 2); /* Verify that we have a matrix or higher-dimensional (rank >= 2). */
    return ok;
}
static bool mag_validate_op_binary(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    bool ok = true;
    ok = ok && mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace);     /* Check if inplace operation is allowed */
    ok = ok && mag_verify_dtype_compat(ss, result->op, inputs);                     /* Check if the operator is defined between the given dtypes */
    ok = ok && mag_verify_is_shape_eq(ss, result, inputs[0]);                   /* Check if result shape matches first input */
    ok = ok && mag_verify_can_broadcast(ss, inputs[0], inputs[1]);              /* Check if second input can be broadcasted to first input */
    return ok;
}
static bool mag_validate_op_transpose(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    return true; /* TODO */
}
static bool mag_validate_op_scalar(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    return mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace) &&   /* Check if inplace operation is allowed */
        mag_verify_dtype_compat(ss, result->op, inputs);                      /* Check if the operator is defined between the given dtypes */
}
static bool mag_validate_op_matmul(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    return mag_verify_can_matmul(ss, inputs[0], inputs[1]) &&   /* Check if inputs can be matrix-multiplied */
        mag_verify_dtype_compat(ss, result->op, inputs);          /* Check if the operator is defined between the given dtypes */
}
static bool mag_validate_op_repeat_rev(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    return mag_verify_can_broadcast(ss, inputs[0], inputs[1]) &&    /* Check if inputs can be matrix-multiplied */
        mag_verify_dtype_compat(ss, result->op, inputs);              /* Check if the operator is defined between the given dtypes */
}
static bool mag_validate_op_gather(mag_sstream_t** ss, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_opparam_t* params) {
    bool ok = true;
    ok = ok && mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace);     /* Check if inplace operation is allowed */
    /*ok = ok && mag_verify_dtype_compat(ss, result->op, inputs); TODO: index always has type i32, and x any */
    ok = ok && mag_verify_is_min_rank(ss, inputs[1], 1);                    /* Check if index tensor has at least rank 1 */
    return ok;
}

/*
** ###################################################################################################################
** Operator Result Constructors
** ###################################################################################################################
*/

static mag_tensor_t* mag_result_constructor_routine_isomorph(mag_tensor_t** inputs, const mag_opparam_t* params) {
    return mag_tensor_empty_like(*inputs);
}

static mag_tensor_t* mag_result_constructor_routine_bool_isomorph(mag_tensor_t** inputs, const mag_opparam_t* params) {
    mag_tensor_t* base = *inputs;
    return mag_tensor_empty(base->ctx, MAG_DTYPE_BOOL, base->rank, base->shape);
}

bool mag_solve_view_strides(
    int64_t (*out)[MAG_MAX_DIMS],   /* Output strides */
    const int64_t* osz,             /* Old shape */
    const int64_t* ost,             /* Old strides */
    int64_t ork,                    /* Old rank */
    const int64_t* nsz,             /* New shape */
    int64_t nrk                     /* New rank */
) {
    int64_t numel = 1;
    for (int64_t i=0; i < ork; ++i)
        mag_assert2(!mag_mulov64(numel, osz[i], &numel));
    if (!numel) {
        if (!nrk) return false;
        (*out)[nrk-1] = 1;
        for (int64_t d=nrk-2; d >= 0; --d)
            mag_assert2(!mag_mulov64((*out)[d+1], nsz[d+1], &(*out)[d]));
        return true;
    }
    int64_t oi = ork-1;
    int64_t ni = nrk-1;
    while (oi >= 0 && ni >= 0) {
        if (nsz[ni] == 1) { (*out)[ni] = 0; --ni; continue; }
        for (; oi >= 0 && osz[oi] == 1; --oi);
        if (oi < 0) return false;
        if (nsz[ni] == osz[oi]) {
            (*out)[ni] = ost[oi];
            --ni; --oi;
            continue;
        }
        int64_t nc = nsz[ni];
        int64_t oc = osz[oi];
        int64_t cs = ost[oi];
        int64_t nkf = ni;
        while (nc != oc) {
            if (nc < oc) {
                --ni;
                if (ni < 0) return false;
                nc *= nsz[ni];
            } else {
                --oi;
                for (; oi >= 0 && osz[oi] == 1; --oi);
                if (oi < 0) return false;
                if (ost[oi] != osz[oi+1]*ost[oi+1])
                    return false;
                oc *= osz[oi];
            }
        }
        int64_t stride = cs;
        for (int64_t k=ni; k <= nkf; ++k) {
            (*out)[k] = stride;
            mag_assert2(!mag_mulov64(stride, nsz[k], &stride));
        }
        --ni; --oi;
    }
    while (ni >= 0) { (*out)[ni] = 0; --ni; }
    for (; oi >= 0 && osz[oi] == 1; --oi);
    return oi < 0;
}

static void mag_infer_missing_dim(int64_t(*out)[MAG_MAX_DIMS], const int64_t* dims, int64_t rank, int64_t numel) {
    int64_t known_prod = 1;
    int64_t infer_dim = -1;
    for (int64_t i=0; i < rank; ++i) {
        int64_t ax = dims[i];
        if (ax == -1) {
            mag_assert(infer_dim == -1, "only one dimension can be -1");
            infer_dim = i;
            (*out)[i] = 1;
        } else {
            mag_assert(ax > 0, "dimension must be > 0 or -1");
            (*out)[i] = ax;
            mag_assert2(!mag_mulov64(known_prod, ax, &known_prod));
        }
    }
    if (infer_dim >= 0) {
        mag_assert(numel % known_prod == 0, "cannot infer dimension size from %" PRIi64 " and known product %" PRIi64, numel, known_prod);
        (*out)[infer_dim] = numel / known_prod;
    } else {
        mag_assert(known_prod == numel, "total shape size mismatch: expected %" PRIi64 ", got %" PRIi64, numel, known_prod);
    }
}

static mag_tensor_t* mag_result_constructor_routine_view(mag_tensor_t** inputs, const mag_opparam_t* params) {
    mag_tensor_t* base = *inputs;
    int64_t rank = mag_op_param_unpack_i64_or_panic(params[0]);
    if (rank <= 0) return mag_tensor_as_strided(base->ctx, base, base->rank, base->shape, base->strides, base->storage_offset);
    int64_t new_dims[MAG_MAX_DIMS];
    for (int64_t i=0; i < rank; ++i)
        new_dims[i] = mag_op_param_unpack_i64_or_panic(params[i + 1]);
    int64_t shape[MAG_MAX_DIMS];
    mag_infer_missing_dim(&shape, new_dims, rank, base->numel);
    int64_t strides[MAG_MAX_DIMS];
    if (rank == base->rank && !memcmp(shape, base->shape, rank*sizeof(*shape))) { /* Stride strategy: same shape as base */
        memcpy(strides, base->strides, rank*sizeof(*shape));
    } else if (mag_tensor_is_contiguous(base)) { /* Stride strategy: contiguous row-major */
        strides[rank-1] = 1;
        for (int64_t d = rank-2; d >= 0; --d)
            mag_assert2(!mag_mulov64(shape[d+1], strides[d+1], strides+d));
    } else { /* Stride strategy: solve generc strides */
        bool ok = mag_solve_view_strides(&strides, base->shape, base->strides, base->rank, shape, rank);
        mag_assert(ok,
            "Tensor is not contiguous enough to be viewed. "
            "Consider calling contiguous() or reshape() instead"
        );
    }
    return mag_tensor_as_strided(base->ctx, base, rank, shape, strides, base->storage_offset);
}

static mag_tensor_t* mag_result_constructor_routine_scalar(mag_tensor_t** inputs,  const mag_opparam_t* params) {
    mag_tensor_t* base = *inputs;
    return mag_tensor_empty_scalar(base->ctx, base->dtype);
}

static mag_tensor_t* mag_result_constructor_routine_transposed(mag_tensor_t** inputs,  const mag_opparam_t* params) {
    mag_tensor_t* base = *inputs;
    int64_t ax0 = mag_op_param_unpack_i64_or_panic(params[0]);
    int64_t ax1 = mag_op_param_unpack_i64_or_panic(params[1]);
    int64_t shape[MAG_MAX_DIMS];
    int64_t stride[MAG_MAX_DIMS];
    memcpy(shape, base->shape, sizeof shape);
    memcpy(stride, base->strides, sizeof stride);
    mag_swap(int64_t, shape[ax0], shape[ax1]);
    mag_swap(int64_t, stride[ax0], stride[ax1]);
    return mag_tensor_as_strided(base->ctx, base, base->rank, shape, stride, base->storage_offset);
}

static mag_tensor_t* mag_result_constructor_routine_permuted(mag_tensor_t** inputs,  const mag_opparam_t* params) {
    mag_tensor_t* base = *inputs;
    int64_t axes[MAG_MAX_DIMS];
    for (int64_t i=0; i < base->rank; ++i)
        axes[i] = mag_op_param_unpack_i64_or_panic(params[i]);
    for (int64_t i=0; i < base->rank; ++i)
        for (int64_t j = i+1; j < base->rank; ++j)
            mag_assert(axes[i] != axes[j], "Axes must be unique: %" PRIi64 " == %" PRIi64, axes[i], axes[j]);
    int64_t shape[MAG_MAX_DIMS];
    int64_t stride[MAG_MAX_DIMS];
    for (int64_t i=0; i < base->rank; ++i) {
        shape[i] = base->shape[axes[i]];
        stride[i] = base->strides[axes[i]];
    }
    return mag_tensor_as_strided(base->ctx, base, base->rank, shape, stride, base->storage_offset);
}

static mag_tensor_t* mag_result_constructor_routine_matmul(mag_tensor_t** inputs,  const mag_opparam_t* params) { /* MxR = MxN * NxR */
    (void)params;
    mag_tensor_t* a = inputs[0];
    mag_tensor_t* b = inputs[1];
    /* Handle degenerate 1D cases first */
    if (a->rank == 1 && b->rank == 1) { /* (K)x(K) -> () */
        int64_t shape[1] = {1};
        return mag_tensor_new(a->ctx, a->dtype, 1, shape);
    }
    if (a->rank == 1 && b->rank == 2) { /* (K)x(K,N) -> (N) */
        int64_t shape[1] = {b->shape[1]};
        return mag_tensor_new(a->ctx, a->dtype, 1, shape);
    }
    if (a->rank == 2 && b->rank == 1) { /* (M,K)x(K) -> (M) */
        int64_t shape[1] = {a->shape[0]};
        return mag_tensor_new(a->ctx, a->dtype, 1, shape);
    }
    /* Batched ND version */
    int64_t a_bd = a->rank-2;
    int64_t b_bd = b->rank-2;
    int64_t r_bd = a_bd > b_bd ? a_bd : b_bd;
    int64_t shape[MAG_MAX_DIMS] = {0};
    for (int64_t i=0; i < r_bd; ++i) {
        int64_t a_dim = i < r_bd-a_bd ? 1 : a->shape[i-(r_bd-a_bd)];
        int64_t b_dim = i < r_bd-b_bd ? 1 : b->shape[i-(r_bd-b_bd)];
        shape[i] = a_dim > b_dim ? a_dim : b_dim;
    }
    shape[r_bd] = a->shape[a->rank-2];    /* M */
    shape[r_bd+1] = b->shape[b->rank-1];    /* N */
    return mag_tensor_new(a->ctx, a->dtype, r_bd+2, shape);
}

static mag_tensor_t* mag_result_constructor_routine_repeat_back(mag_tensor_t** inputs,  const mag_opparam_t* params) {
    return mag_tensor_new(inputs[0]->ctx, inputs[0]->dtype, inputs[1]->rank, inputs[1]->shape);
}

static mag_tensor_t* mag_result_constructor_routine_gather(mag_tensor_t** inputs,  const mag_opparam_t* params) {
    mag_tensor_t* src = inputs[0];
    mag_tensor_t* index = inputs[1];
    mag_assert2(index->dtype == MAG_DTYPE_I32);
    int64_t axis = mag_op_param_unpack_i64_or_panic(params[0]);
    if (axis < 0) axis += src->rank;
    mag_assert2(axis >= 0 && axis < src->rank);
    int64_t out_shape[MAG_MAX_DIMS];
    int64_t out_rank = 0;
    bool full_rank_gather = false;
    if (index->rank == src->rank) {
        full_rank_gather = true;
        for (int64_t d = 0; d < src->rank; ++d) {
            if (d == axis) continue;
            if (index->shape[d] != src->shape[d]) {
                full_rank_gather = false;
                break;
            }
        }
    }
    if (full_rank_gather) {
        for (int64_t d = 0; d < src->rank; ++d) {
            if (d == axis) {
                out_shape[out_rank++] = index->shape[d];
            } else {
                out_shape[out_rank++] = index->shape[d];
            }
        }
    } else if (index->rank == 1) {
        for (int64_t d = 0; d < src->rank; ++d) {
            if (d == axis) {
                out_shape[out_rank++] = index->shape[0];
            } else {
                out_shape[out_rank++] = src->shape[d];
            }
        }
    } else {
        for (int64_t d = 0; d < axis; ++d) {
            out_shape[out_rank++] = src->shape[d];
        }
        for (int64_t i = 0; i < index->rank; ++i) {
            out_shape[out_rank++] = index->shape[i];
        }
        for (int64_t d = axis + 1; d < src->rank; ++d) {
            out_shape[out_rank++] = src->shape[d];
        }
    }

    mag_assert2(out_rank >= 1 && out_rank <= MAG_MAX_DIMS);
    return mag_tensor_empty(src->ctx, src->dtype, out_rank, out_shape);
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

/* Execute an operator on the active compute device and return result tensor. */
static mag_tensor_t* MAG_HOTPROC mag_tensor_operator(
    mag_context_t* ctx,                 /* Context to use. All involved tensors must be allocated from this context. */
    mag_opcode_t op,                    /* Operator code */
    bool inplace,                   /* Attempt to perform inplace operation? e.g. r <- x += y instead of      r <- x + y */
    mag_tensor_t** inputs,          /* Input tensors. Must point to an array of 'num_inputs' (N) non-null tensors. */
    uint32_t num_inputs,            /* Number of valid non-null input tensors in the inputs array. Must be same as specified in the op metadata. */
    const mag_opparam_t* params,   /* Operation parameters or NULL. Must be same as specified in the op metadata. */
    uint32_t num_params,            /* Number of operation parameters. Must be same as specified in the op metadata. */
    mag_exec_stage_t stage          /* Graph evaluation direction. */
) {
    /* Assert that general operator data is correct and valid */
    mag_assert_correct_op_data(op, inputs, num_inputs, params, num_params);

    /* Query validate and result constructor functions for the scheduled opcode. */
    const mag_opmeta_t* meta = mag_op_meta_of(op);
    mag_tensor_t* (*r_alloc)(mag_tensor_t**, const mag_opparam_t*) = meta->r_alloc;                                        /* Get result allocator function. */
    bool (*validate_op)(mag_sstream_t**, bool, mag_tensor_t*, mag_tensor_t**, const mag_opparam_t*) = meta->validator;   /* Get validator function. */
    inplace &= !!(meta->flags & MAG_OP_FLAG_SUPPORTS_INPLACE);                                                              /* Inplace operation requested and supported? */

    /* Allocate result tensor and validate operation */
    mag_tensor_t* base = *inputs;
    mag_tensor_t* result = inplace ? mag_tensor_as_strided(ctx, base, base->rank, base->shape, base->strides, base->storage_offset)
        : (*r_alloc)(inputs, params);     /* If inplace, result views x (input 0), else a new result tensor is allocated. */
    result->op = op;                                                                                    /* Set opcode of result. */
    mag_sstream_t* msg = NULL;
    if (mag_unlikely(!(*validate_op)(&msg, inplace, result, inputs, params))) { /* Operation is invalid */
        const char* err = msg ? msg->buf : "Unknown error";
        FILE* out = stdout;
        fprintf(out, MAG_CC_RED "%s" MAG_CC_RESET, err); /* Print error message to stdout. */
        fflush(out);
        if (msg) { /* Free error message stream if it was created. */
            mag_sstream_free(msg);
            (*mag_alloc)(msg, 0, 0);
        }
        mag_panic("Invalid operation '%s'", meta->mnemonic);
    }

    /* Apply input tensor's gradient rules and increase their lifetime. */
    bool is_recording_grads = !!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER);
    for (uint32_t i=0; i < num_inputs; ++i) {
        mag_tensor_t* input = inputs[i];
        result->op_inputs[i] = input;
        /* If gradient tracking is enabled, the result tensor inherits the input's gradient rules. */
        if (is_recording_grads) {
            result->flags |= input->flags & MAG_TFLAG_REQUIRES_GRAD; /* Set gradient tracking flag if set in input. */
            mag_tensor_incref(input);                                /* Keep input alive for the backward pass. */
        }
    }
    if (params) /* If available, copy operation parameters to result */
        memcpy(result->op_params, params, num_params*sizeof(*params));
    mag_op_exec(result, ctx->device, stage);  /* Execute the operator. */
    if (inplace) mag_bump_version(result);   /* result aliases the modified storage */
    if (!is_recording_grads)
        mag_tensor_detach_inplace(result); /* If gradient are not recorded, detach the tensor's parents (clear parent and opcode). TODO: why are we doing this? */
    return result;
}

mag_tensor_t* mag_clone(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CLONE, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_view(mag_tensor_t* x, const int64_t* dims, int64_t rank) {
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid view dimensions count, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank));
    if (dims)
        for (int64_t i=0; i < rank; ++i)
            mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_VIEW, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_reshape(mag_tensor_t* x, const int64_t* dims, int64_t rank) {
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
    mag_assert(len > 0, "slice length is 0");
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
    mag_assert(dim1 != dim2, "transposition axes must be unequal, but: %" PRIi64 " = %" PRIi64, dim1, dim2);
    int64_t ra = x->rank;
    int64_t ax0 = dim1;
    int64_t ax1 = dim2;
    if (ax0 < 0) ax0 += ra;
    if (ax1 < 0) ax1 += ra;
    mag_assert(ax0 >= 0 && ax0 < ra, "invalid transposition axis: %" PRIi64, dim1);
    mag_assert(ax1 >= 0 && ax1 < ra, "invalid transposition axis: %" PRIi64, dim2);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(ax0));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(ax1));
    return mag_tensor_operator(x->ctx, MAG_OP_TRANSPOSE, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_permute(mag_tensor_t* x, const int64_t* dims, int64_t rank) {
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    for (int64_t i=0; i < rank; ++i)
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_PERMUTE, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mean(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (int64_t i=2; i < rank; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MEAN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_min(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (int64_t i=2; i < rank; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_max(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (int64_t i=2; i < rank; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sum(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (int64_t i=2; i < rank; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_SUM, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_argmin(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (int64_t i=2; i < rank; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_argmax(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim) {
    mag_assert(rank >= 0 && rank <= MAG_MAX_DIMS, "invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(rank)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (int64_t i=2; i < rank; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_abs(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ABS, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_abs_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ABS, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sgn(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SGN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sgn_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SGN, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_neg(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NEG, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_neg_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NEG, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_log(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_LOG, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_log_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_LOG, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqr(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQR, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqr_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQR, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqrt(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQRT, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqrt_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQRT, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sin(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sin_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIN, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_cos(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_COS, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_cos_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_COS, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_step(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_STEP, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_step_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_STEP, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_exp(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_EXP, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_exp_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_EXP, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_floor(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_FLOOR, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_floor_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_FLOOR, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_ceil(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CEIL, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_ceil_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CEIL, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_round(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ROUND, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_round_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ROUND, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_hard_sigmoid(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_hard_sigmoid_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_add(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_ADD, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_add_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_ADD, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sub(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUB, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sub_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUB, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mul(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mul_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MUL, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_div(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_DIV, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_div_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_DIV, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_matmul(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MATMUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_repeat_back(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_REPEAT_BACK, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gather(mag_tensor_t* x, int64_t dim, mag_tensor_t* idx){
    mag_assert(dim >= 0 && dim < x->rank, "gather dimension %" PRIi64 " out of range for rank %" PRIi64, dim, x->rank);
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dim)); /* Store dimension in op_params[0] */
    return mag_tensor_operator(x->ctx, MAG_OP_GATHER, false, (mag_tensor_t*[]){x, idx}, 2, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_and(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_AND, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_and_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_AND, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_or(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_OR, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_or_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_OR, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_xor(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_XOR, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_xor_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_XOR, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_not(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NOT, false, (mag_tensor_t*[]){x}, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_not_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NOT, true, (mag_tensor_t*[]){x}, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_shl(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SHL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_shl_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SHL, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_shr(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SHR, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_shr_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SHR, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_eq(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_EQ, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_ne(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_NE, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_le(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_LE, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_ge(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_GE, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_lt(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_LT, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gt(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_GT, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* _Nonnull mag_tril(mag_tensor_t* _Nonnull x, int32_t diag) {
    mag_opparam_t param = mag_op_param_wrap_i64(diag);
    return mag_tensor_operator(x->ctx, MAG_OP_TRIL, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* _Nonnull mag_tril_(mag_tensor_t* _Nonnull x, int32_t diag) {
    mag_opparam_t param = mag_op_param_wrap_i64(diag);
    return mag_tensor_operator(x->ctx, MAG_OP_TRIL, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* _Nonnull mag_triu(mag_tensor_t* _Nonnull x, int32_t diag) {
    mag_opparam_t param = mag_op_param_wrap_i64(diag);
    return mag_tensor_operator(x->ctx, MAG_OP_TRIU, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* _Nonnull mag_triu_(mag_tensor_t* _Nonnull x, int32_t diag) {
    mag_opparam_t param = mag_op_param_wrap_i64(diag);
    return mag_tensor_operator(x->ctx, MAG_OP_TRIU, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
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
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(x));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_int(mag_tensor_t* t, int32_t x) {
    mag_assert2(mag_tensor_is_integral_typed(t));
    t->init_op = MAG_IOP_BROADCAST;
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(x));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_masked_fill_float(mag_tensor_t* t, mag_tensor_t* mask, mag_e8m23_t x) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    mag_assert2(mask->dtype == MAG_DTYPE_BOOL);
    t->init_op = MAG_IOP_MASKED_BROADCAST;
    mag_opparam_layout_t layout;
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
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(x));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64((int64_t)(uintptr_t)mask));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_uniform_float(mag_tensor_t* t, mag_e8m23_t min, mag_e8m23_t max) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    t->init_op = MAG_IOP_RAND_UNIFORM;
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(min));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(max));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_uniform_int(mag_tensor_t* t, int32_t min, int32_t max) {
    mag_assert2(mag_tensor_is_integral_typed(t));
    t->init_op = MAG_IOP_RAND_UNIFORM;
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(min));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(max));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_normal(mag_tensor_t* t, mag_e8m23_t mean, mag_e8m23_t stddev) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    t->init_op = MAG_IOP_RAND_NORMAL;
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(mean));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(stddev));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_bernoulli(mag_tensor_t* t, mag_e8m23_t p) {
    mag_assert2(t->dtype == MAG_DTYPE_BOOL);
    t->init_op = MAG_IOP_RAND_BERNOULLI;
    mag_opparam_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(p));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
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
            .r_alloc = NULL,
            .validator = NULL
        },
        [MAG_OP_CLONE] = {
            .mnemonic = "clone",
            .desc = "clone",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_clone,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_VIEW] = {
            .mnemonic = "view",
            .desc = "view",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {
                MAG_OPP_I64, /* view shape count : u32 */
                MAG_OPP_I64, /* view shape : u32 */
                MAG_OPP_I64, /* view shape : u32 */
                MAG_OPP_I64, /* view shape : u32 */
                MAG_OPP_I64, /* view shape : u32 */
                MAG_OPP_I64, /* view shape : u32 */
                MAG_OPP_I64, /* view shape : u32 */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_view,
            .r_alloc = &mag_result_constructor_routine_view,
            .validator = &mag_validate_op_view
        },
        [MAG_OP_TRANSPOSE] = {
            .mnemonic = "transpose",
            .desc = "𝑥ᵀ",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {
                MAG_OPP_I64,   /* axis index 0 : u32 */
                MAG_OPP_I64,  /* axis index 1 : u32 */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_transpose,
            .r_alloc = &mag_result_constructor_routine_transposed,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_PERMUTE] = {
            .mnemonic = "permute",
            .desc = "permute",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_ALL,
            .op_param_layout = {
                MAG_OPP_I64, /* perm axis : u32 */
                MAG_OPP_I64, /* perm axis : u32 */
                MAG_OPP_I64, /* perm axis : u32 */
                MAG_OPP_I64, /* perm axis : u32 */
                MAG_OPP_I64, /* perm axis : u32 */
                MAG_OPP_I64, /* perm axis : u32 */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_permuted,
            .validator = &mag_validate_op_transpose
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
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
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
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
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
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
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
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_ABS] = {
            .mnemonic = "abs",
            .desc = "|𝑥|",
            .input_count = 1,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_abs,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary_matrix,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary_matrix,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_MATMUL] = {
            .mnemonic = "matmul",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_matmul,
            .r_alloc = &mag_result_constructor_routine_matmul,
            .validator = &mag_validate_op_matmul,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 10000
            }
        },
        [MAG_OP_REPEAT_BACK] = {
            .mnemonic = "repeat_back",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_FLOATING,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_repeat_back,
            .validator = mag_validate_op_repeat_rev
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
            .r_alloc = &mag_result_constructor_routine_gather,
            .validator = &mag_validate_op_gather
        },
        [MAG_OP_AND] = {
            .mnemonic = "and",
            .desc = "𝑥 ∧ 𝑦",
            .input_count = 2,
            .dtype_mask = MAG_DTYPE_MASK_INTEGRAL,
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_bool_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_bool_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_bool_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_bool_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_bool_isomorph,
            .validator = &mag_validate_op_binary,
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
            .r_alloc = &mag_result_constructor_routine_bool_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
    };
    return infos+opc;
}
