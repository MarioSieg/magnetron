/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
*/

#include "magnetron_internal.h"

#include <stdarg.h>

extern mag_Tensor* mag_tensor_init_internal(mag_Context* ctx, mag_DType type, int64_t rank, const int64_t* shape, mag_Tensor* view, size_t view_offs);

static mag_Tensor* mag_tensor_inplace_view(mag_Tensor* base) {
    return mag_tensor_init_internal(base->ctx, base->dtype, base->rank, base->shape, base, 0);
}

/*
** ###################################################################################################################
** Operator Validation Helpers
** ###################################################################################################################
*/

static bool mag_op_requires_op_params(mag_Operator op) { /* Returns true if the op requires any op params and thus requires validation of them. */
    const mag_OPMetadata* meta = mag_op_meta_of(op);
    for (int i=0; i < MAG_MAX_OP_PARAMS; ++i) {
        if (meta->op_param_layout[i].is_required) {
            return true;
        }
    }
    return false;
}

static void mag_assert_correct_op_data(
    mag_Operator op,
    mag_Tensor** inputs,
    uint32_t num_inputs,
    const mag_OPParam* op_params,
    uint32_t num_op_params
) {
    mag_assert(op != MAG_OP_NOP, "invalid operation: %d", op);
    const mag_OPMetadata* meta = mag_op_meta_of(op);

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
            if (meta->op_param_layout[i].is_required) { /* Only check for type equality if op param is required */
                mag_assert(op_params[i].type == meta->op_param_layout[i].type,
                    "invalid operation parameter type for operation '%s': %d != %d",
                    meta->mnemonic, op_params[i].type, meta->op_param_layout[i].type
                );
            }
        }
    }
}

static void mag_push_verification_error(mag_StrStream** ss, const char* fmt, ...) {
    if (!*ss) {  /* Lazy init error stream if needed */
        *ss = (*mag_alloc)(NULL, sizeof(**ss));
        mag_strstream_init(*ss);
    }
    va_list ap;
    va_start(ap, fmt);
    mag_strstream_vappend(*ss, fmt, ap);
    va_end(ap);
    mag_strstream_putc(*ss, '\n');
}

static bool mag_verify_is_shape_eq(mag_StrStream** ss, const mag_Tensor* x, const mag_Tensor* y) {
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
static bool mag_verify_is_min_rank(mag_StrStream** ss, const mag_Tensor* x, int64_t min_rank) {
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
static bool mag_verify_can_broadcast(mag_StrStream** ss, const mag_Tensor* x, const mag_Tensor* y) {
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
static bool mag_verify_dtype_compat(mag_StrStream** ss, mag_Operator op, mag_Tensor** inputs) {
    const mag_OPMetadata* meta = mag_op_meta_of(op);
    for (uint32_t i=0; i < meta->input_count; ++i) { /* Check that the input data types are supported by the operator. */
        const mag_DTypeSlot (*supported)[MAG_DTYPE__NUM] = &meta->input_dtypes[i];
        bool found = false;
        for (int j=0; j < sizeof(*supported)/sizeof(**supported); ++j) { /* Linear search for a supported dtype */
            if ((*supported)[j].is_used && (*supported)[j].type == inputs[i]->dtype) {
                found = true;
                break;
            }
        }
        if (mag_unlikely(!found)) {
            const char* dtype = mag_dtype_meta_of(inputs[i]->dtype)->name;
            mag_StrStream supported_dtypes;
            mag_strstream_init(&supported_dtypes);
            for (int j=0; j < sizeof(*supported)/sizeof(**supported); ++j) {
                if (!(*supported)[j].is_used) continue;
                if (j > 0) mag_strstream_append(&supported_dtypes, ", ");
                mag_strstream_append(&supported_dtypes, mag_dtype_meta_of((*supported)[j].type)->name);
            }
            mag_push_verification_error(ss,
                "Unsupported data type '%s' for operator: %s\n"
                "    Hint: Cast or convert the tensor to a supported data type: %s.\n",
                dtype, meta->mnemonic, supported_dtypes.buf
            );
            mag_strstream_free(&supported_dtypes);
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
static bool mag_verify_can_matmul(mag_StrStream** ss, const mag_Tensor* x, const mag_Tensor* y) {
    if (mag_unlikely(y->shape[0] != x->shape[1])) {
        char fmt_shape_x[MAG_FMT_DIM_BUF_SIZE];
        char fmt_shape_y[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&fmt_shape_x, &x->shape, x->rank);
        mag_fmt_shape(&fmt_shape_y, &y->shape, y->rank);
        mag_push_verification_error(ss,
            "Shape mismatch: %s cannot be matrix-multiplied with %s\n"
            "    Hint: Tensors must have compatible shapes for matrix multiplication.\n",
            fmt_shape_x, fmt_shape_y
        );
        return false;
    }
    return true;
}
static bool mag_verify_is_contiguous(mag_StrStream** ss, const mag_Tensor* x) {
    if (mag_unlikely(!mag_tensor_is_contiguous(x))) {
        mag_push_verification_error(ss,
            "Tensor is not contiguous: %s\n"
            "    Hint: Tensors must be contiguous for this operation.\n",
            x->name
        );
    }
    return true;
}
static bool mag_verify_is_inplace_and_grad_mode_off(mag_StrStream** ss, const mag_Tensor* result, bool is_inplace) {
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

static bool mag_validate_op_unary(mag_StrStream** ss, bool is_inplace, mag_Tensor* result, mag_Tensor** inputs, const mag_OPParam* params) {
    bool ok = true;
    ok = ok && mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace);     /* Check if inplace operation is allowed */
    ok = ok && mag_verify_dtype_compat(ss, result->op, inputs);                     /* Check if the operator is defined between the given dtypes */
    ok = ok && mag_verify_is_shape_eq(ss, result, inputs[0]);                   /* Check if result shape matches input */
    ok = ok && mag_verify_is_contiguous(ss, result);                                /* Check if result is contiguous */
    return ok;
}
static bool mag_validate_op_view(mag_StrStream** ss, bool is_inplace, mag_Tensor* result, mag_Tensor** inputs, const mag_OPParam* params) {
    bool ok = true;
    ok = ok && mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace);     /* Check if inplace operation is allowed */
    ok = ok && mag_verify_dtype_compat(ss, result->op, inputs);                     /* Check if the operator is defined between the given dtypes */
    ok = ok && mag_verify_is_contiguous(ss, result);                                /* Check if result is contiguous */
    ok = ok && mag_verify_is_contiguous(ss, inputs[0]);                             /* Check if x is contiguous */
    return ok;
}
static bool mag_validate_op_unary_matrix(mag_StrStream** ss, bool is_inplace, mag_Tensor* result, mag_Tensor** inputs, const mag_OPParam* params) {
    bool ok = mag_validate_op_unary(ss, is_inplace, result, inputs, params);
    ok = ok && mag_verify_is_min_rank(ss, result, 2); /* Verify that we have a matrix or higher-dimensional (rank >= 2). */
    return ok;
}
static bool mag_validate_op_binary(mag_StrStream** ss, bool is_inplace, mag_Tensor* result, mag_Tensor** inputs, const mag_OPParam* params) {
    bool ok = true;
    ok = ok && mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace);     /* Check if inplace operation is allowed */
    ok = ok && mag_verify_dtype_compat(ss, result->op, inputs);                     /* Check if the operator is defined between the given dtypes */
    ok = ok && mag_verify_is_shape_eq(ss, result, inputs[0]);                   /* Check if result shape matches first input */
    ok = ok && mag_verify_can_broadcast(ss, inputs[0], inputs[1]);              /* Check if second input can be broadcasted to first input */
    ok = ok && mag_verify_is_contiguous(ss, result);                                /* Check if result is contiguous TODO: this is always true? */
    return ok;
}
static bool mag_validate_op_transpose(mag_StrStream** ss, bool is_inplace, mag_Tensor* result, mag_Tensor** inputs, const mag_OPParam* params) {
    return true; /* TODO */
}
static bool mag_validate_op_scalar(mag_StrStream** ss, bool is_inplace, mag_Tensor* result, mag_Tensor** inputs, const mag_OPParam* params) {
    return mag_verify_is_inplace_and_grad_mode_off(ss, result, is_inplace) &&   /* Check if inplace operation is allowed */
        mag_verify_dtype_compat(ss, result->op, inputs) &&                      /* Check if the operator is defined between the given dtypes */
        mag_verify_is_contiguous(ss, inputs[0]);                                /* Check if X is contiguous */
}
static bool mag_validate_op_matmul(mag_StrStream** ss, bool is_inplace, mag_Tensor* result, mag_Tensor** inputs, const mag_OPParam* params) {
    return mag_verify_can_matmul(ss, inputs[0], inputs[1]) &&   /* Check if inputs can be matrix-multiplied */
        mag_verify_dtype_compat(ss, result->op, inputs) &&          /* Check if the operator is defined between the given dtypes */
        mag_verify_is_contiguous(ss, result);                       /* Check if result is contiguous TODO: this is always true? */
}
static bool mag_validate_op_repeat_rev(mag_StrStream** ss, bool is_inplace, mag_Tensor* result, mag_Tensor** inputs, const mag_OPParam* params) {
    return mag_verify_can_broadcast(ss, inputs[0], inputs[1]) &&    /* Check if inputs can be matrix-multiplied */
        mag_verify_dtype_compat(ss, result->op, inputs) &&              /* Check if the operator is defined between the given dtypes */
        mag_verify_is_contiguous(ss, result);                           /* Check if result is contiguous. TODO: this is always true? */
}

/*
** ###################################################################################################################
** Operator Result Constructors
** ###################################################################################################################
*/

static mag_Tensor* mag_result_constructor_routine_isomorph(mag_Tensor** inputs, const mag_OPParam* params) {
    return mag_tensor_empty_like(*inputs);
}

static mag_Tensor* mag_result_constructor_routine_bool_isomorph(mag_Tensor** inputs, const mag_OPParam* params) {
    mag_Tensor* base = *inputs;
    return mag_tensor_empty(base->ctx, MAG_DTYPE_BOOL, base->rank, base->shape);
}

static mag_Tensor* mag_result_constructor_routine_view(mag_Tensor** inputs, const mag_OPParam* params) {
    mag_Tensor* base = *inputs;
    uint32_t nd_new = (uint32_t)mag_op_param_unpack_i64_or_panic(params[0]);
    if (!nd_new)
        return mag_tensor_inplace_view(base); /* If no new dimensions are specified, return the inplace view of the base tensor. */
    int64_t old_numel = base->numel;
    int64_t canon[MAG_MAX_DIMS];
    int64_t infer_dim = -1;
    int64_t known_prod = 1;
    for (uint32_t i=0; i < nd_new; ++i) {
        int64_t s = mag_op_param_unpack_i64_or_panic(params[i+1]);
        if (s == -1) {
            mag_assert2(infer_dim == -1);
            infer_dim = i;
            canon[i] = 1;
        } else {
            mag_assert2(s > 0);
            canon[i] = s;
            known_prod *= s;
        }
    }
    if (infer_dim >= 0) {
        mag_assert2(old_numel % known_prod == 0);
        canon[infer_dim] = old_numel / known_prod;
    } else {
        mag_assert2(known_prod == old_numel);
    }
    mag_Tensor* result = mag_tensor_init_internal(base->ctx, base->dtype, nd_new, canon, base, 0);
    if (*base->name)
        mag_tensor_fmt_name(result, "%s (view)", base->name);
    return result;
}

static mag_Tensor* mag_result_constructor_routine_scalar(mag_Tensor** inputs,  const mag_OPParam* params) {
    mag_Tensor* base = *inputs;
    return mag_tensor_empty_scalar(base->ctx, base->dtype);
}

static mag_Tensor* mag_result_constructor_routine_transposed(mag_Tensor** inputs,  const mag_OPParam* params) {
    mag_Tensor* transposed = mag_tensor_inplace_view(*inputs);
    uint64_t ax0 = mag_op_param_unpack_i64_or_panic(params[0]);
    uint64_t ax1 = mag_op_param_unpack_i64_or_panic(params[1]);
    mag_swap(int64_t, transposed->shape[ax0], transposed->shape[ax1]);
    mag_swap(int64_t, transposed->strides[ax0], transposed->strides[ax1]);
    if (*inputs[0]->name)
        mag_tensor_fmt_name(transposed, "%s (T)", inputs[0]->name);
    return transposed;
}

static mag_Tensor* mag_result_constructor_routine_permuted(mag_Tensor** inputs,  const mag_OPParam* params) {
    mag_assert2(params != NULL);
    const mag_Tensor* base = inputs[0];
    mag_Tensor* permuted = mag_tensor_inplace_view(*inputs);
    int64_t axes[MAG_MAX_DIMS];
    for (int64_t i=0; i < base->rank; ++i)
        axes[i] = mag_op_param_unpack_i64_or_panic(params[i]);
    for (int64_t i=0; i < base->rank; ++i) {
        mag_assert2(axes[i] < base->rank);
        for (int64_t j=i+1; j < base->rank; ++j)
            mag_assert(axes[i] != axes[j], "Axes must be unique: %zu == %zu", axes[i], axes[j]);
    }
    int64_t tmp_shape[MAG_MAX_DIMS];
    int64_t tmp_stride[MAG_MAX_DIMS];
    memcpy(tmp_shape, base->shape, sizeof(tmp_shape));
    memcpy(tmp_stride, base->strides, sizeof(tmp_stride));
    for (int64_t i=0; i < base->rank; ++i) {
        permuted->shape[i] = tmp_shape[axes[i]];
        permuted->strides[i] = tmp_stride[axes[i]];
    }
    if (*base->name)
        mag_tensor_fmt_name(permuted, "%s (perm)", base->name);
    return permuted;
}

static mag_Tensor* mag_result_constructor_routine_matmul(mag_Tensor** inputs,  const mag_OPParam* params) { /* MxR = MxN * NxR */
    (void)params;
    int64_t shape[MAG_MAX_DIMS] = {0};
    int64_t* rd0 = shape;
    int64_t* rd1 = shape+1;
    int64_t rank = 0;
    if (inputs[0]->rank == 1 && inputs[1]->rank == 2) { /* (ℝⁿ)(ℝⁿˣʳ) → ℝʳ */
        *rd0 = 1;
        *rd1 = inputs[1]->shape[1];
        rank = 2;
    } else if (inputs[0]->rank == 2 && inputs[1]->rank == 1) { /* (ℝᵐˣⁿ)(ℝⁿ) → ℝᵐ */
        *rd0 = inputs[0]->shape[0];
        rank = 1;
    } else if (inputs[0]->rank == 1 && inputs[1]->rank == 1) { /* (ℝⁿ)(ℝⁿ) → ℝ */
        rank = 1;
    } else { /* (ℝᵐˣⁿ)(ℝⁿˣʳ) → ℝᵐˣʳ */
        *rd0 = inputs[0]->shape[0];
        *rd1 = inputs[1]->shape[1];
        rank = 2;
    }
    return mag_tensor_init_internal(inputs[0]->ctx, inputs[0]->dtype, rank, shape, NULL, 0);
}

static mag_Tensor* mag_result_constructor_routine_repeat_back(mag_Tensor** inputs,  const mag_OPParam* params) {
    return mag_tensor_init_internal(inputs[0]->ctx, inputs[0]->dtype, inputs[1]->rank, inputs[1]->shape, NULL, 0);
}

/*
** ###################################################################################################################
** Operator Backprop Impls
** ###################################################################################################################
*/

static void mag_op_backward_clone(mag_Tensor* node, mag_Tensor** grads) {
    *grads = mag_clone(node->grad);
}

static void mag_op_backward_view(mag_Tensor* node, mag_Tensor** grads) {
    *grads = mag_clone(node->grad);
}

static void mag_op_backward_transpose(mag_Tensor* node, mag_Tensor** grads) {
    uint64_t ax0 = mag_op_param_unpack_i64_or_panic(node->op_params[0]);
    uint64_t ax1 = mag_op_param_unpack_i64_or_panic(node->op_params[1]);
    *grads = mag_transpose(node->grad, ax0, ax1);
}

static void mag_op_backward_mean(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* scale = mag_tensor_full_like(x, (mag_E8M23)(1.0/(mag_E11M52)x->numel));
    *grads = mag_mul(scale, node->grad);
    mag_tensor_decref(scale);
}

static void mag_op_backward_sum(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* ones = mag_tensor_full_like(x, 1.0f);
    *grads = mag_mul(ones, node->grad);
    mag_tensor_decref(ones);
}

static void mag_op_backward_abs(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* step = mag_step(x);
    mag_Tensor* one = mag_tensor_scalar(x->ctx, x->dtype, 1.0f);
    mag_Tensor* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_Tensor* step2 = mag_mul(step, two);
    mag_Tensor* sign = mag_sub(step2, one);
    grads[0] = mag_mul(node->grad, sign);
    mag_tensor_decref(two);
    mag_tensor_decref(one);
    mag_tensor_decref(step);
    mag_tensor_decref(step2);
    mag_tensor_decref(sign);
}

static void mag_op_backward_neg(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* m1 = mag_tensor_scalar(node->grad->ctx, node->grad->dtype, -1.f);
    grads[0] = mag_mul(node->grad, m1);
    mag_tensor_decref(m1);
}

static void mag_op_backward_log(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    grads[0] = mag_div(node->grad, x);
}

static void mag_op_backward_sqr(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_Tensor* two_x = mag_mul(x, two);
    grads[0] = mag_mul(node->grad, two_x);
    mag_tensor_decref(two);
    mag_tensor_decref(two_x);
}

static void mag_op_backward_sqrt(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* sqrt_x = mag_sqrt(x);
    mag_Tensor* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_Tensor* denom = mag_mul(sqrt_x, two);
    grads[0] = mag_div(node->grad, denom);
    mag_tensor_decref(two);
    mag_tensor_decref(sqrt_x);
    mag_tensor_decref(denom);
}

static void mag_op_backward_sin(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* cos_x = mag_cos(x);
    grads[0] = mag_mul(node->grad, cos_x);
    mag_tensor_decref(cos_x);
}

static void mag_op_backward_cos(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* sinx = mag_sin(x);
    mag_Tensor* nsinx = mag_neg(sinx);
    grads[0] = mag_mul(node->grad, nsinx);
    mag_tensor_decref(sinx);
    mag_tensor_decref(nsinx);
}

static void mag_op_backward_exp(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* exp_x = mag_exp(x);
    grads[0] = mag_mul(node->grad, exp_x);
    mag_tensor_decref(exp_x);
}

static void mag_op_backward_softmax(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* y = mag_softmax(x);
    mag_Tensor* tmp = mag_mul(node->grad, y);
    mag_Tensor* sum_tmp = mag_sum(tmp, NULL, 0, false);
    mag_Tensor* diff = mag_sub(node->grad, sum_tmp);
    grads[0] = mag_mul(y, diff);
    mag_tensor_decref(tmp);
    mag_tensor_decref(sum_tmp);
    mag_tensor_decref(diff);
    mag_tensor_decref(y);
}

static void mag_op_backward_sigmoid(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* dv = mag_sigmoid_dv(x);
    grads[0] = mag_mul(dv, node->grad);
    mag_tensor_decref(dv);
}

static void mag_op_backward_silu(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* dv = mag_silu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_tanh(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* dv = mag_tanh_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_relu(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* mask = mag_step(x);
    grads[0] = mag_mul(node->grad, mask);
    mag_tensor_decref(mask);
}

static void mag_op_backward_gelu(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* dv = mag_gelu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_add(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_clone(node->grad);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_Tensor* grad = node->grad;
        if (!mag_tensor_is_shape_eq(x, y)) {
            grad = mag_repeat_back(grad, y);
        } else {
            grad = mag_clone(grad); /* Output gradients must be a new allocated tensor, so we clone. */
        }
        grads[1] = grad;
    }
}

static void mag_op_backward_sub(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_clone(node->grad);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_Tensor* mg = mag_neg(node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_Tensor* pmg = mg;
            mg = mag_repeat_back(pmg, y);
            mag_tensor_decref(pmg);
        }
        grads[1] = mg;
    }
}

static void mag_op_backward_mul(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_mul(node->grad, y);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_Tensor* xg = mag_mul(x, node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_Tensor* pxg = xg;
            xg = mag_repeat_back(pxg, y);
            mag_tensor_decref(pxg);
        }
        grads[1] = xg;
    }
}

static void mag_op_backward_div(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_div(node->grad, y);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_Tensor* gx = mag_mul(node->grad, x);
        mag_Tensor* yy = mag_mul(y, y);
        mag_Tensor* gxyy = mag_div(gx, yy);
        mag_Tensor* mgxyy = mag_neg(gxyy);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_Tensor* pmgxyy = mgxyy;
            mgxyy = mag_repeat_back(pmgxyy, y);
            mag_tensor_decref(pmgxyy);
        }
        grads[1] = mgxyy;
        mag_tensor_decref(gxyy);
        mag_tensor_decref(yy);
        mag_tensor_decref(gx);
    }
}

static void mag_op_backward_matmul(mag_Tensor* node, mag_Tensor** grads) {
    mag_Tensor* x = node->op_inputs[0];
    mag_Tensor* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_Tensor* yt = mag_transpose(y, 0, 1);
        grads[0] = mag_matmul(node->grad, yt);
        mag_tensor_decref(yt);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_Tensor* xt = mag_transpose(x, 0, 1);
        grads[1] = mag_matmul(xt, node->grad);
        mag_tensor_decref(xt);
    }
}

/* Execute init/normal operator on R. */
static void MAG_HOTPROC mag_op_exec(mag_Tensor* R, mag_IComputeDevice* dvc, mag_ExecStage stage) {
    void (*exec)(mag_IComputeDevice*, mag_Tensor*) = stage == MAG_STAGE_INIT ? dvc->eager_exec_init : dvc->eager_exec_fwd;
    (*exec)(dvc, R); /* Dispatch to backend. */
}

extern void mag_tensor_detach_inplace(mag_Tensor* target);

/* Execute an operator on the active compute device and return result tensor. */
static mag_Tensor* MAG_HOTPROC mag_tensor_operator(
    mag_Context* ctx,                 /* Context to use. All involved tensors must be allocated from this context. */
    mag_Operator op,                    /* Operator code */
    bool inplace,                   /* Attempt to perform inplace operation? e.g. r <- x += y instead of      r <- x + y */
    mag_Tensor** inputs,          /* Input tensors. Must point to an array of 'num_inputs' (N) non-null tensors. */
    uint32_t num_inputs,            /* Number of valid non-null input tensors in the inputs array. Must be same as specified in the op metadata. */
    const mag_OPParam* params,   /* Operation parameters or NULL. Must be same as specified in the op metadata. */
    uint32_t num_params,            /* Number of operation parameters. Must be same as specified in the op metadata. */
    mag_ExecStage stage          /* Graph evaluation direction. */
) {
    /* Assert that general operator data is correct and valid */
    mag_assert_correct_op_data(op, inputs, num_inputs, params, num_params);

    /* Query validate and result constructor functions for the scheduled opcode. */
    const mag_OPMetadata* meta = mag_op_meta_of(op);
    mag_Tensor* (*r_alloc)(mag_Tensor**, const mag_OPParam*) = meta->r_alloc;                                        /* Get result allocator function. */
    bool (*validate_op)(mag_StrStream**, bool, mag_Tensor*, mag_Tensor**, const mag_OPParam*) = meta->validator;   /* Get validator function. */
    inplace &= !!(meta->flags & MAG_OP_FLAG_SUPPORTS_INPLACE);                                                              /* Inplace operation requested and supported? */

    /* Allocate result tensor and validate operation */
    mag_Tensor* result = inplace ? mag_tensor_inplace_view(*inputs) : (*r_alloc)(inputs, params);     /* If inplace, result views x (input 0), else a new result tensor is allocated. */
    result->op = op;                                                                                    /* Set opcode of result. */
    mag_StrStream* msg = NULL;
    if (mag_unlikely(!(*validate_op)(&msg, inplace, result, inputs, params))) { /* Operation is invalid */
        const char* err = msg ? msg->buf : "Unknown error";
        FILE* out = stdout;
        fprintf(out, MAG_CC_RED "%s" MAG_CC_RESET, err); /* Print error message to stdout. */
        fflush(out);
        if (msg) { /* Free error message stream if it was created. */
            mag_strstream_free(msg);
            (*mag_alloc)(msg, 0);
        }
        mag_panic("Invalid operation '%s'", meta->mnemonic);
    }

    /* Apply input tensor's gradient rules and increase their lifetime. */
    bool is_recording_grads = !!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER);
    for (uint32_t i=0; i < num_inputs; ++i) {
        mag_Tensor* input = inputs[i];
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
    if (!is_recording_grads)
        mag_tensor_detach_inplace(result); /* If gradient are not recorded, detach the tensor's parents (clear parent and opcode). TODO: why are we doing this? */
    return result;
}

mag_Tensor* mag_clone(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CLONE, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_view(mag_Tensor* x, const int64_t* dims, uint32_t num_dims) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "invalid view dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(num_dims));
    if (dims)
        for (uint32_t i=0; i < num_dims; ++i)
            mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_VIEW, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_reshape(mag_Tensor* x, const int64_t* dims, uint32_t num_dims) {
  if (x->rank == num_dims && !memcmp(x->shape, dims, sizeof(*dims)*num_dims)) {
    mag_tensor_incref(x); /* Incref by one because return value of function must be a "new" tensor */
    return x;
  }
  if (mag_tensor_is_contiguous(x)) /* If cont, just view directly */
    return mag_view(x, dims, num_dims);
  mag_Tensor* base = mag_clone(x);
  mag_Tensor* view = mag_view(base, dims, num_dims);
  mag_tensor_decref(base); /* Decref so base goes out of scope, if RC > 1 base would be leaked due to lost instance */
  return view;
}

mag_Tensor* mag_view_slice(mag_Tensor* x, int64_t dim, int64_t start, int64_t len, int64_t step) {
    mag_assert(step > 0, "negative step not supported");
    mag_assert(dim  >= 0 && dim < x->rank, "dim out of range");
    int64_t sz = x->shape[dim];
    if (start < 0) start += sz;
    mag_assert(start >= 0 && start < sz, "start out of bounds");
    if (len < 0) len = sz - start;
    mag_assert(len > 0, "len must be > 0");
    mag_assert(start + (len-1)*step < sz, "slice exceeds tensor bounds");
    int64_t shape[MAG_MAX_DIMS];
    int64_t strides[MAG_MAX_DIMS];
    memcpy(shape, x->shape, sizeof(shape));
    memcpy(strides, x->strides, sizeof(strides));
    shape[dim] = len;
    strides[dim] *= step;
    size_t byte_offs = start*x->strides[dim]*mag_dtype_meta_of(x->dtype)->size;
    return mag_tensor_init_internal(x->ctx, x->dtype, x->rank, shape, x, byte_offs);
}

mag_Tensor* mag_transpose(mag_Tensor* x, int64_t dim1, int64_t dim2) {
    mag_assert(dim1 != dim2, "transposition axes must be unequal, but: %" PRIi64 " = %" PRIi64, dim1, dim2);
    int64_t ra = x->rank;
    int64_t ax0 = dim1;
    int64_t ax1 = dim2;
    if (ax0 < 0) ax0 += ra;
    if (ax1 < 0) ax1 += ra;
    mag_assert(ax0 >= 0 && ax0 < ra, "invalid transposition axis: %" PRIi64, dim1);
    mag_assert(ax1 >= 0 && ax1 < ra, "invalid transposition axis: %" PRIi64, dim2);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(ax0));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(ax1));
    return mag_tensor_operator(x->ctx, MAG_OP_TRANSPOSE, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_permute(mag_Tensor* x, const int64_t* dims, uint32_t num_dims) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "invalid permutation dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    for (uint32_t i=0; i < num_dims; ++i)
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_PERMUTE, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_mean(mag_Tensor* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MEAN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_min(mag_Tensor* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_max(mag_Tensor* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sum(mag_Tensor* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_SUM, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_argmin(mag_Tensor* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_argmax(mag_Tensor* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_Tensor* mag_abs(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ABS, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_abs_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ABS, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sgn(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SGN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sgn_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SGN, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_neg(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NEG, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_neg_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NEG, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_log(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_LOG, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_log_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_LOG, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sqr(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQR, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sqr_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQR, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sqrt(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQRT, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sqrt_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQRT, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sin(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sin_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIN, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_cos(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_COS, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_cos_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_COS, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_step(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_STEP, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_step_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_STEP, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_exp(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_EXP, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_exp_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_EXP, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_floor(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_FLOOR, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_floor_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_FLOOR, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_ceil(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CEIL, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_ceil_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CEIL, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_round(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ROUND, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_round_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ROUND, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_softmax(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_softmax_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_softmax_dv(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_softmax_dv_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sigmoid(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sigmoid_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sigmoid_dv(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sigmoid_dv_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_hard_sigmoid(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_hard_sigmoid_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_silu(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_silu_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_silu_dv(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_silu_dv_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_tanh(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_tanh_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_tanh_dv(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_tanh_dv_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_relu(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_relu_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_relu_dv(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_relu_dv_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_gelu(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_gelu_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_gelu_dv(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_gelu_dv_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_add(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_ADD, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_add_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_ADD, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sub(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUB, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_sub_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUB, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_mul(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MUL, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_mul_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MUL, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_div(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_DIV, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_div_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_DIV, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_matmul(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MATMUL, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_repeat_back(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_REPEAT_BACK, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_and(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_AND, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_and_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_AND, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_or(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_OR, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_or_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_OR, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_xor(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_XOR, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_xor_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_XOR, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_not(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NOT, false, (mag_Tensor*[]){x}, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_not_(mag_Tensor* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NOT, true, (mag_Tensor*[]){x}, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_shl(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SHL, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_shl_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SHL, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_shr(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SHR, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_shr_(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SHR, true, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_eq(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_EQ, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_ne(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_NE, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_le(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_LE, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_ge(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_GE, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_lt(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_LT, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* mag_gt(mag_Tensor* x, mag_Tensor* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_GT, false, (mag_Tensor*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_Tensor* _Nonnull mag_tril(mag_Tensor* _Nonnull x, int32_t diag) {
    mag_OPParam param = mag_op_param_wrap_i64(diag);
    return mag_tensor_operator(x->ctx, MAG_OP_TRIL, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_Tensor* _Nonnull mag_tril_(mag_Tensor* _Nonnull x, int32_t diag) {
    mag_OPParam param = mag_op_param_wrap_i64(diag);
    return mag_tensor_operator(x->ctx, MAG_OP_TRIL, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_Tensor* _Nonnull mag_triu(mag_Tensor* _Nonnull x, int32_t diag) {
    mag_OPParam param = mag_op_param_wrap_i64(diag);
    return mag_tensor_operator(x->ctx, MAG_OP_TRIU, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_Tensor* _Nonnull mag_triu_(mag_Tensor* _Nonnull x, int32_t diag) {
    mag_OPParam param = mag_op_param_wrap_i64(diag);
    return mag_tensor_operator(x->ctx, MAG_OP_TRIU, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}


void mag_tensor_fill_from_floats(mag_Tensor* t, const mag_E8M23* data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_IStorageBuffer* sto = t->storage;
    (*sto->convert)(sto, MAG_TRANSFER_DIR_H2D, 0, (void*)data, len*sizeof(*data), MAG_DTYPE_E8M23);
}

void mag_tensor_fill_from_raw_bytes(mag_Tensor* t, const void* data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_IStorageBuffer* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, 0, (void*)data, len);
}

void mag_tensor_fill_float(mag_Tensor* t, mag_E8M23 x) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    t->init_op = MAG_IOP_BROADCAST;
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(x));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_int(mag_Tensor* t, int32_t x) {
    mag_assert2(mag_tensor_is_integral_typed(t));
    t->init_op = MAG_IOP_BROADCAST;
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(x));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_uniform_float(mag_Tensor* t, mag_E8M23 min, mag_E8M23 max) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    t->init_op = MAG_IOP_RAND_UNIFORM;
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(min));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(max));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_uniform_int(mag_Tensor* t, int32_t min, int32_t max) {
    mag_assert2(mag_tensor_is_integral_typed(t));
    t->init_op = MAG_IOP_RAND_UNIFORM;
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(min));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_i64(max));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_normal(mag_Tensor* t, mag_E8M23 mean, mag_E8M23 stddev) {
    mag_assert2(mag_tensor_is_floating_point_typed(t));
    t->init_op = MAG_IOP_RAND_NORMAL;
    mag_OPParamLayout layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(mean));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(stddev));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_bernoulli(mag_Tensor* t, mag_E8M23 p) {
    mag_assert2(t->dtype == MAG_DTYPE_BOOL);
    t->init_op = MAG_IOP_RAND_BERNOULLI;
    mag_OPParamLayout layout;
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

const mag_OPMetadata* mag_op_meta_of(mag_Operator opc) {
    static const mag_OPMetadata infos[MAG_OP__NUM] = {
        [MAG_OP_NOP] = {
            .mnemonic = "nop",
            .desc = "nop",
            .input_count = 0,
            .input_dtypes = {},
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true}, /* view shape count : u32 */
                {.type=MAG_OPP_I64, .is_required=false}, /* view shape : u32 */
                {.type=MAG_OPP_I64, .is_required=false}, /* view shape : u32 */
                {.type=MAG_OPP_I64, .is_required=false}, /* view shape : u32 */
                {.type=MAG_OPP_I64, .is_required=false}, /* view shape : u32 */
                {.type=MAG_OPP_I64, .is_required=false}, /* view shape : u32 */
                {.type=MAG_OPP_I64, .is_required=false}, /* view shape : u32 */
                {.type=MAG_OPP_NONE, .is_required=false},
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true},   /* axis index 0 : u32 */
                {.type=MAG_OPP_I64, .is_required=true},  /* axis index 1 : u32 */
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_I64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_I64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_I64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_I64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_I64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true},  /* reduction dim count : u32 */
                {.type=MAG_OPP_I64, .is_required=true},  /* keepdim : bool */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true},  /* reduction dim count : u32 */
                {.type=MAG_OPP_I64, .is_required=true},  /* keepdim : bool */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true},  /* reduction dim count : u32 */
                {.type=MAG_OPP_I64, .is_required=true},  /* keepdim : bool */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true},  /* reduction dim count : u32 */
                {.type=MAG_OPP_I64, .is_required=true},  /* keepdim : bool */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_I64, .is_required=false}, /* reduction dim : u32 [optional] */
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true},
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                }
            },
            .op_param_layout = {
                {.type=MAG_OPP_I64, .is_required=true},
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                }
            },
            .op_param_layout = {},
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_repeat_back,
            .validator = mag_validate_op_repeat_rev
        },
        [MAG_OP_AND] = {
            .mnemonic = "and",
            .desc = "𝑥 ∧ 𝑦",
            .input_count = 2,
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_BOOL, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
            .input_dtypes = {
                [0] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                },
                [1] = {
                    {.type=MAG_DTYPE_E8M23, .is_used=true},
                    {.type=MAG_DTYPE_E5M10, .is_used=true},
                    {.type=MAG_DTYPE_I32, .is_used=true},
                }
            },
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
