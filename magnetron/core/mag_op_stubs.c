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
#include "mag_reduce_plan.h"

mag_scalar_t mag_scalar_float(double value) {
    return (mag_scalar_t){.type = MAG_SCALAR_TYPE_F64, .value.f64 = value};
}

mag_scalar_t mag_scalar_int(int64_t value) {
    return (mag_scalar_t){.type = MAG_SCALAR_TYPE_I64, .value.i64 = value};
}

mag_scalar_t mag_scalar_uint(uint64_t value) {
    return (mag_scalar_t){.type = MAG_SCALAR_TYPE_U64, .value.u64 = value};
}

bool mag_scalar_is_f64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_F64; }

bool mag_scalar_is_i64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_I64; }

bool mag_scalar_is_u64(mag_scalar_t s) { return s.type == MAG_SCALAR_TYPE_U64; }

double mag_scalar_as_f64(mag_scalar_t s) {
    switch (s.type) {
        case MAG_SCALAR_TYPE_F64: return s.value.f64;
        case MAG_SCALAR_TYPE_I64: return (double)s.value.i64;
        case MAG_SCALAR_TYPE_U64: return (double)s.value.u64;
        default: mag_panic("Invalid scalar type tag: %d", s.type);
    }
}

int64_t mag_scalar_as_i64(mag_scalar_t s) {
    switch (s.type) {
        case MAG_SCALAR_TYPE_I64: return s.value.i64;
        case MAG_SCALAR_TYPE_U64: return (int64_t)s.value.u64;
        case MAG_SCALAR_TYPE_F64: return (int64_t)s.value.f64;
        default: mag_panic("Invalid scalar type tag: %d", s.type);
    }
}

uint64_t mag_scalar_as_u64(mag_scalar_t s) {
    switch (s.type) {
        case MAG_SCALAR_TYPE_U64: return s.value.u64;
        case MAG_SCALAR_TYPE_I64: return (uint64_t)s.value.i64;
        case MAG_SCALAR_TYPE_F64: return (uint64_t)s.value.f64;
        default: mag_panic("Invalid scalar type tag: %d", s.type);
    }
}

static mag_op_attr_t mag_scalar_to_op_attr(mag_dtype_t dtype, mag_scalar_t x) {
    mag_dtype_mask_t dtb = mag_dtype_bit(dtype);
    if (dtb & MAG_DTYPE_MASK_FP) return mag_op_attr_float64(mag_scalar_as_f64(x));
    if (dtb & MAG_DTYPE_MASK_SINT) return mag_op_attr_int64(mag_scalar_as_i64(x));
    if (dtb & MAG_DTYPE_MASK_UINT || dtype == MAG_DTYPE_BOOLEAN) return mag_op_attr_uint64(mag_scalar_as_u64(x));
    mag_panic("Unsupported dtype %s for scalar conversion", mag_type_trait(dtype)->name);
}

static bool mag_scalar_same_type(mag_scalar_t a, mag_scalar_t b) {
    return a.type == b.type;
}

static void mag_norm_axis(int64_t *ax, int64_t ra) {
    if (*ax < 0) *ax += ra;
}

static const mag_dtype_t mag_type_promotion_rules[MAG_DTYPE__NUM][MAG_DTYPE__NUM] = {
    [MAG_DTYPE_FLOAT32] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_FLOAT32,
    },
    [MAG_DTYPE_FLOAT16] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_FLOAT32,
    },
    [MAG_DTYPE_BOOLEAN] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_BOOLEAN,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT8,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT8,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_UINT8] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_UINT8,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT8,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_INT8] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_INT8,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT8,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_INT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_UINT16] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_UINT16,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT16,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_INT16] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_INT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT16,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_UINT32] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_UINT32,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT32,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_INT32] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_INT32,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT32,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_UINT64] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_UINT64,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_UINT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
    [MAG_DTYPE_INT64] = {
        [MAG_DTYPE_FLOAT32] = MAG_DTYPE_FLOAT32,
        [MAG_DTYPE_FLOAT16] = MAG_DTYPE_FLOAT16,
        [MAG_DTYPE_BOOLEAN] = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT8]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT8]    = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT16]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT16]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT32]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT32]   = MAG_DTYPE_INT64,
        [MAG_DTYPE_UINT64]  = MAG_DTYPE_INT64,
        [MAG_DTYPE_INT64]   = MAG_DTYPE_INT64,
    },
};

static bool mag_promote_type(mag_dtype_t *out, mag_dtype_t lhs, mag_dtype_t rhs) {
    if (mag_unlikely(lhs >= MAG_DTYPE__NUM || rhs >= MAG_DTYPE__NUM)) return false;
    *out = mag_type_promotion_rules[lhs][rhs];
    return *out < MAG_DTYPE__NUM;
}

static bool mag_op_requires_op_params(mag_opcode_t op) { /* Returns true if the op requires any op params and thus requires validation of them. */
    const mag_op_traits_t *meta = mag_op_traits(op);
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
    const mag_op_traits_t *meta = mag_op_traits(op);

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

static mag_status_t mag_tensor_strided_view(mag_tensor_t **out_result, mag_tensor_t *base) {
    return mag_as_strided(out_result, base->ctx, base, base->coords.rank, base->coords.shape, base->coords.strides, base->storage_offset);
}

static void MAG_COLDPROC mag_dbg_trace_op_ir(mag_opcode_t op, bool inplace, mag_tensor_t **in, uint32_t num_in, mag_tensor_t **out, uint32_t num_out) {
    const mag_op_traits_t *meta = mag_op_traits(op);
    const char *dvc = in ? (*in)->ctx->device->id : (*out)->ctx->device->id;
    bool cont = true;
    for (uint32_t i=0; i < num_in; ++i) cont &= mag_tensor_is_contiguous(in[i]);
    for (uint32_t i=0; i < num_out; ++i) cont &= mag_tensor_is_contiguous(out[i]);
    char opcode[64];
    snprintf(opcode, sizeof(opcode), "%s", meta->mnemonic);
    for (char *p=opcode; *p; ++p) if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) *p = (char)(*p|0x20);
    printf("%02X %s.%s%s%s ", op, opcode, cont ? "cont." : "", inplace ? "inplace." : "", dvc);
    for (uint32_t i=0; i < num_in; ++i) {
        mag_tensor_t *tensor = in[i];
        printf("in[%u](%s,", i, mag_type_trait(tensor->dtype)->name);
        for (int64_t j=0; j < tensor->coords.rank; ++j) {
            printf("%" PRIi64, tensor->coords.shape[j]);
            if (j+1 < tensor->coords.rank) printf("x");
        }
        printf(") ");
    }
    printf("-> ");
    for (uint32_t i=0; i < num_out; ++i) {
        mag_tensor_t *tensor = out[i];
        printf("out[%u](%s,", i, mag_type_trait(tensor->dtype)->name);
        for (int64_t j=0; j < tensor->coords.rank; ++j) {
            printf("%" PRIi64, tensor->coords.shape[j]);
            if (j+1 < tensor->coords.rank) printf("x");
        }
        printf(") ");
    }
    printf("\n");
}

/* Execute an operator on the active compute device and return result tensor. */
static void MAG_HOTPROC mag_dispatch(mag_opcode_t op, bool inplace, const mag_op_attr_registry_t *layout, mag_tensor_t **in, uint32_t num_in, mag_tensor_t **out, uint32_t num_out) {
    const mag_op_traits_t *meta = mag_op_traits(op);
    mag_assert2((in && num_in) || (out && num_out));
    mag_assert2(op != MAG_OP_NOP);
#if 0 /* Debug: print dispatched ops */
    mag_dbg_trace_op_ir(op, inplace, in, num_in, out, num_out);
#endif
    mag_context_t *ctx = in ? (*in)->ctx : (*out)->ctx;
    const mag_op_attr_t *params = layout ? layout->slots : NULL;
    uint32_t num_params = layout ? layout->count : 0;
    mag_assert_correct_op_data(op, in, num_in, out, num_out, params, num_params);
    if (!!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER) && meta->backward) {
        for (uint32_t i=0; i < num_out; ++i) {
            mag_tensor_t *r = out[i];
            mag_au_state_t *au = mag_au_state_lazy_alloc(&r->au_state, r->ctx);
            au->op = op;
            for (uint32_t j=0; j < num_in; ++j) {
                mag_tensor_t *input = in[j];
                au->op_inputs[j] = input;
                if (input->flags & MAG_TFLAG_REQUIRES_GRAD &&
                    !(r->flags & MAG_TFLAG_REQUIRES_GRAD))
                    mag_tensor_set_requires_grad(r, true);
                mag_rc_incref(input);
            }
            if (params)
                memcpy(au->op_attrs, params, num_params * sizeof(*params));
        }
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
    }
    ++ctx->ops_dispatched;
}

static void mag_assert_dtype_compat(mag_opcode_t op, mag_tensor_t **inputs) {
    const mag_op_traits_t *meta = mag_op_traits(op);
    for (uint32_t i=0; i < meta->in; ++i) { /* Check that the input data types are supported by the operator. */
        bool supported = meta->dtype_mask & mag_dtype_bit(inputs[i]->dtype);
        if (mag_unlikely(!supported)) {
            const char *dtype = mag_type_trait(inputs[i]->dtype)->name;
            mag_panic(
                "Data type '%s' is not supported by operator '%s'.\n"
                "    Hint: Use a different data type or operator.\n",
                dtype, meta->mnemonic
            );
        }
    }
    if (mag_unlikely(meta->in == 2 && inputs[0]->dtype != inputs[1]->dtype)) { /* For binary operators, check that both inputs have the same data type. */
        const char *dtype_x = mag_type_trait(inputs[0]->dtype)->name;
        const char *dtype_y = mag_type_trait(inputs[1]->dtype)->name;
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

mag_status_t mag_empty_like(mag_tensor_t **out_result, mag_tensor_t *like) {
    return mag_empty(out_result, like->ctx, like->dtype, like->coords.rank, like->coords.shape);
}

mag_status_t mag_empty_scalar(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type) {
    return mag_empty(out_result, ctx, type, 0, NULL);
}

mag_status_t mag_scalar(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t value) {
    mag_status_t stat = mag_empty_scalar(out_result, ctx, type);
    if (mag_iserr(stat)) return stat;
    mag_fill_(*out_result, value);
    return MAG_STATUS_OK;
}

mag_status_t mag_full(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t value) {
    mag_status_t stat = mag_empty(out_result, ctx, type, rank, shape);
    if (mag_iserr(stat)) return stat;
    mag_fill_(*out_result, value);
    return MAG_STATUS_OK;
}

mag_status_t mag_full_like(mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t value) {
    mag_status_t stat = mag_empty_like(out_result, like);
    if (mag_iserr(stat)) return stat;
    mag_fill_(*out_result, value);
    return MAG_STATUS_OK;
}

mag_status_t mag_zeros(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape) {
    return mag_full(out_result, ctx, type, rank, shape, mag_scalar_uint(0));
}

mag_status_t mag_zeros_like(mag_tensor_t **out_result, mag_tensor_t *like) {
    return mag_full_like(out_result, like, mag_scalar_uint(0));
}

mag_status_t mag_ones(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape) {
    return mag_full(out_result, ctx, type, rank, shape, mag_scalar_uint(0));
}

mag_status_t mag_ones_like(mag_tensor_t **out_result, mag_tensor_t *like) {
    return mag_full_like(out_result, like, mag_scalar_uint(0));
}

mag_status_t mag_uniform(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t min, mag_scalar_t max) {
    mag_status_t stat = mag_empty(out_result, ctx, type, rank, shape);
    if (mag_iserr(stat)) return stat;
    return mag_uniform_(*out_result, min, max);
}

mag_status_t mag_uniform_like(mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t min, mag_scalar_t max) {
    mag_status_t stat = mag_empty_like(out_result, like);
    if (mag_iserr(stat)) return stat;
    return mag_uniform_(*out_result, min, max);
}

mag_status_t mag_normal(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t mean, mag_scalar_t stddev) {
    mag_status_t stat = mag_empty(out_result, ctx, type, rank, shape);
    if (mag_iserr(stat)) return stat;
    return mag_normal_(*out_result, mean, stddev);
}

mag_status_t mag_normal_like(mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t mean, mag_scalar_t stddev) {
    mag_status_t stat = mag_empty_like(out_result, like);
    if (mag_iserr(stat)) return stat;
    return mag_normal_(*out_result, mean, stddev);
}

mag_status_t mag_bernoulli(mag_tensor_t **out_result, mag_context_t *ctx, int64_t rank, const int64_t *shape, mag_scalar_t p) {
    mag_status_t stat = mag_empty(out_result, ctx, MAG_DTYPE_BOOLEAN, rank, shape);
    if (mag_iserr(stat)) return stat;
    return mag_bernoulli_(*out_result, p);
}

mag_status_t mag_bernoulli_like(mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t p) {
    mag_status_t stat = mag_empty(out_result, like->ctx, MAG_DTYPE_BOOLEAN, like->coords.rank, like->coords.shape);
    if (mag_iserr(stat)) return stat;
    return mag_bernoulli_(*out_result, p);
}

static bool mag_arange_numel_int(int64_t start, int64_t stop, int64_t step, int64_t *numel) {
    if (step == 0) {
        *numel = 0;
        return false;
    }
    int64_t delta = stop - start;
    if (step > 0) {
        if (delta <= 0) {
            *numel = 0;
            return true;
        }
        *numel = (delta + step - 1)/step;
        return true;
    }
    if (delta >= 0) {
        *numel = 0;
        return true;
    }
    int64_t step_pos = -step;
    int64_t diff_pos = -delta;
    *numel = (diff_pos + step_pos - 1)/step_pos;
    return true;
}

static bool mag_arange_numel_float(double start, double end, double step, int64_t *numel) {
    if (step == 0.0) {
        *numel = 0;
        return false;
    }
    double delta = end - start;
    if ((step > 0.0 && delta <= 0.0) || (step < 0.0 && delta >= 0.0)) {
        *numel = 0;
        return true;
    }
    double nc = ceil(delta/step - 1e-12);
    if (nc <= 0.0) {
        *numel = 0;
        return true;
    }
    if (nc > (double)INT64_MAX) return false;
    *numel = (int64_t)nc;
    return true;
}

mag_status_t mag_arange(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t start, mag_scalar_t end, mag_scalar_t step) {
    *out_result = NULL;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_scalar_same_type(start, end) && mag_scalar_same_type(start, step), "Start, end and step scalars must have the same type");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_dtype_bit(type) & MAG_DTYPE_MASK_NUMERIC, "Data type must be numeric");
    mag_tensor_t *result;
    int64_t numel = 0;
    bool ok = false;
    if (mag_dtype_bit(type) & MAG_DTYPE_MASK_INTEGER) ok = mag_arange_numel_int(mag_scalar_as_i64(start), mag_scalar_as_i64(end), mag_scalar_as_i64(step), &numel);
    else ok = mag_arange_numel_float(mag_scalar_as_f64(start), mag_scalar_as_f64(end), mag_scalar_as_f64(step), &numel);
    if (mag_unlikely(!ok) || numel <= 0) {
       mag_contract(ctx, ERR_INVALID_PARAM, {}, false, "Invalid parameters for arange");
       return MAG_STATUS_ERR_INVALID_PARAM;
    }
    mag_status_t stat = mag_empty(&result, ctx, type, 1, &numel);
    if (mag_iserr(stat)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_float64(mag_scalar_as_f64(start))); /* TODO: this looses information for int64/uint64 ranges that exceed f64 precision */
    mag_op_attr_registry_insert(&layout, mag_op_attr_float64(mag_scalar_as_f64(step)));
    mag_dispatch(MAG_OP_ARANGE, false, &layout, NULL, 0, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_rand_perm(mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t n) {
    *out_result = NULL;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_dtype_bit(type) & MAG_DTYPE_MASK_INTEGER, "Data type must be integer");
    mag_tensor_t *result;
    mag_status_t stat = mag_empty(&result, ctx, type, 1, &n);
    if (mag_iserr(stat)) return stat;
    mag_dispatch(MAG_OP_RAND_PERM, false, NULL, NULL, 0, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_clone(mag_tensor_t **out_result, mag_tensor_t *x) {
    *out_result = NULL;
    mag_tensor_t *result;
    mag_status_t stat = mag_empty_like(&result, x);
    if (mag_iserr(stat)) return stat;
    mag_dispatch(MAG_OP_CLONE, false, NULL, &x, 1, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_cast(mag_tensor_t **out_result, mag_tensor_t *x, mag_dtype_t dst_type) {
    if (x->dtype == dst_type) return mag_clone(out_result, x); /* If dtypes match, we just clone */
    *out_result = NULL;
    mag_tensor_t *result;
    mag_status_t stat = mag_empty(&result, x->ctx, dst_type, x->coords.rank, x->coords.shape);
    if (mag_iserr(stat)) return stat;
        mag_dispatch(MAG_OP_CAST, false, NULL, &x, 1, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_view(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank >= 0 && rank <= MAG_MAX_DIMS, "Invalid dimensions rank, must be [0, %d], but is %" PRIi64, MAG_MAX_DIMS, rank);
    if (rank == 0) {
        mag_contract(ctx, ERR_INVALID_PARAM, {}, x->numel == 1, "view([]) only allowed for tensors with numel == 1, got %" PRIi64, x->numel);
        stat = mag_as_strided(&result, x->ctx, x, 0, NULL, NULL, x->storage_offset);
        if (mag_iserr(stat)) return stat;
    } else {
        mag_contract(ctx, ERR_INVALID_PARAM, {}, dims != NULL, "Dims cannot be NULL if rank > 0");
        int64_t oshape[MAG_MAX_DIMS] = {0};
        memcpy(oshape, dims, rank*sizeof(*dims));
        int64_t shape[MAG_MAX_DIMS];
        mag_contract(ctx, ERR_INVALID_DIM, {}, mag_infer_missing_dim(&shape, oshape, rank, x->numel), "Cannot infer missing dimension for view");
        int64_t strides[MAG_MAX_DIMS];
        if (rank == x->coords.rank && !memcmp(shape, x->coords.shape, rank*sizeof(*shape))) { /* Stride strategy: same shape as base */
            memcpy(strides, x->coords.strides, rank*sizeof(*shape));
        } else if (rank == x->coords.rank+1 && shape[rank-2]*shape[rank-1] == x->coords.shape[x->coords.rank-1]) { /* Stride strategy: last dim only */
            memcpy(strides, x->coords.strides, (rank-2)*sizeof(*strides));
            strides[rank-2] = x->coords.strides[x->coords.rank-1]*shape[rank-1];
            strides[rank-1] = x->coords.strides[x->coords.rank-1];
        } else if (mag_tensor_is_contiguous(x)) { /* Stride strategy: contiguous row-major */
            strides[rank-1] = 1;
            for (int64_t i=rank-2; i >= 0; --i) {
                mag_contract(ctx, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[i+1], strides[i+1], strides+i), "Dimension overflow when calculating strides for view");
            }
        } else { /* Stride strategy: solve generic strides */
            mag_contract(ctx, ERR_STRIDE_SOLVER_FAILED, {}, mag_solve_view_strides(&strides, x->coords.shape, x->coords.strides, x->coords.rank, shape, rank),
               "Tensor is not contiguous enough to be viewed\n"
               "Consider calling contiguous() or reshape() instead"
            );
        }
        stat = mag_as_strided(&result, x->ctx, x, rank, shape, strides, x->storage_offset);
        if (mag_iserr(stat)) return stat;
    }

    mag_dispatch(MAG_OP_VIEW, false, NULL, &x, 1, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_reshape(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    int64_t shape[MAG_MAX_DIMS];
    mag_contract(ctx, ERR_INVALID_DIM, {}, mag_infer_missing_dim(&shape, dims, rank, x->numel), "Cannot infer missing dimension for reshape");
    if (x->coords.rank == rank && !memcmp(x->coords.shape, shape, sizeof(*dims)*rank)) {
        mag_rc_incref(x);
        *out_result = x;
        return MAG_STATUS_OK;
    }
    if (mag_tensor_is_contiguous(x)) {
        int64_t strides[MAG_MAX_DIMS];
        strides[rank-1] = 1;
        for (int64_t i=rank-2; i >= 0; --i) {
            mag_contract(ctx, ERR_DIM_OVERFLOW, {}, !mag_mulov64(shape[i+1], strides[i+1], strides+i), "Dimension overflow when calculating strides for reshape")
        }
        stat = mag_as_strided(&result, x->ctx, x, rank, shape, strides, x->storage_offset);
        if (mag_iserr(stat)) return stat;
        *out_result = result;
        return MAG_STATUS_OK;
    }
    if (mag_tensor_can_view(x, shape, rank)) {
        stat = mag_view(&result, x, shape, rank);
        if (mag_iserr(stat)) return stat;
        *out_result = result;
        return MAG_STATUS_OK;
    }
    stat = mag_contiguous(&result, x);
    if (mag_iserr(stat)) return stat;
    int64_t strides[MAG_MAX_DIMS];
    strides[rank-1] = 1;
    for (int64_t i=rank-2; i >= 0; --i)
        mag_assert2(!mag_mulov64(shape[i+1], strides[i+1], strides+i));
    mag_tensor_t *reshaped;
    stat = mag_as_strided(&reshaped, result->ctx, result, rank, shape, strides, result->storage_offset);
    if (mag_iserr(stat)) {
        mag_rc_decref(result);
        return stat;
    }
    mag_rc_decref(result);
    *out_result = reshaped;
    return MAG_STATUS_OK;
}

mag_status_t mag_view_slice(mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t len, int64_t step) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0, "Cannot slice scalar tensor");
    mag_norm_axis(&dim, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= dim && dim < rank, "Dim %" PRIi64 " out of range for rank %" PRIi64, dim, rank);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, step > 0, "Slice step must be > 0, got %" PRIi64, step);
    int64_t sz = x->coords.shape[dim];
    mag_norm_axis(&start, sz);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, 0 <= start && start < sz, "Slice start out of bounds for dim %" PRIi64 ": %" PRIi64 " (size=%" PRIi64 ")", dim, start, sz);
    if (len < 0) len = (sz - start + step - 1)/step;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, len > 0, "Slice length must be > 0, got %" PRIi64, len);
    int64_t last = start + (len - 1)*step;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, 0 <= last && last < sz, "Slice exceeds bounds for dim %" PRIi64 ": last index %" PRIi64 " >= %" PRIi64, dim, last, sz);
    int64_t shape[MAG_MAX_DIMS];
    int64_t strides[MAG_MAX_DIMS];
    memcpy(shape, x->coords.shape, rank*sizeof(*shape));
    memcpy(strides, x->coords.strides, rank*sizeof(*strides));
    shape[dim] = len;
    strides[dim] = x->coords.strides[dim] * step;
    int64_t offset = x->storage_offset + start*x->coords.strides[dim];
    return mag_as_strided(out_result, x->ctx, x, rank, shape, strides, offset);
}

mag_status_t mag_transpose(mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim1, int64_t dim2) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, x->coords.rank >= 2, "Transpose requires rank >= 2, but got: %" PRIi64, x->coords.rank);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, dim1 != dim2, "Transposition axes must be unequal, but: %" PRIi64 " = %" PRIi64, dim1, dim2);
    int64_t ra = x->coords.rank;
    int64_t ax0 = dim1;
    int64_t ax1 = dim2;
    mag_norm_axis(&ax0, ra);
    mag_norm_axis(&ax1, ra);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, ax0 >= 0 && ax0 < ra, "Invalid transposition axis: %" PRIi64, dim1);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, ax1 >= 0 && ax1 < ra, "Invalid transposition axis: %" PRIi64, dim2);
    int64_t shape[MAG_MAX_DIMS];
    int64_t stride[MAG_MAX_DIMS];
    memcpy(shape, x->coords.shape, sizeof shape);
    memcpy(stride, x->coords.strides, sizeof stride);
    mag_swap(int64_t, shape[ax0], shape[ax1]);
    mag_swap(int64_t, stride[ax0], stride[ax1]);
    stat = mag_as_strided(&result, x->ctx, x, x->coords.rank, shape, stride, x->storage_offset);
    if (mag_iserr(stat)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(ax0));
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(ax1));
    mag_dispatch(MAG_OP_TRANSPOSE, false, &layout, &x, 1, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_permute(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank) {
    *out_result = NULL;
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
    stat = mag_as_strided(&result, x->ctx, x, x->coords.rank, shape, stride, x->storage_offset);
    if (mag_iserr(stat)) return stat;
    mag_dispatch(MAG_OP_PERMUTE, false, NULL, &x, 1, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_contiguous(mag_tensor_t **out_result, mag_tensor_t *x) {
    if (!x->storage_offset && mag_tensor_is_contiguous(x)) {
        mag_rc_incref(x); /* If already contiguous, just incref */
        *out_result = x;
        return MAG_STATUS_OK;
    }
    return mag_clone(out_result, x);
}

mag_status_t mag_squeeze_all(mag_tensor_t **out_result, mag_tensor_t *x) {
    *out_result = NULL;
    int64_t rank = x->coords.rank;
    if (!rank) return mag_view(out_result, x, x->coords.shape, 0);
    int64_t shape[MAG_MAX_DIMS];
    int64_t nrank = 0;
    for (int64_t i=0; i < rank; ++i) {
        int64_t sz = x->coords.shape[i];
        if (sz != 1) shape[nrank++] = sz;
    }
    return nrank == rank ? mag_view(out_result, x, x->coords.shape, rank) : mag_view(out_result, x, shape, nrank);
}

mag_status_t mag_squeeze_dim(mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0, "Cannot squeeze dim of scalar tensor");
    mag_norm_axis(&dim, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= dim && dim < rank, "Dim %" PRIi64 " out of range for rank %" PRIi64, dim, rank);
    int64_t sz = x->coords.shape[dim];
    if (sz != 1) return mag_view(out_result, x, x->coords.shape, rank);
    int64_t shape[MAG_MAX_DIMS];
    int64_t nrank = 0;
    for (int64_t i=0; i < rank; ++i) {
        if (i == dim) continue;
        shape[nrank++] = x->coords.shape[i];
    }
    return mag_view(out_result, x, shape, nrank);
}

mag_status_t mag_unsqueeze(mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    int64_t nrank = rank+1;
    mag_contract(ctx, ERR_INVALID_RANK, {}, nrank <= MAG_MAX_DIMS, "Unsqueeze would exceed MAG_MAX_DIMS (%d)", MAG_MAX_DIMS);
    mag_norm_axis(&dim, nrank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= dim && dim < nrank, "Unsqueeze dim %" PRIi64 " out of range for new rank %" PRIi64, dim, nrank);
    int64_t shape[MAG_MAX_DIMS];
    for (int64_t i=0, j=0; i < nrank; ++i)
        shape[i] = i == dim ? 1 : x->coords.shape[j++];
    return mag_view(out_result, x, shape, nrank);
}

mag_status_t mag_flatten(mag_tensor_t **out_result, mag_tensor_t *x, int64_t start_dim, int64_t end_dim) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    if (!rank) return mag_view(out_result, x, x->coords.shape, 0);
    mag_norm_axis(&start_dim, rank);
    mag_norm_axis(&end_dim, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= start_dim && start_dim < rank, "Flatten start_dim %" PRIi64 " out_result of range for rank %" PRIi64, start_dim, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= end_dim && end_dim < rank, "Flatten end_dim %" PRIi64 " out_result of range for rank %" PRIi64, end_dim, rank);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, start_dim <= end_dim, "Flatten requires start_dim <= end_dim (got %" PRIi64 " > %" PRIi64 ")", start_dim, end_dim);
    int64_t shape[MAG_MAX_DIMS];
    int64_t nrank = 0;
    for (int64_t i=0; i < start_dim; ++i)
        shape[nrank++] = x->coords.shape[i];
    int64_t sz=1;
    for (int64_t i=start_dim; i <= end_dim; ++i) {
        sz *= x->coords.shape[i];
    }
    shape[nrank++] = sz;
    for (int64_t i=end_dim+1; i < rank; ++i) {
        shape[nrank++] = x->coords.shape[i];
    }
    mag_contract(ctx, ERR_INVALID_RANK, {}, nrank <= MAG_MAX_DIMS, "Flatten result rank %" PRIi64 " exceeds MAG_MAX_DIMS (%d)", nrank, MAG_MAX_DIMS);
    mag_status_t stat = mag_view(out_result, x, shape, nrank); /* Try view first */
    if (mag_iserr(stat))
        stat = mag_reshape(out_result, x, shape, nrank);
    return stat;
}

mag_status_t mag_unflatten(mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, const int64_t *sizes, int64_t sizes_rank) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, sizes_rank > 0, "unflatten requires sizes_rank > 0");
    mag_norm_axis(&dim, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= dim && dim < rank, "Unflatten dim %" PRIi64 " out_result of range for rank %" PRIi64, dim, rank);
    int64_t dim_sz = x->coords.shape[dim];
    int64_t prod = 1;
    for (int64_t i=0; i < sizes_rank; ++i) {
        mag_contract(ctx, ERR_INVALID_PARAM, {}, sizes[i] > 0, "unflatten sizes[% " PRIi64 "] must be > 0, got %" PRIi64, i, sizes[i]);
        prod *= sizes[i];
    }
    mag_contract(ctx, ERR_INVALID_PARAM, {}, prod == dim_sz, "unflatten sizes product %" PRIi64 " does not match dim size %" PRIi64, prod, dim_sz);
    int64_t nr = rank - 1 + sizes_rank;
    mag_contract(ctx, ERR_INVALID_RANK, {}, nr <= MAG_MAX_DIMS, "Unflatten result rank %" PRIi64 " exceeds MAG_MAX_DIMS (%d)", nr, MAG_MAX_DIMS);
    int64_t shape[MAG_MAX_DIMS];
    int64_t k=0;
    for (int64_t i=0; i < dim; ++i) shape[k++] = x->coords.shape[i];
    for (int64_t i=0; i < sizes_rank; ++i) shape[k++] = sizes[i];
    for (int64_t i=dim+1; i < rank; ++i) shape[k++] = x->coords.shape[i];
    mag_status_t stat = mag_view(out_result, x, shape, nr); /* Try view first */
    if (mag_iserr(stat))
        stat = mag_reshape(out_result, x, shape, nr);
    return stat;
}

mag_status_t mag_narrow(mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t length) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0, "Cannot narrow a scalar tensor");
    mag_norm_axis(&dim, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= dim && dim < rank, "narrow dim %" PRIi64 " out_result of range for rank %" PRIi64, dim, rank);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, length >= 0, "narrow length must be >= 0, got %" PRIi64, length);
    int64_t sz = x->coords.shape[dim];
    mag_contract(ctx, ERR_INVALID_PARAM, {}, start >= 0 && start <= sz, "narrow start %" PRIi64 " out_result of bounds for size %" PRIi64, start, sz);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, start + length <= sz, "narrow (start+length) %" PRIi64 " exceeds size %" PRIi64, start + length, sz);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, length > 0, "narrow with length<=0 not supported yet");
    return mag_view_slice(out_result, x, dim, start, length, 1);
}

mag_status_t mag_movedim(mag_tensor_t **out_result, mag_tensor_t *x, int64_t src, int64_t dst) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0, "Cannot movedim on scalar tensor");
    mag_norm_axis(&src, rank);
    mag_norm_axis(&dst, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= src && src < rank, "movedim src %" PRIi64 " out_result of range for rank %" PRIi64, src, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= dst && dst < rank, "movedim dst %" PRIi64 " out_result of range for rank %" PRIi64, dst, rank);
    if (src == dst)
        return mag_view(out_result, x, x->coords.shape, rank);
    int64_t perm[MAG_MAX_DIMS];
    for (int64_t i = 0; i < rank; ++i) {
        perm[i] = i;
    }
    int64_t tmp = perm[src];
    if (src < dst) {
        for (int64_t i = src; i < dst; ++i) {
            perm[i] = perm[i + 1];
        }
        perm[dst] = tmp;
    } else {
        for (int64_t i = src; i > dst; --i) {
            perm[i] = perm[i - 1];
        }
        perm[dst] = tmp;
    }
    return mag_permute(out_result, x, perm, rank);
}

mag_status_t mag_select(mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t index) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0, "Cannot select from scalar tensor");
    mag_norm_axis(&dim, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= dim && dim < rank, "select dim %" PRIi64 " out_result of range for rank %" PRIi64, dim, rank);
    int64_t sz = x->coords.shape[dim];
    mag_norm_axis(&index, sz);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, 0 <= index && index < sz, "select index %" PRIi64 " out_result of bounds for size %" PRIi64, index, sz);
    mag_tensor_t *tmp = NULL;
    mag_status_t stat = mag_view_slice(&tmp, x, dim, index, 1, 1);
    if (mag_iserr(stat)) return stat;
    stat = mag_squeeze_dim(out_result, tmp, dim);
    mag_tensor_decref(tmp);
    return stat;
}

mag_status_t mag_split(mag_tensor_t **outs, int64_t num_splits, mag_tensor_t *x, int64_t split_size, int64_t dim) {
    mag_context_t *ctx = x->ctx;
    int64_t rank = x->coords.rank;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, split_size > 0, "split_size must be > 0, got %" PRIi64, split_size);
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0, "Cannot split scalar tensor");
    mag_norm_axis(&dim, rank);
    mag_contract(ctx, ERR_INVALID_RANK, {}, 0 <= dim && dim < rank, "split dim %" PRIi64 " out_result of range for rank %" PRIi64, dim, rank);
    int64_t sz = x->coords.shape[dim];
    int64_t expected_chunks = 0;
    if (sz > 0) expected_chunks = (sz + split_size-1)/split_size;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, num_splits >= 0, "num_splits must be >= 0, got %" PRIi64, num_splits);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, num_splits == expected_chunks, "num_splits (%" PRIi64 ") does not match expected chunk count (%" PRIi64 ")", num_splits, expected_chunks);
    if (!num_splits) return MAG_STATUS_OK;
    for (int64_t i=0; i < num_splits; ++i) outs[i] = NULL;
    int64_t start = 0;
    for (int64_t i=0; i < num_splits; ++i) {
        int64_t remaining = sz - start;
        int64_t length = remaining < split_size ? remaining : split_size;  /* min */
        mag_status_t stat = mag_view_slice(outs+i, x, dim, start, length, 1);
        if (mag_iserr(stat)) {
            for (int64_t j=0; j < i; ++j) {
                mag_tensor_decref(outs[j]);
                outs[j] = NULL;
            }
            return stat;
        }
        start += length;
    }
    return MAG_STATUS_OK;
}

static mag_status_t mag_op_stub_reduction(mag_tensor_t **out_result, mag_opcode_t op, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    mag_status_t stat;
    mag_reduce_plan_t plan;
    stat = mag_reduce_plan_init(ctx, &plan, &x->coords, dims, rank, keepdim);
    if (mag_iserr(stat)) return stat;
    mag_tensor_t *result = NULL;
    mag_dtype_t otype;
    if ((op == MAG_OP_SUM || op == MAG_OP_PROD) && mag_tensor_is_integer_typed(x)) {
        /* For sum/prod use large int64/uint64 as result dtype to store big accumulators */
        otype = mag_dtype_bit(x->dtype) & MAG_DTYPE_MASK_UINT ? MAG_DTYPE_UINT64 : MAG_DTYPE_INT64;
    } else if (op == MAG_OP_ANY || op == MAG_OP_ALL) { /* For logical reductions, use boolean dtype */
        otype = MAG_DTYPE_BOOLEAN;
    } else if (op == MAG_OP_ARGMIN || op == MAG_OP_ARGMAX) { /* For argmin/argmax, use int64 dtype */
        otype = MAG_DTYPE_INT64;
    }else { /* For other reductions, use same dtype as input */
        otype = x->dtype;
    }
    if (!keepdim && !plan.out_rank) stat = mag_empty_scalar(&result, x->ctx,otype);
    else stat = mag_empty(&result, x->ctx,otype, plan.out_rank, plan.out_shape);
    if (mag_iserr(stat)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_ptr(&plan));
    mag_dispatch(op, false, &layout, &x, 1, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_mean(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_MEAN, x, dims, rank, keepdim);
}

mag_status_t mag_min(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_MIN, x, dims, rank, keepdim);
}

mag_status_t mag_max(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_MAX, x, dims, rank, keepdim);
}

mag_status_t mag_argmin(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_ARGMIN, x, dims, rank, keepdim);
}

mag_status_t mag_argmax(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_ARGMAX, x, dims, rank, keepdim);
}

mag_status_t mag_sum(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_SUM, x, dims, rank, keepdim);
}

mag_status_t mag_prod(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_PROD, x, dims, rank, keepdim);
}

mag_status_t mag_all(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_ALL, x, dims, rank, keepdim);
}

mag_status_t mag_any(mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) {
    return mag_op_stub_reduction(out_result, MAG_OP_ANY, x, dims, rank, keepdim);
}

mag_status_t mag_topk(mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t k, int64_t dim, bool largest, bool sorted) {
    *out_values  = NULL;
    *out_indices = NULL;
    mag_contract(NULL, ERR_INVALID_PARAM, {}, x != NULL, "Input tensor cannot be NULL");
    mag_context_t *ctx = x->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, k > 0, "k must be > 0, got: %" PRIi64, k);
    int64_t rank = x->coords.rank;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank > 0, "topk requires rank > 0");
    if (dim < 0) dim += rank;
    mag_contract(ctx, ERR_INVALID_DIM, {}, 0 <= dim && dim < rank, "topk dim %" PRIi64 " out of range for rank %" PRIi64, dim, rank);
    int64_t dim_size = x->coords.shape[dim];
    mag_contract(ctx, ERR_INVALID_PARAM, {}, k <= dim_size, "topk k=%" PRIi64 " must be <= size of dim (%" PRIi64 ")", k, dim_size);
    int64_t shape[MAG_MAX_DIMS];
    memcpy(shape, x->coords.shape, sizeof(*shape)*rank);
    shape[dim] = k;
    mag_tensor_t *values  = NULL;
    mag_tensor_t *indices = NULL;
    mag_status_t stat;
    stat = mag_empty(&values, ctx, x->dtype, rank, shape);
    if (mag_iserr(stat)) return stat;
    stat = mag_empty(&indices, ctx, MAG_DTYPE_INT64,  rank, shape);
    if (mag_iserr(stat)) {
        mag_tensor_decref(values);
        return stat;
    }
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(k));
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim));
    mag_op_attr_registry_insert(&layout, mag_op_attr_bool(largest));
    mag_op_attr_registry_insert(&layout, mag_op_attr_bool(sorted));
    mag_dispatch(MAG_OP_TOPK, false, &layout, &x, 1, (mag_tensor_t*[2]){values, indices}, 2);
    *out_values = values;
    *out_indices = indices;
    return MAG_STATUS_OK;
}

static mag_status_t mag_op_stub_unary(mag_tensor_t **out_result, mag_opcode_t op, mag_tensor_t *x, const mag_op_attr_registry_t *layout, bool inplace) {
    *out_result = NULL;
    mag_assert_dtype_compat(op, &x);
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    if (inplace) {
        stat = mag_tensor_strided_view(&result, x); /* Use the same storage as x */
        if (mag_iserr(stat)) return stat;
        mag_assert_inplace_and_grad_mode_off(x);
    } else {
        stat = mag_empty_like(&result, x); /* Allocate a new tensor for the result */
        if (mag_iserr(stat)) return stat;
    }
    mag_dispatch(op, inplace, layout, &x, 1, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

#define mag_impl_unary_pair(name, op) \
    mag_status_t mag_##name(mag_tensor_t **out_result, mag_tensor_t* x) { return mag_op_stub_unary(out_result, MAG_OP_##op, x, NULL, false); } \
    mag_status_t mag_##name##_(mag_tensor_t **out_result, mag_tensor_t* x) { return mag_op_stub_unary(out_result, MAG_OP_##op, x, NULL, true); }

mag_impl_unary_pair(not, NOT)
mag_impl_unary_pair(abs, ABS)
mag_impl_unary_pair(sgn, SGN)
mag_impl_unary_pair(neg, NEG)
mag_impl_unary_pair(log, LOG)
mag_impl_unary_pair(log10, LOG10)
mag_impl_unary_pair(log1p, LOG1P)
mag_impl_unary_pair(log2, LOG2)
mag_impl_unary_pair(sqr, SQR)
mag_impl_unary_pair(rcp, RCP)
mag_impl_unary_pair(sqrt, SQRT)
mag_impl_unary_pair(rsqrt, RSQRT)
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

mag_status_t mag_tril(mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag) {
    *out_result = NULL;
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, tensor->coords.rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, tensor->coords.rank);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(diag));
    return mag_op_stub_unary(out_result, MAG_OP_TRIL, tensor, &layout, false);
}

mag_status_t mag_tril_(mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag) {
    *out_result = NULL;
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, tensor->coords.rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, tensor->coords.rank);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(diag));
    return mag_op_stub_unary(out_result, MAG_OP_TRIL, tensor, &layout, true);
}

mag_status_t mag_triu(mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag) {
    *out_result = NULL;
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, tensor->coords.rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, tensor->coords.rank);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(diag));
    return mag_op_stub_unary(out_result, MAG_OP_TRIU, tensor, &layout, false);
}

mag_status_t mag_triu_(mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag) {
    *out_result = NULL;
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, tensor->coords.rank >= 2, "Diagonal matrix operator requires rank >= 2, but got: %" PRIi64, tensor->coords.rank);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(diag));
    return mag_op_stub_unary(out_result, MAG_OP_TRIU, tensor, &layout, true);
}

mag_status_t mag_multinomial(mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t num_samples, bool replacement) {
    *out_result = NULL;
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, tensor->coords.rank == 1 || tensor->coords.rank == 2, "Multinomial dist requires rank 1 or 2, but got: %" PRIi64, tensor->coords.rank);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_contiguous(tensor), "Input tensor must be contiguous row-major");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, num_samples > 0, "Number of samples must be > 0, but got: %" PRIi64, num_samples);
    mag_assert_dtype_compat(MAG_OP_MULTINOMIAL, &tensor);
    int64_t shape[MAG_MAX_DIMS] = {0};
    if (tensor->coords.rank > 1) memcpy(shape, tensor->coords.shape, (tensor->coords.rank - 1)*sizeof(*shape));
    shape[tensor->coords.rank-1] = num_samples;
    mag_tensor_t *result;
    mag_status_t stat = mag_empty(&result, tensor->ctx, MAG_DTYPE_INT64, tensor->coords.rank, shape);
    if (mag_iserr(stat)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(num_samples));
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(!!replacement));
    mag_dispatch(MAG_OP_MULTINOMIAL, false, &layout, &tensor, 1, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_cat(mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count, int64_t dim) {
    *out_result = NULL;
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
    memcpy(shape, t0->coords.shape, rank*sizeof(*shape));
    shape[dim] = 0;
    for (size_t i=0; i < count; ++i) {
        mag_tensor_t *ti = tensors[i];
        mag_contract(ctx, ERR_INVALID_PARAM, {}, ti != NULL, "Tensor %" PRIu64 " cannot be NULL", (uint64_t)i);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, ti->coords.rank == rank, "All tensors must have same rank (%" PRIi64 " != %" PRIi64 ")", ti->coords.rank, rank);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, ti->dtype == dtype, "All tensors must have same dtype (%d != %d)", ti->dtype, dtype);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_contiguous(ti), "All tensors must be contiguous row-major");
        for (int64_t j=0; j < rank; ++j) {
            if (j == dim) continue;
            mag_contract(ctx, ERR_INVALID_PARAM, {}, ti->coords.shape[j] == t0->coords.shape[j], "Shapes must match on non-concat dims (dim=%" PRIi64 " mismatch on axis %" PRIi64 ")", dim, j);
        }
        shape[dim] += ti->coords.shape[dim];
    }
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim));
    stat = mag_empty(&result, t0->ctx, dtype, rank, shape);
    if (mag_iserr(stat)) return stat;
    mag_dispatch(MAG_OP_CAT, false, &layout, tensors, count, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_one_hot(mag_tensor_t **out_result, mag_tensor_t *indices, int64_t num_classes) {
    *out_result = NULL;
    mag_context_t *ctx = indices->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, indices->dtype == MAG_DTYPE_INT64, "one_hot: indices dtype must be int64, got %s", mag_type_trait(indices->dtype)->name);
    mag_contract(ctx, ERR_INVALID_PARAM, {},  num_classes >= -1, "one_hot: num_classes must be >= -1, got %" PRIi64,  num_classes);
    mag_status_t stat;
    if (num_classes == -1) {
        mag_tensor_t *maxv = NULL;
        stat = mag_max(&maxv, indices, NULL, 0, false);
        if (mag_iserr(stat)) return stat;
        mag_scalar_t max_scalar;
        stat = mag_tensor_item(maxv, &max_scalar);
        if (mag_iserr(stat)) {
            mag_tensor_decref(maxv);
            return stat;
        }
        int64_t max_class = mag_scalar_as_i64(max_scalar);
        mag_tensor_decref(maxv);
        num_classes = max_class >= 0 ? 1+max_class : 0;
    }
    mag_contract(ctx, ERR_INVALID_PARAM, {}, num_classes > 0, "one_hot: inferred num_classes must be > 0, got %" PRIi64, num_classes);
    int64_t rank = indices->coords.rank;
    mag_contract(ctx, ERR_INVALID_RANK, {}, rank + 1 <= MAG_MAX_DIMS, "one_hot: rank(indices)+1 must be <= MAG_MAX_DIMS");
    int64_t orank = rank+1;
    int64_t oshape[MAG_MAX_DIMS];
    for (int64_t i=0; i < rank; ++i)
        oshape[i] = indices->coords.shape[i];
    oshape[rank] = num_classes;
    mag_tensor_t *result;
    stat = mag_zeros(&result, ctx, MAG_DTYPE_INT64, orank, oshape);
    if (mag_iserr(stat)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(num_classes));
    mag_dispatch(MAG_OP_ONE_HOT, false, &layout, &indices, 1, &result, 1);
    if (mag_iserr(stat)) {
        mag_tensor_decref(result);
        return stat;
    }
    *out_result = result;
    return MAG_STATUS_OK;
}

typedef enum mag_binop_flags_t {
    MAG_BINOP_NONE = 0,
    MAG_BINOP_LOGICAL = 1<<0,
    MAG_BINOP_INPLACE = 1<<1
} mag_binop_flags_t;

static mag_status_t mag_op_stub_binary(mag_tensor_t **out_result, mag_opcode_t op, mag_tensor_t *x, mag_tensor_t *y, mag_binop_flags_t flags) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_dtype_t prom_type; /* common compute dtype for x,y */
    mag_dtype_t res_type;  /* dtype of 'result' tensor */
    bool x_int = mag_tensor_is_integer_typed(x);
    bool y_int = mag_tensor_is_integer_typed(y);
    if (flags & MAG_BINOP_INPLACE) {
        switch (op) {
            case MAG_OP_DIV: {
                mag_contract(ctx, ERR_INVALID_PARAM, {}, !x_int, "Inplace truediv is not allowed on integer tensors, got dtype %s", mag_type_trait(x->dtype)->name);
            } break;
            case MAG_OP_FLOORDIV: {
                mag_contract(ctx, ERR_INVALID_PARAM, {}, x_int && y_int, "Inplace floordiv is only allowed for integer tensors, got dtypes %s and %s", mag_type_trait(x->dtype)->name, mag_type_trait(y->dtype)->name);
            } break;
            default: { /* Inplace ops must keep x's dtype */
                mag_dtype_t prom;
                bool prom_ok = mag_promote_type(&prom, x->dtype, y->dtype);
                mag_contract(ctx, ERR_INVALID_PARAM, {},  prom_ok && prom == x->dtype,  "Inplace binary op '%s' would change the dtype from %s to %s", mag_op_traits(op)->mnemonic, mag_type_trait(x->dtype)->name, mag_type_trait(prom)->name);
            } break;
        }
        prom_type = x->dtype;
        res_type = x->dtype;
    } else if (flags & MAG_BINOP_LOGICAL) { /* Inplace keeps x's dtype, but cast y to x's dtype if needed */
        bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
        mag_contract(ctx, ERR_INVALID_PARAM, {}, prom_ok, "Logical binary operator '%s' not supported for dtypes %s and %s",
            mag_op_traits(op)->mnemonic,
            mag_type_trait(x->dtype)->name,
            mag_type_trait(y->dtype)->name
        );
        res_type = MAG_DTYPE_BOOLEAN; /* logical ops always yield boolean result */
        mag_assert2(!(flags & MAG_BINOP_INPLACE));
    } else { /* Pure out of place -> full promotion */
        switch (op) {
            case MAG_OP_DIV: { /* Special case for truediv */
                if (x_int && y_int) { /* Integer division always promotes to default float dtype */
                    prom_type = res_type = MAG_DTYPE_FLOAT32;
                } else {
                    bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
                    mag_contract(ctx, ERR_INVALID_PARAM, {}, prom_ok,
                        "Binary operator '%s' not supported for dtypes %s and %s",
                        mag_op_traits(op)->mnemonic,
                        mag_type_trait(x->dtype)->name,
                        mag_type_trait(y->dtype)->name
                    );
                    res_type = prom_type;  /* will be a floating dtype */
                }
            } break;
            case MAG_OP_FLOORDIV: {
                bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
                mag_contract(ctx, ERR_INVALID_PARAM, {}, prom_ok,
                    "Binary operator '%s' not supported for dtypes %s and %s",
                    mag_op_traits(op)->mnemonic,
                    mag_type_trait(x->dtype)->name,
                        mag_type_trait(y->dtype)->name
                );
                if (x_int && y_int) { /* Integer floor division keeps integer dtype */
                    res_type = prom_type;
                } else { /* Non-integer floor division promotes to floating dtype */
                    if (!(mag_dtype_bit(prom_type) & MAG_DTYPE_MASK_FP))
                        prom_type = MAG_DTYPE_FLOAT32;
                    res_type = prom_type;
                }
            } break;
            default: {
                bool prom_ok = mag_promote_type(&prom_type, x->dtype, y->dtype);
                mag_contract(ctx, ERR_INVALID_PARAM, {}, prom_ok, "Binary operator '%s' not supported for dtypes %s and %s",
                    mag_op_traits(op)->mnemonic,
                    mag_type_trait(x->dtype)->name,
                    mag_type_trait(y->dtype)->name
                );
                res_type = prom_type;
            } break;
        }
    }
    if (flags & MAG_BINOP_INPLACE) {
        mag_assert2(!(flags & MAG_BINOP_LOGICAL));
        mag_assert_inplace_and_grad_mode_off(x);
        stat = mag_tensor_strided_view(&result, x);
        if (mag_iserr(stat)) return stat;
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
                sx, sy, mag_op_traits(op)->mnemonic
            );
        }
        stat = rank ? mag_empty(&result, x->ctx, res_type, rank, dims) : mag_empty_scalar(&result, x->ctx, res_type);
        if (mag_iserr(stat)) return stat;
    }
    mag_tensor_t *prom_x = x;
    mag_tensor_t *prom_y = y;
    mag_tensor_t *tmp_x = NULL;
    mag_tensor_t *tmp_y = NULL;
    if (x->dtype != prom_type) { /* Cast x only if its dtype != promote_dtype and the op semantics say so */
        stat = mag_cast(&tmp_x, x, prom_type); /* For inplace, x->dtype == promote_dtype, so this is skipped */
        if (mag_iserr(stat)) {
            if (!(flags & MAG_BINOP_INPLACE) && result)
                mag_tensor_decref(result);
            return stat;
        }
        prom_x = tmp_x;
    }
    if (y->dtype != prom_type) { /* Cast y if needed */
        stat = mag_cast(&tmp_y, y, prom_type);
        if (mag_iserr(stat)) {
            if (tmp_x) mag_tensor_decref(tmp_x);
            if (!(flags & MAG_BINOP_INPLACE) && result)
                mag_tensor_decref(result);
            return stat;
        }
        prom_y = tmp_y;
    }
    mag_tensor_t *in[2] = {prom_x, prom_y};
    mag_assert_dtype_compat(op, in);
    mag_dispatch(op, flags & MAG_BINOP_INPLACE, NULL, in, sizeof(in)/sizeof(*in), &result, 1);
    if (tmp_x) mag_tensor_decref(tmp_x);
    if (tmp_y) mag_tensor_decref(tmp_y);
    *out_result = result;
    return MAG_STATUS_OK;
}

#define mag_impl_binary_pair(name, op, logical) \
    mag_status_t mag_##name(mag_tensor_t **out_result, mag_tensor_t* x, mag_tensor_t* y) { return mag_op_stub_binary(out_result, MAG_OP_##op, x, y, logical ? MAG_BINOP_LOGICAL : 0); } \
    mag_status_t mag_##name##_(mag_tensor_t **out_result, mag_tensor_t* x, mag_tensor_t* y) { return mag_op_stub_binary(out_result, MAG_OP_##op, x, y, (logical ? MAG_BINOP_LOGICAL : 0)+MAG_BINOP_INPLACE); }

mag_impl_binary_pair(add, ADD, false)
mag_impl_binary_pair(sub, SUB, false)
mag_impl_binary_pair(mul, MUL, false)
mag_impl_binary_pair(div, DIV, false)
mag_impl_binary_pair(floordiv, FLOORDIV, false)
mag_impl_binary_pair(mod, MOD, false)
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

mag_status_t mag_matmul(mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
    *out_result = NULL;
    mag_context_t *ctx = x->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_floating_point_typed(x) && mag_tensor_is_floating_point_typed(y), "matmul: both tensors must be floating point typed");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, x->coords.rank >= 1 && y->coords.rank >= 1, "matmul: both tensors must be at least rank 1");
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_assert_dtype_compat(MAG_OP_MATMUL, (mag_tensor_t *[]){x, y});
    int64_t kx = x->coords.shape[x->coords.rank-1];
    int64_t ky = y->coords.rank == 1 ? *y->coords.shape : y->coords.rank == 2 && x->coords.rank == 1 ? *y->coords.shape : y->coords.shape[y->coords.rank-2];
    if (kx != ky) {
        char sx[MAG_FMT_DIM_BUF_SIZE];
        char sy[MAG_FMT_DIM_BUF_SIZE];
        mag_fmt_shape(&sx, &x->coords.shape, x->coords.rank);
        mag_fmt_shape(&sy, &y->coords.shape, y->coords.rank);
        mag_contract(
            ctx, ERR_OPERATOR_IMPOSSIBLE, {}, 0,
            "Cannot perform matmul on tensors with shapes %s and %s: "
            "last dimension of first tensor (%" PRIi64 ") does not match second tensor (%" PRIi64 ").\n"
            "    Hint: Ensure that the last dimension of the first tensor matches the second-to-last dimension of the second tensor.\n",
            sx, sy, kx, ky
        );
    }
    /* verify broadcastability of batch dims */
    int64_t xbd = x->coords.rank > 2 ? x->coords.rank-2 : 0;
    int64_t ybd = y->coords.rank > 2 ? y->coords.rank-2 : 0;
    int64_t rbd = xbd > ybd ? xbd : ybd;
    for (int64_t i=0; i < rbd; ++i) {
        int64_t xd = i < rbd-xbd ? 1 : x->coords.shape[i-(rbd-xbd)];
        int64_t yd = i < rbd-ybd ? 1 : y->coords.shape[i-(rbd-ybd)];
        if (xd != yd && xd != 1 && yd != 1) {
            char sx[MAG_FMT_DIM_BUF_SIZE];
            char sy[MAG_FMT_DIM_BUF_SIZE];
            mag_fmt_shape(&sx, &x->coords.shape, x->coords.rank);
            mag_fmt_shape(&sy, &y->coords.shape, y->coords.rank);
            mag_contract(
                ctx, ERR_OPERATOR_IMPOSSIBLE, {}, 0,
                "Cannot perform matmul on tensors with shapes %s and %s: "
                "batch dimensions at index %" PRIi64 " do not match (%" PRIi64 " != %" PRIi64 ").\n"
                "    Hint: Ensure that the batch dimensions are compatible for broadcasting.\n",
                sx, sy, i, xd, yd
            );
        }
    }
    if (x->coords.rank == 1 && y->coords.rank == 1) stat = mag_empty_scalar(&result, x->ctx, x->dtype); /* (K)x(K) -> () */
    else if (x->coords.rank == 1 && y->coords.rank == 2) stat = mag_empty(&result, x->ctx, x->dtype, 1, y->coords.shape+1); /* (K)x(K,N) -> (N) */
    else if (x->coords.rank == 2 && y->coords.rank == 1) stat = mag_empty(&result, x->ctx, x->dtype, 1, x->coords.shape); /* (M,K)x(K) -> (M) */
    else { /* Batched ND version */
        xbd = x->coords.rank-2;
        ybd = y->coords.rank-2;
        int64_t shape[MAG_MAX_DIMS] = {0};
        for (int64_t i=0; i < rbd; ++i) {
            int64_t da = i < rbd-xbd ? 1 : x->coords.shape[i-(rbd-xbd)];
            int64_t db = i < rbd-ybd ? 1 : y->coords.shape[i-(rbd-ybd)];
            shape[i] = da > db ? da : db;
        }
        shape[rbd] = x->coords.shape[x->coords.rank-2];
        shape[rbd+1] = y->coords.shape[y->coords.rank-1];
        stat = mag_empty(&result,x->ctx, x->dtype, rbd+2, shape);
    }
    if (mag_iserr(stat)) return stat;
    mag_dispatch(MAG_OP_MATMUL, false, false, (mag_tensor_t *[2]) {x, y}, 2, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_repeat_back(mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y) {
    *out_result = NULL;
    mag_tensor_t *result = NULL;
    mag_assert_dtype_compat(MAG_OP_REPEAT_BACK, (mag_tensor_t *[]) {x, y});
    mag_status_t stat = mag_empty(&result, x->ctx, x->dtype, y->coords.rank, y->coords.shape);
    if (mag_iserr(stat)) return stat;
    /* TODO: Check for broadcastability of x and y */
    mag_dispatch(MAG_OP_REPEAT_BACK, false, NULL, (mag_tensor_t *[2]) {x, y}, 2, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_gather(mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t dim, mag_tensor_t *idx) {
    *out_result = NULL;
    mag_context_t *ctx = tensor->ctx;
    mag_tensor_t *result = NULL;
    mag_status_t stat;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, idx->dtype == MAG_DTYPE_INT64, "Index tensor must be of type: int64_t");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, dim >= 0 && dim < tensor->coords.rank, "Gather dim must be in [0, %" PRIi64 "), but got: %" PRIi64, tensor->coords.rank, dim);
    mag_contract(ctx, ERR_INVALID_PARAM, {}, idx->coords.rank <= tensor->coords.rank, "Index tensor rank must be <= input tensor rank (%" PRIi64 " <= %" PRIi64")", idx->coords.rank, tensor->coords.rank);
    mag_norm_axis(&dim, tensor->coords.rank);
    mag_assert2(dim >= 0 && dim < tensor->coords.rank);
    int64_t ax[MAG_MAX_DIMS];
    int64_t ork = 0;
    bool full = false;
    if (idx->coords.rank == tensor->coords.rank) {
        full = true;
        for (int64_t i=0; i < tensor->coords.rank; ++i) {
            if (i == dim) continue;
            if (idx->coords.shape[i] != tensor->coords.shape[i]) {
                full = false;
                break;
            }
        }
    }
    if (full)
        for (int64_t i=0; i < tensor->coords.rank; ++i)
            ax[ork++] = idx->coords.shape[i];
    else if (idx->coords.rank == 1)
        for (int64_t i=0; i < tensor->coords.rank; ++i)
            ax[ork++] = i == dim ? idx->coords.shape[0] : tensor->coords.shape[i];
    else {
        for (int64_t i=0; i < dim; ++i) ax[ork++] = tensor->coords.shape[i];
        for (int64_t i=0; i < idx->coords.rank; ++i) ax[ork++] = idx->coords.shape[i];
        for (int64_t i=dim+1; i < tensor->coords.rank; ++i) ax[ork++] = tensor->coords.shape[i];
    }
    mag_contract(ctx, ERR_INVALID_RANK, {}, ork >= 1 && ork <= MAG_MAX_DIMS, "Gather output rank must be in [1, %d], but got: %" PRIi64, MAG_MAX_DIMS, ork);
    stat = mag_empty(&result, tensor->ctx, tensor->dtype, ork, ax);
    if (mag_iserr(stat)) return stat;
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_int64(dim)); /* Store dimension in op_params[0] */
    mag_dispatch(MAG_OP_GATHER, false, &layout, (mag_tensor_t *[2]) {tensor, idx}, 2, &result, 1);
    *out_result = result;
    return MAG_STATUS_OK;
}

mag_status_t mag_copy_raw_(mag_tensor_t *tensor, const void *data, size_t size_bytes) {
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, data != NULL && size_bytes > 0, "invalid data pointer or length");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_device_is(tensor->storage->device, "cpu"), "tensor storage device must be CPU for mag_copy_raw_");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, data && size_bytes, "invalid data pointer or length");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_numbytes(tensor) == size_bytes, "data length (%" PRIu64 ") does not match tensor nbytes (%" PRIu64 ")", (uint64_t)size_bytes, (uint64_t)mag_tensor_numbytes(tensor));
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_contiguous(tensor), "tensor must be contiguous");
    void *dst = (void *)mag_tensor_data_ptr_mut(tensor);
    memcpy(dst, data, size_bytes);
    return MAG_STATUS_OK;
}

mag_status_t mag_zero_(mag_tensor_t *tensor) {
    return mag_fill_(tensor, mag_scalar_uint(0));
}

mag_status_t mag_fill_(mag_tensor_t *tensor, mag_scalar_t value) {
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_scalar_to_op_attr(tensor->dtype, value));
    mag_dispatch(MAG_OP_FILL, false, &layout, NULL, 0, &tensor, 1);
    return MAG_STATUS_OK;
}

mag_status_t mag_masked_fill_(mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value) {
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mask->dtype == MAG_DTYPE_BOOLEAN, "mask tensor dtype must be boolean, got %s", mag_type_trait(mask->dtype)->name);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_scalar_to_op_attr(tensor->dtype, value));
    mag_op_attr_registry_insert(&layout, mag_op_attr_ptr(mask));
    mag_dispatch(MAG_OP_MASKED_FILL, false, &layout, NULL, 0, &tensor, 1);
    return MAG_STATUS_OK;
}

mag_status_t mag_uniform_(mag_tensor_t *tensor, mag_scalar_t min, mag_scalar_t max) {
    mag_context_t *ctx = tensor->ctx;
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_scalar_same_type(min, max), "min and max must be of the same type");
    mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_numeric_typed(tensor), "tensor dtype must be integer, got %s", mag_type_trait(tensor->dtype)->name);
    if (mag_scalar_is_f64(min)) {
        mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_scalar_as_f64(min) < mag_scalar_as_f64(max), "min must be < max (got min=%f, max=%f)", mag_scalar_as_f64(min), mag_scalar_as_f64(max));
    } else if (mag_scalar_is_i64(min)) {
        mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_scalar_as_i64(min) < mag_scalar_as_i64(max), "min must be < max (got min=%" PRIi64 ", max=%" PRIi64 ")", mag_scalar_as_i64(min), mag_scalar_as_i64(max));
    } else if (mag_scalar_is_u64(min)) {
        mag_contract(ctx, ERR_INVALID_PARAM, {}, mag_scalar_as_u64(min) < mag_scalar_as_u64(max), "min must be < max (got min=%" PRIu64 ", max=%" PRIu64 ")", mag_scalar_as_u64(min), mag_scalar_as_u64(max));
    } else {
        mag_contract(ctx, ERR_INVALID_PARAM, {}, false, "unsupported scalar type for min/max");
    }
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_scalar_to_op_attr(tensor->dtype, min));
    mag_op_attr_registry_insert(&layout, mag_scalar_to_op_attr(tensor->dtype, max));
    mag_dispatch(MAG_OP_RAND_UNIFORM, false, &layout, NULL, 0, &tensor, 1);
    return MAG_STATUS_OK;
}

mag_status_t mag_normal_(mag_tensor_t *tensor, mag_scalar_t mean, mag_scalar_t stddev) {
    mag_contract(tensor->ctx, ERR_INVALID_PARAM, {}, mag_scalar_is_f64(mean) && mag_scalar_is_f64(stddev), "mean and stddev must be float scalars");
    mag_contract(tensor->ctx, ERR_INVALID_PARAM, {}, mag_tensor_is_floating_point_typed(tensor), "tensor dtype must be floating point, got %s", mag_type_trait(tensor->dtype)->name);
    double stddev_f = mag_scalar_as_f64(stddev);
    double mean_f = mag_scalar_as_f64(mean);
    mag_contract(tensor->ctx, ERR_INVALID_PARAM, {}, stddev_f >= 0.0, "stddev must be >= 0 (got stddev=%f)", stddev_f);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_float64(mean_f));
    mag_op_attr_registry_insert(&layout, mag_op_attr_float64(stddev_f));
    mag_dispatch(MAG_OP_RAND_NORMAL, false, &layout, NULL, 0, &tensor, 1);
    return MAG_STATUS_OK;
}

mag_status_t mag_bernoulli_(mag_tensor_t *tensor, mag_scalar_t p) {
    mag_contract(tensor->ctx, ERR_INVALID_PARAM, {}, mag_scalar_is_f64(p), "p must be a ");
    mag_contract(tensor->ctx, ERR_INVALID_PARAM, {}, tensor->dtype == MAG_DTYPE_BOOLEAN, "tensor dtype must be boolean, got %s", mag_type_trait(tensor->dtype)->name);
    double pf = mag_scalar_as_f64(p);
    mag_contract(tensor->ctx, ERR_INVALID_PARAM, {}, pf >= 0.0 && pf <= 1.0, "probability p must be in [0.0, 1.0], got p=%f", pf);
    mag_op_attr_registry_t layout;
    mag_op_attr_registry_init(&layout);
    mag_op_attr_registry_insert(&layout, mag_op_attr_float64(pf));
    mag_dispatch(MAG_OP_RAND_BERNOULLI, false, &layout, NULL, 0, &tensor, 1);
    return MAG_STATUS_OK;
}
