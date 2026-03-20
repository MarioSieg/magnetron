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

#ifndef MAGNETRON_H
#define MAGNETRON_H

#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_MAX_DIMS 16 /* Maximum number of dimensions for a tensor. Currently fixed. */

#ifndef MAG_EXPORT
#ifdef _MSC_VER
#define MAG_EXPORT __declspec(dllexport)
#else
#define MAG_EXPORT __attribute__((visibility("default")))
#endif
#endif

#define mag_assert_name2(name, line) name ## line
#define mag_assert_name(line) mag_assert_name2(_assert_, line)
#define mag_static_assert(expr) extern void mag_assert_name(__LINE__)(bool STATIC_ASSERTION_FAILED[((expr)?1:-1)])

#define mag_ver_encode(maj, min, patch) ((maj)*10000u + (min)*100u + (patch))
#define mag_ver_major(v) ((v)/10000u)
#define mag_ver_minor(v) (((v)/100u)%100u)
#define mag_ver_patch(v) ((v)%100u)
#define MAG_VERSION mag_ver_encode(0, 1, 6)
#define MAG_SNAPSHOT_VERSION mag_ver_encode(0, 1, 0)

typedef enum mag_log_level_t {
    MAG_LOG_LEVEL_NONE,
    MAG_LOG_LEVEL_ERROR,
    MAG_LOG_LEVEL_WARN,
    MAG_LOG_LEVEL_INFO,
    MAG_LOG_LEVEL_DEBUG
} mag_log_level_t;

extern MAG_EXPORT void mag_set_log_level(mag_log_level_t level); /* Set global log level. */
extern MAG_EXPORT mag_log_level_t mag_log_level(void); /* Get current global log level. */

/**
 * Status return codes for magnetron library functions.
 */
#define mag_statusdef(_) \
    _(MAG_STATUS_OK, "OK") \
    _(MAG_STATUS_ERR_PENDING, "Operation already pending") \
    _(MAG_STATUS_ERR_THREAD_MISMATCH, "Thread mismatch") \
    _(MAG_STATUS_ERR_INVALID_RANK, "Invalid tensor rank") \
    _(MAG_STATUS_ERR_INVALID_DIM, "Invalid dimension") \
    _(MAG_STATUS_ERR_DIM_OVERFLOW, "Dimension overflow") \
    _(MAG_STATUS_ERR_INVALID_INDEX, "Invalid index") \
    _(MAG_STATUS_ERR_OUT_OF_BOUNDS, "Index out of bounds") \
    _(MAG_STATUS_ERR_INVALID_PARAM, "Invalid parameter") \
    _(MAG_STATUS_ERR_STRIDE_SOLVER_FAILED, "Stride solver failed") \
    _(MAG_STATUS_ERR_BROADCAST_IMPOSSIBLE, "Broadcast not possible") \
    _(MAG_STATUS_ERR_OPERATOR_IMPOSSIBLE, "Operator cannot be applied to given tensors") \
    _(MAG_STATUS_ERR_INVALID_STATE, "Invalid state") \
    _(MAG_STATUS_ERR_IMAGE_ERROR, "Image processing error") \
    _(MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED, "Memory allocation failed") \
    _(MAG_STATUS_ERR_MEMORY_DEALLOCATION_FAILED, "Memory deallocation failed") \
    _(MAG_STATUS_ERR_UNKNOWN, "Unknown error")

typedef enum mag_status_t {
#define _(code, msg) code,
    mag_statusdef(_)
#undef _
} mag_status_t;
extern MAG_EXPORT const char *mag_status_get_name(mag_status_t op);
extern MAG_EXPORT const char *mag_status_get_message(mag_status_t op);

/**
 * @brief Error structure for magnetron library functions.
 */
typedef struct mag_error_t {
    mag_status_t code;
    char message[256];
    const char *file;
    int line;
    const char *func;
} mag_error_t;

/**
* @brief The context is used to create and manage tensors, operations, and other resources.
* Since all tensors, storages, devices and backends are associated with a context, the context must stay alive until all associated resources are destroyed.
* The context is not thread safe, in multiple threads, use one context per thread.
*/
typedef struct mag_context_t mag_context_t;

extern MAG_EXPORT mag_context_t *mag_ctx_create(const char *device_id);                                                 /* Create context with default config, and only specify device type. */
extern MAG_EXPORT const char *mag_ctx_get_compute_device_name(const mag_context_t *ctx);                                /* Get the name of the compute device */
extern MAG_EXPORT const char *mag_ctx_get_os_name(const mag_context_t *ctx);                                            /* Get the name of the operating system */
extern MAG_EXPORT const char *mag_ctx_get_cpu_name(const mag_context_t *ctx);                                           /* Get the name of the CPU */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_virtual_cores(const mag_context_t *ctx);                                     /* Get the number of virtual cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_physical_cores(const mag_context_t *ctx);                                    /* Get the number of physical cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_sockets(const mag_context_t *ctx);                                           /* Get the number of CPU sockets */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_total(const mag_context_t *ctx);                                 /* Get the total physical memory in bytes */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_free(const mag_context_t *ctx);                                  /* Get the free physical memory in bytes */
extern MAG_EXPORT bool mag_ctx_is_numa_system(const mag_context_t *ctx);                                                /* Check if the system is NUMA */
extern MAG_EXPORT size_t mag_ctx_get_total_tensors_created(const mag_context_t *ctx);                                   /* Get total tensors created. (Including views) */
extern MAG_EXPORT void mag_ctx_grad_recorder_start(mag_context_t *ctx);                                                 /* Start gradient recording */
extern MAG_EXPORT void mag_ctx_grad_recorder_stop(mag_context_t *ctx);                                                  /* Stop gradient recording */
extern MAG_EXPORT bool mag_ctx_grad_recorder_is_running(const mag_context_t *ctx);                                      /* Check if gradient recording is running */
extern MAG_EXPORT void mag_ctx_lazy_exec_start(mag_context_t *ctx);                                                     /* Enable lazy execution. */
extern MAG_EXPORT void mag_ctx_lazy_exec_stop(mag_context_t *ctx);                                                      /* Disable lazy execution. */
extern MAG_EXPORT bool mag_ctx_lazy_exec_is_running(const mag_context_t *ctx);                                          /* Check if lazy execution is enabled. */
extern MAG_EXPORT void mag_ctx_manual_seed(mag_context_t *ctx, uint64_t seed);                                          /* Manually seed the PRNG. */
extern MAG_EXPORT void mag_ctx_destroy(mag_context_t *ctx, bool suppress_leak_detection);                               /* Destroy context and free memory */

/**
 * @brief Multidimensional tensor of arbitrary rank and data type.
 *      The tensor is reference counted and can be shared between multiple tensors.
 *      Rule of Thumb for Reference Counting:
 *          - If you only use the reference temporarily and do not store it, no need to adjust the reference count.
 *          - If you store the reference (e.g., in a data structure), increase the reference count when storing and decrease it when removing.
 *      The rank is > 0 and <= MAG_MAX_DIMS. The shape of the tensor is an array of dimensions of size MAG_MAX_DIMS.
 *      Is a node in a static or dynamic computation graph, depending on the context execution mode.
 */
typedef struct mag_tensor_t mag_tensor_t;

/**
* Type tag discriminating between different scalar types.
*/
typedef enum mag_scalar_type_t {
    MAG_SCALAR_TYPE_F64,
    MAG_SCALAR_TYPE_I64,
    MAG_SCALAR_TYPE_U64,
} mag_scalar_type_t;

/**
 * @brief Represents a scalar value that can be of different types (float, int, uint, bool).
 * Used to pass scalar values to tensor factories,
 * to avoid overloading or multiple versions of functions for different scalar types.
 * (e.g. we don't want mag_full_f64, mag_full_i64, mag_full_u64, etc.).
 * Also used for metadata records in snapshots.
 */
typedef struct mag_scalar_t {
    mag_scalar_type_t type;
    union {
        double f64;
        int64_t i64;
        uint64_t u64;
    } value;
} mag_scalar_t;

extern MAG_EXPORT mag_scalar_t mag_scalar_from_f64(double value);
extern MAG_EXPORT mag_scalar_t mag_scalar_from_i64(int64_t value);
extern MAG_EXPORT mag_scalar_t mag_scalar_from_u64(uint64_t value);

extern MAG_EXPORT bool mag_scalar_is_f64(mag_scalar_t s);
extern MAG_EXPORT bool mag_scalar_is_i64(mag_scalar_t s);
extern MAG_EXPORT bool mag_scalar_is_u64(mag_scalar_t s);

extern MAG_EXPORT double mag_scalar_as_f64(mag_scalar_t s);
extern MAG_EXPORT int64_t mag_scalar_as_i64(mag_scalar_t s);
extern MAG_EXPORT uint64_t mag_scalar_as_u64(mag_scalar_t s);

/**
 * @brief Data types for tensors.
 */
typedef enum mag_dtype_t {
    MAG_DTYPE_FLOAT32,
    MAG_DTYPE_FLOAT16,
    MAG_DTYPE_BFLOAT16,
    MAG_DTYPE_BOOLEAN,
    MAG_DTYPE_UINT8,
    MAG_DTYPE_INT8,
    MAG_DTYPE_UINT16,
    MAG_DTYPE_INT16,
    MAG_DTYPE_UINT32,
    MAG_DTYPE_INT32,
    MAG_DTYPE_UINT64,
    MAG_DTYPE_INT64,

    MAG_DTYPE__NUM
} mag_dtype_t;
mag_static_assert(MAG_DTYPE__NUM <= 0xff); /* Must fit in 1 byte */

extern MAG_EXPORT bool mag_promote_type(mag_dtype_t *out, mag_dtype_t lhs, mag_dtype_t rhs);

/**
* @brief Contains metadata about a data type such as its name, size, and alignment.
*/
typedef struct mag_type_traits_t {
    const char *name;           /* Name of the data type. eg. bfloat16 */
    const char *short_name;     /* Short name of the data type. eg. bf16 */
    size_t size;                /* Size of the data type in bytes. Must be a power of two. */
    size_t alignment;           /* CPU Alignment of the data type in bytes. Must be a power of two. */
    mag_scalar_t min_val;       /* Minimum finite value representable by this data type, as a scalar. For integer types, this is the smallest integer. For floating point types, this is the smallest normalized positive value. */
    mag_scalar_t max_val;       /* Maximum finite value representable by this data type, as a scalar. For integer types, this is the largest integer. For floating point types, this is the largest finite value. */
} mag_type_traits_t;
extern MAG_EXPORT const mag_type_traits_t *mag_type_trait(mag_dtype_t type);
extern MAG_EXPORT bool mag_type_category_is_floating_point(mag_dtype_t type);
extern MAG_EXPORT bool mag_type_category_is_unsigned_integer(mag_dtype_t type);
extern MAG_EXPORT bool mag_type_category_is_signed_integer(mag_dtype_t type);
extern MAG_EXPORT bool mag_type_category_is_integer(mag_dtype_t type);
extern MAG_EXPORT bool mag_type_category_is_integral(mag_dtype_t type);
extern MAG_EXPORT bool mag_type_category_is_numeric(mag_dtype_t type);

extern MAG_EXPORT mag_status_t mag_empty(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape);
extern MAG_EXPORT mag_status_t mag_as_strided(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset);
extern MAG_EXPORT mag_status_t mag_empty_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like);
extern MAG_EXPORT mag_status_t mag_empty_scalar(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type);
extern MAG_EXPORT mag_status_t mag_scalar(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_full(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_full_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_zeros(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape);
extern MAG_EXPORT mag_status_t mag_zeros_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like);
extern MAG_EXPORT mag_status_t mag_ones(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape);
extern MAG_EXPORT mag_status_t mag_ones_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like);
extern MAG_EXPORT mag_status_t mag_uniform(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t min, mag_scalar_t max);
extern MAG_EXPORT mag_status_t mag_uniform_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t min, mag_scalar_t max);
extern MAG_EXPORT mag_status_t mag_normal(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t mean, mag_scalar_t stddev);
extern MAG_EXPORT mag_status_t mag_normal_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t mean, mag_scalar_t stddev);
extern MAG_EXPORT mag_status_t mag_bernoulli(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, int64_t rank, const int64_t *shape, mag_scalar_t p);
extern MAG_EXPORT mag_status_t mag_bernoulli_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t p);
extern MAG_EXPORT mag_status_t mag_arange(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t start, mag_scalar_t end, mag_scalar_t step);
extern MAG_EXPORT mag_status_t mag_one_hot(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *indices, int64_t num_classes);
extern MAG_EXPORT mag_status_t mag_rand_perm(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t n);
extern MAG_EXPORT mag_status_t mag_load_image(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, const char *file, const char *channels, uint32_t resize_width, uint32_t resize_height);
extern MAG_EXPORT mag_status_t mag_copy_raw_(mag_error_t *err, mag_tensor_t *tensor, const void *data, size_t size_bytes);
extern MAG_EXPORT mag_status_t mag_zero_(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_masked_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_uniform_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t min, mag_scalar_t max);
extern MAG_EXPORT mag_status_t mag_normal_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t mean, mag_scalar_t stddev);
extern MAG_EXPORT mag_status_t mag_bernoulli_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t p);
extern MAG_EXPORT mag_status_t mag_clone(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_cast(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_dtype_t dst_type);
extern MAG_EXPORT mag_status_t mag_view(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank);
extern MAG_EXPORT mag_status_t mag_view_slice(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t len, int64_t step);
extern MAG_EXPORT mag_status_t mag_reshape(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank);
extern MAG_EXPORT mag_status_t mag_transpose(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim1, int64_t dim2);
extern MAG_EXPORT mag_status_t mag_T(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_permute(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank);
extern MAG_EXPORT mag_status_t mag_contiguous(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_squeeze_all(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_squeeze_dim(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim);
extern MAG_EXPORT mag_status_t mag_unsqueeze(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim) ;
extern MAG_EXPORT mag_status_t mag_flatten(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t start_dim, int64_t end_dim);
extern MAG_EXPORT mag_status_t mag_unflatten(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, const int64_t *sizes, int64_t sizes_rank);
extern MAG_EXPORT mag_status_t mag_narrow(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t length);
extern MAG_EXPORT mag_status_t mag_movedim(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t src, int64_t dst);
extern MAG_EXPORT mag_status_t mag_select(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t index);
extern MAG_EXPORT mag_status_t mag_split(mag_error_t *err, mag_tensor_t **outs, int64_t num_splits, mag_tensor_t *x, int64_t split_size, int64_t dim);
extern MAG_EXPORT mag_status_t mag_mean(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_min(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_max(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_argmin(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_argmax(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_sum(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_prod(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_all(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_any(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_topk(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t k, int64_t dim, bool largest, bool sorted);
extern MAG_EXPORT mag_status_t mag_abs(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_abs_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sgn(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sgn_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_neg(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_neg_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_log(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_log_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_log10(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_log10_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_log1p(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_log1p_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_log2(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_log2_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sqr(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sqr_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_rcp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_rcp_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sqrt(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sqrt_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_rsqrt(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_rsqrt_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sin(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sin_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_cos(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_cos_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_tan(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_tan_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sinh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sinh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_cosh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_cosh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_tanh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_tanh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_asin(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_asin_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_acos(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_acos_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_atan(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_atan_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_asinh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_asinh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_acosh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_acosh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_atanh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_atanh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_step(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_step_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_erf(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_erf_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_erfc(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_erfc_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_exp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_exp_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_exp2(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_exp2_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_expm1(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_expm1_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_floor(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_floor_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_ceil(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_ceil_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_round(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_round_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_trunc(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_trunc_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_softmax(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_softmax_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_softmax_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_softmax_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sigmoid(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sigmoid_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sigmoid_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_sigmoid_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_hard_sigmoid(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_hard_sigmoid_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_silu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_silu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_silu_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_silu_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_tanh_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_tanh_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_relu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_relu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_relu_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_relu_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_gelu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_gelu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_gelu_approx(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_gelu_approx_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_gelu_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_gelu_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_add(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_add_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_sub(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_sub_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_mul(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_mul_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_div(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_div_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_floordiv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_floordiv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_mod(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_mod_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_pow(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_pow_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_matmul(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_repeat_back(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_gather(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t dim, mag_tensor_t *idx);
extern MAG_EXPORT mag_status_t mag_and(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_and_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_or(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_or_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_xor(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_xor_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_not(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_not_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_shl(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_shl_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_shr(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_shr_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_eq(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_ne(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_le(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_ge(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_lt(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_gt(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_where(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *cond, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_tril(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag);
extern MAG_EXPORT mag_status_t mag_tril_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag);
extern MAG_EXPORT mag_status_t mag_triu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag);
extern MAG_EXPORT mag_status_t mag_triu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int32_t diag);
extern MAG_EXPORT mag_status_t mag_multinomial(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t num_samples, bool replacement);
extern MAG_EXPORT mag_status_t mag_cat(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count, int64_t dim);

extern MAG_EXPORT int64_t mag_tensor_rank(const mag_tensor_t *tensor);
extern MAG_EXPORT const int64_t *mag_tensor_shape_ptr(const mag_tensor_t *tensor);
extern MAG_EXPORT const int64_t *mag_tensor_strides_ptr(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_dtype_t mag_tensor_type(const mag_tensor_t *tensor);
extern MAG_EXPORT size_t mag_tensor_data_offset(const mag_tensor_t *tensor);
extern MAG_EXPORT uintptr_t mag_tensor_data_ptr(const mag_tensor_t *tensor);
extern MAG_EXPORT uintptr_t mag_tensor_data_ptr_mut(const mag_tensor_t *tensor);
extern MAG_EXPORT uintptr_t mag_tensor_data_storage_ptr(const mag_tensor_t *tensor);
extern MAG_EXPORT uintptr_t mag_tensor_data_storage_ptr_mut(const mag_tensor_t *tensor);
extern MAG_EXPORT size_t mag_tensor_numbytes(const mag_tensor_t *tensor);
extern MAG_EXPORT int64_t mag_tensor_numel(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_context_t *mag_tensor_context(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_view(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_floating_point_typed(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_integral_typed(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_integer_typed(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_unsigned_integer_typed(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_signed_integer_typed(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_numeric_typed(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_shape_eq(const mag_tensor_t *x, const mag_tensor_t *y);
extern MAG_EXPORT bool mag_tensor_are_strides_eq(const mag_tensor_t *x, const mag_tensor_t *y);
extern MAG_EXPORT bool mag_tensor_can_broadcast(const mag_tensor_t *small, const mag_tensor_t *big);
extern MAG_EXPORT bool mag_tensor_is_transposed(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_permuted(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_contiguous(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_can_view(const mag_tensor_t *tensor, const int64_t *dims, int64_t rank);
extern MAG_EXPORT mag_status_t mag_tensor_grad(mag_error_t *err, const mag_tensor_t *tensor, mag_tensor_t **out_grad);
extern MAG_EXPORT bool mag_tensor_requires_grad(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_tensor_set_requires_grad(mag_error_t *err, mag_tensor_t *tensor, bool requires_grad);
extern MAG_EXPORT mag_status_t mag_tensor_backward(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_tensor_zero_grad(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_tensor_copy_data(mag_error_t *err, mag_tensor_t *tensor, void **out_buf, size_t *out_size_bytes);
extern MAG_EXPORT void mag_tensor_copy_data_free(void *ret_val);
extern MAG_EXPORT mag_status_t mag_tensor_item(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t *out_value);
extern MAG_EXPORT mag_status_t mag_eval(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_evalv(mag_error_t *err, mag_tensor_t **tensors, size_t count);
extern MAG_EXPORT mag_tensor_t *mag_tensor_detach(mag_tensor_t *tensor);
extern MAG_EXPORT const char *mag_tensor_to_string(mag_tensor_t *tensor, int64_t head, int64_t tail, int64_t threshold);
extern MAG_EXPORT void mag_tensor_to_string_free_data(const char *ret_val);
extern MAG_EXPORT void mag_tensor_incref(mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_decref(mag_tensor_t *tensor);
extern MAG_EXPORT void mag_tensor_visualize_backprop_graph(mag_tensor_t *tensor, const char *file);
extern MAG_EXPORT void mag_tensor_visualize_execution_graph(mag_tensor_t *tensor, const char *file);

typedef struct mag_snapshot_t mag_snapshot_t;

/* TODO: Migrate to new error system */

extern MAG_EXPORT mag_snapshot_t *mag_snapshot_new(mag_context_t *ctx);
extern MAG_EXPORT mag_snapshot_t *mag_snapshot_deserialize(mag_context_t *ctx, const char *filename);
extern MAG_EXPORT bool mag_snapshot_serialize(mag_snapshot_t *snap, const char *filename);

extern MAG_EXPORT mag_tensor_t *mag_snapshot_get_tensor(mag_snapshot_t *snap, const char *key);
extern MAG_EXPORT bool mag_snapshot_put_tensor(mag_snapshot_t *snap, const char *key, mag_tensor_t *tensor);
extern MAG_EXPORT size_t mag_snapshot_get_num_tensors(mag_snapshot_t *snap);
extern MAG_EXPORT const char **mag_snapshot_get_tensor_keys(mag_snapshot_t *snap, size_t *out_num_keys);
extern MAG_EXPORT void mag_snapshot_free_tensor_keys(const char **keys, size_t num_keys);

extern MAG_EXPORT void mag_snapshot_print_info(mag_snapshot_t *snap);
extern MAG_EXPORT void mag_snapshot_free(mag_snapshot_t *snap);

#ifdef __cplusplus
}
#endif
#endif
