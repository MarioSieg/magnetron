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
#define MAG_VERSION mag_ver_encode(0, 1, 9)
#define MAG_SNAPSHOT_VERSION mag_ver_encode(0, 3, 0)

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

/**
 * Status return codes for magnetron library functions.
 */
#define mag_statusdef(_) \
  _(MAG_OK, "Success") \
  _(MAG_ERR_PENDING, "Operation already in progress") \
  _(MAG_ERR_THREAD, "Called from the wrong thread") \
  _(MAG_ERR_RANK, "Invalid tensor rank") \
  _(MAG_ERR_DIM, "Invalid tensor dimension") \
  _(MAG_ERR_SHAPE, "Invalid tensor shape") \
  _(MAG_ERR_INDEX, "Invalid index") \
  _(MAG_ERR_DEVICE, "Invalid device") \
  _(MAG_ERR_BOUNDS, "Index out of bounds") \
  _(MAG_ERR_PARAM, "Invalid argument") \
  _(MAG_ERR_STRIDES, "Failed to compute tensor strides") \
  _(MAG_ERR_BROADCAST, "Broadcasting is not possible") \
  _(MAG_ERR_OP, "Operation is not supported for the given operands") \
  _(MAG_ERR_STATE, "Invalid object state") \
  _(MAG_ERR_IMAGE, "Image processing failed") \
  _(MAG_ERR_OOM, "Out of memory") \
  _(MAG_ERR_FREE, "Memory deallocation failed") \
  _(MAG_ERR_MMAP, "Failed to memory-map file") \
  _(MAG_ERR_IO, "I/O error") \
  _(MAG_ERR_SERIALIZE, "Serialization failed") \
  _(MAG_ERR_KERNEL, "Compute kernel execution failed") \
  _(MAG_ERR_EINSUM, "Einsum operation failed") \
  _(MAG_ERR_NOFILE, "File not found") \
  _(MAG_ERR_OS, "Operating system error") \
  _(MAG_ERR_BACKEND, "Backend error") \
  _(MAG_ERR_AUTOGRAD, "Autograd error") \
  _(MAG_ERR_UNKNOWN, "Unknown error")

typedef enum mag_status_t {
#define _(code, msg) code,
  mag_statusdef(_)
#undef _
} mag_status_t;
extern MAG_EXPORT const char *mag_status_get_name(mag_status_t op);
extern MAG_EXPORT const char *mag_status_get_message(mag_status_t op);

/* Name, ID, Required */
#define mag_backenddef(_)\
  _(CPU, cpu, true)\
  _(CUDA, cuda, false)\
  _(CUSTOM, custom, false)\

typedef enum mag_backend_type_t {
#define _(name, id, required) MAG_BACKEND_TYPE_##name,
  mag_backenddef(_)
  MAG_BACKEND_TYPE__COUNT
#undef _
} mag_backend_type_t;
mag_static_assert(MAG_BACKEND_TYPE__COUNT <= 0xff);
extern MAG_EXPORT const char *mag_backend_type_to_str(mag_backend_type_t type);
extern MAG_EXPORT bool mag_backend_type_is_required(mag_backend_type_t type);
extern MAG_EXPORT bool mag_backend_type_has_device_ordinals(mag_backend_type_t type);

#define MAG_DEVICE_ORDINAL_MAX ((1u<<15u)-1u)

typedef struct mag_device_id_t {
  bool is_virtual : 1;                      /* If true - device is a virtual device (called meta device in PyTorch). */
  uint32_t device_ordinal : 15;             /* !Ignored if is_virtual=true! 15-bit device index for the given backend type, (e.g. 0 for cuda:0). */
  mag_backend_type_t type : 8;              /* !Ignored if is_virtual=true! 8-bit backend type, (e.g. CPU, CUDA, etc..) */
} mag_device_id_t;
mag_static_assert(sizeof(mag_device_id_t) <= 8); /* We want this compact <= 8B or 4 */
extern MAG_EXPORT void mag_device_id_to_str(mag_device_id_t id, char (*buf)[32]);
extern MAG_EXPORT bool mag_device_id_eq(mag_device_id_t a, mag_device_id_t b);

/* Designated, not positional: 'is_virtual' is the first member, so a positional list would land the backend
   type in it and leave 'type' defaulted to CPU. That happened to work only for mag_device(CPU, 0). */
#define mag_device(name, ordinal) ((mag_device_id_t){.is_virtual=false, .device_ordinal=(ordinal), .type=MAG_BACKEND_TYPE_##name})

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

/* === Scalar Value === */

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
    double float64;
    int64_t int64;
    uint64_t uint64;
  } value;
} mag_scalar_t;

extern MAG_EXPORT mag_scalar_t mag_scalar_from_float64(double value);
extern MAG_EXPORT mag_scalar_t mag_scalar_from_int64(int64_t value);
extern MAG_EXPORT mag_scalar_t mag_scalar_from_uint64(uint64_t value);
extern MAG_EXPORT bool mag_scalar_is_float64(mag_scalar_t s);
extern MAG_EXPORT bool mag_scalar_is_int64(mag_scalar_t s);
extern MAG_EXPORT bool mag_scalar_is_uint64(mag_scalar_t s);
extern MAG_EXPORT double mag_scalar_as_float64(mag_scalar_t s);
extern MAG_EXPORT int64_t mag_scalar_as_int64(mag_scalar_t s);
extern MAG_EXPORT uint64_t mag_scalar_as_uint64(mag_scalar_t s);
extern MAG_EXPORT bool mag_scalar_same_type(mag_scalar_t a, mag_scalar_t b);
extern MAG_EXPORT bool mag_scalar_same_type_and_value(mag_scalar_t a, mag_scalar_t b);

/* === Data Type Handling === */

/**
 * @brief Data types for tensors. Never
 * @warning The ordinals are used on disk - never reorder, append only.
 */
typedef enum mag_dtype_t {
  MAG_DTYPE_FLOAT32,
  MAG_DTYPE_FLOAT16,
  MAG_DTYPE_BFLOAT16,
  MAG_DTYPE_FLOAT8_E4M3FN,
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

/* === Context === */

typedef struct mag_context_t mag_context_t;
extern MAG_EXPORT mag_status_t mag_ctx_create(mag_error_t *err, mag_context_t **out_ctx);                               /* Create context with default config, and only specify device type. */
extern MAG_EXPORT bool mag_ctx_is_device_available(mag_context_t *ctx, mag_device_id_t id);                             /* Check if a device is available in the context. */
extern MAG_EXPORT void mag_ctx_grad_recorder_start(mag_context_t *ctx);                                                 /* Start gradient recording */
extern MAG_EXPORT void mag_ctx_grad_recorder_stop(mag_context_t *ctx);                                                  /* Stop gradient recording */
extern MAG_EXPORT bool mag_ctx_grad_recorder_is_running(const mag_context_t *ctx);                                      /* Check if gradient recording is running */
extern MAG_EXPORT void mag_ctx_manual_seed(mag_context_t *ctx, uint64_t seed);                                          /* Manually seed the PRNG. */
extern MAG_EXPORT mag_dtype_t mag_ctx_default_dtype(mag_context_t *ctx);                                                /* Get default floating point dtype for the context. This is used by factory functions when the dtype is not specified. */
extern MAG_EXPORT bool mag_ctx_set_default_dtype(mag_context_t *ctx, mag_dtype_t type);                                 /* Set default floating point dtype for the context. This is used by factory functions when the dtype is not specified. Must be a floating point type. */
extern MAG_EXPORT mag_device_id_t mag_ctx_default_device(mag_context_t *ctx);                                           /* Get the device used when a caller names none. Defaults to the CPU device - having a GPU present does not change it. */
extern MAG_EXPORT mag_status_t mag_ctx_set_default_device(mag_error_t *err, mag_context_t *ctx, mag_device_id_t id);    /* Set the device used when a caller names none. Fails if the device is not available. */
extern MAG_EXPORT mag_status_t mag_ctx_best_device(mag_error_t *err, mag_context_t *ctx, mag_backend_type_t type, mag_device_id_t *out_id); /* Backend's own pick of its fastest device, e.g. mag_ctx_best_device(err, ctx, MAG_BACKEND_TYPE_CUDA, &id). */
extern MAG_EXPORT void mag_ctx_destroy(mag_context_t *ctx, bool suppress_leak_detection);                               /* Destroy context and free memory */

/* === Tensor Factories === */

typedef struct mag_tensor_t mag_tensor_t;
extern MAG_EXPORT mag_status_t mag_empty(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_strided_view(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset);
extern MAG_EXPORT mag_status_t mag_broadcast(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, int64_t rank, const int64_t *shape);
extern MAG_EXPORT mag_status_t mag_expand(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, int64_t rank, const int64_t *shape);
extern MAG_EXPORT mag_status_t mag_empty_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like);
extern MAG_EXPORT mag_status_t mag_empty_scalar(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_scalar(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t value, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_full(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t value, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_full_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_zeros(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_zeros_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like);
extern MAG_EXPORT mag_status_t mag_ones(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_ones_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like);
extern MAG_EXPORT mag_status_t mag_uniform(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t min, mag_scalar_t max, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_uniform_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t min, mag_scalar_t max);
extern MAG_EXPORT mag_status_t mag_normal(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t mean, mag_scalar_t stddev, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_normal_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, mag_scalar_t mean, mag_scalar_t stddev);
extern MAG_EXPORT mag_status_t mag_bernoulli(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, int64_t rank, const int64_t *shape, double p, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_bernoulli_like(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *like, double p);
extern MAG_EXPORT mag_status_t mag_arange(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t start, mag_scalar_t end, mag_scalar_t step, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_linspace(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t start, mag_scalar_t end, int64_t steps, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_eye(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t n, int64_t m, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_meshgrid(mag_error_t *err, mag_tensor_t **out_results, mag_tensor_t **tensors, size_t count) ;
extern MAG_EXPORT mag_status_t mag_one_hot(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *indices, int64_t num_classes);
extern MAG_EXPORT mag_status_t mag_rand_perm(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, mag_dtype_t type, int64_t n, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_load_image(mag_error_t *err, mag_tensor_t **out_result, mag_context_t *ctx, const char *file, const char *channels, uint32_t resize_width, uint32_t resize_height, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_save_image(mag_error_t *err, mag_tensor_t *tensor, const char *file);
extern MAG_EXPORT mag_status_t mag_load_audio(mag_error_t *err, mag_tensor_t **out, mag_context_t *ctx, const char *file, uint32_t *out_sample_rate, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_save_audio(mag_error_t *err, mag_tensor_t *tensor, const char *file, uint32_t sample_rate);
extern MAG_EXPORT mag_status_t mag_borrow_cpu_buffer(
    mag_error_t *err,
    mag_tensor_t **out,
    mag_context_t *ctx,
    void *data,
    size_t num_bytes,
    mag_dtype_t dtype,
    int64_t rank,
    const int64_t *shape,
    bool is_writeable,
    void (*release_cb)(void *usr),
    void *usr
);

/* === Tensor Inplace Fill Operators === */

extern MAG_EXPORT mag_status_t mag_copy_(mag_error_t *err, mag_tensor_t *dst, mag_tensor_t *src);
extern MAG_EXPORT mag_status_t mag_copy_raw_(mag_error_t *err, mag_tensor_t *tensor, const void *data, size_t size_bytes);
extern MAG_EXPORT mag_status_t mag_zeros_(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_ones_(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_masked_fill(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_masked_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_uniform_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t low, mag_scalar_t high);
extern MAG_EXPORT mag_status_t mag_normal_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t mean, mag_scalar_t stddev);
extern MAG_EXPORT mag_status_t mag_bernoulli_(mag_error_t *err, mag_tensor_t *tensor, double p);

/* === Tensor Operators === */

extern MAG_EXPORT mag_status_t mag_clone(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_cast(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_dtype_t dst_type);
extern MAG_EXPORT mag_status_t mag_transfer(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_device_id_t device);
extern MAG_EXPORT mag_status_t mag_view(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank);
extern MAG_EXPORT mag_status_t mag_reinterpret_view(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_dtype_t dtype, const int64_t *dims, int64_t rank);
extern MAG_EXPORT mag_status_t mag_view_slice(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim, int64_t start, int64_t len, int64_t step);
extern MAG_EXPORT mag_status_t mag_reshape(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank);
extern MAG_EXPORT mag_status_t mag_transpose(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim1, int64_t dim2);
extern MAG_EXPORT mag_status_t mag_T(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_permute(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank);
extern MAG_EXPORT mag_status_t mag_flip(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t ndims);
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
extern MAG_EXPORT mag_status_t mag_minima(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_maxima(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_argmin(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_argmax(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_sum(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_prod(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_all(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_any(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_status_t mag_topk(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t k, int64_t dim, bool largest, bool sorted);
extern MAG_EXPORT mag_status_t mag_cusum(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim);
extern MAG_EXPORT mag_status_t mag_cuprod(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim);
extern MAG_EXPORT mag_status_t mag_cumax(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t dim);
extern MAG_EXPORT mag_status_t mag_cumin(mag_error_t *err, mag_tensor_t **out_values, mag_tensor_t **out_indices, mag_tensor_t *x, int64_t dim);
extern MAG_EXPORT mag_status_t mag_outer(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *a, mag_tensor_t *b);
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
extern MAG_EXPORT mag_status_t mag_repeat(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *repeats, int64_t repeats_len);
extern MAG_EXPORT mag_status_t mag_repeat_interleave(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, bool flatten, int64_t dim, const int64_t *counts, int64_t count_len);
extern MAG_EXPORT mag_status_t mag_gather(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t dim, mag_tensor_t *idx);
extern MAG_EXPORT mag_status_t mag_embedding(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *weight, mag_tensor_t *indices);
extern MAG_EXPORT mag_status_t mag_index_add_(mag_error_t *err, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *source, double alpha);
extern MAG_EXPORT mag_status_t mag_scatter(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src);
extern MAG_EXPORT mag_status_t mag_scatter_(mag_error_t *err, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src);
extern MAG_EXPORT mag_status_t mag_scatter_add(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src);
extern MAG_EXPORT mag_status_t mag_scatter_add_(mag_error_t *err, mag_tensor_t *self, int64_t dim, mag_tensor_t *index, mag_tensor_t *src);
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
extern MAG_EXPORT mag_status_t mag_min(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_max(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_where(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *cond, mag_tensor_t *x, mag_tensor_t *y);
extern MAG_EXPORT mag_status_t mag_clamp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *min, mag_tensor_t *max);
extern MAG_EXPORT mag_status_t mag_clamp_min(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *min);
extern MAG_EXPORT mag_status_t mag_clamp_max(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *max);
extern MAG_EXPORT mag_status_t mag_lerp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *start, mag_tensor_t *end, mag_tensor_t *weight);
extern MAG_EXPORT mag_status_t mag_lerp_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *start, mag_tensor_t *end, mag_tensor_t *weight);
extern MAG_EXPORT mag_status_t mag_pad(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *pad, int64_t pad_len, const char *mode,  mag_scalar_t value);
extern MAG_EXPORT mag_status_t mag_tril(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag);
extern MAG_EXPORT mag_status_t mag_tril_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag);
extern MAG_EXPORT mag_status_t mag_triu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag);
extern MAG_EXPORT mag_status_t mag_triu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag);
extern MAG_EXPORT mag_status_t mag_multinomial(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t num_samples, bool replacement);
extern MAG_EXPORT mag_status_t mag_cat(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count, int64_t dim);
extern MAG_EXPORT mag_status_t mag_stack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count, int64_t dim);
extern MAG_EXPORT mag_status_t mag_hstack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count);
extern MAG_EXPORT mag_status_t mag_vstack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count);
extern MAG_EXPORT mag_status_t mag_dstack(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t **tensors, size_t count);
extern MAG_EXPORT mag_status_t mag_einsum(mag_error_t *err, mag_tensor_t **out_result, const char *equation, mag_tensor_t **args, size_t num_args);
extern MAG_EXPORT mag_status_t mag_detach(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor);

/* === Tensor Methods === */

extern MAG_EXPORT int64_t mag_tensor_rank(const mag_tensor_t *tensor);
extern MAG_EXPORT const int64_t *mag_tensor_shape_ptr(const mag_tensor_t *tensor);
extern MAG_EXPORT const int64_t *mag_tensor_strides_ptr(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_dtype_t mag_tensor_type(const mag_tensor_t *tensor);
extern MAG_EXPORT size_t mag_tensor_data_offset(const mag_tensor_t *tensor);
extern MAG_EXPORT uintptr_t mag_tensor_data_ptr(const mag_tensor_t *tensor);
extern MAG_EXPORT uintptr_t mag_tensor_data_ptr_mut(const mag_tensor_t *tensor);
extern MAG_EXPORT uintptr_t mag_tensor_data_storage_ptr(const mag_tensor_t *tensor);
extern MAG_EXPORT uintptr_t mag_tensor_data_storage_ptr_mut(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_device_id_t mag_tensor_device_id(const mag_tensor_t *tensor);
extern MAG_EXPORT size_t mag_tensor_numbytes(const mag_tensor_t *tensor);
extern MAG_EXPORT size_t mag_tensor_storage_numbytes(const mag_tensor_t *tensor);
extern MAG_EXPORT int64_t mag_tensor_numel(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_context_t *mag_tensor_context(const mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_view(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_tensor_t *mag_tensor_view_base(const mag_tensor_t *tensor);
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
extern MAG_EXPORT mag_tensor_t *mag_tensor_grad(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_tensor_set_grad(mag_error_t *err, mag_tensor_t *tensor, mag_tensor_t *grad);
extern MAG_EXPORT bool mag_tensor_requires_grad(const mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_tensor_set_requires_grad(mag_error_t *err, mag_tensor_t *tensor, bool requires_grad);
extern MAG_EXPORT mag_status_t mag_tensor_backward(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_tensor_zero_grad(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_tensor_copy_data(mag_error_t *err, mag_tensor_t *tensor, void **out_buf, size_t *out_size_bytes);
extern MAG_EXPORT void mag_tensor_copy_data_free(void *ret_val);
extern MAG_EXPORT mag_status_t mag_tensor_item(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t *out_value);
extern MAG_EXPORT const char *mag_tensor_to_string(mag_tensor_t *tensor, int64_t head, int64_t tail, int64_t threshold);
extern MAG_EXPORT void mag_tensor_to_string_free_data(const char *ret_val);
extern MAG_EXPORT void mag_tensor_incref(mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_decref(mag_tensor_t *tensor);
extern MAG_EXPORT bool mag_tensor_is_cpu(mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_tensor_visualize_backprop_graph(mag_error_t *err, mag_tensor_t *tensor, const char *file);

/* === Snapshot De/Serialization === */

typedef struct mag_snapshot_stream_writer_t mag_snapshot_stream_writer_t;
extern MAG_EXPORT mag_status_t mag_snapshot_stream_writer_open(
  mag_error_t *err,
  mag_snapshot_stream_writer_t **writer,
  mag_context_t *ctx,
  const char *filepath,
  const char *meta_document,
  uint64_t meta_len,
  uint64_t blob_len
);
extern MAG_EXPORT mag_status_t mag_snapshot_stream_writer_submit_blob(
  mag_error_t *err,
  mag_snapshot_stream_writer_t *writer,
  const void *blob,
  uint64_t size
);
extern MAG_EXPORT mag_status_t mag_snapshot_stream_writer_close(mag_error_t *err, mag_snapshot_stream_writer_t *writer);
extern MAG_EXPORT void mag_snapshot_stream_writer_abort(mag_snapshot_stream_writer_t *writer);

typedef struct mag_snapshot_stream_reader_t mag_snapshot_stream_reader_t;
extern MAG_EXPORT mag_status_t mag_snapshot_stream_reader_open(
  mag_error_t *err,
  mag_snapshot_stream_reader_t **reader,
  mag_context_t *ctx,
  const char *filepath
);
extern MAG_EXPORT const char *mag_snapshot_stream_reader_meta(const mag_snapshot_stream_reader_t *reader, uint64_t *out_len); /* Warning! NOT NUL terminated!! */
extern MAG_EXPORT uint64_t mag_snapshot_stream_reader_blob_len(const mag_snapshot_stream_reader_t *reader);
extern MAG_EXPORT uint32_t mag_snapshot_stream_reader_version(const mag_snapshot_stream_reader_t *reader);
extern MAG_EXPORT mag_status_t mag_snapshot_stream_reader_borrow_tensor(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_snapshot_stream_reader_t *reader,
  uint64_t offset,
  uint64_t size,
  mag_dtype_t dtype,
  int64_t rank,
  const int64_t *shape
);
extern MAG_EXPORT void mag_snapshot_stream_reader_close(mag_snapshot_stream_reader_t *reader);


/* === Distributed & Process Group === */

typedef struct mag_process_group_t mag_process_group_t;
extern MAG_EXPORT mag_status_t mag_pgroup_init_tcp(
  mag_error_t *err,
  mag_process_group_t **out,
  const char *master_addr,
  uint16_t master_port,
  uint32_t rank,
  uint32_t world_size
);
extern MAG_EXPORT void mag_pgroup_destroy(mag_process_group_t *pgroup);
extern MAG_EXPORT uint32_t mag_pgroup_rank(const mag_process_group_t *pgroup);
extern MAG_EXPORT uint32_t mag_pgroup_world_size(const mag_process_group_t *pgroup);
extern MAG_EXPORT mag_status_t mag_pgroup_validate(mag_error_t *err, mag_process_group_t *pgroup);
extern MAG_EXPORT mag_status_t mag_pgroup_verify_tensor_is_wireable(mag_error_t *err, mag_tensor_t *tensor);
extern MAG_EXPORT mag_status_t mag_pgroup_send_bytes(mag_error_t *err, mag_process_group_t *pgroup, uint32_t dst_rank, const void *buf, size_t nb);
extern MAG_EXPORT mag_status_t mag_pgroup_recv_bytes(mag_error_t *err, mag_process_group_t *pgroup, uint32_t src_rank, void *buf, size_t nb);
extern MAG_EXPORT mag_status_t mag_pgroup_barrier(mag_error_t *err, mag_process_group_t *pgroup);
extern MAG_EXPORT mag_status_t mag_pgroup_broadcast_(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *x);
extern MAG_EXPORT mag_status_t mag_pgroup_all_reduce_sum_(mag_error_t *err, mag_process_group_t *pgroup, mag_tensor_t *x);

#ifdef __cplusplus
}
#endif
#endif
