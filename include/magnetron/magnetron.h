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

#define MAG_MAX_DIMS 16 /* Maximum number of dimensions for a tensor. Currently fixed. If 16-D is not enough for ur cursed use-case, bump this and recompile (: */

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
#define MAG_VERSION mag_ver_encode(0, 2, 1)
#define MAG_SNAPSHOT_VERSION mag_ver_encode(0, 3, 0)

typedef enum mag_log_level_t {
  MAG_LOG_LEVEL_NONE,
  MAG_LOG_LEVEL_ERROR,
  MAG_LOG_LEVEL_WARN,
  MAG_LOG_LEVEL_INFO,
  MAG_LOG_LEVEL_DEBUG
} mag_log_level_t;

/**
 * Set the global log level.
 *
 * @param level New log level; messages more verbose than it are suppressed.
 */
extern MAG_EXPORT void mag_set_log_level(mag_log_level_t level);

/**
 * Get the global log level.
 *
 * @return Current log level.
 */
extern MAG_EXPORT mag_log_level_t mag_log_level(void);

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
  _(MAG_ERR_COMM, "Distributed communicator error") \
  _(MAG_ERR_AUTOGRAD, "Autograd error") \
  _(MAG_ERR_UNKNOWN, "Unknown error")

/**
 * Status code returned by library functions. MAG_OK signals success; every other value is an error.
 */
typedef enum mag_status_t {
#define _(code, msg) code,
  mag_statusdef(_)
#undef _
} mag_status_t;

/**
 * Get the short descriptive string of a status code.
 *
 * @param op Status code.
 * @return Static NUL-terminated string; must not be freed.
 */
extern MAG_EXPORT const char *mag_status_get_name(mag_status_t op);

/**
 * Get the descriptive message of a status code.
 *
 * @param op Status code.
 * @return Static NUL-terminated string; must not be freed.
 */
extern MAG_EXPORT const char *mag_status_get_message(mag_status_t op);

/* Name, ID, Required */
#define mag_backenddef(_)\
  _(CPU, cpu, true)\
  _(CUDA, cuda, false)\
  _(CUSTOM, custom, false)\


/**
 * Compute backend kinds. Values fit in one byte and are embedded in mag_device_id_t.
 */
typedef enum mag_backend_type_t {
#define _(name, id, required) MAG_BACKEND_TYPE_##name,
  mag_backenddef(_)
  MAG_BACKEND_TYPE__COUNT
#undef _
} mag_backend_type_t;
mag_static_assert(MAG_BACKEND_TYPE__COUNT <= 0xff);

/**
 * Get the identifier string of a backend type.
 *
 * @param type Backend type.
 * @return Static string such as "cpu" or "cuda".
 */
extern MAG_EXPORT const char *mag_backend_type_to_str(mag_backend_type_t type);

/**
 * Check whether a backend must be loadable for context creation to succeed.
 *
 * @param type Backend type.
 * @return true if the backend is mandatory.
 */
extern MAG_EXPORT bool mag_backend_type_is_required(mag_backend_type_t type);

/**
 * Check whether a backend distinguishes multiple devices by ordinal.
 *
 * @param type Backend type.
 * @return true if device ordinals are meaningful; false for CPU.
 */
extern MAG_EXPORT bool mag_backend_type_has_device_ordinals(mag_backend_type_t type);

#define MAG_DEVICE_ORDINAL_MAX ((1u<<15u)-1u)

/**
 * Compact identifier of a compute device: backend type plus device ordinal, or a virtual device.
 */
typedef struct mag_device_id_t {
  bool is_virtual : 1;                      /* If true - device is a virtual device (called meta device in PyTorch). */
  uint32_t device_ordinal : 15;             /* !Ignored if is_virtual=true! 15-bit device index for the given backend type, (e.g. 0 for cuda:0). */
  mag_backend_type_t type : 8;              /* !Ignored if is_virtual=true! 8-bit backend type, (e.g. CPU, CUDA, etc..) */
} mag_device_id_t;
mag_static_assert(sizeof(mag_device_id_t) <= 8); /* We want this compact <= 8B or 4 */

/**
 * Format a device identifier as a string such as "cpu", "cuda:0", or "virtual".
 *
 * @param id Device identifier.
 * @param buf Output buffer for the NUL-terminated string.
 */
extern MAG_EXPORT void mag_device_id_to_str(mag_device_id_t id, char (*buf)[32]);

/**
 * Compare two device identifiers for equality.
 *
 * @param a First identifier.
 * @param b Second identifier.
 * @return true if all fields are equal.
 */
extern MAG_EXPORT bool mag_device_id_eq(mag_device_id_t a, mag_device_id_t b);

/**
 * Construct a non-virtual device identifier.
 *
 * @param name Backend name as spelled in mag_backenddef, e.g. CPU or CUDA.
 * @param ordinal Device ordinal within the backend.
 */
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

/**
 * Wrap a double value in a tagged scalar.
 *
 * @param value Value to wrap.
 * @return Scalar tagged as MAG_SCALAR_TYPE_F64.
 */
extern MAG_EXPORT mag_scalar_t mag_scalar_from_float64(double value);

/**
 * Wrap a int64_t value in a tagged scalar.
 *
 * @param value Value to wrap.
 * @return Scalar tagged as MAG_SCALAR_TYPE_I64.
 */
extern MAG_EXPORT mag_scalar_t mag_scalar_from_int64(int64_t value);

/**
 * Wrap a uint64_t value in a tagged scalar.
 *
 * @param value Value to wrap.
 * @return Scalar tagged as MAG_SCALAR_TYPE_U64.
 */
extern MAG_EXPORT mag_scalar_t mag_scalar_from_uint64(uint64_t value);

/**
 * Check whether a scalar holds a double value.
 *
 * @param s Scalar to inspect.
 * @return true if the type tag is MAG_SCALAR_TYPE_F64.
 */
extern MAG_EXPORT bool mag_scalar_is_float64(mag_scalar_t s);

/**
 * Check whether a scalar holds a int64_t value.
 *
 * @param s Scalar to inspect.
 * @return true if the type tag is MAG_SCALAR_TYPE_I64.
 */
extern MAG_EXPORT bool mag_scalar_is_int64(mag_scalar_t s);

/**
 * Check whether a scalar holds a uint64_t value.
 *
 * @param s Scalar to inspect.
 * @return true if the type tag is MAG_SCALAR_TYPE_U64.
 */
extern MAG_EXPORT bool mag_scalar_is_uint64(mag_scalar_t s);

/**
 * Read a scalar as double, converting from the stored type if needed.
 *
 * @param s Scalar to read.
 * @return Converted value.
 */
extern MAG_EXPORT double mag_scalar_as_float64(mag_scalar_t s);

/**
 * Read a scalar as int64_t, converting from the stored type if needed.
 *
 * @param s Scalar to read.
 * @return Converted value.
 */
extern MAG_EXPORT int64_t mag_scalar_as_int64(mag_scalar_t s);

/**
 * Read a scalar as uint64_t, converting from the stored type if needed.
 *
 * @param s Scalar to read.
 * @return Converted value.
 */
extern MAG_EXPORT uint64_t mag_scalar_as_uint64(mag_scalar_t s);

/**
 * Check whether two scalars carry the same type tag.
 *
 * @param a First scalar.
 * @param b Second scalar.
 * @return true if the type tags are equal.
 */
extern MAG_EXPORT bool mag_scalar_same_type(mag_scalar_t a, mag_scalar_t b);

/**
 * Check whether two scalars carry the same type tag and bit-identical payload.
 *
 * @param a First scalar.
 * @param b Second scalar.
 * @return true if type tags and payload bits are equal.
 */
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

/**
 * Compute the common data type two operand types are promoted to.
 *
 * @param out Receives the promoted type on success.
 * @param lhs Left operand type.
 * @param rhs Right operand type.
 * @return true if a promotion rule exists, false otherwise.
 */
extern MAG_EXPORT bool mag_promote_type(mag_dtype_t *out, mag_dtype_t lhs, mag_dtype_t rhs);

/**
 * Static properties of a data type.
 */
typedef struct mag_type_traits_t {
  const char *name;           /* Name of the data type. eg. bfloat16 */
  const char *short_name;     /* Short name of the data type. eg. bf16 */
  size_t size;                /* Size of the data type in bytes. Must be a power of two. */
  size_t alignment;           /* CPU Alignment of the data type in bytes. Must be a power of two. */
  mag_scalar_t min_val;       /* Minimum finite value representable by this data type, as a scalar. For integer types, this is the smallest integer. For floating point types, this is the smallest normalized positive value. */
  mag_scalar_t max_val;       /* Maximum finite value representable by this data type, as a scalar. For integer types, this is the largest integer. For floating point types, this is the largest finite value. */
} mag_type_traits_t;

/**
 * Get the traits of a data type.
 *
 * @param type Data type.
 * @return Pointer to static traits; valid for the lifetime of the process.
 */
extern MAG_EXPORT const mag_type_traits_t *mag_type_trait(mag_dtype_t type);

/**
 * Check whether a data type is a floating-point type.
 *
 * @param type Data type.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_type_category_is_floating_point(mag_dtype_t type);

/**
 * Check whether a data type is an unsigned integer type.
 *
 * @param type Data type.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_type_category_is_unsigned_integer(mag_dtype_t type);

/**
 * Check whether a data type is a signed integer type.
 *
 * @param type Data type.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_type_category_is_signed_integer(mag_dtype_t type);

/**
 * Check whether a data type is a signed or unsigned integer type. Boolean is excluded.
 *
 * @param type Data type.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_type_category_is_integer(mag_dtype_t type);

/**
 * Check whether a data type is an integer or boolean type.
 *
 * @param type Data type.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_type_category_is_integral(mag_dtype_t type);

/**
 * Check whether a data type is a floating-point or integer type. Boolean is excluded.
 *
 * @param type Data type.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_type_category_is_numeric(mag_dtype_t type);

/* === Context === */

/**
 * Opaque runtime context. Owns the backend registry, devices, allocators, and tensor bookkeeping.
 */
typedef struct mag_context_t mag_context_t;

/**
 * Create a context with the default configuration and load the required backends.
 *
 * @param err Error output; set when the call fails.
 * @param out_ctx Receives the new context.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ctx_create(mag_error_t *err, mag_context_t **out_ctx);

/**
 * Check whether a device can be used through the context.
 *
 * @param ctx Context.
 * @param id Device identifier.
 * @return true if the backend is loaded and the device exists.
 */
extern MAG_EXPORT bool mag_ctx_is_device_available(mag_context_t *ctx, mag_device_id_t id);

/**
 * Enable gradient recording on the calling thread.
 *
 * @param ctx Context.
 */
extern MAG_EXPORT void mag_ctx_grad_recorder_start(mag_context_t *ctx);

/**
 * Disable gradient recording on the calling thread. Operations issued while stopped are not tracked by autograd.
 *
 * @param ctx Context.
 */
extern MAG_EXPORT void mag_ctx_grad_recorder_stop(mag_context_t *ctx);

/**
 * Check whether gradient recording is enabled on the calling thread.
 *
 * @param ctx Context.
 * @return true if recording is enabled.
 */
extern MAG_EXPORT bool mag_ctx_grad_recorder_is_running(const mag_context_t *ctx);

/**
 * Seed the random number generators of all devices. The seed is replayed onto backends loaded later.
 *
 * @param ctx Context.
 * @param seed Seed value.
 */
extern MAG_EXPORT void mag_ctx_manual_seed(mag_context_t *ctx, uint64_t seed);

/**
 * Get the default floating-point dtype used by factories when none is specified.
 *
 * @param ctx Context.
 * @return Current default dtype.
 */
extern MAG_EXPORT mag_dtype_t mag_ctx_default_dtype(mag_context_t *ctx);

/**
 * Set the default floating-point dtype used by factories when none is specified.
 *
 * @param ctx Context.
 * @param type New default; must be a floating-point type.
 * @return true on success, false if @p type is not a floating-point type.
 */
extern MAG_EXPORT bool mag_ctx_set_default_dtype(mag_context_t *ctx, mag_dtype_t type);

/**
 * Get the device used when a caller names none. Defaults to the CPU device.
 *
 * @param ctx Context.
 * @return Current default device.
 */
extern MAG_EXPORT mag_device_id_t mag_ctx_default_device(mag_context_t *ctx);

/**
 * Set the device used when a caller names none. Fails if the device is not available.
 *
 * @param err Error output; set when the call fails.
 * @param ctx Context.
 * @param id New default device.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ctx_set_default_device(mag_error_t *err, mag_context_t *ctx, mag_device_id_t id);

/**
 * Query a backend for its preferred device.
 *
 * @param err Error output; set when the call fails.
 * @param ctx Context.
 * @param type Backend to query.
 * @param out_id Receives the device identifier.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ctx_best_device(mag_error_t *err, mag_context_t *ctx, mag_backend_type_t type, mag_device_id_t *out_id);

/**
 * Destroy a context and release its resources. All tensors created from it should have been released.
 *
 * @param ctx Context to destroy.
 * @param suppress_leak_detection If true, leaked tensors are reported as a warning instead of an error.
 */
extern MAG_EXPORT void mag_ctx_destroy(mag_context_t *ctx, bool suppress_leak_detection);

/* === Tensor Factories === */

/**
 * Opaque reference-counted tensor handle.
 */
typedef struct mag_tensor_t mag_tensor_t;

/**
 * Create a tensor with uninitialized contents.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Element data type.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_empty(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t rank,
  const int64_t *shape,
  mag_device_id_t device
);

/**
 * Create a view over the storage of @p base with explicit shape, strides, and offset. The view shares storage and must lie within it.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param base Tensor whose storage is viewed.
 * @param rank Number of dimensions in @p shape and @p strides.
 * @param shape Dimension sizes, @p rank entries.
 * @param strides Element strides, @p rank entries.
 * @param offset Storage offset of the first element, in elements.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_strided_view(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_tensor_t *base,
  int64_t rank,
  const int64_t *shape,
  const int64_t *strides,
  int64_t offset
);

/**
 * Create a broadcast view of @p x. Size-1 dims and newly prepended dims are stretched with stride 0.
 *
 * @param err Error output; set when the call fails.
 * @param out Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param rank Target rank; must be >= the rank of @p x.
 * @param shape Target shape, @p rank entries.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_broadcast(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_tensor_t *x,
  int64_t rank,
  const int64_t *shape
);

/**
 * Create an expanded view of @p x. Like mag_broadcast(), but -1 in @p shape keeps the existing size of that dim.
 *
 * @param err Error output; set when the call fails.
 * @param out Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param rank Target rank; must be >= the rank of @p x.
 * @param shape Target shape, @p rank entries; -1 keeps a dim.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_expand(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_tensor_t *x,
  int64_t rank,
  const int64_t *shape
);

/**
 * Create a tensor with uninitialized contents and the shape, dtype, and device of @p like.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param like Tensor whose shape, dtype, and device are reused.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_empty_like(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *like
);

/**
 * Create a rank-0 tensor with uninitialized contents.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Element data type.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_empty_scalar(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  mag_device_id_t device
);

/**
 * Create a rank-0 tensor holding @p value.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Element data type.
 * @param value Initial value, converted to @p type.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_scalar(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  mag_scalar_t value,
  mag_device_id_t device
);

/**
 * Create a tensor filled with @p value.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Element data type.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @param value Fill value, converted to @p type.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_full(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t rank,
  const int64_t *shape,
  mag_scalar_t value,
  mag_device_id_t device
);

/**
 * Create a tensor filled with @p value, with the shape, dtype, and device of @p like.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param like Tensor whose shape, dtype, and device are reused.
 * @param value Fill value, converted to the dtype of @p like.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_full_like(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *like,
  mag_scalar_t value
);

/**
 * Create a tensor filled with zeros.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Element data type.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_zeros(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t rank,
  const int64_t *shape,
  mag_device_id_t device
);

/**
 * Create a tensor filled with zeros, with the shape, dtype, and device of @p like.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param like Tensor whose shape, dtype, and device are reused.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_zeros_like(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *like
);

/**
 * Create a tensor filled with ones.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Element data type.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ones(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t rank,
  const int64_t *shape,
  mag_device_id_t device
);

/**
 * Create a tensor filled with ones, with the shape, dtype, and device of @p like.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param like Tensor whose shape, dtype, and device are reused.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ones_like(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *like
);

/**
 * Create a tensor with values drawn from a uniform distribution between @p min and @p max.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Element data type.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @param min Lower bound.
 * @param max Upper bound; same scalar type as @p min and greater than it.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_uniform(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t rank,
  const int64_t *shape,
  mag_scalar_t min,
  mag_scalar_t max,
  mag_device_id_t device
);

/**
 * Create a uniformly distributed tensor with the shape, dtype, and device of @p like.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param like Tensor whose shape, dtype, and device are reused.
 * @param min Lower bound.
 * @param max Upper bound; same scalar type as @p min and greater than it.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_uniform_like(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *like,
  mag_scalar_t min,
  mag_scalar_t max
);

/**
 * Create a tensor with values drawn from a normal distribution.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Floating-point data type.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @param mean Mean, a floating-point scalar.
 * @param stddev Standard deviation, a non-negative floating-point scalar.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_normal(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t rank,
  const int64_t *shape,
  mag_scalar_t mean,
  mag_scalar_t stddev,
  mag_device_id_t device
);

/**
 * Create a normally distributed tensor with the shape, dtype, and device of @p like.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param like Tensor whose shape, dtype, and device are reused.
 * @param mean Mean, a floating-point scalar.
 * @param stddev Standard deviation, a non-negative floating-point scalar.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_normal_like(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *like,
  mag_scalar_t mean,
  mag_scalar_t stddev
);

/**
 * Create a boolean tensor whose elements are true with probability @p p.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @param p Probability in [0, 1].
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_bernoulli(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  int64_t rank,
  const int64_t *shape,
  double p,
  mag_device_id_t device
);

/**
 * Create a Bernoulli-distributed boolean tensor with the shape and device of @p like.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param like Tensor whose shape and device are reused.
 * @param p Probability in [0, 1].
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_bernoulli_like(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *like,
  double p
);

/**
 * Create a 1-D tensor with values from @p start up to, but excluding, @p end in increments of @p step.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Numeric data type.
 * @param start First value.
 * @param end Exclusive end value; same scalar type as @p start.
 * @param step Increment; same scalar type as @p start.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_arange(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  mag_scalar_t start,
  mag_scalar_t end,
  mag_scalar_t step,
  mag_device_id_t device
);

/**
 * Create a 1-D tensor of @p steps values evenly spaced from @p start to @p end inclusive.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Numeric data type.
 * @param start First value.
 * @param end Last value; same scalar type as @p start.
 * @param steps Number of values, > 0.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_linspace(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  mag_scalar_t start,
  mag_scalar_t end,
  int64_t steps,
  mag_device_id_t device
);

/**
 * Create a 2-D tensor with ones on the main diagonal and zeros elsewhere.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Element data type.
 * @param n Number of rows.
 * @param m Number of columns.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_eye(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t n,
  int64_t m,
  mag_device_id_t device
);

/**
 * Create coordinate grids from 1-D tensors. Output i is @p tensors[i] expanded along every other axis (matrix indexing).
 *
 * @param err Error output; set when the call fails.
 * @param out_results Array of @p count entries receiving the grid views; the caller owns one reference each.
 * @param tensors Array of @p count 1-D tensors.
 * @param count Number of tensors, at most MAG_MAX_DIMS.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_meshgrid(
  mag_error_t *err,
  mag_tensor_t **out_results,
  mag_tensor_t **tensors,
  size_t count
);

/**
 * Encode int64 indices as one-hot vectors along a new trailing dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param indices Index tensor, dtype int64.
 * @param num_classes Width of the one-hot dim; -1 infers it from the maximum index.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_one_hot(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *indices,
  int64_t num_classes
);

/**
 * Create a 1-D tensor holding a random permutation of the integers 0 to n-1.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param type Integer data type.
 * @param n Permutation length.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_rand_perm(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  mag_dtype_t type,
  int64_t n,
  mag_device_id_t device
);

/**
 * Decode an image file into a uint8 tensor of shape (channels, height, width).
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param file Path of the image file.
 * @param channels Channel layout to decode to: "GRAY", "GRAY_ALPHA", "RGB", or "RGBA".
 * @param resize_width Target width; 0 keeps the source width.
 * @param resize_height Target height; 0 keeps the source height.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_load_image(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_context_t *ctx,
  const char *file,
  const char *channels,
  uint32_t resize_width,
  uint32_t resize_height,
  mag_device_id_t device
);

/**
 * Encode a uint8 tensor of shape (channels, height, width) to an image file. The format follows the file extension.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Image tensor, dtype uint8, 1 to 4 channels.
 * @param file Destination path.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_save_image(
  mag_error_t *err,
  mag_tensor_t *tensor,
  const char *file
);

/**
 * Decode an audio file (.wav, .flac, or .mp3) into a float32 tensor of shape (channels, frames).
 *
 * @param err Error output; set when the call fails.
 * @param out Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param file Path of the audio file.
 * @param out_sample_rate Receives the sample rate in Hz; may be NULL.
 * @param device Device the tensor is allocated on.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_load_audio(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_context_t *ctx,
  const char *file,
  uint32_t *out_sample_rate,
  mag_device_id_t device
);

/**
 * Encode a float32 tensor of shape (channels, frames) to a .wav file.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Audio tensor, dtype float32.
 * @param file Destination path with a .wav extension.
 * @param sample_rate Sample rate in Hz, > 0.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_save_audio(
  mag_error_t *err,
  mag_tensor_t *tensor,
  const char *file,
  uint32_t sample_rate
);

/**
 * Wrap a caller-owned host buffer in a tensor without copying. The buffer must stay valid until @p release_cb is invoked.
 *
 * @param err Error output; set when the call fails.
 * @param out Receives the result tensor; the caller owns one reference.
 * @param ctx Owning context.
 * @param data Host buffer; must not be NULL.
 * @param num_bytes Buffer size in bytes; must cover the described tensor.
 * @param dtype Element data type.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @param is_writeable If false, the tensor is read-only.
 * @param release_cb Called with @p usr when the last reference to the storage is dropped; must not be NULL.
 * @param usr Opaque pointer passed to @p release_cb.
 * @return MAG_OK on success, an error status otherwise.
 */
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

/**
 * Copy the contents of @p src into @p dst. Shapes and dtypes must match.
 *
 * @param err Error output; set when the call fails.
 * @param dst Destination tensor.
 * @param src Source tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_copy_(mag_error_t *err, mag_tensor_t *dst, mag_tensor_t *src);

/**
 * Copy raw bytes into a contiguous CPU tensor.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Destination tensor; must be contiguous and on CPU.
 * @param data Source bytes.
 * @param size_bytes Byte count; must equal the tensor size in bytes.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_copy_raw_(mag_error_t *err, mag_tensor_t *tensor, const void *data, size_t size_bytes);

/**
 * Fill @p tensor with zeros in place.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor to fill.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_zeros_(mag_error_t *err, mag_tensor_t *tensor);

/**
 * Fill @p tensor with ones in place.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor to fill.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ones_(mag_error_t *err, mag_tensor_t *tensor);

/**
 * Fill @p tensor with @p value in place.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor to fill.
 * @param value Fill value, converted to the tensor dtype.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t value);

/**
 * Copy @p tensor, replacing the elements selected by @p mask with @p value.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensor Input tensor.
 * @param mask Mask tensor selecting the elements to replace.
 * @param value Replacement value.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_masked_fill(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value);

/**
 * Replace the elements of @p tensor selected by @p mask with @p value, in place.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor to modify; must not require grad while recording.
 * @param mask Mask tensor selecting the elements to replace.
 * @param value Replacement value.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_masked_fill_(mag_error_t *err, mag_tensor_t *tensor, mag_tensor_t *mask, mag_scalar_t value);

/**
 * Fill @p tensor in place with values drawn from a uniform distribution between @p low and @p high.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Numeric tensor to fill.
 * @param low Lower bound.
 * @param high Upper bound; same scalar type as @p low and greater than it.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_uniform_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t low, mag_scalar_t high);

/**
 * Fill @p tensor in place with values drawn from a normal distribution.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Floating-point tensor to fill.
 * @param mean Mean, a floating-point scalar.
 * @param stddev Standard deviation, a non-negative floating-point scalar.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_normal_(mag_error_t *err, mag_tensor_t *tensor, mag_scalar_t mean, mag_scalar_t stddev);

/**
 * Fill a boolean tensor in place with elements that are true with probability @p p.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Boolean tensor to fill.
 * @param p Probability in [0, 1].
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_bernoulli_(mag_error_t *err, mag_tensor_t *tensor, double p);

/* === Tensor Operators === */

/**
 * Create a contiguous deep copy of @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_clone(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Convert @p x to @p dst_type. If the dtype already matches, the result is a copy.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dst_type Target data type.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cast(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_dtype_t dst_type);

/**
 * Copy @p x to @p device. If @p x already resides there, it is returned itself with an added reference.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param device Destination device.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_transfer(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_device_id_t device);

/**
 * Create a view of @p x with a new shape. Fails if the existing strides cannot express the shape.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims New shape, @p rank entries.
 * @param rank Number of entries in @p dims.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_view(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank
);

/**
 * Reinterpret the bytes of @p x as @p dtype with a new shape, without copying. Not allowed on tensors that require grad.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dtype Data type to reinterpret the bytes as.
 * @param dims New shape, @p rank entries.
 * @param rank Number of entries in @p dims.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_reinterpret_view(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  mag_dtype_t dtype,
  const int64_t *dims,
  int64_t rank
);

/**
 * Create a strided slice view along one dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @param start First index; negative values count from the end.
 * @param len Number of elements; negative takes all remaining elements.
 * @param step Stride between elements, > 0.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_view_slice(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  int64_t dim,
  int64_t start,
  int64_t len,
  int64_t step
);

/**
 * Return @p x with a new shape, as a view when possible and otherwise as a contiguous copy. One entry of @p dims may be -1 and is inferred.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims New shape, @p rank entries; at most one -1.
 * @param rank Number of entries in @p dims.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_reshape(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank);

/**
 * Create a view of @p x with two dims swapped.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim1 First axis.
 * @param dim2 Second axis.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_transpose(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim1, int64_t dim2);

/**
 * Create a view of @p x with all dims reversed. Tensors of rank below 2 are returned as is with an added reference.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_T(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Create a view of @p x with its dims reordered.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Permutation of the axes, @p rank entries.
 * @param rank Number of entries in @p dims; must equal the rank of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_permute(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t rank);

/**
 * Create a view of @p x with the given axes reversed.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to flip, @p ndims entries, no duplicates.
 * @param ndims Number of entries in @p dims.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_flip(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, const int64_t *dims, int64_t ndims);

/**
 * Return a contiguous version of @p x: @p x itself with an added reference if already contiguous, otherwise a copy.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_contiguous(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Create a view of @p x with all size-1 dims removed.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_squeeze_all(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Create a view of @p x with the size-1 dim @p dim removed.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis to remove; must have size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_squeeze_dim(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim);

/**
 * Create a view of @p x with a size-1 dim inserted at @p dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Position of the new axis.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_unsqueeze(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, int64_t dim);

/**
 * Create a view of @p x with dims @p start_dim through @p end_dim merged into one.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param start_dim First axis to merge.
 * @param end_dim Last axis to merge, inclusive.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_flatten(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  int64_t start_dim,
  int64_t end_dim
);

/**
 * Create a view of @p x with dim @p dim split into several dims.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis to split.
 * @param sizes Sizes of the new dims, @p sizes_rank entries; their product must equal the size of @p dim.
 * @param sizes_rank Number of entries in @p sizes.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_unflatten(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  int64_t dim,
  const int64_t *sizes,
  int64_t sizes_rank
);

/**
 * Create a view of @p length consecutive elements along @p dim starting at @p start.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @param start First index.
 * @param length Number of elements.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_narrow(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  int64_t dim,
  int64_t start,
  int64_t length
);

/**
 * Create a view of @p x with dim @p src moved to position @p dst.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param src Axis to move.
 * @param dst Destination position.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_movedim(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  int64_t src,
  int64_t dst
);

/**
 * Create a view of @p x with dim @p dim fixed at @p index and removed.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @param index Index along @p dim.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_select(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  int64_t dim,
  int64_t index
);

/**
 * Split @p x into chunks of @p split_size along @p dim. Chunks are views; the last one may be smaller.
 *
 * @param err Error output; set when the call fails.
 * @param outs Array of @p num_splits entries receiving the chunk views; the caller owns one reference each.
 * @param num_splits Expected chunk count, ceil(size / split_size).
 * @param x Input tensor.
 * @param split_size Elements per chunk, > 0.
 * @param dim Axis; negative values count from the end.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_split(
  mag_error_t *err,
  mag_tensor_t **outs,
  int64_t num_splits,
  mag_tensor_t *x,
  int64_t split_size,
  int64_t dim
);

/**
 * Remove @p dim from @p x and return one view per index along it. Views share storage with @p x.
 *
 * @param err Error output; set when the call fails.
 * @param outs Array of @p num_outs entries receiving the views; the caller owns one reference each.
 * @param num_outs Expected output count, equal to the size of @p dim.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_unbind(
  mag_error_t *err,
  mag_tensor_t **outs,
  int64_t num_outs,
  mag_tensor_t *x,
  int64_t dim
);

/**
 * Compute the mean of @p x over the given dims.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_mean(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Compute the minimum of @p x over the given dims.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_minima(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Compute the maximum of @p x over the given dims.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_maxima(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Compute the index of the minimum of @p x over the given dims. The result has dtype int64.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_argmin(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Compute the index of the maximum of @p x over the given dims. The result has dtype int64.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_argmax(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Compute the sum of @p x over the given dims. Integer inputs accumulate in int64 or uint64.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sum(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Compute the product of @p x over the given dims. Integer inputs accumulate in int64 or uint64.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_prod(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Check whether all elements of @p x are true over the given dims. The result has dtype boolean.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_all(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Check whether any element of @p x is true over the given dims. The result has dtype boolean.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dims Axes to reduce, @p rank entries; NULL with @p rank 0 reduces all dims.
 * @param rank Number of entries in @p dims.
 * @param keepdim If true, reduced dims are kept with size 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_any(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *dims,
  int64_t rank,
  bool keepdim
);

/**
 * Select the @p k largest or smallest elements of @p x along @p dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_values Receives the selected values; the caller owns one reference.
 * @param out_indices Receives the int64 source indices; the caller owns one reference.
 * @param x Input tensor.
 * @param k Number of elements to select, in [1, size of @p dim].
 * @param dim Axis; negative values count from the end.
 * @param largest If true, select the largest elements; otherwise the smallest.
 * @param sorted If true, the results are sorted.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_topk(
  mag_error_t *err,
  mag_tensor_t **out_values,
  mag_tensor_t **out_indices,
  mag_tensor_t *x,
  int64_t k,
  int64_t dim,
  bool largest,
  bool sorted
);

/**
 * Sort @p x along @p dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_values Receives the sorted values; the caller owns one reference.
 * @param out_indices Receives the int64 source indices; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @param descending If true, sort in descending order.
 * @param stable If true, equal elements keep their input order.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sort(
  mag_error_t *err,
  mag_tensor_t **out_values,
  mag_tensor_t **out_indices,
  mag_tensor_t *x,
  int64_t dim,
  bool descending,
  bool stable
);

/**
 * Compute the int64 indices that sort @p x along @p dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_indices Receives the int64 indices; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @param descending If true, sort in descending order.
 * @param stable If true, equal elements keep their input order.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_argsort(
  mag_error_t *err,
  mag_tensor_t **out_indices,
  mag_tensor_t *x,
  int64_t dim,
  bool descending,
  bool stable
);

/**
 * Count the occurrences of each value in a 1-D integer tensor.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x 1-D integer tensor of non-negative values.
 * @param weights Optional 1-D numeric weights of the same length; NULL counts occurrences.
 * @param min_len Minimum length of the result.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_bincount(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  mag_tensor_t *weights,
  int64_t min_len
);

/**
 * Compute the indices of all non-zero elements of @p x. NaN counts as non-zero.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives an int64 tensor of shape [N, rank] holding one row-major multi-index per non-zero element; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_nonzero(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x
);

/**
 * Compute the cumulative sum of @p x along @p dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cusum(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  int64_t dim
);

/**
 * Compute the cumulative product of @p x along @p dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cuprod(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  int64_t dim
);

/**
 * Compute the running maximum of @p x along @p dim and the index at which each maximum occurred.
 *
 * @param err Error output; set when the call fails.
 * @param out_values Receives the running maxima; the caller owns one reference.
 * @param out_indices Receives the int64 indices; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cumax(
  mag_error_t *err,
  mag_tensor_t **out_values,
  mag_tensor_t **out_indices,
  mag_tensor_t *x,
  int64_t dim
);

/**
 * Compute the running minimum of @p x along @p dim and the index at which each minimum occurred.
 *
 * @param err Error output; set when the call fails.
 * @param out_values Receives the running minima; the caller owns one reference.
 * @param out_indices Receives the int64 indices; the caller owns one reference.
 * @param x Input tensor.
 * @param dim Axis; negative values count from the end.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cumin(
  mag_error_t *err,
  mag_tensor_t **out_values,
  mag_tensor_t **out_indices,
  mag_tensor_t *x,
  int64_t dim
);

/**
 * Compute the outer product of two 1-D tensors.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First 1-D tensor.
 * @param y Second 1-D tensor on the same device.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_outer(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  mag_tensor_t *y
);

/**
 * Compute the absolute value element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_abs(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the absolute value element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_abs_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the sign (-1, 0, or 1) element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sgn(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the sign (-1, 0, or 1) element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sgn_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the negation element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_neg(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the negation element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_neg_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the natural logarithm element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_log(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the natural logarithm element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_log_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the base-10 logarithm element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_log10(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the base-10 logarithm element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_log10_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute log(1 + x) element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_log1p(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute log(1 + x) element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_log1p_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the base-2 logarithm element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_log2(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the base-2 logarithm element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_log2_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the square element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sqr(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the square element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sqr_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the reciprocal element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_rcp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the reciprocal element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_rcp_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the square root element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sqrt(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the square root element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sqrt_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the reciprocal square root element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_rsqrt(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the reciprocal square root element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_rsqrt_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the sine element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sin(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the sine element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sin_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the cosine element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cos(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the cosine element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cos_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the tangent element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tan(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the tangent element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tan_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the hyperbolic sine element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sinh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the hyperbolic sine element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sinh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the hyperbolic cosine element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cosh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the hyperbolic cosine element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cosh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the hyperbolic tangent element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tanh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the hyperbolic tangent element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tanh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the arc sine element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_asin(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the arc sine element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_asin_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the arc cosine element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_acos(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the arc cosine element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_acos_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the arc tangent element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_atan(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the arc tangent element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_atan_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the inverse hyperbolic sine element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_asinh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the inverse hyperbolic sine element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_asinh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the inverse hyperbolic cosine element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_acosh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the inverse hyperbolic cosine element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_acosh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the inverse hyperbolic tangent element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_atanh(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the inverse hyperbolic tangent element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_atanh_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the Heaviside step function (1 for x > 0, otherwise 0) element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_step(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the Heaviside step function (1 for x > 0, otherwise 0) element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_step_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the error function element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_erf(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the error function element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_erf_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the complementary error function element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_erfc(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the complementary error function element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_erfc_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the exponential element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_exp(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the exponential element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_exp_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the base-2 exponential element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_exp2(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the base-2 exponential element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_exp2_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute exp(x) - 1 element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_expm1(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute exp(x) - 1 element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_expm1_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the floor element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_floor(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the floor element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_floor_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the ceiling element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ceil(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the ceiling element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ceil_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the value rounded to the nearest integer element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_round(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the value rounded to the nearest integer element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_round_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the value rounded toward zero element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_trunc(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the value rounded toward zero element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_trunc_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the softmax along the last dim element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_softmax(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the softmax along the last dim element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_softmax_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the softmax along the last dim element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_softmax_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the softmax along the last dim element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_softmax_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the logistic sigmoid element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sigmoid(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the logistic sigmoid element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sigmoid_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the logistic sigmoid element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sigmoid_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the logistic sigmoid element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sigmoid_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the hard sigmoid, clamp((x + 3) / 6, 0, 1) element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_hard_sigmoid(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the hard sigmoid, clamp((x + 3) / 6, 0, 1) element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_hard_sigmoid_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the SiLU activation, x * sigmoid(x) element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_silu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the SiLU activation, x * sigmoid(x) element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_silu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the SiLU activation element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_silu_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the SiLU activation element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_silu_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the hyperbolic tangent element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tanh_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the hyperbolic tangent element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tanh_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the ReLU activation, max(x, 0) element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_relu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the ReLU activation, max(x, 0) element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_relu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the ReLU activation element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_relu_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the ReLU activation element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_relu_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the exact (erf-based) GELU activation element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_gelu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the exact (erf-based) GELU activation element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_gelu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the tanh approximation of the GELU activation element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_gelu_approx(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the tanh approximation of the GELU activation element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_gelu_approx_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the GELU activation element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_gelu_dv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the derivative of the GELU activation element-wise, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_gelu_dv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute x + y element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_add(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x + y element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_add_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x - y element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sub(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x - y element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_sub_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x * y element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_mul(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x * y element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_mul_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x / y (true division) element-wise with broadcasting and dtype promotion. Integer operands are promoted to the default floating-point dtype.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_div(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x / y (true division) element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording. Must not be an integer tensor.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_div_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute floor(x / y) element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_floordiv(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute floor(x / y) element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording. Must be an integer tensor.
 * @param y Second operand; must be an integer tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_floordiv_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the remainder of x / y element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_mod(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the remainder of x / y element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_mod_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x raised to the power y element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_pow(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x raised to the power y element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_pow_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the matrix product of two floating-point tensors. Leading batch dims are broadcast.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Left operand, rank >= 1.
 * @param y Right operand, rank >= 1, same dtype as @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_matmul(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Reduce @p x to the shape of @p y by summing over broadcast dims. This is the adjoint of broadcasting and repeating.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Tensor to reduce.
 * @param y Tensor whose shape the result takes.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_repeat_back(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Tile @p x along each dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param repeats Repeat count per dim, @p num_reps entries, aligned to the trailing dims of @p x.
 * @param num_reps Number of entries in @p repeats, > 0.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_repeat(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *repeats,
  int64_t num_reps
);

/**
 * Repeat each element of @p x consecutively.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param flatten If true, operate on the flattened tensor and return a 1-D result.
 * @param dim Axis to repeat along; ignored when @p flatten is true.
 * @param counts Repeat counts, @p num_counts entries: one per element along the axis or a single count for all.
 * @param num_counts Number of entries in @p counts, > 0.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_repeat_interleave(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  bool flatten,
  int64_t dim,
  const int64_t *counts,
  int64_t num_counts
);

/**
 * Gather values from @p tensor along @p dim at the positions given by @p idx. The result has the shape of @p idx.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensor Source tensor.
 * @param dim Axis to index.
 * @param idx Index tensor, dtype int64, same rank as @p tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_gather(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *tensor,
  int64_t dim,
  mag_tensor_t *idx
);

/**
 * Look up rows of @p weight by index. The result shape is the shape of @p indices followed by the trailing dims of @p weight.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param weight Embedding table, rank >= 1.
 * @param indices Index tensor, dtype int64, rank >= 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_embedding(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *weight,
  mag_tensor_t *indices
);

/**
 * Add @p alpha times @p source into @p self at the positions of @p index along @p dim, in place.
 *
 * @param err Error output; set when the call fails.
 * @param self Tensor to accumulate into.
 * @param dim Axis; negative values count from the end.
 * @param index 1-D index tensor, dtype int64.
 * @param source Tensor to add; same rank and dtype as @p self, with the size of @p dim equal to the length of @p index.
 * @param alpha Scale factor applied to @p source.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_index_add_(
  mag_error_t *err,
  mag_tensor_t *self,
  int64_t dim,
  mag_tensor_t *index,
  mag_tensor_t *source,
  double alpha
);

/**
 * Copy @p self and write the elements of @p src into the copy at the positions given by @p index along @p dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param self Tensor to copy.
 * @param dim Axis to index.
 * @param index Index tensor, dtype int64.
 * @param src Values to write.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_scatter(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *self,
  int64_t dim,
  mag_tensor_t *index,
  mag_tensor_t *src
);

/**
 * Write the elements of @p src into @p self at the positions given by @p index along @p dim, in place.
 *
 * @param err Error output; set when the call fails.
 * @param self Tensor to modify; must not require grad while recording.
 * @param dim Axis to index.
 * @param index Index tensor, dtype int64.
 * @param src Values to write.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_scatter_(
  mag_error_t *err,
  mag_tensor_t *self,
  int64_t dim,
  mag_tensor_t *index,
  mag_tensor_t *src
);

/**
 * Copy @p self and add the elements of @p src into the copy at the positions given by @p idx along @p dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param self Tensor to copy.
 * @param dim Axis to index.
 * @param idx Index tensor, dtype int64.
 * @param src Values to add.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_scatter_add(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *self,
  int64_t dim,
  mag_tensor_t *idx,
  mag_tensor_t *src
);

/**
 * Add the elements of @p src into @p self at the positions given by @p idx along @p dim, in place.
 *
 * @param err Error output; set when the call fails.
 * @param self Tensor to modify; must not require grad while recording.
 * @param dim Axis to index.
 * @param idx Index tensor, dtype int64.
 * @param src Values to add.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_scatter_add_(
  mag_error_t *err,
  mag_tensor_t *self,
  int64_t dim,
  mag_tensor_t *idx,
  mag_tensor_t *src
);

/**
 * Compute the bitwise AND of x and y (logical for boolean tensors) element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_and(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the bitwise AND of x and y (logical for boolean tensors) element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_and_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the bitwise OR of x and y (logical for boolean tensors) element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_or(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the bitwise OR of x and y (logical for boolean tensors) element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_or_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the bitwise XOR of x and y (logical for boolean tensors) element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_xor(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the bitwise XOR of x and y (logical for boolean tensors) element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_xor_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the bitwise NOT element-wise (logical for boolean tensors).
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_not(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute the bitwise NOT element-wise, in place (logical for boolean tensors).
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_not_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x);

/**
 * Compute x shifted left by y bits element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_shl(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x shifted left by y bits element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_shl_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x shifted right by y bits element-wise with broadcasting and dtype promotion.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_shr(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute x shifted right by y bits element-wise into @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p x with an added reference.
 * @param x Tensor to operate on; overwritten with the result. Must not require grad while recording.
 * @param y Second operand; broadcast against @p x. The promoted dtype must equal the dtype of @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_shr_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compare x == y element-wise with broadcasting. The result has dtype boolean.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_eq(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compare x != y element-wise with broadcasting. The result has dtype boolean.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ne(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compare x <= y element-wise with broadcasting. The result has dtype boolean.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_le(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compare x >= y element-wise with broadcasting. The result has dtype boolean.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_ge(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compare x < y element-wise with broadcasting. The result has dtype boolean.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_lt(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compare x > y element-wise with broadcasting. The result has dtype boolean.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_gt(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the element-wise minimum of @p x and @p y with broadcasting.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_min(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Compute the element-wise maximum of @p x and @p y with broadcasting.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x First operand.
 * @param y Second operand; broadcast against @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_max(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *y);

/**
 * Select elements from @p x where @p cond is true and from @p y elsewhere. All operands are broadcast together.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param cond Condition tensor, dtype boolean.
 * @param x Values taken where @p cond is true.
 * @param y Values taken where @p cond is false; same dtype as @p x.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_where(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *cond,
  mag_tensor_t *x,
  mag_tensor_t *y
);

/**
 * Clamp @p x to the range [min, max] element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param min Lower bound tensor.
 * @param max Upper bound tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_clamp(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  mag_tensor_t *min,
  mag_tensor_t *max
);

/**
 * Clamp @p x from below element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param min Lower bound tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_clamp_min(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *min);

/**
 * Clamp @p x from above element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param max Upper bound tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_clamp_max(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *x, mag_tensor_t *max);

/**
 * Compute start + weight * (end - start) element-wise.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param start Start values.
 * @param end End values.
 * @param weight Interpolation weights.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_lerp(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *start,
  mag_tensor_t *end,
  mag_tensor_t *weight
);

/**
 * Compute start + weight * (end - start) element-wise into @p start.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p start with an added reference.
 * @param start Start values; overwritten with the result. Must not require grad while recording.
 * @param end End values.
 * @param weight Interpolation weights.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_lerp_(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *start,
  mag_tensor_t *end,
  mag_tensor_t *weight
);

/**
 * Compute an N-dimensional convolution over an input of layout (batch, channels, spatial...).
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input of rank @p spatial + 2, floating-point.
 * @param weight Filter of shape (out_channels, in_channels / groups, kernel...); same dtype as @p x.
 * @param bias Optional bias of shape (out_channels), or NULL.
 * @param spatial Number of spatial dims: 1, 2, or 3.
 * @param stride Stride per spatial dim, @p spatial entries.
 * @param padding Zero padding per spatial dim, @p spatial entries.
 * @param dilation Dilation per spatial dim, @p spatial entries.
 * @param groups Number of channel groups, >= 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_conv(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  mag_tensor_t *weight,
  mag_tensor_t *bias,
  int64_t spatial,
  const int64_t *stride,
  const int64_t *padding,
  const int64_t *dilation,
  int64_t groups
);

/**
 * Compute an N-dimensional transposed convolution over an input of layout (batch, channels, spatial...).
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input of rank @p spatial + 2, floating-point.
 * @param weight Filter of shape (in_channels, out_channels / groups, kernel...); same dtype as @p x.
 * @param bias Optional bias of shape (out_channels), or NULL.
 * @param spatial Number of spatial dims: 1, 2, or 3.
 * @param stride Stride per spatial dim, @p spatial entries.
 * @param padding Padding per spatial dim, @p spatial entries.
 * @param output_padding Extra size added to each output spatial dim, @p spatial entries.
 * @param dilation Dilation per spatial dim, @p spatial entries.
 * @param groups Number of channel groups, >= 1.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_convT(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  mag_tensor_t *weight,
  mag_tensor_t *bias,
  int64_t spatial,
  const int64_t *stride,
  const int64_t *padding,
  const int64_t *output_padding,
  const int64_t *dilation,
  int64_t groups
);

/**
 * Interpolation modes for tensor resizing and resampling operations.
 */
typedef enum mag_interp_mode_t {
  /** Nearest-neighbor interpolation. */
  MAG_INTERP_MODE_NEAREST = 0,
  /** Nearest-neighbor interpolation using exact source-index mapping. */
  MAG_INTERP_MODE_NEAREST_EXACT = 1,
  /** Linear interpolation for 1D inputs. */
  MAG_INTERP_MODE_LINEAR = 2,
  /** Bilinear interpolation for 2D inputs. */
  MAG_INTERP_MODE_BILINEAR = 3,
  /** Bicubic interpolation for 2D inputs. */
  MAG_INTERP_MODE_BICUBIC = 4,
  /** Trilinear interpolation for 3D inputs. */
  MAG_INTERP_MODE_TRILINEAR = 5,
  /** Area-based interpolation, typically used for downsampling. */
  MAG_INTERP_MODE_AREA = 6,
} mag_interp_mode_t;

/**
 * Resample the spatial dims of an input of layout (batch, channels, spatial...) to a new size.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input of rank 3, 4, or 5.
 * @param out_size Target size per spatial dim, @p out_len entries.
 * @param out_len Number of entries in @p out_size; must equal rank - 2.
 * @param scale_factor Optional scale per spatial dim used for coordinate mapping, or NULL.
 * @param mode One of mag_interp_mode_t::*.
 * @param align_corners If true, align corner samples; only valid for the linear, bilinear, bicubic, and trilinear modes.
 * @param antialias If true, apply antialiasing; only valid for the bilinear and bicubic modes.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_interpolate(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *out_size,
  int64_t out_len,
  const double *scale_factor,
  mag_interp_mode_t mode,
  bool align_corners,
  bool antialias
);

/**
 * Padding modes for mag_pad.
 */
typedef enum mag_pad_mode_t {
  /** Fill padded elements with a constant value. */
  MAG_PAD_MODE_CONSTANT = 0,
  /** Pad by reflecting values at the input boundaries. */
  MAG_PAD_MODE_REFLECT = 1,
  /** Pad by replicating the boundary values. */
  MAG_PAD_MODE_REPLICATE = 2,
  /** Pad by wrapping values around from the opposite boundary. */
  MAG_PAD_MODE_CIRCULAR = 3,
} mag_pad_mode_t;

/**
 * Pad the dims of @p x.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param x Input tensor.
 * @param pad Padding as (before, after) pairs starting with the last dim, @p pad_len entries; omitted dims get no padding.
 * @param pad_len Number of entries in @p pad, at most 2 * rank.
 * @param mode One of mag_pad_mode_t::*.
 * @param value Fill value for constant mode.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_pad(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *x,
  const int64_t *pad,
  int64_t pad_len,
  mag_pad_mode_t mode,
  mag_scalar_t value
);

/**
 * Zero the elements above the given diagonal of the last two dims.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensor Input tensor, rank >= 2.
 * @param diag Diagonal offset; 0 is the main diagonal, positive values are above it, negative values below.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tril(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag);

/**
 * Zero the elements above the given diagonal of the last two dims, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p tensor with an added reference.
 * @param tensor Tensor to modify, rank >= 2; must not require grad while recording.
 * @param diag Diagonal offset; 0 is the main diagonal, positive values are above it, negative values below.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tril_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag);

/**
 * Zero the elements below the given diagonal of the last two dims.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensor Input tensor, rank >= 2.
 * @param diag Diagonal offset; 0 is the main diagonal, positive values are above it, negative values below.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_triu(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag);

/**
 * Zero the elements below the given diagonal of the last two dims, in place.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives @p tensor with an added reference.
 * @param tensor Tensor to modify, rank >= 2; must not require grad while recording.
 * @param diag Diagonal offset; 0 is the main diagonal, positive values are above it, negative values below.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_triu_(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor, int64_t diag);

/**
 * Draw @p num_samples category indices per row from a tensor of probabilities. The result has dtype int64.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensor Contiguous 1-D or 2-D tensor of non-negative weights per category.
 * @param num_samples Samples per row, > 0.
 * @param replacement If true, sample with replacement.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_multinomial(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t *tensor,
  int64_t num_samples,
  bool replacement
);

/**
 * Concatenate tensors along an existing dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensors Array of @p count tensors with matching shapes except along @p dim.
 * @param count Number of tensors, > 0.
 * @param dim Axis; negative values count from the end.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_cat(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t **tensors,
  size_t count,
  int64_t dim
);

/**
 * Stack tensors along a new dim.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensors Array of @p count tensors of identical shape.
 * @param count Number of tensors, > 0.
 * @param dim Position of the new axis; negative values count from the end.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_stack(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t **tensors,
  size_t count,
  int64_t dim
);

/**
 * Concatenate tensors horizontally: along dim 1, or dim 0 for 1-D tensors.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensors Array of @p count tensors.
 * @param count Number of tensors, > 0.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_hstack(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t **tensors,
  size_t count
);

/**
 * Concatenate tensors vertically along dim 0. 1-D tensors are stacked as rows.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensors Array of @p count tensors.
 * @param count Number of tensors, > 0.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_vstack(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t **tensors,
  size_t count
);

/**
 * Concatenate tensors along dim 2. Tensors of lower rank are promoted to rank 3 first.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensors Array of @p count tensors.
 * @param count Number of tensors, > 0.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_dstack(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_tensor_t **tensors,
  size_t count
);

/**
 * Evaluate an Einstein summation over the given operands.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param equation Subscript specification, e.g. "ij,jk->ik".
 * @param args Operand tensors, @p num_args entries.
 * @param num_args Number of operands.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_einsum(
  mag_error_t *err,
  mag_tensor_t **out_result,
  const char *equation,
  mag_tensor_t **args,
  size_t num_args
);

/**
 * Create a view of @p tensor that is cut off from the autograd graph and does not require grad.
 *
 * @param err Error output; set when the call fails.
 * @param out_result Receives the result tensor; the caller owns one reference.
 * @param tensor Input tensor.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_detach(mag_error_t *err, mag_tensor_t **out_result, mag_tensor_t *tensor);

/* === Tensor Methods === */

/**
 * Get the number of dimensions.
 *
 * @param tensor Tensor to query.
 * @return Rank, in [0, MAG_MAX_DIMS].
 */
extern MAG_EXPORT int64_t mag_tensor_rank(const mag_tensor_t *tensor);

/**
 * Get the dimension sizes.
 *
 * @param tensor Tensor to query.
 * @return Pointer to rank entries; valid while the tensor is alive.
 */
extern MAG_EXPORT const int64_t *mag_tensor_shape_ptr(const mag_tensor_t *tensor);

/**
 * Get the element strides.
 *
 * @param tensor Tensor to query.
 * @return Pointer to rank entries; valid while the tensor is alive.
 */
extern MAG_EXPORT const int64_t *mag_tensor_strides_ptr(const mag_tensor_t *tensor);

/**
 * Get the element data type.
 *
 * @param tensor Tensor to query.
 * @return Data type.
 */
extern MAG_EXPORT mag_dtype_t mag_tensor_type(const mag_tensor_t *tensor);

/**
 * Get the offset of the first element from the storage base.
 *
 * @param tensor Tensor to query.
 * @return Offset in bytes.
 */
extern MAG_EXPORT size_t mag_tensor_data_offset(const mag_tensor_t *tensor);

/**
 * Get the address of the first element: the storage base plus the data offset.
 *
 * @param tensor Tensor to query.
 * @return Address as an integer; a device address for non-host tensors.
 */
extern MAG_EXPORT uintptr_t mag_tensor_data_ptr(const mag_tensor_t *tensor);

/**
 * Get the address of the first element for writing. Asserts that the storage is writable.
 *
 * @param tensor Tensor to query.
 * @return Address as an integer; a device address for non-host tensors.
 */
extern MAG_EXPORT uintptr_t mag_tensor_data_ptr_mut(const mag_tensor_t *tensor);

/**
 * Get the base address of the underlying storage buffer.
 *
 * @param tensor Tensor to query.
 * @return Address as an integer; a device address for non-host tensors.
 */
extern MAG_EXPORT uintptr_t mag_tensor_data_storage_ptr(const mag_tensor_t *tensor);

/**
 * Get the base address of the underlying storage buffer for writing. Asserts that the storage is writable.
 *
 * @param tensor Tensor to query.
 * @return Address as an integer; a device address for non-host tensors.
 */
extern MAG_EXPORT uintptr_t mag_tensor_data_storage_ptr_mut(const mag_tensor_t *tensor);

/**
 * Get the device the tensor resides on.
 *
 * @param tensor Tensor to query.
 * @return Device identifier.
 */
extern MAG_EXPORT mag_device_id_t mag_tensor_device_id(const mag_tensor_t *tensor);

/**
 * Get the size of the tensor elements: element count times element size.
 *
 * @param tensor Tensor to query.
 * @return Size in bytes.
 */
extern MAG_EXPORT size_t mag_tensor_numbytes(const mag_tensor_t *tensor);

/**
 * Get the size of the underlying storage buffer, which may exceed the tensor size for views.
 *
 * @param tensor Tensor to query.
 * @return Size in bytes.
 */
extern MAG_EXPORT size_t mag_tensor_storage_numbytes(const mag_tensor_t *tensor);

/**
 * Get the number of elements.
 *
 * @param tensor Tensor to query.
 * @return Element count.
 */
extern MAG_EXPORT int64_t mag_tensor_numel(const mag_tensor_t *tensor);

/**
 * Get the context the tensor belongs to.
 *
 * @param tensor Tensor to query.
 * @return Owning context.
 */
extern MAG_EXPORT mag_context_t *mag_tensor_context(const mag_tensor_t *tensor);

/**
 * Check whether the tensor is a view of another tensor's storage.
 *
 * @param tensor Tensor to query.
 * @return true if the tensor is a view.
 */
extern MAG_EXPORT bool mag_tensor_is_view(const mag_tensor_t *tensor);

/**
 * Get the tensor a view was created from.
 *
 * @param tensor Tensor to query.
 * @return Base tensor with an added reference, or NULL if @p tensor is not a view.
 */
extern MAG_EXPORT mag_tensor_t *mag_tensor_view_base(const mag_tensor_t *tensor);

/**
 * Check whether the element type is floating-point.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_floating_point_typed(const mag_tensor_t *tensor);

/**
 * Check whether the element type is an integer or boolean type.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_integral_typed(const mag_tensor_t *tensor);

/**
 * Check whether the element type is a signed or unsigned integer type. Boolean is excluded.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_integer_typed(const mag_tensor_t *tensor);

/**
 * Check whether the element type is an unsigned integer type.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_unsigned_integer_typed(const mag_tensor_t *tensor);

/**
 * Check whether the element type is a signed integer type.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_signed_integer_typed(const mag_tensor_t *tensor);

/**
 * Check whether the element type is floating-point or integer. Boolean is excluded.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_numeric_typed(const mag_tensor_t *tensor);

/**
 * Check whether two tensors have identical shapes.
 *
 * @param x First tensor.
 * @param y Second tensor.
 * @return true if rank and all dimension sizes match.
 */
extern MAG_EXPORT bool mag_tensor_is_shape_eq(const mag_tensor_t *x, const mag_tensor_t *y);

/**
 * Check whether two tensors have identical strides.
 *
 * @param x First tensor.
 * @param y Second tensor.
 * @return true if rank and all strides match.
 */
extern MAG_EXPORT bool mag_tensor_are_strides_eq(const mag_tensor_t *x, const mag_tensor_t *y);

/**
 * Check whether @p small can be broadcast to the shape of @p big.
 *
 * @param small Tensor to broadcast.
 * @param big Tensor providing the target shape.
 * @return true if broadcasting is possible.
 */
extern MAG_EXPORT bool mag_tensor_can_broadcast(const mag_tensor_t *small, const mag_tensor_t *big);

/**
 * Check whether the tensor's strides describe a transposed layout.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_transposed(const mag_tensor_t *tensor);

/**
 * Check whether the tensor's strides describe a permuted layout.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_permuted(const mag_tensor_t *tensor);

/**
 * Check whether the tensor is laid out contiguously in row-major order.
 *
 * @param tensor Tensor to query.
 * @return true if the property holds, false otherwise.
 */
extern MAG_EXPORT bool mag_tensor_is_contiguous(const mag_tensor_t *tensor);

/**
 * Check whether the tensor can be viewed with the given shape without copying.
 *
 * @param tensor Tensor to query.
 * @param dims Candidate shape, @p rank entries.
 * @param rank Number of entries in @p dims.
 * @return true if the existing strides can express the shape.
 */
extern MAG_EXPORT bool mag_tensor_can_view(const mag_tensor_t *tensor, const int64_t *dims, int64_t rank);

/**
 * Get the accumulated gradient of the tensor.
 *
 * @param tensor Tensor to query.
 * @return Gradient tensor with an added reference, or NULL if the tensor does not require grad or has no gradient yet.
 */
extern MAG_EXPORT mag_tensor_t *mag_tensor_grad(const mag_tensor_t *tensor);

/**
 * Assign @p grad as the gradient of @p tensor, or clear it. Enables gradient tracking on @p tensor if needed.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor to modify.
 * @param grad New gradient, or NULL to clear the current one.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tensor_set_grad(mag_error_t *err, mag_tensor_t *tensor, mag_tensor_t *grad);

/**
 * Check whether gradient tracking is enabled for the tensor.
 *
 * @param tensor Tensor to query.
 * @return true if the tensor requires grad.
 */
extern MAG_EXPORT bool mag_tensor_requires_grad(const mag_tensor_t *tensor);

/**
 * Enable or disable gradient tracking. Tracking requires a floating-point dtype.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor to modify.
 * @param requires_grad true to enable tracking, false to disable it.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tensor_set_requires_grad(mag_error_t *err, mag_tensor_t *tensor, bool requires_grad);

/**
 * Run backpropagation from a scalar tensor and populate the gradients of the tensors it was computed from.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Root of the backward pass; must be rank 0 and require grad.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tensor_backward(mag_error_t *err, mag_tensor_t *tensor);

/**
 * Zero the gradient of the tensor, if it has one.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor whose gradient is zeroed.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tensor_zero_grad(mag_error_t *err, mag_tensor_t *tensor);

/**
 * Copy the tensor contents into a newly allocated contiguous host buffer.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor to copy; may reside on any device.
 * @param out_buf Receives the buffer; release it with mag_tensor_copy_data_free().
 * @param out_size_bytes Receives the buffer size in bytes.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tensor_copy_data(
  mag_error_t *err,
  mag_tensor_t *tensor,
  void **out_buf,
  size_t *out_size_bytes
);

/**
 * Release a buffer returned by mag_tensor_copy_data().
 *
 * @param ret_val Buffer to free.
 */
extern MAG_EXPORT void mag_tensor_copy_data_free(void *ret_val);

/**
 * Read the single element of a one-element tensor as a scalar.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Tensor with exactly one element; may reside on any device.
 * @param out_value Receives the value.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tensor_item(
  mag_error_t *err,
  mag_tensor_t *tensor,
  mag_scalar_t *out_value
);

/**
 * Format the tensor contents and metadata as a string.
 *
 * @param tensor Tensor to format; may reside on any device.
 * @param head Elements shown at the start of each truncated dim; negative selects the default.
 * @param tail Elements shown at the end of each truncated dim; negative selects the default.
 * @param threshold Element count above which the output is truncated; negative selects the default.
 * @return NUL-terminated string to release with mag_tensor_to_string_free_data(), or NULL on failure.
 */
extern MAG_EXPORT const char *mag_tensor_to_string(
  mag_tensor_t *tensor,
  int64_t head,
  int64_t tail,
  int64_t threshold
);

/**
 * Release a string returned by mag_tensor_to_string().
 *
 * @param ret_val String to free.
 */
extern MAG_EXPORT void mag_tensor_to_string_free_data(const char *ret_val);

/**
 * Add a reference to the tensor.
 *
 * @param tensor Tensor to retain.
 */
extern MAG_EXPORT void mag_tensor_incref(mag_tensor_t *tensor);

/**
 * Drop a reference to the tensor and destroy it when the count reaches zero.
 *
 * @param tensor Tensor to release.
 * @return true if the tensor was destroyed by this call.
 */
extern MAG_EXPORT bool mag_tensor_decref(mag_tensor_t *tensor);

/**
 * Check whether the tensor resides on the CPU device.
 *
 * @param tensor Tensor to query.
 * @return true if the tensor is on CPU.
 */
extern MAG_EXPORT bool mag_tensor_is_cpu(mag_tensor_t *tensor);

/**
 * Write the autograd graph reachable from @p tensor to a file in Graphviz DOT format.
 *
 * @param err Error output; set when the call fails.
 * @param tensor Root tensor of the graph.
 * @param file Destination path.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_tensor_visualize_backprop_graph(mag_error_t *err, mag_tensor_t *tensor, const char *file);

/* === Snapshot De/Serialization === */

/**
 * Opaque streaming writer for snapshot (.mag) files.
 */
typedef struct mag_snapshot_stream_writer_t mag_snapshot_stream_writer_t;

/**
 * Create a snapshot file and write its header and metadata. Output goes to a temporary file that is renamed into place on close.
 *
 * @param err Error output; set when the call fails.
 * @param writer Receives the writer.
 * @param ctx Owning context.
 * @param filepath Destination path; must end in .mag.
 * @param meta_document UTF-8 metadata document without NUL bytes; may be NULL if @p meta_len is 0.
 * @param meta_len Length of @p meta_document in bytes.
 * @param blob_len Declared size of the data section in bytes, including alignment padding between blobs; > 0.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_snapshot_stream_writer_open(
  mag_error_t *err,
  mag_snapshot_stream_writer_t **writer,
  mag_context_t *ctx,
  const char *filepath,
  const char *meta_document,
  uint64_t meta_len,
  uint64_t blob_len
);

/**
 * Append a blob to the data section. Each blob starts at the tensor blob alignment; the padding counts toward the declared data section size.
 *
 * @param err Error output; set when the call fails.
 * @param writer Open writer.
 * @param blob Bytes to append; may be NULL if @p size is 0.
 * @param size Number of bytes to append.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_snapshot_stream_writer_submit_blob(
  mag_error_t *err,
  mag_snapshot_stream_writer_t *writer,
  const void *blob,
  uint64_t size
);

/**
 * Finish the snapshot: verify the data section is complete, flush it, and rename the file into place. The writer is released whether or not the call succeeds.
 *
 * @param err Error output; set when the call fails.
 * @param writer Writer to close.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_snapshot_stream_writer_close(mag_error_t *err, mag_snapshot_stream_writer_t *writer);

/**
 * Discard an unfinished snapshot, remove the temporary file, and release the writer. NULL is ignored.
 *
 * @param writer Writer to abort.
 */
extern MAG_EXPORT void mag_snapshot_stream_writer_abort(mag_snapshot_stream_writer_t *writer);

/**
 * Opaque memory-mapped reader for snapshot (.mag) files.
 */
typedef struct mag_snapshot_stream_reader_t mag_snapshot_stream_reader_t;

/**
 * Open a snapshot file, validate its header, and map it into memory.
 *
 * @param err Error output; set when the call fails.
 * @param reader Receives the reader.
 * @param ctx Owning context.
 * @param filepath Path of the snapshot file.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_snapshot_stream_reader_open(
  mag_error_t *err,
  mag_snapshot_stream_reader_t **reader,
  mag_context_t *ctx,
  const char *filepath
);

/**
 * Get the metadata document. The returned bytes are not NUL-terminated.
 *
 * @param reader Open reader.
 * @param out_len Receives the metadata length in bytes.
 * @return Pointer into the mapping; valid while the reader is open.
 */
extern MAG_EXPORT const char *mag_snapshot_stream_reader_meta(const mag_snapshot_stream_reader_t *reader, uint64_t *out_len); /* Warning! NOT NUL terminated!! */

/**
 * Get the size of the data section.
 *
 * @param reader Open reader.
 * @return Size in bytes.
 */
extern MAG_EXPORT uint64_t mag_snapshot_stream_reader_blob_len(const mag_snapshot_stream_reader_t *reader);

/**
 * Get the format version recorded in the file header.
 *
 * @param reader Open reader.
 * @return Encoded version; decode with the mag_ver_* macros.
 */
extern MAG_EXPORT uint32_t mag_snapshot_stream_reader_version(const mag_snapshot_stream_reader_t *reader);

/**
 * Create a read-only tensor that references a region of the mapped data section without copying. The mapping stays alive until the tensor is released.
 *
 * @param err Error output; set when the call fails.
 * @param out Receives the result tensor; the caller owns one reference.
 * @param reader Open reader.
 * @param offset Byte offset within the data section; must be a multiple of the tensor blob alignment.
 * @param size Region size in bytes, > 0.
 * @param dtype Element data type.
 * @param rank Number of dimensions in @p shape.
 * @param shape Dimension sizes, @p rank entries.
 * @return MAG_OK on success, an error status otherwise.
 */
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

/**
 * Release the reader. The mapping is unmapped once all borrowed tensors are released. NULL is ignored.
 *
 * @param reader Reader to close.
 */
extern MAG_EXPORT void mag_snapshot_stream_reader_close(mag_snapshot_stream_reader_t *reader);


/* === Distributed === */

/**
 * Element-wise reduction applied by the reducing collectives. MAG_REDUCE_AVG divides the sum by the communicator size; the bitwise ops require integral tensors.
 */
typedef enum mag_reduce_op_t {
  MAG_REDUCE_SUM,
  MAG_REDUCE_AVG,
  MAG_REDUCE_PROD,
  MAG_REDUCE_MIN,
  MAG_REDUCE_MAX,
  MAG_REDUCE_AND,
  MAG_REDUCE_OR,
  MAG_REDUCE_XOR,
} mag_reduce_op_t;

/**
 * Backend-specific communicator configuration passed to mag_comm_init. Reserved; pass NULL for defaults.
 */
typedef struct mag_comm_backend_desc_t {
  int dummy;
} mag_comm_backend_desc_t;

/**
 * Opaque handle to a group of ranks that exchange tensors through a communication backend. Every collective must be called by all ranks of the communicator.
 */
typedef struct mag_communicator_t mag_communicator_t;

/**
 * Create a communicator for the calling rank and connect it to the given backend.
 *
 * @param err Error output; set when the call fails.
 * @param out_comm Receives the communicator; release it with mag_comm_destroy.
 * @param rank Rank of the caller within the communicator, in [0, size).
 * @param size Number of ranks in the communicator; must be > 0.
 * @param backend Name of the communication backend to use.
 * @param backend_desc Optional backend configuration; NULL selects the defaults.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_init(
  mag_error_t *err,
  mag_context_t *ctx,
  mag_communicator_t **out_comm,
  uint32_t rank,
  uint32_t size,
  const char *backend,
  const mag_comm_backend_desc_t *backend_desc
);

/**
 * Tear down the communicator and release its backend resources. NULL is ignored.
 *
 * @param comm Communicator to destroy.
 */
extern MAG_EXPORT void mag_comm_destroy(mag_communicator_t *comm);


/**
 * Query the rank of the caller within the communicator.
 *
 * @param comm Communicator to query.
 * @return Rank in [0, size).
 */
extern MAG_EXPORT uint32_t mag_comm_rank(mag_communicator_t *comm);


/**
 * Query the number of ranks in the communicator.
 *
 * @param comm Communicator to query.
 * @return Number of participating ranks.
 */
extern MAG_EXPORT uint32_t mag_comm_size(mag_communicator_t *comm);


/**
 * Query the name of the communication backend behind the communicator.
 *
 * @param comm Communicator to query.
 * @return Static backend name string; owned by the library.
 */
extern MAG_EXPORT const char *mag_comm_backend_name(mag_communicator_t *comm);


/**
 * Block until every rank of the communicator has entered the barrier.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to synchronize.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_barrier(
  mag_error_t *err,
  mag_communicator_t *comm
);

/**
 * Copy @p tensor from the root rank into the same-shaped @p tensor of every other rank.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param tensor Source on the root rank, destination on all other ranks; updated in place.
 * @param root Rank whose data is broadcast, in [0, size).
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_broadcast(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  uint32_t root
);

/**
 * Reduce @p tensor across all ranks with @p red and store the result in @p tensor on the root rank only.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param tensor Contribution of the caller; overwritten with the result on the root rank.
 * @param root Rank that receives the reduced result, in [0, size).
 * @param red Reduction to apply element-wise.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_reduce(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  uint32_t root,
  mag_reduce_op_t red
);

/**
 * Reduce @p tensor across all ranks with @p red and store the result in @p tensor on every rank.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param tensor Contribution of the caller; overwritten with the reduced result in place.
 * @param red Reduction to apply element-wise.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_all_reduce(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  mag_reduce_op_t red
);

/**
 * Concatenate the @p in tensor of every rank, ordered by rank, into @p out on every rank.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param out Receives size copies of the input shape stacked along the first dim.
 * @param in Contribution of the caller; the same shape on every rank.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_all_gather(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *out,
  mag_tensor_t *in
);

/**
 * Reduce @p in across all ranks with @p red, then scatter the result so that rank i receives the i-th equal chunk in @p out.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param out Receives the caller's chunk of the reduced result; holds numel(in)/size elements.
 * @param in Contribution of the caller; the same shape on every rank and divisible into size chunks.
 * @param red Reduction to apply element-wise.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_reduce_scatter(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *out,
  mag_tensor_t *in,
  mag_reduce_op_t red
);

/**
 * Split @p in into size equal chunks and send chunk j to rank j; @p out receives the chunk addressed to the caller from every rank, ordered by source rank.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param out Receives size chunks; the same shape as @p in.
 * @param in Data to distribute; the same shape on every rank and divisible into size chunks.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_all_to_all(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *out,
  mag_tensor_t *in
);

/**
 * Ragged all-to-all: send a per-peer slice of @p in to every rank and receive a per-peer slice from every rank into @p out. Counts and offsets are in elements and have size entries indexed by peer rank.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param out Receives the slices addressed to the caller; must hold recv_offsets[j] + recv_counts[j] elements for every j.
 * @param in Data to distribute; must hold send_offsets[j] + send_counts[j] elements for every j.
 * @param send_counts Number of elements sent to rank j.
 * @param send_offsets Element offset into @p in of the slice sent to rank j.
 * @param recv_counts Number of elements received from rank j; must equal that rank's send_counts entry for the caller.
 * @param recv_offsets Element offset into @p out of the slice received from rank j.
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_all_to_all_v(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *out,
  mag_tensor_t *in,
  const size_t *send_counts,
  const size_t *send_offsets,
  const size_t *recv_counts,
  const size_t *recv_offsets
);

/**
 * Send @p tensor to a single peer. Blocks until the peer has posted a matching mag_comm_recv with the same shape and dtype.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param tensor Data to send.
 * @param dst Destination rank, in [0, size).
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_send(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  uint32_t dst
);

/**
 * Receive into @p tensor from a single peer. Blocks until the peer's matching mag_comm_send with the same shape and dtype completes.
 *
 * @param err Error output; set when the call fails.
 * @param comm Communicator to use.
 * @param tensor Preallocated destination; overwritten in place.
 * @param src Source rank, in [0, size).
 * @return MAG_OK on success, an error status otherwise.
 */
extern MAG_EXPORT mag_status_t mag_comm_recv(
  mag_error_t *err,
  mag_communicator_t *comm,
  mag_tensor_t *tensor,
  uint32_t src
);

#ifdef __cplusplus
}
#endif
#endif
