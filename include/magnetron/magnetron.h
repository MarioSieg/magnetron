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

#ifndef MAGNETRON_H
#define MAGNETRON_H

#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_MAX_DIMS 6                      /* Maximum number of dimensions for a tensor */
#define MAG_MAX_OP_INPUTS 2                 /* Maximum number of input tensors for an operation */
#define MAG_MAX_OP_PARAMS 8                 /* Maximum number of parameters for an operation */

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

#define mag_version_pack(maj, mi) ((uint32_t)((((maj)&255)<<8)+((mi)&255)))
#define mag_version_major(ver) (((ver)>>8)&255)
#define mag_version_minor(ver) ((ver)&255)
#define MAG_VERSION mag_version_pack(0, 1) /* magnetron library version. */
#define MAG_STORAGE_VERSION 1 /* magnetron storage format version. */

/**
 * @brief Compute device types. Determines the type of hardware used for computation.
 */
typedef enum mag_device_type_t {
    MAG_DEVICE_TYPE_CPU = 0,        /* CPU compute device */
    MAG_DEVICE_TYPE_GPU_CUDA = 1,   /* CUDA GPU compute device */
    MAG_DEVICE_TYPE__NUM
} mag_device_type_t;
extern MAG_EXPORT const char* mag_device_type_get_name(mag_device_type_t op);

/**
 * @brief Pseudo-random number generator (PRNG) algorithms used for number generation and random tensor initialization.
 *      The default PRNG is Mersenne Twister, switching to PCG can yield better performance in some cases.
 */
typedef enum mag_prng_algo_t {
    MAG_PRNG_MERSENNE_TWISTER = 0,  /* Mersenne Twister PRNG */
    MAG_PRNG_PCG = 1,               /* Permuted Congruential Generator PRNG */

    MAG_PRNG__NUM
} mag_prng_algo_t;

/**
 * @brief Thread scheduling priority for CPU compute.
 *      This set the OS scheduling priority of the thread that executes the compute operations.
 *      The priority is only used if the compute device is a CPU.
 */
typedef enum mag_thread_prio_t {       /* Thread scheduling priority for CPU compute */
    MAG_THREAD_PRIO_NORMAL = 0,     /* Normal thread priority */
    MAG_THREAD_PRIO_MEDIUM = 1,     /* Medium thread priority */
    MAG_THREAD_PRIO_HIGH = 2,       /* High thread priority */
    MAG_THREAD_PRIO_REALTIME = 3,   /* Real-time thread priority */
} mag_thread_prio_t;

/**
 * @brief Desired Color channels for loading image tensors.
 */
typedef enum mag_color_channels_t {
    MAG_COLOR_CHANNELS_AUTO,        /* Automatically detect number of color channels */
    MAG_COLOR_CHANNELS_GRAY,        /* Grayscale F32 */
    MAG_COLOR_CHANNELS_GRAY_A,      /* Grayscale F32 + Alpha F32 */
    MAG_COLOR_CHANNELS_RGB,         /* R32G32B32 */
    MAG_COLOR_CHANNELS_RGBA,        /* R32G32B32A32 */

    MAG_COLOR_CHANNELS__NUM
} mag_color_channels_t;

extern MAG_EXPORT void* (*mag_get_alloc_fn(void))(void*, size_t, size_t); /* Get global allocator. */
extern MAG_EXPORT void mag_set_alloc_fn(void* (*alloc)(void*, size_t, size_t)); /* Set global allocator. */
extern MAG_EXPORT void mag_set_log_mode(bool enabled); /* Enable/disable logging. */
extern MAG_EXPORT uint32_t mag_pack_color_u8(uint8_t r, uint8_t g, uint8_t b);
extern MAG_EXPORT uint32_t mag_pack_color_f32(float r, float g, float b);

/**
* @brief Context for the magnetron library.
*      The context is used to create and manage tensors, operations, and other resources.
*      It is also used to configure the compute device and PRNG algorithm.
*      The context must be kept alive as long as any tensor is used.
*      The context itself is not thread-safe, use a thread-local context or synchronize access. (Multiple contexts can also be used.)
*/
typedef struct mag_context_t mag_context_t;

/**
 * @brief Compute device descriptor.
 *      Used to specify the compute device type and configuration when creating a context.
 *      The device type must be one of the mag_device_type_t values.
 *      If the type is MAG_DEVICE_TYPE_CPU, thread_count can be set to 0 to detect hardware concurrency.
 *      If the type is MAG_DEVICE_TYPE_GPU_CUDA, cuda_device_id can be set to select a specific GPU.
 */
typedef struct mag_device_desc_t {
    mag_device_type_t type;  /* Device type */
    uint32_t cpu_thread_count;   /* Number of threads if type == MAG_DEVICE_TYPE_CPU. If set to 0, hardware concurrency of host CPU is detected. */
    uint32_t cuda_device_id;     /* CUDA device ID if type == MAG_DEVICE_TYPE_GPU_CUDA. Default: 0 (first GPU). */
} mag_device_desc_t;
extern MAG_EXPORT mag_device_desc_t mag_compute_device_desc_cpu(uint32_t thread_count);                                 /* Helper to fill device descriptor for CPU compute device. */
extern MAG_EXPORT mag_device_desc_t mag_compute_device_desc_cuda(uint32_t cuda_device_id);                              /* Helper to fill device descriptor for CUDA GPU compute device. */

extern MAG_EXPORT mag_context_t* mag_ctx_create(mag_device_type_t device);                                              /* Create context with default config, and only specify device type. */
extern MAG_EXPORT mag_context_t* mag_ctx_create2(const mag_device_desc_t* device_info);                                 /* Create context with customized device config, and only specify device type. */
extern MAG_EXPORT mag_prng_algo_t mag_ctx_get_prng_algorithm(const mag_context_t* ctx);                                 /* Get PRNG algorithm */
extern MAG_EXPORT void mag_ctx_set_prng_algorithm(mag_context_t* ctx, mag_prng_algo_t algorithm, uint64_t seed);        /* Set PRNG algorithm */
extern MAG_EXPORT mag_device_type_t mag_ctx_get_compute_device_type(const mag_context_t* ctx);                          /* Get compute device type */
extern MAG_EXPORT const char* mag_ctx_get_compute_device_name(const mag_context_t* ctx);                                /* Get the name of the compute device */
extern MAG_EXPORT const char* mag_ctx_get_os_name(const mag_context_t* ctx);                                            /* Get the name of the operating system */
extern MAG_EXPORT const char* mag_ctx_get_cpu_name(const mag_context_t* ctx);                                           /* Get the name of the CPU */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_virtual_cores(const mag_context_t* ctx);                                     /* Get the number of virtual cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_physical_cores(const mag_context_t* ctx);                                    /* Get the number of physical cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_sockets(const mag_context_t* ctx);                                           /* Get the number of CPU sockets */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_total(const mag_context_t* ctx);                                 /* Get the total physical memory in bytes */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_free(const mag_context_t* ctx);                                  /* Get the free physical memory in bytes */
extern MAG_EXPORT bool mag_ctx_is_numa_system(const mag_context_t* ctx);                                                /* Check if the system is NUMA */
extern MAG_EXPORT size_t mag_ctx_get_total_tensors_created(const mag_context_t* ctx);                                   /* Get total tensors created. (Including views) */
extern MAG_EXPORT void mag_ctx_grad_recorder_start(mag_context_t* ctx);                                                 /* Start gradient recording */
extern MAG_EXPORT void mag_ctx_grad_recorder_stop(mag_context_t* ctx);                                                  /* Stop gradient recording */
extern MAG_EXPORT bool mag_ctx_grad_recorder_is_running(const mag_context_t* ctx);                                      /* Check if gradient recording is running */
extern MAG_EXPORT void mag_ctx_destroy(mag_context_t* ctx, bool suppress_leak_detection);                               /* Destroy context and free memory */

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
 * @brief Data types for tensors.
 */
typedef enum mag_dtype_t {
    MAG_DTYPE_E8M23,        /* IEEE-754 32-bit floating point number. Commonly known as float32, f32. */
    MAG_DTYPE_E5M10,        /* IEEE-754 16-bit floating point number. Commonly known as float16, f16, half. */
    MAG_DTYPE_BOOL,         /* 1-byte boolean */
    MAG_DTYPE_I32,          /* 32-bit signed integer */

    MAG_DTYPE__NUM
} mag_dtype_t;

mag_static_assert(MAG_DTYPE__NUM <= 0xff); /* Must fix in 1 byte */

/**
 * @brief Stores various information about a data type such as size, name, etc.
 */
typedef struct mag_dtype_meta_t {
    const char* name;   /* Name of the data type */
    size_t size;                 /* Size of the data type in bytes. Must be a power of two. */
    size_t align;                /* CPU Alignment of the data type in bytes. Must be a power of two. */
} mag_dtype_meta_t;
extern MAG_EXPORT const mag_dtype_meta_t* mag_dtype_meta_of(mag_dtype_t type);

extern MAG_EXPORT mag_tensor_t* mag_tensor_new(mag_context_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape);
extern MAG_EXPORT mag_tensor_t* mag_tensor_as_strided(mag_context_t* ctx, mag_tensor_t* base, int64_t rank, const int64_t* shape, const int64_t* strides, int64_t offset);
extern MAG_EXPORT mag_tensor_t* mag_tensor_empty(mag_context_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape);
extern MAG_EXPORT mag_tensor_t* mag_tensor_empty_like(mag_tensor_t* isomorph);
extern MAG_EXPORT mag_tensor_t* mag_tensor_empty_scalar(mag_context_t* ctx, mag_dtype_t type);
extern MAG_EXPORT mag_tensor_t* mag_tensor_scalar(mag_context_t* ctx, mag_dtype_t type, float value);
extern MAG_EXPORT mag_tensor_t* mag_tensor_full(mag_context_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape, float value);
extern MAG_EXPORT mag_tensor_t* mag_tensor_full_like(mag_tensor_t* isomorph, float value);

/* ============ Tensor Operators ============ */

extern MAG_EXPORT mag_tensor_t* mag_clone(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_view(mag_tensor_t* x, const int64_t* dims, int64_t rank);
extern MAG_EXPORT mag_tensor_t* mag_view_slice(mag_tensor_t* x, int64_t dim, int64_t start, int64_t len, int64_t step);
extern MAG_EXPORT mag_tensor_t* mag_reshape(mag_tensor_t* x, const int64_t* dims, int64_t rank);
extern MAG_EXPORT mag_tensor_t* mag_transpose(mag_tensor_t* x, int64_t dim1, int64_t dim2);
extern MAG_EXPORT mag_tensor_t* mag_permute(mag_tensor_t* x, const int64_t* dims, int64_t rank);
extern MAG_EXPORT mag_tensor_t* mag_contiguous(mag_tensor_t* x);

extern MAG_EXPORT mag_tensor_t* mag_mean(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_tensor_t* mag_min(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_tensor_t* mag_max(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_tensor_t* mag_sum(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_tensor_t* mag_argmin(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim);
extern MAG_EXPORT mag_tensor_t* mag_argmax(mag_tensor_t* x, const int64_t* dims, int64_t rank, bool keepdim);

extern MAG_EXPORT mag_tensor_t* mag_abs(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_abs_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sgn(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sgn_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_neg(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_neg_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_log(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_log_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sqr(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sqr_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sqrt(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sqrt_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sin(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sin_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_cos(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_cos_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_step(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_step_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_exp(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_exp_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_floor(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_floor_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_ceil(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_ceil_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_round(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_round_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_softmax(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_softmax_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_softmax_dv(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_softmax_dv_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sigmoid(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sigmoid_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sigmoid_dv(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_sigmoid_dv_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_hard_sigmoid(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_hard_sigmoid_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_silu(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_silu_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_silu_dv(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_silu_dv_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_tanh(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_tanh_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_tanh_dv(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_tanh_dv_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_relu(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_relu_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_relu_dv(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_relu_dv_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_gelu(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_gelu_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_gelu_approx(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_gelu_approx_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_gelu_dv(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_gelu_dv_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_add(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_add_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_sub( mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_sub_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_mul( mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_mul_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_div(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_div_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_matmul(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_repeat_back(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_gather(mag_tensor_t* x, int64_t dim, mag_tensor_t* idx);
extern MAG_EXPORT mag_tensor_t* mag_and(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_and_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_or(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_or_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_xor(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_xor_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_not(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_not_(mag_tensor_t* x);
extern MAG_EXPORT mag_tensor_t* mag_shl(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_shl_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_shr(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_shr_(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_eq(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_ne(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_le(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_ge(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_lt(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_gt(mag_tensor_t* x, mag_tensor_t* y);
extern MAG_EXPORT mag_tensor_t* mag_tril(mag_tensor_t* x, int32_t diag);
extern MAG_EXPORT mag_tensor_t* mag_tril_(mag_tensor_t* x, int32_t diag);
extern MAG_EXPORT mag_tensor_t* mag_triu(mag_tensor_t* x, int32_t diag);
extern MAG_EXPORT mag_tensor_t* mag_triu_(mag_tensor_t* x, int32_t diag);
extern MAG_EXPORT mag_tensor_t* mag_multinomial(mag_tensor_t* x, int64_t num_samples, bool replacement);

/* ============ Tensor Init Operators ============ */

extern MAG_EXPORT void mag_tensor_fill_from_floats(mag_tensor_t* t, const float* data, size_t len);       /* Copy floats into tensor buffer. If the tensors datatype is not float, the values are converted to the tensors dtype. */
extern MAG_EXPORT void mag_tensor_fill_from_raw_bytes(mag_tensor_t* t, const void* data, size_t len);     /* Copy raw bytes into tensor buffer */
extern MAG_EXPORT void mag_tensor_fill_float(mag_tensor_t* t, float x);                                            /* Set all tensor elements to a specific value. */
extern MAG_EXPORT void mag_tensor_fill_int(mag_tensor_t* t, int32_t x);                                            /* Set all tensor elements to a specific value. */
extern MAG_EXPORT void mag_tensor_masked_fill_float(mag_tensor_t* t, mag_tensor_t* mask, float x);          /* Set all tensor elements to a specific value if the mask value at the same index is true. */
extern MAG_EXPORT void mag_tensor_masked_fill_int(mag_tensor_t* t, mag_tensor_t* mask, int32_t x);          /* Set all tensor elements to a specific value if the mask value at the same index is true. */
extern MAG_EXPORT void mag_tensor_fill_random_uniform_float(mag_tensor_t* t, float min, float max);                /* Fill tensor with random values from uniform distribution within [min, max] */
extern MAG_EXPORT void mag_tensor_fill_random_uniform_int(mag_tensor_t* t, int32_t min, int32_t max);              /* Fill tensor with random values from uniform distribution within [min, max] */
extern MAG_EXPORT void mag_tensor_fill_random_normal(mag_tensor_t* t, float mean, float stddev);                   /* Fill tensor with random values from the normal distribution. */
extern MAG_EXPORT void mag_tensor_fill_random_bernoulli(mag_tensor_t* t, float p);                                 /* Fill bool tensor with random values from the bernoulli distribution. */
extern MAG_EXPORT void mag_tensor_fill_arange(mag_tensor_t* t, float start, float step);                            /* Fill tensor with values from start to end with step. */

/* ============ Tensor Property Accessors ============ */

extern MAG_EXPORT uint64_t mag_tensor_get_refcount(const mag_tensor_t* t);                                         /* Return reference count of tensor itself. */
extern MAG_EXPORT uint64_t mag_tensor_get_storage_refcount(const mag_tensor_t* t);                                 /* Return reference count of tensor itself. */
extern MAG_EXPORT size_t mag_tensor_get_memory_usage(const mag_tensor_t* t);                                       /* Return memory used by this tensor in bytes. */
extern MAG_EXPORT int64_t mag_tensor_get_rank(const mag_tensor_t* t);                                              /* Get the rank (number of dimensions) of the tensor */
extern MAG_EXPORT const int64_t* mag_tensor_get_shape(const mag_tensor_t* t);                             /* Get the dimensions of the tensor */
extern MAG_EXPORT const int64_t* mag_tensor_get_strides(const mag_tensor_t* t);                           /* Get the strides of the tensor */
extern MAG_EXPORT mag_dtype_t mag_tensor_get_dtype(const mag_tensor_t* t);                                           /* Get the data type of the tensor */
extern MAG_EXPORT size_t mag_tensor_get_data_offset(const mag_tensor_t* t);
extern MAG_EXPORT void* mag_tensor_get_data_ptr(const mag_tensor_t* t);                                   /* Get pointer to first element. Might point to GPU or any other device memory. */
extern MAG_EXPORT void* mag_tensor_get_storage_base_ptr(const mag_tensor_t* t);                           /* Get pointer to storage memory block base. Might point to GPU or any other device memory. */
extern MAG_EXPORT int64_t mag_tensor_get_data_size(const mag_tensor_t* );                                          /* Get the size of the tensor buffer in bytes. */
extern MAG_EXPORT int64_t mag_tensor_get_numel(const mag_tensor_t* t);                                             /* Get the total amount of elements in the tensor. */
extern MAG_EXPORT mag_context_t* mag_tensor_get_ctx(const mag_tensor_t* t);                                 /* Get the context of the tensor */
extern MAG_EXPORT int64_t mag_tensor_get_width(const mag_tensor_t* t);
extern MAG_EXPORT int64_t mag_tensor_get_height(const mag_tensor_t* t);
extern MAG_EXPORT int64_t mag_tensor_get_channels(const mag_tensor_t* t);
extern MAG_EXPORT bool mag_tensor_is_view(const mag_tensor_t* t);                                                /* Check if the tensor is a view of another tensor */
extern MAG_EXPORT bool mag_tensor_is_floating_point_typed(const mag_tensor_t* t);
extern MAG_EXPORT bool mag_tensor_is_integral_typed(const mag_tensor_t* t);
extern MAG_EXPORT bool mag_tensor_is_integer_typed(const mag_tensor_t* t);
extern MAG_EXPORT bool mag_tensor_is_numeric_typed(const mag_tensor_t* t);

/* ============ Tensor Shape Utils ============ */

extern MAG_EXPORT bool mag_tensor_is_shape_eq(const mag_tensor_t* x, const mag_tensor_t* y);              /* Checks if a and b have the same shape. */
extern MAG_EXPORT bool mag_tensor_are_strides_eq(const mag_tensor_t* x, const mag_tensor_t* y);           /* Checks if a and b have the same strides. */
extern MAG_EXPORT bool mag_tensor_can_broadcast(const mag_tensor_t* small, const mag_tensor_t* big);      /* Checks if b can be broadcasted into a. */
extern MAG_EXPORT bool mag_tensor_is_transposed(const mag_tensor_t* t);                                          /* Check if the tensor is transposed */
extern MAG_EXPORT bool mag_tensor_is_permuted(const mag_tensor_t* t);                                            /* Check if the tensor is permuted */
extern MAG_EXPORT bool mag_tensor_is_contiguous(const mag_tensor_t* t);                                          /* Check if the tensor memory is contiguous */
extern MAG_EXPORT bool mag_tensor_can_view(const mag_tensor_t* t, const int64_t* dims, int64_t rank); /* Check if the tensor can be viewed with the given dimensions */

/* ============ Gradient & Backprop API ============ */

extern MAG_EXPORT mag_tensor_t* mag_tensor_get_grad(const mag_tensor_t* t);                          /* Get the gradient tensor of the tensor */
extern MAG_EXPORT bool mag_tensor_requires_grad(const mag_tensor_t* t);                                        /* Check if the tensor requires gradient computation */
extern MAG_EXPORT void mag_tensor_set_requires_grad(mag_tensor_t* t, bool requires_grad);                      /* Set if the tensor requires gradient computation */
extern MAG_EXPORT void mag_tensor_backward(mag_tensor_t* t);                                                   /* Compute the gradient of the tensor */
extern MAG_EXPORT void mag_tensor_zero_grad(mag_tensor_t* t);                                                  /* Zero the gradient of the tensor */

/* ============ Tensor Data Access API ============ */

extern MAG_EXPORT float mag_tensor_subscript_get_multi(mag_tensor_t* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5);
extern MAG_EXPORT void mag_tensor_subscript_set_multi(mag_tensor_t* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5, float val);
extern MAG_EXPORT float mag_tensor_subscript_get_flattened(mag_tensor_t* t, int64_t idx);
extern MAG_EXPORT void mag_tensor_subscript_set_flattened(mag_tensor_t* t, int64_t idx, float val);
extern MAG_EXPORT void* mag_tensor_get_raw_data_as_bytes(mag_tensor_t* t);
extern MAG_EXPORT void mag_tensor_get_raw_data_as_bytes_free(void* ret_val);
extern MAG_EXPORT float* mag_tensor_get_data_as_floats(mag_tensor_t* t);
extern MAG_EXPORT void mag_tensor_get_data_as_floats_free(float* ret_val);
extern MAG_EXPORT float mag_tensor_get_item_float(const mag_tensor_t* t);
extern MAG_EXPORT int32_t mag_tensor_get_item_int(const mag_tensor_t* t);
extern MAG_EXPORT bool mag_tensor_get_item_bool(const mag_tensor_t* t);

/* ============ Tensor Misc API ============ */

extern MAG_EXPORT void mag_tensor_incref(mag_tensor_t* t);
extern MAG_EXPORT bool mag_tensor_decref(mag_tensor_t* t);
extern MAG_EXPORT mag_tensor_t* mag_tensor_detach(mag_tensor_t* t);
extern MAG_EXPORT char* mag_tensor_to_string(mag_tensor_t* t, bool with_header, size_t from_start_count, size_t from_end_count);
extern MAG_EXPORT void mag_tensor_to_string_free_data(char* ret_val);
extern MAG_EXPORT mag_tensor_t* mag_tensor_load_image(mag_context_t* ctx, const char* file, mag_color_channels_t channels, uint32_t resize_w, uint32_t resize_h);    /* Create a tensor from an image file. */
extern MAG_EXPORT void mag_tensor_save_image(const mag_tensor_t* t, const char* file);                                                                                    /* Save tensor data as an image */
extern MAG_EXPORT void mag_tensor_export_forward_graph_graphviz(mag_tensor_t* t, const char* file);                                                                      /* Export tensor computation graph as Graphviz DOT file *//* Get image channels from tensor */
extern MAG_EXPORT void mag_tensor_export_backward_graph_graphviz(mag_tensor_t* t, const char* file);


/* ============ Storage Archive API ============ */

typedef struct mag_storage_archive_t mag_storage_archive_t;

typedef enum mag_record_type_t {
    MAG_RECORD_TYPE_TENSOR,
    MAG_RECORD_TYPE_I64,
    MAG_RECORD_TYPE_F64,

    MAG_RECORD_TYPE__COUNT
} mag_record_type_t;

extern MAG_EXPORT mag_storage_archive_t* mag_storage_open(mag_context_t* ctx, const char* filename, char mode);
extern MAG_EXPORT bool mag_storage_close(mag_storage_archive_t* archive);

extern MAG_EXPORT const char** mag_storage_get_all_tensor_keys(mag_storage_archive_t* archive, size_t* out_len);
extern MAG_EXPORT const char** mag_storage_get_all_metadata_keys(mag_storage_archive_t* archive, size_t* out_len);
extern MAG_EXPORT void mag_storage_get_all_keys_free(const char** keys, size_t len);

extern MAG_EXPORT bool mag_storage_set_tensor(mag_storage_archive_t* archive, const char* key, mag_tensor_t* tensor);
extern MAG_EXPORT mag_tensor_t* mag_storage_get_tensor(mag_storage_archive_t* archive, const char* key);

extern MAG_EXPORT mag_record_type_t mag_storage_get_metadata_type(mag_storage_archive_t* archive, const char* key);
extern MAG_EXPORT bool mag_storage_set_metadata_i64(mag_storage_archive_t* archive, const char* key, int64_t value);
extern MAG_EXPORT bool mag_storage_get_metadata_i64(mag_storage_archive_t* archive, const char* key, int64_t* value);
extern MAG_EXPORT bool mag_storage_set_metadata_f64(mag_storage_archive_t* archive, const char* key, double value);
extern MAG_EXPORT bool mag_storage_get_metadata_f64(mag_storage_archive_t* archive, const char* key, double* value);

#ifdef __cplusplus
}
#endif
#endif
