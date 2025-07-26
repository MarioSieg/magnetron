/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
**
** Magnetron Machine Learning Framework, public C99 API. For a modern C++ API, see magnetron.hpp.
** This header is also used from Python and by the modern C++ API.
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
#define MAG_MAX_TENSOR_NAME_LEN 64          /* Maximum length for tensor name */
#define MAG_MAX_OP_INPUTS 2                 /* Maximum number of input tensors for an operation */
#define MAG_MAX_OP_PARAMS 8                 /* Maximum number of parameters for an operation */

#ifndef MAG_EXPORT
#ifdef _MSC_VER
#define MAG_EXPORT __declspec(dllexport)
#else
#define MAG_EXPORT __attribute__((visibility("default")))
#endif
#endif
#ifndef __clang__
#ifndef _Nonnull
#define _Nonnull
#endif
#ifndef _Nullable
#define _Nullable
#endif
#endif

#define mag_assert_name2(name, line) name ## line
#define mag_assert_name(line) mag_assert_name2(_assert_, line)
#define mag_static_assert(expr) extern void mag_assert_name(__LINE__)(bool STATIC_ASSERTION_FAILED[_Nonnull((expr)?1:-1)])

#define mag_version_pack(maj, mi) ((uint32_t)((((maj)&255)<<8)+((mi)&255)))
#define mag_version_major(ver) (((ver)>>8)&255)
#define mag_version_minor(ver) ((ver)&255)
#define MAG_VERSION mag_version_pack(0, 1) /* magnetron library version. */
#define MAG_STORAGE_VERSION 1 /* magnetron storage format version. */

/**
 * @brief Compute device types. Determines the type of hardware used for computation.
 */
typedef enum mag_ComputeDeviceType {
    MAG_COMPUTE_DEVICE_TYPE_CPU = 0,        /* CPU compute device */
    MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA = 1,   /* CUDA GPU compute device */

    MAG_COMPUTE_DEVICE_TYPE__NUM
} mag_ComputeDeviceType;
extern MAG_EXPORT const char* _Nonnull mag_device_type_get_name(mag_ComputeDeviceType op);

/**
 * @brief Pseudo-random number generator (PRNG) algorithms used for number generation and random tensor initialization.
 *      The default PRNG is Mersenne Twister, switching to PCG can yield better performance in some cases.
 */
typedef enum mag_PRNGAlgo {
    MAG_PRNG_MERSENNE_TWISTER = 0,  /* Mersenne Twister PRNG */
    MAG_PRNG_PCG = 1,               /* Permuted Congruential Generator PRNG */

    MAG_PRNG__NUM
} mag_PRNGAlgo;

/**
 * @brief Thread scheduling priority for CPU compute.
 *      This set the OS scheduling priority of the thread that executes the compute operations.
 *      The priority is only used if the compute device is a CPU.
 */
typedef enum mag_ThreadPrio {       /* Thread scheduling priority for CPU compute */
    MAG_THREAD_PRIO_NORMAL = 0,     /* Normal thread priority */
    MAG_THREAD_PRIO_MEDIUM = 1,     /* Medium thread priority */
    MAG_THREAD_PRIO_HIGH = 2,       /* High thread priority */
    MAG_THREAD_PRIO_REALTIME = 3,   /* Real-time thread priority */
} mag_ThreadPrio;

/**
 * @brief Desired Color channels for loading image tensors.
 */
typedef enum mag_ColorChannels {
    MAG_COLOR_CHANNELS_AUTO,        /* Automatically detect number of color channels */
    MAG_COLOR_CHANNELS_GRAY,        /* Grayscale F32 */
    MAG_COLOR_CHANNELS_GRAY_A,      /* Grayscale F32 + Alpha F32 */
    MAG_COLOR_CHANNELS_RGB,         /* R32G32B32 */
    MAG_COLOR_CHANNELS_RGBA,        /* R32G32B32A32 */

    MAG_COLOR_CHANNELS__NUM
} mag_ColorChannels;

extern MAG_EXPORT void* _Nonnull (*_Nonnull mag_get_alloc_fn(void))(void* _Nullable, size_t, size_t); /* Get global allocator. */
extern MAG_EXPORT void mag_set_alloc_fn(void* _Nonnull (*_Nonnull alloc)(void* _Nullable, size_t, size_t)); /* Set global allocator. */
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
typedef struct mag_Context mag_Context;

/**
 * @brief Compute device descriptor.
 *      Used to specify the compute device type and configuration when creating a context.
 *      The device type must be one of the mag_ComputeDeviceType values.
 *      If the type is MAG_COMPUTE_DEVICE_TYPE_CPU, thread_count can be set to 0 to detect hardware concurrency.
 *      If the type is MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA, cuda_device_id can be set to select a specific GPU.
 */
typedef struct mag_ComputeDeviceDesc {
    mag_ComputeDeviceType type;  /* Device type */
    uint32_t cpu_thread_count;   /* Number of threads if type == MAG_COMPUTE_DEVICE_TYPE_CPU. If set to 0, hardware concurrency of host CPU is detected. */
    uint32_t cuda_device_id;     /* CUDA device ID if type == MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA. Default: 0 (first GPU). */
} mag_ComputeDeviceDesc;
extern MAG_EXPORT mag_ComputeDeviceDesc mag_compute_device_desc_cpu(uint32_t thread_count);                           /* Helper to fill device descriptor for CPU compute device. */
extern MAG_EXPORT mag_ComputeDeviceDesc mag_compute_device_desc_cuda(uint32_t cuda_device_id);                        /* Helper to fill device descriptor for CUDA GPU compute device. */

extern MAG_EXPORT mag_Context* _Nonnull mag_ctx_create(mag_ComputeDeviceType device);                                 /* Create context with default config, and only specify device type. */
extern MAG_EXPORT mag_Context* _Nonnull mag_ctx_create2(const mag_ComputeDeviceDesc* _Nonnull device_info);           /* Create context with customized device config, and only specify device type. */
extern MAG_EXPORT mag_PRNGAlgo mag_ctx_get_prng_algorithm(const mag_Context* _Nonnull ctx);                           /* Get PRNG algorithm */
extern MAG_EXPORT void mag_ctx_set_prng_algorithm(mag_Context* _Nonnull ctx, mag_PRNGAlgo algorithm, uint64_t seed);  /* Set PRNG algorithm */
extern MAG_EXPORT mag_ComputeDeviceType mag_ctx_get_compute_device_type(const mag_Context* _Nonnull ctx);             /* Get compute device type */
extern MAG_EXPORT const char* _Nonnull mag_ctx_get_compute_device_name(const mag_Context* _Nonnull ctx);              /* Get the name of the compute device */
extern MAG_EXPORT const char* _Nonnull mag_ctx_get_os_name(const mag_Context* _Nonnull ctx);                          /* Get the name of the operating system */
extern MAG_EXPORT const char* _Nonnull mag_ctx_get_cpu_name(const mag_Context* _Nonnull ctx);                         /* Get the name of the CPU */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_virtual_cores(const mag_Context* _Nonnull ctx);                            /* Get the number of virtual cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_physical_cores(const mag_Context* _Nonnull ctx);                           /* Get the number of physical cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_sockets(const mag_Context* _Nonnull ctx);                                  /* Get the number of CPU sockets */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_total(const mag_Context* _Nonnull ctx);                        /* Get the total physical memory in bytes */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_free(const mag_Context* _Nonnull ctx);                         /* Get the free physical memory in bytes */
extern MAG_EXPORT bool mag_ctx_is_numa_system(const mag_Context* _Nonnull ctx);                                       /* Check if the system is NUMA */
extern MAG_EXPORT size_t mag_ctx_get_total_tensors_created(const mag_Context* _Nonnull ctx);                          /* Get total tensors created. (Including views) */
extern MAG_EXPORT void mag_ctx_grad_recorder_start(mag_Context* _Nonnull ctx);                                        /* Start gradient recording */
extern MAG_EXPORT void mag_ctx_grad_recorder_stop(mag_Context* _Nonnull ctx);                                         /* Stop gradient recording */
extern MAG_EXPORT bool mag_ctx_grad_recorder_is_running(const mag_Context* _Nonnull ctx);                             /* Check if gradient recording is running */
extern MAG_EXPORT void mag_ctx_destroy(mag_Context* _Nonnull ctx);                                                    /* Destroy context and free memory */

/**
 * @brief Multidimensional tensor of arbitrary rank and data type.
 *      The tensor is reference counted and can be shared between multiple tensors.
 *      Rule of Thumb for Reference Counting:
 *          - If you only use the reference temporarily and do not store it, no need to adjust the reference count.
 *          - If you store the reference (e.g., in a data structure), increase the reference count when storing and decrease it when removing.
 *      The rank is > 0 and <= MAG_MAX_DIMS. The shape of the tensor is an array of dimensions of size MAG_MAX_DIMS.
 *      Is a node in a static or dynamic computation graph, depending on the context execution mode.
 */
typedef struct mag_Tensor mag_Tensor;

/**
 * @brief Data types for tensors.
 */
typedef enum mag_DType {
    MAG_DTYPE_E8M23,        /* IEEE-754 32-bit floating point number. Commonly known as float32, f32. */
    MAG_DTYPE_E5M10,        /* IEEE-754 16-bit floating point number. Commonly known as float16, f16, half. */
    MAG_DTYPE_BOOL,         /* 1-byte boolean */
    MAG_DTYPE_I32,          /* 32-bit signed integer */

    MAG_DTYPE__NUM
} mag_DType;

mag_static_assert(MAG_DTYPE__NUM <= 0xff); /* Must fix in 1 byte */

/**
 * @brief Stores various information about a data type such as size, name, etc.
 */
typedef struct mag_DTypeMetadata {
    const char* _Nonnull name;   /* Name of the data type */
    size_t size;                 /* Size of the data type in bytes. Must be a power of two. */
    size_t align;                /* CPU Alignment of the data type in bytes. Must be a power of two. */
} mag_DTypeMetadata;
extern MAG_EXPORT const mag_DTypeMetadata* _Nonnull mag_dtype_meta_of(mag_DType type);

/**
 * @brief Create a new N-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tensor_empty(mag_Context* _Nonnull ctx, mag_DType type, int64_t rank, const int64_t* _Nonnull shape);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tensor_empty_like(mag_Tensor* _Nonnull isomorph);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tensor_empty_scalar(mag_Context* _Nonnull ctx, mag_DType type);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tensor_scalar(mag_Context* _Nonnull ctx, mag_DType type, float value);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tensor_full(mag_Context* _Nonnull ctx, mag_DType type, int64_t rank, const int64_t* _Nonnull shape, float value);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tensor_full_like(mag_Tensor* _Nonnull isomorph, float value);

/* ============ Tensor Operators ============ */

extern MAG_EXPORT mag_Tensor* _Nonnull mag_clone(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_view(mag_Tensor* _Nonnull x, const int64_t* _Nullable dims, uint32_t num_dims);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_view_slice(mag_Tensor* _Nonnull x, int64_t dim, int64_t start, int64_t len, int64_t step);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_reshape(mag_Tensor* _Nonnull x, const int64_t* _Nullable dims, uint32_t num_dims);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_transpose(mag_Tensor* _Nonnull x, int64_t dim1, int64_t dim2);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_permute(mag_Tensor* _Nonnull x, const int64_t* _Nonnull dims, uint32_t num_dims);

extern MAG_EXPORT mag_Tensor* _Nonnull mag_mean(mag_Tensor* _Nonnull x, const int64_t* _Nullable dims, uint32_t num_dims, bool keepdim);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_min(mag_Tensor* _Nonnull x, const int64_t* _Nullable dims, uint32_t num_dims, bool keepdim);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_max(mag_Tensor* _Nonnull x, const int64_t* _Nullable dims, uint32_t num_dims, bool keepdim);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sum(mag_Tensor* _Nonnull x, const int64_t* _Nullable dims, uint32_t num_dims, bool keepdim);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_argmin(mag_Tensor* _Nonnull x, const int64_t* _Nullable dims, uint32_t num_dims, bool keepdim);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_argmax(mag_Tensor* _Nonnull x, const int64_t* _Nullable dims, uint32_t num_dims, bool keepdim);

extern MAG_EXPORT mag_Tensor* _Nonnull mag_abs(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_abs_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sgn(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sgn_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_neg(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_neg_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_log(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_log_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sqr(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sqr_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sqrt(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sqrt_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sin(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sin_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_cos(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_cos_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_step(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_step_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_exp(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_exp_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_floor(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_floor_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_ceil(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_ceil_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_round(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_round_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_softmax(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_softmax_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_softmax_dv(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_softmax_dv_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sigmoid(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sigmoid_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sigmoid_dv(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sigmoid_dv_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_hard_sigmoid(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_hard_sigmoid_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_silu(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_silu_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_silu_dv(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_silu_dv_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tanh(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tanh_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tanh_dv(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tanh_dv_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_relu(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_relu_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_relu_dv(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_relu_dv_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_gelu(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_gelu_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_gelu_dv(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_gelu_dv_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_add(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_add_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sub( mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_sub_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_mul( mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_mul_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_div(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_div_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_matmul(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_repeat_back(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_and(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_and_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_or(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_or_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_xor(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_xor_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_not(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_not_(mag_Tensor* _Nonnull x);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_shl(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_shl_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_shr(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_shr_(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_eq(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_ne(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_le(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_ge(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_lt(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_gt(mag_Tensor* _Nonnull x, mag_Tensor* _Nonnull y);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tril(mag_Tensor* _Nonnull x, int32_t diag);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tril_(mag_Tensor* _Nonnull x, int32_t diag);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_triu(mag_Tensor* _Nonnull x, int32_t diag);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_triu_(mag_Tensor* _Nonnull x, int32_t diag);

/* ============ Tensor Init Operators ============ */

extern MAG_EXPORT void mag_tensor_fill_from_floats(mag_Tensor* _Nonnull t, const float* _Nonnull data, size_t len);       /* Copy floats into tensor buffer. If the tensors datatype is not float, the values are converted to the tensors dtype. */
extern MAG_EXPORT void mag_tensor_fill_from_raw_bytes(mag_Tensor* _Nonnull t, const void* _Nonnull data, size_t len);     /* Copy raw bytes into tensor buffer */
extern MAG_EXPORT void mag_tensor_fill_float(mag_Tensor* _Nonnull t, float x);                                            /* Set all tensor elements to a specific value. */
extern MAG_EXPORT void mag_tensor_fill_int(mag_Tensor* _Nonnull t, int32_t x);                                            /* Set all tensor elements to a specific value. */
extern MAG_EXPORT void mag_tensor_masked_fill_float(mag_Tensor* _Nonnull t, mag_Tensor* _Nonnull mask, float x);          /* Set all tensor elements to a specific value if the mask value at the same index is true. */
extern MAG_EXPORT void mag_tensor_masked_fill_int(mag_Tensor* _Nonnull t, mag_Tensor* _Nonnull mask, int32_t x);          /* Set all tensor elements to a specific value if the mask value at the same index is true. */
extern MAG_EXPORT void mag_tensor_fill_random_uniform_float(mag_Tensor* _Nonnull t, float min, float max);                /* Fill tensor with random values from uniform distribution within [min, max] */
extern MAG_EXPORT void mag_tensor_fill_random_uniform_int(mag_Tensor* _Nonnull t, int32_t min, int32_t max);              /* Fill tensor with random values from uniform distribution within [min, max] */
extern MAG_EXPORT void mag_tensor_fill_random_normal(mag_Tensor* _Nonnull t, float mean, float stddev);                   /* Fill tensor with random values from the normal distribution. */
extern MAG_EXPORT void mag_tensor_fill_random_bernoulli(mag_Tensor* _Nonnull t, float p);                                 /* Fill bool tensor with random values from the bernoulli distribution. */

/* ============ Tensor Property Accessors ============ */

extern MAG_EXPORT uint64_t mag_tensor_get_refcount(const mag_Tensor* _Nonnull t);                                         /* Return reference count of tensor itself. */
extern MAG_EXPORT uint64_t mag_tensor_get_storage_refcount(const mag_Tensor* _Nonnull t);                                 /* Return reference count of tensor itself. */
extern MAG_EXPORT size_t mag_tensor_get_memory_usage(const mag_Tensor* _Nonnull t);                                       /* Return memory used by this tensor in bytes. */
extern MAG_EXPORT void mag_tensor_set_name(mag_Tensor* _Nonnull t, const char* _Nonnull name);                            /* Set the name of the tensor */
extern MAG_EXPORT void mag_tensor_fmt_name(mag_Tensor* _Nonnull t, const char* _Nonnull fmt, ...);                        /* Format the name of the tensor */
extern MAG_EXPORT const char* _Nonnull mag_tensor_get_name(const mag_Tensor* _Nonnull t);                                 /* Get the name of the tensor */
extern MAG_EXPORT int64_t mag_tensor_get_rank(const mag_Tensor* _Nonnull t);                                              /* Get the rank (number of dimensions) of the tensor */
extern MAG_EXPORT const int64_t* _Nonnull mag_tensor_get_shape(const mag_Tensor* _Nonnull t);                             /* Get the dimensions of the tensor */
extern MAG_EXPORT const int64_t* _Nonnull mag_tensor_get_strides(const mag_Tensor* _Nonnull t);                           /* Get the strides of the tensor */
extern MAG_EXPORT mag_DType mag_tensor_get_dtype(const mag_Tensor* _Nonnull t);                                           /* Get the data type of the tensor */
extern MAG_EXPORT void* _Nonnull mag_tensor_get_data_ptr(const mag_Tensor* _Nonnull t);                                   /* Get pointer to first element. Might point to GPU or any other device memory. */
extern MAG_EXPORT void* _Nonnull mag_tensor_get_storage_base_ptr(const mag_Tensor* _Nonnull t);                           /* Get pointer to storage memory block base. Might point to GPU or any other device memory. */
extern MAG_EXPORT int64_t mag_tensor_get_data_size(const mag_Tensor* _Nonnull );                                          /* Get the size of the tensor buffer in bytes. */
extern MAG_EXPORT int64_t mag_tensor_get_numel(const mag_Tensor* _Nonnull t);                                             /* Get the total amount of elements in the tensor. */
extern MAG_EXPORT mag_Context* _Nonnull mag_tensor_get_ctx(const mag_Tensor* _Nonnull t);                                 /* Get the context of the tensor */
extern MAG_EXPORT void* _Nullable mag_tensor_get_user_data(const mag_Tensor* _Nonnull t);                                 /* Get the user data of the tensor */
extern MAG_EXPORT void mag_tensor_set_user_data(mag_Tensor* _Nonnull t, void* _Nullable ud);
extern MAG_EXPORT int64_t mag_tensor_get_width(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT int64_t mag_tensor_get_height(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT int64_t mag_tensor_get_channels(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT bool mag_tensor_is_view(const mag_Tensor* _Nonnull t);                                                /* Check if the tensor is a view of another tensor */
extern MAG_EXPORT mag_Tensor* _Nullable mag_tensor_get_view_base(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT size_t mag_tensor_get_view_offset(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT bool mag_tensor_is_floating_point_typed(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT bool mag_tensor_is_integral_typed(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT bool mag_tensor_is_integer_typed(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT bool mag_tensor_is_numeric_typed(const mag_Tensor* _Nonnull t);

/* ============ Tensor Shape Utils ============ */

extern MAG_EXPORT bool mag_tensor_is_shape_eq(const mag_Tensor* _Nonnull x, const mag_Tensor* _Nonnull y);              /* Checks if a and b have the same shape. */
extern MAG_EXPORT bool mag_tensor_are_strides_eq(const mag_Tensor* _Nonnull x, const mag_Tensor* _Nonnull y);           /* Checks if a and b have the same strides. */
extern MAG_EXPORT bool mag_tensor_can_broadcast(const mag_Tensor* _Nonnull small, const mag_Tensor* _Nonnull big);      /* Checks if b can be broadcasted into a. */
extern MAG_EXPORT bool mag_tensor_is_transposed(const mag_Tensor* _Nonnull t);                                          /* Check if the tensor is transposed */
extern MAG_EXPORT bool mag_tensor_is_permuted(const mag_Tensor* _Nonnull t);                                            /* Check if the tensor is permuted */
extern MAG_EXPORT bool mag_tensor_is_contiguous(const mag_Tensor* _Nonnull t);                                          /* Check if the tensor memory is contiguous */

/* ============ Gradient & Backprop API ============ */

extern MAG_EXPORT mag_Tensor* _Nullable mag_tensor_get_grad(const mag_Tensor* _Nonnull t);                          /* Get the gradient tensor of the tensor */
extern MAG_EXPORT bool mag_tensor_requires_grad(const mag_Tensor* _Nonnull t);                                        /* Check if the tensor requires gradient computation */
extern MAG_EXPORT void mag_tensor_set_requires_grad(mag_Tensor* _Nonnull t, bool requires_grad);                      /* Set if the tensor requires gradient computation */
extern MAG_EXPORT void mag_tensor_backward(mag_Tensor* _Nonnull t);                                                   /* Compute the gradient of the tensor */
extern MAG_EXPORT void mag_tensor_zero_grad(mag_Tensor* _Nonnull t);                                                  /* Zero the gradient of the tensor */

/* ============ Tensor Data Access API ============ */

extern MAG_EXPORT float mag_tensor_subscript_get_multi(mag_Tensor* _Nonnull t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5);
extern MAG_EXPORT void mag_tensor_subscript_set_multi(mag_Tensor* _Nonnull t, int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4, int64_t i5, float val);
extern MAG_EXPORT float mag_tensor_subscript_get_flattened(mag_Tensor* _Nonnull t, int64_t idx);
extern MAG_EXPORT void mag_tensor_subscript_set_flattened(mag_Tensor* _Nonnull t, int64_t idx, float val);
extern MAG_EXPORT void* _Nonnull mag_tensor_get_raw_data_as_bytes(mag_Tensor* _Nonnull t);
extern MAG_EXPORT void mag_tensor_get_raw_data_as_bytes_free(void* _Nonnull ret_val);
extern MAG_EXPORT float* _Nonnull mag_tensor_get_data_as_floats(mag_Tensor* _Nonnull t);
extern MAG_EXPORT void mag_tensor_get_data_as_floats_free(float* _Nonnull ret_val);
extern MAG_EXPORT float mag_tensor_get_item_float(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT int32_t mag_tensor_get_item_int(const mag_Tensor* _Nonnull t);
extern MAG_EXPORT bool mag_tensor_get_item_bool(const mag_Tensor* _Nonnull t);

/* ============ Tensor Misc API ============ */

extern MAG_EXPORT void mag_tensor_incref(mag_Tensor* _Nonnull t);
extern MAG_EXPORT bool mag_tensor_decref(mag_Tensor* _Nonnull t);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tensor_detach(mag_Tensor* _Nonnull t);
extern MAG_EXPORT uint32_t mag_tensor_weak_hash(const mag_Tensor* _Nonnull t); /* Returns hash of the tensor properties, not including data. */
extern MAG_EXPORT char* _Nonnull mag_tensor_to_string(mag_Tensor* _Nonnull t, bool with_header, size_t from_start_count, size_t from_end_count);
extern MAG_EXPORT void mag_tensor_to_string_free_data(char* _Nonnull ret_val);
extern MAG_EXPORT void mag_tensor_img_draw_box(mag_Tensor* _Nonnull t, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t wi, uint32_t rgb);
extern MAG_EXPORT void mag_tensor_img_draw_text(mag_Tensor* _Nonnull t, int32_t x, int32_t y, int32_t size, uint32_t rgb, const char* _Nonnull txt);
extern MAG_EXPORT mag_Tensor* _Nonnull mag_tensor_load_image(mag_Context* _Nonnull ctx, const char* _Nonnull file, mag_ColorChannels channels, uint32_t resize_w, uint32_t resize_h);    /* Create a tensor from an image file. */
extern MAG_EXPORT void mag_tensor_save_image(const mag_Tensor* _Nonnull t, const char* _Nonnull file);                                                                                    /* Save tensor data as an image */
extern MAG_EXPORT void mag_tensor_export_forward_graph_graphviz(mag_Tensor* _Nonnull t, const char* _Nonnull file);                                                                      /* Export tensor computation graph as Graphviz DOT file *//* Get image channels from tensor */
extern MAG_EXPORT void mag_tensor_export_backward_graph_graphviz(mag_Tensor* _Nonnull t, const char* _Nonnull file);

#ifdef __cplusplus
}
#endif
#endif
