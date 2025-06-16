/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
*/

#include "magnetron_internal.h"

extern mag_IComputeDevice* mag_init_device_cpu(mag_Context* ctx, const mag_ComputeDeviceDesc* desc);   /* Initialize CPU compute device. Implemented in magnetron_cpu.c */
extern void mag_destroy_device_cpu(mag_IComputeDevice* dvc);      /* Destroy CPU compute device. Implemented in magnetron_cpu.c */

#ifdef MAG_ENABLE_CUDA
extern mag_IComputeDevice* mag_init_device_cuda(mag_Context* ctx, const mag_ComputeDeviceDesc* desc);  /* Initialize GPU compute device. Implemented in magnetron_cuda.cu */
extern void mag_destroy_device_cuda(mag_IComputeDevice* dvc);     /* Destroy GPU compute device. Implemented in magnetron_cuda.cu */
#endif

#define MAG_DEVICE_FALLBACK MAG_COMPUTE_DEVICE_TYPE_CPU

static const mag_IDeviceFactory* const mag_device_factories[MAG_COMPUTE_DEVICE_TYPE__NUM] = {
    [MAG_COMPUTE_DEVICE_TYPE_CPU] = &(mag_IDeviceFactory){
        .init = &mag_init_device_cpu,
        .destroy = &mag_destroy_device_cpu,
    },
#ifdef MAG_ENABLE_CUDA
    [MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA] = &(mag_IDeviceFactory){
        .init = &mag_init_device_cuda,
        .destroy = &mag_destroy_device_cuda,
    },
#else
    [MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA] = NULL, /* CUDA not enabled. */
#endif
};

mag_IComputeDevice* mag_init_dynamic_device(mag_Context* ctx, const mag_ComputeDeviceDesc* desc) {
    mag_assert2(ctx && desc);
    mag_ComputeDeviceType type = desc->type;
    mag_assert2(type < MAG_COMPUTE_DEVICE_TYPE__NUM);
    mag_assert2(mag_device_factories[MAG_DEVICE_FALLBACK]);     /* Fallback factory must be present. */
    const mag_IDeviceFactory* factory = mag_device_factories[type];
    if (mag_unlikely(!factory)) {
        mag_log_error("No device factory for type '%s', falling back to CPU.", mag_device_type_get_name(type));
        goto fallback;
    }
    mag_IComputeDevice* dvc = (*factory->init)(ctx, desc);
    if (mag_unlikely(!dvc)) {
        mag_log_error("Failed to initialize device of type '%s', falling back to CPU.", mag_device_type_get_name(type));
        goto fallback;
    }
    return dvc;
    fallback:
        type = MAG_DEVICE_FALLBACK;
        factory = mag_device_factories[type];
        mag_assert2(factory);
        dvc = (*factory->init)(ctx, desc);
        mag_assert2(dvc);  /* Ensure fallback device is created. */
        return dvc;
}

void mag_destroy_dynamic_device(mag_IComputeDevice* dvc) {
    if (!dvc) return;
    (*mag_device_factories[dvc->type]->destroy)(dvc);
}
