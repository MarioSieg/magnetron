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

#include "mag_cpu.h"
#include "mag_cpu_threadpool.h"
#include "mag_cpu_detect_optimal.h"

#include <core/mag_context.h>
#include <core/mag_thread.h>
#include <core/mag_tensor.h>
#include <core/mag_alloc.h>

static MAG_HOTPROC void mag_cpu_submit(mag_device_t *dvc, const mag_command_t *cmd) {
    mag_cpu_device_t *cpu_dvc = dvc->impl;
    uint32_t intraop_workers = mag_cpu_tune_heuristics_intraop_workers(cmd, dvc); /* Determine number of intra-op workers */
    if (intraop_workers <= 1) { /* Main thread does the work (single threaded mode). */
        volatile mag_atomic64_t next_tile = 0; /* Tile index for the next tile to process. */
        mag_kernel_payload_t *yy = &cpu_dvc->pool->workers[0].payload; /* TODO: Ugly */
        mag_kernel_payload_t payload = {
            .cmd = cmd,
            .thread_idx = 0,
            .thread_num = 1,
            .prng = &cpu_dvc->primary_prng,
            .mm_next_tile = &next_tile,
            .mm_params = yy->mm_params
        };
        mag_worker_exec_thread_local(&cpu_dvc->kernels, &payload);
        return; /* We're done */
    }
    mag_threadpool_parallel_compute(cpu_dvc->pool, cmd, intraop_workers); /* Multithreaded exec + barrier */
}

#define mag_ranges_overlap(a, asz, b, bsz) (((uintptr_t)(a)+(asz)) > (uintptr_t)(b) && ((uintptr_t)(b)+(bsz)) > (uintptr_t)(a))

static void mag_cpu_transfer(mag_storage_buffer_t *sto, mag_transfer_dir_t dir, size_t offs, void *inout, size_t size) {
    mag_assert2(inout && size);
    mag_assert(!(size % sto->granularity), "(%zu) mod (%zu) != 0", size, sto->granularity);
    uintptr_t base = sto->base;
    uintptr_t dp = base+offs;
    uintptr_t dpe = dp+size;
    mag_assert2(dpe > dp);
    mag_assert2(dpe <= base+sto->size);
    mag_assert2(!(offs % sto->granularity));
    uintptr_t hp = (uintptr_t)inout;
    uintptr_t hpe = hp+size;
    mag_assert2(hpe > hp);
    mag_assert2(!mag_ranges_overlap((void *)hp, size, (void *)dp, size));
    void *restrict device = (void *)dp;
    void *restrict host = inout;
    if (dir == MAG_TRANSFER_DIR_H2D) memcpy(device, host, size);
    else memcpy(host, device, size);
}

static void mag_cpu_convert(mag_storage_buffer_t *sto, mag_transfer_dir_t dir, size_t offs, void *restrict host, size_t size, mag_dtype_t hdt) {
    mag_assert2(host && size);
    mag_dtype_t ddt = sto->dtype;
    if (ddt == hdt) { /* identical dtype – delegate to raw copy */
        (*sto->transfer)(sto, dir, offs, host, size);
        return;
    }
    uintptr_t base = sto->base;
    size_t hsz = mag_type_trait(hdt)->size;
    size_t dsz = mag_type_trait(ddt)->size;
    mag_assert2(!(hsz & (hsz-1)));              /* pow2 */
    mag_assert2(!(size & (hsz-1)));             /* multiple of dtype size */
    mag_assert2(!((uintptr_t)host & (hsz-1)));  /* aligned */
    mag_assert2(!(dsz & (dsz-1)));              /* pow2 */
    mag_assert2(!(offs & (dsz-1)));             /* multiple of dtype size */
    size_t helem = size / hsz;                  /* host numel */
    mag_assert2(helem > 0);
    mag_assert2(!(sto->granularity & (dsz-1))); /* sane granularity */
    size_t tnb = helem * sto->granularity;      /* device bytes */
    mag_assert2(tnb <= sto->size-offs);
    void *dvb = (void *)(base+offs);
    void (*kern)(size_t, const void *, mag_dtype_t, void *, mag_dtype_t)
        = ((mag_cpu_device_t *)sto->device->impl)->kernels.vector_cast;
    mag_assert2(kern);
    mag_assert2(!mag_ranges_overlap(host, size, dvb, tnb));
    void *restrict device = dvb;
    if (dir == MAG_TRANSFER_DIR_H2D) (*kern)(size, host, hdt, device, ddt);
    else (*kern)(tnb, device, ddt, host, hdt);
}

static void mag_cpu_storage_dtor(void *self) {
    mag_storage_buffer_t *buf = self;
    mag_context_t *ctx = buf->ctx;
    mag_assert(ctx->num_alive_storages > 0, "double freed storage");
    --ctx->num_alive_storages;
    if (!(buf->flags & MAG_STORAGE_FLAG_INTRUSIVE))
        (*mag_alloc)((void *)buf->base, 0, MAG_CPU_BUF_ALIGN);
    mag_fixed_pool_free_block(&ctx->storage_pool, buf);
}

static void mag_cpu_alloc_storage(mag_device_t *host, mag_storage_buffer_t **out, size_t size, mag_dtype_t dtype) {
    mag_context_t *ctx = host->ctx;
    mag_storage_buffer_t *buf = mag_fixed_pool_alloc_block(&ctx->storage_pool);
    *buf = (mag_storage_buffer_t) { /* Set up storage buffer. */
        .ctx = ctx,
        .aux = {},
        .flags = MAG_STORAGE_FLAG_ACCESS_W,
        .base = 0,
        .size = size,
        .alignment = size <= sizeof(void *) ? MAG_CPU_BUF_ALIGN : 1,
        .dtype = dtype,
        .granularity = mag_type_trait(dtype)->size,
        .device = host,
        .transfer = &mag_cpu_transfer,
        .convert = &mag_cpu_convert
    };
    if (size <= sizeof(void *)) { /* Store value intrusive (scalar storage optimization) */
        buf->base = (uintptr_t)&buf->aux.inline_buf[0]; /* Use 8-byte impl pointer for storage. TODO: this does NOT guarantee MAG_CPU_BUF_ALIGN alignment. */
        buf->flags |= MAG_STORAGE_FLAG_INTRUSIVE;
    } else {
        buf->base = (uintptr_t)(*mag_alloc)(NULL, size, MAG_CPU_BUF_ALIGN);
    }
    mag_rc_init_object(buf, &mag_cpu_storage_dtor);
    ++host->ctx->num_alive_storages;
    *out = buf;
}

static void mag_cpu_manual_seed(mag_device_t *dvc, uint64_t seed) {
    mag_cpu_device_t *cpu_dvc = dvc->impl;
    mag_philox4x32_stream_seed(&cpu_dvc->primary_prng, seed, 0);
    if (cpu_dvc->pool) {
        for (uint32_t i=0; i < cpu_dvc->pool->num_allocated_workers; ++i) {
            mag_worker_t *worker = &cpu_dvc->pool->workers[i];
            mag_philox4x32_stream_seed(&worker->prng, seed, worker->payload.thread_idx+1);
        }
    }
}

static mag_cpu_device_t *mag_cpu_init_device(mag_context_t *ctx, uint32_t num_threads) {
    mag_thread_prio_t sched_prio = MAG_THREAD_PRIO_HIGH;
    mag_cpu_device_t *dvc = (*mag_alloc)(NULL, sizeof(*dvc), 0);
    memset(dvc, 0, sizeof(*dvc));
    *dvc = (mag_cpu_device_t) {
        .ctx = ctx,
        .pool = NULL,
        .num_allocated_workers = 0,
        .kernels = {},
        .primary_prng = {}
    };
    mag_blas_detect_optimal_specialization(ctx, &dvc->kernels);
    if (num_threads > 1) {
        dvc->pool = mag_threadpool_create(ctx, num_threads, &dvc->kernels, sched_prio);
        dvc->num_allocated_workers = num_threads;
    }
    if (*dvc->kernels.init) (*dvc->kernels.init)();
    return dvc;
}

static void mag_cpu_destroy_device(mag_cpu_device_t *dvc) {
    if (*dvc->kernels.deinit) (*dvc->kernels.deinit)();
    if (dvc->pool) mag_threadpool_destroy(dvc->pool);
    (*mag_alloc)(dvc, 0, 0);
}

static mag_device_t *mag_cpu_init_interface(mag_context_t *ctx, uint32_t num_threads) {
    mag_cpu_device_t *cpu_dvc = mag_cpu_init_device(ctx, num_threads);
    mag_device_t *device = (*mag_alloc)(NULL, sizeof(*device), 0);
    *device = (mag_device_t) { /* Initialize device interface */
        .ctx = ctx,
        .physical_device_name = "CPU",
        .impl = cpu_dvc,
        .is_async = false,
        .submit = &mag_cpu_submit,
        .alloc_storage = &mag_cpu_alloc_storage,
        .manual_seed = &mag_cpu_manual_seed
    };
    snprintf(device->id, sizeof(device->id), "cpu:0");
    snprintf(device->physical_device_name, sizeof(device->physical_device_name), "%s", ctx->machine.cpu_name);
    return device;
}

static void mag_cpu_release_interface(mag_device_t *ctx) {
    mag_cpu_device_t *cpu_dvc = ctx->impl;
    mag_cpu_destroy_device(cpu_dvc);
    (*mag_alloc)(ctx, 0, 0); /* Free all memory */
}

static uint32_t mag_cpu_backend_version(mag_backend_t *bck) { return MAG_CPU_BACKEND_VERSION; }
static uint32_t mag_cpu_backend_runtime_version(mag_backend_t *bck) { return MAG_VERSION; }
static uint32_t mag_cpu_backend_score(mag_backend_t *bck) { return 10; }
static const char* mag_cpu_backend_id(mag_backend_t *bck) { return "cpu"; }
static uint32_t mag_cpu_backend_num_devices(mag_backend_t *bck) { return 1; }
static uint32_t mag_cpu_backend_best_device_idx(mag_backend_t *bck) { return 0; }
mag_device_t *mag_cpu_backend_init_device(mag_backend_t *bck, mag_context_t *ctx, uint32_t idx) {
    uint32_t hw_concurrency = mag_xmax(1, ctx->machine.cpu_virtual_cores);
    /*uint32_t num_threads = desc->cpu_thread_count; TODO */
    uint32_t num_threads = ctx->machine.cpu_virtual_cores;
    num_threads = num_threads ? num_threads : hw_concurrency;
    mag_device_t *dvc = mag_cpu_init_interface(ctx, num_threads);
    return dvc;
}
void mag_cpu_backend_destroy_device(mag_backend_t *bck, mag_device_t *dvc) {
    mag_cpu_release_interface(dvc);
}

uint32_t MAG_BACKEND_SYM_ABI_COOKIE(void){
    return mag_pack_abi_cookie('M', 'A', 'G', MAG_BACKEND_MODULE_ABI_VER);
}

mag_backend_t *MAG_BACKEND_SYM_INIT(mag_context_t *ctx) {
    mag_backend_t *backend = (*mag_alloc)(NULL, sizeof(*backend), 0);
    memset(backend, 0, sizeof(*backend));
    *backend = (mag_backend_t){
        .backend_version = &mag_cpu_backend_version,
        .runtime_version = &mag_cpu_backend_runtime_version,
        .score = &mag_cpu_backend_score,
        .id = &mag_cpu_backend_id,
        .num_devices = &mag_cpu_backend_num_devices,
        .best_device_idx = &mag_cpu_backend_best_device_idx,
        .init_device = &mag_cpu_backend_init_device,
        .destroy_device = &mag_cpu_backend_destroy_device,
        .impl = NULL
    };
    return backend;
}

void MAG_BACKEND_SYM_SHUTDOWN(mag_backend_t *backend) {
    (*mag_alloc)(backend, 0, 0);
}

