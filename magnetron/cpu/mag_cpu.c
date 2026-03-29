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

#include "mag_cpu.h"
#include "mag_cpu_threadpool.h"
#include "mag_cpu_detect_optimal.h"
#include "mag_tls_arena.h"

#include <core/mag_context.h>
#include <core/mag_thread.h>
#include <core/mag_tensor.h>
#include <core/mag_alloc.h>

MAG_THREAD_LOCAL mag_scratch_arena_t mag_tls_arena = MAG_SCRATCH_ARENA_INIT(4ull<<20);

const uint32_t mag_crc32c_lut[256] = {
    0x00000000, 0xf26b8303, 0xe13b70f7, 0x1350f3f4, 0xc79a971f, 0x35f1141c,
    0x26a1e7e8, 0xd4ca64eb, 0x8ad958cf, 0x78b2dbcc, 0x6be22838, 0x9989ab3b,
    0x4d43cfd0, 0xbf284cd3, 0xac78bf27, 0x5e133c24, 0x105ec76f, 0xe235446c,
    0xf165b798, 0x030e349b, 0xd7c45070, 0x25afd373, 0x36ff2087, 0xc494a384,
    0x9a879fa0, 0x68ec1ca3, 0x7bbcef57, 0x89d76c54, 0x5d1d08bf, 0xaf768bbc,
    0xbc267848, 0x4e4dfb4b, 0x20bd8ede, 0xd2d60ddd, 0xc186fe29, 0x33ed7d2a,
    0xe72719c1, 0x154c9ac2, 0x061c6936, 0xf477ea35, 0xaa64d611, 0x580f5512,
    0x4b5fa6e6, 0xb93425e5, 0x6dfe410e, 0x9f95c20d, 0x8cc531f9, 0x7eaeb2fa,
    0x30e349b1, 0xc288cab2, 0xd1d83946, 0x23b3ba45, 0xf779deae, 0x05125dad,
    0x1642ae59, 0xe4292d5a, 0xba3a117e, 0x4851927d, 0x5b016189, 0xa96ae28a,
    0x7da08661, 0x8fcb0562, 0x9c9bf696, 0x6ef07595, 0x417b1dbc, 0xb3109ebf,
    0xa0406d4b, 0x522bee48, 0x86e18aa3, 0x748a09a0, 0x67dafa54, 0x95b17957,
    0xcba24573, 0x39c9c670, 0x2a993584, 0xd8f2b687, 0x0c38d26c, 0xfe53516f,
    0xed03a29b, 0x1f682198, 0x5125dad3, 0xa34e59d0, 0xb01eaa24, 0x42752927,
    0x96bf4dcc, 0x64d4cecf, 0x77843d3b, 0x85efbe38, 0xdbfc821c, 0x2997011f,
    0x3ac7f2eb, 0xc8ac71e8, 0x1c661503, 0xee0d9600, 0xfd5d65f4, 0x0f36e6f7,
    0x61c69362, 0x93ad1061, 0x80fde395, 0x72966096, 0xa65c047d, 0x5437877e,
    0x4767748a, 0xb50cf789, 0xeb1fcbad, 0x197448ae, 0x0a24bb5a, 0xf84f3859,
    0x2c855cb2, 0xdeeedfb1, 0xcdbe2c45, 0x3fd5af46, 0x7198540d, 0x83f3d70e,
    0x90a324fa, 0x62c8a7f9, 0xb602c312, 0x44694011, 0x5739b3e5, 0xa55230e6,
    0xfb410cc2, 0x092a8fc1, 0x1a7a7c35, 0xe811ff36, 0x3cdb9bdd, 0xceb018de,
    0xdde0eb2a, 0x2f8b6829, 0x82f63b78, 0x709db87b, 0x63cd4b8f, 0x91a6c88c,
    0x456cac67, 0xb7072f64, 0xa457dc90, 0x563c5f93, 0x082f63b7, 0xfa44e0b4,
    0xe9141340, 0x1b7f9043, 0xcfb5f4a8, 0x3dde77ab, 0x2e8e845f, 0xdce5075c,
    0x92a8fc17, 0x60c37f14, 0x73938ce0, 0x81f80fe3, 0x55326b08, 0xa759e80b,
    0xb4091bff, 0x466298fc, 0x1871a4d8, 0xea1a27db, 0xf94ad42f, 0x0b21572c,
    0xdfeb33c7, 0x2d80b0c4, 0x3ed04330, 0xccbbc033, 0xa24bb5a6, 0x502036a5,
    0x4370c551, 0xb11b4652, 0x65d122b9, 0x97baa1ba, 0x84ea524e, 0x7681d14d,
    0x2892ed69, 0xdaf96e6a, 0xc9a99d9e, 0x3bc21e9d, 0xef087a76, 0x1d63f975,
    0x0e330a81, 0xfc588982, 0xb21572c9, 0x407ef1ca, 0x532e023e, 0xa145813d,
    0x758fe5d6, 0x87e466d5, 0x94b49521, 0x66df1622, 0x38cc2a06, 0xcaa7a905,
    0xd9f75af1, 0x2b9cd9f2, 0xff56bd19, 0x0d3d3e1a, 0x1e6dcdee, 0xec064eed,
    0xc38d26c4, 0x31e6a5c7, 0x22b65633, 0xd0ddd530, 0x0417b1db, 0xf67c32d8,
    0xe52cc12c, 0x1747422f, 0x49547e0b, 0xbb3ffd08, 0xa86f0efc, 0x5a048dff,
    0x8ecee914, 0x7ca56a17, 0x6ff599e3, 0x9d9e1ae0, 0xd3d3e1ab, 0x21b862a8,
    0x32e8915c, 0xc083125f, 0x144976b4, 0xe622f5b7, 0xf5720643, 0x07198540,
    0x590ab964, 0xab613a67, 0xb831c993, 0x4a5a4a90, 0x9e902e7b, 0x6cfbad78,
    0x7fab5e8c, 0x8dc0dd8f, 0xe330a81a, 0x115b2b19, 0x020bd8ed, 0xf0605bee,
    0x24aa3f05, 0xd6c1bc06, 0xc5914ff2, 0x37faccf1, 0x69e9f0d5, 0x9b8273d6,
    0x88d28022, 0x7ab90321, 0xae7367ca, 0x5c18e4c9, 0x4f48173d, 0xbd23943e,
    0xf36e6f75, 0x0105ec76, 0x12551f82, 0xe03e9c81, 0x34f4f86a, 0xc69f7b69,
    0xd5cf889d, 0x27a40b9e, 0x79b737ba, 0x8bdcb4b9, 0x988c474d, 0x6ae7c44e,
    0xbe2da0a5, 0x4c4623a6, 0x5f16d052, 0xad7d5351
};

static MAG_HOTPROC mag_status_t mag_cpu_submit(mag_device_t *dvc, const mag_command_t *cmd) {
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
        return MAG_STATUS_OK; /* We're done */
    }
    mag_threadpool_parallel_compute(cpu_dvc->pool, cmd, intraop_workers); /* Multithreaded exec + barrier */
    return MAG_STATUS_OK;
}

static mag_status_t mag_cpu_storage_dtor(void *self) {
    mag_storage_buffer_t *buf = self;
    mag_context_t *ctx = buf->ctx;
    mag_assert(ctx->telemetry.num_alive_storages > 0, "double freed storage");
    --ctx->telemetry.num_alive_storages;
    if (!(buf->flags & MAG_STORAGE_FLAG_BORROWED))
        (*mag_try_alloc)((void *)buf->base, 0, MAG_CPU_BUF_ALIGN);
    mag_slab_free(&ctx->storage_slab, buf);
    return MAG_STATUS_OK;
}

static mag_status_t mag_cpu_alloc_storage(mag_device_t *host, mag_storage_buffer_t **out, size_t size, mag_dtype_t dtype) {
    mag_context_t *ctx = host->ctx;
    mag_storage_buffer_t *buf = mag_slab_alloc(&ctx->storage_slab);
    *buf = (mag_storage_buffer_t) { /* Set up storage buffer. */
        .ctx = ctx,
        .aux = {},
        .flags = MAG_STORAGE_FLAG_ACCESS_W,
        .alignment = MAG_CPU_BUF_ALIGN,
        .base = 0,
        .size = size,
        .device = host,
    };
    if (size <= MAG_CPU_BUF_INTRUSIVE_CAP) { /* Store value intrusive (scalar storage optimization) */
        buf->base = (uintptr_t)buf->aux.intrusive_storage;
        buf->flags |= MAG_STORAGE_FLAG_BORROWED;
    } else {
        void *base = (*mag_try_alloc)(NULL, size, MAG_CPU_BUF_ALIGN);
        if (mag_unlikely(!base)) { /* Allocation failed (OOM). */
            mag_slab_free(&ctx->storage_slab, buf);
            return MAG_STATUS_ERR_MEMORY_ALLOCATION_FAILED;
        }
        buf->base = (uintptr_t)base;
    }
    mag_assert2(!(buf->base&(MAG_CPU_BUF_ALIGN-1))); /* Ensure alignment */
    mag_rc_init_object(buf, &mag_cpu_storage_dtor);
    ++host->ctx->telemetry.num_alive_storages;
    *out = buf;
    return MAG_STATUS_OK;
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
    mag_numa_init(&dvc->numa_ctrl, MAG_NUMA_STRATEGY_DISTRIBUTE); /* TODO: make configureable */
    mag_blas_detect_optimal_specialization(ctx, &dvc->kernels);
    if (num_threads > 1) {
        dvc->pool = mag_threadpool_create(ctx, num_threads, &dvc->kernels, &dvc->numa_ctrl, sched_prio);
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
        .id = mag_device(CPU, 0),
        .physical_device_name = "CPU",
        .impl = cpu_dvc,
        .is_async = false,
        .submit = &mag_cpu_submit,
        .alloc_storage = &mag_cpu_alloc_storage,
        .manual_seed = &mag_cpu_manual_seed
    };
    snprintf(device->physical_device_name, sizeof(device->physical_device_name), "%s", ctx->machine.cpu_name);
    return device;
}

static void mag_cpu_release_interface(mag_device_t *ctx) {
    mag_cpu_device_t *cpu_dvc = ctx->impl;
    mag_cpu_destroy_device(cpu_dvc);
    (*mag_alloc)(ctx, 0, 0); /* Free all memory */
}

static bool mag_cpu_init(mag_backend_t *self, mag_context_t *ctx) {
    mag_assert2(!self->impl);
    uint32_t hwc = mag_xmax(1, ctx->machine.cpu_virtual_cores);
    uint32_t nt = ctx->machine.cpu_virtual_cores;
    nt = nt ? nt : hwc;
    self->impl = mag_cpu_init_interface(ctx, nt);
    return true;
}
static bool mag_cpu_shutdown(mag_backend_t *self) {
    mag_assert2(self->impl);
    mag_cpu_release_interface(self->impl);
    self->impl = NULL;
    return true;
}
static uint32_t mag_cpu_backend_version(mag_backend_t *bck) { return MAG_CPU_BACKEND_VERSION; }
static uint32_t mag_cpu_backend_runtime_version(mag_backend_t *bck) { return MAG_VERSION; }
static const char* mag_cpu_backend_id(mag_backend_t *bck) { return "cpu"; }
static uint32_t mag_cpu_backend_num_devices(mag_backend_t *bck) { return 1; }
static uint32_t mag_cpu_backend_best_device_idx(mag_backend_t *bck) { return 0; }
mag_device_t *mag_cpu_backend_get_device(mag_backend_t *bck, uint32_t idx) {
    return bck->impl;
}

uint32_t MAG_BACKEND_SYM_ABI_COOKIE(void){
    return mag_pack_abi_cookie('M', 'A', 'G', MAG_BACKEND_MODULE_ABI_VER);
}

mag_backend_t *MAG_BACKEND_SYM_INIT(mag_context_t *ctx) { /* Create and return interface struct */
    mag_backend_t *backend = (*mag_alloc)(NULL, sizeof(*backend)+sizeof(mag_device_t *), 0);
    memset(backend, 0, sizeof(*backend)+sizeof(mag_device_t *));
    *backend = (mag_backend_t){
        .impl = NULL,
        .init = &mag_cpu_init,
        .shutdown = &mag_cpu_shutdown,
        .backend_version = &mag_cpu_backend_version,
        .runtime_version = &mag_cpu_backend_runtime_version,
        .id = &mag_cpu_backend_id,
        .num_devices = &mag_cpu_backend_num_devices,
        .best_device_id = &mag_cpu_backend_best_device_idx,
        .get_device = &mag_cpu_backend_get_device,
    };
    return backend;
}

void MAG_BACKEND_SYM_SHUTDOWN(mag_backend_t *backend) { /* Free interface struct */
    (*mag_alloc)(backend, 0, 0);
}

