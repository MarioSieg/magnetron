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

#include "mag_context.h"
#include "mag_alloc.h"
#include "mag_os.h"
#include "mag_tensor.h"
#include "mag_autodiff.h"
#include "mag_machine.h"

#include <time.h>

/* Print host system and machine information. */
static void mag_system_host_info_dump(mag_context_t *ctx) {
    mag_log_info("OS/Kernel: %s", ctx->machine.os_name);
    const char *cpu_arch = "?";
#if defined(__x86_64__) || defined(_M_X64)
    cpu_arch = "x86-64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    cpu_arch = "aarch64";
#else
#error "Unknwon CPU arch"
#endif
    mag_log_info(
        "CPU: %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u, L1D: %.01f KiB, L2: %.01f KiB, L3: %.01f MiB",
        ctx->machine.cpu_name,
        ctx->machine.cpu_virtual_cores,
        ctx->machine.cpu_physical_cores,
        ctx->machine.cpu_sockets,
        (mag_e11m52_t)ctx->machine.cpu_l1_size/1024.0,
        (mag_e11m52_t) ctx->machine.cpu_l2_size/1024.0,
        (mag_e11m52_t)ctx->machine.cpu_l3_size/1024.0/1024.0
    );
#if defined(__x86_64__) || defined(_M_X64) /* Print detected CPU features for x86-64 platforms. */
    if (mag_log_enabled) {
        mag_log_info("%s CPU flags:", cpu_arch);
        for (unsigned i=0, j=0; i < MAG_AMD64_CAP__NUM; ++i) {
            if (i == MAG_AMD64_CAP_AMD || i == MAG_AMD64_CAP_INTEL) continue; /* Skip vendor caps */
            if (ctx->machine.amd64_cpu_caps & mag_amd64_cap_bit(i)) {
                if (!(j++&7)) printf(j-1 ? "\n\t" : "\t");
                printf("%s ", mag_amd64_cpu_cap_names[i]);
            }
        }
        putchar('\n');
    }
#elif defined(__aarch64__) /* Print detected CPU features for ARM64 platforms. */
    if (mag_log_enabled) {
        printf(MAG_CC_CYAN "[magnetron] " MAG_CC_RESET "%s caps: ", cpu_arch);
        for (uint32_t i=0; i < MAG_ARM64_CAP__NUM; ++i)
            if (ctx->machine.arm64_cpu_caps & (1ull<<i))
                printf("%s ", mag_arm64_cpu_cap_names[i]);
        putchar('\n');
    }
#endif
    /* Now print memory information. */
    mag_e11m52_t mem_total, mem_free, mem_used;
    const char *mem_unit_total, *mem_unit_free, *mem_unit_used;
    mag_humanize_memory_size(ctx->machine.phys_mem_total, &mem_total, &mem_unit_total);
    mag_humanize_memory_size(ctx->machine.phys_mem_free, &mem_free, &mem_unit_free);
    mag_humanize_memory_size((size_t)llabs((int64_t)ctx->machine.phys_mem_total-(int64_t)ctx->machine.phys_mem_free), &mem_used, &mem_unit_used);
    mag_e11m52_t mem_used_percent = fabs((mag_e11m52_t)(ctx->machine.phys_mem_total-ctx->machine.phys_mem_free))/(mag_e11m52_t)ctx->machine.phys_mem_total*100.0;
    mag_log_info("Physical Machine Memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);
}

/* Print compiler information such as name, version and build time. */
static MAG_COLDPROC void mag_ctx_dump_compiler_info(void) {
    const char *compiler_name = "Unknown";
    int compiler_version_major = 0, compiler_version_minor = 0;
#ifdef __clang__
    compiler_name = "Clang";
    compiler_version_major = __clang_major__;
    compiler_version_minor = __clang_minor__;
#elif defined(__GNUC__)
    compiler_name = "GCC";
    compiler_version_major = __GNUC__;
    compiler_version_minor = __GNUC_MINOR__;
#elif defined(_MSC_VER)
    compiler_name = "MSVC";
    compiler_version_major = _MSC_VER / 100;
    compiler_version_minor = _MSC_VER % 100;
#endif
    mag_log_info("magnetron v.%d.%d - " __DATE__ " " __TIME__ " - %s %d.%d", mag_version_major(MAG_VERSION), mag_version_minor(MAG_VERSION), compiler_name, compiler_version_major, compiler_version_minor);
}

/* Create a magnetron context with the selected compute device. */
mag_context_t *mag_ctx_create(mag_device_type_t device) {
    const mag_device_desc_t info = {device};
    return mag_ctx_create2(&info);
}

/* Create context with compute device descriptor. */
mag_context_t *mag_ctx_create2(const mag_device_desc_t *device_info) {
    mag_log_info("Creating magnetron context...");

    uint64_t time_stamp_start = mag_hpc_clock_ns();
    mag_ctx_dump_compiler_info(); /* Dump compiler info. */

    /* Initialize context with default values or from context info. */
    mag_context_t *ctx = (*mag_alloc)(NULL, sizeof(*ctx), 0); /* Allocate context. */
    memset(ctx, 0, sizeof(*ctx));

    /* Init memory pools */
    mag_fixed_pool_init(&ctx->tensor_pool, sizeof(mag_tensor_t), __alignof(mag_tensor_t), 0x1000);
    mag_fixed_pool_init(&ctx->storage_pool, sizeof(mag_storage_buffer_t), __alignof(mag_storage_buffer_t), 0x1000);
    mag_fixed_pool_init(&ctx->view_meta_pool, sizeof(mag_view_meta_t), __alignof(mag_view_meta_t), 0x1000);
    mag_fixed_pool_init(&ctx->au_state_pool, sizeof(mag_au_state_t), __alignof(mag_au_state_t), 0x1000);

    ctx->tr_id = mag_thread_id(); /* Get thread ID. */
    ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER; /* Enable gradient recording by default. */

    /* Query and print host system information. */
    mag_machine_probe(ctx);
    mag_system_host_info_dump(ctx);

    /* Create selected compute device. */
    ctx->device_type = device_info->type;
    ctx->backend_registry = mag_backend_registry_init(ctx);
    const char** backend_paths = NULL;
    size_t num_backend_paths = 0;
    mag_backend_registry_get_search_paths(ctx->backend_registry, &backend_paths, &num_backend_paths);
    mag_assert(mag_backend_registry_scan(ctx->backend_registry),
        "\nNo magnetron compute backends found!"
        "\nBackends are loaded dynamically as shared libraries in the directory containing the magnetron_core library, but none were found."
        "\nMake sure you have at least one backend (e.g. magnetron_cpu) next to the magnetron_core library within the venv or installation path."
        "\nThe backend shared library must be named magnetron_<backend>.{so|dylib|dll}"
        "\nCheck the searched path manually to see if any backends were found: %s",
        num_backend_paths && backend_paths && *backend_paths && **backend_paths ? *backend_paths : "(no paths)"
    );
    ctx->backend = mag_backend_registry_best_backend(ctx->backend_registry);
    mag_assert2(ctx->backend);
    ctx->device = (*ctx->backend->init_device)(ctx->backend, ctx, (*ctx->backend->best_device_idx)(ctx->backend));
    mag_assert2(ctx->device);

    /* Seed prng once with secure system entropy */
    uint64_t global_seed = 0;
    if (mag_unlikely(!mag_sec_crypto_entropy(&global_seed, sizeof(global_seed)))) /* Fallback to weak seeding */
        global_seed = (uint64_t)time(NULL)^ctx->tr_id^((uintptr_t)ctx>>3)^mag_cycles()^((uintptr_t)&global_seed>>3);
    mag_ctx_manual_seed(ctx, global_seed);

    /* Print context initialization time. */
    mag_log_info("magnetron context initialized in %.05f ms", mag_hpc_clock_elapsed_ms(time_stamp_start));
    return ctx;
}

void mag_ctx_destroy(mag_context_t *ctx, bool suppress_leak_detection) { /* Destroy magnetron context. */
#ifdef MAG_DEBUG
    mag_leak_detector_dump_results(ctx);  /* Provide detailed leak check info */
#endif
    bool leaks_detected = ctx->num_tensors || ctx->num_storages;
    if (mag_unlikely(leaks_detected)) {
        char msg[256] = {0};
        snprintf(msg, sizeof(msg), "magnetron context destroyed with %zu leaked tensors and %zu leaked storages", ctx->num_tensors, ctx->num_storages);
        if (suppress_leak_detection) mag_log_warn("%s", msg);
        else mag_panic("%s", msg);
    }
    mag_fixed_pool_destroy(&ctx->au_state_pool);
    mag_fixed_pool_destroy(&ctx->view_meta_pool);
    mag_fixed_pool_destroy(&ctx->tensor_pool);
    mag_fixed_pool_destroy(&ctx->storage_pool);
    (*ctx->backend->destroy_device)(ctx->backend, ctx->device);
    ctx->device = NULL;
    ctx->backend = NULL;
    mag_backend_registry_free(ctx->backend_registry);
    memset(ctx, 255, sizeof(*ctx)); /* Poison context memory range. */
    (*mag_alloc)(ctx, 0, 0); /* Free ctx. */
    ctx = NULL;
    mag_log_info("magnetron context destroyed");
}

mag_device_type_t mag_ctx_get_compute_device_type(const mag_context_t *ctx) {
    return ctx->device_type;
}
const char *mag_ctx_get_compute_device_name(const mag_context_t *ctx) {
    return ctx->device->name;
}
const char *mag_ctx_get_os_name(const mag_context_t *ctx) {
    return ctx->machine.os_name;
}
const char *mag_ctx_get_cpu_name(const mag_context_t *ctx) {
    return ctx->machine.cpu_name;
}
uint32_t mag_ctx_get_cpu_virtual_cores(const mag_context_t *ctx) {
    return ctx->machine.cpu_virtual_cores;
}
uint32_t mag_ctx_get_cpu_physical_cores(const mag_context_t *ctx) {
    return ctx->machine.cpu_physical_cores;
}
uint32_t mag_ctx_get_cpu_sockets(const mag_context_t *ctx) {
    return ctx->machine.cpu_sockets;
}
uint64_t mag_ctx_get_physical_memory_total(const mag_context_t *ctx) {
    return ctx->machine.phys_mem_total;
}
uint64_t mag_ctx_get_physical_memory_free(const mag_context_t *ctx) {
    return ctx->machine.phys_mem_free;
}
bool mag_ctx_is_numa_system(const mag_context_t *ctx) {
    return false; /* TODO */
}
size_t mag_ctx_get_total_tensors_created(const mag_context_t *ctx) {
    return 0; /* TODO */
}

void mag_ctx_grad_recorder_start(mag_context_t *ctx) {
    ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER;
}
void mag_ctx_grad_recorder_stop(mag_context_t *ctx) {
    ctx->flags &= ~MAG_CTX_FLAG_GRAD_RECORDER;
}
bool mag_ctx_grad_recorder_is_running(const mag_context_t *ctx) {
    return ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER;
}
void mag_ctx_manual_seed(mag_context_t *ctx, uint64_t seed) {
    (*ctx->device->manual_seed)(ctx->device, seed);
}

const mag_dtype_meta_t *mag_dtype_meta_of(mag_dtype_t type) {
    static const mag_dtype_meta_t infos[MAG_DTYPE__NUM] = {
        [MAG_DTYPE_E8M23] = {
            .name="e8m23",
            .size=sizeof(mag_e8m23_t),
            .align=__alignof(mag_e8m23_t),
        },
        [MAG_DTYPE_E5M10] = {
            .name="e5m10",
            .size=sizeof(mag_e5m10_t),
            .align=__alignof(mag_e5m10_t),
        },
        [MAG_DTYPE_BOOL] = {
            .name="bool",
            .size=sizeof(uint8_t),
            .align=__alignof(uint8_t),
        },
        [MAG_DTYPE_I32] = {
            .name="i32",
            .size=sizeof(int32_t),
            .align=__alignof(int32_t),
        },
    };
    return &infos[type];
}
