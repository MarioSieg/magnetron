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
#include "mag_float16.h"
#include "mag_bfloat16.h"

#include <time.h>
#include <ctype.h>

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
        (double)ctx->machine.cpu_l1_size/1024.0,
        (double) ctx->machine.cpu_l2_size/1024.0,
        (double)ctx->machine.cpu_l3_size/1024.0/1024.0
    );
#if defined(__x86_64__) || defined(_M_X64) /* Print detected CPU features for x86-64 platforms. */
    if (mag_log_level() >= MAG_LOG_LEVEL_INFO) {
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
    if (mag_log_level() >= MAG_LOG_LEVEL_INFO) {
        printf(MAG_CC_CYAN "[magnetron] " MAG_CC_RESET "%s caps: ", cpu_arch);
        for (uint32_t i=0; i < MAG_ARM64_CAP__NUM; ++i)
            if (ctx->machine.arm64_cpu_caps & (1ull<<i))
                printf("%s ", mag_arm64_cpu_cap_names[i]);
        putchar('\n');
    }
#endif
    /* Now print memory information. */
    double mem_total, mem_free, mem_used;
    const char *mem_unit_total, *mem_unit_free, *mem_unit_used;
    mag_humanize_memory_size(ctx->machine.phys_mem_total, &mem_total, &mem_unit_total);
    mag_humanize_memory_size(ctx->machine.phys_mem_free, &mem_free, &mem_unit_free);
    mag_humanize_memory_size((size_t)llabs((int64_t)ctx->machine.phys_mem_total-(int64_t)ctx->machine.phys_mem_free), &mem_used, &mem_unit_used);
    double mem_used_percent = fabs((double)(ctx->machine.phys_mem_total-ctx->machine.phys_mem_free))/(double)ctx->machine.phys_mem_total*100.0;
    mag_log_info("Physical Machine Memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);
}

static void mag_setup_environ(void) {
    /* Parse MAG_LOG_LEVEL environment variable. */
    const char *v = getenv("MAG_LOG_LEVEL");
    if (!v || !*v) return;
    if (mag_casecmp(v, "off")) mag_set_log_level(MAG_LOG_LEVEL_NONE);
    else if (mag_casecmp(v, "error")) mag_set_log_level(MAG_LOG_LEVEL_ERROR);
    else if (mag_casecmp(v, "warn") || mag_casecmp(v, "warning")) mag_set_log_level(MAG_LOG_LEVEL_WARN);
    else if (mag_casecmp(v, "info")) mag_set_log_level(MAG_LOG_LEVEL_INFO);
    else if (mag_casecmp(v, "debug")) mag_set_log_level(MAG_LOG_LEVEL_DEBUG);
    else mag_log_error("Invalid MAG_LOG_LEVEL value '%s' (valid: off, error, warn, info)", v);
}

/* Print compiler information such as name, version and build time. */
static void mag_ctx_dump_banner(void) {
    const char *compiler_name = "Unknown";
    int cmaj = 0, cmin = 0, cpatch = 0;
#if defined(__clang__)
    compiler_name = "Clang";
    cmaj = __clang_major__;
    cmin = __clang_minor__;
    cpatch = __clang_patchlevel__;
#elif defined(__GNUC__)
    compiler_name = "GCC";
    cmaj = __GNUC__;
    cmin = __GNUC_MINOR__;
    cpatch = __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
    compiler_name = "MSVC";
    cmaj = _MSC_VER/100;
    cmin = _MSC_VER%100;
#endif
    mag_log_info("------------------------------------------------------------");
    mag_log_info("Magnetron");
    mag_log_info("Version        : v.%d.%d.%d (storage v.%d.%d.%d)",
        mag_ver_major(MAG_VERSION),
        mag_ver_minor(MAG_VERSION),
        mag_ver_patch(MAG_VERSION),
        mag_ver_major(MAG_SNAPSHOT_VERSION),
        mag_ver_minor(MAG_SNAPSHOT_VERSION),
        mag_ver_patch(MAG_SNAPSHOT_VERSION)
    );
    mag_log_info("Copyright      : (c) 2024–2026 Mario Sieg");
    mag_log_info("License        : Apache-2.0");
    mag_log_info("Source         : https://github.com/MarioSieg/magnetron");
    mag_log_info("Build          : " __DATE__ " " __TIME__);
    mag_log_info("Compiler       : %s %d.%d.%d", compiler_name, cmaj, cmin, cpatch);
    mag_log_info("------------------------------------------------------------");
}

/* Create context with compute device descriptor. */
mag_context_t *mag_ctx_create(void) {
    mag_setup_environ(); /* Parse and apply environment variables. */

    mag_log_info("Creating magnetron context...");

    uint64_t time_stamp_start = mag_hpc_clock_ns();
    mag_ctx_dump_banner();

    /* Initialize context with default values or from context info. */
    mag_context_t *ctx = (*mag_alloc)(NULL, sizeof(*ctx), 0); /* Allocate context. */
    memset(ctx, 0, sizeof(*ctx));

    /* Init memory pools */
    mag_slab_init(&ctx->tensor_slab, sizeof(mag_tensor_t), __alignof(mag_tensor_t), 0x1000);
    mag_slab_init(&ctx->storage_slab, sizeof(mag_storage_buffer_t), __alignof(mag_storage_buffer_t), 0x1000);
    mag_slab_init(&ctx->view_meta_slab, sizeof(mag_view_meta_t), __alignof(mag_view_meta_t), 0x1000);
    mag_slab_init(&ctx->au_state_slab, sizeof(mag_au_state_t), __alignof(mag_au_state_t), 0x1000);

    ctx->tr_id = mag_thread_id(); /* Get thread ID. */
    ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER; /* Enable gradient recording by default. */

    /* Query and print host system information. */
    mag_machine_info_probe(&ctx->machine);
    mag_system_host_info_dump(ctx);

    /* Create selected compute device. */
    ctx->backend_registry = mag_backend_registry_init(ctx);
    const char** backend_paths = NULL;
    size_t num_backend_paths = 0;
    mag_assert(mag_backend_registry_load_all_available(ctx->backend_registry),
        "\nNo magnetron compute backends found!"
        "\nBackends are loaded dynamically as shared libraries in the directory containing the magnetron_core library, but none were found."
        "\nMake sure you have at least one backend (e.g. magnetron_cpu) next to the magnetron_core library within the venv or installation path."
        "\nThe backend shared library must be named magnetron_<backend>.{so|dylib|dll}"
        "\nCheck the searched path manually to see if any backends were found: %s",
        num_backend_paths && backend_paths && *backend_paths && **backend_paths ? *backend_paths : "(no paths)"
    );
    mag_device_id_t cpu_dvc = MAG_DEVICE_ID_CPU; /* TODO: maybe allow users to specify a preferred default device type via env var or context info struct in the future? */
    ctx->backend = mag_backend_registry_get_by_device_id(ctx->backend_registry, &ctx->active_device, &cpu_dvc);
    if (mag_unlikely(!ctx->backend || !ctx->active_device)) {
        mag_log_error(
            "\nNo suitable magnetron compute backend found for device id %s:%u !"
           "\nMake sure the specified device id is correct and that the corresponding backend is available.",
           mag_backend_type_to_str(cpu_dvc.type), cpu_dvc.device_ordinal
        );
        cpu_dvc = MAG_DEVICE_ID_CPU;
        ctx->backend = mag_backend_registry_get_by_device_id(ctx->backend_registry, &ctx->active_device, &cpu_dvc);
        mag_assert(ctx->backend && ctx->active_device,
            "\nFailed to initialize fallback CPU compute backend!"
            "\nMake sure the magnetron_cpu backend is available next to the magnetron_core library."
        );
    }

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
    bool leaks_detected = ctx->telemetry.num_alive_tensors || ctx->telemetry.num_alive_storages;
    if (mag_unlikely(leaks_detected)) {
        char msg[256] = {0};
        snprintf(msg, sizeof(msg), "magnetron context destroyed with %zu leaked tensors and %zu leaked storages", ctx->telemetry.num_alive_tensors, ctx->telemetry.num_alive_storages);
        if (suppress_leak_detection) mag_log_warn("%s", msg);
        else mag_panic("%s", msg);
    }
    mag_slab_destroy(&ctx->au_state_slab);
    mag_slab_destroy(&ctx->view_meta_slab);
    mag_slab_destroy(&ctx->tensor_slab);
    mag_slab_destroy(&ctx->storage_slab);
    ctx->active_device = NULL;
    ctx->backend = NULL;
    mag_backend_registry_free(ctx->backend_registry);
    size_t num_created_tensors = ctx->telemetry.num_created_tensors;
    size_t storage_bytes = ctx->telemetry.storage_bytes_allocated;
    size_t ops_dispatched = ctx->telemetry.ops_dispatched;
    memset(ctx, 255, sizeof(*ctx)); /* Poison context memory range. */
    (*mag_alloc)(ctx, 0, 0); /* Free ctx. */
    ctx = NULL;
    mag_log_info(
        "runtime metrics: ops: %zu, tensors: %zuK, storage alloc: %.02fGiB",
        ops_dispatched/1000,
        num_created_tensors/1000,
        (double)storage_bytes / (double)(1<<30)
    );
    mag_log_info("magnetron context offline");
    fflush(stdout);
    fflush(stderr);
}

const mag_error_t *mag_ctx_get_last_error(const mag_context_t *ctx) {
    return &ctx->error_status;
}

void mag_ctx_set_last_error(mag_context_t *ctx, const mag_error_t *error){
    ctx->error_status = *error;
}

mag_status_t mag_ctx_get_last_error_code(const mag_context_t *ctx) {
    return ctx->error_status.code;
}

void mag_ctx_clear_last_error(mag_context_t *ctx) {
    memset(&ctx->error_status, 0, sizeof(ctx->error_status));
    ctx->error_status.code = MAG_STATUS_OK;
}

void mag_ctx_take_last_error(mag_context_t *ctx, mag_error_t *err){
    *err = ctx->error_status;
    mag_ctx_clear_last_error(ctx);
}

bool mag_ctx_has_error(const mag_context_t *ctx){
    return ctx->error_status.code != MAG_STATUS_OK;
}

const char *mag_ctx_get_compute_device_name(const mag_context_t *ctx) {
    return ctx->active_device->physical_device_name;
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
    (*ctx->active_device->manual_seed)(ctx->active_device, seed);
}

const mag_type_traits_t *mag_type_trait(mag_dtype_t type) {
    static const mag_type_traits_t infos[MAG_DTYPE__NUM] = {
        [MAG_DTYPE_FLOAT32] = {
            .name="float32",
            .short_name="f32",
            .size=sizeof(float),
            .align=__alignof(float),
        },
        [MAG_DTYPE_FLOAT16] = {
            .name="float16",
            .short_name="f16",
            .size=sizeof(mag_float16_t),
            .align=__alignof(mag_float16_t),
        },
        [MAG_DTYPE_BFLOAT16] = {
            .name="bfloat16",
            .short_name="bf16",
            .size=sizeof(mag_bfloat16_t),
            .align=__alignof(mag_bfloat16_t),
        },
        [MAG_DTYPE_BOOLEAN] = {
            .name="boolean",
            .short_name="b8",
            .size=sizeof(uint8_t),
            .align=__alignof(uint8_t),
        },
        [MAG_DTYPE_UINT8] = {
            .name="uint8",
            .short_name="u8",
            .size=sizeof(uint8_t),
            .align=__alignof(uint8_t),
        },
        [MAG_DTYPE_INT8] = {
            .name="int8",
            .short_name="i8",
            .size=sizeof(int8_t),
            .align=__alignof(int8_t),
        },
        [MAG_DTYPE_UINT16] = {
            .name="uint16",
            .short_name="u16",
            .size=sizeof(uint16_t),
            .align=__alignof(uint16_t),
        },
        [MAG_DTYPE_INT16] = {
            .name="int16",
            .short_name="i16",
            .size=sizeof(int16_t),
            .align=__alignof(int16_t),
        },
        [MAG_DTYPE_UINT32] = {
            .name="uint32",
            .short_name="u32",
            .size=sizeof(uint32_t),
            .align=__alignof(uint32_t),
        },
        [MAG_DTYPE_INT32] = {
            .name="int32",
            .short_name="i32",
            .size=sizeof(int32_t),
            .align=__alignof(int32_t),
        },
        [MAG_DTYPE_UINT64] = {
            .name="int64",
            .short_name="u64",
            .size=sizeof(uint64_t),
            .align=__alignof(uint64_t),
        },
        [MAG_DTYPE_INT64] = {
            .name="int64",
            .short_name="i64",
            .size=sizeof(int64_t),
            .align=__alignof(int64_t),
        },
    };
    return &infos[type];
}
