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

#include "mag_context.h"
#include "mag_alloc.h"
#include "mag_os.h"
#include "mag_tensor.h"
#include "mag_autodiff.h"
#include "mag_machine.h"

#include <time.h>
#include <ctype.h>

/* Print host system and machine information. */
static void mag_system_host_info_dump(mag_context_t *ctx) {
  mag_log_info("OS/Kernel: %s", ctx->machine.os_name);
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
    mag_log_info("AMD64 (x86-64) CPU flags:");
    for (uint32_t i=0, j=0; i < MAG_AMD64_CAP__NUM; ++i) {
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
    mag_log_info("ARM64 (aarch64) CPU flags:");
    for (uint32_t i=0, j=0; i < MAG_ARM64_CAP__NUM; ++i) {
      if (ctx->machine.arm64_cpu_caps & mag_arm64_cap_bit(i)) {
        if (!(j++&7)) printf(j-1 ? "\n\t" : "\t");
        printf("%s ", mag_arm64_cpu_cap_names[i]);
      }

    }
    putchar('\n');
  }
#elif defined(__loongarch64) /* Print detected CPU features for Loongson / Godson */
  if (mag_log_level() >= MAG_LOG_LEVEL_INFO) {
    mag_log_info("LOONGSON (loongarch64) CPU flags:");
    for (uint32_t i=0, j=0; i < MAG_LOONGARCH64_CAP__NUM; ++i) {
      if (ctx->machine.loongarch64_cpu_caps & mag_loongarch64_cap_bit(i)) {
        if (!(j++&7)) printf(j-1 ? "\n\t" : "\t");
        printf("%s ", mag_loongarch64_cpu_cap_names[i]);
      }
    }
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
mag_status_t mag_ctx_create(mag_error_t *err, mag_context_t **out_ctx) {
  if (mag_unlikely(!out_ctx))
    return mag_set_error(err, MAG_ERR_PARAM, "context: out_ctx pointer is NULL.");
  *out_ctx = NULL;

  mag_setup_environ(); /* Parse and apply environment variables. */

  mag_log_info("Creating magnetron context...");

  uint64_t time_stamp_start = mag_hpc_clock_ns();
  mag_ctx_dump_banner();

  /* Initialize context with default values or from context info. */
  mag_context_t *ctx = (*mag_try_alloc)(NULL, sizeof(*ctx), 0); /* Allocate context. */
  if (mag_unlikely(!ctx))
    return mag_set_error(err, MAG_ERR_OOM, "context: failed to allocate context structure.");
  memset(ctx, 0, sizeof(*ctx));

  /* Slab allocators */
  bool slab_ok = true;
  slab_ok &= mag_slab_init(&ctx->tensor_slab, sizeof(mag_tensor_t), __alignof(mag_tensor_t), 0x1000);
  slab_ok &= mag_slab_init(&ctx->storage_slab, sizeof(mag_storage_buffer_t), __alignof(mag_storage_buffer_t), 0x1000);
  slab_ok &= mag_slab_init(&ctx->view_meta_slab, sizeof(mag_view_meta_t), __alignof(mag_view_meta_t), 0x1000);
  slab_ok &= mag_slab_init(&ctx->au_state_slab, sizeof(mag_au_state_t), __alignof(mag_au_state_t), 0x1000);
  slab_ok &= mag_slab_init(&ctx->au_state_op_params_slab, sizeof(mag_op_params_t), __alignof(mag_op_params_t), 0x1000);
  if (mag_unlikely(!slab_ok)) {
    mag_slab_destroy(&ctx->au_state_op_params_slab);
    mag_slab_destroy(&ctx->au_state_slab);
    mag_slab_destroy(&ctx->view_meta_slab);
    mag_slab_destroy(&ctx->tensor_slab);
    mag_slab_destroy(&ctx->storage_slab);
    (*mag_alloc)(ctx, 0, 0);
    return mag_set_error(err, MAG_ERR_OOM, "context: failed to initialize context memory pools.");
  }

  /* Toposort stuff */
  ctx->topo_traversal_epoch = 0;
  if (mag_unlikely(!mag_topo_set_init(&ctx->topo_set, 0x2000) || !mag_topo_stack_init(&ctx->topo_stack, 0x2000))) {
    mag_topo_set_free(&ctx->topo_set);
    mag_topo_stack_free(&ctx->topo_stack);
    mag_slab_destroy(&ctx->au_state_op_params_slab);
    mag_slab_destroy(&ctx->au_state_slab);
    mag_slab_destroy(&ctx->view_meta_slab);
    mag_slab_destroy(&ctx->tensor_slab);
    mag_slab_destroy(&ctx->storage_slab);
    (*mag_alloc)(ctx, 0, 0);
    return mag_set_error(err, MAG_ERR_OOM, "context: failed to initialize toposort structures.");
  }

  ctx->tr_id = mag_thread_id(); /* Get thread ID. */
  ctx->default_dtype = MAG_DTYPE_FLOAT32; /* Use fp32 by default */
  ctx->flags|=MAG_CTX_FLAG_GRAD_RECORDER; /* Enable gradient recording by default. */

  /* Query and print host system information. */
  mag_machine_info_probe(&ctx->machine);
  mag_system_host_info_dump(ctx);

  /* Create compute backends and devices. On failure 'err' carries the specific reason (e.g. missing .so, ABI mismatch, device init failure). */
  mag_status_t status = mag_backend_registry_init(err, ctx, &ctx->backend_registry);
  if (mag_unlikely(mag_iserr(status))) {
    mag_topo_set_free(&ctx->topo_set);
    mag_topo_stack_free(&ctx->topo_stack);
    mag_slab_destroy(&ctx->au_state_op_params_slab);
    mag_slab_destroy(&ctx->au_state_slab);
    mag_slab_destroy(&ctx->view_meta_slab);
    mag_slab_destroy(&ctx->tensor_slab);
    mag_slab_destroy(&ctx->storage_slab);
    (*mag_alloc)(ctx, 0, 0); /* Free ctx. */
    return status;
  }

  /* Seed prng once with secure system entropy */
  uint64_t global_seed = 0;
  if (mag_unlikely(!mag_sec_crypto_entropy(&global_seed, sizeof(global_seed)))) /* Fallback to weak seeding */
    global_seed = (uint64_t)time(NULL)^ctx->tr_id^((uintptr_t)ctx>>3)^mag_cycles()^((uintptr_t)&global_seed>>3);
  mag_ctx_manual_seed(ctx, global_seed);

  /* Print context initialization time. */
  mag_log_info("magnetron context initialized in %.05f ms", mag_hpc_clock_elapsed_ms(time_stamp_start));
  *out_ctx = ctx;
  return MAG_OK;
}

bool mag_ctx_is_device_available(mag_context_t *ctx, mag_device_id_t id) {
  mag_device_t *device=NULL;
  mag_backend_t *backend=NULL;
  return mag_backend_registry_lookup_device_id(ctx->backend_registry, id, &backend, &device) && device && backend;
}

void mag_ctx_destroy(mag_context_t *ctx, bool suppress_leak_detection) { /* Destroy magnetron context. */
#ifdef MAG_DEBUG
  mag_leak_detector_dump_results(ctx);  /* Provide detailed leak check info */
#endif
  bool leaks_detected = ctx->telemetry.num_alive_tensors || ctx->telemetry.num_alive_storages;
  if (mag_unlikely(leaks_detected)) {
    char msg[256] = {0};
    snprintf(msg, sizeof(msg), "context: destroyed with %zu leaked tensors and %zu leaked storage buffers.", ctx->telemetry.num_alive_tensors, ctx->telemetry.num_alive_storages);
    if (suppress_leak_detection) mag_log_warn("%s", msg);
    else mag_log_error("%s", msg); /* Never abort from Python - report the leak instead of panicking. */
  }
  mag_topo_set_free(&ctx->topo_set);
  mag_topo_stack_free(&ctx->topo_stack);
  mag_slab_destroy(&ctx->au_state_op_params_slab);
  mag_slab_destroy(&ctx->au_state_slab);
  mag_slab_destroy(&ctx->view_meta_slab);
  mag_slab_destroy(&ctx->tensor_slab);
  mag_slab_destroy(&ctx->storage_slab);
  mag_backend_registry_shutdown(NULL, ctx->backend_registry); /* TODO: propagate error */
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

void mag_ctx_grad_recorder_start(mag_context_t *ctx) {
  ctx->flags |= MAG_CTX_FLAG_GRAD_RECORDER;
}

void mag_ctx_grad_recorder_stop(mag_context_t *ctx) {
  ctx->flags &= ~MAG_CTX_FLAG_GRAD_RECORDER;
}

bool mag_ctx_grad_recorder_is_running(const mag_context_t *ctx) {
  return ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER;
}

static void mag_seed_callback(mag_backend_t *bck, mag_device_t *dvc, void *usr) {
  (*dvc->manual_seed)(NULL, dvc, *(const uint64_t *)usr);
}

void mag_ctx_manual_seed(mag_context_t *ctx, uint64_t seed) {
  mag_backend_registry_iter_devices(ctx->backend_registry, &mag_seed_callback, &seed);
}

mag_dtype_t mag_ctx_default_dtype(mag_context_t *ctx) {
  return ctx->default_dtype;
}

bool mag_ctx_set_default_dtype(mag_context_t *ctx, mag_dtype_t type) {
  if (!mag_type_category_is_floating_point(type)) {
    mag_log_error("Cannot set default floating point dtype to non-floating point type '%s'", mag_type_trait(type)->name);
    return false;
  }
  ctx->default_dtype = type;
  return true;
}
