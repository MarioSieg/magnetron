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
#include "mag_envcfg.h"
#include "mag_tensor.h"
#include "mag_autodiff.h"
#include "mag_machine.h"

#include <time.h>
#include <ctype.h>

/* Print host system and machine information. */
static void mag_system_host_info_dump(mag_context_t *ctx) {
  mag_log_info("OS/Kernel: %s", ctx->machine.os_name);
  mag_log_info(
    "CPU: %s, Virtual Cores: %u, Performance Virtual Cores: %u, Physical Cores: %u, Sockets: %u, L1D: %.01f KiB, L2: %.01f KiB, L3: %.01f MiB",
    ctx->machine.cpu_name,
    ctx->machine.cpu_virtual_cores,
    ctx->machine.cpu_perf_virtual_cores,
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

MAG_THREAD_LOCAL mag_tls_state_t mag_tls_state;

/* Create context with compute device descriptor. */
mag_status_t mag_ctx_create(mag_error_t *err, mag_context_t **out_ctx) {
  if (mag_unlikely(!out_ctx))
    return mag_set_error(err, MAG_ERR_PARAM, "context: out_ctx pointer is NULL.");
  *out_ctx = NULL;

  mag_envcfg_apply_log_level(); /* Parse and apply environment variables, see mag_envcfg.h */

  mag_log_info("Creating magnetron context...");

  uint64_t time_stamp_start = mag_hpc_clock_ns();
  mag_ctx_dump_banner();

  /* Initialize context with default values or from context info. */
  mag_context_t *ctx = (*mag_try_alloc)(NULL, sizeof(*ctx), 0); /* Allocate context. */
  if (mag_unlikely(!ctx))
    return mag_set_error(err, MAG_ERR_OOM, "context: failed to allocate context structure.");
  memset(ctx, 0, sizeof(*ctx));
  ctx->boot_timestamp_ns = time_stamp_start;

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

  mag_atomic64_store(&ctx->topo_traversal_epoch, 0, MAG_MO_RELAXED);

  /* Query and print host system information. */
  mag_machine_info_probe(&ctx->machine);
  mag_system_host_info_dump(ctx);

  /* Create compute backends and devices. On failure, err carries reason. */
  mag_status_t status = mag_backend_registry_init(err, ctx, &ctx->backend_registry);
  if (mag_unlikely(mag_iserr(status))) {
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
  if (mag_unlikely(!mag_query_crypto_entropy(&global_seed, sizeof(global_seed)))) /* Fallback to weak seeding */
    global_seed = (uint64_t)time(NULL)^mag_thread_id()^((uintptr_t)ctx>>3)^mag_cycles()^((uintptr_t)&global_seed>>3);
  mag_ctx_manual_seed(ctx, global_seed);

  /* Print context initialization time. */
  mag_log_info("context: magnetron initialized in %.05f ms.", mag_hpc_clock_elapsed_ms(time_stamp_start));
  *out_ctx = ctx;
  return MAG_OK;
}

bool mag_ctx_is_device_available(mag_context_t *ctx, mag_device_id_t id) {
  mag_device_t *device=NULL;
  mag_backend_t *backend=NULL;
  return mag_backend_registry_lookup_device_id(ctx->backend_registry, id, &backend, &device) && device && backend;
}

typedef struct mag_slab_stats_t {
  const char *name;
  uint32_t num_allocs;
  uint32_t num_freelist_hits;
  uint32_t num_pool_hits;
  uint32_t num_chunks;
  size_t blocks_per_chunk;
  size_t capacity_bytes;
} mag_slab_stats_t;

static mag_slab_stats_t mag_slab_snapshot(const mag_slab_alloc_t *slab, const char *name) {
  return (mag_slab_stats_t) {
    .name = name,
    .num_allocs = slab->num_allocs,
    .num_freelist_hits = slab->num_freelist_hits,
    .num_pool_hits = slab->num_pool_hits,
    .num_chunks = slab->num_chunks,
    .blocks_per_chunk = slab->blocks_per_chunk,
    .capacity_bytes = (size_t)slab->num_chunks*slab->blocks_per_chunk*slab->block_size
  };
}

static void mag_fmt_duration(char *buf, size_t n, uint64_t ns) {
  uint64_t whole = ns/1000000000ull;
  uint64_t days = whole/86400, hours = whole%86400/3600, mins = whole%3600/60;
  size_t off = (size_t)snprintf(buf, n, "up ");
  if (days) off += (size_t)snprintf(buf+off, n-off, "%" PRIu64 " day%s, ", days, days == 1 ? "" : "s");
  if (hours) snprintf(buf+off, n-off, "%" PRIu64 ":%02" PRIu64, hours, mins);
  else if (mins || days) snprintf(buf+off, n-off, "%" PRIu64 " min", mins);
  else snprintf(buf+off, n-off, "%" PRIu64 " sec", whole);
}

void mag_ctx_destroy(mag_context_t *ctx, bool suppress_leak_detection) { /* Destroy magnetron context. */
#ifdef MAG_DEBUG
  mag_leak_detector_dump_results(ctx);  /* Provide detailed leak check info */
#endif
  uint64_t uptime_ns = mag_hpc_clock_elapsed_ns(ctx->boot_timestamp_ns);
  int64_t alive_tensors = mag_atomic64_load(&ctx->telemetry.num_alive_tensors, MAG_MO_RELAXED);
  int64_t alive_storages = mag_atomic64_load(&ctx->telemetry.num_alive_storages, MAG_MO_RELAXED);
  bool leaks_detected = alive_tensors || alive_storages;
  if (mag_unlikely(leaks_detected)) {
    char msg[256] = {0};
    snprintf(msg, sizeof(msg), "context: destroyed with %" PRIi64 " leaked tensors and %" PRIi64 " leaked storage buffers.", alive_tensors, alive_storages);
    if (suppress_leak_detection) mag_log_warn("%s", msg);
    else mag_log_error("%s", msg); /* Never abort from Python - report the leak instead of panicking. */
  }
  mag_slab_stats_t slabs[] = {
    mag_slab_snapshot(&ctx->tensor_slab, "tensor"),
    mag_slab_snapshot(&ctx->storage_slab, "storage"),
    mag_slab_snapshot(&ctx->view_meta_slab, "view_meta"),
    mag_slab_snapshot(&ctx->au_state_slab, "au_state"),
    mag_slab_snapshot(&ctx->au_state_op_params_slab, "au_op_params"),
  };
  mag_slab_destroy(&ctx->au_state_op_params_slab);
  mag_slab_destroy(&ctx->au_state_slab);
  mag_slab_destroy(&ctx->view_meta_slab);
  mag_slab_destroy(&ctx->tensor_slab);
  mag_slab_destroy(&ctx->storage_slab);
  mag_backend_registry_shutdown(NULL, ctx->backend_registry); /* TODO: propagate error */
  int64_t num_created_tensors = mag_atomic64_load(&ctx->telemetry.num_created_tensors, MAG_MO_RELAXED);
  int64_t num_created_views = mag_atomic64_load(&ctx->telemetry.num_created_views, MAG_MO_RELAXED);
  int64_t storage_bytes = mag_atomic64_load(&ctx->telemetry.storage_bytes_allocated, MAG_MO_RELAXED);
  int64_t ops_dispatched = mag_atomic64_load(&ctx->telemetry.ops_dispatched, MAG_MO_RELAXED);
  int64_t backward_passes = mag_atomic64_load(&ctx->telemetry.backward_passes, MAG_MO_RELAXED);
  int64_t backward_nodes = mag_atomic64_load(&ctx->telemetry.backward_nodes_visited, MAG_MO_RELAXED);
  int64_t grads_materialized = mag_atomic64_load(&ctx->telemetry.grads_materialized, MAG_MO_RELAXED);
  uint32_t cpu_workers = ctx->telemetry.cpu_workers;
  memset(ctx, 255, sizeof(*ctx)); /* Poison context memory range. */
  (*mag_alloc)(ctx, 0, 0); /* Free ctx. */
  ctx = NULL;
  double uptime_s = (double)uptime_ns*1e-9;
  double rate_div = uptime_s > 0.0 ? uptime_s : 1.0;
  char uptime_str[64];
  mag_fmt_duration(uptime_str, sizeof(uptime_str), uptime_ns);
  double storage_alloc, tensors_num, views_num, ops_num, ops_rate, tensors_rate, bwd_num, bwd_nodes_num, grads_num;
  const char *storage_unit, *tensors_unit, *views_unit, *ops_unit, *ops_rate_unit, *tensors_rate_unit, *bwd_unit, *bwd_nodes_unit, *grads_unit;
  mag_humanize_memory_size(storage_bytes, &storage_alloc, &storage_unit);
  mag_humanize_amount(num_created_tensors, &tensors_num, &tensors_unit);
  mag_humanize_amount(num_created_views, &views_num, &views_unit);
  double views_pct = num_created_tensors ? 100.0*(double)num_created_views/(double)num_created_tensors : 0.0;
  mag_humanize_amount(ops_dispatched, &ops_num, &ops_unit);
  mag_humanize_amount((size_t)((double)ops_dispatched/rate_div), &ops_rate, &ops_rate_unit);
  mag_humanize_amount((size_t)((double)num_created_tensors/rate_div), &tensors_rate, &tensors_rate_unit);
  mag_humanize_amount(backward_passes, &bwd_num, &bwd_unit);
  mag_humanize_amount(backward_nodes, &bwd_nodes_num, &bwd_nodes_unit);
  mag_humanize_amount(grads_materialized, &grads_num, &grads_unit);
  double nodes_per_pass = backward_passes ? (double)backward_nodes/(double)backward_passes : 0.0;
  mag_log_info("runtime metrics: %s, cpu workers: %u", uptime_str, cpu_workers);
  mag_log_info(
    "runtime metrics: operators dispatched: %.01f%s (%.01f%s/s), tensors created: %.01f%s (%.01f%s/s), of which views: %.01f%s (%.01f%%), total storage memory allocated: %.01f%s.",
    ops_num, ops_unit, ops_rate, ops_rate_unit,
    tensors_num, tensors_unit, tensors_rate, tensors_rate_unit,
    views_num, views_unit, views_pct,
    storage_alloc, storage_unit
  );
  mag_log_info(
    "runtime metrics: backward passes: %.01f%s, autograd nodes visited: %.01f%s (%.01f/pass), grad tensors materialized: %.01f%s.",
    bwd_num, bwd_unit,
    bwd_nodes_num, bwd_nodes_unit, nodes_per_pass,
    grads_num, grads_unit
  );
  for (size_t i=0; i < sizeof(slabs)/sizeof(*slabs); ++i) {
    const mag_slab_stats_t *st = slabs+i;
    double cap_val, reuse = st->num_allocs ? 100.0*(double)st->num_freelist_hits/(double)st->num_allocs : 0.0;
    const char *cap_unit;
    mag_humanize_memory_size(st->capacity_bytes, &cap_val, &cap_unit);
    mag_log_info(
      "runtime metrics: slab %-12s allocs: %u, freelist hits: %u (%.01f%% reuse), pool hits: %u, chunks: %u x %zu blocks, capacity: %.01f%s",
      st->name, st->num_allocs, st->num_freelist_hits, reuse, st->num_pool_hits, st->num_chunks, st->blocks_per_chunk, cap_val, cap_unit
    );
  }
  mag_log_info("magnetron context offline");
  fflush(stdout);
  fflush(stderr);
}

void mag_ctx_grad_recorder_start(mag_context_t *ctx) {
  (void)ctx;
  mag_tls_state.no_grad = false;
}

void mag_ctx_grad_recorder_stop(mag_context_t *ctx) {
  (void)ctx;
  mag_tls_state.no_grad = true;
}

bool mag_ctx_grad_recorder_is_running(const mag_context_t *ctx) {
  (void)ctx;
  return !mag_tls_state.no_grad;
}

void mag_ctx_manual_seed(mag_context_t *ctx, uint64_t seed) {
  mag_backend_registry_manual_seed(ctx->backend_registry, seed); /* Also replayed onto backends that are loaded lazily later on */
}

mag_device_id_t mag_ctx_default_device(mag_context_t *ctx) {
  (void)ctx;
  return mag_tls_state.device;
}

mag_status_t mag_ctx_set_default_device(mag_error_t *err, mag_context_t *ctx, mag_device_id_t id) {
  if (mag_unlikely(!mag_ctx_is_device_available(ctx, id))) {
    char device_name[32];
    mag_device_id_to_str(id, &device_name);
    return mag_set_error(err, MAG_ERR_DEVICE, "set_default_device: device '%s' is not available.", device_name);
  }
  mag_tls_state.device = id;
  return MAG_OK;
}

mag_status_t mag_ctx_best_device(mag_error_t *err, mag_context_t *ctx, mag_backend_type_t type, mag_device_id_t *out_id) {
  mag_device_t *device = NULL;
  if (mag_unlikely(!mag_backend_registry_best_device(ctx->backend_registry, type, NULL, &device)))
    return mag_set_error(err, MAG_ERR_DEVICE, "best_device: backend '%s' has no usable device.", mag_backend_type_to_str(type));
  *out_id = device->id;
  return MAG_OK;
}

bool mag_ctx_tensors_alive(mag_context_t* ctx) {
  return mag_atomic64_load(&ctx->telemetry.num_alive_tensors, MAG_MO_ACQUIRE) > 0
    || mag_atomic64_load(&ctx->telemetry.num_alive_storages, MAG_MO_ACQUIRE) > 0;
}

mag_dtype_t mag_ctx_default_dtype(mag_context_t *ctx) { /* TODO: maybe remove ctx here */
  (void)ctx;
  return mag_tls_state.dtype;
}

bool mag_ctx_set_default_dtype(mag_context_t *ctx, mag_dtype_t type) { /* TODO: maybe remove ctx here */
  if (!mag_type_category_is_floating_point(type)) {
    mag_log_error("Cannot set default floating point dtype to non-floating point type '%s'", mag_type_trait(type)->name);
    return false;
  }
  mag_tls_state.dtype = type;
  return true;
}
