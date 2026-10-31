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

#include <prelude.hpp>
#include <core/mag_envcfg.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace magnetron;

namespace {
    class env_override final { /* Sets an env var for the lifetime of the object, restores the previous value after */
    public:
        env_override(const char *name, const char *value) : m_name{name} {
            if (const char *prev = std::getenv(name)) m_prev = prev;
            setenv(name, value, 1);
        }
        env_override(const env_override &) = delete;
        auto operator=(const env_override &) -> env_override & = delete;
        ~env_override() {
            if (m_prev) setenv(m_name, m_prev->c_str(), 1);
            else unsetenv(m_name);
        }
    private:
        const char *m_name;
        std::optional<std::string> m_prev {};
    };

    class raw_ctx final {
    public:
        raw_ctx() {
            mag_error_t err {};
            if (mag_iserr(mag_ctx_create(&err, &m_ctx))) throw std::runtime_error {err.message};
        }
        raw_ctx(const raw_ctx &) = delete;
        auto operator=(const raw_ctx &) -> raw_ctx & = delete;
        ~raw_ctx() { mag_ctx_destroy(m_ctx, false); }
        [[nodiscard]] auto operator*() const noexcept -> mag_context_t * { return m_ctx; }
    private:
        mag_context_t *m_ctx {};
    };

    // TODO: ugly
    class ref final { /* Owning tensor reference */
    public:
        ref() noexcept = default;
        explicit ref(mag_tensor_t *t) noexcept : m_t{t} {}
        ref(const ref &) = delete;
        auto operator=(const ref &) -> ref & = delete;
        ref(ref &&o) noexcept : m_t{o.m_t} { o.m_t = nullptr; }
        auto operator=(ref &&o) noexcept -> ref & { if (this != &o) { reset(); m_t = o.m_t; o.m_t = nullptr; } return *this; }
        ~ref() { reset(); }
        auto reset() noexcept -> void { if (m_t) mag_tensor_decref(m_t); m_t = nullptr; }
        [[nodiscard]] auto get() const noexcept -> mag_tensor_t * { return m_t; }
        [[nodiscard]] auto out() noexcept -> mag_tensor_t ** { reset(); return &m_t; }
    private:
        mag_tensor_t *m_t {};
    };

    auto check(mag_status_t stat, const mag_error_t &err) -> void {
        ASSERT_EQ(stat, MAG_OK) << err.message;
    }

    auto cpu_device(mag_context_t *ctx) -> mag_device_t * {
        mag_device_t *dvc {};
        mag_backend_registry_best_device(ctx->backend_registry, MAG_BACKEND_TYPE_CPU, nullptr, &dvc);
        return dvc;
    }

    auto full(mag_context_t *ctx, int64_t n, float v) -> ref {
        mag_error_t err {};
        ref r {};
        check(mag_full(&err, r.out(), ctx, MAG_DTYPE_FLOAT32, 1, &n, mag_scalar_from_float64(v), mag_device(CPU, 0)), err);
        return r;
    }

    auto scalar(mag_context_t *ctx, float v) -> ref {
        mag_error_t err {};
        ref r {};
        check(mag_scalar(&err, r.out(), ctx, MAG_DTYPE_FLOAT32, mag_scalar_from_float64(v), mag_device(CPU, 0)), err);
        return r;
    }

    auto sum_item(mag_tensor_t *x) -> double {
        mag_error_t err {};
        ref s {};
        check(mag_sum(&err, s.out(), x, nullptr, 0, false), err);
        mag_scalar_t v {};
        check(mag_tensor_item(&err, s.get(), &v), err);
        return mag_scalar_as_float64(v);
    }
}

TEST(async_stream, inplace_chain_executes_in_order) {
    raw_ctx ctx {};
    constexpr int64_t n {4096};
    constexpr int iters {2000};
    ref x {full(*ctx, n, 0.0f)};
    ref one {scalar(*ctx, 1.0f)};
    mag_error_t err {};
    for (int i=0; i < iters; ++i) {
        ref out {};
        check(mag_add_(&err, out.out(), x.get(), one.get()), err);
    }
    ASSERT_EQ(sum_item(x.get()), static_cast<double>(n*iters)); /* item() waits for the last writer (RAW) */
}

TEST(async_stream, host_write_waits_for_queued_readers) {
    raw_ctx ctx {};
    constexpr int64_t n {1<<16};
    ref x {full(*ctx, n, 1.0f)};
    ref two {scalar(*ctx, 2.0f)};
    mag_error_t err {};
    std::vector<ref> readers {};
    for (int i=0; i < 64; ++i) { /* Queue many commands reading x */
        ref y {};
        check(mag_mul(&err, y.out(), x.get(), two.get()), err);
        readers.emplace_back(std::move(y));
    }
    std::vector<float> fives(n, 5.0f);
    check(mag_copy_raw_(&err, x.get(), fives.data(), fives.size()*sizeof(float)), err); /* Must not overwrite x before the readers ran (WAR) */
    for (const ref &y : readers)
        ASSERT_EQ(sum_item(y.get()), 2.0*n);
    ASSERT_EQ(sum_item(x.get()), 5.0*n);
}

TEST(async_stream, deferred_error_is_reported_once_and_clears) {
    raw_ctx ctx {};
    mag_error_t err {};
    ref src {full(*ctx, 10, 1.0f)};
    ref idx {}; /* Larger than MAG_CPU_INLINE_MAX_NUMEL, so the gather is always queued and never runs inline */
    int64_t n_idx {1<<15};
    check(mag_full(&err, idx.out(), *ctx, MAG_DTYPE_INT64, 1, &n_idx, mag_scalar_from_int64(100), mag_device(CPU, 0)), err);
    ref g {};
    mag_status_t stat {mag_gather(&err, g.out(), src.get(), 0, idx.get())};
    if (!(cpu_device(*ctx)->flags&MAG_DEVICE_FLAGS_ASYNC) || mag_envcfg_launch_blocking()) { /* Synchronous and launch blocking mode report at the call */
        ASSERT_EQ(stat, MAG_ERR_BOUNDS);
        return;
    }
    ASSERT_EQ(stat, MAG_OK) << "async submit must not execute the kernel";
    mag_error_t err2 {};
    ref h {};
    mag_status_t submit_stat {mag_add(&err2, h.out(), src.get(), src.get())}; /* May already see the poisoned stream */
    mag_error_t sync_err {};
    mag_status_t sync_stat {mag_ctx_sync(&sync_err, *ctx)};
    bool reported_at_submit {submit_stat != MAG_OK};
    bool reported_at_sync {sync_stat != MAG_OK};
    ASSERT_NE(reported_at_submit, reported_at_sync) << "the deferred error must be reported exactly once";
    const mag_error_t &e {reported_at_submit ? err2 : sync_err};
    ASSERT_EQ(e.code, MAG_ERR_BOUNDS);
    ASSERT_NE(std::strstr(e.message, "gather"), nullptr) << e.message;
    ASSERT_NE(std::strstr(e.message, "async cpu error"), nullptr) << e.message;
    mag_error_t clean {}; /* Error cleared, the stream works again */
    ASSERT_EQ(mag_ctx_sync(&clean, *ctx), MAG_OK) << clean.message;
    ref k {};
    check(mag_add(&clean, k.out(), src.get(), src.get()), clean);
    ASSERT_EQ(sum_item(k.get()), 20.0);
}

TEST(async_stream, tiny_queue_backpressure) {
    for (const char *depth : {"1", "2", "3"}) {
        env_override async {"MAG_CPU_ASYNC", "1"};
        env_override queue {"MAG_CPU_QUEUE_DEPTH", depth};
        raw_ctx ctx {};
        ASSERT_TRUE(cpu_device(*ctx)->flags&MAG_DEVICE_FLAGS_ASYNC);
        ref x {full(*ctx, 1<<14, 0.0f)};
        ref one {scalar(*ctx, 1.0f)};
        mag_error_t err {};
        for (int i=0; i < 500; ++i) {
            ref out {};
            check(mag_add_(&err, out.out(), x.get(), one.get()), err);
        }
        ASSERT_EQ(sum_item(x.get()), 500.0*(1<<14)) << "depth " << depth;
    }
}

TEST(async_stream, shutdown_with_pending_work) {
    env_override async {"MAG_CPU_ASYNC", "1"};
    env_override queue {"MAG_CPU_QUEUE_DEPTH", "8"};
    raw_ctx ctx {};
    mag_error_t err {};
    ref x {full(*ctx, 1<<20, 1.0f)};
    for (int i=0; i < 100; ++i) {
        ref out {};
        check(mag_mul_(&err, out.out(), x.get(), x.get()), err);
    }
    /* raw_ctx destroys the context with commands still queued: it must drain them first */
}

TEST(async_stream, manual_seed_is_ordered_with_rng_ops) {
    raw_ctx ctx {};
    constexpr int64_t n {1<<16};
    auto draw = [&] {
        mag_error_t err {};
        mag_ctx_manual_seed(*ctx, 0x1234);
        ref u {};
        check(mag_uniform(&err, u.out(), *ctx, MAG_DTYPE_FLOAT32, 1, &n, mag_scalar_from_float64(0.0), mag_scalar_from_float64(1.0), mag_device(CPU, 0)), err);
        return u;
    };
    ref a {draw()};
    ref b {draw()}; /* Reseeding must not affect a, even if its RNG command is still queued */
    mag_error_t err {};
    void *pa {}, *pb {};
    size_t na {}, nb {};
    check(mag_tensor_copy_data(&err, a.get(), &pa, &na), err);
    check(mag_tensor_copy_data(&err, b.get(), &pb, &nb), err);
    ASSERT_EQ(na, nb);
    ASSERT_EQ(std::memcmp(pa, pb, na), 0);
    mag_tensor_copy_data_free(pa);
    mag_tensor_copy_data_free(pb);
}

TEST(async_stream, scalar_pow_does_not_wait) {
    env_override async {"MAG_CPU_ASYNC", "1"};
    env_override budget {"MAG_CPU_QUEUE_MB", "1024"}; /* No in-flight budget backpressure, which also counts as a host wait */
    raw_ctx ctx {};
    ASSERT_TRUE(cpu_device(*ctx)->flags&MAG_DEVICE_FLAGS_ASYNC);
    ref x {full(*ctx, 1<<20, 3.0f)};
    ref busy {full(*ctx, 1<<20, 0.0f)};
    ref one {scalar(*ctx, 1.0f)};
    mag_error_t err {};
    check(mag_ctx_sync(&err, *ctx), err);
    for (int i=0; i < 64; ++i) { /* Keep the queue busy: the exponent created below is queued behind this */
        ref out {};
        check(mag_add_(&err, out.out(), busy.get(), one.get()), err);
    }
    int64_t waits_before {(*ctx)->telemetry.host_waits};
    ref e {scalar(*ctx, 2.0f)};
    ref r {};
    check(mag_pow(&err, r.out(), x.get(), e.get()), err);
    if (!mag_envcfg_launch_blocking()) /* Launch blocking waits for every command by design */
        ASSERT_EQ((*ctx)->telemetry.host_waits, waits_before) << "pow with a scalar exponent must not drain the queue";
    ASSERT_EQ(sum_item(r.get()), 9.0*(1<<20));
    ASSERT_EQ(sum_item(busy.get()), 64.0*(1<<20));
}

TEST(async_stream, pow_scalar_does_not_wait) {
    env_override async {"MAG_CPU_ASYNC", "1"};
    env_override budget {"MAG_CPU_QUEUE_MB", "1024"};
    raw_ctx ctx {};
    ref x {full(*ctx, 1<<20, 3.0f)};
    ref busy {full(*ctx, 1<<20, 0.0f)};
    ref one {scalar(*ctx, 1.0f)};
    mag_error_t err {};
    check(mag_ctx_sync(&err, *ctx), err);
    for (int i=0; i < 64; ++i) {
        ref out {};
        check(mag_add_(&err, out.out(), busy.get(), one.get()), err);
    }
    int64_t waits_before {(*ctx)->telemetry.host_waits};
    ref r2 {};
    check(mag_pow_scalar(&err, r2.out(), x.get(), mag_scalar_from_int64(2)), err); /* Rewritten into sqr */
    ref r25 {};
    check(mag_pow_scalar(&err, r25.out(), x.get(), mag_scalar_from_float64(0.5)), err); /* Rewritten into sqrt */
    ref rg {};
    check(mag_pow_scalar(&err, rg.out(), x.get(), mag_scalar_from_float64(1.5)), err); /* Generic kernel */
    if (!mag_envcfg_launch_blocking())
        ASSERT_EQ((*ctx)->telemetry.host_waits, waits_before) << "pow_scalar must not drain the queue";
    ASSERT_EQ(sum_item(r2.get()), 9.0*(1<<20));
    ASSERT_NEAR(sum_item(r25.get()), std::sqrt(3.0)*(1<<20), 1e-3*std::sqrt(3.0)*(1<<20)); /* float32 accumulation over 1M elements */
    ASSERT_NEAR(sum_item(rg.get()), std::pow(3.0, 1.5)*(1<<20), 1e-3*std::pow(3.0, 1.5)*(1<<20));
    ASSERT_EQ(sum_item(busy.get()), 64.0*(1<<20));
}

TEST(async_stream, pow_with_pending_exponent_tensor_does_not_wait) {
    env_override async {"MAG_CPU_ASYNC", "1"};
    env_override budget {"MAG_CPU_QUEUE_MB", "1024"};
    raw_ctx ctx {};
    ref x {full(*ctx, 1<<20, 3.0f)};
    ref busy {full(*ctx, 1<<20, 0.0f)};
    ref one {scalar(*ctx, 1.0f)};
    mag_error_t err {};
    check(mag_ctx_sync(&err, *ctx), err);
    for (int i=0; i < 64; ++i) {
        ref out {};
        check(mag_add_(&err, out.out(), busy.get(), one.get()), err);
    }
    int64_t waits_before {(*ctx)->telemetry.host_waits};
    ref e {};
    check(mag_add(&err, e.out(), one.get(), one.get()), err); /* Exponent 2, still queued behind busy */
    ref r {};
    check(mag_pow(&err, r.out(), x.get(), e.get()), err); /* Must take the generic kernel instead of reading e */
    if (!mag_envcfg_launch_blocking())
        ASSERT_EQ((*ctx)->telemetry.host_waits, waits_before) << "pow must not wait for a pending exponent";
    ASSERT_EQ(sum_item(r.get()), 9.0*(1<<20));
}

TEST(async_stream, pow_scalar_matches_pow) {
    raw_ctx ctx {};
    mag_error_t err {};
    ref x {full(*ctx, 64, 1.75f)};
    for (double e : {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5, -0.5, -1.0, 2.5, -3.0}) {
        ref et {scalar(*ctx, static_cast<float>(e))};
        ref a {};
        check(mag_pow(&err, a.out(), x.get(), et.get()), err);
        ref b {};
        check(mag_pow_scalar(&err, b.out(), x.get(), mag_scalar_from_float64(e)), err);
        ASSERT_NEAR(sum_item(a.get()), sum_item(b.get()), 1e-3*std::abs(sum_item(a.get()))) << "exponent " << e;
        ref c {full(*ctx, 64, 1.75f)};
        ref co {};
        check(mag_pow_scalar_(&err, co.out(), c.get(), mag_scalar_from_float64(e)), err);
        ASSERT_EQ(co.get(), c.get());
        ASSERT_NEAR(sum_item(c.get()), sum_item(a.get()), 1e-3*std::abs(sum_item(a.get()))) << "in-place exponent " << e;
    }
}

TEST(async_stream, host_aliased_storage_is_synchronous) {
    raw_ctx ctx {};
    constexpr int64_t n {1<<12};
    ref x {full(*ctx, n, 0.0f)};
    ref one {scalar(*ctx, 1.0f)};
    mag_error_t err {};
    check(mag_tensor_mark_host_aliased(&err, x.get()), err);
    const auto *p {reinterpret_cast<const volatile float *>(mag_tensor_data_ptr(x.get()))};
    for (int i=1; i <= 50; ++i) {
        ref out {};
        check(mag_add_(&err, out.out(), x.get(), one.get()), err);
        ASSERT_EQ(p[n-1], static_cast<float>(i)) << "external readers must observe program order without a sync";
    }
}

TEST(async_stream, concurrent_producers) {
    raw_ctx ctx {};
    constexpr int producers {4};
    constexpr int iters {300};
    constexpr int64_t n {1<<12};
    std::vector<ref> xs {};
    for (int i=0; i < producers; ++i) xs.emplace_back(full(*ctx, n, 0.0f));
    ref one {scalar(*ctx, 1.0f)};
    std::vector<std::thread> threads {};
    for (int t=0; t < producers; ++t) {
        threads.emplace_back([&, t] {
            mag_error_t err {};
            for (int i=0; i < iters; ++i) {
                ref out {};
                if (mag_iserr(mag_add_(&err, out.out(), xs[t].get(), one.get()))) std::abort();
            }
        });
    }
    for (std::thread &th : threads) th.join();
    for (const ref &x : xs)
        ASSERT_EQ(sum_item(x.get()), static_cast<double>(n*iters));
}

TEST(async_stream, sync_releases_command_references) {
    raw_ctx ctx {};
    ref x {full(*ctx, 1<<16, 1.0f)};
    mag_error_t err {};
    {
        ref y {};
        check(mag_clone(&err, y.out(), x.get()), err);
    }
    check(mag_ctx_sync(&err, *ctx), err);
    ASSERT_EQ(x.get()->__rcb.rc_strong, 1) << "completed commands must not keep tensors alive after a sync";
}

TEST(async_stream, inline_fast_path_never_overtakes_queued_work) {
    raw_ctx ctx {};
    constexpr int64_t n {1<<22}; /* Large enough to be queued and multithreaded */
    ref big {full(*ctx, n, 0.0f)};
    ref one {scalar(*ctx, 1.0f)};
    mag_error_t err {};
    for (int i=0; i < 8; ++i) {
        ref out {};
        check(mag_add_(&err, out.out(), big.get(), one.get()), err); /* Queued */
        ref head {};
        check(mag_view_slice(&err, head.out(), big.get(), 0, 0, 4, 1), err);
        ref out2 {};
        check(mag_add_(&err, out2.out(), head.get(), one.get()), err); /* Tiny: may only run inline once the queue drained */
    }
    ref head {};
    check(mag_view_slice(&err, head.out(), big.get(), 0, 0, 4, 1), err);
    ASSERT_EQ(sum_item(head.get()), 4.0*16);
    ASSERT_EQ(sum_item(big.get()), 8.0*n + 8.0*4);
}
