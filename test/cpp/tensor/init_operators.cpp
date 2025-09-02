// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

static constexpr std::int64_t lim {4};

TEST(cpu_tensor_init_ops, copy_e8m23) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        tensor t {ctx, dtype::e8m23, shape};
        std::vector<e8m23_t> fill_data {};
        fill_data.resize(t.numel());
        std::uniform_real_distribution<e8m23_t> dist {dtype_traits<e8m23_t>::min, dtype_traits<e8m23_t>::max};
        std::ranges::generate(fill_data, [&] { return dist(gen); });
        t.fill_from(fill_data);
        std::vector<e8m23_t> data {t.to_float_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (std::size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], fill_data[i]);
        }
    });
}

TEST(cpu_tensor_init_ops, copy_e5m10) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        tensor t {ctx, dtype::e5m10, shape};
        std::vector<e8m23_t> fill_data {};
        fill_data.resize(t.numel());
        std::uniform_real_distribution<e8m23_t> dist {-1.0f, 1.0f};
        std::ranges::generate(fill_data, [&] { return dist(gen); });
        t.fill_from(fill_data);
        std::vector<e8m23_t> data {t.to_float_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (std::size_t i {}; i < data.size(); ++i) {
            ASSERT_NEAR(data[i], fill_data[i], dtype_traits<e5m10_t>::test_eps);
        }
    });
}

TEST(cpu_tensor_init_ops, copy_bool) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        tensor t {ctx, dtype::boolean, shape};
        std::vector<bool> fill_data {};
        fill_data.reserve(t.numel());
        std::bernoulli_distribution dist {0.5f};
        for (std::size_t i {}; i < t.numel(); ++i) {
            fill_data.emplace_back(dist(gen));
        }
        t.fill_from(fill_data);
        std::vector<bool> data {t.to_bool_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (std::size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], fill_data[i]);
        }
    });
}

TEST(cpu_tensor_init_ops, fill_e8m23) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        std::uniform_real_distribution<e8m23_t> dist {dtype_traits<e8m23_t>::min, dtype_traits<e8m23_t>::max};
        e8m23_t fill_val {dist(gen)};
        tensor t {ctx, dtype::e8m23, shape};
        t.fill_float(fill_val);
        std::vector<e8m23_t> data {t.to_float_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (std::size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], fill_val);
        }
    });
}

TEST(cpu_tensor_init_ops, fill_e5m10) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        std::uniform_real_distribution<e8m23_t> dist {-1.0f, 1.0f};
        e8m23_t fill_val {dist(gen)};
        tensor t {ctx, dtype::e5m10, shape};
        t.fill_float(fill_val);
        std::vector<e8m23_t> data {t.to_float_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (std::size_t i {}; i < data.size(); ++i) {
            ASSERT_NEAR(data[i], fill_val, 1e-3f);
        }
    });
}

TEST(cpu_tensor_init_ops, fill_bool) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        std::bernoulli_distribution dist {};
        bool fill_val {dist(gen)};
        tensor t {ctx, dtype::boolean, shape};
        t.fill_int(fill_val);
        std::vector<bool> data {t.to_bool_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (std::size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], !!fill_val);
        }
    });
}

TEST(cpu_tensor_init_ops, fill_random_uniform_e8m23) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        std::uniform_real_distribution dist {dtype_traits<e8m23_t>::min, dtype_traits<e8m23_t>::max};
        e8m23_t min {dist(gen)};
        e8m23_t max {std::uniform_real_distribution{min, dtype_traits<e8m23_t>::max}(gen)};
        tensor t {ctx, dtype::e8m23, shape};
        t.fill_rand_uniform_float(min, max);
        std::vector<e8m23_t> data {t.to_float_vector()};
        ASSERT_EQ(data.size(), t.numel());
        for (auto x : data) {
           ASSERT_GE(x, min);
           ASSERT_LE(x, max);
       }
    });
}

#if 0 // TODO
TEST(cpu_tensor_init_ops, fill_random_uniform_e5m10) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        std::uniform_real_distribution dist {-1.0f, 1.0f};
         e8m23_t min {dist(gen)};
         e8m23_t max {std::uniform_real_distribution{min, dtype_traits<e8m23_t>::max}(gen)};
         e8m23_t qmin {static_cast<e8m23_t>(e5m10_t{min})};
         e8m23_t qmax {static_cast<e8m23_t>(e5m10_t{max})};
         tensor t {ctx, dtype::e5m10, shape};
         t.fill_rand_uniform(qmin, qmax);
         std::vector<e8m23_t> data {t.to_vector()};
         ASSERT_EQ(data.size(), t.numel());
         for (auto x : data) {
             ASSERT_GE(x, qmin);
             ASSERT_LE(x, qmax);
         }
    });
}
#endif

#if 0 // TODO
TEST(cpu_tensor_init_ops, fill_random_normal_e8m23) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        e8m23_t mean {std::uniform_real_distribution{0.0f, 5.0f}(gen)};
        e8m23_t stddev {std::uniform_real_distribution{0.0f, 5.0f}(gen)};
        tensor t {ctx, dtype::e8m23, shape};
        t.fill_rand_normal(mean, stddev);
        std::vector<e8m23_t> data {t.to_vector()};
        ASSERT_FLOAT_EQ(mean, compute_mean<e8m23_t>(data));
        ASSERT_FLOAT_EQ(stddev, compute_std<e8m23_t>(data));
    });
}

TEST(cpu_tensor_init_ops, fill_random_normal_e5m10) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        e8m23_t mean {std::uniform_real_distribution{0.0f, 5.0f}(gen)};
        e8m23_t stddev {std::uniform_real_distribution{0.0f, 5.0f}(gen)};
        tensor t {ctx, dtype::e5m10, shape};
        t.fill_rand_normal(mean, stddev);
        std::vector<e8m23_t> data {t.to_vector()};
        ASSERT_FLOAT_EQ(mean, compute_mean<e8m23_t>(data));
        ASSERT_FLOAT_EQ(stddev, compute_std<e8m23_t>(data));
    });
}
#endif

#if 0
TEST(cpu_tensor_init_ops, fill_random_bool) {
    context ctx {compute_device::cpu};
    for_all_shape_perms(lim, 1, [&](std::span<const std::int64_t> shape) {
        std::uniform_real_distribution<e8m23_t> dist {0.01f, 0.99f};
        e8m23_t p {dist(gen)};
        tensor t {ctx, dtype::boolean, shape};
        t.fill_rand_bernoulli(p);
        std::vector<bool> data {t.to_bool_vector()};
        int64_t samples {};
        for (bool k : data)
            samples += k ? 1 : 0;
        double phat {static_cast<double>(samples) / data.size()};
        double z  {(phat - p) / std::sqrt(p*(1.0 - p) / static_cast<double>(data.size()))};
        EXPECT_LT(std::fabs(z), 3.49) << "phat=" << phat << "  z=" << z;
    });
}
#endif
