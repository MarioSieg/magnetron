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

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

template <typename T>
[[nodiscard]] auto compute_mean(const std::vector<T>& data) -> float {
    float sum {};
    for (const T x : data) sum += x;
    return sum / static_cast<float>(data.size());
}

template <typename T>
[[nodiscard]] auto compute_mean(const mag_tensor_t* tensor) -> float {
    return compute_mean(std::vector<T>{reinterpret_cast<const T*>(mag_tensor_data_ptr(tensor)), static_cast<size_t>(tensor->numel)});
}

template <typename T>
[[nodiscard]] auto compute_std(const std::vector<T>& data) -> double {
    float sum {};
    float mean {compute_mean(data)};
    for (const T x : data) {
        sum += std::pow(x-mean, 2.0f);
    }
    return std::sqrt(sum / static_cast<float>(data.size()));
}

template <typename T>
[[nodiscard]] auto compute_std(const mag_tensor_t* tensor) -> double {
    return compute_std(std::vector<T>{reinterpret_cast<const T*>(mag_tensor_data_ptr(tensor)), static_cast<size_t>(tensor->numel)});
}

TEST(cpu_tensor_init_ops, copy_float32) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        tensor t {ctx, dtype::float32, shape};
        std::vector<float> fill_data {};
        fill_data.resize(t.numel());
        std::uniform_real_distribution dist {dtype_traits<float>::min, dtype_traits<float>::max};
        std::generate(fill_data.begin(), fill_data.end(), [&] { return dist(gen); });
        t.fill_from(fill_data);
        std::vector data {t.to_vector<float>()};
        ASSERT_EQ(data.size(), t.numel());
        for (size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], fill_data[i]);
        }
    });
}

TEST(cpu_tensor_init_ops, copy_float16) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        tensor t {ctx, dtype::float16, shape};
        std::vector<float> fill_data {};
        fill_data.resize(t.numel());
        std::uniform_real_distribution dist {-1.0f, 1.0f};
        std::generate(fill_data.begin(), fill_data.end(), [&] { return dist(gen); });
        t.fill_from(fill_data);
        std::vector data {t.to_vector<float16>()};
        ASSERT_EQ(data.size(), t.numel());
        for (size_t i {}; i < data.size(); ++i) {
            ASSERT_NEAR(data[i], fill_data[i], dtype_traits<float16>::test_eps);
        }
    });
}

TEST(cpu_tensor_init_ops, copy_bool) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        tensor t {ctx, dtype::boolean, shape};
        std::vector<bool> fill_data {};
        fill_data.reserve(t.numel());
        std::bernoulli_distribution dist {0.5f};
        for (size_t i {}; i < t.numel(); ++i) {
            fill_data.emplace_back(dist(gen));
        }
        t.fill_from(fill_data);
        std::vector data {t.to_vector<bool>()};
        ASSERT_EQ(data.size(), t.numel());
        for (size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], fill_data[i]);
        }
    });
}

TEST(cpu_tensor_init_ops, fill_float32) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        std::uniform_real_distribution dist {dtype_traits<float>::min, dtype_traits<float>::max};
        float fill_val {dist(gen)};
        tensor t {ctx, dtype::float32, shape};
        t.fill_(fill_val);
        std::vector data {t.to_vector<float>()};
        ASSERT_EQ(data.size(), t.numel());
        for (size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], fill_val);
        }
    });
}

TEST(cpu_tensor_init_ops, fill_float16) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        std::uniform_real_distribution dist {-1.0f, 1.0f};
        float fill_val {dist(gen)};
        tensor t {ctx, dtype::float16, shape};
        t.fill_(fill_val);
        std::vector data {t.to_vector<float16>()};
        ASSERT_EQ(data.size(), t.numel());
        for (size_t i {}; i < data.size(); ++i) {
            ASSERT_NEAR(data[i], fill_val, 1e-3f);
        }
    });
}

TEST(cpu_tensor_init_ops, fill_bool) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        std::bernoulli_distribution dist {};
        bool fill_val {dist(gen)};
        tensor t {ctx, dtype::boolean, shape};
        t.fill_(fill_val);
        std::vector data {t.to_vector<bool>()};
        ASSERT_EQ(data.size(), t.numel());
        for (size_t i {}; i < data.size(); ++i) {
            ASSERT_EQ(data[i], !!fill_val);
        }
    });
}

TEST(cpu_tensor_init_ops, fill_random_uniform_float32) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        std::uniform_real_distribution dist {dtype_traits<float>::min, dtype_traits<float>::max};
        float min {dist(gen)};
        float max {std::uniform_real_distribution{min+0.1f, dtype_traits<float>::max}(gen)};
        tensor t {ctx, dtype::float32, shape};
        t.uniform_(min, max);
        std::vector data {t.to_vector<float>()};
        ASSERT_EQ(data.size(), t.numel());
        for (size_t i {}; i < data.size(); ++i) {
           ASSERT_GE(data[i], min);
           ASSERT_LE(data[i], max);
       }
    });
}

#if 0 // TODO
TEST(cpu_tensor_init_ops, fill_random_uniform_float16) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        std::uniform_real_distribution dist {-1.0f, 1.0f};
         float min {dist(gen)};
         float max {std::uniform_real_distribution{min, dtype_traits<float>::max}(gen)};
         float qmin {static_cast<float>(float16{min})};
         float qmax {static_cast<float>(float16{max})};
         tensor t {ctx, dtype::float16, shape};
         t.uniform_(qmin, qmax);
         std::vector<float> data {t.to_vector()};
         ASSERT_EQ(data.size(), t.numel());
         for (auto x : data) {
             ASSERT_GE(x, qmin);
             ASSERT_LE(x, qmax);
         }
    });
}
#endif

#if 0 // TODO
TEST(cpu_tensor_init_ops, fill_random_normal_float32) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        float mean {std::uniform_real_distribution{0.0f, 5.0f}(gen)};
        float stddev {std::uniform_real_distribution{0.0f, 5.0f}(gen)};
        tensor t {ctx, dtype::float32, shape};
        t.normal_(mean, stddev);
        std::vector<float> data {t.to_vector()};
        ASSERT_FLOAT_EQ(mean, compute_mean<float>(data));
        ASSERT_FLOAT_EQ(stddev, compute_std<float>(data));
    });
}

TEST(cpu_tensor_init_ops, fill_random_normal_float16) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        float mean {std::uniform_real_distribution{0.0f, 5.0f}(gen)};
        float stddev {std::uniform_real_distribution{0.0f, 5.0f}(gen)};
        tensor t {ctx, dtype::float16, shape};
        t.normal_(mean, stddev);
        std::vector<float> data {t.to_vector()};
        ASSERT_FLOAT_EQ(mean, compute_mean<float>(data));
        ASSERT_FLOAT_EQ(stddev, compute_std<float>(data));
    });
}
#endif

#if 0
TEST(cpu_tensor_init_ops, fill_random_bool) {
    context ctx {};
    for_all_test_shapes([&](const std::vector<int64_t>& shape) {
        std::uniform_real_distribution<float> dist {0.01f, 0.99f};
        float p {dist(gen)};
        tensor t {ctx, dtype::boolean, shape};
        t.bernoulli_(p);
        std::vector<bool> data {t.to_vector<bool>()};
        int64_t samples {};
        for (bool k : data)
            samples += k ? 1 : 0;
        double phat {static_cast<double>(samples) / data.size()};
        double z  {(phat - p) / std::sqrt(p*(1.0 - p) / static_cast<double>(data.size()))};
        EXPECT_LT(std::fabs(z), 3.49) << "phat=" << phat << "  z=" << z;
    });
}
#endif
