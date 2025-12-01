// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#pragma once

#include <bit>
#include <random>
#include <functional>
#include <span>

#include <magnetron.hpp>
#include <core/mag_context.h>
#include <core/mag_tensor.h>

#include <gtest/gtest.h>

#include <half.hpp>

using namespace testing;

namespace magnetron::test {
    enum class device_kind {
        cpu,
        cuda,
    };

    [[nodiscard]] constexpr const char* get_device_kind_name(device_kind dvc) {
        switch (dvc) {
            case device_kind::cpu: return "cpu";
            case device_kind::cuda: return "cuda";
            default: return "";
        }
    }

    [[nodiscard]] extern std::vector<device_kind> get_supported_test_backends();
    [[nodiscard]] extern context& get_cached_context(device_kind dev);

    using float16 = half_float::half;

    template <typename T>
    struct dtype_traits final {
        static constexpr T min {std::numeric_limits<T>::min()};
        static constexpr T max {std::numeric_limits<T>::min()};
        static constexpr float eps {std::numeric_limits<T>::epsilon()};
        static inline const float test_eps {std::numeric_limits<T>::epsilon()};
    };

    template <>
    struct dtype_traits<float16> final {
        static constexpr float16 min {std::numeric_limits<float16>::min()};
        static constexpr float16 max {std::numeric_limits<float16>::min()};
        static inline const float eps {std::numeric_limits<float16>::epsilon()};
        static inline const float test_eps {std::numeric_limits<float16>::epsilon()+0.04f}; // We increase the epsilon for f16 a little, as multiplication fails if not
    };

    [[nodiscard]] extern std::string get_gtest_backend_name(const TestParamInfo<device_kind>& info);

    [[nodiscard]] extern std::vector<int64_t> shape_as_vec(tensor t);
    [[nodiscard]] extern std::vector<int64_t> strides_as_vec(tensor t);
    [[nodiscard]] extern std::string shape_to_string(const std::vector<int64_t>& shape);
    [[nodiscard]] extern auto make_random_view(tensor base) -> tensor;

    extern thread_local std::random_device rd;
    extern thread_local std::mt19937_64 gen;

    extern const std::unordered_map<dtype, float> dtype_eps_map;

    extern void for_all_test_shapes(std::function<void (const std::vector<int64_t>&)>&& f);
}
