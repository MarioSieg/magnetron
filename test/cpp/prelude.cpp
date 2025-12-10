// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

namespace magnetron::test {
    std::vector<device_kind> get_supported_test_backends() {
        static std::optional<std::vector<device_kind> > backends = std::nullopt;
        if (!backends) {
            backends.emplace({device_kind::cpu});
            #ifdef MAG_ENABLE_CUDA
                //backends->emplace_back(device_kind::cuda);
            #endif
        }
        return *backends;
    }

    context &get_cached_context(device_kind dev) {
        static std::unordered_map<device_kind, std::unique_ptr<context> > cached;
        if (cached.find(dev) == cached.end()) {
            cached[dev] = std::make_unique<context>(get_device_kind_name(dev));
            cached[dev]->stop_grad_recorder();
        }
        return *cached[dev];
    }

    std::string get_gtest_backend_name(const TestParamInfo<device_kind> &info) {
        return get_device_kind_name(info.param);
    }

    auto shape_as_vec(tensor t) -> std::vector<int64_t> {
        mag_tensor_t *internal{&*t};
        return {std::begin(internal->coords.shape), std::end(internal->coords.shape)};
    }

    auto strides_as_vec(tensor t) -> std::vector<int64_t> {
        mag_tensor_t *internal{&*t};
        return {std::begin(internal->coords.strides), std::end(internal->coords.strides)};
    }

    auto shape_to_string(const std::vector<int64_t>& shape) -> std::string {
        std::stringstream ss{};
        ss << "(";
        for (size_t i{}; i < shape.size(); ++i) {
            ss << shape[i];
            if (i != shape.size() - 1) {
                ss << ", ";
            }
        }
        ss << ")";
        return ss.str();
    }

    thread_local std::random_device rd{};
    thread_local std::mt19937_64 gen{rd()};

    const std::unordered_map<dtype, float> dtype_eps_map{
        {dtype::float32, dtype_traits<float>::test_eps},
        {dtype::float16, dtype_traits<float16>::test_eps},
    };

    static const std::vector<std::vector<int64_t> > TEST_SHAPES = {
    {},
    {1,},
    {2,},
    {3,},
    {1, 1},
    {2, 2},
    {2, 3},
    {3, 2},
    {3, 3},
    {1, 2, 3},
    {3, 2, 1},
    {2, 3, 5},
    {1, 2},
    {2, 1},
    {1, 2, 1},
    {1, 2, 3, 1},
    {2, 1, 3, 1},
    {1, 1, 2, 3},
    {1, 2, 3, 4},
    {1, 1, 2, 3, 4},
    {1, 3, 8, 8},
    {2, 3, 8, 8},
    {2, 3, 16, 16},
    {1, 16, 64},
    {2, 16, 64},
    {2, 32, 64},
    {2, 32, 128},
    {7, 13},
    {13, 7},
    {5, 21},
    {21, 5},
    };

    void for_all_test_shapes(std::function<void (const std::vector<int64_t>&)> &&f) {
        for (const auto &shape: TEST_SHAPES) {
            std::invoke(f, shape);
        }
    }

    tensor make_random_view(tensor base) {
        std::mt19937_64 &rng{gen};
        if (base.rank() == 0) return base.view();
        bool all_one = true;
        for (auto s: base.shape()) {
            if (s > 1) {
                all_one = false;
                break;
            }
        }
        if (all_one) return base.view();
        std::vector<int64_t> slicable = {};
        for (int64_t d{}; d < base.rank(); ++d)
            if (base.shape()[d] > 1) slicable.push_back(d);
        std::uniform_int_distribution<size_t> dim_dis(0, slicable.size() - 1);
        int64_t dim{slicable[dim_dis(rng)]};
        int64_t size{base.shape()[dim]};
        std::uniform_int_distribution<int64_t> step_dis{2, std::min<int64_t>(4, size)};
        int64_t step{step_dis(rng)};
        int64_t max_start{size - step};
        std::uniform_int_distribution<int64_t> start_dis{0, max_start};
        int64_t start{start_dis(rng)};
        int64_t max_len{(size - start + step - 1) / step};
        std::uniform_int_distribution<int64_t> len_dis{1, max_len};
        int64_t len{len_dis(rng)};
        return base.view_slice(dim, start, len, step);
    }
}
