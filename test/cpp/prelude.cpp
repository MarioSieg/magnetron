// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

namespace magnetron::test {
    std::vector<device_kind> get_supported_test_backends() {
        static std::optional<std::vector<device_kind> > backends = std::nullopt;
        if (!backends) {
            backends.emplace({device_kind::cpu});
            #ifdef MAG_ENABLE_CUDA
                backends->emplace_back(device_kind::cuda);
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
        {1},
        {1, 1},
        {1, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {2},
        {3},
        {4},
        {1, 2},
        {2, 1},
        {1, 3},
        {3, 1},
        {1, 1, 2},
        {1, 2, 1},
        {2, 1, 1},
        {2, 3},
        {3, 2},
        {1, 2, 3},
        {3, 2, 1},
        {1, 3, 1, 3},
        {3, 5},
        {4, 7},
        {7, 4},
        {5, 3, 4},
        {4, 5, 3},
        {3, 4, 5},
        {1, 16},
        {16, 1},
        {1, 1, 16},
        {8, 1, 16},
        {1, 8, 16},
        {2, 3, 1, 16},
        {2, 1, 3, 16},
        {1, 2, 3, 1},
        {1, 2, 1, 3},
        {2, 1, 1, 3},
        {1, 32, 128},
        {32, 1, 128},
        {32, 128, 1},
        {16},
        {32},
        {64},
        {128},
        {256},
        {512},
        {1024},
        {2048},
        {4096},
        {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024},
        {1, 3, 224, 224},
        {1, 3, 512, 512},
        {8, 3, 32, 32},
        {4, 3, 1024, 1024},
        {64, 3, 7, 7},
        {128, 64, 3, 3},
        {256, 128, 1, 1},
        {512, 256, 3, 3},
        {1, 197, 768},
        {1, 577, 768},
        {1, 197, 1024},
        {1, 4096},
        {1, 5120},
        {1, 8192},
        {1, 12288},
        {16, 4096},
        {32, 4096},
        {8, 8192},
        {8, 12288},
        {32, 128},
        {40, 128},
        {64, 128},
        {32, 96},
        {4096 / 4, 12288 / 4},
        {8192 / 4, 8192 / 4},
        {12288 / 4, 12288 / 4},
        {11008 / 4, 4096 / 4},
        {4096 / 4, 11008 / 4},
        {8192 / 4, 22016 / 6},
        {12288 / 4, 49152 / 8},
        {1, 32, 2048 / 4, 128 / 4},
        {4, 32, 2048 / 4, 128},
        {1, 40, 4096 / 4, 96},
        {8, 64, 1024 / 4, 128 / 4},
        {2, 32, 4096 / 4, 128},
        {16, 128},
        {128, 128},
        {2048, 128},
        {3, 1, 7},
        {1, 3, 7},
        {7, 1, 3},
        {3, 7, 1},
        {1, 7, 3},
        {7, 3, 1},
        {2, 3, 5, 7},
        {7, 5, 3, 2},
        {17},
        {31},
        {67},
        {127},
        {257},
        {2, 3, 4, 5, 6},
        {6, 5, 4, 3, 2},
        {1, 2, 3, 4, 5, 6},
        {1, 1, 4096, 4096},
        {1024 / 2, 1024 / 2},
        {2048 / 2, 2048 / 2},
        {4096 / 2, 1024 / 2},
        {1024 / 2, 4096 / 2},
        {4096 / 2, 4096 / 2},
        {8192 / 2, 2048 / 2},
        {6, 66, 666},
        {4, 4, 1024, 1024},
        {2, 8, 512, 512},
        {5},
        {7},
        {9},
        {11},
        {13},
        {2, 2},
        {2, 2, 2},
        {2, 2, 2, 2},
        {3, 3},
        {3, 3, 3},
        {4, 1, 4},
        {1, 4, 1, 4},
        {1, 2, 2, 1},
        {2, 1, 2, 1},
        {1, 1, 2, 2},
        {2, 2, 1, 1},
        {1, 3, 5},
        {5, 3, 1},
        {1, 5, 3},
        {3, 1, 5},
        {1, 1, 8},
        {8, 1, 1},
        {1, 8, 1},
        {1, 1, 1, 8},
        {8, 1, 1, 1},
        {1, 8, 1, 1},
        {1, 1, 8, 1},
        {1, 1, 16, 1},
        {1, 16, 1, 1},
        {1, 4, 1, 16},
        {4, 1, 16, 1},
        {1, 1, 4, 16},
        {1, 7, 1, 1, 7},
        {7, 1, 7, 1, 1},
        {1, 1, 7, 1, 7},
        {1, 3, 7, 7},
        {1, 3, 9, 9},
        {1, 3, 11, 11},
        {2, 3, 17, 17},
        {1, 1, 15, 15},
        {4, 3, 13, 13},
        {8, 3, 17, 17},
        {1, 16, 64},
        {2, 16, 64},
        {4, 16, 64},
        {1, 32, 64},
        {2, 32, 64},
        {4, 32, 64},
        {1, 32, 128},
        {2, 32, 128},
        {4, 32, 128},
        {1, 64, 64},
        {2, 64, 64},
        {4, 64, 64},
        {1, 64, 128},
        {2, 64, 128},
        {4, 64, 128},
        {1, 1, 16, 16},
        {1, 2, 16, 32},
        {2, 2, 16, 32},
        {1, 4, 32, 32},
        {2, 4, 32, 32},
        {4, 4, 32, 32},
        {1, 8, 64, 32},
        {2, 8, 64, 32},
        {1, 8, 32, 64},
        {2, 8, 32, 64},
        {7, 13},
        {13, 7},
        {15, 33},
        {33, 15},
        {31, 64},
        {64, 31},
        {63, 64},
        {64, 63},
        {127, 64},
        {64, 127},
        {255, 64},
        {64, 255},
        {16, 20},
        {20, 16},
        {8, 24},
        {24, 8},
        {7, 17},
        {17, 7},
        {5, 21},
        {21, 5},
        {4, 4, 16},
        {4, 8, 16},
        {8, 4, 16},
        {2, 8, 32},
        {2, 16, 32},
        {4, 8, 32},
        {2, 4, 8, 16},
        {2, 4, 16, 8},
        {4, 2, 8, 16},
        {1, 4, 8, 16},
        {1, 8, 4, 16},
        {1, 2, 3, 4, 5},
        {5, 4, 3, 2, 1},
        {1, 1, 2, 3, 4},
        {1, 2, 1, 3, 4},
        {2, 1, 3, 1, 4},
        {1, 2, 3, 1, 4},
        {1, 2, 3, 4, 1},
        {1, 2, 3, 4, 5, 1},
        {1, 1, 2, 3, 4, 5},
        {2, 3, 1, 4, 1, 5},
        {1, 16, 8, 8},
        {1, 8, 8, 16},
        {2, 16, 8, 8},
        {2, 8, 8, 16},
        {4, 16, 4, 4},
        {4, 4, 4, 16},
        {1, 1, 32},
        {32, 1, 1},
        {1, 32, 1},
        {2, 1, 32},
        {2, 32, 1},
        {1, 2, 32},
        {1, 2, 1, 32},
        {2, 1, 1, 32},
        {1, 2, 32, 1},
        {1, 15},
        {1, 31},
        {1, 63},
        {2, 63},
        {4, 63},
        {3, 21},
        {3, 63},
        {5, 19},
        {7, 8},
        {8, 7},
        {15, 16},
        {16, 15},
        {31, 32},
        {32, 31},
        {1, 2, 2, 2, 2, 2, 2},
        {2, 1, 2, 2, 2, 2, 2},
        {1, 3, 1, 3, 1, 3, 1},
        {2, 3, 4, 1, 2, 3, 4},
        {1, 2, 3, 4, 1, 2, 3, 4},
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
