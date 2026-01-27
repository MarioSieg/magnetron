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

#include <filesystem>
#include <prelude.hpp>

using namespace magnetron;

TEST(snapshot, read_write_tensors) {
    context ctx {};
    std::mt19937_64 rng {1234};
    std::vector<std::pair<std::string, tensor>> tensors {};
    for (int type = 0; type < MAG_DTYPE__NUM; ++type) {
        size_t num_tensors = std::uniform_int_distribution<size_t>{1, 10}(rng);
        for (size_t i=0; i < num_tensors; ++i) {
            std::string name = "tensor_" + std::to_string(i) + "_typeid_" + std::to_string(type);
            std::vector<int64_t> shape = {};
            int64_t rank = std::uniform_int_distribution<int64_t>{1, 6}(rng);
            shape.reserve(rank);
            for (int64_t r=0; r < rank; ++r)
                shape.emplace_back(std::uniform_int_distribution<int64_t>{1, 10}(rng));
            tensor data {ctx, static_cast<dtype>(type), shape};
            if (type == MAG_DTYPE_BOOLEAN) data.bernoulli_();
            else if (data.is_floating_point_typed()) data.uniform_(-100.f, 100.f);
            else data.uniform_(0, 100);
            tensors.emplace_back(name, data);
        }
    }
    { // write
        mag_snapshot_t *snap = mag_snapshot_new(&*ctx);
        ASSERT_NE(snap, nullptr);
        test::scope_guard sg {[&] { mag_snapshot_free(snap); }};
        for (auto&& [name, t] : tensors)
            ASSERT_TRUE(mag_snapshot_put_tensor(snap, name.c_str(), &*t));
        ASSERT_TRUE(mag_snapshot_serialize(snap, "snap.mag"));
    }
    ASSERT_TRUE(std::filesystem::exists("snap.mag"));
    { // read
        mag_snapshot_t *snap = mag_snapshot_deserialize(&*ctx, "snap.mag");
        ASSERT_NE(snap, nullptr);
        test::scope_guard sg {[&] { mag_snapshot_free(snap); }};
        {
            mag_snapshot_print_info(snap);
            for (auto&& [name, t_orig] : tensors) {
                mag_tensor_t *t_loaded = mag_snapshot_get_tensor(snap, name.c_str());
                ASSERT_NE(t_loaded, nullptr);
                tensor loaded {t_loaded};
                ASSERT_EQ(loaded.dtype(), t_orig.dtype()) << "Dtype mismatch for tensor: "
                    << name << ": "
                    << mag_type_trait(static_cast<mag_dtype_t>(loaded.dtype()))->name << " != "
                    << mag_type_trait(static_cast<mag_dtype_t>(t_orig.dtype()))->name;
                ASSERT_EQ(loaded.shape(), t_orig.shape()) << "Shape mismatch for tensor: " << name;
                ASSERT_EQ(loaded.numel(), t_orig.numel()) << "Numel mismatch for tensor: " << name;
                ASSERT_TRUE((loaded == t_orig).all()) << "Tensor data mismatch for tensor: " << name;
            }
        }
    }
    tensors.clear();
    ASSERT_TRUE(std::filesystem::remove("snap.mag"));
}
