// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>


#include <filesystem>
#include <numbers>

using namespace magnetron;

TEST(storage, new_close) {
  context ctx {cpu_device{}};

  mag_storage_archive_t* archive = mag_storage_open(&*ctx, "test.mag", 'w');
  ASSERT_NE(nullptr, archive);
  mag_storage_close(archive);
}

TEST(storage, write_inmemory_metadata_only) {
  context ctx {cpu_device{}};

  mag_storage_archive_t* archive = mag_storage_open(&*ctx, "test.mag", 'r');
  ASSERT_NE(nullptr, archive);

  // Let's put some metadata
  ASSERT_EQ(mag_storage_get_metadata_type(archive, "x"), MAG_RECORD_TYPE__COUNT);
  ASSERT_TRUE(mag_storage_set_metadata_i64(archive, "x", std::numeric_limits<int64_t>::max()));
  ASSERT_EQ(mag_storage_get_metadata_type(archive, "x"), MAG_RECORD_TYPE_I64);

  ASSERT_EQ(mag_storage_get_metadata_type(archive, "x.x"), MAG_RECORD_TYPE__COUNT);
  ASSERT_TRUE(mag_storage_set_metadata_i64(archive, "x.x", -128));
  ASSERT_EQ(mag_storage_get_metadata_type(archive, "x.x"), MAG_RECORD_TYPE_I64);
  ASSERT_TRUE(mag_storage_set_metadata_i64(archive, "y", 0));

  ASSERT_TRUE(mag_storage_set_metadata_i64(archive, "meow.128.noel", 128));

  ASSERT_FALSE(mag_storage_set_metadata_i64(archive, "y", -3));
  ASSERT_FALSE(mag_storage_set_metadata_i64(archive, "meow.128.noel", -300));

  ASSERT_EQ(mag_storage_get_metadata_type(archive, "pi"), MAG_RECORD_TYPE__COUNT);
  ASSERT_TRUE(mag_storage_set_metadata_f64(archive, "pi", std::numbers::pi_v<double>));
  ASSERT_EQ(mag_storage_get_metadata_type(archive, "pi"), MAG_RECORD_TYPE_F64);

  std::int64_t vi64 {};
  double vf64 {};
  ASSERT_TRUE(mag_storage_get_metadata_i64(archive, "x", &vi64));
  ASSERT_EQ(std::numeric_limits<int64_t>::max(), vi64);
  ASSERT_TRUE(mag_storage_get_metadata_i64(archive, "x.x", &vi64));
  ASSERT_EQ(-128, vi64);
  ASSERT_TRUE(mag_storage_get_metadata_i64(archive, "y", &vi64));
  ASSERT_EQ(0, vi64);
  ASSERT_TRUE(mag_storage_get_metadata_i64(archive, "meow.128.noel", &vi64));
  ASSERT_EQ(128, vi64);
  ASSERT_TRUE(mag_storage_get_metadata_f64(archive, "pi", &vf64));
  ASSERT_FLOAT_EQ(std::numbers::pi_v<double>, vf64);

  mag_storage_close(archive);
}

TEST(storage, read_write_disk_metadata_only) {
  context ctx {cpu_device{}};

  std::mt19937_64 rng {std::random_device{}()};
  std::uniform_int_distribution<std::int64_t> i64 {std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()};
  std::uniform_real_distribution<double> f64 {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max()};

  std::vector<std::int64_t> i64s {};
  i64s.resize(1000);
  std::ranges::generate(i64s, [&]() { return i64(rng); });
  std::vector<double> f64s {};
  f64s.resize(1000);
  std::ranges::generate(f64s, [&]() { return f64(rng); });

  {
    mag_storage_archive_t* archive = mag_storage_open(&*ctx, "test.mag", 'w');
    ASSERT_NE(nullptr, archive);
    for (std::size_t i=0; i < i64s.size(); ++i) {
      std::string name {"i64." + std::to_string(i)};
      ASSERT_TRUE(mag_storage_set_metadata_i64(archive, name.c_str(), i64s[i]));
    }
    for (std::size_t i=0; i < f64s.size(); ++i) {
      std::string name {"f64." + std::to_string(i)};
      ASSERT_TRUE(mag_storage_set_metadata_f64(archive, name.c_str(), f64s[i]));
    }
    ASSERT_TRUE(mag_storage_close(archive));
    ASSERT_TRUE(std::filesystem::exists("test.mag"));
  }

  {
    mag_storage_archive_t* archive = mag_storage_open(&*ctx, "test.mag", 'r');
    ASSERT_NE(nullptr, archive);
    for (std::size_t i=0; i < i64s.size(); ++i) {
      std::int64_t v {};
      std::string name {"i64." + std::to_string(i)};
      ASSERT_TRUE(mag_storage_get_metadata_i64(archive, name.c_str(), &v)) << name;
      ASSERT_EQ(i64s[i], v);
    }
    for (std::size_t i=0; i < f64s.size(); ++i) {
      double v {};
      std::string name {"f64." + std::to_string(i)};
      ASSERT_TRUE(mag_storage_get_metadata_f64(archive, name.c_str(), &v)) << name;
      ASSERT_EQ(f64s[i], v);
    }
    ASSERT_TRUE(mag_storage_close(archive));
  }

  ASSERT_TRUE(std::filesystem::remove("test.mag"));
}

TEST(storage, write_read_tensor_to_disk) {
  std::random_device rd {};
  std::normal_distribution<float> dist {};
  std::vector<float> data {};
  data.resize(1*2*3*4*5*6);
  for (auto& v : data) {
    v = dist(rd);
  }
  {
    context ctx {cpu_device{}};
    tensor t {ctx, dtype::e8m23, 1, 2, 3, 4, 5, 6};
    t.fill_from(data);
    mag_storage_archive_t* archive = mag_storage_open(&*ctx, "test2.mag", 'w');
    ASSERT_NE(archive, nullptr);
    ASSERT_TRUE(mag_storage_set_tensor(archive, "mat32x32x2", &*t));
    ASSERT_TRUE(mag_storage_close(archive));
    ASSERT_TRUE(std::filesystem::exists("test2.mag"));
  }
  {
    context ctx {cpu_device{}};
    mag_storage_archive_t* archive = mag_storage_open(&*ctx, "test2.mag", 'r');
    ASSERT_NE(archive, nullptr);
    mag_tensor_t* tensor_ptr = mag_storage_get_tensor(archive, "mat32x32x2");
    ASSERT_NE(nullptr, tensor_ptr);
    tensor tensor {tensor_ptr};
    ASSERT_TRUE(tensor.is_contiguous());
    ASSERT_EQ(tensor.shape().size(), 6);
    for (std::size_t i=0; i < tensor.shape().size(); ++i) {
      ASSERT_EQ(tensor.shape()[i], i+1);
    }
    auto out_data = tensor.to_vector<float>();
    for (std::size_t i=0; i < data.size(); ++i) {
      ASSERT_EQ(std::bit_cast<std::uint32_t>(data[i]), std::bit_cast<std::uint32_t>(out_data[i])); // Test for bit-exact equality
    }
    ASSERT_TRUE(mag_storage_close(archive));
  }

  ASSERT_TRUE(std::filesystem::remove("test2.mag"));
}

TEST(storage, read_gpt2_weights) {
  context ctx {cpu_device{}};
  mag_storage_archive_t* archive = mag_storage_open(&*ctx, "gpt2-fp32.mag", 'r');
  ASSERT_NE(archive, nullptr);
  std::size_t num_keys = 0;
  const char** keys = mag_storage_get_all_tensor_keys(archive, &num_keys);
  for (std::size_t i=0; i < num_keys; ++i) {
    const char* key = keys[i];
    mag_tensor_t* tensor_ptr = mag_storage_get_tensor(archive, key);
    ASSERT_NE(nullptr, tensor_ptr) << "Key: " << key;
    tensor tensor {tensor_ptr};
    ASSERT_TRUE(tensor.is_contiguous()) << "Key: " << key;
    ASSERT_GT(tensor.shape().size(), 0) << "Key: " << key;
    std::cout << "Key: " << key << ", shape: ";
    for (std::size_t j=0; j < tensor.shape().size(); ++j) {
      std::cout << tensor.shape()[j];
      if (j + 1 < tensor.shape().size()) {
        std::cout << " x ";
      }
    }
    std::cout << std::endl;
  }
}
