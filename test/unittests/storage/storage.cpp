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
  {
    context ctx {cpu_device{}};
    tensor t {ctx, dtype::e8m23, 32, 32, 2};
    t.fill_float(-2.5f);
    mag_storage_archive_t* archive = mag_storage_open(&*ctx, "test2.mag", 'w');
    ASSERT_TRUE(mag_storage_set_tensor(archive, "mat32x32x2", &*t));
    ASSERT_TRUE(mag_storage_close(archive));
    ASSERT_TRUE(std::filesystem::exists("test2.mag"));
  }

  ASSERT_TRUE(std::filesystem::remove("test2.mag"));
}
