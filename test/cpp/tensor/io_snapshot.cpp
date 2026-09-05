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

#include <filesystem>
#include <fstream>
#include <prelude.hpp>

#include <core/mag_io_snapshot_layout.h>

using namespace magnetron;

namespace {
    constexpr const char *k_meta = R"({"manifest_ver":1,"tensor_map":{}})";

    struct blob final {
        std::vector<float> data;
        uint64_t offset;
        [[nodiscard]] uint64_t numbytes() const noexcept { return data.size()*sizeof(float); }
    };

    std::vector<blob> write_snapshot(mag_context_t *ctx, const char *path, std::vector<std::vector<float>> tensors) {
        std::vector<blob> blobs {};
        uint64_t end = 0;
        for (auto &&t : tensors) {
            end = mag_align_up(end, MAG_SNAP_TENSOR_BLOB_ALIGN);
            blob b {};
            b.data = std::move(t);
            b.offset = end;
            blobs.emplace_back(std::move(b));
            end += blobs.back().numbytes();
        }
        mag_snapshot_stream_writer_t *writer = nullptr;
        mag_error_t err {};
        EXPECT_EQ(mag_snapshot_stream_writer_open(&err, &writer, ctx, path, k_meta, std::strlen(k_meta), end), MAG_OK) << err.message;
        for (auto &&b : blobs)
            EXPECT_EQ(mag_snapshot_stream_writer_submit_blob(&err, writer, b.data.data(), b.numbytes()), MAG_OK) << err.message;
        EXPECT_EQ(mag_snapshot_stream_writer_close(&err, writer), MAG_OK) << err.message;
        return blobs;
    }

    [[nodiscard]] std::vector<float> ramp(std::size_t n, float base) {
        std::vector<float> v (n);
        for (std::size_t i=0; i < n; ++i) v[i] = base + static_cast<float>(i);
        return v;
    }

    void write_raw(const char *path, const std::vector<std::uint8_t> &bytes) {
        std::ofstream f {path, std::ios::binary|std::ios::trunc};
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    [[nodiscard]] std::vector<std::uint8_t> read_raw(const char *path) {
        std::ifstream f {path, std::ios::binary};
        return std::vector<std::uint8_t> {std::istreambuf_iterator<char> {f}, std::istreambuf_iterator<char> {}};
    }
}

TEST(snapshot, write_read_roundtrip) {
    context ctx {};
    constexpr const char *path = "test_roundtrip.mag";
    test::scope_guard rm {[&] { std::filesystem::remove(path); }};
    std::vector<blob> blobs = write_snapshot(&*ctx, path, {ramp(7, 0.0f), ramp(3, 100.0f), ramp(129, 1000.0f)});
    ASSERT_TRUE(std::filesystem::exists(path));

    mag_snapshot_stream_reader_t *reader = nullptr;
    mag_error_t err {};
    ASSERT_EQ(mag_snapshot_stream_reader_open(&err, &reader, &*ctx, path), MAG_OK) << err.message;
    test::scope_guard close {[&] { mag_snapshot_stream_reader_close(reader); }};

    ASSERT_EQ(mag_snapshot_stream_reader_version(reader), MAG_SNAPSHOT_VERSION);
    uint64_t meta_len = 0;
    const char *meta = mag_snapshot_stream_reader_meta(reader, &meta_len);
    ASSERT_EQ(std::string_view(meta, meta_len), k_meta);
    ASSERT_EQ(mag_snapshot_stream_reader_blob_len(reader), blobs.back().offset+blobs.back().numbytes());

    for (auto &&b : blobs) {
        auto numel = static_cast<int64_t>(b.data.size());
        mag_tensor_t *t = nullptr;
        ASSERT_EQ(mag_snapshot_stream_reader_borrow_tensor(&err, &t, reader, b.offset, b.numbytes(), MAG_DTYPE_FLOAT32, 1, &numel), MAG_OK) << err.message;
        ASSERT_NE(t, nullptr);
        test::scope_guard drop {[&] { mag_rc_decref(t); }};
        ASSERT_EQ(mag_tensor_numel(t), numel);
        ASSERT_EQ(mag_tensor_type(t), MAG_DTYPE_FLOAT32);
        ASSERT_TRUE(mag_tensor_is_contiguous(t));
        ASSERT_EQ(std::memcmp(reinterpret_cast<const void *>(mag_tensor_data_ptr(t)), b.data.data(), b.numbytes()), 0);
        ASSERT_EQ(mag_tensor_data_ptr(t) % MAG_SNAP_TENSOR_BLOB_ALIGN, 0); /* Page aligned, so O_DIRECT and GDS can start here */
    }
}

TEST(snapshot, borrowing_is_zero_copy_and_outlives_the_reader) {
    context ctx {};
    constexpr const char *path = "test_borrow.mag";
    test::scope_guard rm {[&] { std::filesystem::remove(path); }};
    std::vector<blob> blobs = write_snapshot(&*ctx, path, {ramp(16, 7.0f)});

    mag_snapshot_stream_reader_t *reader = nullptr;
    mag_error_t err {};
    ASSERT_EQ(mag_snapshot_stream_reader_open(&err, &reader, &*ctx, path), MAG_OK) << err.message;
    int64_t numel = 16;
    mag_tensor_t *a = nullptr;
    mag_tensor_t *b = nullptr;
    ASSERT_EQ(mag_snapshot_stream_reader_borrow_tensor(&err, &a, reader, blobs[0].offset, blobs[0].numbytes(), MAG_DTYPE_FLOAT32, 1, &numel), MAG_OK) << err.message;
    ASSERT_EQ(mag_snapshot_stream_reader_borrow_tensor(&err, &b, reader, blobs[0].offset, blobs[0].numbytes(), MAG_DTYPE_FLOAT32, 1, &numel), MAG_OK) << err.message;
    ASSERT_EQ(mag_tensor_data_ptr(a), mag_tensor_data_ptr(b));

    mag_snapshot_stream_reader_close(reader);
    mag_rc_decref(b);
    ASSERT_EQ(std::memcmp(reinterpret_cast<const void *>(mag_tensor_data_ptr(a)), blobs[0].data.data(), blobs[0].numbytes()), 0);
    mag_rc_decref(a);
}

TEST(snapshot, writer_refuses_to_close_a_short_data_section) {
    context ctx {};
    constexpr const char *path = "test_short.mag";
    std::vector<float> data = ramp(8, 0.0f);
    mag_snapshot_stream_writer_t *writer = nullptr;
    mag_error_t err {};
    ASSERT_EQ(mag_snapshot_stream_writer_open(&err, &writer, &*ctx, path, k_meta, std::strlen(k_meta), 2*data.size()*sizeof(float)), MAG_OK) << err.message;
    ASSERT_EQ(mag_snapshot_stream_writer_submit_blob(&err, writer, data.data(), data.size()*sizeof(float)), MAG_OK) << err.message;
    ASSERT_NE(mag_snapshot_stream_writer_close(&err, writer), MAG_OK);
    ASSERT_FALSE(std::filesystem::exists(path));
    ASSERT_FALSE(std::filesystem::exists(std::string {path}+".tmp"));
}

TEST(snapshot, writer_rejects_more_data_than_declared) {
    context ctx {};
    constexpr const char *path = "test_overlong.mag";
    test::scope_guard rm {[&] { std::filesystem::remove(std::string {path}+".tmp"); }};
    std::vector<float> data = ramp(8, 0.0f);
    mag_snapshot_stream_writer_t *writer = nullptr;
    mag_error_t err {};
    ASSERT_EQ(mag_snapshot_stream_writer_open(&err, &writer, &*ctx, path, k_meta, std::strlen(k_meta), data.size()*sizeof(float)), MAG_OK) << err.message;
    ASSERT_EQ(mag_snapshot_stream_writer_submit_blob(&err, writer, data.data(), data.size()*sizeof(float)), MAG_OK) << err.message;
    ASSERT_NE(mag_snapshot_stream_writer_submit_blob(&err, writer, data.data(), sizeof(float)), MAG_OK);
    mag_snapshot_stream_writer_abort(writer);
    ASSERT_FALSE(std::filesystem::exists(path));
}

TEST(snapshot, writer_requires_the_mag_extension) {
    context ctx {};
    mag_snapshot_stream_writer_t *writer = nullptr;
    mag_error_t err {};
    ASSERT_NE(mag_snapshot_stream_writer_open(&err, &writer, &*ctx, "test_wrong.bin", k_meta, std::strlen(k_meta), 4), MAG_OK);
    ASSERT_EQ(writer, nullptr);
    ASSERT_FALSE(std::filesystem::exists("test_wrong.bin"));
}

TEST(snapshot, reader_rejects_files_that_are_not_snapshots) {
    context ctx {};
    constexpr const char *good_path = "test_corrupt_src.mag";
    constexpr const char *bad_path = "test_corrupt.mag";
    test::scope_guard rm {[&] { std::filesystem::remove(good_path); std::filesystem::remove(bad_path); }};
    write_snapshot(&*ctx, good_path, {ramp(64, 0.0f)});
    std::vector<std::uint8_t> good = read_raw(good_path);

    const auto rejects = [&](std::vector<std::uint8_t> bytes, const char *what) {
        write_raw(bad_path, bytes);
        mag_snapshot_stream_reader_t *reader = nullptr;
        mag_error_t err {};
        EXPECT_NE(mag_snapshot_stream_reader_open(&err, &reader, &*ctx, bad_path), MAG_OK) << "accepted " << what;
        EXPECT_EQ(reader, nullptr) << what;
    };

    rejects({}, "an empty file");
    rejects({good.begin(), good.begin()+16}, "a file shorter than the header");
    std::vector<std::uint8_t> bad_magic = good;
    bad_magic[0] = 'X';
    rejects(bad_magic, "a bad magic");
    std::vector<std::uint8_t> bad_version = good;
    uint32_t future = MAG_SNAPSHOT_VERSION+10000;
    std::memcpy(bad_version.data()+4, &future, sizeof(future));
    rejects(bad_version, "a newer format version");
    std::vector<std::uint8_t> foreign_endian = good;
    uint32_t aux = MAG_SNAP_AUX_BIG_ENDIAN ^ MAG_SNAP_AUX_HOST_ENDIAN;
    std::memcpy(foreign_endian.data()+8, &aux, sizeof(aux));
    rejects(foreign_endian, "foreign endianness");
    std::vector<std::uint8_t> reserved = good;
    uint32_t bits = MAG_SNAP_AUX_HOST_ENDIAN|(1u<<9);
    std::memcpy(reserved.data()+8, &bits, sizeof(bits));
    rejects(reserved, "reserved aux bits");
    rejects({good.begin(), good.begin()+static_cast<std::ptrdiff_t>(good.size()/2)}, "a truncated file");
}

TEST(snapshot, reader_rejects_borrows_outside_or_across_the_layout) {
    context ctx {};
    constexpr const char *path = "test_bounds.mag";
    test::scope_guard rm {[&] { std::filesystem::remove(path); }};
    std::vector<blob> blobs = write_snapshot(&*ctx, path, {ramp(32, 0.0f)});
    uint64_t blob_len = blobs[0].offset+blobs[0].numbytes();

    mag_snapshot_stream_reader_t *reader = nullptr;
    mag_error_t err {};
    ASSERT_EQ(mag_snapshot_stream_reader_open(&err, &reader, &*ctx, path), MAG_OK) << err.message;
    test::scope_guard close {[&] { mag_snapshot_stream_reader_close(reader); }};

    int64_t numel = 32;
    mag_tensor_t *t = nullptr;
    ASSERT_NE(mag_snapshot_stream_reader_borrow_tensor(&err, &t, reader, blob_len, blobs[0].numbytes(), MAG_DTYPE_FLOAT32, 1, &numel), MAG_OK);
    ASSERT_NE(mag_snapshot_stream_reader_borrow_tensor(&err, &t, reader, 0, blob_len+1, MAG_DTYPE_FLOAT32, 1, &numel), MAG_OK);
    ASSERT_NE(mag_snapshot_stream_reader_borrow_tensor(&err, &t, reader, 64, blobs[0].numbytes(), MAG_DTYPE_FLOAT32, 1, &numel), MAG_OK);
    ASSERT_NE(mag_snapshot_stream_reader_borrow_tensor(&err, &t, reader, 0, blobs[0].numbytes(), static_cast<mag_dtype_t>(MAG_DTYPE__NUM), 1, &numel), MAG_OK);
    int64_t too_many = numel+1;
    ASSERT_NE(mag_snapshot_stream_reader_borrow_tensor(&err, &t, reader, 0, blobs[0].numbytes(), MAG_DTYPE_FLOAT32, 1, &too_many), MAG_OK);
    ASSERT_EQ(t, nullptr);
    ASSERT_EQ(mag_snapshot_stream_reader_borrow_tensor(&err, &t, reader, 0, blobs[0].numbytes(), MAG_DTYPE_FLOAT32, 1, &numel), MAG_OK) << err.message;
    mag_rc_decref(t);
}
