// +---------------------------------------------------------------------+
// | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
// | Licensed under the Apache License, Version 2.0                      |
// |                                                                     |
// | Website : https://mariosieg.com                                     |
// | GitHub  : https://github.com/MarioSieg                              |
// | License : https://www.apache.org/licenses/LICENSE-2.0               |
// +---------------------------------------------------------------------+

// Command line options:
// -jobs=48 -workers=48 -max_len=16384 -rss_limit_mb=16384 -max_total_time=60 -exact_artifact_path="bin/fuzz" --dict=fuzzer/dict.txt
// 3600 = 1hr


#include <cstddef>
#include <cstdint>

#include <magnetron.hpp>
#include "../magnetron/magnetron_internal.h"

using namespace magnetron;

extern "C" auto mag_storage_read_from_buffer(mag_storage_archive_t* archive, const std::uint8_t* buf, std::size_t size) -> bool; // External C function

static constexpr std::size_t file_header_size = MAG_STO_STATIC_FILE_HEADER_SIZE;
static constexpr std::uint32_t file_magic = MAG_STO_FILE_MAGIC;
static std::unique_ptr<context> ctx = std::make_unique<context>(device_type::cpu);

extern "C" auto LLVMFuzzerTestOneInput(const std::int8_t* data, std::size_t nb) -> int {
    if (nb < file_header_size) {
        return -1; // Input too small to contain a valid header, do not add to corpus
    }
    if (*reinterpret_cast<const std::uint32_t*>(data) != file_magic) {
        return -1; // Invalid magic, do not add to corpus
    }
    // Data is now big enough for a header, add the magic
    mag_storage_archive_t* archive = mag_storage_new(&**ctx, "__mem__", 'r');
    bool ok = mag_storage_read_from_buffer(archive, reinterpret_cast<const std::uint8_t*>(data), nb);
    mag_storage_close(archive);
    return ok ? 0 : -1;
}