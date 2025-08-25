// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

// Command line options:
// -jobs=64 -workers=64 -max_len=16384 -rss_limit_mb=16384 -max_total_time=3600 -exact_artifact_path="bin/fuzz"
// 3600 = 1hr

#include <cstddef>
#include <cstdint>

#include <magnetron.hpp>

extern "C" [[nodiscard]] auto mag__sto_read_buffered(mag_ctx_t* ctx, const std::uint8_t* buf, std::size_t size, std::size_t* out_n_tensors) -> mag_tensor_t**; // Imported from msml.c

extern "C" auto LLVMFuzzerTestOneInput(const std::int8_t* data, std::size_t dize) -> int {
    static std::unique_ptr<magnetron::ctx> ctx {new magnetron::ctx{}};
    std::size_t n_tensors = 0;
    [[maybe_unused]]
    mag_tensor_t** volatile tensors = mag__sto_read_buffered(**ctx, reinterpret_cast<const std::uint8_t*>(data), dize, &n_tensors);
    return !tensors ? -1 : 0;
}