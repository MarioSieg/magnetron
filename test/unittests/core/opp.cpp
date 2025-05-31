// (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

TEST(op_param, pack_e8m23) {
    e8m23_t val {3.1415f};
    mag_OPParam p {mag_op_param_wrap_e8m23(val)};
    ASSERT_EQ(p.type, MAG_OPP_E8M23);
    ASSERT_EQ(p.u64, std::bit_cast<std::uint32_t>(val));
    ASSERT_EQ(mag_op_param_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_e8m23_or(p, 0.0f), val);

    val = std::numeric_limits<e8m23_t>::min();
    p = mag_op_param_wrap_e8m23(val);
    ASSERT_EQ(p.type, MAG_OPP_E8M23);
    ASSERT_EQ(p.u64, std::bit_cast<std::uint32_t>(val));
    ASSERT_EQ(mag_op_param_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_e8m23_or(p, 0.0f), val);

    val = std::numeric_limits<e8m23_t>::max();
    p = mag_op_param_wrap_e8m23(val);
    ASSERT_EQ(p.type, MAG_OPP_E8M23);
    ASSERT_EQ(p.u64, std::bit_cast<std::uint32_t>(val));
    ASSERT_EQ(mag_op_param_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_e8m23_or(p, 0.0f), val);
}

TEST(op_param, pack_i64) {
    std::int64_t val {-123456};
    mag_OPParam p {mag_op_param_wrap_i64(val)};
    ASSERT_EQ(p.type, MAG_OPP_I64);
    ASSERT_EQ(mag_op_param_unpack_i64_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_i64_or(p, 0), val);

    val = -(((1ll<<62)>>1)); /* min val for int62 */
    p = mag_op_param_wrap_i64(val);
    ASSERT_EQ(p.type, MAG_OPP_I64);
    ASSERT_EQ(mag_op_param_unpack_i64_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_i64_or(p, 0), val);

    val = ((1ll<<62)>>1)-1; /* max val for int62 */
    p = mag_op_param_wrap_i64(val);
    ASSERT_EQ(p.type, MAG_OPP_I64);
    ASSERT_EQ(mag_op_param_unpack_i64_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_i64_or(p, 0), val);
}

TEST(op_param, pack_u64) {
    std::uint64_t val {123456};
    mag_OPParam p {mag_op_param_wrap_u64(val)};
    ASSERT_EQ(p.type, MAG_OPP_U64);
    ASSERT_EQ(p.u64, val);
    ASSERT_EQ(mag_op_param_unpack_u64_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_u64_or(p, 0), val);

    val = 0; /* min val for uint62 */
    p = mag_op_param_wrap_u64(val);
    ASSERT_EQ(p.type, MAG_OPP_U64);
    ASSERT_EQ(p.u64, val);
    ASSERT_EQ(mag_op_param_unpack_u64_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_u64_or(p, 0), val);

    val = (1ull<<62)-1; /* max val for uint62 */
    p = mag_op_param_wrap_u64(val);
    ASSERT_EQ(p.type, MAG_OPP_U64);
    ASSERT_EQ(p.u64, val);
    ASSERT_EQ(mag_op_param_unpack_u64_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_u64_or(p, 0), val);
}
