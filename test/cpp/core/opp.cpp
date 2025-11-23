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

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

TEST(op_param, pack_e8m23) {
    float val {3.1415f};
    mag_opparam_t p {mag_op_param_wrap_e8m23(val)};
    ASSERT_EQ(p.type, MAG_OPP_E8M23);
    ASSERT_EQ(p.i64, std::bit_cast<uint32_t>(val));
    ASSERT_EQ(mag_op_param_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_e8m23_or(p, 0.0f), val);

    val = std::numeric_limits<float>::min();
    p = mag_op_param_wrap_e8m23(val);
    ASSERT_EQ(p.type, MAG_OPP_E8M23);
    ASSERT_EQ(p.i64, std::bit_cast<uint32_t>(val));
    ASSERT_EQ(mag_op_param_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_e8m23_or(p, 0.0f), val);

    val = std::numeric_limits<float>::max();
    p = mag_op_param_wrap_e8m23(val);
    ASSERT_EQ(p.type, MAG_OPP_E8M23);
    ASSERT_EQ(p.i64, std::bit_cast<uint32_t>(val));
    ASSERT_EQ(mag_op_param_unpack_e8m23_or_panic(p), val);
    ASSERT_EQ(mag_op_param_unpack_e8m23_or(p, 0.0f), val);
}

TEST(op_param, pack_i64) {
    int64_t val {-123456};
    mag_opparam_t p {mag_op_param_wrap_i64(val)};
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
