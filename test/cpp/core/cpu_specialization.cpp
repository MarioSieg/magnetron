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

#include <prelude.hpp>

#include <core/mag_cpuid.h>
#include <cpu/mag_cpu_specialization_detector.h>

#include <string>
#include <vector>

namespace {
    constexpr uint64_t bit(uint32_t i) { return 1ull<<i; }

    uint64_t req_123() { return bit(1)|bit(2)|bit(3); }
    uint64_t req_12() { return bit(1)|bit(2); }
    uint64_t req_1() { return bit(1); }
    uint64_t req_1234() { return bit(1)|bit(2)|bit(3)|bit(4); }

    const mag_cpu_specialization_t synthetic[] = {
        {"synth-abc", &req_123, nullptr},
        {"synth-ab", &req_12, nullptr},
        {"synth-a", &req_1, nullptr},
    };
    constexpr size_t num_synthetic = sizeof(synthetic)/sizeof(*synthetic);

    const mag_cpu_specialization_t gapped[] = {
        {"synth-abcd", &req_1234, nullptr},
        {"synth-abc", &req_123, nullptr},
        {"synth-a", &req_1, nullptr},
    };
    constexpr size_t num_gapped = sizeof(gapped)/sizeof(*gapped);

    const mag_cpu_specialization_t *find_built(const char *name) {
        size_t num = 0;
        const mag_cpu_specialization_t *impls = mag_cpu_specializations(&num);
        for (size_t i=0; i < num; ++i)
            if (std::string{impls[i].name} == name) return impls+i;
        return nullptr;
    }
}

TEST(cpu_specialization, select_strongest_satisfied) {
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, num_synthetic, ~0ull), synthetic+0);
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, num_synthetic, bit(1)|bit(2)|bit(3)), synthetic+0);
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, num_synthetic, bit(1)|bit(2)), synthetic+1);
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, num_synthetic, bit(1)|bit(2)|bit(7)), synthetic+1);
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, num_synthetic, bit(1)), synthetic+2);
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, num_synthetic, bit(1)|bit(3)), synthetic+2);
}

TEST(cpu_specialization, reject_partial_match) {
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, num_synthetic, bit(2)|bit(3)), nullptr);
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, num_synthetic, 0), nullptr);
    ASSERT_EQ(mag_cpu_select_specialization(synthetic, 0, ~0ull), nullptr);
    ASSERT_EQ(mag_cpu_select_specialization(gapped, num_gapped, bit(1)|bit(3)|bit(4)), gapped+2);
    ASSERT_EQ(mag_cpu_select_specialization(gapped, num_gapped, bit(1)|bit(2)|bit(3)), gapped+1);
}

TEST(cpu_specialization, format_caps_names_every_bit) {
    char buf[512];
    ASSERT_EQ(mag_cpu_format_caps(0, buf, sizeof(buf)), 0u);
    ASSERT_STREQ(buf, "");
    size_t len = mag_cpu_format_caps(bit(1)|bit(63), buf, sizeof(buf));
    ASSERT_EQ(len, strlen(buf));
    std::string s {buf};
    ASSERT_NE(s.find(mag_cpu_cap_name(1)), std::string::npos);
    ASSERT_NE(s.find("bit63"), std::string::npos);
    char tiny[4];
    ASSERT_LT(mag_cpu_format_caps(~0ull, tiny, sizeof(tiny)), sizeof(tiny));
}

TEST(cpu_specialization, built_list_is_ordered_strongest_first) {
    size_t num = 0;
    const mag_cpu_specialization_t *impls = mag_cpu_specializations(&num);
    if (!num) GTEST_SKIP() << "no specializations built";
    ASSERT_EQ(mag_cpu_select_specialization(impls, num, ~0ull), impls+0);
    ASSERT_EQ(mag_cpu_select_specialization(impls, num, 0), nullptr);
    for (size_t i=0; i < num; ++i) {
        uint64_t own = (*impls[i].get_feature_bitset)();
        ASSERT_NE(own, 0u) << impls[i].name;
        const mag_cpu_specialization_t *sel = mag_cpu_select_specialization(impls, num, own);
        ASSERT_NE(sel, nullptr) << impls[i].name;
        ASSERT_LE(sel-impls, static_cast<ptrdiff_t>(i)) << impls[i].name;
        uint64_t sel_req = (*sel->get_feature_bitset)();
        ASSERT_EQ(sel_req & ~own, 0u) << sel->name << " selected for " << impls[i].name;
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)

TEST(cpu_specialization, arm64_github_runner_caps) {
    size_t num = 0;
    const mag_cpu_specialization_t *impls = mag_cpu_specializations(&num);
    if (!num) GTEST_SKIP() << "no specializations built";
    uint64_t without_cvt =
        mag_arm64_cap(NEON)|mag_arm64_cap(PMULL)|mag_arm64_cap(CRC32)|mag_arm64_cap(DOTPROD)|mag_arm64_cap(I8MM)|
        mag_arm64_cap(F16SCALAR)|mag_arm64_cap(F16VECTOR)|mag_arm64_cap(BF16)|mag_arm64_cap(SVE)|mag_arm64_cap(SVE2);
    ASSERT_EQ(without_cvt, 0xefeull);
    ASSERT_EQ(mag_cpu_select_specialization(impls, num, without_cvt), nullptr);
    uint64_t runner = without_cvt|mag_arm64_cap(F16CVT);
    ASSERT_EQ(runner, 0xffeull);
    ASSERT_EQ(mag_cpu_select_specialization(impls, num, runner), impls+0);
    if (const mag_cpu_specialization_t *v9 = find_built("arm64-v9_sve2"))
        ASSERT_EQ(mag_cpu_select_specialization(impls, num, runner), v9);
}

TEST(cpu_specialization, arm64_requirements_track_build_flags) {
    uint64_t f16 = mag_arm64_cap(F16SCALAR)|mag_arm64_cap(F16VECTOR)|mag_arm64_cap(F16CVT);
    uint64_t v82 = mag_arm64_cap(NEON)|mag_arm64_cap(CRC32)|mag_arm64_cap(DOTPROD)|f16;
    uint64_t v86 = v82|mag_arm64_cap(I8MM)|mag_arm64_cap(BF16);
    struct { const char *name; uint64_t req; } expected[] = {
        {"arm64-v82", v82},
        {"arm64-v82_sve", mag_arm64_cap(NEON)|mag_arm64_cap(CRC32)|f16|mag_arm64_cap(SVE)},
        {"arm64-v86", v86},
        {"arm64-v86_crypto", v86|mag_arm64_cap(PMULL)},
        {"arm64-v86_sve", v86|mag_arm64_cap(PMULL)|mag_arm64_cap(SVE)},
        {"arm64-v9_sve2", v86|mag_arm64_cap(PMULL)|mag_arm64_cap(SVE)|mag_arm64_cap(SVE2)},
    };
    size_t checked = 0;
    for (const auto &e : expected) {
        const mag_cpu_specialization_t *spec = find_built(e.name);
        if (!spec) continue;
        char want[512], got[512];
        mag_cpu_format_caps(e.req, want, sizeof(want));
        mag_cpu_format_caps((*spec->get_feature_bitset)(), got, sizeof(got));
        ASSERT_EQ((*spec->get_feature_bitset)(), e.req) << e.name << " expected [" << want << "] got [" << got << "]";
        ++checked;
    }
    if (!checked) GTEST_SKIP() << "no arm64 specializations built";
}

TEST(cpu_specialization, arm64_representative_hosts) {
    size_t num = 0;
    const mag_cpu_specialization_t *impls = mag_cpu_specializations(&num);
    if (!num) GTEST_SKIP() << "no specializations built";
    uint64_t f16 = mag_arm64_cap(F16SCALAR)|mag_arm64_cap(F16VECTOR)|mag_arm64_cap(F16CVT);
    uint64_t neoverse_n1 = mag_arm64_cap(NEON)|mag_arm64_cap(PMULL)|mag_arm64_cap(CRC32)|mag_arm64_cap(DOTPROD)|f16;
    uint64_t apple_m2 = neoverse_n1|mag_arm64_cap(I8MM)|mag_arm64_cap(BF16);
    uint64_t graviton3 = apple_m2|mag_arm64_cap(SVE);
    uint64_t a64fx = mag_arm64_cap(NEON)|mag_arm64_cap(PMULL)|mag_arm64_cap(CRC32)|f16|mag_arm64_cap(SVE);
    uint64_t armv8_base = mag_arm64_cap(NEON)|mag_arm64_cap(F16CVT);
    struct { uint64_t host; const char *want; } cases[] = {
        {neoverse_n1, "arm64-v82"},
        {apple_m2, "arm64-v86_crypto"},
        {graviton3, "arm64-v86_sve"},
        {a64fx, "arm64-v82_sve"},
    };
    for (const auto &c : cases) {
        const mag_cpu_specialization_t *want = find_built(c.want);
        if (!want) continue;
        const mag_cpu_specialization_t *sel = mag_cpu_select_specialization(impls, num, c.host);
        ASSERT_NE(sel, nullptr) << c.want;
        ASSERT_EQ(std::string{sel->name}, std::string{c.want});
    }
    ASSERT_EQ(mag_cpu_select_specialization(impls, num, armv8_base), nullptr);
}

#endif
