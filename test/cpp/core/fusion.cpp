#include <prelude.hpp>

#include <core/mag_fusion.h>

#include <cmath>
#include <vector>

using namespace magnetron;

namespace {
  mag_fuse_operand_t reg(int32_t r) { return mag_fuse_operand_t{MAG_FUSE_REG, static_cast<uint8_t>(r)}; }
  mag_fuse_operand_t imm(uint8_t s) { return mag_fuse_operand_t{MAG_FUSE_IMM, s}; }

  std::string codegen(const mag_fuse_plan_t &plan) {
    char *src = mag_fuse_codegen(&plan);
    std::string out {src ? src : ""};
    mag_fuse_source_free(src);
    return out;
  }
}

TEST(fusion, classifies_fusible_ops) {
  ASSERT_TRUE(mag_fuse_op_is_pointwise(MAG_OP_ADD));
  ASSERT_TRUE(mag_fuse_op_is_pointwise(MAG_OP_SQRT));
  ASSERT_TRUE(mag_fuse_op_is_pointwise(MAG_OP_GELU));
  /* Anything that reads more than element i must end a chain. */
  ASSERT_FALSE(mag_fuse_op_is_pointwise(MAG_OP_MATMUL));
  ASSERT_FALSE(mag_fuse_op_is_pointwise(MAG_OP_SUM));
  ASSERT_FALSE(mag_fuse_op_is_pointwise(MAG_OP_SOFTMAX));
  ASSERT_FALSE(mag_fuse_op_is_pointwise(MAG_OP_SCATTER_ADD));
}

TEST(fusion, rejects_wrong_arity_and_overflow) {
  mag_fuse_plan_t plan;
  mag_fuse_plan_init(&plan, MAG_DTYPE_FLOAT32);
  int32_t a = mag_fuse_load(&plan, 0);
  ASSERT_GE(a, 0);
  mag_fuse_operand_t one[1] = {reg(a)};
  ASSERT_LT(mag_fuse_emit(&plan, MAG_OP_ADD, one, 1), 0);       /* ADD needs two operands */
  ASSERT_LT(mag_fuse_emit(&plan, MAG_OP_MATMUL, one, 1), 0);    /* not a pointwise op */
  mag_fuse_operand_t bad[2] = {reg(a), reg(99)};
  ASSERT_LT(mag_fuse_emit(&plan, MAG_OP_ADD, bad, 2), 0);       /* register not yet defined */
}

TEST(fusion, generated_source_is_one_loop) {
  mag_fuse_plan_t plan;
  mag_fuse_plan_init(&plan, MAG_DTYPE_FLOAT32);
  int32_t x = mag_fuse_load(&plan, 0);
  mag_fuse_operand_t mul[2] = {reg(x), imm(0)};
  int32_t scaled = mag_fuse_emit(&plan, MAG_OP_MUL, mul, 2);
  int32_t act = mag_fuse_emit(&plan, MAG_OP_RELU, (mag_fuse_operand_t[]){reg(scaled)}, 1);
  ASSERT_TRUE(mag_fuse_store(&plan, 1, act));
  std::string src = codegen(plan);
  /* The whole chain must collapse into a single loop, otherwise there is no fusion. */
  ASSERT_EQ(src.find("for (int64_t i=begin"), src.rfind("for (int64_t i=begin"));
  ASSERT_NE(src.find("fmaxf"), std::string::npos);
  ASSERT_NE(src.find("b1[i] ="), std::string::npos);
  ASSERT_NE(src.find("const float *restrict b0"), std::string::npos); /* read-only input */
}

/* A buffer that is read and written must not be marked restrict, or the compiler is free to
   assume it does not alias the others. */
TEST(fusion, in_place_buffer_is_not_restrict) {
  mag_fuse_plan_t plan;
  mag_fuse_plan_init(&plan, MAG_DTYPE_FLOAT32);
  int32_t p = mag_fuse_load(&plan, 0);
  int32_t g = mag_fuse_load(&plan, 1);
  mag_fuse_operand_t add[2] = {reg(p), reg(g)};
  ASSERT_TRUE(mag_fuse_store(&plan, 0, mag_fuse_emit(&plan, MAG_OP_ADD, add, 2)));
  std::string src = codegen(plan);
  ASSERT_NE(src.find("float *b0 = (float *)bufs[0]"), std::string::npos);
  ASSERT_NE(src.find("const float *restrict b1"), std::string::npos);
}

/* The point of the whole exercise: a compiled kernel must produce exactly what the ops it
   replaces would have produced. */
TEST(fusion, compiled_adam_matches_reference) {
  context ctx {};
  mag_fuse_plan_t plan;
  mag_fuse_plan_init(&plan, MAG_DTYPE_FLOAT32);
  /* bufs: 0=param 1=grad 2=m 3=v   imms: 0=b1 1=b2 2=lr 3=eps 4=c1 5=c2 */
  int32_t p = mag_fuse_load(&plan, 0);
  int32_t g = mag_fuse_load(&plan, 1);
  int32_t m = mag_fuse_load(&plan, 2);
  int32_t v = mag_fuse_load(&plan, 3);
  int32_t b1m = mag_fuse_emit(&plan, MAG_OP_MUL, (mag_fuse_operand_t[]){reg(m), imm(0)}, 2);
  int32_t one_b1 = mag_fuse_emit(&plan, MAG_OP_SUB, (mag_fuse_operand_t[]){imm(6), imm(0)}, 2);
  int32_t gp = mag_fuse_emit(&plan, MAG_OP_MUL, (mag_fuse_operand_t[]){reg(g), reg(one_b1)}, 2);
  int32_t mn = mag_fuse_emit(&plan, MAG_OP_ADD, (mag_fuse_operand_t[]){reg(b1m), reg(gp)}, 2);
  int32_t b2v = mag_fuse_emit(&plan, MAG_OP_MUL, (mag_fuse_operand_t[]){reg(v), imm(1)}, 2);
  int32_t one_b2 = mag_fuse_emit(&plan, MAG_OP_SUB, (mag_fuse_operand_t[]){imm(6), imm(1)}, 2);
  int32_t g2 = mag_fuse_emit(&plan, MAG_OP_SQR, (mag_fuse_operand_t[]){reg(g)}, 1);
  int32_t g2p = mag_fuse_emit(&plan, MAG_OP_MUL, (mag_fuse_operand_t[]){reg(g2), reg(one_b2)}, 2);
  int32_t vn = mag_fuse_emit(&plan, MAG_OP_ADD, (mag_fuse_operand_t[]){reg(b2v), reg(g2p)}, 2);
  int32_t mh = mag_fuse_emit(&plan, MAG_OP_DIV, (mag_fuse_operand_t[]){reg(mn), imm(4)}, 2);
  int32_t vh = mag_fuse_emit(&plan, MAG_OP_DIV, (mag_fuse_operand_t[]){reg(vn), imm(5)}, 2);
  int32_t vs = mag_fuse_emit(&plan, MAG_OP_SQRT, (mag_fuse_operand_t[]){reg(vh)}, 1);
  int32_t den = mag_fuse_emit(&plan, MAG_OP_ADD, (mag_fuse_operand_t[]){reg(vs), imm(3)}, 2);
  int32_t q = mag_fuse_emit(&plan, MAG_OP_DIV, (mag_fuse_operand_t[]){reg(mh), reg(den)}, 2);
  int32_t upd = mag_fuse_emit(&plan, MAG_OP_MUL, (mag_fuse_operand_t[]){reg(q), imm(2)}, 2);
  int32_t pn = mag_fuse_emit(&plan, MAG_OP_SUB, (mag_fuse_operand_t[]){reg(p), reg(upd)}, 2);
  ASSERT_GE(pn, 0);
  ASSERT_TRUE(mag_fuse_store(&plan, 0, pn));
  ASSERT_TRUE(mag_fuse_store(&plan, 2, mn));
  ASSERT_TRUE(mag_fuse_store(&plan, 3, vn));

  mag_fused_fn_t fn = nullptr;
  mag_error_t err {};
  mag_status_t st = mag_fuse_compile(&err, &*ctx, &plan, &fn);
  if (mag_iserr(st)) GTEST_SKIP() << "no host compiler available: " << err.message;
  ASSERT_NE(fn, nullptr);

  constexpr int64_t n = 1024;
  std::vector<float> p_j(n), g_j(n), m_j(n), v_j(n), p_r(n), m_r(n), v_r(n);
  for (int64_t i=0; i < n; ++i) {
    p_j[i] = p_r[i] = 0.01f*static_cast<float>(i%97);
    g_j[i] = 0.001f*static_cast<float>(i%13) - 0.003f;   /* spans negative, zero and positive */
    m_j[i] = m_r[i] = 0.002f*static_cast<float>(i%7);
    v_j[i] = v_r[i] = 0.0005f*static_cast<float>(i%11) + 1e-4f; /* strictly positive, so eps never dominates */
  }
  const double b1 = 0.9, b2 = 0.999, lr = 1e-3, eps = 1e-8;
  const double c1 = 1.0 - std::pow(b1, 10.0), c2 = 1.0 - std::pow(b2, 10.0);
  const double imms[7] = {b1, b2, lr, eps, c1, c2, 1.0};
  void *bufs[4] = {p_j.data(), g_j.data(), m_j.data(), v_j.data()};
  fn(bufs, imms, 0, n);

  /* The reference must mirror the plan operation for operation, including where each value is
     rounded to float. Computing (float)(1.0-b1) in double and narrowing gives a different last bit
     than 1.0f - (float)b1, which is what the generated kernel does. */
  const float b1f = static_cast<float>(b1), b2f = static_cast<float>(b2);
  const float lrf = static_cast<float>(lr), epsf = static_cast<float>(eps);
  const float c1f = static_cast<float>(c1), c2f = static_cast<float>(c2);
  const float one_b1f = 1.0f - b1f, one_b2f = 1.0f - b2f;
  for (int64_t i=0; i < n; ++i) {
    float mn_ = b1f*m_r[i] + g_j[i]*one_b1f;
    float vn_ = b2f*v_r[i] + (g_j[i]*g_j[i])*one_b2f;
    m_r[i] = mn_;
    v_r[i] = vn_;
    float den = std::sqrt(vn_/c2f) + epsf;
    p_r[i] = p_r[i] - (mn_/c1f)/den*lrf;
  }

  /* Built with -ffp-contract=off precisely so fusing a chain cannot change its arithmetic:
     the generated kernel must agree with the op-by-op reference bit for bit. */
  for (int64_t i=0; i < n; ++i) {
    ASSERT_FLOAT_EQ(p_j[i], p_r[i]) << "param mismatch at " << i;
    ASSERT_FLOAT_EQ(m_j[i], m_r[i]) << "moment mismatch at " << i;
    ASSERT_FLOAT_EQ(v_j[i], v_r[i]) << "variance mismatch at " << i;
  }
}

/* The same chain must not be compiled twice, and changing only a scalar must not force a rebuild. */
TEST(fusion, cache_reuses_kernel_across_immediates) {
  context ctx {};
  mag_fuse_plan_t plan;
  mag_fuse_plan_init(&plan, MAG_DTYPE_FLOAT32);
  int32_t x = mag_fuse_load(&plan, 0);
  int32_t y = mag_fuse_emit(&plan, MAG_OP_MUL, (mag_fuse_operand_t[]){reg(x), imm(0)}, 2);
  ASSERT_TRUE(mag_fuse_store(&plan, 1, y));
  mag_fused_fn_t a = nullptr, b = nullptr;
  mag_error_t err {};
  if (mag_iserr(mag_fuse_compile(&err, &*ctx, &plan, &a))) GTEST_SKIP() << "no host compiler: " << err.message;
  ASSERT_FALSE(mag_iserr(mag_fuse_compile(&err, &*ctx, &plan, &b)));
  ASSERT_EQ(a, b);
  uint64_t hits = 0, compiles = 0;
  mag_fuse_cache_stats(&*ctx, &hits, &compiles);
  ASSERT_EQ(compiles, 1u);
  ASSERT_GE(hits, 1u);

  /* Scaling by a different constant reuses the same compiled code. */
  const double s1[1] = {2.0}, s2[1] = {3.0};
  std::vector<float> in {1.0f, 2.0f, 3.0f, 4.0f}, out(4, 0.0f);
  void *bufs[2] = {in.data(), out.data()};
  a(bufs, s1, 0, 4);
  ASSERT_FLOAT_EQ(out[3], 8.0f);
  a(bufs, s2, 0, 4);
  ASSERT_FLOAT_EQ(out[3], 12.0f);
  mag_fuse_cache_stats(&*ctx, &hits, &compiles);
  ASSERT_EQ(compiles, 1u);
}
