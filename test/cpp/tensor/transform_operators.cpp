// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

#include <prelude.hpp>

using namespace magnetron;
using namespace test;

TEST(cpu_tensor_transform_ops, view_no_axes) {
  auto ctx = context{compute_device::cpu};
  auto base = tensor{ctx, dtype::e8m23, 2, 2, 3, 1};
  auto v = base.view();
  ASSERT_FALSE(base.is_view());
  ASSERT_TRUE(v.is_view());
  ASSERT_EQ(base.storage_base_ptr(), v.storage_base_ptr());
}
