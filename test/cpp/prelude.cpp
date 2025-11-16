// (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>


namespace magnetron::test {
  std::vector<device_kind> get_supported_test_backends() {
    return {device_kind::cpu, device_kind::cuda};
  }

  std::string get_gtest_backend_name(const TestParamInfo<device_kind>& info){
    return get_device_kind_id(info.param);
  }

  auto shape_as_vec(tensor t) -> std::vector<std::int64_t> {
    mag_tensor_t* internal {&*t};
    return {std::begin(internal->coords.shape), std::end(internal->coords.shape)};
  }

  auto strides_as_vec(tensor t) -> std::vector<std::int64_t> {
    mag_tensor_t* internal {&*t};
    return {std::begin(internal->coords.strides), std::end(internal->coords.strides)};
  }

  auto shape_to_string(std::span<const std::int64_t> shape) -> std::string {
    std::stringstream ss {};
    ss << "(";
    for (std::size_t i {}; i < shape.size(); ++i) {
      ss << shape[i];
      if (i != shape.size() - 1) {
        ss << ", ";
      }
    }
    ss << ")";
    return ss.str();
  }

  thread_local std::random_device rd {};
  thread_local std::mt19937_64 gen {rd()};

  const std::unordered_map<dtype, float> dtype_eps_map {
    {dtype::e8m23, dtype_traits<float>::test_eps},
    {dtype::e5m10,  dtype_traits<float16>::test_eps},
    {dtype::boolean, 0.0f},
    {dtype::i32, 0.0f},
  };


  void for_all_shape_perms(std::int64_t lim, std::int64_t fac, std::function<void (std::span<const std::int64_t>)>&& f) {
    if (lim <= 0)
      throw std::runtime_error("Invalid limit");
    ++lim;
    std::vector<std::int64_t> shape {};
    shape.reserve(MAG_MAX_DIMS);
    for (std::int64_t i0 = 1; i0 < lim; ++i0) {
      for (std::int64_t i1 = 0; i1 < lim; ++i1) {
        for (std::int64_t i2 = 0; i2 < lim; ++i2) {
          for (std::int64_t i3 = 0; i3 < lim; ++i3) {
            for (std::int64_t i4 = 0; i4 < lim; ++i4) {
              for (std::int64_t i5 = 0; i5 < lim; ++i5) {
                for (std::int64_t i6 = 0; i6 < lim; ++i6) {
                  for (std::int64_t i7 = 0; i7 < lim; ++i7) {
                    shape.clear();
                    if (i0 > 0) shape.emplace_back(i0*fac);
                    if (i1 > 0) shape.emplace_back(i1*fac);
                    if (i2 > 0) shape.emplace_back(i2*fac);
                    if (i3 > 0) shape.emplace_back(i3*fac);
                    if (i4 > 0) shape.emplace_back(i4*fac);
                    if (i5 > 0) shape.emplace_back(i5*fac);
                    if (i6 > 0) shape.emplace_back(i6*fac);
                    if (i7 > 0) shape.emplace_back(i7*fac);
                    f(std::span{shape});
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  tensor make_random_view(tensor base) {
    std::mt19937_64& rng {gen};
    if (base.rank() == 0) return base.view();
    bool all_one = true;
    for (auto s : base.shape())
      if (s > 1) { all_one = false; break; }
    if (all_one) return base.view();
    std::vector<std::int64_t> slicable;
    for (std::int64_t d {}; d < base.rank(); ++d)
      if (base.shape()[d] > 1) slicable.push_back(d);
    std::uniform_int_distribution<size_t> dim_dis(0, slicable.size() - 1);
    std::int64_t dim {slicable[dim_dis(rng)]};
    std::int64_t size {base.shape()[dim]};
    std::uniform_int_distribution<std::int64_t> step_dis {2, std::min<std::int64_t>(4, size)};
    std::int64_t step {step_dis(rng)};
    std::int64_t max_start {size - step};
    std::uniform_int_distribution<std::int64_t> start_dis {0, max_start};
    std::int64_t start {start_dis(rng)};
    std::int64_t max_len {(size - start + step - 1)/step};
    std::uniform_int_distribution<std::int64_t> len_dis {1, max_len};
    std::int64_t len {len_dis(rng)};
    return base.view_slice(dim, start, len, step);
  }
}

