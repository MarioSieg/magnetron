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

#include "prelude.hpp"

#include <sstream>

namespace mag::bindings {
  // Resolve the dtype/device kwargs of a window factory, falling back to the context defaults.
  [[nodiscard]] static std::pair<mag_dtype_t, mag_device_id_t> resolve_factory_opts(const std::optional<dtype_wrapper> &dtype, const std::optional<std::string> &device) {
    mag_context_t *ctx = get_ctx();
    mag_dtype_t dt = dtype ? dtype->v : mag_ctx_default_dtype(ctx);
    std::optional<mag_device_id_t> dev = device ? resolve_device_id_str(*device) : std::optional{mag_ctx_default_device(ctx)};
    if (!dev) throw std::runtime_error {"Invalid device id"};
    return {dt, *dev};
  }

  [[nodiscard]] static tensor_wrapper finish_factory(tensor_wrapper t, bool requires_grad) {
    if (requires_grad) {
      mag_error_t err {};
      throw_if_error(mag_tensor_set_requires_grad(&err, *t, true), err);
    }
    return t;
  }

  using window_fn = mag_status_t (mag_error_t *, mag_tensor_t **, mag_context_t *, mag_dtype_t, int64_t, bool, mag_device_id_t);

  template <window_fn *Fn>
  [[nodiscard]] static tensor_wrapper cosine_window(int64_t n, bool periodic, const std::optional<dtype_wrapper> &dtype, const std::optional<std::string> &device, bool requires_grad) {
    auto [dt, dev] = resolve_factory_opts(dtype, device);
    return finish_factory(call_op(Fn, get_ctx(), dt, n, periodic, dev), requires_grad);
  }

  [[nodiscard]] static nb::tuple complex_pair(mag_tensor_t *re, mag_tensor_t *im) {
    tensor_wrapper r {re};
    tensor_wrapper i {im};
    return nb::make_tuple(std::move(r), std::move(i));
  }

  void init_bindings_dsp(nb::module_ &m) {
    auto dsp = m.def_submodule("_dsp", "Digital signal processing operators: windows, Fourier transforms and resampling.");

    dsp.def("i0", [](const tensor_wrapper &x) -> tensor_wrapper {
      return call_op_maybe_without_gil(x, mag_i0, *x);
    }, "x"_a, "Zeroth-order modified Bessel function of the first kind, element-wise.");

    dsp.def("hann_window", &cosine_window<mag_hann_window>,
      "n"_a, "periodic"_a = true, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
      "1-D Hann window of length n. periodic=True for spectral analysis, False for a symmetric filter window.");
    dsp.def("hamming_window", &cosine_window<mag_hamming_window>,
      "n"_a, "periodic"_a = true, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
      "1-D Hamming window of length n. periodic=True for spectral analysis, False for a symmetric filter window.");
    dsp.def("blackman_window", &cosine_window<mag_blackman_window>,
      "n"_a, "periodic"_a = true, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
      "1-D Blackman window of length n. periodic=True for spectral analysis, False for a symmetric filter window.");

    dsp.def("kaiser_window", [](int64_t n, double beta, bool periodic, const std::optional<dtype_wrapper> &dtype, const std::optional<std::string> &device, bool requires_grad) -> tensor_wrapper {
      auto [dt, dev] = resolve_factory_opts(dtype, device);
      return finish_factory(call_op(mag_kaiser_window, get_ctx(), dt, n, beta, periodic, dev), requires_grad);
    }, "n"_a, "beta"_a = 12.0, "periodic"_a = true, nb::kw_only(), "dtype"_a = nb::none(), "device"_a = nb::none(), "requires_grad"_a = false,
    "1-D Kaiser window of length n with shape parameter beta.");

    dsp.def("fft", [](const tensor_wrapper &xr, const std::optional<tensor_wrapper> &xi, int64_t dim) -> nb::tuple {
      mag_tensor_t *re = nullptr, *im = nullptr;
      mag_error_t err {};
      throw_if_error(invoke_auto_gil(xr, [&] { return mag_fft(&err, &re, &im, *xr, xi ? **xi : nullptr, dim); }), err);
      return complex_pair(re, im);
    }, "xr"_a, "xi"_a = nb::none(), "dim"_a = -1,
    "Complex DFT along dim with split real/imaginary parts. xi=None treats the input as real. Returns (re, im).");

    dsp.def("ifft", [](const tensor_wrapper &xr, const std::optional<tensor_wrapper> &xi, int64_t dim) -> nb::tuple {
      mag_tensor_t *re = nullptr, *im = nullptr;
      mag_error_t err {};
      throw_if_error(invoke_auto_gil(xr, [&] { return mag_ifft(&err, &re, &im, *xr, xi ? **xi : nullptr, dim); }), err);
      return complex_pair(re, im);
    }, "xr"_a, "xi"_a = nb::none(), "dim"_a = -1,
    "Inverse complex DFT along dim with split real/imaginary parts. Returns (re, im).");

    dsp.def("rfft", [](const tensor_wrapper &x, int64_t dim) -> nb::tuple {
      mag_tensor_t *re = nullptr, *im = nullptr;
      mag_error_t err {};
      throw_if_error(invoke_auto_gil(x, [&] { return mag_rfft(&err, &re, &im, *x, dim); }), err);
      return complex_pair(re, im);
    }, "x"_a, "dim"_a = -1,
    "Real-input DFT along dim, keeping the n//2+1 non-negative frequency bins. Returns (re, im).");

    dsp.def("irfft", [](const tensor_wrapper &xr, const tensor_wrapper &xi, int64_t n, int64_t dim) -> tensor_wrapper {
      return call_op_maybe_without_gil(xr, mag_irfft, *xr, *xi, n, dim);
    }, "xr"_a, "xi"_a, "n"_a, "dim"_a = -1,
    "Inverse of rfft: real signal of length n along dim from n//2+1 frequency bins.");

    dsp.def("resample", [](const tensor_wrapper &x, int64_t orig_freq, int64_t new_freq, int64_t lowpass_filter_width, double rolloff, std::optional<double> beta) -> nb::object {
      if (orig_freq <= 0 || new_freq <= 0) {
        std::ostringstream oss;
        oss << "Sample rates must be positive, got " << orig_freq << " -> " << new_freq;
        throw nb::value_error(oss.str().c_str());
      }
      if (orig_freq == new_freq) return nb::find(x); /* Identity: hand back the same Python object */
      mag_resample_window_t window = beta ? MAG_RESAMPLE_WINDOW_KAISER : MAG_RESAMPLE_WINDOW_HANN;
      return nb::cast(call_op_maybe_without_gil(x, mag_resample, *x, orig_freq, new_freq, lowpass_filter_width, rolloff, window, beta.value_or(0.0)));
    }, "x"_a, "orig_freq"_a, "new_freq"_a, "lowpass_filter_width"_a = 64, "rolloff"_a = 0.9475937167399596, "beta"_a.none() = 14.769656459379492,
    "Resample along the last dim with a windowed-sinc polyphase filter. beta=None uses a Hann window, otherwise a Kaiser window with that beta.");
  }
}
