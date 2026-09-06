/* Private bindings for the fused Adam kernel. optim.Adam is the only caller. */

#include "prelude.hpp"

#include "core/mag_fusion.h"

namespace mag::bindings {
  void init_bindings_fused(nb::module_ &m) {
    m.def("_fused_adam_supported", [](const tensor_wrapper &p, const tensor_wrapper &g, const tensor_wrapper &mom, const tensor_wrapper &vel) -> bool {
      std::lock_guard lock {get_global_mutex()};
      return mag_fused_adam_supported(*p, *g, *mom, *vel);
    }, "True when the fused Adam kernel can take these four tensors, without compiling anything.");
    m.def("_fused_adam_step", [](tensor_wrapper &p, const tensor_wrapper &g, tensor_wrapper &mom, tensor_wrapper &vel,
                                 double lr, double beta1, double beta2, double eps, double c1, double c2) -> bool {
      std::lock_guard lock {get_global_mutex()};
      mag_error_t err {};
      return !mag_iserr(mag_fused_adam_step(&err, *p, *g, *mom, *vel, lr, beta1, beta2, eps, c1, c2));
    }, "Apply one Adam step in a single generated kernel, updating the parameter, moment and "
       "variance in place. Returns false when the JIT declines - unsupported operands, or no host "
       "compiler - and the caller must run the eager update instead.");
  }
}
