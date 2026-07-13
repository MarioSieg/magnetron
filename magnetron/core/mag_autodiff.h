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

#ifndef MAG_AUTODIFF_H
#define MAG_AUTODIFF_H

#include "mag_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_AU_STATE_INTRUSIVE_STORAGE_NUM 4 /* How many tensors to store inline before moving to dynamic mem */

/* Autodiff state for parameters */
struct mag_au_state_t {
  MAG_RC_INJECT_HEADER; /* RC Control block must be first */

  mag_context_t *ctx;
  mag_opcode_t op;
  mag_tensor_t **in;
  mag_tensor_t *in_intrusive[MAG_AU_STATE_INTRUSIVE_STORAGE_NUM];
  uint32_t num_in;
  uint32_t cap_in;
  mag_op_params_t *params;
  mag_tensor_t *grad;
};
MAG_RC_OBJECT_IS_VALID(mag_au_state_t);

extern mag_au_state_t *mag_au_state_lazy_alloc(mag_au_state_t **au, mag_context_t *ctx);
extern bool mag_au_state_reserve_more_input_cap(mag_au_state_t *au, uint32_t extra);
extern bool mag_au_state_set_op_params(mag_au_state_t *au, const mag_op_params_t *params);
extern bool mag_au_state_set_input(mag_au_state_t *au, mag_tensor_t *x);

#ifdef __cplusplus
}
#endif

#endif
