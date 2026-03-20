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
#include "mag_reduce_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Autodiff state for parameters */
struct mag_au_state_t {
    MAG_RC_INJECT_HEADER; /* RC Control block must be first */

    mag_context_t *ctx;
    mag_opcode_t op;
    mag_tensor_t *op_inputs[MAG_MAX_OP_INPUTS];
    mag_op_attr_t op_attrs[MAG_MAX_OP_PARAMS];
    uint32_t op_num_inputs;
    uint32_t op_num_outputs;
    uint32_t op_num_attrs;
    bool op_inplace;
    mag_reduce_plan_t *owned_reduce_plan;
    mag_tensor_t *grad;
};
MAG_RC_OBJECT_IS_VALID(mag_au_state_t);

extern mag_au_state_t *mag_au_state_lazy_alloc(mag_au_state_t **au_state, mag_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
