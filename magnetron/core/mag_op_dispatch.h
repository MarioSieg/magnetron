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

#ifndef MAG_OP_DISPATCH_H
#define MAG_OP_DISPATCH_H

#include "mag_def.h"
#include "mag_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

extern mag_status_t mag_dispatch(
  mag_error_t *err,
  mag_opcode_t op,
  bool inplace,
  mag_tensor_t **in,
  uint32_t num_in,
  mag_tensor_t **out,
  uint32_t num_out,
  const mag_op_params_t *params
);

#ifdef __cplusplus
}
#endif
#endif
