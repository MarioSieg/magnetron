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

#pragma once

#include "mag_cuda.cuh"

namespace mag {
    extern void binary_op_add(const mag_command_t *cmd);
    extern void binary_op_sub(const mag_command_t *cmd);
    extern void binary_op_mul(const mag_command_t *cmd);
    extern void binary_op_div(const mag_command_t *cmd);
}
