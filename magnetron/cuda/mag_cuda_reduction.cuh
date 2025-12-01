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
    constexpr int REDUCTION_BLOCK_SIZE = 256;

    extern void reduce_op_mean(const mag_command_t *cmd);
    extern void reduce_op_min(const mag_command_t *cmd);
    extern void reduce_op_max(const mag_command_t *cmd);
    extern void reduce_op_sum(const mag_command_t *cmd);
    extern void reduce_op_prod(const mag_command_t *cmd);
}