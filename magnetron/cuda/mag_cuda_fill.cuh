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
    constexpr int FILL_BLOCK_SIZE = 256;

    extern void fill_op_fill(const mag_command_t *cmd);
    extern void fill_op_masked_fill(const mag_command_t *cmd);
    extern void fill_op_fill_rand_uniform(const mag_command_t *cmd);
    extern void fill_op_fill_rand_normal(const mag_command_t *cmd);
}
