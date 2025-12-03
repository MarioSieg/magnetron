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
    constexpr int BINARY_BLOCK_SIZE = 256;

    extern void binary_op_add(const mag_command_t *cmd);
    extern void binary_op_sub(const mag_command_t *cmd);
    extern void binary_op_mul(const mag_command_t *cmd);
    extern void binary_op_div(const mag_command_t *cmd);
    extern void binary_op_mod(const mag_command_t *cmd);
    extern void binary_op_and(const mag_command_t *cmd);
    extern void binary_op_or(const mag_command_t *cmd);
    extern void binary_op_xor(const mag_command_t *cmd);
    extern void binary_op_shl(const mag_command_t *cmd);
    extern void binary_op_shr(const mag_command_t *cmd);
    extern void binary_op_eq(const mag_command_t *cmd);
    extern void binary_op_ne(const mag_command_t *cmd);
    extern void binary_op_le(const mag_command_t *cmd);
    extern void binary_op_ge(const mag_command_t *cmd);
    extern void binary_op_lt(const mag_command_t *cmd);
    extern void binary_op_gt(const mag_command_t *cmd);
}
