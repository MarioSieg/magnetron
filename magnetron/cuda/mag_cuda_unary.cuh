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
    extern void unary_op_abs(const mag_command_t *cmd);
    extern void unary_op_sgn(const mag_command_t *cmd);
    extern void unary_op_neg(const mag_command_t *cmd);
    extern void unary_op_log(const mag_command_t *cmd);
    extern void unary_op_sqr(const mag_command_t *cmd);
    extern void unary_op_sqrt(const mag_command_t *cmd);
    extern void unary_op_sin(const mag_command_t *cmd);
    extern void unary_op_cos(const mag_command_t *cmd);
    extern void unary_op_step(const mag_command_t *cmd);
    extern void unary_op_exp(const mag_command_t *cmd);
    extern void unary_op_floor(const mag_command_t *cmd);
    extern void unary_op_ceil(const mag_command_t *cmd);
    extern void unary_op_round(const mag_command_t *cmd);
    extern void unary_op_softmax(const mag_command_t *cmd);
    extern void unary_op_softmax_dv(const mag_command_t *cmd);
    extern void unary_op_sigmoid(const mag_command_t *cmd);
    extern void unary_op_sigmoid_dv(const mag_command_t *cmd);
    extern void unary_op_hard_sigmoid(const mag_command_t *cmd);
    extern void unary_op_silu(const mag_command_t *cmd);
    extern void unary_op_silu_dv(const mag_command_t *cmd);
    extern void unary_op_tanh(const mag_command_t *cmd);
    extern void unary_op_tanh_dv(const mag_command_t *cmd);
    extern void unary_op_relu(const mag_command_t *cmd);
    extern void unary_op_relu_dv(const mag_command_t *cmd);
    extern void unary_op_gelu(const mag_command_t *cmd);
    extern void unary_op_gelu_dv(const mag_command_t *cmd);
}
