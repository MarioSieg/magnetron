/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** +---------------------------------------------------------------------+
*/

#pragma once

#include "mag_cuda.cuh"

namespace mag {
    void misc_op_one_hot(const mag_command_t &cmd);
    void misc_op_topk(const mag_command_t &cmd);
    void misc_op_tril(const mag_command_t &cmd);
    void misc_op_triu(const mag_command_t &cmd);
    void misc_op_multinomial(const mag_command_t &cmd);
    void misc_op_cat(const mag_command_t &cmd);
    void misc_op_matmul(const mag_command_t &cmd);
    void misc_op_repeat_back(const mag_command_t &cmd);
    void misc_op_gather(const mag_command_t &cmd);
    void misc_op_where(const mag_command_t &cmd);
}
