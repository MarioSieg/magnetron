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

#include "mag_operator.h"
#include "mag_gradients.h"

const mag_opmeta_t *mag_op_meta_of(mag_opcode_t opc) {
    static const mag_opmeta_t infos[MAG_OP__NUM] = {
    #define mag_op_backward_NULL NULL
    #define _(enu, in, out, dtm, opp, flags, diff) [MAG_OP_##enu] = (mag_opmeta_t){ \
        #enu, \
        in, \
        out, \
        MAG_DTYPE_MASK_##dtm, \
        opp, \
        flags, \
        mag_op_backward_##diff \
        }
        mag_opdef(_, MAG_SEP)
    #undef _
    #undef mag_op_backward_NULL
    };
    return infos+opc;
}
