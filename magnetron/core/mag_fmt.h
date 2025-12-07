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

/*
** This file contains code derived from LuaJIT's floating-point number formatting implementation.
** LuaJIT Project Readme:
**
**      README for LuaJIT 2.1
**      ---------------------
**
**      LuaJIT is a Just-In-Time (JIT) compiler for the Lua programming language.
**
**      Project Homepage: https://luajit.org/
**
**      LuaJIT is Copyright (C) 2005-2025 Mike Pall.
**      LuaJIT is free software, released under the MIT license.
**      See full Copyright Notice in the COPYRIGHT file or in luajit.h.
**
**      Documentation for LuaJIT is available in HTML format.
**      Please point your favorite browser to:
**
**     doc/luajit.html
**
** Original Source File Notice:
**      String formatting for floating-point numbers.
**      Copyright (C) 2005-2025 Mike Pall. See Copyright Notice in luajit.h
**      Contributed by Peter Cawley.
**
** Modifications:
**   This file includes modifications by Mario Sieg for the
**   Magnetron project. All modifications are licensed under the
**   Apache License, Version 2.0.
*/

#ifndef MAG_FMT_H
#define MAG_FMT_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum mag_format_type_t {
    MAG_FMT_EOF, MAG_FMT_ERR, MAG_FMT_LIT, MAG_FMT_INT,
    MAG_FMT_UINT, MAG_FMT_NUM, MAG_FMT_STR, MAG_FMT_CHAR,
    MAG_FMT_PTR
} mag_format_type_t;

typedef uint32_t mag_format_flags_t; /* Flags for formatting output */

/* Format flags */
#define MAG_FMT_F_LEFT  0x0100 /* Left-align the output */
#define MAG_FMT_F_PLUS  0x0200 /* Prefix positive numbers with a plus sign */
#define MAG_FMT_F_ZERO  0x0400 /* Pad with zeros instead of spaces */
#define MAG_FMT_F_SPACE 0x0800 /* Prefix a space for positive numbers */
#define MAG_FMT_F_ALT   0x1000 /* Alternate format flag */
#define MAG_FMT_F_UPPER 0x2000 /* Use uppercase letters for hex output */

/* Format subtypes (bits reused) */
#define MAG_FMT_T_HEX    0x0010 /* Hexadecimal format for unsigned integers */
#define MAG_FMT_T_OCT    0x0020 /* Octal format for unsigned integers */
#define MAG_FMT_T_FP_A   0x0000 /* 'a' format for floating-point numbers */
#define MAG_FMT_T_FP_E   0x0010 /* 'e' format for floating-point numbers */
#define MAG_FMT_T_FP_F   0x0020 /* 'f' format for floating-point numbers */
#define MAG_FMT_T_FP_G   0x0030 /* 'g' format for floating-point numbers */
#define MAG_FMT_T_QUOTED 0x0010 /* Quoted string format */

#define MAG_FMT_SH_WIDTH 16    /* Shift width for formatting */
#define MAG_FMT_SH_PREC  24    /* Shift precision for formatting */
#define MAG_FMT_TYPE(sf) ((mag_format_type)((sf) & 15))  /* Extract format type */
#define MAG_FMT_WIDTH(sf) (((sf) >> MAG_FMT_SH_WIDTH) & 255u) /* Extract width */
#define MAG_FMT_PREC(sf) ((((sf) >> MAG_FMT_SH_PREC) & 255u) - 1u) /* Extract precision */
#define MAG_FMT_FP(sf) (((sf) >> 4) & 3) /* Extract floating-point format */

/* Formats for conversion characters */
#define MAG_FMT_A (MAG_FMT_NUM|MAG_FMT_T_FP_A) /* 'a' format */
#define MAG_FMT_C (MAG_FMT_CHAR) /* 'c' format */
#define MAG_FMT_D (MAG_FMT_INT)  /* 'd' format */
#define MAG_FMT_E (MAG_FMT_NUM|MAG_FMT_T_FP_E) /* 'e' format */
#define MAG_FMT_F (MAG_FMT_NUM|MAG_FMT_T_FP_F) /* 'f' format */
#define MAG_FMT_G (MAG_FMT_NUM|MAG_FMT_T_FP_G) /* 'g' format */
#define MAG_FMT_I MAG_FMT_D /* 'i' format (same as 'd') */
#define MAG_FMT_O (MAG_FMT_UINT|MAG_FMT_T_OCT) /* 'o' format */
#define MAG_FMT_P (MAG_FMT_PTR) /* 'p' format */
#define MAG_FMT_Q (MAG_FMT_STR|MAG_FMT_T_QUOTED) /* Quoted string */
#define MAG_FMT_S (MAG_FMT_STR) /* 's' format */
#define MAG_FMT_U (MAG_FMT_UINT) /* 'u' format */
#define MAG_FMT_X (MAG_FMT_UINT|MAG_FMT_T_HEX) /* 'x' format */
#define MAG_FMT_G14 (MAG_FMT_G | ((14+1) << MAG_FMT_SH_PREC)) /* 'g' format with precision 14 */
#define MAG_FMT_G5 (MAG_FMT_G | ((5+1) << MAG_FMT_SH_PREC)) /* 'g' format with precision 5 */

#define MAG_FMT_TENSOR_DEFAULT_HEAD_ELEMS 3
#define MAG_FMT_TENSOR_DEFAULT_TAIL_ELEMS 3
#define MAG_FMT_TENSOR_DEFAULT_THRESHOLD 1000
#define MAG_FMT_TENSOR_DEFAULT_LINE_WIDTH 80
#define MAG_FMT_BUF_MAX 64

extern char *mag_fmt_int64(char *p, int64_t n);
extern char *mag_fmt_uint64(char *p, uint64_t n);
extern char *mag_fmt_e11m52(char *p, double n, mag_format_flags_t sf);

#ifdef __cplusplus
}
#endif

#endif