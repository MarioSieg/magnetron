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

#include "mag_einsum.h"

#include <ctype.h>

#include "mag_alloc.h"
#include "mag_tensor.h"

/*
** The magnetron einsum impl is based on:
** https://github.com/ml-explore/mlx/blob/main/mlx/einsum.cpp
** https://github.com/numpy/numpy/blob/1d49c7f7ff527c696fc26ab2278ad51632a66660/numpy/_core/einsumfunc.py#L743
** https://github.com/dgasmith/opt_einsum
*/

typedef struct mag_einsum_str_t { char buf[64]; } mag_einsum_str_t;

typedef struct mag_parsed_einsum_t {
  mag_einsum_str_t inputs[MAG_EINSUM_MAX_INPUTS];
  uint32_t num_inputs;
  bool has_output;
  mag_einsum_str_t output;
} mag_parsed_einsum_t;

static void mag_einsum_remove_spaces(char *s) {
  char *w = s;
  for (; *s; ++s)
    if (*s != ' ' && *s != '\t' && *s != '\n' && *s != '\r')
      *w++ = *s;
  *w = '\0';
}

static mag_status_t mag_einsum_parse(
  mag_error_t *err,
  char *subscripts,
  mag_parsed_einsum_t *out
) {
  char *arrow = strstr(subscripts, "->");
  char *lhs = NULL;
  char *rhs = NULL;
  if (arrow) {
    *arrow = '\0';
    lhs = subscripts;
    rhs = arrow+2;
    snprintf(out->output.buf, sizeof(out->output.buf), "%s", rhs);
  } else {
    lhs = subscripts;
    int counts[256] = {0};
    bool has_ellipsis = false;
    for (const char *p=subscripts; *p; ++p) {
      if (*p == ',') continue;
      if (*p == '.') {
        mag_contract(err, ERR_EINSUM, {}, p[1] == '.' && p[2] == '.', "Malformed ellipsis in einsum equation string");
        if (!has_ellipsis) has_ellipsis = true;
        p += 2;
        continue;
      }
      counts[(unsigned char)*p]++;
    }
    char tmp[512];
    size_t n=0;
    if (has_ellipsis) {
      memcpy(tmp, "...", sizeof("...")-1);
      n += sizeof("...")-1;
    }
    for (size_t c=0; c < sizeof(counts)/sizeof(*counts); ++c)
      if (counts[c] == 1)
        tmp[n++] = (char)c;
    tmp[n] = '\0';
    snprintf(out->output.buf, sizeof(out->output.buf), "%s", tmp);
  }
  size_t num_inputs=1;
  for (char *p=lhs; *p; ++p)
    if (*p == ',') ++num_inputs;
  size_t idx = 0;
  char *save = NULL;
  char *tok = strtok_r(lhs, ",", &save);
  for (; tok; (tok=strtok_r(NULL, ",", &save)), ++idx) {
    mag_contract(err, ERR_EINSUM, {}, idx < num_inputs, "Einsum parse error");
    size_t len = strlen(tok);
    mag_contract(err, ERR_EINSUM, {}, len <= MAG_EINSUM_MAX_SPEC, "einsum: input spec too long");
    mag_einsum_str_t *input = out->inputs+idx;
    snprintf(input->buf, sizeof(input->buf), "%s", tok);
  }
  mag_contract(err, ERR_EINSUM, {}, num_inputs < UINT32_MAX, "Too many einsum inputs");
  out->num_inputs = (uint32_t)num_inputs;
  mag_contract(err, ERR_EINSUM, {}, idx == num_inputs, "Empty einsum input index");
  out->has_output = arrow != NULL;
  return MAG_STATUS_OK;
}

static int mag_einsum_label_id(char c) {
  if (c >= 'a' && c <= 'z') return c-'a';
  if (c >= 'A' && c <= 'Z') return 26+(c-'A');
  mag_panic("Invalid label character in einsum equation string: '%c'", c);
  return -1;
}

typedef uint64_t mag_einsum_charset_t;

typedef struct mag_einsum_subscript_t {
  mag_einsum_str_t str;
  mag_einsum_charset_t charset;
} mag_einsum_subscript_t;

typedef struct mag_einsum_path_heuristics_t {
  size_t naive_cost;
  size_t naive_scaling;
  size_t opt_cost;
  size_t opt_scaling;
  size_t max_term;
} mag_einsum_path_heuristics_t;

typedef struct mag_einsum_path_node_t {
  mag_einsum_subscript_t inputs[MAG_EINSUM_MAX_INPUTS];
  uint32_t num_inputs;
  mag_einsum_subscript_t output;
  uint32_t positions[MAG_EINSUM_MAX_INPUTS];
} mag_einsum_path_node_t;

static mag_status_t mag_einsum_check_alnum_and_expand_ellipsis(
  mag_error_t *err,
  mag_einsum_str_t *subscript,
  const mag_tensor_t *operand,
  size_t operand_idx,
  const char *remaining_chars,
  size_t remaining_len,
  int *max_ellipsis_len
) {
  bool ellipsis = false;
  int pre_count=0;
  int post_count=0;
  const char *str = subscript->buf;
  size_t len = strlen(str);
  for (size_t i=0; i < len; ++i) {
    if (!isalpha(str[i])) {
      mag_contract(err, ERR_EINSUM, {},
        !(i+2 >= len || str[i] != '.' || str[i+1] != '.' || str[i+2] != '.'),
        "Subscripts must be letters, but got: %c", str[i]
      );
      mag_contract(err, ERR_EINSUM, {},
        !ellipsis,
        "Only one ellipsis per subscript is allowed but found more in: %s", str
      );
      ellipsis = true;
      i += 2;
      continue;
    }
    if (ellipsis) ++post_count;
    else ++pre_count;
  }
  if (!ellipsis) return MAG_STATUS_OK;
  int ellipsis_len;
  if (operand) {
    ellipsis_len = (int)operand->coords.rank-pre_count-post_count;
    char shape_fmt[MAG_FMT_DIM_BUF_SIZE];
    if (mag_unlikely(ellipsis_len < 0))
      mag_fmt_shape(&shape_fmt, &operand->coords.shape, operand->coords.rank);
    mag_contract(err, ERR_EINSUM, {},
      ellipsis_len >= 0,
      "Operand %zu with shape %s has insufficient dimensions for subscript %s. "
      "The ellipsis requires at least %d dimensions but the operand has %d dimensions.",
      operand_idx, shape_fmt, subscript->buf, pre_count+post_count, (int)operand->coords.rank
    );
    *max_ellipsis_len = mag_xmax(*max_ellipsis_len, ellipsis_len);
  } else ellipsis_len = *max_ellipsis_len;
  size_t prefix_len = (size_t)pre_count;
  size_t repl_len = (size_t)ellipsis_len;
  size_t suffix_start = prefix_len+3;
  size_t suffix_len = len - suffix_start;
  mag_contract(err, ERR_EINSUM, {}, prefix_len+repl_len+suffix_len < sizeof(subscript->buf), "Expanded subscript too long");
  const char *replacement = remaining_chars + remaining_len - repl_len;
  char tmp[sizeof(subscript->buf)];
  memcpy(tmp, str, prefix_len);
  memcpy(tmp+prefix_len, replacement, repl_len);
  memcpy(tmp+prefix_len+repl_len, str+suffix_start, suffix_len);
  tmp[prefix_len+repl_len+suffix_len] = '\0';
  memcpy(subscript->buf, tmp, prefix_len+repl_len+suffix_len+1);
  return MAG_STATUS_OK;
}

typedef struct mag_einsum_dim_map_t {
  int64_t dims[52];
  mag_einsum_charset_t present;
} mag_einsum_dim_map_t;

static size_t mag_einsum_term_size(const char *term, const mag_einsum_dim_map_t *dim_map) {
  size_t size=1;
  for (const char *p=term; *p; ++p)
    size *= (size_t)dim_map->dims[mag_einsum_label_id(*p)];
  return size;
}

static size_t mag_einsum_term_size_str(const char *term, const mag_einsum_dim_map_t *dim_map) {
  size_t size=1;
  for (const char *p=term; *p; ++p)
    size *= (size_t)dim_map->dims[mag_einsum_label_id(*p)];
  return size;
}

static size_t mag_einsum_term_size_set(mag_einsum_charset_t term, const mag_einsum_dim_map_t *dim_map) {
  size_t size=1;
  for (int id=0; id < 52; ++id)
    if (term&(1ull<<id))
      size *= (size_t)dim_map->dims[id];
  return size;
}

static size_t mag_einsum_flop_count(mag_einsum_charset_t term, bool inner, uint32_t num_terms, const mag_einsum_dim_map_t *dim_map) {
  size_t size = mag_einsum_term_size_set(term, dim_map);
  uint32_t op_factor = 1;
  if ((num_terms-1) > op_factor) op_factor = num_terms - 1;
  if (inner) op_factor += 1;
  return size*(size_t)op_factor;
}

static void mag_einsum_compute_cost_and_scaling(
  const mag_einsum_subscript_t *inputs,
  uint32_t num_inputs,
  const mag_einsum_subscript_t *output,
  const mag_einsum_dim_map_t *dim_map,
  size_t *out_cost,
  size_t *out_scaling
) {
  mag_einsum_charset_t contractions = 0;
  for (uint32_t i=0; i < num_inputs; ++i)
    contractions |= inputs[i].charset;
  bool inner = false;
  for (int id=0; id < 52; ++id) {
    if ((contractions&(1ull<<id)) && !(output->charset&(1ull<<id))) {
      inner = true;
      break;
    }
  }
  *out_cost = mag_einsum_flop_count(
    contractions,
    inner,
    num_inputs,
    dim_map
  );
  *out_scaling = (size_t)mag_popcnt64(contractions);
}

static mag_status_t mag_einsum_compute_path(
  mag_error_t *err,
  char *equation,
  const mag_tensor_t **args,
  size_t num_args,
  mag_einsum_path_heuristics_t *out_heuristics,
  mag_einsum_path_node_t *out_nodes,
  size_t *out_num_nodes
) {
  mag_parsed_einsum_t parsed = {0};
  mag_try(mag_einsum_parse(err, equation, &parsed));
  mag_contract(err, ERR_EINSUM, {}, parsed.num_inputs == num_args, "Number of operands does not match number of input subscripts: %u != %zu", parsed.num_inputs, num_args);
  mag_einsum_charset_t used_chars = 0;
  for (const char *p = equation; *p; ++p) {
    if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z'))
      used_chars|=1ull<<mag_einsum_label_id(*p);
  }
  char rem_chars[53];
  size_t rem_len = 0;
  for (char c = 'a'; c <= 'z'; ++c)
    if (!(used_chars&(1ull<<mag_einsum_label_id(c))))
      rem_chars[rem_len++] = c;
  for (char c = 'A'; c <= 'Z'; ++c)
    if (!(used_chars&(1ull<<mag_einsum_label_id(c))))
      rem_chars[rem_len++] = c;
  rem_chars[rem_len] = '\0';
  int max_ellipsis_len = 0;
  for (size_t i=0; i < parsed.num_inputs; ++i) {
    mag_try(mag_einsum_check_alnum_and_expand_ellipsis(
      err,
      &parsed.inputs[i],
      args[i],
      i,
      rem_chars,
      rem_len,
      &max_ellipsis_len
    ));
  }
  mag_try(mag_einsum_check_alnum_and_expand_ellipsis(
    err,
    &parsed.output,
    NULL,
    0,
    rem_chars,
    rem_len,
    &max_ellipsis_len
  ));
  mag_einsum_charset_t out_set = 0;
  const char *out_str = parsed.output.buf;
  size_t out_len = strlen(out_str);
  for (size_t i=0; i < out_len; ++i) {
    char c = out_str[i];
    mag_contract(err, ERR_EINSUM, {}, isalpha((unsigned char)c), "Subscripts must be letters, but got: %c", c);
    out_set|=1ull<<mag_einsum_label_id(c);
  }
  mag_contract(err, ERR_EINSUM, {}, (size_t)mag_popcnt64(out_set) == out_len, "Repeat indices not allowed in output.");
  mag_einsum_dim_map_t dim_map = {0};
  mag_einsum_subscript_t inputs[MAG_EINSUM_MAX_INPUTS];
  for (size_t i=0; i < parsed.num_inputs; ++i) {
    const char *in = parsed.inputs[i].buf;
    size_t in_len = strlen(in);
    int64_t ndim = args[i]->coords.rank;
    mag_einsum_charset_t in_set = 0;
    for (size_t j = 0; j < in_len; ++j)
      in_set |= 1ull << mag_einsum_label_id(in[j]);
    inputs[i].str = parsed.inputs[i];
    inputs[i].charset = in_set;
    mag_contract(err, ERR_EINSUM, {}, in_len == (size_t)ndim, "Invalid number of subscripts %zu for input %zu with %d dimensions.", in_len, i, (int)ndim);
    if (mag_popcnt64(in_set) < in_len) {
      int64_t local_dims[52] = {0};
      mag_einsum_charset_t local_present = 0;
      for (size_t j = 0; j < in_len; ++j) {
        char c = in[j];
        int id = mag_einsum_label_id(c);
        int64_t dim = args[i]->coords.shape[j];
        if (local_present&(1ull<<id)) {
          mag_contract(err, ERR_EINSUM, {}, local_dims[id] == dim,
            "Dimensions of repeated subscripts do not have the same size (%lld != %lld).",
            (long long)local_dims[id],
            (long long)dim);
        } else {
          local_present |= 1ull << id;
          local_dims[id] = dim;
        }
      }
    }
    for (size_t j = 0; j < in_len; ++j) {
      char c = in[j];
      int id = mag_einsum_label_id(c);
      int64_t dim = args[i]->coords.shape[j];
      if (dim_map.present&(1ull<<id)) {
        int64_t old = dim_map.dims[id];
        mag_contract(err, ERR_EINSUM, {},
          dim == 1 || old == 1 || old == dim,
          "Cannot broadcast dimension %zu of input %zu to size %" PRIi64 ".",
          j, i, old);
        if (dim > old)
          dim_map.dims[id] = dim;
      } else {
        dim_map.present |= 1ull << id;
        dim_map.dims[id] = dim;
      }
    }
  }
  for (size_t i=0; i < out_len; ++i) {
    mag_contract(err, ERR_EINSUM, {},
      dim_map.present & (1ull << mag_einsum_label_id(out_str[i])),
      "Output subscript '%c' does not appear in any input.", out_str[i]);
  }
  size_t max_size = mag_einsum_term_size(parsed.output.buf, &dim_map);
  for (size_t i=0; i < parsed.num_inputs; ++i) {
    size_t s = mag_einsum_term_size(parsed.inputs[i].buf, &dim_map);
    if (s > max_size)
      max_size = s;
  }
  mag_einsum_subscript_t output = {
    .str = parsed.output,
    .charset = out_set,
  };
  memset(out_heuristics, 0, sizeof(*out_heuristics));
  out_heuristics->max_term = max_size;
  mag_einsum_compute_cost_and_scaling(
    inputs,
    parsed.num_inputs,
    &output,
    &dim_map,
    &out_heuristics->naive_cost,
    &out_heuristics->naive_scaling
  );
  out_heuristics->opt_cost = out_heuristics->naive_cost;
  out_heuristics->opt_scaling = out_heuristics->naive_scaling;
  memset(out_nodes, 0, sizeof(out_nodes[0]));
  out_nodes[0].num_inputs = parsed.num_inputs;
  for (size_t i=0; i < parsed.num_inputs; ++i) {
    out_nodes[0].inputs[i] = inputs[i];
    out_nodes[0].positions[i] = (uint32_t)i;
  }
  out_nodes[0].output = output;
  *out_num_nodes = 1;
  return MAG_STATUS_OK;
}

typedef struct mag_einsum_char_axis_t {
  char c;
  int64_t ax;
} mag_einsum_char_axis_t;

static int mag_einsum_pair_cmp_by_canonical_axis(const void *a, const void *b, void *usr) {
  const mag_einsum_char_axis_t *x = a;
  const mag_einsum_char_axis_t *y = b;
  int64_t ax = ((const int64_t *)usr)[mag_einsum_label_id(x->c)];
  int64_t ay = ((const int64_t *)usr)[mag_einsum_label_id(y->c)];
  return (ax>ay) - (ax<ay);
}

static bool mag_einsum_axes_sorted_by_original_axis(const mag_einsum_char_axis_t *xs, int64_t n) {
  for (int64_t i=1; i < n; ++i)
    if (xs[i-1].ax > xs[i].ax)
      return false;
  return true;
}

static void mag_einsum_sort_str_ax(mag_einsum_char_axis_t *xs, int64_t n, const int64_t char_to_ax[52]) {
  for (int64_t i=1; i < n; ++i) {
    mag_einsum_char_axis_t v = xs[i];
    int64_t vkey = char_to_ax[mag_einsum_label_id(v.c)];
    int64_t j = i - 1;
    while (j >= 0) {
      int64_t jkey = char_to_ax[mag_einsum_label_id(xs[j].c)];
      if (jkey <= vkey)
        break;
      xs[j+1] = xs[j];
      --j;
    }
    xs[j+1] = v;
  }
}

static mag_status_t mag_einsum_collapse_repeats(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_einsum_subscript_t *subscript,
  mag_tensor_t *x
) {
  const char *str = subscript->str.buf;
  int64_t rank = mag_tensor_rank(x);
  const int64_t *shape = mag_tensor_shape_ptr(x);
  const int64_t *strides = mag_tensor_strides_ptr(x);
  int64_t new_shape[MAG_EINSUM_MAX_SPEC];
  int64_t new_strides[MAG_EINSUM_MAX_SPEC];
  char new_str[MAG_EINSUM_MAX_SPEC];
  uint64_t seen = 0;
  int64_t new_rank = 0;
  for (int64_t i=0; i < rank; ++i) {
    int id = mag_einsum_label_id(str[i]);
    if (seen & (1ull<<id)) continue;
    seen|=(1ull<<id);
    int64_t dim = shape[i];
    int64_t stride_sum = 0;
    for (int64_t j=i; j < rank; ++j) {
      if (str[j] == str[i]) {
        mag_contract(err, ERR_EINSUM, {}, shape[j] == dim, "Dimensions of repeated subscripts do not have the same size.");
        stride_sum += strides[j];
      }
    }
    new_shape[new_rank] = dim;
    new_strides[new_rank] = stride_sum;
    new_str[new_rank] = str[i];
    ++new_rank;
  }
  new_str[new_rank] = '\0';
  mag_try(mag_as_strided(
    err,
    out,
    x->ctx,
    x,
    new_rank,
    new_shape,
    new_strides,
    (int64_t)mag_tensor_data_offset(x)
  ));
  snprintf(subscript->str.buf, sizeof(subscript->str.buf), "%s", new_str);
  subscript->charset = seen;
  return MAG_STATUS_OK;
}

static mag_status_t mag_einsum_naive(mag_error_t *err, mag_tensor_t **out_result, mag_einsum_path_node_t *node, mag_tensor_t **operands) {
  for (uint32_t i=0; i < node->num_inputs; ++i) {
    uint32_t pos = node->positions[i];
    if ((size_t)mag_popcnt64(node->inputs[i].charset) <
        strlen(node->inputs[i].str.buf)) {
      mag_tensor_t *collapsed = NULL;
      mag_try(mag_einsum_collapse_repeats(
        err,
        &collapsed,
        &node->inputs[i],
        operands[pos]
      ));
      mag_tensor_decref(operands[pos]);
      operands[pos] = collapsed;
    }
  }
  int64_t char_to_ax[52];
  for (int i=0; i < 52; ++i)
    char_to_ax[i] = -1;
  int64_t num_axes = 0;
  for (uint32_t i=0; i < node->num_inputs; ++i) {
    const char *s = node->inputs[i].str.buf;
    for (const char *p=s; *p; ++p) {
      int id = mag_einsum_label_id(*p);
      if (char_to_ax[id] < 0) char_to_ax[id] = num_axes++;
    }
  }
  for (uint32_t i=0; i < node->num_inputs; ++i) {
    uint32_t pos = node->positions[i];
    mag_tensor_t *op = operands[pos];
    int64_t op_rank = op->coords.rank;
    if (op_rank != num_axes) {
      const int64_t *old_shape = mag_tensor_shape_ptr(op);
      int64_t shape[MAG_EINSUM_MAX_SPEC];
      for (int64_t ax=0; ax < op_rank; ++ax)
        shape[ax] = old_shape[ax];
      for (int64_t ax=op_rank; ax < num_axes; ++ax)
        shape[ax] = 1;
      mag_tensor_t *tmp = NULL;
      mag_try(mag_reshape(err, &tmp, op, shape, num_axes));
      mag_tensor_decref(operands[pos]);
      operands[pos] = tmp;
      op = tmp;
    }
    mag_einsum_char_axis_t str_ax[MAG_EINSUM_MAX_SPEC];
    int64_t str_ax_len = 0;
    const char *str = node->inputs[i].str.buf;
    for (const char *p=str; *p; ++p) {
      int64_t ax = str_ax_len;
      str_ax[str_ax_len++] = (mag_einsum_char_axis_t){.c = *p, .ax = ax,};
    }
    for (int id=0; id < 52; ++id) {
      if (char_to_ax[id] < 0)
        continue;
      char c = id < 26 ? (char)('a' + id) : (char)('A' + id - 26);
      if (!(node->inputs[i].charset&(1ull<<id))) {
        int64_t ax = str_ax_len;
        str_ax[str_ax_len++] = (mag_einsum_char_axis_t){.c = c, .ax = ax,};
      }
    }
    mag_einsum_sort_str_ax(str_ax, str_ax_len, char_to_ax);
    if (mag_einsum_axes_sorted_by_original_axis(str_ax, str_ax_len))
      continue;
    int64_t reorder[MAG_EINSUM_MAX_SPEC];
    for (int64_t ax=0; ax < str_ax_len; ++ax)
      reorder[ax] = str_ax[ax].ax;
    mag_tensor_t *tmp = NULL;
    mag_try(mag_permute(err, &tmp, op, reorder, str_ax_len));
    mag_tensor_decref(operands[pos]);
    operands[pos] = tmp;
  }
  mag_tensor_t *out = operands[node->positions[0]];
  mag_tensor_incref(out);
  for (uint32_t i=1; i < node->num_inputs; ++i) {
    mag_tensor_t *tmp = NULL;
    mag_try(mag_mul(err, &tmp, out, operands[node->positions[i]]));
    mag_tensor_decref(out);
    out = tmp;
  }
  int64_t sum_axes[MAG_EINSUM_MAX_SPEC];
  int64_t num_sum_axes = 0;
  for (int id=0; id < 52; ++id) {
    if (char_to_ax[id] < 0)
      continue;
    if (!(node->output.charset&(1ull<<id)))
      sum_axes[num_sum_axes++] = char_to_ax[id];
  }
  if (num_sum_axes > 0) {
    mag_tensor_t *tmp = NULL;
    mag_try(mag_sum(err, &tmp, out, sum_axes, num_sum_axes, false));
    mag_tensor_decref(out);
    out = tmp;
  }
  const char *out_str = node->output.str.buf;
  int64_t out_rank = (int64_t)strlen(out_str);
  if (out_rank <= 1) {
    *out_result = out;
    return MAG_STATUS_OK;
  }
  int64_t reorder[MAG_EINSUM_MAX_SPEC];
  for (int64_t i=0; i < out_rank; ++i) {
    int id = mag_einsum_label_id(out_str[i]);
    reorder[i] = char_to_ax[id];
    int64_t offset = 0;
    for (int64_t j = 0; j < num_sum_axes; ++j)
      if (reorder[i] > sum_axes[j])
        ++offset;
    reorder[i] -= offset;
  }
  bool identity = true;
  for (int64_t i=0; i < out_rank; ++i) {
    if (reorder[i] != i) {
      identity = false;
      break;
    }
  }
  if (identity) {
    *out_result = out;
  } else {
    mag_tensor_t *tmp = NULL;
    mag_try(mag_permute(err, &tmp, out, reorder, out_rank));
    mag_tensor_decref(out);
    *out_result = tmp;
  }
  return MAG_STATUS_OK;
}

static MAG_COLDPROC void mag_einsum_debug_print_path(
  const char *equation,
  const mag_einsum_path_heuristics_t *h,
  const mag_einsum_path_node_t *nodes,
  size_t num_nodes
) {
  printf("einsum: %s\n", equation);
  printf("  naive cost:       %zu\n", h->naive_cost);
  printf("  naive scaling:    %zu\n", h->naive_scaling);
  printf("  optimized cost:   %zu\n", h->opt_cost);
  printf("  optimized scaling:%zu\n", h->opt_scaling);
  printf("  max term:         %zu\n", h->max_term);
  printf("  nodes:            %zu\n", num_nodes);
  for (size_t n=0; n < num_nodes; ++n) {
    const mag_einsum_path_node_t *node = &nodes[n];
    printf("  node[%zu]: ", n);
    for (uint32_t i=0; i < node->num_inputs; ++i) {
      if (i) printf(", ");
      printf("%u:%s", node->positions[i], node->inputs[i].str.buf);
    }
    printf(" -> %s\n", node->output.str.buf);
  }
}

mag_status_t mag_einsum_eval(
  mag_error_t *err,
  mag_tensor_t **out_result,
  const char *equation,
  const mag_tensor_t **args,
  size_t num_args
) {
  size_t len = strlen(equation);
  mag_contract(err, ERR_EINSUM, {}, mag_utf8_validate((const uint8_t *)equation, len), "Invalid UTF-8 in equation string");
  mag_contract(err, ERR_EINSUM, {}, num_args > 0, "At least one input tensor is required");
  char *cloned = mag_strdup(equation);
  mag_einsum_remove_spaces(cloned);
  len = strlen(cloned);
  mag_contract(err, ERR_EINSUM, {}, len > 0, "Empty equation string");
  mag_einsum_path_heuristics_t heuristics = {0};
  mag_einsum_path_node_t nodes[MAG_EINSUM_MAX_INPUTS] = {0};
  size_t num_nodes = 0;
  mag_status_t st = mag_einsum_compute_path(
    err,
    cloned,
    args,
    num_args,
    &heuristics,
    nodes,
    &num_nodes
  );
  if (st != MAG_STATUS_OK) {
    (*mag_alloc)(cloned, 0, 0);
    return st;
  }
  #if MAG_DEBUG
    mag_einsum_debug_print_path(...);
  #endif
  mag_tensor_t *operands[MAG_EINSUM_MAX_INPUTS];
  for (size_t i=0; i < num_args; ++i) {
    operands[i] = (mag_tensor_t *)args[i];
    mag_tensor_incref(operands[i]);
  }
  st = mag_einsum_naive(err, out_result, &nodes[0], operands);
  for (size_t i=0; i < num_args; ++i)
    mag_tensor_decref(operands[i]);
  (*mag_alloc)(cloned, 0, 0);
  return st;
}
