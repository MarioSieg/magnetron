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

#define MAG_EIN_MAX_INPUTS 64
#define MAG_EIN_MAX_SPEC 128
#define MAG_EIN_STR_BUF_LEN (MAG_EIN_MAX_SPEC+1)
#define MAG_EIN_NUM_LETTERS (26+26) /* a-z + A-Z */
#define MAG_EIN_ASCII_TABLE_SIZE 256
#define MAG_EIN_MAX_CONTRACTIONS ((MAG_EIN_MAX_INPUTS*(MAG_EIN_MAX_INPUTS-1))>>1)

mag_static_assert(MAG_EIN_STR_BUF_LEN >= MAG_EIN_NUM_LETTERS+sizeof("...")-1+1);

static int mag_ein_label_id(char c) {
  if (c >= 'a' && c <= 'z') return c - 'a';
  if (c >= 'A' && c <= 'Z') return 26 + (c - 'A');
  mag_assert(false, "einsum: invalid label character '%c'.", c);
  return -1;
}

typedef uint64_t mag_ein_charset_t;
#define mag_ein_charset_add_bit(set, id) ((set)|=(1ull<<(id)))
#define mag_ein_charset_add(set, c) mag_ein_charset_add_bit(set, mag_ein_label_id(c))
#define mag_ein_charset_has_bit(set, id) ((set)&(1ull<<(id)))
#define mag_ein_charset_has(set, c) mag_ein_charset_has_bit(set, mag_ein_label_id(c))
#define mag_ein_charset_union(a,b) ((a)|(b))
#define mag_ein_charset_intersects(a,b) ((a)&(b))
#define mag_ein_chatset_len(set) (mag_popcnt64(set))

typedef struct mag_ein_str_t { char buf[MAG_EIN_STR_BUF_LEN]; } mag_ein_str_t;

typedef struct mag_parsed_ein_t {
  mag_ein_str_t inputs[MAG_EIN_MAX_INPUTS];
  uint32_t num_inputs;
  bool has_output;
  mag_ein_str_t output;
} mag_parsed_ein_t;

typedef struct mag_ein_subscript_t {
  mag_ein_str_t str;
  mag_ein_charset_t charset;
} mag_ein_subscript_t;

typedef struct mag_ein_path_heuristics_t {
  size_t naive_cost;
  size_t naive_scaling;
  size_t opt_cost;
  size_t opt_scaling;
  size_t max_term;
} mag_ein_path_heuristics_t;

typedef struct mag_ein_path_node_t {
  mag_ein_subscript_t inputs[MAG_EIN_MAX_INPUTS];
  uint32_t num_inputs;
  mag_ein_subscript_t output;
  uint32_t positions[MAG_EIN_MAX_INPUTS];
} mag_ein_path_node_t;

typedef struct mag_ein_dim_map_t {
  int64_t dims[MAG_EIN_NUM_LETTERS];
  mag_ein_charset_t present;
} mag_ein_dim_map_t;

typedef struct mag_ein_axes_t {
  int64_t v[MAG_EIN_MAX_SPEC];
  int64_t n;
} mag_ein_axes_t;

typedef struct mag_ein_contraction_t {
  int64_t size;
  size_t cost;
  uint32_t dims;
  uint32_t x;
  uint32_t y;
  mag_ein_charset_t output;
} mag_ein_contraction_t;

typedef struct mag_ein_char_axis_t {
  char c;
  int64_t ax;
} mag_ein_char_axis_t;

static void mag_ein_remove_spaces(char *s) {
  char *w = s;
  for (; *s; ++s)
    if (*s != ' ' && *s != '\t' && *s != '\n' && *s != '\r')
      *w++ = *s;
  *w = '\0';
}

static mag_status_t mag_ein_parse(
  mag_error_t *err,
  char *subscripts,
  mag_parsed_ein_t *out
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
    int counts[MAG_EIN_ASCII_TABLE_SIZE] = {0};
    bool has_ellipsis = false;
    for (const char *p=subscripts; *p; ++p) {
      if (*p == ',') continue;
      if (*p == '.') {
        if (mag_unlikely(!(p[1] == '.' && p[2] == '.')))
          return mag_set_error(err, MAG_ERR_EINSUM, "einsum: malformed ellipsis in equation; expected '...'.");
        if (!has_ellipsis) has_ellipsis = true;
        p += 2;
        continue;
      }
      counts[(unsigned char)*p]++;
    }
    char tmp[MAG_EIN_STR_BUF_LEN];
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
    if (mag_unlikely(!(idx < num_inputs)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: failed to parse input subscripts.");
    size_t len = strlen(tok);
    if (mag_unlikely(!(len < MAG_EIN_MAX_SPEC)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: input subscript too long (max %d characters).", MAG_EIN_MAX_SPEC);
    mag_ein_str_t *input = out->inputs+idx;
    snprintf(input->buf, sizeof(input->buf), "%s", tok);
  }
  if (mag_unlikely(!(num_inputs < UINT32_MAX)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: too many input subscripts.");
  out->num_inputs = (uint32_t)num_inputs;
  if (mag_unlikely(!(idx == num_inputs)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: input subscript list contains an empty entry.");
  out->has_output = arrow != NULL;
  return MAG_OK;
}

static mag_status_t mag_ein_check_alnum_and_expand_ellipsis(
  mag_error_t *err,
  mag_ein_str_t *subscript,
  const mag_tensor_t *operand,
  size_t operand_idx,
  const char *remaining_chars,
  size_t remaining_len,
  int64_t *max_ellipsis_len
) {
  bool ellipsis = false;
  size_t pre_count=0;
  size_t post_count=0;
  const char *str = subscript->buf;
  size_t len = strlen(str);
  for (size_t i=0; i < len; ++i) {
    if (!isalpha((unsigned char)str[i])) {
      if (mag_unlikely(i+2 >= len || str[i] != '.' || str[i+1] != '.' || str[i+2] != '.'))
        return mag_set_error(err, MAG_ERR_EINSUM, "einsum: subscripts must be letters, but got '%c'.", str[i]);
      if (mag_unlikely(ellipsis))
        return mag_set_error(err, MAG_ERR_EINSUM, "einsum: only one ellipsis is allowed per subscript, but found more in '%s'.", str);
      ellipsis = true;
      i += 2;
      continue;
    }
    if (ellipsis) ++post_count;
    else ++pre_count;
  }
  if (!ellipsis) return MAG_OK;
  int64_t ellipsis_len;
  if (operand) {
    ellipsis_len = operand->coords.rank-(int64_t)pre_count-(int64_t)post_count;
    char shape_fmt[MAG_FMT_DIM_BUF_SIZE];
    if (mag_unlikely(ellipsis_len < 0))
      mag_fmt_shape(&shape_fmt, &operand->coords.shape, operand->coords.rank);
    if (mag_unlikely(!(ellipsis_len >= 0)))
      return mag_set_error(err, MAG_ERR_EINSUM,
        "einsum: operand %zu with shape %s has too few dimensions for subscript '%s' (needs at least %zu, got %zu).",
        operand_idx, shape_fmt, subscript->buf, pre_count+post_count, (size_t)operand->coords.rank
      );
    *max_ellipsis_len = mag_xmax(*max_ellipsis_len, ellipsis_len);
  } else ellipsis_len = *max_ellipsis_len;
  size_t prefix_len = pre_count;
  size_t repl_len = ellipsis_len;
  size_t suffix_start = prefix_len+3;
  size_t suffix_len = len - suffix_start;
  if (mag_unlikely(!(prefix_len+repl_len+suffix_len < sizeof(subscript->buf))))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: subscript is too long after ellipsis expansion.");
  if (mag_unlikely(!(repl_len <= remaining_len)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: too many dimensions for ellipsis expansion.");
  const char *replacement = remaining_chars + remaining_len - repl_len;
  char tmp[sizeof(subscript->buf)];
  memcpy(tmp, str, prefix_len);
  memcpy(tmp+prefix_len, replacement, repl_len);
  memcpy(tmp+prefix_len+repl_len, str+suffix_start, suffix_len);
  tmp[prefix_len+repl_len+suffix_len] = '\0';
  memcpy(subscript->buf, tmp, prefix_len+repl_len+suffix_len+1);
  return MAG_OK;
}

static size_t mag_ein_term_size(const char *term, const mag_ein_dim_map_t *dim_map) {
  size_t size=1;
  for (const char *p=term; *p; ++p)
    size *= (size_t)dim_map->dims[mag_ein_label_id(*p)];
  return size;
}

static size_t mag_ein_term_size_set(mag_ein_charset_t term, const mag_ein_dim_map_t *dim_map) {
  size_t size=1;
  for (int id=0; id < MAG_EIN_NUM_LETTERS; ++id)
    if (mag_ein_charset_has_bit(term, id))
      size *= (size_t)dim_map->dims[id];
  return size;
}

static size_t mag_ein_flop_count(mag_ein_charset_t term, bool inner, uint32_t num_terms, const mag_ein_dim_map_t *dim_map) {
  size_t size = mag_ein_term_size_set(term, dim_map);
  uint32_t op_factor = 1;
  if ((num_terms-1) > op_factor) op_factor = num_terms-1;
  if (inner) op_factor += 1;
  return size*(size_t)op_factor;
}

static void mag_ein_compute_cost_and_scaling(
  const mag_ein_subscript_t *inputs,
  uint32_t num_inputs,
  const mag_ein_subscript_t *output,
  const mag_ein_dim_map_t *dim_map,
  size_t *out_cost,
  size_t *out_scaling
) {
  mag_ein_charset_t contractions = 0;
  for (uint32_t i=0; i < num_inputs; ++i)
    contractions = mag_ein_charset_union(contractions, inputs[i].charset);
  bool inner = false;
  for (int id=0; id < MAG_EIN_NUM_LETTERS; ++id) {
    if (mag_ein_charset_has_bit(contractions, id) && !(mag_ein_charset_has_bit(output->charset, id))) {
      inner = true;
      break;
    }
  }
  *out_cost = mag_ein_flop_count(
    contractions,
    inner,
    num_inputs,
    dim_map
  );
  *out_scaling = (size_t)mag_ein_chatset_len(contractions);
}

static void mag_ein_axes_push(mag_ein_axes_t *xs, int64_t v) { xs->v[xs->n++] = v; }
static int64_t mag_ein_find_axis(const char *s, char c) {
  const char *p = strchr(s, c);
  return p ? p-s : -1;
}

static int64_t mag_ein_shape_prod(
  const mag_tensor_t *x,
  const mag_ein_axes_t *axes
) {
  int64_t r=1;
  const int64_t *shape = x->coords.shape;
  for (int64_t i=0; i < axes->n; ++i)
    r *= shape[axes->v[i]];
  return r;
}

static mag_status_t mag_ein_transpose_reshape_for_dot(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_tensor_t *x,
  const mag_ein_axes_t *i_axes,
  const mag_ein_axes_t *j_axes,
  const mag_ein_axes_t *k_axes
) {
  int64_t reorder[MAG_EIN_MAX_SPEC];
  int64_t reorder_n = 0;
  for (int64_t i=0; i < i_axes->n; ++i)
    reorder[reorder_n++] = i_axes->v[i];
  for (int64_t i=0; i < j_axes->n; ++i)
    reorder[reorder_n++] = j_axes->v[i];
  for (int64_t i=0; i < k_axes->n; ++i)
    reorder[reorder_n++] = k_axes->v[i];

  mag_tensor_t *xt = NULL;
  mag_status_t stat = mag_permute(err, &xt, x, reorder, reorder_n);
  if (mag_iserr(stat)) return stat;
  int64_t size1 = mag_ein_shape_prod(x, j_axes);
  int64_t size2 = mag_ein_shape_prod(x, k_axes);
  int64_t shape[MAG_EIN_MAX_SPEC];
  int64_t rank = 0;
  const int64_t *x_shape = x->coords.shape;
  for (int64_t i=0; i < i_axes->n; ++i)
    shape[rank++] = x_shape[i_axes->v[i]];
  shape[rank++] = size1;
  shape[rank++] = size2;
  mag_tensor_t *xr = NULL;
  stat = mag_reshape(err, &xr, xt, shape, rank);
  mag_tensor_decref(xt);
  if (mag_iserr(stat)) return stat;
  *out = xr;
  return MAG_OK;
}

static mag_status_t mag_ein_broadcast_contract_dims(
  mag_error_t *err,
  mag_tensor_t **out_a,
  mag_tensor_t **out_b,
  mag_tensor_t *a,
  mag_tensor_t *b,
  const mag_ein_axes_t *a_contract,
  const mag_ein_axes_t *b_contract
) {
  int64_t a_rank = mag_tensor_rank(a);
  int64_t b_rank = mag_tensor_rank(b);
  const int64_t *a_shape_old = a->coords.shape;
  const int64_t *b_shape_old = b->coords.shape;
  int64_t a_shape[MAG_EIN_MAX_SPEC];
  int64_t b_shape[MAG_EIN_MAX_SPEC];
  memcpy(a_shape, a_shape_old, (size_t)a_rank * sizeof(a_shape[0]));
  memcpy(b_shape, b_shape_old, (size_t)b_rank * sizeof(b_shape[0]));
  if (mag_unlikely(!(a_contract->n == b_contract->n)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: internal error, contract axis count mismatch.");
  for (int64_t i=0; i < a_contract->n; ++i) {
    int64_t aa = a_contract->v[i];
    int64_t ba = b_contract->v[i];
    int64_t da = a_shape[aa];
    int64_t db = b_shape[ba];
    if (mag_unlikely(!(da == db || da == 1 || db == 1)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: cannot broadcast contracting dimensions of size %" PRIi64 " and %" PRIi64 ".", da, db);
    int64_t d = da > db ? da : db;
    a_shape[aa] = d;
    b_shape[ba] = d;
  }
  mag_tensor_t *ab = NULL;
  mag_tensor_t *bb = NULL;
  mag_status_t stat = mag_broadcast_to(err, &ab, a, a_rank, a_shape);
  if (mag_iserr(stat)) return stat;
  stat = mag_broadcast_to(err, &bb, b, b_rank, b_shape);
  if (mag_iserr(stat)) {
    mag_tensor_decref(ab);
    return stat;
  }
  *out_a = ab;
  *out_b = bb;
  return MAG_OK;
}

static mag_status_t mag_ein_batch_tensordot(
  mag_error_t *err,
  mag_tensor_t **out,
  mag_tensor_t *a,
  mag_tensor_t *b,
  const mag_ein_axes_t *a_contract,
  const mag_ein_axes_t *a_batch,
  const mag_ein_axes_t *a_concat,
  const mag_ein_axes_t *b_contract,
  const mag_ein_axes_t *b_batch,
  const mag_ein_axes_t *b_concat
) {
  mag_tensor_t *a_bcast = NULL;
  mag_tensor_t *b_bcast = NULL;
  mag_status_t stat = mag_ein_broadcast_contract_dims(err, &a_bcast, &b_bcast, a, b, a_contract, b_contract);
  if (mag_iserr(stat)) return stat;
  mag_tensor_t *ar = NULL;
  mag_tensor_t *br = NULL;
  stat = mag_ein_transpose_reshape_for_dot( err, &ar, a_bcast, a_batch, a_concat, a_contract);
  if (mag_iserr(stat)) {
    mag_tensor_decref(a_bcast);
    mag_tensor_decref(b_bcast);
    return stat;
  }
  stat = mag_ein_transpose_reshape_for_dot( err, &br, b_bcast, b_batch, b_contract, b_concat);
  if (mag_iserr(stat)) {
    mag_tensor_decref(ar);
    mag_tensor_decref(a_bcast);
    mag_tensor_decref(b_bcast);
    return stat;
  }
  mag_tensor_t *mm = NULL;
  stat = mag_matmul(err, &mm, ar, br);
  mag_tensor_decref(ar);
  mag_tensor_decref(br);
  if (mag_iserr(stat)) {
    mag_tensor_decref(a_bcast);
    mag_tensor_decref(b_bcast);
    return stat;
  }
  int64_t out_shape[MAG_EIN_MAX_SPEC];
  int64_t out_rank = 0;
  const int64_t *sa = a_bcast->coords.shape;
  const int64_t *sb = b_bcast->coords.shape;
  for (int64_t i=0; i < a_batch->n; ++i)
    out_shape[out_rank++] = sa[a_batch->v[i]];
  for (int64_t i=0; i < a_concat->n; ++i)
    out_shape[out_rank++] = sa[a_concat->v[i]];
  for (int64_t i=0; i < b_concat->n; ++i)
    out_shape[out_rank++] = sb[b_concat->v[i]];
  mag_tensor_t *reshaped = NULL;
  stat = mag_reshape(err, &reshaped, mm, out_shape, out_rank);
  mag_tensor_decref(mm);
  mag_tensor_decref(a_bcast);
  mag_tensor_decref(b_bcast);
  if (mag_iserr(stat))
    return stat;
  *out = reshaped;
  return MAG_OK;
}

static mag_status_t mag_ein_dot_node(mag_error_t *err, mag_tensor_t **out, mag_ein_path_node_t *node, mag_tensor_t **operands) {
  mag_ein_subscript_t *in_a = node->inputs;
  mag_ein_subscript_t *in_b = node->inputs+1;
  mag_ein_axes_t a_contract = {0};
  mag_ein_axes_t a_batch = {0};
  mag_ein_axes_t a_concat = {0};
  mag_ein_axes_t b_contract = {0};
  mag_ein_axes_t b_batch = {0};
  mag_ein_axes_t b_concat = {0};
  for (int64_t i=0; in_a->str.buf[i]; ++i) {
    char c=in_a->str.buf[i];
    int id = mag_ein_label_id(c);
    if (!(mag_ein_charset_has_bit(node->output.charset, id))) mag_ein_axes_push(&a_contract, i);
    else if (mag_ein_charset_has_bit(in_b->charset, id)) mag_ein_axes_push(&a_batch, i);
    else mag_ein_axes_push(&a_concat, i);
  }
  for (int64_t i=0; i < a_contract.n; ++i) {
    int64_t ax = mag_ein_find_axis(in_b->str.buf, in_a->str.buf[a_contract.v[i]]);
    if (mag_unlikely(!(ax >= 0)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: internal error, contract axis not found.");
    mag_ein_axes_push(&b_contract, ax);
  }
  for (int64_t i=0; i < a_batch.n; ++i) {
    int64_t ax = mag_ein_find_axis(in_b->str.buf, in_a->str.buf[a_batch.v[i]]);
    if (mag_unlikely(!(ax >= 0)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: internal error, batch axis not found.");
    mag_ein_axes_push(&b_batch, ax);
  }
  for (int64_t i=0; in_b->str.buf[i]; ++i) {
    int id = mag_ein_label_id(in_b->str.buf[i]);
    if (mag_ein_charset_has_bit(node->output.charset, id) && !(mag_ein_charset_has_bit(in_a->charset, id)))
      mag_ein_axes_push(&b_concat, i);
  }
  mag_tensor_t *a = operands[node->positions[0]];
  mag_tensor_t *b = operands[node->positions[1]];
  int64_t char_map[MAG_EIN_NUM_LETTERS];
  for (int i=0; i < MAG_EIN_NUM_LETTERS; ++i)
    char_map[i] = -1;
  int64_t out_axis = 0;
  for (int64_t i=0; i < a_batch.n; ++i)
    char_map[mag_ein_label_id(in_a->str.buf[a_batch.v[i]])] = out_axis++;
  for (int64_t i=0; i < a_concat.n; ++i)
    char_map[mag_ein_label_id(in_a->str.buf[a_concat.v[i]])] = out_axis++;
  for (int64_t i=0; i < b_concat.n; ++i)
    char_map[mag_ein_label_id(in_b->str.buf[b_concat.v[i]])] = out_axis++;
  mag_tensor_t *td = NULL;
  mag_status_t stat = mag_ein_batch_tensordot(err, &td, a, b, &a_contract, &a_batch, &a_concat, &b_contract, &b_batch, &b_concat);
  if (mag_iserr(stat)) return stat;
  int64_t out_rank = (int64_t)strlen(node->output.str.buf);
  if (out_rank <= 1) {
    *out = td;
    return MAG_OK;
  }
  int64_t reorder[MAG_EIN_MAX_SPEC];
  for (int64_t i=0; i < out_rank; ++i) {
    int id = mag_ein_label_id(node->output.str.buf[i]);
    if (mag_unlikely(!(char_map[id] >= 0)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: internal error, output reorder label missing.");
    reorder[i] = char_map[id];
  }
  bool identity = true;
  for (int64_t i=0; i < out_rank; ++i) {
    if (reorder[i] != i) {
      identity = false;
      break;
    }
  }
  if (identity) {
    *out = td;
  } else {
    mag_tensor_t *tmp = NULL;
    stat = mag_permute(err, &tmp, td, reorder, out_rank);
    mag_tensor_decref(td);
    if (mag_iserr(stat))
      return stat;
    *out = tmp;
  }
  return MAG_OK;
}

static void mag_ein_subscript_from_set_sorted_by_dim(mag_ein_subscript_t *out, mag_ein_charset_t set, const mag_ein_dim_map_t *dim_map) {
  int ids[MAG_EIN_NUM_LETTERS];
  int n=0;
  for (int id=0; id < MAG_EIN_NUM_LETTERS; ++id)
    if (mag_ein_charset_has_bit(set, id))
      ids[n++] = id;
  for (int i=1; i < n; ++i) {
    int v = ids[i];
    int64_t vd = dim_map->dims[v];
    int j=i-1;
    while (j >= 0 && dim_map->dims[ids[j]] > vd) {
      ids[j+1] = ids[j];
      --j;
    }
    ids[j+1] = v;
  }
  for (int i=0; i < n; ++i) {
    int id = ids[i];
    out->str.buf[i] = id < 26 ? (char)('a'+id) : (char)('A'+id-26);
  }
  out->str.buf[n] = '\0';
  out->charset = set;
}

static void mag_ein_remove_inputs_2(mag_ein_subscript_t *inputs, uint32_t *num_inputs, uint32_t x, uint32_t y) {
  if (x > y) mag_swap(uint32_t, x,y);
  memmove(inputs+y, inputs+y+1, (size_t)(*num_inputs-y-1)*sizeof(*inputs));
  --*num_inputs;
  memmove(inputs+x, inputs+x+1, (size_t)(*num_inputs-x-1)*sizeof(*inputs));
  --*num_inputs;
}

static void mag_ein_remove_operand_at(mag_tensor_t **operands, size_t *num_operands, uint32_t pos) {
  mag_tensor_decref(operands[pos]);
  memmove(operands+pos, operands+pos+1, (*num_operands-(size_t)pos-1)*sizeof(*operands));
  --*num_operands;
}

static bool mag_ein_add_contraction(
  mag_ein_contraction_t *possible,
  uint32_t *num_possible,
  uint32_t max_possible,
  const mag_ein_subscript_t *inputs,
  uint32_t num_inputs,
  const mag_ein_subscript_t *output,
  const mag_ein_dim_map_t *dim_map,
  uint32_t p1,
  uint32_t p2,
  size_t path_cost,
  size_t cost_limit,
  size_t memory_limit
) {
  mag_ein_charset_t contractions = inputs[p1].charset | inputs[p2].charset;
  mag_ein_charset_t new_term = 0;
  for (uint32_t i=0; i < num_inputs; ++i) {
    if (i == p1 || i == p2) continue;
    new_term = mag_ein_charset_union(new_term, mag_ein_charset_intersects(inputs[i].charset, contractions));
  }
  new_term = mag_ein_charset_union(new_term, mag_ein_charset_intersects(output->charset, contractions));
  size_t new_size = mag_ein_term_size_set(new_term, dim_map);
  if (new_size > memory_limit)
    return false;
  int64_t removed_size = (int64_t)mag_ein_term_size_set(inputs[p1].charset, dim_map) + (int64_t)mag_ein_term_size_set(inputs[p2].charset, dim_map) - (int64_t)new_size;
  bool inner = mag_ein_chatset_len(contractions) > mag_ein_chatset_len(new_term);
  size_t cost = mag_ein_flop_count(contractions, inner, 2, dim_map);
  if (path_cost+cost > cost_limit) return false;
  if (*num_possible >= max_possible) return false;
  possible[*num_possible] = (mag_ein_contraction_t){
    .size = removed_size,
    .cost = cost,
    .output = new_term,
    .dims = (uint32_t)mag_ein_chatset_len(contractions),
    .x = p1,
    .y = p2,
  };
  ++*num_possible;
  return true;
}

static uint32_t mag_ein_find_best_contraction(const mag_ein_contraction_t *possible, uint32_t num_possible) {
  uint32_t best = 0;
  for (uint32_t i=1; i < num_possible; ++i) {
    const mag_ein_contraction_t *x = possible+i;
    const mag_ein_contraction_t *y = possible+best;
    if (x->size > y->size || (x->size == y->size && x->cost < y->cost))
      best = i;
  }
  return best;
}

static mag_status_t mag_ein_greedy_path(
  mag_error_t *err,
  const mag_ein_subscript_t *inputs_src,
  uint32_t num_inputs_src,
  const mag_ein_subscript_t *output,
  const mag_ein_dim_map_t *dim_map,
  size_t cost_limit,
  size_t memory_limit,
  mag_ein_path_node_t *out_nodes,
  size_t *out_num_nodes,
  size_t *out_cost,
  size_t *out_scaling
) {
  mag_ein_subscript_t inputs[MAG_EIN_MAX_INPUTS];
  memcpy(inputs, inputs_src, (size_t)num_inputs_src*sizeof(*inputs));
  uint32_t num_inputs = num_inputs_src;
  size_t path_cost = 0;
  size_t path_scaling = 0;
  size_t num_nodes = 0;
  for (uint32_t step=0; step < num_inputs_src-1; ++step) {
    mag_ein_contraction_t possible[MAG_EIN_MAX_CONTRACTIONS];
    uint32_t num_possible = 0;
    for (uint32_t i=0; i < num_inputs; ++i) {
      for (uint32_t j=i+1; j < num_inputs; ++j) {
        if (mag_ein_charset_intersects(inputs[i].charset, inputs[j].charset) == 0)
          continue;
        mag_ein_add_contraction(
          possible,
          &num_possible,
          sizeof(possible)/sizeof(*possible),
          inputs,
          num_inputs,
          output,
          dim_map,
          i,
          j,
          path_cost,
          cost_limit,
          memory_limit
        );
      }
    }
    if (num_possible == 0) {
      for (uint32_t i=0; i < num_inputs; ++i) {
        for (uint32_t j=i+1; j < num_inputs; ++j) {
          mag_ein_add_contraction(
            possible,
            &num_possible,
            sizeof(possible)/sizeof(*possible),
            inputs,
            num_inputs,
            output,
            dim_map,
            i,
            j,
            path_cost,
            cost_limit,
            memory_limit
          );
        }
      }
    }
    if (num_possible == 0) {
      mag_ein_path_node_t *node = out_nodes+num_nodes++;
      memset(node, 0, sizeof(*node));
      node->num_inputs = num_inputs;
      for (uint32_t i=0; i < num_inputs; ++i) {
        node->inputs[i] = inputs[i];
        node->positions[i] = i;
      }
      node->output = *output;
      size_t cost = 0;
      size_t scaling = 0;
      mag_ein_compute_cost_and_scaling(inputs, num_inputs, output, dim_map, &cost, &scaling);
      path_cost += cost;
      if (scaling > path_scaling)
        path_scaling = scaling;
      break;
    }
    uint32_t best_i = mag_ein_find_best_contraction(possible, num_possible);
    mag_ein_contraction_t best = possible[best_i];
    if (best.dims > path_scaling)
      path_scaling = best.dims;
    mag_ein_subscript_t new_output;
    mag_ein_subscript_from_set_sorted_by_dim(&new_output, best.output, dim_map);
    if (mag_unlikely(!(num_nodes < MAG_EIN_MAX_INPUTS)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: contraction path is too long (%zu >= %d).", num_nodes, MAG_EIN_MAX_INPUTS);
    {
      mag_ein_path_node_t *node = out_nodes+num_nodes++;
      memset(node, 0, sizeof(*node));
      node->num_inputs = 2;
      node->inputs[0] = inputs[best.x];
      node->inputs[1] = inputs[best.y];
      node->positions[0] = best.x;
      node->positions[1] = best.y;
      node->output = new_output;
    }
    mag_ein_remove_inputs_2(inputs, &num_inputs, best.x, best.y);
    if (mag_unlikely(!(num_inputs < MAG_EIN_MAX_INPUTS)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: too many intermediate inputs (%u >= %d).", num_inputs, MAG_EIN_MAX_INPUTS);
    inputs[num_inputs++] = new_output;
    path_cost += best.cost;
  }
  *out_num_nodes = num_nodes;
  *out_cost = path_cost;
  *out_scaling = path_scaling;
  return MAG_OK;
}

static mag_status_t mag_ein_compute_path(
  mag_error_t *err,
  char *equation,
  const mag_tensor_t **args,
  size_t num_args,
  mag_ein_path_heuristics_t *out_heuristics,
  mag_ein_path_node_t *out_nodes,
  size_t *out_num_nodes
) {
  mag_parsed_ein_t parsed = {0};
  mag_status_t stat = mag_ein_parse(err, equation, &parsed);
  if (mag_iserr(stat)) return stat;
  if (mag_unlikely(!(parsed.num_inputs == num_args)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: equation has %u input subscripts but got %zu operands.", parsed.num_inputs, num_args);
  mag_ein_charset_t used_chars = 0;
  for (size_t i=0; i < parsed.num_inputs; ++i) {
    for (const char *p = parsed.inputs[i].buf; *p; ++p) {
      if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
        mag_ein_charset_add(used_chars, *p);
      }
    }
  }
  for (const char *p=parsed.output.buf; *p; ++p) {
    if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z')) {
      mag_ein_charset_add(used_chars, *p);
    }
  }
  char rem_chars[MAG_EIN_NUM_LETTERS+1];
  size_t rem_len = 0;
  for (char c='a'; c <= 'z'; ++c) {
    if (!mag_ein_charset_has(used_chars, c))
      rem_chars[rem_len++] = c;
  }
  for (char c='A'; c <= 'Z'; ++c) {
    if  (!mag_ein_charset_has(used_chars, c))
      rem_chars[rem_len++] = c;
  }
  rem_chars[rem_len] = '\0';
  int64_t max_ellipsis_len = 0;
  for (size_t i=0; i < parsed.num_inputs; ++i) {
    stat = mag_ein_check_alnum_and_expand_ellipsis(err, &parsed.inputs[i], args[i], i, rem_chars, rem_len, &max_ellipsis_len);
    if (mag_iserr(stat)) return stat;
  }
  stat = mag_ein_check_alnum_and_expand_ellipsis(err, &parsed.output, NULL, 0, rem_chars, rem_len, &max_ellipsis_len);
  if (mag_iserr(stat)) return stat;
  mag_ein_charset_t out_set = 0;
  const char *out_str = parsed.output.buf;
  size_t out_len = strlen(out_str);
  for (size_t i=0; i < out_len; ++i) {
    char c = out_str[i];
    if (mag_unlikely(!isalpha((unsigned char)c)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: subscripts must be letters, but got '%c'.", c);
    mag_ein_charset_add(out_set, c);
  }
  if (mag_unlikely(!((size_t)mag_ein_chatset_len(out_set) == out_len)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: repeated indices are not allowed in the output subscript.");
  mag_ein_dim_map_t dim_map = {0};
  mag_ein_subscript_t inputs[MAG_EIN_MAX_INPUTS];
  for (size_t i=0; i < parsed.num_inputs; ++i) {
    const char *in = parsed.inputs[i].buf;
    size_t in_len = strlen(in);
    int64_t ndim = args[i]->coords.rank;
    mag_ein_charset_t in_set = 0;
    for (size_t j=0; j < in_len; ++j)
      mag_ein_charset_add(in_set, in[j]);
    inputs[i].str = parsed.inputs[i];
    inputs[i].charset = in_set;
    if (mag_unlikely(!(in_len == (size_t)ndim)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: input %zu has %" PRIi64 " dimensions but its subscript has %zu labels.", i, ndim, in_len);
    if (mag_ein_chatset_len(in_set) < in_len) {
      int64_t local_dims[MAG_EIN_NUM_LETTERS] = {0};
      mag_ein_charset_t local_present = 0;
      for (size_t j=0; j < in_len; ++j) {
        char c = in[j];
        int id = mag_ein_label_id(c);
        int64_t dim = args[i]->coords.shape[j];
        if (mag_ein_charset_has_bit(local_present, id)) {
          if (mag_unlikely(!(local_dims[id] == dim)))
            return mag_set_error(err, MAG_ERR_EINSUM, "einsum: repeated subscript dimensions must have the same size, but got %" PRIi64 " and %" PRIi64 ".", local_dims[id], dim);
        } else {
          mag_ein_charset_add_bit(local_present, id);
          local_dims[id] = dim;
        }
      }
    }
    for (size_t j=0; j < in_len; ++j) {
      char c = in[j];
      int id = mag_ein_label_id(c);
      int64_t dim = args[i]->coords.shape[j];
      if (mag_ein_charset_has_bit(dim_map.present, id)) {
        int64_t old = dim_map.dims[id];
        if (mag_unlikely(!(dim == 1 || old == 1 || old == dim)))
          return mag_set_error(err, MAG_ERR_EINSUM, "einsum: cannot broadcast dim %zu of input %zu (size %" PRIi64 ") with previously seen size %" PRIi64 ".", j, i, dim, old);
        if (dim > old) dim_map.dims[id] = dim;
      } else {
        mag_ein_charset_add_bit(dim_map.present, id);
        dim_map.dims[id] = dim;
      }
    }
  }
  for (size_t i=0; i < out_len; ++i) {
    if (mag_unlikely(!mag_ein_charset_has(dim_map.present, out_str[i])))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: output subscript '%c' does not appear in any input.", out_str[i]);
  }
  size_t max_size = mag_ein_term_size(parsed.output.buf, &dim_map);
  for (size_t i=0; i < parsed.num_inputs; ++i) {
    size_t s = mag_ein_term_size(parsed.inputs[i].buf, &dim_map);
    if (s > max_size)
      max_size = s;
  }
  mag_ein_subscript_t output = {
    .str = parsed.output,
    .charset = out_set,
  };
  memset(out_heuristics, 0, sizeof(*out_heuristics));
  out_heuristics->max_term = max_size;
  mag_ein_compute_cost_and_scaling(
    inputs,
    parsed.num_inputs,
    &output,
    &dim_map,
    &out_heuristics->naive_cost,
    &out_heuristics->naive_scaling
  );
  memset(out_nodes, 0, sizeof(out_nodes[0]) * MAG_EIN_MAX_INPUTS);
  if (parsed.num_inputs <= 2) {
    out_nodes[0].num_inputs = parsed.num_inputs;
    for (size_t i=0; i < parsed.num_inputs; ++i) {
      out_nodes[0].inputs[i] = inputs[i];
      out_nodes[0].positions[i] = (uint32_t)i;
    }
    out_nodes[0].output = output;
    *out_num_nodes = 1;
    out_heuristics->opt_cost = out_heuristics->naive_cost;
    out_heuristics->opt_scaling = out_heuristics->naive_scaling;
  } else {
    stat = mag_ein_greedy_path(err, inputs, parsed.num_inputs, &output, &dim_map, out_heuristics->naive_cost, max_size, out_nodes, out_num_nodes, &out_heuristics->opt_cost, &out_heuristics->opt_scaling);
    if (mag_iserr(stat)) return stat;
    if (mag_unlikely(!(*out_num_nodes > 0)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: failed to produce a contraction path.");
    out_nodes[*out_num_nodes - 1].output = output;
  }
  return MAG_OK;
}

static bool mag_ein_axes_sorted_by_original_axis(const mag_ein_char_axis_t *xs, int64_t n) {
  for (int64_t i=1; i < n; ++i)
    if (xs[i-1].ax > xs[i].ax)
      return false;
  return true;
}

static void mag_ein_sort_str_ax(mag_ein_char_axis_t *xs, int64_t n, const int64_t char_to_ax[MAG_EIN_NUM_LETTERS]) {
  for (int64_t i=1; i < n; ++i) {
    mag_ein_char_axis_t v = xs[i];
    int64_t vkey = char_to_ax[mag_ein_label_id(v.c)];
    int64_t j=i - 1;
    while (j >= 0) {
      int64_t jkey = char_to_ax[mag_ein_label_id(xs[j].c)];
      if (jkey <= vkey)
        break;
      xs[j+1] = xs[j];
      --j;
    }
    xs[j+1] = v;
  }
}

static mag_status_t mag_ein_collapse_repeats(mag_error_t *err, mag_tensor_t **out, mag_ein_subscript_t *subscript, mag_tensor_t *x) {
  const char *str = subscript->str.buf;
  int64_t rank = mag_tensor_rank(x);
  const int64_t *shape = x->coords.shape;
  const int64_t *strides = x->coords.strides;
  int64_t new_shape[MAG_EIN_MAX_SPEC];
  int64_t new_strides[MAG_EIN_MAX_SPEC];
  char new_str[MAG_EIN_MAX_SPEC];
  uint64_t seen = 0;
  int64_t new_rank = 0;
  for (int64_t i=0; i < rank; ++i) {
    int id = mag_ein_label_id(str[i]);
    if (mag_ein_charset_has_bit(seen, id)) continue;
    mag_ein_charset_add_bit(seen, id);
    int64_t dim = shape[i];
    int64_t stride_sum = 0;
    for (int64_t j=i; j < rank; ++j) {
      if (str[j] == str[i]) {
        if (mag_unlikely(!(shape[j] == dim)))
          return mag_set_error(err, MAG_ERR_EINSUM, "einsum: repeated subscript dimensions must have the same size.");
        stride_sum += strides[j];
      }
    }
    new_shape[new_rank] = dim;
    new_strides[new_rank] = stride_sum;
    new_str[new_rank] = str[i];
    ++new_rank;
  }
  new_str[new_rank] = '\0';
  mag_status_t stat = mag_as_strided(err, out, x->ctx, x, new_rank, new_shape, new_strides, (int64_t)mag_tensor_data_offset(x));
  if (mag_iserr(stat)) return stat;
  snprintf(subscript->str.buf, sizeof(subscript->str.buf), "%s", new_str);
  subscript->charset = seen;
  return MAG_OK;
}

static bool mag_ein_can_dot(const mag_ein_path_node_t *node) {
  if (node->num_inputs != 2)
    return false;
  const mag_ein_subscript_t *a = node->inputs;
  for (const char *p=a->str.buf; *p; ++p) {
    if (!mag_ein_charset_has(node->output.charset, *p))
      return true;
  }
  return false;
}

static mag_status_t mag_ein_naive(mag_error_t *err, mag_tensor_t **out_result, mag_ein_path_node_t *node, mag_tensor_t **operands) {
  int64_t char_to_ax[MAG_EIN_NUM_LETTERS];
  for (int i=0; i < MAG_EIN_NUM_LETTERS; ++i)
    char_to_ax[i] = -1;
  int64_t num_axes = 0;
  mag_status_t stat;
  for (uint32_t i=0; i < node->num_inputs; ++i) {
    const char *s = node->inputs[i].str.buf;
    for (const char *p=s; *p; ++p) {
      int id = mag_ein_label_id(*p);
      if (char_to_ax[id] < 0) char_to_ax[id] = num_axes++;
    }
  }
  for (uint32_t i=0; i < node->num_inputs; ++i) {
    uint32_t pos = node->positions[i];
    mag_tensor_t *op = operands[pos];
    int64_t op_rank = op->coords.rank;
    if (op_rank != num_axes) {
      const int64_t *old_shape = mag_tensor_shape_ptr(op);
      int64_t shape[MAG_EIN_MAX_SPEC];
      for (int64_t ax=0; ax < op_rank; ++ax)
        shape[ax] = old_shape[ax];
      for (int64_t ax=op_rank; ax < num_axes; ++ax)
        shape[ax] = 1;
      mag_tensor_t *tmp = NULL;
      stat = mag_reshape(err, &tmp, op, shape, num_axes);
      if (mag_iserr(stat)) return stat;
      mag_tensor_decref(operands[pos]);
      operands[pos] = tmp;
      op = tmp;
    }
    mag_ein_char_axis_t str_ax[MAG_EIN_MAX_SPEC];
    int64_t str_ax_len = 0;
    const char *str = node->inputs[i].str.buf;
    for (const char *p=str; *p; ++p) {
      int64_t ax = str_ax_len;
      str_ax[str_ax_len++] = (mag_ein_char_axis_t){.c = *p, .ax = ax,};
    }
    for (int id=0; id < MAG_EIN_NUM_LETTERS; ++id) {
      if (char_to_ax[id] < 0)
        continue;
      char c = id < 26 ? (char)('a' + id) : (char)('A' + id - 26);
      if (!mag_ein_charset_has_bit(node->inputs[i].charset, id)) {
        int64_t ax = str_ax_len;
        str_ax[str_ax_len++] = (mag_ein_char_axis_t){.c = c, .ax = ax,};
      }
    }
    mag_ein_sort_str_ax(str_ax, str_ax_len, char_to_ax);
    if (mag_ein_axes_sorted_by_original_axis(str_ax, str_ax_len))
      continue;
    int64_t reorder[MAG_EIN_MAX_SPEC];
    for (int64_t ax=0; ax < str_ax_len; ++ax)
      reorder[ax] = str_ax[ax].ax;
    mag_tensor_t *tmp = NULL;
    stat = mag_permute(err, &tmp, op, reorder, str_ax_len);
    if (mag_iserr(stat)) return stat;
    mag_tensor_decref(operands[pos]);
    operands[pos] = tmp;
  }
  mag_tensor_t *out = operands[node->positions[0]];
  mag_tensor_incref(out);
  for (uint32_t i=1; i < node->num_inputs; ++i) {
    mag_tensor_t *tmp = NULL;
    stat = mag_mul(err, &tmp, out, operands[node->positions[i]]);
    if (mag_iserr(stat)) return stat;
    mag_tensor_decref(out);
    out = tmp;
  }
  int64_t sum_axes[MAG_EIN_MAX_SPEC];
  int64_t num_sum_axes = 0;
  for (int id=0; id < MAG_EIN_NUM_LETTERS; ++id) {
    if (char_to_ax[id] < 0)
      continue;
    if (!mag_ein_charset_has_bit(node->output.charset, id))
      sum_axes[num_sum_axes++] = char_to_ax[id];
  }
  if (num_sum_axes > 0) {
    mag_tensor_t *tmp = NULL;
    stat = mag_sum(err, &tmp, out, sum_axes, num_sum_axes, false);
    if (mag_iserr(stat)) return stat;
    mag_tensor_decref(out);
    out = tmp;
  }
  const char *out_str = node->output.str.buf;
  int64_t out_rank = (int64_t)strlen(out_str);
  if (out_rank <= 1) {
    *out_result = out;
    return MAG_OK;
  }
  int64_t reorder[MAG_EIN_MAX_SPEC];
  for (int64_t i=0; i < out_rank; ++i) {
    int id = mag_ein_label_id(out_str[i]);
    reorder[i] = char_to_ax[id];
    int64_t offset = 0;
    for (int64_t j=0; j < num_sum_axes; ++j)
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
    stat = mag_permute(err, &tmp, out, reorder, out_rank);
    if (mag_iserr(stat)) return stat;
    mag_tensor_decref(out);
    *out_result = tmp;
  }
  return MAG_OK;
}

static mag_status_t mag_ein_preprocess_node(mag_error_t *err, mag_ein_path_node_t *node, mag_tensor_t **operands) {
  mag_status_t stat;
  for (uint32_t i=0; i < node->num_inputs; ++i) {
    uint32_t pos = node->positions[i];
    if ((size_t)mag_ein_chatset_len(node->inputs[i].charset) < strlen(node->inputs[i].str.buf)) {
      mag_tensor_t *collapsed = NULL;
      stat = mag_ein_collapse_repeats(err, &collapsed, &node->inputs[i], operands[pos]);
      if (mag_iserr(stat)) return stat;
      mag_tensor_decref(operands[pos]);
      operands[pos] = collapsed;
    }
  }
  int counts[MAG_EIN_NUM_LETTERS] = {0};
  for (uint32_t i=0; i < node->num_inputs; ++i) {
    mag_ein_charset_t set = node->inputs[i].charset;
    for (int id=0; id < MAG_EIN_NUM_LETTERS; ++id)
      if (mag_ein_charset_has_bit(set, id))
        ++counts[id];
  }
  for (int id=0; id < MAG_EIN_NUM_LETTERS; ++id)
    if (mag_ein_charset_has_bit(node->output.charset, id))
      ++counts[id];
  for (uint32_t i=0; i < node->num_inputs; ++i) {
    mag_ein_subscript_t *in = &node->inputs[i];
    uint32_t pos = node->positions[i];
    int64_t sum_axes[MAG_EIN_MAX_SPEC];
    int64_t num_sum_axes = 0;
    for (int64_t ax=0; in->str.buf[ax]; ++ax) {
      int id = mag_ein_label_id(in->str.buf[ax]);
      if (counts[id] == 1)
        sum_axes[num_sum_axes++] = ax;
    }
    if (!num_sum_axes) continue;
    mag_tensor_t *summed = NULL;
    stat = mag_sum(err, &summed, operands[pos], sum_axes, num_sum_axes, false);
    if (mag_iserr(stat)) return stat;
    mag_tensor_decref(operands[pos]);
    operands[pos] = summed;
    char new_str[MAG_EIN_MAX_SPEC];
    int64_t w = 0;
    for (int64_t ax=0; in->str.buf[ax]; ++ax) {
      bool remove = false;
      for (int64_t j=0; j < num_sum_axes; ++j) {
        if (sum_axes[j] == ax) {
          remove = true;
          break;
        }
      }
      if (!remove)
        new_str[w++] = in->str.buf[ax];
    }
    new_str[w] = '\0';
    snprintf(in->str.buf, sizeof(in->str.buf), "%s", new_str);
    in->charset = 0;
    for (int64_t ax=0; in->str.buf[ax]; ++ax)
      mag_ein_charset_add(in->charset, in->str.buf[ax]);
  }
  return MAG_OK;
}

static mag_status_t mag_ein_execute_path(
  mag_error_t *err,
  mag_tensor_t **out_result,
  mag_ein_path_node_t *nodes,
  size_t num_nodes,
  const mag_tensor_t **args,
  size_t num_args
) {
  mag_tensor_t *operands[MAG_EIN_MAX_INPUTS * 2];
  size_t num_operands = num_args;
  for (size_t i=0; i < num_args; ++i) {
    operands[i] = (mag_tensor_t *)args[i];
    mag_tensor_incref(operands[i]);
  }
  mag_status_t status = MAG_OK;
  for (size_t n=0; n < num_nodes; ++n) {
    mag_tensor_t *result = NULL;
    status = mag_ein_preprocess_node(err, &nodes[n], operands);
    if (mag_iserr(status))
      goto cleanup;
    if (mag_ein_can_dot(&nodes[n])) status = mag_ein_dot_node(err, &result, &nodes[n], operands);
    else status = mag_ein_naive(err, &result, &nodes[n], operands);
    if (mag_iserr(status))
      goto cleanup;
    if (mag_unlikely(!(num_operands < MAG_EIN_MAX_INPUTS * 2)))
      return mag_set_error(err, MAG_ERR_EINSUM, "einsum: too many intermediate operands.");
    operands[num_operands++] = result;
    for (int64_t i=(int64_t)nodes[n].num_inputs-1; i >= 0; --i) {
      uint32_t pos = nodes[n].positions[i];
      if (mag_unlikely(!(pos < num_operands)))
        return mag_set_error(err, MAG_ERR_EINSUM, "einsum: internal error, invalid path operand position.");
      mag_ein_remove_operand_at(operands, &num_operands, pos);
    }
  }
  if (mag_unlikely(!(num_operands == 1)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: internal error, expected a single final operand but got %zu.", num_operands);
  *out_result = *operands;
  return MAG_OK;
cleanup:
  for (size_t i=0; i < num_operands; ++i)
    mag_tensor_decref(operands[i]);
  return status;
}

static MAG_COLDPROC void mag_ein_debug_print_path(const char *equation, const mag_ein_path_heuristics_t *h, const mag_ein_path_node_t *nodes, size_t num_nodes) {
  printf("einsum: %s\n", equation);
  printf("  naive cost:       %zu\n", h->naive_cost);
  printf("  naive scaling:    %zu\n", h->naive_scaling);
  printf("  optimized cost:   %zu\n", h->opt_cost);
  printf("  optimized scaling:%zu\n", h->opt_scaling);
  printf("  max term:         %zu\n", h->max_term);
  printf("  nodes:            %zu\n", num_nodes);
  for (size_t n=0; n < num_nodes; ++n) {
    const mag_ein_path_node_t *node = &nodes[n];
    printf("  node[%zu]: ", n);
    for (uint32_t i=0; i < node->num_inputs; ++i) {
      if (i) printf(", ");
      printf("%u:%s", node->positions[i], node->inputs[i].str.buf);
    }
    printf(" -> %s\n", node->output.str.buf);
  }
}

mag_status_t mag_einsum_eval(mag_error_t *err, mag_tensor_t **out_result, const char *equation, const mag_tensor_t **args, size_t num_args) {
  size_t len = strlen(equation);
  if (mag_unlikely(!mag_utf8_validate((const uint8_t *)equation, len)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: equation string contains invalid UTF-8.");
  if (mag_unlikely(!(num_args > 0)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: requires at least one input tensor.");
  char *cloned = mag_strdup(equation);
  if (mag_unlikely(!cloned))
    return mag_set_error(err, MAG_ERR_OOM, "einsum: failed to allocate %zu bytes for equation string.", len+1);
  mag_ein_remove_spaces(cloned);
  len = strlen(cloned);
  if (mag_unlikely(!(len > 0)))
    return mag_set_error(err, MAG_ERR_EINSUM, "einsum: equation string is empty.");
  mag_ein_path_heuristics_t heuristics = {0};
  mag_ein_path_node_t nodes[MAG_EIN_MAX_INPUTS] = {0};
  size_t num_nodes = 0;
  mag_status_t stat = mag_ein_compute_path(err, cloned, args, num_args, &heuristics, nodes, &num_nodes);
  if (mag_iserr(stat)) {
    (*mag_alloc)(cloned, 0, 0);
    return stat;
  }
  #ifdef MAG_DEBUG
    mag_ein_debug_print_path(cloned, &heuristics, nodes, num_nodes);
  #endif
  stat = mag_ein_execute_path(err, out_result, nodes, num_nodes, args, num_args);
  (*mag_alloc)(cloned, 0, 0);
  return stat;
}
