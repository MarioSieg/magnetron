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

#include "mag_interp_plan.h"

typedef enum mag_interp_kind_t {
  MAG_INTERP_KIND_NEAREST,
  MAG_INTERP_KIND_NEAREST_EXACT,
  MAG_INTERP_KIND_LINEAR,
  MAG_INTERP_KIND_CUBIC,
  MAG_INTERP_KIND_AREA,
} mag_interp_kind_t;

bool mag_interp_mode_is_nearest(mag_interp_mode_t mode) {
  return mode == MAG_INTERP_MODE_NEAREST || mode == MAG_INTERP_MODE_NEAREST_EXACT;
}

static mag_interp_kind_t mag_interp_kind_of(mag_interp_mode_t mode) {
  switch (mode) {
    case MAG_INTERP_MODE_NEAREST: return MAG_INTERP_KIND_NEAREST;
    case MAG_INTERP_MODE_NEAREST_EXACT: return MAG_INTERP_KIND_NEAREST_EXACT;
    case MAG_INTERP_MODE_LINEAR:
    case MAG_INTERP_MODE_BILINEAR:
    case MAG_INTERP_MODE_TRILINEAR: return MAG_INTERP_KIND_LINEAR;
    case MAG_INTERP_MODE_BICUBIC: return MAG_INTERP_KIND_CUBIC;
    case MAG_INTERP_MODE_AREA: return MAG_INTERP_KIND_AREA;
    default: return MAG_INTERP_KIND_NEAREST;
  }
}

static float mag_interp_scale(int64_t in, int64_t out, double sf, bool align) {
  if (align) return out > 1 ? (float)(in-1)/(float)(out-1) : 0.f;
  return sf > 0.0 ? (float)(1.0/sf) : (float)in/(float)out;
}
static float mag_interp_source_index(float scale, int64_t dst, bool align, bool cubic) {
  if (align) return scale*(float)dst;
  float src = scale*((float)dst + 0.5f) - 0.5f;
  return !cubic && src < 0.f ? 0.f : src;
}
static float mag_cubic_conv1(float x, float a) { return ((a + 2.f)*x - (a + 3.f))*x*x + 1.f; }
static float mag_cubic_conv2(float x, float a) { return ((a*x - 5.f*a)*x + 8.f*a)*x - 4.f*a; }
static float mag_aa_filter_linear(float x) { x = fabsf(x); return x < 1.f ? 1.f - x : 0.f; }
static float mag_aa_filter_cubic(float x) {
  float a = -0.5f;
  x = fabsf(x);
  if (x < 1.f) return ((a + 2.f)*x - (a + 3.f))*x*x + 1.f;
  if (x < 2.f) return (((x - 5.f)*x + 8.f)*x - 4.f)*a;
  return 0.f;
}

static bool mag_interp_axis_alloc(mag_interp_axis_t *ax, int64_t in, int64_t out, int64_t max_taps, mag_interp_alloc_fn *alloc, void *ud) {
  ax->in = in;
  ax->out = out;
  ax->max_taps = max_taps;
  ax->ntaps = (*alloc)(ud, (size_t)out*sizeof(*ax->ntaps));
  ax->idx = (*alloc)(ud, (size_t)(out*max_taps)*sizeof(*ax->idx));
  ax->w = (*alloc)(ud, (size_t)(out*max_taps)*sizeof(*ax->w));
  return ax->ntaps && ax->idx && ax->w;
}

static bool mag_interp_axis_build_forward(
  mag_interp_axis_t *ax,
  mag_interp_kind_t kind,
  int64_t in,
  int64_t out,
  double scale_factor,
  bool align_corners,
  bool antialias,
  mag_interp_alloc_fn alloc,
  void *ud
) {
  float scale = mag_interp_scale(in, out, scale_factor, align_corners);
  int64_t interp_size = kind == MAG_INTERP_KIND_CUBIC ? 4 : 2;
  float support = scale >= 1.f ? (float)interp_size*0.5f*scale : (float)interp_size*0.5f;
  int64_t max_taps;
  switch (kind) {
    case MAG_INTERP_KIND_NEAREST:
    case MAG_INTERP_KIND_NEAREST_EXACT: max_taps = 1; break;
    case MAG_INTERP_KIND_LINEAR: max_taps = antialias ? ((int64_t)ceilf(support)<<1) + 1 : 2; break;
    case MAG_INTERP_KIND_CUBIC: max_taps = antialias ? ((int64_t)ceilf(support)<<1) + 1 : 4; break;
    case MAG_INTERP_KIND_AREA: max_taps = (in + out - 1)/out + 1; break;
    default: return false;
  }
  if (!mag_interp_axis_alloc(ax, in, out, max_taps, alloc, ud)) return false;
  for (int64_t o=0; o < out; ++o) {
    int64_t *idx = ax->idx + o*max_taps;
    float *w = ax->w + o*max_taps;
    int64_t n = 0;
    switch (kind) {
      case MAG_INTERP_KIND_NEAREST: {
        int64_t src;
        if (out == in) src = o;
        else if (out == in<<1) src = o>>1;
        else src = (int64_t)floorf((float)o*scale);
        idx[0] = src < in-1 ? src : in-1;
        w[0] = 1.f;
        n = 1;
      } break;
      case MAG_INTERP_KIND_NEAREST_EXACT: {
        int64_t src = (int64_t)floorf(((float)o + 0.5f)*scale);
        idx[0] = src < in-1 ? src : in-1;
        w[0] = 1.f;
        n = 1;
      } break;
      case MAG_INTERP_KIND_LINEAR:
      case MAG_INTERP_KIND_CUBIC: {
        if (antialias) {
          float center = scale*((float)o + 0.5f);
          float invscale = scale >= 1.f ? 1.f/scale : 1.f;
          int64_t xmin = (int64_t)(center - support + 0.5f);
          if (xmin < 0) xmin = 0;
          int64_t xmax = (int64_t)(center + support + 0.5f);
          if (xmax > in) xmax = in;
          int64_t xsize = xmax - xmin;
          if (xsize > max_taps) xsize = max_taps;
          float total = 0.f;
          for (int64_t j=0; j < xsize; ++j) {
            float arg = ((float)(j + xmin) - center + 0.5f)*invscale;
            float wj = kind == MAG_INTERP_KIND_CUBIC ? mag_aa_filter_cubic(arg) : mag_aa_filter_linear(arg);
            idx[j] = xmin + j;
            w[j] = wj;
            total += wj;
          }
          if (total != 0.f)
            for (int64_t j=0; j < xsize; ++j) w[j] /= total;
          n = xsize;
        } else if (kind == MAG_INTERP_KIND_LINEAR) {
          if (in == out) {
            idx[0] = o;
            w[0] = 1.f;
            n = 1;
          } else {
            float real = mag_interp_source_index(scale, o, align_corners, false);
            int64_t i0 = (int64_t)real;
            if (i0 > in-1) i0 = in-1;
            float l1 = real - (float)i0;
            l1 = l1 < 0.f ? 0.f : (l1 > 1.f ? 1.f : l1);
            int64_t i1 = i0 + (i0 < in-1 ? 1 : 0);
            idx[0] = i0;
            w[0] = 1.f - l1;
            idx[1] = i1;
            w[1] = l1;
            n = 2;
          }
        } else {
          const float a = -0.75f;
          float real = mag_interp_source_index(scale, o, align_corners, true);
          float fx = floorf(real);
          int64_t ix = (int64_t)fx;
          float t = real - fx;
          float coeffs[4];
          coeffs[0] = mag_cubic_conv2(t + 1.f, a);
          coeffs[1] = mag_cubic_conv1(t, a);
          coeffs[2] = mag_cubic_conv1(1.f - t, a);
          coeffs[3] = mag_cubic_conv2(2.f - t, a);
          for (int64_t k=0; k < 4; ++k) {
            idx[k] = mag_vclamp(ix - 1 + k, 0, in-1);
            w[k] = coeffs[k];
          }
          n = 4;
        }
      } break;
      case MAG_INTERP_KIND_AREA: {
        int64_t start = o*in/out;
        int64_t end = ((o+1)*in + out - 1)/out;
        float wv = 1.f/(float)(end - start);
        for (int64_t i=start; i < end; ++i) {
          idx[n] = i;
          w[n] = wv;
          ++n;
        }
      } break;
      default: return false;
    }
    ax->ntaps[o] = n;
  }
  return true;
}

static bool mag_interp_axis_transpose(mag_interp_axis_t *dst, const mag_interp_axis_t *src, mag_interp_alloc_fn alloc, void *ud) {
  int64_t in = src->out;
  int64_t out = src->in;
  int64_t *counts = (*alloc)(ud, (size_t)out*sizeof(int64_t));
  if (!counts) return false;
  for (int64_t i=0; i < out; ++i) counts[i] = 0;
  for (int64_t o=0; o < src->out; ++o)
    for (int64_t k=0; k < src->ntaps[o]; ++k)
      ++counts[src->idx[o*src->max_taps + k]];
  int64_t max_taps = 1;
  for (int64_t i=0; i < out; ++i) if (counts[i] > max_taps) max_taps = counts[i];
  if (!mag_interp_axis_alloc(dst, in, out, max_taps, alloc, ud)) return false;
  for (int64_t i=0; i < out; ++i) dst->ntaps[i] = 0;
  for (int64_t o=0; o < src->out; ++o) {
    for (int64_t k=0; k < src->ntaps[o]; ++k) {
      int64_t i = src->idx[o*src->max_taps + k];
      int64_t slot = dst->ntaps[i]++;
      dst->idx[i*max_taps + slot] = o;
      dst->w[i*max_taps + slot] = src->w[o*src->max_taps + k];
    }
  }
  return true;
}

bool mag_interp_plan_build(
  mag_interp_plan_t *plan,
  const mag_op_params_t *params,
  const int64_t *small_shape,
  const int64_t *big_shape,
  int64_t rank,
  bool transposed,
  mag_interp_alloc_fn *alloc,
  void *ud
) {
  int64_t spatial = rank-2;
  int64_t off = 3-spatial;
  plan->planes = small_shape[0]*small_shape[1];
  mag_interp_kind_t kind = mag_interp_kind_of(params->interp.mode);
  for (int64_t d=0; d < 3; ++d) {
    int64_t in = 1;
    int64_t out = 1;
    double sf = 0.0;
    mag_interp_kind_t kd = MAG_INTERP_KIND_NEAREST;
    bool aa = false;
    bool ac = false;
    if (d >= off) {
      int64_t i = d - off;
      in = small_shape[2+i];
      out = big_shape[2+i];
      sf = params->interp.scale[i];
      kd = kind;
      aa = params->interp.antialias;
      ac = params->interp.align_corners;
    }
    mag_interp_axis_t fwd;
    if (!mag_interp_axis_build_forward(&fwd, kd, in, out, sf, ac, aa, alloc, ud)) return false;
    if (transposed) {
      if (!mag_interp_axis_transpose(&plan->axes[d], &fwd, alloc, ud)) return false;
    } else {
      plan->axes[d] = fwd;
    }
  }
  return true;
}
