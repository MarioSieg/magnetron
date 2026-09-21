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

#include "mag_cuda_conv.cuh"

#include <cuda_runtime.h>

namespace mag {
  constexpr unsigned CONV_BLOCK_SIZE = 256;

  struct conv_geom final {
    int N;
    int cin;
    int cout;
    int groups;
    int big[3];
    int small[3];
    int k[3];
    int s[3];
    int p[3];
    int d[3];
  };

  static conv_geom make_conv_geom(const mag_op_params_t &params, const int64_t *big, const int64_t *small, const int64_t *k) {
    conv_geom g {};
    int spatial = static_cast<int>(params.conv.spatial);
    int off = 3 - spatial;
    g.groups = static_cast<int>(params.conv.groups);
    for (int i=0; i < 3; ++i) {
      g.big[i] = 1;
      g.small[i] = 1;
      g.k[i] = 1;
      g.s[i] = 1;
      g.p[i] = 0;
      g.d[i] = 1;
    }
    for (int i=0; i < spatial; ++i) {
      g.big[off+i] = static_cast<int>(big[i]);
      g.small[off+i] = static_cast<int>(small[i]);
      g.k[off+i] = static_cast<int>(k[i]);
      g.s[off+i] = static_cast<int>(params.conv.stride[i]);
      g.p[off+i] = static_cast<int>(params.conv.padding[i]);
      g.d[off+i] = static_cast<int>(params.conv.dilation[i]);
    }
    return g;
  }

  template <typename T>
  __global__ static void conv_kernel(int64_t total, conv_geom g, T *__restrict__ br, const T *__restrict__ bx, const T *__restrict__ bw, const T *__restrict__ bb) {
    int64_t idx = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*gridDim.x;
    int cinG = g.cin/g.groups;
    int coutG = g.cout/g.groups;
    int Id = g.big[0], Ih = g.big[1], Iw = g.big[2];
    int Od = g.small[0], Oh = g.small[1], Ow = g.small[2];
    int Kd = g.k[0], Kh = g.k[1], Kw = g.k[2];
    for (; idx < total; idx += step) {
      int64_t tmp = idx;
      int ow = static_cast<int>(tmp % Ow); tmp /= Ow;
      int oh = static_cast<int>(tmp % Oh); tmp /= Oh;
      int od = static_cast<int>(tmp % Od); tmp /= Od;
      int co = static_cast<int>(tmp % g.cout); tmp /= g.cout;
      int n = static_cast<int>(tmp);
      int grp = co/coutG;
      float acc = bb ? static_cast<float>(bb[co]) : 0.0f;
      for (int cig=0; cig < cinG; ++cig) {
        int ci = grp*cinG + cig;
        const T *xplane = bx + (static_cast<int64_t>(n)*g.cin + ci)*Id*Ih*Iw;
        const T *wplane = bw + (static_cast<int64_t>(co)*cinG + cig)*Kd*Kh*Kw;
        for (int kd=0; kd < Kd; ++kd) {
          int id = od*g.s[0] - g.p[0] + kd*g.d[0];
          if (id < 0 || id >= Id) continue;
          for (int kh=0; kh < Kh; ++kh) {
            int ih = oh*g.s[1] - g.p[1] + kh*g.d[1];
            if (ih < 0 || ih >= Ih) continue;
            const T *xrow = xplane + (static_cast<int64_t>(id)*Ih + ih)*Iw;
            const T *wrow = wplane + (kd*Kh + kh)*Kw;
            for (int kw=0; kw < Kw; ++kw) {
              int iw = ow*g.s[2] - g.p[2] + kw*g.d[2];
              if (iw < 0 || iw >= Iw) continue;
              acc += static_cast<float>(wrow[kw])*static_cast<float>(xrow[iw]);
            }
          }
        }
      }
      br[idx] = static_cast<T>(acc);
    }
  }

  template <typename T>
  __global__ static void conv_transpose_kernel(int64_t total, conv_geom g, T *__restrict__ br, const T *__restrict__ bx, const T *__restrict__ bw, const T *__restrict__ bb) {
    int64_t idx = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t step = static_cast<int64_t>(blockDim.x)*gridDim.x;
    int cinG = g.cin/g.groups;
    int coutG = g.cout/g.groups;
    int Id = g.small[0], Ih = g.small[1], Iw = g.small[2];
    int Od = g.big[0], Oh = g.big[1], Ow = g.big[2];
    int Kd = g.k[0], Kh = g.k[1], Kw = g.k[2];
    for (; idx < total; idx += step) {
      int64_t tmp = idx;
      int ow = static_cast<int>(tmp % Ow); tmp /= Ow;
      int oh = static_cast<int>(tmp % Oh); tmp /= Oh;
      int od = static_cast<int>(tmp % Od); tmp /= Od;
      int co = static_cast<int>(tmp % g.cout); tmp /= g.cout;
      int n = static_cast<int>(tmp);
      int grp = co/coutG;
      int cog = co - grp*coutG;
      float acc = bb ? static_cast<float>(bb[co]) : 0.0f;
      for (int cig=0; cig < cinG; ++cig) {
        int ci = grp*cinG + cig;
        const T *xplane = bx + (static_cast<int64_t>(n)*g.cin + ci)*Id*Ih*Iw;
        const T *wplane = bw + (static_cast<int64_t>(ci)*coutG + cog)*Kd*Kh*Kw;
        for (int kd=0; kd < Kd; ++kd) {
          int td = od + g.p[0] - kd*g.d[0];
          if (td < 0 || td % g.s[0]) continue;
          int id = td/g.s[0];
          if (id >= Id) continue;
          for (int kh=0; kh < Kh; ++kh) {
            int th = oh + g.p[1] - kh*g.d[1];
            if (th < 0 || th % g.s[1]) continue;
            int ih = th/g.s[1];
            if (ih >= Ih) continue;
            const T *xrow = xplane + (static_cast<int64_t>(id)*Ih + ih)*Iw;
            const T *wrow = wplane + (kd*Kh + kh)*Kw;
            for (int kw=0; kw < Kw; ++kw) {
              int tw = ow + g.p[2] - kw*g.d[2];
              if (tw < 0 || tw % g.s[2]) continue;
              int iw = tw/g.s[2];
              if (iw >= Iw) continue;
              acc += static_cast<float>(wrow[kw])*static_cast<float>(xrow[iw]);
            }
          }
        }
      }
      br[idx] = static_cast<T>(acc);
    }
  }

  template <typename T>
  __global__ static void conv_wgrad_kernel(conv_geom g, T *__restrict__ br, const T *__restrict__ ba, const T *__restrict__ bb) {
    __shared__ float red[CONV_BLOCK_SIZE];
    int caG = g.cin/g.groups;
    int cbG = g.cout/g.groups;
    int Id = g.big[0], Ih = g.big[1], Iw = g.big[2];
    int Od = g.small[0], Oh = g.small[1], Ow = g.small[2];
    int Kd = g.k[0], Kh = g.k[1], Kw = g.k[2];
    int64_t item = blockIdx.x;
    int64_t tmp = item;
    int kw = static_cast<int>(tmp % Kw); tmp /= Kw;
    int kh = static_cast<int>(tmp % Kh); tmp /= Kh;
    int kd = static_cast<int>(tmp % Kd); tmp /= Kd;
    int cag = static_cast<int>(tmp % caG); tmp /= caG;
    int cb = static_cast<int>(tmp);
    int grp = cb/cbG;
    int ca = grp*caG + cag;
    int64_t plane = static_cast<int64_t>(Od)*Oh*Ow;
    int64_t positions = g.N*plane;
    float acc = 0.0f;
    for (int64_t pos = threadIdx.x; pos < positions; pos += blockDim.x) {
      int64_t t = pos;
      int ow = static_cast<int>(t % Ow); t /= Ow;
      int oh = static_cast<int>(t % Oh); t /= Oh;
      int od = static_cast<int>(t % Od); t /= Od;
      int n = static_cast<int>(t);
      int id = od*g.s[0] - g.p[0] + kd*g.d[0];
      int ih = oh*g.s[1] - g.p[1] + kh*g.d[1];
      int iw = ow*g.s[2] - g.p[2] + kw*g.d[2];
      if (id < 0 || id >= Id || ih < 0 || ih >= Ih || iw < 0 || iw >= Iw) continue;
      const T *bv = bb + (static_cast<int64_t>(n)*g.cout + cb)*plane + (static_cast<int64_t>(od)*Oh + oh)*Ow + ow;
      const T *av = ba + ((static_cast<int64_t>(n)*g.cin + ca)*Id + id)*Ih*Iw + static_cast<int64_t>(ih)*Iw + iw;
      acc += static_cast<float>(*bv)*static_cast<float>(*av);
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned off = blockDim.x>>1; off > 0; off >>= 1) {
      if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
      __syncthreads();
    }
    if (threadIdx.x == 0) br[item] = static_cast<T>(red[0]);
  }

  static int64_t grid_for(int64_t total) {
    int64_t blocks = (total + CONV_BLOCK_SIZE - 1)/CONV_BLOCK_SIZE;
    return blocks < 1 ? 1 : (blocks > 65535*16 ? 65535*16 : blocks);
  }

  template <typename T>
  static mag_status_t launch_conv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, bool transposed) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *w = cmd.in[1];
    const mag_tensor_t *b = cmd.num_in > 2 ? cmd.in[2] : nullptr;
    int64_t total = mag_tensor_numel(r);
    if (!total) return MAG_OK;
    conv_geom g = transposed
      ? make_conv_geom(*cmd.params, r->meta.coords.shape+2, x->meta.coords.shape+2, w->meta.coords.shape+2)
      : make_conv_geom(*cmd.params, x->meta.coords.shape+2, r->meta.coords.shape+2, w->meta.coords.shape+2);
    g.N = static_cast<int>(x->meta.coords.shape[0]);
    g.cin = static_cast<int>(x->meta.coords.shape[1]);
    g.cout = static_cast<int>(r->meta.coords.shape[1]);
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    const auto *bw = reinterpret_cast<const T *>(mag_tensor_data_ptr(w));
    const auto *bb = b ? reinterpret_cast<const T *>(mag_tensor_data_ptr(b)) : nullptr;
    int64_t blocks = grid_for(total);
    if (transposed) conv_transpose_kernel<T><<<blocks, CONV_BLOCK_SIZE, 0, stream>>>(total, g, br, bx, bw, bb);
    else conv_kernel<T><<<blocks, CONV_BLOCK_SIZE, 0, stream>>>(total, g, br, bx, bw, bb);
    mag_cu_rt_check(err, cudaGetLastError(), transposed ? "conv_transpose" : "conv");
    return MAG_OK;
  }

  template <typename T>
  static mag_status_t launch_conv_wgrad(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *a = cmd.in[0];
    const mag_tensor_t *b = cmd.in[1];
    int64_t total = mag_tensor_numel(r);
    if (!total) return MAG_OK;
    conv_geom g = make_conv_geom(*cmd.params, a->meta.coords.shape+2, b->meta.coords.shape+2, r->meta.coords.shape+2);
    g.N = static_cast<int>(a->meta.coords.shape[0]);
    g.cin = static_cast<int>(a->meta.coords.shape[1]);
    g.cout = static_cast<int>(b->meta.coords.shape[1]);
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *ba = reinterpret_cast<const T *>(mag_tensor_data_ptr(a));
    const auto *bb = reinterpret_cast<const T *>(mag_tensor_data_ptr(b));
    conv_wgrad_kernel<T><<<static_cast<unsigned>(total), CONV_BLOCK_SIZE, 0, stream>>>(g, br, ba, bb);
    mag_cu_rt_check(err, cudaGetLastError(), "conv_wgrad");
    return MAG_OK;
  }

  static mag_status_t conv_dispatch(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, int which) {
    const mag_tensor_t *x = cmd.in[0];
    switch (x->meta.dtype) {
      case MAG_DTYPE_FLOAT32: return which == 2 ? launch_conv_wgrad<float>(err, cmd, stream) : launch_conv<float>(err, cmd, stream, which == 1);
      case MAG_DTYPE_FLOAT16: return which == 2 ? launch_conv_wgrad<half>(err, cmd, stream) : launch_conv<half>(err, cmd, stream, which == 1);
      case MAG_DTYPE_BFLOAT16: return which == 2 ? launch_conv_wgrad<__nv_bfloat16>(err, cmd, stream) : launch_conv<__nv_bfloat16>(err, cmd, stream, which == 1);
      case MAG_DTYPE_FLOAT8_E4M3FN: return which == 2 ? launch_conv_wgrad<__nv_fp8_e4m3>(err, cmd, stream) : launch_conv<__nv_fp8_e4m3>(err, cmd, stream, which == 1);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: conv: unsupported dtype: %s.", mag_type_trait(x->meta.dtype)->name);
    }
  }

  mag_status_t conv_op_conv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    return conv_dispatch(err, cmd, stream, 0);
  }

  mag_status_t conv_op_conv_transpose(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    return conv_dispatch(err, cmd, stream, 1);
  }

  mag_status_t conv_op_conv_wgrad(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    return conv_dispatch(err, cmd, stream, 2);
  }
}
