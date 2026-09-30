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
#include "mag_cuda_gemm.cuh"

#include <cuda_runtime.h>

#include <algorithm>

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

  struct im2col_params final {
    conv_geom g;
    int Kp;
    int Kpad;
    int P;
    int p0;
    int pcount;
    int pitch;
    int n0;
    int nb;
    int grp;
  };

  template <typename T, bool TRANSPOSED>
  __global__ static void im2col_kernel(im2col_params p, const T *__restrict__ x, T *__restrict__ col) {
    const conv_geom &g = p.g;
    const int cinG = g.cin/g.groups;
    const int Id = TRANSPOSED ? g.small[0] : g.big[0], Ih = TRANSPOSED ? g.small[1] : g.big[1], Iw = TRANSPOSED ? g.small[2] : g.big[2];
    const int Oh = TRANSPOSED ? g.big[1] : g.small[1], Ow = TRANSPOSED ? g.big[2] : g.small[2];
    const int Kd = g.k[0], Kh = g.k[1], Kw = g.k[2];
    const int Kprod = Kd*Kh*Kw;
    const int64_t total = static_cast<int64_t>(p.nb)*p.Kpad*p.pcount;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<int64_t>(blockDim.x)*gridDim.x) {
      int pos = static_cast<int>(idx % p.pcount);
      int64_t t = idx / p.pcount;
      int kp = static_cast<int>(t % p.Kpad);
      int nbi = static_cast<int>(t / p.Kpad);
      float v = 0.0f;
      T tv = static_cast<T>(v);
      if (kp < p.Kp) {
        int cig = kp / Kprod;
        int kk = kp - cig*Kprod;
        int kw = kk % Kw;
        int kh = (kk / Kw) % Kh;
        int kd = kk / (Kw*Kh);
        int pidx = p.p0 + pos;
        int ow = pidx % Ow;
        int oh = (pidx / Ow) % Oh;
        int od = pidx / (Ow*Oh);
        int ci = p.grp*cinG + cig;
        int n = p.n0 + nbi;
        int id, ih, iw;
        bool ok;
        if constexpr (!TRANSPOSED) {
          id = od*g.s[0] - g.p[0] + kd*g.d[0];
          ih = oh*g.s[1] - g.p[1] + kh*g.d[1];
          iw = ow*g.s[2] - g.p[2] + kw*g.d[2];
          ok = id >= 0 && id < Id && ih >= 0 && ih < Ih && iw >= 0 && iw < Iw;
        } else {
          int td = od + g.p[0] - kd*g.d[0];
          int th = oh + g.p[1] - kh*g.d[1];
          int tw = ow + g.p[2] - kw*g.d[2];
          ok = td >= 0 && th >= 0 && tw >= 0 && !(td % g.s[0]) && !(th % g.s[1]) && !(tw % g.s[2]);
          id = td/g.s[0];
          ih = th/g.s[1];
          iw = tw/g.s[2];
          ok = ok && id < Id && ih < Ih && iw < Iw;
        }
        if (ok) tv = x[((static_cast<int64_t>(n)*g.cin + ci)*Id + id)*Ih*Iw + static_cast<int64_t>(ih)*Iw + iw];
      }
      col[(static_cast<int64_t>(nbi)*p.Kpad + kp)*p.pitch + pos] = tv;
    }
  }

  /* wp[cout][Kpad] <- w, zero padded past Kp. TRANSPOSED_W reads the conv-transpose weight layout [cin][coutG][k]. */
  template <typename T, bool TRANSPOSED_W>
  __global__ static void pack_weights_kernel(int cout, int coutG, int cinG, int Kprod, int Kp, int Kpad, const T *__restrict__ w, T *__restrict__ wp) {
    const int64_t total = static_cast<int64_t>(cout)*Kpad;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<int64_t>(blockDim.x)*gridDim.x) {
      int kp = static_cast<int>(idx % Kpad);
      int co = static_cast<int>(idx / Kpad);
      T v = static_cast<T>(0.0f);
      if (kp < Kp) {
        int cig = kp / Kprod;
        int kk = kp - cig*Kprod;
        if constexpr (!TRANSPOSED_W) {
          v = w[(static_cast<int64_t>(co)*cinG + cig)*Kprod + kk];
        } else {
          int grp = co / coutG;
          int cog = co - grp*coutG;
          int ci = grp*cinG + cig;
          v = w[(static_cast<int64_t>(ci)*coutG + cog)*Kprod + kk];
        }
      }
      wp[idx] = v;
    }
  }

  [[nodiscard]] static size_t conv_scratch_budget(const physical_device &dev) noexcept {
    size_t budget = dev.vram()/8;
    budget = std::max<size_t>(budget, size_t{64}<<20);
    budget = std::min<size_t>(budget, size_t{2}<<30);
    return budget;
  }

  [[nodiscard]] static int64_t round_up(int64_t v, int64_t m) noexcept { return (v + m - 1)/m*m; }

  template <typename T>
  [[nodiscard]] static constexpr bool conv_gemm_dtype() noexcept {
    return std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>;
  }

  template <typename T>
  static mag_status_t launch_conv_gemm(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, bool transposed, bool &handled) {
    handled = false;
    if constexpr (!conv_gemm_dtype<T>()) return MAG_OK;
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *x = cmd.in[0];
    const mag_tensor_t *w = cmd.in[1];
    const mag_tensor_t *b = cmd.num_in > 2 ? cmd.in[2] : nullptr;
    physical_device &dev = device_of(r);
    conv_geom g = transposed
      ? make_conv_geom(*cmd.params, r->meta.coords.shape+2, x->meta.coords.shape+2, w->meta.coords.shape+2)
      : make_conv_geom(*cmd.params, x->meta.coords.shape+2, r->meta.coords.shape+2, w->meta.coords.shape+2);
    g.N = static_cast<int>(x->meta.coords.shape[0]);
    g.cin = static_cast<int>(x->meta.coords.shape[1]);
    g.cout = static_cast<int>(r->meta.coords.shape[1]);
    if (g.groups < 1 || g.cin % g.groups || g.cout % g.groups) return MAG_OK;
    const int cinG = g.cin/g.groups;
    const int coutG = g.cout/g.groups;
    const int64_t Kprod = static_cast<int64_t>(g.k[0])*g.k[1]*g.k[2];
    const int64_t Kp = cinG*Kprod;
    const int64_t Kpad = round_up(Kp, 8);
    const int64_t P = transposed
      ? static_cast<int64_t>(g.big[0])*g.big[1]*g.big[2]
      : static_cast<int64_t>(g.small[0])*g.small[1]*g.small[2];
    if (Kp < 1 || P < 1 || Kpad > INT32_MAX || P > INT32_MAX) return MAG_OK;
    const bool pack_w = transposed || Kpad != Kp;

    /* Scratch: [packed weights][im2col chunk] */
    const size_t budget = conv_scratch_budget(dev);
    const size_t w_bytes = pack_w ? round_up(static_cast<int64_t>(g.cout)*Kpad*sizeof(T), 256) : 0;
    if (w_bytes >= budget) return MAG_OK;
    const int64_t max_pos = static_cast<int64_t>((budget - w_bytes)/(Kpad*sizeof(T)));
    if (max_pos < 8) return MAG_OK;
    int64_t p_chunk = std::min<int64_t>(round_up(P, 64), max_pos/64*64);
    if (p_chunk < 64) p_chunk = max_pos/8*8;
    if (p_chunk < 8) return MAG_OK;
    int64_t nb = p_chunk >= P ? std::clamp<int64_t>(max_pos/p_chunk, 1, g.N) : 1;
    const size_t col_bytes = static_cast<size_t>(nb)*Kpad*p_chunk*sizeof(T);
    if (mag_status_t st = dev.reserve_scratch(err, w_bytes + col_bytes); mag_iserr(st)) return st;
    auto *scratch = static_cast<uint8_t *>(dev.scratch());
    auto *wp = reinterpret_cast<T *>(scratch);
    auto *col = reinterpret_cast<T *>(scratch + w_bytes);

    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    const auto *bw = reinterpret_cast<const T *>(mag_tensor_data_ptr(w));
    const auto *bb = b ? reinterpret_cast<const T *>(mag_tensor_data_ptr(b)) : nullptr;

    if (pack_w) {
      int64_t total = static_cast<int64_t>(g.cout)*Kpad;
      unsigned blocks = static_cast<unsigned>(grid_for(total));
      if (transposed) pack_weights_kernel<T, true><<<blocks, CONV_BLOCK_SIZE, 0, stream>>>(g.cout, coutG, cinG, static_cast<int>(Kprod), static_cast<int>(Kp), static_cast<int>(Kpad), bw, wp);
      else pack_weights_kernel<T, false><<<blocks, CONV_BLOCK_SIZE, 0, stream>>>(g.cout, coutG, cinG, static_cast<int>(Kprod), static_cast<int>(Kp), static_cast<int>(Kpad), bw, wp);
      mag_cu_rt_check(err, cudaGetLastError(), "conv: weight packing");
    }
    const T *a_base = pack_w ? wp : bw;
    const int64_t lda = pack_w ? Kpad : Kp;

    for (int64_t n0=0; n0 < g.N; n0 += nb) {
      const int64_t nbc = std::min<int64_t>(nb, g.N - n0);
      for (int64_t p0=0; p0 < P; p0 += p_chunk) {
        const int64_t pc = std::min<int64_t>(p_chunk, P - p0);
        for (int grp=0; grp < g.groups; ++grp) {
          im2col_params ip {};
          ip.g = g;
          ip.Kp = static_cast<int>(Kp);
          ip.Kpad = static_cast<int>(Kpad);
          ip.P = static_cast<int>(P);
          ip.p0 = static_cast<int>(p0);
          ip.pcount = static_cast<int>(pc);
          ip.pitch = static_cast<int>(p_chunk);
          ip.n0 = static_cast<int>(n0);
          ip.nb = static_cast<int>(nbc);
          ip.grp = grp;
          unsigned blocks = static_cast<unsigned>(grid_for(nbc*Kpad*pc));
          if (transposed) im2col_kernel<T, true><<<blocks, CONV_BLOCK_SIZE, 0, stream>>>(ip, bx, col);
          else im2col_kernel<T, false><<<blocks, CONV_BLOCK_SIZE, 0, stream>>>(ip, bx, col);
          mag_cu_rt_check(err, cudaGetLastError(), "conv: im2col");

          gemm_desc d {};
          d.dtype = r->meta.dtype;
          d.a = a_base + static_cast<int64_t>(grp)*coutG*lda;
          d.a_kmajor = true;
          d.lda = lda;
          d.b = col;
          d.b_kmajor = false;
          d.ldb = p_chunk;
          d.c = br + (n0*g.cout + static_cast<int64_t>(grp)*coutG)*P + p0;
          d.ldc = P;
          d.bias = bb ? bb + static_cast<int64_t>(grp)*coutG : nullptr;
          d.M = coutG;
          d.N = pc;
          d.K = Kpad;
          d.batch = nbc;
          d.a_bstride = 0;
          d.b_bstride = Kpad*p_chunk;
          d.c_bstride = static_cast<int64_t>(g.cout)*P;
          if (mag_status_t st = gemm(err, dev, d, stream); mag_iserr(st)) return st;
        }
      }
    }
    handled = true;
    return MAG_OK;
  }

  template <typename T>
  static mag_status_t launch_conv_wgrad_gemm(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, bool &handled) {
    handled = false;
    if constexpr (!conv_gemm_dtype<T>()) return MAG_OK;
    mag_tensor_t *r = cmd.out[0];
    const mag_tensor_t *a = cmd.in[0];
    const mag_tensor_t *b = cmd.in[1];
    physical_device &dev = device_of(r);
    conv_geom g = make_conv_geom(*cmd.params, a->meta.coords.shape+2, b->meta.coords.shape+2, r->meta.coords.shape+2);
    g.N = static_cast<int>(a->meta.coords.shape[0]);
    g.cin = static_cast<int>(a->meta.coords.shape[1]);
    g.cout = static_cast<int>(b->meta.coords.shape[1]);
    if (g.groups < 1 || g.cin % g.groups || g.cout % g.groups || g.N < 1) return MAG_OK;
    const int cinG = g.cin/g.groups;
    const int coutG = g.cout/g.groups;
    const int64_t Kprod = static_cast<int64_t>(g.k[0])*g.k[1]*g.k[2];
    const int64_t Kp = cinG*Kprod;
    const int64_t P = static_cast<int64_t>(g.small[0])*g.small[1]*g.small[2];
    if (Kp < 1 || P < 1 || Kp > INT32_MAX || P > INT32_MAX) return MAG_OK;
    if (P & 7) return MAG_OK;
    const size_t col_bytes = static_cast<size_t>(g.N)*Kp*P*sizeof(T);
    if (col_bytes > conv_scratch_budget(dev)) return MAG_OK;
    if (mag_status_t st = dev.reserve_scratch(err, col_bytes); mag_iserr(st)) return st;
    auto *col = static_cast<T *>(dev.scratch());
    auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *ba = reinterpret_cast<const T *>(mag_tensor_data_ptr(a));
    const auto *bb = reinterpret_cast<const T *>(mag_tensor_data_ptr(b));
    for (int grp=0; grp < g.groups; ++grp) {
      im2col_params ip {};
      ip.g = g;
      ip.Kp = static_cast<int>(Kp);
      ip.Kpad = static_cast<int>(Kp);
      ip.P = static_cast<int>(P);
      ip.p0 = 0;
      ip.pcount = static_cast<int>(P);
      ip.pitch = static_cast<int>(P);
      ip.n0 = 0;
      ip.nb = g.N;
      ip.grp = grp;
      unsigned blocks = static_cast<unsigned>(grid_for(static_cast<int64_t>(g.N)*Kp*P));
      im2col_kernel<T, false><<<blocks, CONV_BLOCK_SIZE, 0, stream>>>(ip, ba, col);
      mag_cu_rt_check(err, cudaGetLastError(), "conv_wgrad: im2col");

      gemm_desc d {};
      d.dtype = r->meta.dtype;
      d.a = bb + static_cast<int64_t>(grp)*coutG*P;   /* dY[n][coutG][P]: K-major rows of length P */
      d.a_kmajor = true;
      d.lda = P;
      d.b = col;                                       /* col[n][Kp][P]: B^T rows of length P */
      d.b_kmajor = true;
      d.ldb = P;
      d.c = br + static_cast<int64_t>(grp)*coutG*Kp;
      d.ldc = Kp;
      d.M = coutG;
      d.N = Kp;
      d.K = P;
      d.batch = 1;
      d.k_batch = g.N;
      d.a_kbstride = static_cast<int64_t>(g.cout)*P;
      d.b_kbstride = Kp*P;
      if (mag_status_t st = gemm(err, dev, d, stream); mag_iserr(st)) return st;
    }
    handled = true;
    return MAG_OK;
  }

  template <typename T>
  static mag_status_t conv_dispatch_typed(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, int which) {
    if (!mag_tensor_numel(cmd.out[0])) return MAG_OK;
    bool handled = false;
    mag_status_t st = which == 2
      ? launch_conv_wgrad_gemm<T>(err, cmd, stream, handled)
      : launch_conv_gemm<T>(err, cmd, stream, which == 1, handled);
    if (mag_iserr(st) || handled) return st;
    return which == 2 ? launch_conv_wgrad<T>(err, cmd, stream) : launch_conv<T>(err, cmd, stream, which == 1);
  }

  static mag_status_t conv_dispatch(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream, int which) {
    const mag_tensor_t *x = cmd.in[0];
    switch (x->meta.dtype) {
      case MAG_DTYPE_FLOAT32: return conv_dispatch_typed<float>(err, cmd, stream, which);
      case MAG_DTYPE_FLOAT16: return conv_dispatch_typed<half>(err, cmd, stream, which);
      case MAG_DTYPE_BFLOAT16: return conv_dispatch_typed<__nv_bfloat16>(err, cmd, stream, which);
      case MAG_DTYPE_FLOAT8_E4M3FN: return conv_dispatch_typed<__nv_fp8_e4m3>(err, cmd, stream, which);
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
