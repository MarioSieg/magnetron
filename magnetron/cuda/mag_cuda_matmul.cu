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

#include "mag_cuda_matmul.cuh"

#include <algorithm>
#include <core/mag_prng_philox4x32.h>

#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/barrier>
#include <mma.h>

#include <array>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>

#define MAG_CUDA_MATMUL_USE_WMMA 1

namespace mag {
#if MAG_CUDA_MATMUL_USE_WMMA /* WMMA + TMA fast kernel */

  [[nodiscard]] static int64_t tensor_batch_total(const mag_tensor_t *tensor) {
    int64_t ra = tensor->meta.coords.rank;
    if (ra <= 2) return 1;
    int64_t batch=1;
    int64_t delta=ra-2;
    for (int64_t i=0; i < delta; ++i)
      batch *= tensor->meta.coords.shape[i];
    return batch;
  }

  static std::once_flag g_dlsym_once;
  static std::atomic<PFN_cuTensorMapEncodeTiled_v12000> g_tmap_encode_fn = nullptr;
  [[nodiscard]] static PFN_cuTensorMapEncodeTiled_v12000 lookup_proc_address_encode_tmap() {
    std::call_once(g_dlsym_once, [] {
      cudaDriverEntryPointQueryResult stat;
      PFN_cuTensorMapEncodeTiled_v12000 pfn = nullptr;
      auto res = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled",
        reinterpret_cast<void **>(&pfn),
        12000,
        cudaEnableDefault,
        &stat
      );
      if (mag_unlikely(res != cudaSuccess || stat != cudaDriverEntryPointSuccess))
        throw std::runtime_error {"Failed to get address of cuTensorMapEncodeTiled: " + std::string{cudaGetErrorString(res)}};
      g_tmap_encode_fn.store(pfn, std::memory_order_release);
    });
    return g_tmap_encode_fn.load(std::memory_order_acquire);
  }

  template <typename T, const size_t rank>
  [[nodiscard]] static CUtensorMap init_tmap_nd(
    void *base,
    const std::array<int64_t, rank> &dims,
    const std::array<int64_t, rank-1> &strides,
    const std::array<int32_t, rank> &box,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE
  ) {
    for (auto dim : dims)
      if (dim < 1) throw std::invalid_argument("dimensions must be >= 1");
    for (auto stride : strides)
      if (stride & 15) throw std::invalid_argument("strides must be multiples of 16 for TMA");

    std::array<uint64_t, rank> global_dims = {};
    std::transform(dims.begin(), dims.end(), global_dims.begin(), [](auto x) noexcept { return static_cast<uint64_t>(x); });
    std::array<uint64_t, rank-1> global_stride = {};
    std::transform(strides.begin(), strides.end(), global_stride.begin(), [](auto x) noexcept { return static_cast<uint64_t>(x); });
    std::array<uint32_t, rank> box_dim = {};
    std::transform(box.begin(), box.end(), box_dim.begin(), [](auto x) noexcept { return static_cast<uint32_t>(x); });
    std::array<uint32_t, rank> elem_stride = {};
    std::fill(elem_stride.begin(), elem_stride.end(), 1);

    CUtensorMap map{};
    CUtensorMapDataType dtype{};
    if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if constexpr (std::is_same_v<T, half>) dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    else throw std::runtime_error("unsupported dtype for TMA map");

    auto *encode = lookup_proc_address_encode_tmap();
    CUresult rc = (*encode)(
      &map,
      dtype,
      rank,
      base,
      global_dims.data(),
      global_stride.data(),
      box_dim.data(),
      elem_stride.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (rc != CUDA_SUCCESS)
      throw std::runtime_error("cuTensorMapEncodeTiled failed");
    return map;
  }

   template <typename T, bool TA, int BM, int BK>
  [[nodiscard]] static CUtensorMap init_tmap_x(const mag_tensor_t *x) {
    int64_t ra = x->meta.coords.rank;
    int64_t batch_total = tensor_batch_total(x);
    int64_t M = ra == 1 ? 1 : x->meta.coords.shape[ra-2];
    int64_t K = x->meta.coords.shape[ra-1];
    if constexpr (!TA) {
      return init_tmap_nd<T, 3>(
        reinterpret_cast<void *>(mag_tensor_data_ptr(x)),
        { K, M, batch_total },
        { K*static_cast<int64_t>(sizeof(T)), M*K*static_cast<int64_t>(sizeof(T)) },
        { BK, BM, 1 }
      );
    } else {
      return init_tmap_nd<T, 3>(
        reinterpret_cast<void *>(mag_tensor_data_ptr(x)),
        { M, K, batch_total },
        { M*static_cast<int64_t>(sizeof(T)), M*K*static_cast<int64_t>(sizeof(T)) },
        { BM, BK, 1 }
      );
    }
  }

  template <typename T, bool TB, int BK, int BN>
  [[nodiscard]] static CUtensorMap init_tmap_y(const mag_tensor_t *y) {
    int64_t ra = y->meta.coords.rank;
    int64_t batch_total = tensor_batch_total(y);
    int64_t K = ra == 1 ? y->meta.coords.shape[0] : y->meta.coords.shape[ra-2];
    int64_t N = ra == 1 ? 1 : y->meta.coords.shape[ra-1];
    if constexpr (!TB) {
      return init_tmap_nd<T, 3>(
        reinterpret_cast<void *>(mag_tensor_data_ptr(y)),
        { N, K, batch_total },
        { N*static_cast<int64_t>(sizeof(T)), K*N*static_cast<int64_t>(sizeof(T)) },
        { BN, BK, 1 }
      );
    } else {
      return init_tmap_nd<T, 3>(
        reinterpret_cast<void *>(mag_tensor_data_ptr(y)),
        { K, N, batch_total },
        { K*static_cast<int64_t>(sizeof(T)), K*N*static_cast<int64_t>(sizeof(T)) },
        { BK, BN, 1 }
      );
    }
  }

  template <typename T>
  static __device__ __forceinline__ void store_f32x2(T *o, float x, float y);

  template <>
  __device__ __forceinline__ void store_f32x2<half>(half *o, float x, float y) {
    *reinterpret_cast<half2 *>(o) = __halves2half2(__float2half_rn(x), __float2half_rn(y));
  }

  template <>
  __device__ __forceinline__ void store_f32x2<__nv_bfloat16>(__nv_bfloat16 *o, float x, float y) {
    *reinterpret_cast<__nv_bfloat162 *>(o) = __halves2bfloat162(__float2bfloat16(x), __float2bfloat16(y));
  }

  template <typename T>
  static __device__ __forceinline__ void store_tile_16x16(
    T *__restrict__ r_batch,
    int M,
    int N,
    int base_row,
    int base_col,
    const float *__restrict__ c_ptr,
    int lane
  ) {
    bool full_tile = base_row+16 <= M && base_col+16 <= N;
    auto can_store_x2 = [](const void *p) -> bool {
      return !(3&reinterpret_cast<uintptr_t>(p));
    };
    if (full_tile) {
      #pragma unroll
      for (int i=lane<<1; i < 256; i += 64) {
        int row = i>>4;
        int col = i&15;
        int out_idx = (base_row + row)*N + (base_col + col);
        auto *dst = r_batch + out_idx;
        if (can_store_x2(dst)) {
          store_f32x2<T>(dst, c_ptr[i], c_ptr[i+1]);
        } else {
          dst[0] = static_cast<T>(c_ptr[i]);
          dst[1] = static_cast<T>(c_ptr[i+1]);
        }
      }
    } else {
      #pragma unroll
      for (int i=lane<<1; i < 256; i += 64) {
        int row = i>>4;
        int col = i&15;
        int g_row = base_row + row;
        int g_col = base_col + col;
        if (g_row >= M) continue;
        int out_idx = g_row*N + g_col;
        auto *dst = r_batch + out_idx;
        if (g_col+1 < N && can_store_x2(dst)) {
          store_f32x2<T>(dst, c_ptr[i], c_ptr[i+1]);
        } else {
          if (g_col < N) dst[0] = static_cast<T>(c_ptr[i]);
          if (g_col+1 < N) dst[1] = static_cast<T>(c_ptr[i+1]);
        }
      }
    }
  }

  struct barrier final {
    uint64_t bar;

    __device__ void init(const uint32_t &count) {
      asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this))), "r"(count) : "memory");
    }

    /* Make prior generic-proxy shared writes (mbarrier.init above) visible to the async proxy.
       TMA runs in the async proxy, so without this fence it may observe an uninitialized
       mbarrier. Required by the PTX ISA between init and the first cp.async.bulk.tensor. */
    __device__ static void fence_proxy_async_shared_cta() {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }

    __device__ void cp_async_bulk_tensor_3d(void *dst, const void *tmap, const int32_t (&coords)[3]) {
      asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
        "l"(tmap),
        "r"(coords[0]), "r"(coords[1]), "r"(coords[2]),
        "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this)))
        : "memory"
      );
    }

    __device__ void arrive_expect_tx(const uint32_t &tx) {
      [[maybe_unused]] uint64_t state;
      asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
        : "=l"(state)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this))), "r"(tx)
        : "memory"
      );
    }

    [[nodiscard]] __device__ bool try_wait_parity(const uint32_t &phase_parity){
      uint32_t wait_completed;
      asm volatile(
        "{\n"
        ".reg .pred PROT;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 PROT, [%1], %2;\n"
        "selp.b32 %0, 1, 0, PROT;\n"
        "}"
        : "=r"(wait_completed)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this))), "r"(phase_parity)
        : "memory"
      );
      return static_cast<bool>(wait_completed);
    }

    __device__ void arrive(){
      [[maybe_unused]] uint64_t state;
      asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];"
        : "=l"(state)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this)))
        : "memory"
      );
    }
  };
  static_assert(sizeof(barrier) == sizeof(uint64_t));

  template <typename T, bool TA, bool TB, int BM, int BN, int BK, int STAGES>
  __global__ static void matmul_kernel_wmma(
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t batch_total,
    T *__restrict__ br,
    const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b
  ) {
    using namespace nvcuda;

    static_assert(BK % 16 == 0, "BK must be a multiple of 16 for wmma 16x16x16");
    static_assert((BM&15) == 0);
    static_assert((BN&15) == 0);
    static constexpr int WM = BM>>4;
    static constexpr int WN = BN>>4;
    static constexpr int WARP_TILES_OUT = WM*WN;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int CONSUMER_WARPS = WARP_TILES_OUT>>1;
    static constexpr int TOTAL_WARPS = PRODUCER_WARPS + CONSUMER_WARPS;
    static constexpr int BLOCK_THREADS = TOTAL_WARPS<<5;
    static constexpr int A_SIZE = BM*BK;
    static constexpr int B_SIZE = BK*BN;

    static_assert(!(WARP_TILES_OUT&1));
    static_assert(BLOCK_THREADS <= 1024);
    static_assert(CONSUMER_WARPS > 0);

    int batch = blockIdx.z;
    if (batch >= batch_total) return;

    int tile_m = blockIdx.y*BM;
    int tile_n = blockIdx.x*BN;
    int tid = threadIdx.x;
    int lane = tid&31;
    int warp_id = tid>>5;
    bool is_producer = warp_id == 0;
    int consumer_warp = warp_id-1;

    T *__restrict__ r_batch = br + static_cast<int64_t>(batch)*M*N;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    __shared__ barrier a_bar[STAGES];
    __shared__ barrier b_bar[STAGES];
    __shared__ barrier done_bar[STAGES];
    auto *a_smem = reinterpret_cast<T *>(smem_raw);
    auto *b_smem = a_smem + STAGES*A_SIZE;
    auto *c_smem = reinterpret_cast<float *>(b_smem + STAGES*B_SIZE);

    if (tid == 0) {
      #pragma unroll
      for (int s=0; s < STAGES; ++s) {
        a_bar[s].init(1);
        b_bar[s].init(1);
        done_bar[s].init(CONSUMER_WARPS);
      }
      barrier::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    auto init_tma_coords = [=](int ktile, int32_t (&ca)[3], int32_t (&cb)[3]) -> void {
      if constexpr (!TA) { // dims {K, M, batch}, box {BK, BM, 1}
        ca[0] = ktile * BK;
        ca[1] = tile_m;
        ca[2] = batch;
      } else { // dims {M, K, batch}, box {BM, BK, 1}
        ca[0] = tile_m;
        ca[1] = ktile * BK;
        ca[2] = batch;
      }
      if constexpr (!TB) { // dims {N, K, batch}, box {BN, BK, 1}
        cb[0] = tile_n;
        cb[1] = ktile * BK;
        cb[2] = batch;
      } else { // dims {K, N, batch}, box {BK, BN, 1}
        cb[0] = ktile * BK;
        cb[1] = tile_n;
        cb[2] = batch;
      }
    };

    auto issue_tma_stage = [&](int stage, int ktile) -> void {
      if (!is_producer || lane != 0) return;
      auto *a_buf = a_smem + stage * A_SIZE;
      auto *b_buf = b_smem + stage * B_SIZE;
      int32_t a_coords[3];
      int32_t b_coords[3];
      init_tma_coords(ktile, a_coords, b_coords);
      a_bar[stage].cp_async_bulk_tensor_3d(a_buf, &map_a, a_coords);
      a_bar[stage].arrive_expect_tx(sizeof(T)*A_SIZE);
      b_bar[stage].cp_async_bulk_tensor_3d(b_buf, &map_b, b_coords);
      b_bar[stage].arrive_expect_tx(sizeof(T)*B_SIZE);
    };
    auto wait_stage_ready = [&](int stage, int phase) -> void {
      while (!a_bar[stage].try_wait_parity(phase));
      while (!b_bar[stage].try_wait_parity(phase));
    };
    auto producer_wait_stage_reusable = [&](int stage, int phase) -> void {
      if (!is_producer || lane != 0) return;
      while (!done_bar[stage].try_wait_parity(phase));
    };
    auto consumer_mark_stage_done = [&](int stage) -> void {
      if (is_producer || lane != 0) return;
      done_bar[stage].arrive();
    };
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag1;
    int warp_m0 = 0, warp_n0 = 0;
    int warp_m1 = 0, warp_n1 = 0;
    if (!is_producer) {
      wmma::fill_fragment(c_frag0, 0.0f);
      wmma::fill_fragment(c_frag1, 0.0f);
      int tile0 = consumer_warp;
      int tile1 = consumer_warp + CONSUMER_WARPS;
      warp_m0 = tile0 / WN;
      warp_n0 = tile0 % WN;
      warp_m1 = tile1 / WN;
      warp_n1 = tile1 % WN;
    }

    auto compute_stage = [&](int stage) -> void {
      if (is_producer) return;
      auto *a_buf = a_smem + stage*A_SIZE;
      auto *b_buf = b_smem + stage*B_SIZE;
      #pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        if constexpr (!TA && !TB) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a0, a1;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
          wmma::load_matrix_sync(a0,     a_buf + (warp_m0<<4)*BK + kk, BK);
          wmma::load_matrix_sync(a1,     a_buf + (warp_m1<<4)*BK + kk, BK);
          wmma::load_matrix_sync(b_frag, b_buf + kk*BN + (warp_n0<<4), BN);
          wmma::mma_sync(c_frag0, a0, b_frag, c_frag0);
          wmma::mma_sync(c_frag1, a1, b_frag, c_frag1);
        } else if constexpr (TA && !TB) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> a0, a1;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
          wmma::load_matrix_sync(a0,     a_buf + kk*BM + (warp_m0<<4), BM);
          wmma::load_matrix_sync(a1,     a_buf + kk*BM + (warp_m1<<4), BM);
          wmma::load_matrix_sync(b_frag, b_buf + kk*BN + (warp_n0<<4), BN);
          wmma::mma_sync(c_frag0, a0, b_frag, c_frag0);
          wmma::mma_sync(c_frag1, a1, b_frag, c_frag1);
        } else if constexpr (!TA && TB) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a0, a1;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
          wmma::load_matrix_sync(a0,     a_buf + (warp_m0<<4)*BK + kk, BK);
          wmma::load_matrix_sync(a1,     a_buf + (warp_m1<<4)*BK + kk, BK);
          wmma::load_matrix_sync(b_frag, b_buf + (warp_n0<<4)*BK + kk, BK);
          wmma::mma_sync(c_frag0, a0, b_frag, c_frag0);
          wmma::mma_sync(c_frag1, a1, b_frag, c_frag1);
        } else {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> a0, a1;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
          wmma::load_matrix_sync(a0,     a_buf + kk*BM + (warp_m0<<4), BM);
          wmma::load_matrix_sync(a1,     a_buf + kk*BM + (warp_m1<<4), BM);
          wmma::load_matrix_sync(b_frag, b_buf + (warp_n0<<4)*BK + kk, BK);
          wmma::mma_sync(c_frag0, a0, b_frag, c_frag0);
          wmma::mma_sync(c_frag1, a1, b_frag, c_frag1);
        }
      }
    };

    int k_tiles = static_cast<int>((K + BK - 1)/BK);
    int prefetch = k_tiles < STAGES ? k_tiles : STAGES;

    if (is_producer && lane == 0) {
      #pragma unroll
      for (int s=0; s < STAGES; ++s) {
        if (s < prefetch) issue_tma_stage(s, s);
      }
    }
    for (int kt=0; kt < k_tiles; ++kt) {
      int stage = kt % STAGES;
      int phase = (kt / STAGES) & 1;
      int next_kt = kt + STAGES;
      if (!is_producer) {
        wait_stage_ready(stage, phase);
        compute_stage(stage);
        __syncwarp();
        consumer_mark_stage_done(stage);
      }
      if (is_producer && lane == 0 && next_kt < k_tiles) {
        producer_wait_stage_reusable(stage, phase);
        issue_tma_stage(stage, next_kt);
      }
    }

    if (!is_producer) {
      int tile0 = consumer_warp;
      int tile1 = consumer_warp + CONSUMER_WARPS;
      auto *c_ptr0 = c_smem + (tile0<<8);
      auto *c_ptr1 = c_smem + (tile1<<8);

      wmma::store_matrix_sync(c_ptr0, c_frag0, 16, wmma::mem_row_major);
      wmma::store_matrix_sync(c_ptr1, c_frag1, 16, wmma::mem_row_major);

      __syncwarp();

      store_tile_16x16<T>(
        r_batch,
        static_cast<int>(M),
        static_cast<int>(N),
        tile_m + (warp_m0 << 4),
        tile_n + (warp_n0 << 4),
        c_ptr0,
        lane
      );
      store_tile_16x16<T>(
        r_batch,
        static_cast<int>(M),
        static_cast<int>(N),
        tile_m + (warp_m1 << 4),
        tile_n + (warp_n1 << 4),
        c_ptr1,
        lane
      );
    }
  }

  template <typename T>
  static mag_status_t launch_matmul_kernel_wmma(
    mag_error_t *err,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_total,
    T *__restrict__ br,
    mag_tensor_t *x, mag_tensor_t *y,
    bool xT, bool yT
  ) {
    static_assert(std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>);
    static constexpr int BM = 128;
    static constexpr int BN = 64;
    static constexpr int BK = 32;
    static constexpr int STAGES = 4;
    static constexpr int BLOCK_THREADS = (1 + ((BM/16)*(BN/16))/2) * 32;
    int max_smem_real;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    size_t smem = sizeof(T)*STAGES*(BM*BK + BN*BK) + sizeof(float)*((BM>>4)*(BN>>4)<<8);
    if (smem > (unsigned)max_smem_real)
      return mag_set_error(err, MAG_ERR_OP, "cuda: matmul shared memory requirement (%u bytes) exceeds device limit (%d bytes).", static_cast<unsigned>(smem), max_smem_real);
    auto set_kernel_smem_size = [&](auto kernel, size_t size) -> void {
      mag_assert2(size <= INT32_MAX);
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(size));
    };
    dim3 grid_dim(static_cast<unsigned>((N + BN-1)/BN), static_cast<unsigned>((M + BM-1)/BM), static_cast<unsigned>(batch_total));
    dim3 block_dim(BLOCK_THREADS, 1, 1);
    if (!xT && !yT) {
      CUtensorMap map_a = init_tmap_x<T, false, BM, BK>(x);
      CUtensorMap map_b = init_tmap_y<T, false, BK, BN>(y);
      auto *kernel = matmul_kernel_wmma<T, false, false, BM, BN, BK, STAGES>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
    } else if (!xT && yT) {
      CUtensorMap map_a = init_tmap_x<T, false, BM, BK>(x);
      CUtensorMap map_b = init_tmap_y<T, true, BK, BN>(y);
      auto *kernel = matmul_kernel_wmma<T, false, true, BM, BN, BK, STAGES>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
    } else if (xT && !yT) {
      CUtensorMap map_a = init_tmap_x<T, true, BM, BK>(x);
      CUtensorMap map_b = init_tmap_y<T, false, BK, BN>(y);
      auto *kernel = matmul_kernel_wmma<T, true, false, BM, BN, BK, STAGES>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
    } else {
      CUtensorMap map_a = init_tmap_x<T, true, BM, BK>(x);
      CUtensorMap map_b = init_tmap_y<T, true, BK, BN>(y);
      auto *kernel = matmul_kernel_wmma<T, true, true, BM, BN, BK, STAGES>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
    }
    return MAG_OK;
  }

#endif

  // In order
  // https://siboehm.com/articles/22/CUDA-MMM
  // https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
  // https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
  // https://gau-nernst.github.io/tcgen05/

  template <typename T, bool TA, bool TB, int BM, int BN, int BK, int TM, int TN>
  __global__ static void matmul_kernel_fallback(
    int M, int N, int K,
    int batch_total,
    T *br, const T *bx, const T *by
  ) {
    static constexpr int A_SIZE = BM*BK;
    static constexpr int B_SIZE = BK*BN;
    static constexpr int STAGES = 2;
    extern __shared__ uint8_t smem[];
    auto *a_smem = reinterpret_cast<T *>(smem);
    auto *b_smem = reinterpret_cast<T *>(smem) + STAGES*A_SIZE;
    int batch = blockIdx.z;
    if (batch >= batch_total) return;
    bx += batch*M*K;
    by += batch*K*N;
    br += batch*M*N;
    int a_row_stride = TA ? 1 : K;
    int a_col_stride = TA ? M : 1;
    int b_row_stride = TB ? 1 : N;
    int b_col_stride = TB ? K : 1;
    int tile_m = blockIdx.y * BM;
    int tile_n = blockIdx.x * BN;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y*blockDim.x + threadIdx.x;
    int nthreads = blockDim.x*blockDim.y;
    int local_m0 = ty * TM;
    int local_n0 = tx * TN;
    float acc[TM][TN] = {};
    auto load_stage = [&](int stage, int k0) {
      auto *a_buf = a_smem + stage*A_SIZE;
      auto *b_buf = b_smem + stage*B_SIZE;
      #pragma unroll
      for (int i=tid; i < A_SIZE; i += nthreads) {
        int row = i / BK;
        int col = i % BK;
        int g_row = tile_m + row;
        int g_col = k0 + col;
        a_buf[i] = g_row < M && g_col < K ? bx[g_row*a_row_stride + g_col*a_col_stride] : T{};
      }
      #pragma unroll
      for (int i=tid; i < B_SIZE; i += nthreads) {
        int row = i / BN;
        int col = i % BN;
        int g_row = k0 + row;
        int g_col = tile_n + col;
        b_buf[i] = g_row < K && g_col < N ? by[g_row*b_row_stride + g_col*b_col_stride] : T{};
      }
    };

    auto compute_stage = [&](int stage) {
      auto *a_buf = a_smem + stage*A_SIZE;
      auto *b_buf = b_smem + stage*B_SIZE;

      #pragma unroll
      for (int kk=0; kk < BK; ++kk) {
        float a_frag[TM];
        float b_frag[TN];
        #pragma unroll
        for (int i=0; i < TM; ++i) {
          a_frag[i] = static_cast<float>(a_buf[(local_m0 + i)*BK + kk]);
        }
        #pragma unroll
        for (int i=0; i < TN; ++i) {
          b_frag[i] = static_cast<float>(b_buf[kk*BN + (local_n0 + i)]);
        }
        #pragma unroll
        for (int i=0; i < TM; ++i) {
          #pragma unroll
          for (int j=0; j < TN; ++j) {
            acc[i][j] += a_frag[i] * b_frag[j];
          }
        }
      }
    };

    int k0 = 0;
    int stage = 0;
    load_stage(stage, k0);
    __syncthreads();

    for (; k0 < K; k0 += BK) {
      int next_k0 = k0 + BK;
      int next_stage = stage^1;
      if (next_k0 < K)
        load_stage(next_stage, next_k0);
      compute_stage(stage);
      __syncthreads();
      stage = next_stage;
    }

    #pragma unroll
    for (int i=0; i < TM; ++i) {
      int g_row = tile_m + local_m0 + i;
      if (g_row >= M) continue;
      #pragma unroll
      for (int j=0; j < TN; ++j) {
        int g_col = tile_n + local_n0 + j;
        if (g_col >= N) continue;
        br[g_row*N + g_col] = static_cast<T>(acc[i][j]);
      }
    }
  }

  template <typename T>
  static mag_status_t launch_matmul_kernel_fallback(
    mag_error_t *err,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_total,
    T *__restrict__ br,
    const T *bx,
    const T *by,
    bool xT, bool yT
  ) {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 32;
    static constexpr int TM = 4;
    static constexpr int TN = 4;
    static constexpr int STAGES = 2;
    static constexpr int TRX = BN/TN;
    static constexpr int TRY = BM/TM;
    static_assert(TRX*TRY <= 1024);

    int64_t blocks_x = (N + BN-1)/BN;
    int64_t blocks_y = (M + BM-1)/BM;
    dim3 grid_dim = dim3(blocks_x, blocks_y, batch_total);
    dim3 block_dim = dim3(TRX, TRY, 1);

    int max_smem_real;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    size_t smem = STAGES * (BM*BK + BN*BK) * sizeof(T);
    if (smem > (unsigned)max_smem_real)
      return mag_set_error(err, MAG_ERR_OP, "cuda: matmul shared memory requirement (%u bytes) exceeds device limit (%d bytes).", static_cast<unsigned>(smem), max_smem_real);
    auto set_kernel_smem_size = [&](auto kernel, size_t size) -> void {
      mag_assert2(size <= INT32_MAX);
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(size));
    };

    if (!xT && !yT) {
      auto *kernel = matmul_kernel_fallback<T, false, false, BM, BN, BK, TM, TN>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, bx, by);
    } else if (!xT && yT) {
      auto *kernel = matmul_kernel_fallback<T, false, true, BM, BN, BK, TM, TN>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, bx, by);
    } else if (xT && !yT) {
      auto *kernel = matmul_kernel_fallback<T, true, false, BM, BN, BK, TM, TN>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, bx, by);
    } else {
      auto *kernel = matmul_kernel_fallback<T, true, true, BM, BN, BK, TM, TN>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, bx, by);
    }
    return MAG_OK;
  }

  template <typename T>
  [[nodiscard]] static bool is_tensor_tma_compat_x(const mag_tensor_t *x, bool TA) {
    int64_t r = x->meta.coords.rank;
    int64_t batch_total = tensor_batch_total(x);
    int64_t M = r == 1 ? 1 : x->meta.coords.shape[r-2];
    int64_t K = x->meta.coords.shape[r-1];
    if (batch_total < 1) return false;
    if (15 & mag_tensor_data_ptr(x)) return false;
    return !TA ? !(15 & (K*sizeof(T))) && !(15 & (M*K*sizeof(T))) : !(15 & (M*sizeof(T))) && !(15 & (M*K*sizeof(T)));
  }

  template <typename T>
  [[nodiscard]] static bool is_tensor_tma_compat_y(const mag_tensor_t *y, bool TB) {
    int64_t r = y->meta.coords.rank;
    int64_t batch_total = tensor_batch_total(y);
    int64_t K = r == 1 ? y->meta.coords.shape[0] : y->meta.coords.shape[r-2];
    int64_t N = r == 1 ? 1 : y->meta.coords.shape[r-1];
    if (batch_total < 1) return false;
    if (15 & mag_tensor_data_ptr(y)) return false;
    return !TB ? !(15 & (N*sizeof(T))) && !(15 & (K*N*sizeof(T))) : !(15 & (K*sizeof(T))) && !(15 & (K*N*sizeof(T)));
  }

  template <typename T, bool YT, int THREADS_X, int OUTS_PER_BLOCK, bool VEC128>
  __global__ static void gemv_m1_kernel(
    int N, int K, int batch_total,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ by
  ) {
    static_assert(!(THREADS_X & 31));
    static_assert(THREADS_X*OUTS_PER_BLOCK <= 1024);
    static constexpr int WARPS_X = THREADS_X >> 5;
    int batch = blockIdx.y;
    if (batch >= batch_total) return;
    bx += batch*K;
    by += batch*K*N;
    br += batch*N;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int lane = tx & 31;
    int warp_x = tx >> 5;
    int out = blockIdx.x*OUTS_PER_BLOCK + ty;
    if (out >= N) return;
    float sum = 0.0f;
    if constexpr (YT) {
      const T *__restrict__ w_row = by + out*K;
      if constexpr (VEC128 && (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>)) {
        const uint4 *x_vec = reinterpret_cast<const uint4 *>(bx);
        const uint4 *w_vec = reinterpret_cast<const uint4 *>(w_row);
        int Kv = K >> 3;
        #pragma unroll 4
        for (int kv = tx; kv < Kv; kv += THREADS_X) {
          uint4 xv = __ldg(x_vec + kv);
          uint4 wv = __ldg(w_vec + kv);
          if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            const __nv_bfloat162 *xa = reinterpret_cast<const __nv_bfloat162 *>(&xv);
            const __nv_bfloat162 *wa = reinterpret_cast<const __nv_bfloat162 *>(&wv);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              float2 a = __bfloat1622float2(xa[i]);
              float2 b = __bfloat1622float2(wa[i]);
              sum = __fmaf_rn(a.x, b.x, sum);
              sum = __fmaf_rn(a.y, b.y, sum);
            }
          } else {
            const half2 *xa = reinterpret_cast<const half2 *>(&xv);
            const half2 *wa = reinterpret_cast<const half2 *>(&wv);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              float2 a = __half22float2(xa[i]);
              float2 b = __half22float2(wa[i]);
              sum = __fmaf_rn(a.x, b.x, sum);
              sum = __fmaf_rn(a.y, b.y, sum);
            }
          }
        }
        for (int k = (K & ~7) + tx; k < K; k += THREADS_X)
          sum = __fmaf_rn(static_cast<float>(bx[k]), static_cast<float>(w_row[k]), sum);
      } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        const __nv_bfloat162 *x2 = reinterpret_cast<const __nv_bfloat162 *>(bx);
        const __nv_bfloat162 *w2 = reinterpret_cast<const __nv_bfloat162 *>(w_row);
        int K2 = K >> 1;
        for (int k2 = tx; k2 < K2; k2 += THREADS_X) {
          float2 a = __bfloat1622float2(__ldg(x2 + k2));
          float2 b = __bfloat1622float2(__ldg(w2 + k2));
          sum = __fmaf_rn(a.x, b.x, sum);
          sum = __fmaf_rn(a.y, b.y, sum);
        }
        if ((K & 1) && tx == 0)
          sum = __fmaf_rn(static_cast<float>(bx[K-1]), static_cast<float>(w_row[K-1]), sum);
      } else if constexpr (std::is_same_v<T, half>) {
        const half2 *x2 = reinterpret_cast<const half2 *>(bx);
        const half2 *w2 = reinterpret_cast<const half2 *>(w_row);
        int K2 = K >> 1;
        for (int k2 = tx; k2 < K2; k2 += THREADS_X) {
          float2 a = __half22float2(__ldg(x2 + k2));
          float2 b = __half22float2(__ldg(w2 + k2));
          sum = __fmaf_rn(a.x, b.x, sum);
          sum = __fmaf_rn(a.y, b.y, sum);
        }
        if ((K & 1) && tx == 0)
          sum = __fmaf_rn(static_cast<float>(bx[K-1]), static_cast<float>(w_row[K-1]), sum);
      } else {
        for (int k = tx; k < K; k += THREADS_X)
          sum = __fmaf_rn(static_cast<float>(__ldg(bx + k)), static_cast<float>(__ldg(w_row + k)), sum);
      }
    } else {
      for (int k = tx; k < K; k += THREADS_X)
        sum = __fmaf_rn(static_cast<float>(__ldg(bx + k)), static_cast<float>(__ldg(by + k*N + out)), sum);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float warp_sums[OUTS_PER_BLOCK][WARPS_X];
    if (lane == 0) warp_sums[ty][warp_x] = sum;
    __syncthreads();
    if (warp_x == 0) {
      sum = lane < WARPS_X ? warp_sums[ty][lane] : 0.0f;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      if (lane == 0) br[out] = static_cast<T>(sum);
    }
  }

  template <typename T>
  static void launch_gemv_m1_kernel(
    int64_t N, int64_t K, int64_t batch_total,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ by,
    [[maybe_unused]] bool xT, bool yT
  ) {
    static constexpr int THREADS_X = 128;
    static constexpr int OUTS_PER_BLOCK = 4;
    bool vec128 = !(K&6) && !(reinterpret_cast<uintptr_t>(bx) & 15) && !(reinterpret_cast<uintptr_t>(by) & 15);
    dim3 block_dim(THREADS_X, OUTS_PER_BLOCK, 1);
    dim3 grid_dim(static_cast<unsigned>((N + OUTS_PER_BLOCK - 1)/OUTS_PER_BLOCK), static_cast<unsigned>(batch_total), 1);
    if (yT && vec128) {
      auto *k = gemv_m1_kernel<T, true, THREADS_X, OUTS_PER_BLOCK, true>;
      k<<<grid_dim, block_dim>>>(N, K, batch_total, br, bx, by);
    } else if (yT) {
      auto *k = gemv_m1_kernel<T, true, THREADS_X, OUTS_PER_BLOCK, false>;
      k<<<grid_dim, block_dim>>>(N, K, batch_total, br, bx, by);
    } else {
      auto *k = gemv_m1_kernel<T, false, THREADS_X, OUTS_PER_BLOCK, false>;
      k<<<grid_dim, block_dim>>>(N, K, batch_total, br, bx, by);
    }
  }

  template <typename T>
  static mag_status_t launch_matmul(mag_error_t *err, const mag_command_t &cmd) {
    mag_tensor_t *r = cmd.out[0];
    mag_tensor_t *x = cmd.in[0];
    mag_tensor_t *y = cmd.in[1];
    mag_assert2(mag_tensor_is_contiguous(r));
    bool x_batch_packed, y_batch_packed;
    mag_mat_layout_type_t x_layout = mag_mat_layout_detect(&x->meta.coords, &x_batch_packed);
    mag_mat_layout_type_t y_layout = mag_mat_layout_detect(&y->meta.coords, &y_batch_packed);
    bool x_ok = x_layout != MAG_MAT_LAYOUT_TYPE_OTHER && x_batch_packed;
    bool y_ok = y_layout != MAG_MAT_LAYOUT_TYPE_OTHER && y_batch_packed;
    bool xT = x_ok && x_layout == MAG_MAT_LAYOUT_TYPE_TRANSPOSED;
    bool yT = y_ok && y_layout == MAG_MAT_LAYOUT_TYPE_TRANSPOSED;
    bool cloned_x = false;
    bool cloned_y = false;
    if (!x_ok) {
      mag_contiguous(nullptr, &x, x);
      xT = false;
      cloned_x = true;
    }
    if (!y_ok) {
      mag_contiguous(nullptr, &y, y);
      yT = false;
      cloned_y = true;
    }
    int64_t M = x->meta.coords.rank == 1 ? 1 : x->meta.coords.shape[x->meta.coords.rank - 2];
    int64_t Kx = x->meta.coords.shape[x->meta.coords.rank - 1];
    int64_t N = y->meta.coords.rank == 1 ? 1 : y->meta.coords.shape[y->meta.coords.rank - 1];
    int64_t Ky = y->meta.coords.rank == 1 ? y->meta.coords.shape[0] : y->meta.coords.shape[y->meta.coords.rank - 2];
    mag_assert2(Kx == Ky);
    int64_t K = Kx;
    int64_t batch_rank = r->meta.coords.rank > 2 ? r->meta.coords.rank-2 : 0;
    int64_t batch_total = std::accumulate(r->meta.coords.shape, r->meta.coords.shape + batch_rank, 1, std::multiplies<int64_t>());
    auto *__restrict__ br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
    const auto *__restrict__ bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
    const auto *__restrict__ by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));
    mag_matmul_type_t mm_type = mag_matmul_type_detect(x, y);
    mag_status_t st = MAG_OK;
    switch (mm_type) {
      case MAG_MATMUL_TYPE_DOT:
      case MAG_MATMUL_TYPE_BMM_DOT:
      case MAG_MATMUL_TYPE_GEMV_VEC_MAT:
      case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT:
        launch_gemv_m1_kernel(N, K, batch_total, br, bx, by, xT, yT);
        goto end;
      case MAG_MATMUL_TYPE_GEMV_MAT_VEC:
      case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC:
        launch_gemv_m1_kernel(M, K, batch_total, br, by, bx, yT, true);
        goto end;
      default: break;
    }
    #if MAG_CUDA_MATMUL_USE_WMMA
      if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>) {
        if (is_tensor_tma_compat_x<T>(x, xT) && is_tensor_tma_compat_y<T>(y, yT)) {
          st = launch_matmul_kernel_wmma(err, M, N, K, batch_total, br, x, y, xT, yT);
          goto end;
        }
      }
    #endif
    st = launch_matmul_kernel_fallback(err, M, N, K, batch_total, br, bx, by, xT, yT);
    [[maybe_unused]] end:
      if (cloned_x) mag_tensor_decref(x);
      if (cloned_y) mag_tensor_decref(y);
    return st;
  }

  mag_status_t misc_op_matmul(mag_error_t *err, const mag_command_t &cmd) {
    const mag_tensor_t *x = cmd.in[0];
    switch (x->meta.dtype) {
      case MAG_DTYPE_FLOAT32: return launch_matmul<float>(err, cmd);
      case MAG_DTYPE_FLOAT16: return launch_matmul<half>(err, cmd);
      case MAG_DTYPE_BFLOAT16: return launch_matmul<__nv_bfloat16>(err, cmd);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: matmul: unsupported dtype %s.", mag_type_trait(x->meta.dtype)->name);
    }
  }
}
