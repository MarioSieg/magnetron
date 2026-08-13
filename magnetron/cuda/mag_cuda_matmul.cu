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
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
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

  template <typename T, bool TA, bool TB, int BM, int BN, int BK, int WT_M, int WT_N, int STAGES>
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
    static constexpr int TM = BM>>4;  /* 16x16 accumulator tiles spanning the block tile */
    static constexpr int TN = BN>>4;
    static_assert(TM%WT_M == 0 && TN%WT_N == 0, "warp tile must divide the block tile");
    static constexpr int WARPS_M = TM/WT_M;
    static constexpr int WARPS_N = TN/WT_N;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int CONSUMER_WARPS = WARPS_M*WARPS_N;
    static constexpr int TOTAL_WARPS = PRODUCER_WARPS + CONSUMER_WARPS;
    static constexpr int BLOCK_THREADS = TOTAL_WARPS<<5;
    static constexpr int A_SIZE = BM*BK;
    static constexpr int B_SIZE = BK*BN;

    static_assert(BLOCK_THREADS <= 1024);
    static_assert(CONSUMER_WARPS > 0);

    /* Each warp owns a WT_M x WT_N grid of 16x16 accumulators, so one k-step costs
       WT_M+WT_N fragment loads and yields WT_M*WT_N mma ops. Keeping that ratio well
       above 1 is what keeps the tensor cores fed instead of the shared memory pipe. */
    using a_layout = std::conditional_t<TA, wmma::col_major, wmma::row_major>;
    using b_layout = std::conditional_t<TB, wmma::col_major, wmma::row_major>;

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
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[WT_M][WT_N];
    #pragma unroll
    for (int i=0; i < WT_M; ++i)
      #pragma unroll
      for (int j=0; j < WT_N; ++j)
        wmma::fill_fragment(c_frag[i][j], 0.0f);
    int warp_m0 = is_producer ? 0 : (consumer_warp/WARPS_N)*WT_M;
    int warp_n0 = is_producer ? 0 : (consumer_warp%WARPS_N)*WT_N;

    auto compute_stage = [&](int stage) -> void {
      if (is_producer) return;
      auto *a_buf = a_smem + stage*A_SIZE;
      auto *b_buf = b_smem + stage*B_SIZE;
      #pragma unroll
      for (int kk = 0; kk < BK; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, T, a_layout> a_frag[WT_M];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, b_layout> b_frag[WT_N];
        #pragma unroll
        for (int i=0; i < WT_M; ++i) {
          int mt = (warp_m0 + i)<<4;
          if constexpr (!TA) wmma::load_matrix_sync(a_frag[i], a_buf + mt*BK + kk, BK);
          else wmma::load_matrix_sync(a_frag[i], a_buf + kk*BM + mt, BM);
        }
        #pragma unroll
        for (int j=0; j < WT_N; ++j) {
          int nt = (warp_n0 + j)<<4;
          if constexpr (!TB) wmma::load_matrix_sync(b_frag[j], b_buf + kk*BN + nt, BN);
          else wmma::load_matrix_sync(b_frag[j], b_buf + nt*BK + kk, BK);
        }
        #pragma unroll
        for (int i=0; i < WT_M; ++i)
          #pragma unroll
          for (int j=0; j < WT_N; ++j)
            wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
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

    /* Every issued TMA has been waited on by the consumers, so the staging buffer is dead
       from here on and can back the f32 epilogue instead of a second dedicated allocation. */
    __syncthreads();
    auto *c_smem = reinterpret_cast<float *>(smem_raw);

    if (!is_producer) {
      auto *c_ptr = c_smem + (consumer_warp<<8);
      #pragma unroll
      for (int i=0; i < WT_M; ++i) {
        #pragma unroll
        for (int j=0; j < WT_N; ++j) {
          wmma::store_matrix_sync(c_ptr, c_frag[i][j], 16, wmma::mem_row_major);
          __syncwarp();
          store_tile_16x16<T>(
            r_batch,
            static_cast<int>(M),
            static_cast<int>(N),
            tile_m + ((warp_m0 + i)<<4),
            tile_n + ((warp_n0 + j)<<4),
            c_ptr,
            lane
          );
          __syncwarp();
        }
      }
    }
  }

  template <typename T>
  static mag_status_t launch_matmul_kernel_wmma(
    mag_error_t *err,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_total,
    T *__restrict__ br,
    mag_tensor_t *x, mag_tensor_t *y,
    bool xT, bool yT,
    cudaStream_t stream
  ) {
    static_assert(std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>);
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 32;
    static constexpr int WT_M = 4;  /* 4x2 accumulators per warp: 8 mma per 6 fragment loads */
    static constexpr int WT_N = 2;
    static constexpr int STAGES = 3;
    static constexpr int CONSUMER_WARPS = ((BM>>4)/WT_M)*((BN>>4)/WT_N);
    static constexpr int BLOCK_THREADS = (1 + CONSUMER_WARPS)*32;
    int max_smem_real;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    /* The epilogue aliases the A/B staging buffer, so the block only pays for the larger of the two. */
    size_t smem = std::max(sizeof(T)*STAGES*(BM*BK + BN*BK), sizeof(float)*(CONSUMER_WARPS<<8));
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
      auto *kernel = matmul_kernel_wmma<T, false, false, BM, BN, BK, WT_M, WT_N, STAGES>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem, stream>>>(M, N, K, batch_total, br, map_a, map_b);
    } else if (!xT && yT) {
      CUtensorMap map_a = init_tmap_x<T, false, BM, BK>(x);
      CUtensorMap map_b = init_tmap_y<T, true, BK, BN>(y);
      auto *kernel = matmul_kernel_wmma<T, false, true, BM, BN, BK, WT_M, WT_N, STAGES>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem, stream>>>(M, N, K, batch_total, br, map_a, map_b);
    } else if (xT && !yT) {
      CUtensorMap map_a = init_tmap_x<T, true, BM, BK>(x);
      CUtensorMap map_b = init_tmap_y<T, false, BK, BN>(y);
      auto *kernel = matmul_kernel_wmma<T, true, false, BM, BN, BK, WT_M, WT_N, STAGES>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem, stream>>>(M, N, K, batch_total, br, map_a, map_b);
    } else {
      CUtensorMap map_a = init_tmap_x<T, true, BM, BK>(x);
      CUtensorMap map_b = init_tmap_y<T, true, BK, BN>(y);
      auto *kernel = matmul_kernel_wmma<T, true, true, BM, BN, BK, WT_M, WT_N, STAGES>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem, stream>>>(M, N, K, batch_total, br, map_a, map_b);
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
    bool xT, bool yT,
    cudaStream_t stream
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
      kernel<<<grid_dim, block_dim, smem, stream>>>(M, N, K, batch_total, br, bx, by);
    } else if (!xT && yT) {
      auto *kernel = matmul_kernel_fallback<T, false, true, BM, BN, BK, TM, TN>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem, stream>>>(M, N, K, batch_total, br, bx, by);
    } else if (xT && !yT) {
      auto *kernel = matmul_kernel_fallback<T, true, false, BM, BN, BK, TM, TN>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem, stream>>>(M, N, K, batch_total, br, bx, by);
    } else {
      auto *kernel = matmul_kernel_fallback<T, true, true, BM, BN, BK, TM, TN>;
      set_kernel_smem_size(kernel, smem);
      kernel<<<grid_dim, block_dim, smem, stream>>>(M, N, K, batch_total, br, bx, by);
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



  static constexpr int64_t GEMV_MAX_THIN_M = 8;

  template <typename T>
  [[nodiscard]] static __device__ __forceinline__ float dot_vec16(const uint4 &a, const uint4 &b) {
    float s0 = 0.0f;
    float s1 = 0.0f;
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      const auto *p = reinterpret_cast<const __nv_bfloat162 *>(&a);
      const auto *q = reinterpret_cast<const __nv_bfloat162 *>(&b);
      #pragma unroll
      for (int i=0; i < 4; ++i) {
        float2 u = __bfloat1622float2(p[i]);
        float2 v = __bfloat1622float2(q[i]);
        s0 = __fmaf_rn(u.x, v.x, s0);
        s1 = __fmaf_rn(u.y, v.y, s1);
      }
    } else if constexpr (std::is_same_v<T, half>) {
      const auto *p = reinterpret_cast<const half2 *>(&a);
      const auto *q = reinterpret_cast<const half2 *>(&b);
      #pragma unroll
      for (int i=0; i < 4; ++i) {
        float2 u = __half22float2(p[i]);
        float2 v = __half22float2(q[i]);
        s0 = __fmaf_rn(u.x, v.x, s0);
        s1 = __fmaf_rn(u.y, v.y, s1);
      }
    } else {
      const auto *p = reinterpret_cast<const float *>(&a);
      const auto *q = reinterpret_cast<const float *>(&b);
      #pragma unroll
      for (int i=0; i < 4; i += 2) {
        s0 = __fmaf_rn(p[i], q[i], s0);
        s1 = __fmaf_rn(p[i+1], q[i+1], s1);
      }
    }
    return s0 + s1;
  }

  template <typename T>
  static __device__ __forceinline__ void unpack_vec16(const uint4 &v, float *o) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
      const auto *p = reinterpret_cast<const __nv_bfloat162 *>(&v);
      #pragma unroll
      for (int i=0; i < 4; ++i) {
        float2 u = __bfloat1622float2(p[i]);
        o[i<<1] = u.x;
        o[(i<<1)+1] = u.y;
      }
    } else if constexpr (std::is_same_v<T, half>) {
      const auto *p = reinterpret_cast<const half2 *>(&v);
      #pragma unroll
      for (int i=0; i < 4; ++i) {
        float2 u = __half22float2(p[i]);
        o[i<<1] = u.x;
        o[(i<<1)+1] = u.y;
      }
    } else {
      const auto *p = reinterpret_cast<const float *>(&v);
      #pragma unroll
      for (int i=0; i < 4; ++i) o[i] = p[i];
    }
  }

  /* One warp owns ROWS consecutive weight rows: 32 lanes sweep K in 16-byte strides, so a
     warp-step pulls ROWS*512 contiguous bytes with ROWS independent loads in flight, and the
     x fragment is fetched once and reused across all ROWS*MTILE dot products. The tail rows
     are clamped instead of predicated - they read live memory and are dropped at store time,
     which keeps the inner loop branch free. */
  template <typename T, int MTILE, int WARPS, int ROWS, bool VEC>
  static __global__ void __launch_bounds__(WARPS*32) gemv_wt_kernel(
    int M, int N, int K, int batch_total,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw
  ) {
    static constexpr int E = 16/sizeof(T);
    int batch = blockIdx.y;
    if (batch >= batch_total) return;
    bx += static_cast<int64_t>(batch)*M*K;
    bw += static_cast<int64_t>(batch)*N*K;
    br += static_cast<int64_t>(batch)*M*N;
    int lane = threadIdx.x&31;
    int row0 = (blockIdx.x*WARPS + static_cast<int>(threadIdx.x>>5))*ROWS;
    if (row0 >= N) return;
    const T *wr[ROWS];
    #pragma unroll
    for (int r=0; r < ROWS; ++r)
      wr[r] = bw + static_cast<int64_t>(::min(row0 + r, N-1))*K;
    const T *xr[MTILE];
    #pragma unroll
    for (int m=0; m < MTILE; ++m)
      xr[m] = bx + static_cast<int64_t>(::min(m, M-1))*K;
    float acc[MTILE][ROWS] = {};
    if constexpr (VEC) {
      int kv = K/E;
      #pragma unroll 2
      for (int i=lane; i < kv; i += 32) {
        uint4 xv[MTILE];
        #pragma unroll
        for (int m=0; m < MTILE; ++m)
          xv[m] = __ldg(reinterpret_cast<const uint4 *>(xr[m]) + i);
        uint4 wv[ROWS];
        #pragma unroll
        for (int r=0; r < ROWS; ++r)
          wv[r] = __ldcs(reinterpret_cast<const uint4 *>(wr[r]) + i); /* streamed once, keep it out of L1 */
        #pragma unroll
        for (int m=0; m < MTILE; ++m)
          #pragma unroll
          for (int r=0; r < ROWS; ++r)
            acc[m][r] += dot_vec16<T>(xv[m], wv[r]);
      }
    } else {
      for (int k=lane; k < K; k += 32) {
        float xv[MTILE];
        #pragma unroll
        for (int m=0; m < MTILE; ++m) xv[m] = static_cast<float>(xr[m][k]);
        #pragma unroll
        for (int r=0; r < ROWS; ++r) {
          float w = static_cast<float>(wr[r][k]);
          #pragma unroll
          for (int m=0; m < MTILE; ++m) acc[m][r] = __fmaf_rn(xv[m], w, acc[m][r]);
        }
      }
    }
    #pragma unroll
    for (int m=0; m < MTILE; ++m) {
      #pragma unroll
      for (int r=0; r < ROWS; ++r) {
        float s = acc[m][r];
        #pragma unroll
        for (int off=16; off > 0; off >>= 1)
          s += __shfl_down_sync(0xffffffff, s, off);
        acc[m][r] = s;
      }
    }
    if (lane) return;
    #pragma unroll
    for (int m=0; m < MTILE; ++m) {
      if (m >= M) break;
      #pragma unroll
      for (int r=0; r < ROWS; ++r)
        if (row0 + r < N) br[static_cast<int64_t>(m)*N + row0 + r] = static_cast<T>(acc[m][r]);
    }
  }

  template <typename T, int MTILE, int BLOCK, bool VEC>
  static __global__ void __launch_bounds__(BLOCK) gemv_wn_kernel(
    int M, int N, int K, int batch_total,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw
  ) {
    static constexpr int CPT = VEC ? 16/sizeof(T) : 1;
    int batch = blockIdx.y;
    if (batch >= batch_total) return;
    bx += static_cast<int64_t>(batch)*M*K;
    bw += static_cast<int64_t>(batch)*K*N;
    br += static_cast<int64_t>(batch)*M*N;
    int col0 = (blockIdx.x*BLOCK + static_cast<int>(threadIdx.x))*CPT;
    if (col0 >= N) return;
    const T *xr[MTILE];
    #pragma unroll
    for (int m=0; m < MTILE; ++m)
      xr[m] = bx + static_cast<int64_t>(::min(m, M-1))*K;
    float acc[MTILE][CPT] = {};
    bool vectorized = VEC && col0 + CPT <= N;
    if constexpr (VEC) {
      if (vectorized) {
        #pragma unroll 4
        for (int k=0; k < K; ++k) {
          uint4 wv = __ldcs(reinterpret_cast<const uint4 *>(bw + static_cast<int64_t>(k)*N + col0));
          float w[CPT];
          unpack_vec16<T>(wv, w);
          #pragma unroll
          for (int m=0; m < MTILE; ++m) {
            float xv = static_cast<float>(__ldg(xr[m] + k));
            #pragma unroll
            for (int e=0; e < CPT; ++e) acc[m][e] = __fmaf_rn(xv, w[e], acc[m][e]);
          }
        }
      }
    }
    if (!vectorized) {
      for (int k=0; k < K; ++k) {
        float w[CPT];
        #pragma unroll
        for (int e=0; e < CPT; ++e)
          w[e] = col0 + e < N ? static_cast<float>(bw[static_cast<int64_t>(k)*N + col0 + e]) : 0.0f;
        #pragma unroll
        for (int m=0; m < MTILE; ++m) {
          float xv = static_cast<float>(__ldg(xr[m] + k));
          #pragma unroll
          for (int e=0; e < CPT; ++e) acc[m][e] = __fmaf_rn(xv, w[e], acc[m][e]);
        }
      }
    }
    #pragma unroll
    for (int m=0; m < MTILE; ++m) {
      if (m >= M) break;
      #pragma unroll
      for (int e=0; e < CPT; ++e)
        if (col0 + e < N) br[static_cast<int64_t>(m)*N + col0 + e] = static_cast<T>(acc[m][e]);
    }
  }

  template <typename T, int MTILE, int ROWS>
  static void launch_gemv_wt(
    int M, int N, int K, int64_t batch_total,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw,
    bool vec, cudaStream_t stream
  ) {
    static constexpr int WARPS = 4;
    dim3 block_dim(WARPS<<5, 1, 1);
    dim3 grid_dim(static_cast<unsigned>((N + WARPS*ROWS - 1)/(WARPS*ROWS)), static_cast<unsigned>(batch_total), 1);
    if (vec) gemv_wt_kernel<T, MTILE, WARPS, ROWS, true><<<grid_dim, block_dim, 0, stream>>>(M, N, K, batch_total, br, bx, bw);
    else gemv_wt_kernel<T, MTILE, WARPS, ROWS, false><<<grid_dim, block_dim, 0, stream>>>(M, N, K, batch_total, br, bx, bw);
  }

  template <typename T, int MTILE>
  static void launch_gemv_mtile(
    int64_t M, int64_t N, int64_t K, int64_t batch_total,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw,
    bool wT, cudaStream_t stream
  ) {
    static constexpr int64_t ALIGN = 15;
    bool base_aligned = !(ALIGN & reinterpret_cast<uintptr_t>(bx)) && !(ALIGN & reinterpret_cast<uintptr_t>(bw));
    int Mi = static_cast<int>(M);
    int Ni = static_cast<int>(N);
    int Ki = static_cast<int>(K);
    if (wT) {
      bool vec = base_aligned && !(ALIGN & (K*static_cast<int64_t>(sizeof(T))));
      if (N >= 4096) launch_gemv_wt<T, MTILE, 4>(Mi, Ni, Ki, batch_total, br, bx, bw, vec, stream);
      else if (N >= 1024) launch_gemv_wt<T, MTILE, 2>(Mi, Ni, Ki, batch_total, br, bx, bw, vec, stream);
      else launch_gemv_wt<T, MTILE, 1>(Mi, Ni, Ki, batch_total, br, bx, bw, vec, stream);
    } else {
      static constexpr int BLOCK = 128;
      static constexpr int E = 16/sizeof(T);
      bool vec = base_aligned && !(ALIGN & (N*static_cast<int64_t>(sizeof(T))));
      int cpt = vec ? E : 1;
      dim3 block_dim(BLOCK, 1, 1);
      dim3 grid_dim(static_cast<unsigned>((N + BLOCK*cpt - 1)/(BLOCK*cpt)), static_cast<unsigned>(batch_total), 1);
      if (vec) gemv_wn_kernel<T, MTILE, BLOCK, true><<<grid_dim, block_dim, 0, stream>>>(Mi, Ni, Ki, batch_total, br, bx, bw);
      else gemv_wn_kernel<T, MTILE, BLOCK, false><<<grid_dim, block_dim, 0, stream>>>(Mi, Ni, Ki, batch_total, br, bx, bw);
    }
  }

  template <typename T>
  static void launch_gemv(
    int64_t M, int64_t N, int64_t K, int64_t batch_total,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw,
    bool wT, cudaStream_t stream
  ) {
    if (N == 1) wT = true; /* a single output column is K-contiguous under either layout */
    if (M <= 1) launch_gemv_mtile<T, 1>(M, N, K, batch_total, br, bx, bw, wT, stream);
    else if (M <= 2) launch_gemv_mtile<T, 2>(M, N, K, batch_total, br, bx, bw, wT, stream);
    else if (M <= 4) launch_gemv_mtile<T, 4>(M, N, K, batch_total, br, bx, bw, wT, stream);
    else launch_gemv_mtile<T, 8>(M, N, K, batch_total, br, bx, bw, wT, stream);
  }

  template <typename T>
  static mag_status_t launch_matmul(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
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
        launch_gemv(1, N, K, batch_total, br, bx, by, yT, stream);
        goto end;
      case MAG_MATMUL_TYPE_GEMV_MAT_VEC:
      case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC:

        launch_gemv(1, M, K, batch_total, br, by, bx, !xT, stream);
        goto end;
      default: break;
    }
    if (N == 1) {
      launch_gemv(1, M, K, batch_total, br, by, bx, !xT, stream);
      goto end;
    }
    if (M <= GEMV_MAX_THIN_M && !xT) {
      launch_gemv(M, N, K, batch_total, br, bx, by, yT, stream);
      goto end;
    }
    #if MAG_CUDA_MATMUL_USE_WMMA
      if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>) {
        if (is_tensor_tma_compat_x<T>(x, xT) && is_tensor_tma_compat_y<T>(y, yT)) {
          st = launch_matmul_kernel_wmma(err, M, N, K, batch_total, br, x, y, xT, yT, stream);
          goto end;
        }
      }
    #endif
    st = launch_matmul_kernel_fallback(err, M, N, K, batch_total, br, bx, by, xT, yT, stream);
    [[maybe_unused]] end:
      if (cloned_x) mag_tensor_decref(x);
      if (cloned_y) mag_tensor_decref(y);
    return st;
  }

  mag_status_t misc_op_matmul(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    const mag_tensor_t *x = cmd.in[0];
    switch (x->meta.dtype) {
      case MAG_DTYPE_FLOAT32: return launch_matmul<float>(err, cmd, stream);
      case MAG_DTYPE_FLOAT16: return launch_matmul<half>(err, cmd, stream);
      case MAG_DTYPE_BFLOAT16: return launch_matmul<__nv_bfloat16>(err, cmd, stream);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: matmul: unsupported dtype %s.", mag_type_trait(x->meta.dtype)->name);
    }
  }
}
