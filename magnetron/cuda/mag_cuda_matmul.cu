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
#include "mag_cuda_gemm.cuh"
#include "mag_cuda_tma.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <numeric>

namespace mag {
  struct gemm_kargs final {
    int M, N, K;
    int batch;
    int k_batch;
    int a_bcast, b_bcast;
    int64_t lda, ldb, ldc;
    int64_t a_bs, b_bs, c_bs;
    int64_t a_kbs, b_kbs;
    const void *a;
    const void *b;
    void *c;
    const void *bias;
  };

  [[nodiscard]] static gemm_kargs make_kargs(const gemm_desc &d) noexcept {
    gemm_kargs k {};
    k.M = static_cast<int>(d.M);
    k.N = static_cast<int>(d.N);
    k.K = static_cast<int>(d.K);
    k.batch = static_cast<int>(d.batch);
    k.k_batch = static_cast<int>(d.k_batch);
    k.a_bcast = d.a_bstride == 0;
    k.b_bcast = d.b_bstride == 0;
    k.lda = d.lda;
    k.ldb = d.ldb;
    k.ldc = d.ldc;
    k.a_bs = d.a_bstride;
    k.b_bs = d.b_bstride;
    k.c_bs = d.c_bstride;
    k.a_kbs = d.a_kbstride;
    k.b_kbs = d.b_kbstride;
    k.a = d.a;
    k.b = d.b;
    k.c = d.c;
    k.bias = d.bias;
    return k;
  }

  [[nodiscard]] static int64_t tensor_batch_total(const mag_tensor_t *tensor) noexcept {
    int64_t ra = tensor->meta.coords.rank;
    if (ra <= 2) return 1;
    int64_t batch = 1;
    for (int64_t i=0; i < ra-2; ++i)
      batch *= tensor->meta.coords.shape[i];
    return batch;
  }

  bool tcgen05_disabled() noexcept {
    static const bool disabled = [] {
      const char *v = std::getenv("MAG_CUDA_DISABLE_TCGEN05");
      return v && *v && !(v[0] == '0' && !v[1]);
    }();
    return disabled;
  }

  template <typename T>
  static __device__ __forceinline__ void store_f32x2(T *o, float x, float y);

  template <>
  __device__ __forceinline__ void store_f32x2<half>(half *o, float x, float y) {
    *reinterpret_cast<half2 *>(o) = __floats2half2_rn(x, y);
  }

  template <>
  __device__ __forceinline__ void store_f32x2<__nv_bfloat16>(__nv_bfloat16 *o, float x, float y) {
    *reinterpret_cast<__nv_bfloat162 *>(o) = __floats2bfloat162_rn(x, y);
  }

  /* One warp writes a 16x16 fp32 accumulator tile (row-major in shared memory) to C. */
  template <typename T>
  static __device__ __forceinline__ void store_tile_16x16(
    T *__restrict__ c,
    const T *__restrict__ bias,
    int M, int N, int64_t ldc,
    int base_row, int base_col,
    const float *__restrict__ c_ptr,
    int lane
  ) {
    #pragma unroll
    for (int i=lane<<1; i < 256; i += 64) {
      int row = i>>4;
      int col = i&15;
      int g_row = base_row + row;
      int g_col = base_col + col;
      if (g_row >= M || g_col >= N) continue;
      float b = bias ? static_cast<float>(bias[g_row]) : 0.0f;
      T *dst = c + static_cast<int64_t>(g_row)*ldc + g_col;
      float v0 = c_ptr[i] + b;
      float v1 = c_ptr[i+1] + b;
      if (g_col+1 < N && !(3 & reinterpret_cast<uintptr_t>(dst))) {
        store_f32x2<T>(dst, v0, v1);
      } else {
        dst[0] = static_cast<T>(v0);
        if (g_col+1 < N) dst[1] = static_cast<T>(v1);
      }
    }
  }

  /* ---- TMA + WMMA kernel (sm_90+) ------------------------------------------------------------ */

  struct barrier final {
    uint64_t bar;

    __device__ void init(const uint32_t &count) {
      asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(this))), "r"(count) : "memory");
    }

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

  /* TA: A stored [K][M] (MN-major), TB: B stored [N][K] (K-major). Grid (n_tiles, m_tiles, batch). */
  template <typename T, bool TA, bool TB, int BM, int BN, int BK, int WT_M, int WT_N, int STAGES>
  __global__ static void matmul_kernel_wmma(
    const __grid_constant__ gemm_kargs args,
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
    if (batch >= args.batch) return;
    const int M = args.M, N = args.N, K = args.K;

    int tile_m = blockIdx.y*BM;
    int tile_n = blockIdx.x*BN;
    int tid = threadIdx.x;
    int lane = tid&31;
    int warp_id = tid>>5;
    bool is_producer = warp_id == 0;
    int consumer_warp = warp_id-1;

    T *__restrict__ c_batch = static_cast<T *>(args.c) + static_cast<int64_t>(batch)*args.c_bs;
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

    const int k_tiles = (K + BK - 1)/BK;
    const int k_iters = k_tiles*args.k_batch;

    auto init_tma_coords = [=](int it, int32_t (&ca)[3], int32_t (&cb)[3]) -> void {
      int kb = it / k_tiles;
      int k0 = (it - kb*k_tiles)*BK;
      int zsel = args.k_batch > 1 ? kb : batch;
      int za = args.a_bcast ? 0 : zsel;
      int zb = args.b_bcast ? 0 : zsel;
      if constexpr (!TA) { ca[0] = k0; ca[1] = tile_m; ca[2] = za; }  /* dims {K, M, batch}, box {BK, BM, 1} */
      else { ca[0] = tile_m; ca[1] = k0; ca[2] = za; }                 /* dims {M, K, batch}, box {BM, BK, 1} */
      if constexpr (!TB) { cb[0] = tile_n; cb[1] = k0; cb[2] = zb; }  /* dims {N, K, batch}, box {BN, BK, 1} */
      else { cb[0] = k0; cb[1] = tile_n; cb[2] = zb; }                 /* dims {K, N, batch}, box {BK, BN, 1} */
    };

    auto issue_tma_stage = [&](int stage, int it) -> void {
      if (!is_producer || lane != 0) return;
      auto *a_buf = a_smem + stage*A_SIZE;
      auto *b_buf = b_smem + stage*B_SIZE;
      int32_t a_coords[3];
      int32_t b_coords[3];
      init_tma_coords(it, a_coords, b_coords);
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

    int prefetch = k_iters < STAGES ? k_iters : STAGES;
    if (is_producer && lane == 0) {
      #pragma unroll
      for (int s=0; s < STAGES; ++s) {
        if (s < prefetch) issue_tma_stage(s, s);
      }
    }
    for (int it=0; it < k_iters; ++it) {
      int stage = it % STAGES;
      int phase = (it / STAGES) & 1;
      int next_it = it + STAGES;
      if (!is_producer) {
        wait_stage_ready(stage, phase);
        compute_stage(stage);
        __syncwarp();
        consumer_mark_stage_done(stage);
      }
      if (is_producer && lane == 0 && next_it < k_iters) {
        producer_wait_stage_reusable(stage, phase);
        issue_tma_stage(stage, next_it);
      }
    }
    __syncthreads();
    auto *c_smem = reinterpret_cast<float *>(smem_raw);
    if (!is_producer) {
      auto *c_ptr = c_smem + (consumer_warp<<8);
      const auto *bias = static_cast<const T *>(args.bias);
      #pragma unroll
      for (int i=0; i < WT_M; ++i) {
        #pragma unroll
        for (int j=0; j < WT_N; ++j) {
          wmma::store_matrix_sync(c_ptr, c_frag[i][j], 16, wmma::mem_row_major);
          __syncwarp();
          store_tile_16x16<T>(c_batch, bias, M, N, args.ldc, tile_m + ((warp_m0 + i)<<4), tile_n + ((warp_n0 + j)<<4), c_ptr, lane);
          __syncwarp();
        }
      }
    }
  }

  template <typename T>
  static mag_status_t launch_matmul_kernel_wmma(mag_error_t *err, const gemm_desc &d, cudaStream_t stream) {
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

    const bool kb = d.k_batch > 1;
    const int64_t nb = kb ? d.k_batch : d.batch;
    const int64_t a_bs = kb ? d.a_kbstride : d.a_bstride;
    const int64_t b_bs = kb ? d.b_kbstride : d.b_bstride;
    const int64_t a_batch = a_bs ? nb : 1;
    const int64_t b_batch = b_bs ? nb : 1;
    const bool TA = !d.a_kmajor;
    const bool TB = d.b_kmajor;
    CUtensorMap map_a {}, map_b {};
    bool ok_a = !TA
      ? tma::encode_operand_3d<T>(map_a, d.a, d.M, d.K, d.lda, a_batch, a_bs, BK, BM, CU_TENSOR_MAP_SWIZZLE_NONE)
      : tma::encode_operand_3d<T>(map_a, d.a, d.K, d.M, d.lda, a_batch, a_bs, BM, BK, CU_TENSOR_MAP_SWIZZLE_NONE);
    bool ok_b = !TB
      ? tma::encode_operand_3d<T>(map_b, d.b, d.K, d.N, d.ldb, b_batch, b_bs, BN, BK, CU_TENSOR_MAP_SWIZZLE_NONE)
      : tma::encode_operand_3d<T>(map_b, d.b, d.N, d.K, d.ldb, b_batch, b_bs, BK, BN, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (mag_unlikely(!ok_a || !ok_b))
      return mag_set_error(err, MAG_ERR_BACKEND, "cuda: matmul: failed to encode TMA descriptors.");

    gemm_kargs args = make_kargs(d);
    args.a_bcast = a_batch == 1;
    args.b_bcast = b_batch == 1;
    dim3 grid_dim(static_cast<unsigned>((d.N + BN-1)/BN), static_cast<unsigned>((d.M + BM-1)/BM), static_cast<unsigned>(d.batch));
    dim3 block_dim(BLOCK_THREADS, 1, 1);
    auto launch = [&](auto *kernel) -> void {
      mag_assert2(smem <= INT32_MAX);
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
      kernel<<<grid_dim, block_dim, smem, stream>>>(args, map_a, map_b);
    };
    if (!TA && !TB) launch(matmul_kernel_wmma<T, false, false, BM, BN, BK, WT_M, WT_N, STAGES>);
    else if (!TA && TB) launch(matmul_kernel_wmma<T, false, true, BM, BN, BK, WT_M, WT_N, STAGES>);
    else if (TA && !TB) launch(matmul_kernel_wmma<T, true, false, BM, BN, BK, WT_M, WT_N, STAGES>);
    else launch(matmul_kernel_wmma<T, true, true, BM, BN, BK, WT_M, WT_N, STAGES>);
    mag_cu_rt_check(err, cudaGetLastError(), "matmul: wmma kernel launch failed");
    return MAG_OK;
  }

  /* ---- Scalar fallback (fp32 and unaligned corner cases) ------------------------------------- */

  template <typename T, bool TA, bool TB, int BM, int BN, int BK, int TM, int TN>
  __global__ static void matmul_kernel_fallback(const __grid_constant__ gemm_kargs args) {
    static constexpr int A_SIZE = BM*BK;
    static constexpr int B_SIZE = BK*BN;
    static constexpr int STAGES = 2;
    extern __shared__ uint8_t smem[];
    auto *a_smem = reinterpret_cast<T *>(smem);
    auto *b_smem = reinterpret_cast<T *>(smem) + STAGES*A_SIZE;
    int batch = blockIdx.z;
    if (batch >= args.batch) return;
    const int M = args.M, N = args.N, K = args.K;
    const auto *bx = static_cast<const T *>(args.a) + static_cast<int64_t>(batch)*args.a_bs;
    const auto *by = static_cast<const T *>(args.b) + static_cast<int64_t>(batch)*args.b_bs;
    auto *br = static_cast<T *>(args.c) + static_cast<int64_t>(batch)*args.c_bs;
    const auto *bias = static_cast<const T *>(args.bias);
    const int64_t a_row_stride = TA ? 1 : args.lda;
    const int64_t a_col_stride = TA ? args.lda : 1;
    const int64_t b_row_stride = TB ? 1 : args.ldb;
    const int64_t b_col_stride = TB ? args.ldb : 1;
    int tile_m = blockIdx.y * BM;
    int tile_n = blockIdx.x * BN;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y*blockDim.x + threadIdx.x;
    int nthreads = blockDim.x*blockDim.y;
    int local_m0 = ty * TM;
    int local_n0 = tx * TN;
    float acc[TM][TN] = {};
    auto load_stage = [&](int stage, int k0, const T *ax, const T *bxx) {
      auto *a_buf = a_smem + stage*A_SIZE;
      auto *b_buf = b_smem + stage*B_SIZE;
      #pragma unroll
      for (int i=tid; i < A_SIZE; i += nthreads) {
        int row = i / BK;
        int col = i % BK;
        int g_row = tile_m + row;
        int g_col = k0 + col;
        a_buf[i] = g_row < M && g_col < K ? ax[g_row*a_row_stride + g_col*a_col_stride] : T{};
      }
      #pragma unroll
      for (int i=tid; i < B_SIZE; i += nthreads) {
        int row = i / BN;
        int col = i % BN;
        int g_row = k0 + row;
        int g_col = tile_n + col;
        b_buf[i] = g_row < K && g_col < N ? bxx[g_row*b_row_stride + g_col*b_col_stride] : T{};
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
        for (int i=0; i < TM; ++i) a_frag[i] = static_cast<float>(a_buf[(local_m0 + i)*BK + kk]);
        #pragma unroll
        for (int i=0; i < TN; ++i) b_frag[i] = static_cast<float>(b_buf[kk*BN + (local_n0 + i)]);
        #pragma unroll
        for (int i=0; i < TM; ++i)
          #pragma unroll
          for (int j=0; j < TN; ++j)
            acc[i][j] += a_frag[i] * b_frag[j];
      }
    };

    for (int kb=0; kb < args.k_batch; ++kb) {
      const T *ax = bx + static_cast<int64_t>(kb)*args.a_kbs;
      const T *bxx = by + static_cast<int64_t>(kb)*args.b_kbs;
      int k0 = 0;
      int stage = 0;
      load_stage(stage, k0, ax, bxx);
      __syncthreads();
      for (; k0 < K; k0 += BK) {
        int next_k0 = k0 + BK;
        int next_stage = stage^1;
        if (next_k0 < K)
          load_stage(next_stage, next_k0, ax, bxx);
        compute_stage(stage);
        __syncthreads();
        stage = next_stage;
      }
    }

    #pragma unroll
    for (int i=0; i < TM; ++i) {
      int g_row = tile_m + local_m0 + i;
      if (g_row >= M) continue;
      float b = bias ? static_cast<float>(bias[g_row]) : 0.0f;
      #pragma unroll
      for (int j=0; j < TN; ++j) {
        int g_col = tile_n + local_n0 + j;
        if (g_col >= N) continue;
        br[static_cast<int64_t>(g_row)*args.ldc + g_col] = static_cast<T>(acc[i][j] + b);
      }
    }
  }

  /* ---- WMMA kernel with plain vectorized global loads (any alignment) ----------------------- */

  template <typename T, bool TA, bool TB, int BM, int BN, int BK, int WARPS_M, int WARPS_N>
  __global__ static void matmul_kernel_fallback_mma(const __grid_constant__ gemm_kargs args) {
    using namespace nvcuda;
    static_assert(std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>);
    static_assert(BM % (16*WARPS_M) == 0 && BN % (16*WARPS_N) == 0 && BK % 16 == 0);
    static constexpr int WT_M = BM/(16*WARPS_M);
    static constexpr int WT_N = BN/(16*WARPS_N);
    static constexpr int NTHREADS = WARPS_M*WARPS_N*32;
    static constexpr int A_SIZE = BM*BK;
    static constexpr int B_SIZE = BK*BN;
    static constexpr int STAGES = 2;
    using a_layout = std::conditional_t<TA, wmma::col_major, wmma::row_major>;
    using b_layout = std::conditional_t<TB, wmma::col_major, wmma::row_major>;
    extern __shared__ __align__(128) uint8_t smem_raw[];
    auto *a_smem = reinterpret_cast<T *>(smem_raw);
    auto *b_smem = a_smem + STAGES*A_SIZE;
    int batch = blockIdx.z;
    if (batch >= args.batch) return;
    const int M = args.M, N = args.N, K = args.K;
    const auto *bx = static_cast<const T *>(args.a) + static_cast<int64_t>(batch)*args.a_bs;
    const auto *by = static_cast<const T *>(args.b) + static_cast<int64_t>(batch)*args.b_bs;
    T *c_batch = static_cast<T *>(args.c) + static_cast<int64_t>(batch)*args.c_bs;
    int tile_m = blockIdx.y*BM;
    int tile_n = blockIdx.x*BN;
    int tid = threadIdx.x;
    int lane = tid&31;
    int warp = tid>>5;
    int warp_m0 = (warp / WARPS_N)*WT_M;
    int warp_n0 = (warp % WARPS_N)*WT_N;
    const int64_t a_pitch = args.lda;
    const int a_rows_total = TA ? K : M;
    const int a_cols_total = TA ? M : K;
    const int64_t b_pitch = args.ldb;
    const int b_rows_total = TB ? N : K;
    const int b_cols_total = TB ? K : N;
    auto vec_width = [](int64_t pitch, const void *base) -> int {
      if (!(pitch & 7) && !(reinterpret_cast<uintptr_t>(base) & 15)) return 8;
      if (!(pitch & 1) && !(reinterpret_cast<uintptr_t>(base) & 3)) return 2;
      return 1;
    };
    auto load_tile = [&]<int ROWS, int COLS, int V>(T *dst, const T *src, int64_t pitch, int row0, int col0, int rows_total, int cols_total) {
      static_assert(COLS % V == 0 && (ROWS*COLS/V) % NTHREADS == 0);
      using vec_t = std::conditional_t<V == 8, uint4, std::conditional_t<V == 2, uint32_t, T>>;
      static constexpr int PER_THREAD = ROWS*COLS/V/NTHREADS;
      static constexpr int VEC_COLS = COLS/V;
      vec_t v[PER_THREAD];
      #pragma unroll
      for (int t=0; t < PER_THREAD; ++t) {
        int i = tid + t*NTHREADS;
        int row = i / VEC_COLS;
        int col = (i % VEC_COLS)*V;
        int grow = row0 + row;
        int gcol = col0 + col;
        v[t] = vec_t {};
        if (grow < rows_total && gcol < cols_total)
          v[t] = *reinterpret_cast<const vec_t *>(src + static_cast<int64_t>(grow)*pitch + gcol);
      }
      #pragma unroll
      for (int t=0; t < PER_THREAD; ++t) {
        int i = tid + t*NTHREADS;
        int row = i / VEC_COLS;
        int col = (i % VEC_COLS)*V;
        *reinterpret_cast<vec_t *>(dst + row*COLS + col) = v[t];
      }
    };
    auto load_tile_any = [&]<int ROWS, int COLS>(T *dst, const T *src, int64_t pitch, int row0, int col0, int rows_total, int cols_total, int vec) {
      if (vec == 8) load_tile.template operator()<ROWS, COLS, 8>(dst, src, pitch, row0, col0, rows_total, cols_total);
      else if (vec == 2) load_tile.template operator()<ROWS, COLS, 2>(dst, src, pitch, row0, col0, rows_total, cols_total);
      else load_tile.template operator()<ROWS, COLS, 1>(dst, src, pitch, row0, col0, rows_total, cols_total);
    };

    auto load_stage = [&](int stage, int k0, const T *ax, const T *bxx, int a_vec, int b_vec) {
      T *a_buf = a_smem + stage*A_SIZE;
      T *b_buf = b_smem + stage*B_SIZE;
      if constexpr (!TA) load_tile_any.template operator()<BM, BK>(a_buf, ax, a_pitch, tile_m, k0, a_rows_total, a_cols_total, a_vec);
      else load_tile_any.template operator()<BK, BM>(a_buf, ax, a_pitch, k0, tile_m, a_rows_total, a_cols_total, a_vec);
      if constexpr (!TB) load_tile_any.template operator()<BK, BN>(b_buf, bxx, b_pitch, k0, tile_n, b_rows_total, b_cols_total, b_vec);
      else load_tile_any.template operator()<BN, BK>(b_buf, bxx, b_pitch, tile_n, k0, b_rows_total, b_cols_total, b_vec);
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[WT_M][WT_N];
    #pragma unroll
    for (int i=0; i < WT_M; ++i) {
      #pragma unroll
      for (int j=0; j < WT_N; ++j)
        wmma::fill_fragment(c_frag[i][j], 0.0f);
    }

    auto compute_stage = [&](int stage) {
      const T *a_buf = a_smem + stage*A_SIZE;
      const T *b_buf = b_smem + stage*B_SIZE;
      #pragma unroll
      for (int kk=0; kk < BK; kk += 16) {
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
        for (int i=0; i < WT_M; ++i) {
          #pragma unroll
          for (int j=0; j < WT_N; ++j)
            wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
        }
      }
    };

    for (int kb=0; kb < args.k_batch; ++kb) {
      const T *ax = bx + static_cast<int64_t>(kb)*args.a_kbs;
      const T *bxx = by + static_cast<int64_t>(kb)*args.b_kbs;
      const int a_vec = vec_width(a_pitch, ax);
      const int b_vec = vec_width(b_pitch, bxx);
      int stage = 0;
      load_stage(stage, 0, ax, bxx, a_vec, b_vec);
      __syncthreads();
      for (int k0=0; k0 < K; k0 += BK) {
        int next_k0 = k0 + BK;
        int next_stage = stage^1;
        if (next_k0 < K)
          load_stage(next_stage, next_k0, ax, bxx, a_vec, b_vec);
        compute_stage(stage);
        __syncthreads();
        stage = next_stage;
      }
    }
    auto *c_ptr = reinterpret_cast<float *>(smem_raw) + (warp<<8);
    const auto *bias = static_cast<const T *>(args.bias);
    #pragma unroll
    for (int i=0; i < WT_M; ++i) {
      #pragma unroll
      for (int j=0; j < WT_N; ++j) {
        wmma::store_matrix_sync(c_ptr, c_frag[i][j], 16, wmma::mem_row_major);
        __syncwarp();
        store_tile_16x16<T>(c_batch, bias, M, N, args.ldc, tile_m + ((warp_m0 + i)<<4), tile_n + ((warp_n0 + j)<<4), c_ptr, lane);
        __syncwarp();
      }
    }
  }

  template <typename T>
  static mag_status_t launch_matmul_kernel_fallback_mma(mag_error_t *err, const gemm_desc &d, cudaStream_t stream) {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 32;
    static constexpr int WARPS_M = 4;
    static constexpr int WARPS_N = 2;
    static constexpr int STAGES = 2;
    static constexpr int BLOCK_THREADS = WARPS_M*WARPS_N*32;
    dim3 grid_dim(static_cast<unsigned>((d.N + BN-1)/BN), static_cast<unsigned>((d.M + BM-1)/BM), static_cast<unsigned>(d.batch));
    dim3 block_dim(BLOCK_THREADS, 1, 1);
    int max_smem_real;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    size_t smem = std::max(sizeof(T)*STAGES*(BM*BK + BK*BN), sizeof(float)*(WARPS_M*WARPS_N<<8));
    if (smem > (unsigned)max_smem_real)
      return mag_set_error(err, MAG_ERR_OP, "cuda: matmul shared memory requirement (%u bytes) exceeds device limit (%d bytes).", static_cast<unsigned>(smem), max_smem_real);
    gemm_kargs args = make_kargs(d);
    const bool TA = !d.a_kmajor;
    const bool TB = d.b_kmajor;
    auto launch = [&](auto *kernel) -> void {
      mag_assert2(smem <= INT32_MAX);
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
      kernel<<<grid_dim, block_dim, smem, stream>>>(args);
    };
    if (!TA && !TB) launch(matmul_kernel_fallback_mma<T, false, false, BM, BN, BK, WARPS_M, WARPS_N>);
    else if (!TA && TB) launch(matmul_kernel_fallback_mma<T, false, true, BM, BN, BK, WARPS_M, WARPS_N>);
    else if (TA && !TB) launch(matmul_kernel_fallback_mma<T, true, false, BM, BN, BK, WARPS_M, WARPS_N>);
    else launch(matmul_kernel_fallback_mma<T, true, true, BM, BN, BK, WARPS_M, WARPS_N>);
    mag_cu_rt_check(err, cudaGetLastError(), "matmul: wmma fallback kernel launch failed");
    return MAG_OK;
  }

  template <typename T>
  static mag_status_t launch_matmul_kernel_fallback(mag_error_t *err, const gemm_desc &d, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>)
      return launch_matmul_kernel_fallback_mma<T>(err, d, stream);
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 32;
    static constexpr int TM = 4;
    static constexpr int TN = 4;
    static constexpr int STAGES = 2;
    static constexpr int TRX = BN/TN;
    static constexpr int TRY = BM/TM;
    static_assert(TRX*TRY <= 1024);

    dim3 grid_dim(static_cast<unsigned>((d.N + BN-1)/BN), static_cast<unsigned>((d.M + BM-1)/BM), static_cast<unsigned>(d.batch));
    dim3 block_dim(TRX, TRY, 1);

    int max_smem_real;
    int device;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    size_t smem = STAGES * (BM*BK + BN*BK) * sizeof(T);
    if (smem > (unsigned)max_smem_real)
      return mag_set_error(err, MAG_ERR_OP, "cuda: matmul shared memory requirement (%u bytes) exceeds device limit (%d bytes).", static_cast<unsigned>(smem), max_smem_real);
    gemm_kargs args = make_kargs(d);
    const bool TA = !d.a_kmajor;
    const bool TB = d.b_kmajor;
    auto launch = [&](auto *kernel) -> void {
      mag_assert2(smem <= INT32_MAX);
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
      kernel<<<grid_dim, block_dim, smem, stream>>>(args);
    };
    if (!TA && !TB) launch(matmul_kernel_fallback<T, false, false, BM, BN, BK, TM, TN>);
    else if (!TA && TB) launch(matmul_kernel_fallback<T, false, true, BM, BN, BK, TM, TN>);
    else if (TA && !TB) launch(matmul_kernel_fallback<T, true, false, BM, BN, BK, TM, TN>);
    else launch(matmul_kernel_fallback<T, true, true, BM, BN, BK, TM, TN>);
    mag_cu_rt_check(err, cudaGetLastError(), "matmul: fallback kernel launch failed");
    return MAG_OK;
  }

  bool gemm_supported(const gemm_desc &d) noexcept {
    if (d.dtype != MAG_DTYPE_FLOAT32 && d.dtype != MAG_DTYPE_FLOAT16 && d.dtype != MAG_DTYPE_BFLOAT16) return false;
    if (!d.a || !d.b || !d.c) return false;
    if (d.M < 0 || d.N < 0 || d.K < 0 || d.batch < 0 || d.k_batch < 1) return false;
    if (d.M > INT32_MAX || d.N > INT32_MAX || d.K > INT32_MAX || d.batch > INT32_MAX || d.k_batch > INT32_MAX) return false;
    if (d.k_batch > 1 && d.batch != 1) return false;
    if (d.lda < (d.a_kmajor ? d.K : d.M) || d.ldb < (d.b_kmajor ? d.K : d.N) || d.ldc < d.N) return false;
    return true;
  }

  template <typename T>
  [[nodiscard]] static bool tma_compatible(const gemm_desc &d) noexcept {
    auto aligned = [](const void *p) noexcept -> bool { return !(15 & reinterpret_cast<uintptr_t>(p)); };
    auto pitch_ok = [](int64_t n) noexcept -> bool { return !(15 & (n*static_cast<int64_t>(sizeof(T)))); };
    return aligned(d.a) && aligned(d.b) && pitch_ok(d.lda) && pitch_ok(d.ldb)
      && pitch_ok(d.a_bstride) && pitch_ok(d.b_bstride) && pitch_ok(d.a_kbstride) && pitch_ok(d.b_kbstride)
      && d.M <= INT32_MAX && d.N <= INT32_MAX && d.K <= INT32_MAX;
  }

  template <typename T>
  static mag_status_t gemm_dtype(mag_error_t *err, const physical_device &dev, const gemm_desc &d, cudaStream_t stream) {
    if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>) {
#ifdef MAG_HAVE_CUDA_SM_100
      if (!tcgen05_disabled() && dev.has_features(device_features::tcgen05) && sm_100::gemm_supported(d, dev.compute_capability()))
        return sm_100::gemm(err, d, stream, dev.num_sms());
#endif
      if (dev.has_features(device_features::tma) && tma_compatible<T>(d))
        return launch_matmul_kernel_wmma<T>(err, d, stream);
    }
    return launch_matmul_kernel_fallback<T>(err, d, stream);
  }

  mag_status_t gemm(mag_error_t *err, const physical_device &dev, const gemm_desc &d, cudaStream_t stream) {
    if (mag_unlikely(!gemm_supported(d)))
      return mag_set_error(err, MAG_ERR_OP, "cuda: gemm: unsupported problem (dtype %s, M=%lld N=%lld K=%lld batch=%lld k_batch=%lld).",
        mag_type_trait(d.dtype)->name, static_cast<long long>(d.M), static_cast<long long>(d.N), static_cast<long long>(d.K),
        static_cast<long long>(d.batch), static_cast<long long>(d.k_batch));
    if (d.M == 0 || d.N == 0 || d.batch == 0) return MAG_OK;
    switch (d.dtype) {
      case MAG_DTYPE_FLOAT32: return gemm_dtype<float>(err, dev, d, stream);
      case MAG_DTYPE_FLOAT16: return gemm_dtype<half>(err, dev, d, stream);
      case MAG_DTYPE_BFLOAT16: return gemm_dtype<__nv_bfloat16>(err, dev, d, stream);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: gemm: unsupported dtype %s.", mag_type_trait(d.dtype)->name);
    }
  }

  /* ---- GEMV (thin M, memory bound) ------------------------------------------------------------ */

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
    int64_t x_bs, int64_t w_bs,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw
  ) {
    static constexpr int E = 16/sizeof(T);
    int batch = blockIdx.y;
    if (batch >= batch_total) return;
    bx += static_cast<int64_t>(batch)*x_bs;
    bw += static_cast<int64_t>(batch)*w_bs;
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
    int64_t x_bs, int64_t w_bs,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw
  ) {
    static constexpr int CPT = VEC ? 16/sizeof(T) : 1;
    int batch = blockIdx.y;
    if (batch >= batch_total) return;
    bx += static_cast<int64_t>(batch)*x_bs;
    bw += static_cast<int64_t>(batch)*w_bs;
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
    int64_t x_bs, int64_t w_bs,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw,
    bool vec, cudaStream_t stream
  ) {
    static constexpr int WARPS = 4;
    dim3 block_dim(WARPS<<5, 1, 1);
    dim3 grid_dim(static_cast<unsigned>((N + WARPS*ROWS - 1)/(WARPS*ROWS)), static_cast<unsigned>(batch_total), 1);
    if (vec) gemv_wt_kernel<T, MTILE, WARPS, ROWS, true><<<grid_dim, block_dim, 0, stream>>>(M, N, K, batch_total, x_bs, w_bs, br, bx, bw);
    else gemv_wt_kernel<T, MTILE, WARPS, ROWS, false><<<grid_dim, block_dim, 0, stream>>>(M, N, K, batch_total, x_bs, w_bs, br, bx, bw);
  }

  template <typename T, int MTILE>
  static void launch_gemv_mtile(
    int64_t M, int64_t N, int64_t K, int64_t batch_total,
    int64_t x_bs, int64_t w_bs,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw,
    bool wT, cudaStream_t stream
  ) {
    static constexpr int64_t ALIGN = 15;
    bool base_aligned = !(ALIGN & reinterpret_cast<uintptr_t>(bx)) && !(ALIGN & reinterpret_cast<uintptr_t>(bw))
      && !(ALIGN & (x_bs*static_cast<int64_t>(sizeof(T)))) && !(ALIGN & (w_bs*static_cast<int64_t>(sizeof(T))));
    int Mi = static_cast<int>(M);
    int Ni = static_cast<int>(N);
    int Ki = static_cast<int>(K);
    if (wT) {
      bool vec = base_aligned && !(ALIGN & (K*static_cast<int64_t>(sizeof(T))));
      if (N >= 4096) launch_gemv_wt<T, MTILE, 4>(Mi, Ni, Ki, batch_total, x_bs, w_bs, br, bx, bw, vec, stream);
      else if (N >= 1024) launch_gemv_wt<T, MTILE, 2>(Mi, Ni, Ki, batch_total, x_bs, w_bs, br, bx, bw, vec, stream);
      else launch_gemv_wt<T, MTILE, 1>(Mi, Ni, Ki, batch_total, x_bs, w_bs, br, bx, bw, vec, stream);
    } else {
      static constexpr int BLOCK = 128;
      static constexpr int E = 16/sizeof(T);
      bool vec = base_aligned && !(ALIGN & (N*static_cast<int64_t>(sizeof(T))));
      int cpt = vec ? E : 1;
      dim3 block_dim(BLOCK, 1, 1);
      dim3 grid_dim(static_cast<unsigned>((N + BLOCK*cpt - 1)/(BLOCK*cpt)), static_cast<unsigned>(batch_total), 1);
      if (vec) gemv_wn_kernel<T, MTILE, BLOCK, true><<<grid_dim, block_dim, 0, stream>>>(Mi, Ni, Ki, batch_total, x_bs, w_bs, br, bx, bw);
      else gemv_wn_kernel<T, MTILE, BLOCK, false><<<grid_dim, block_dim, 0, stream>>>(Mi, Ni, Ki, batch_total, x_bs, w_bs, br, bx, bw);
    }
  }
  template <typename T>
  static void launch_gemv(
    int64_t M, int64_t N, int64_t K, int64_t batch_total,
    int64_t x_bs, int64_t w_bs,
    T *__restrict__ br,
    const T *__restrict__ bx,
    const T *__restrict__ bw,
    bool wT, cudaStream_t stream
  ) {
    if (N == 1) wT = true;
    if (M <= 1) launch_gemv_mtile<T, 1>(M, N, K, batch_total, x_bs, w_bs, br, bx, bw, wT, stream);
    else if (M <= 2) launch_gemv_mtile<T, 2>(M, N, K, batch_total, x_bs, w_bs, br, bx, bw, wT, stream);
    else if (M <= 4) launch_gemv_mtile<T, 4>(M, N, K, batch_total, x_bs, w_bs, br, bx, bw, wT, stream);
    else launch_gemv_mtile<T, 8>(M, N, K, batch_total, x_bs, w_bs, br, bx, bw, wT, stream);
  }

  static mag_status_t materialize_batch_broadcast(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *t, const mag_tensor_t *r) {
    int64_t rb = r->meta.coords.rank > 2 ? r->meta.coords.rank-2 : 0;
    int64_t tr = t->meta.coords.rank;
    int64_t shape[MAG_MAX_DIMS] = {};
    for (int64_t i=0; i < rb; ++i) shape[i] = r->meta.coords.shape[i];
    shape[rb] = t->meta.coords.shape[tr-2];
    shape[rb+1] = t->meta.coords.shape[tr-1];
    mag_tensor_t *expanded = nullptr;
    if (mag_status_t st = mag_broadcast(err, &expanded, t, rb+2, shape); mag_iserr(st)) return st;
    mag_status_t st = mag_contiguous(err, out, expanded);
    mag_tensor_decref(expanded);
    return st;
  }

  template <typename T>
  static mag_status_t launch_matmul(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream) {
    mag_tensor_t *r = cmd.out[0];
    mag_tensor_t *x = cmd.in[0];
    mag_tensor_t *y = cmd.in[1];
    mag_assert2(mag_tensor_is_contiguous(r));
    const physical_device &dev = device_of(r);
    bool x_batch_packed, y_batch_packed;
    mag_mat_layout_type_t x_layout = mag_mat_layout_detect(&x->meta.coords, &x_batch_packed);
    mag_mat_layout_type_t y_layout = mag_mat_layout_detect(&y->meta.coords, &y_batch_packed);
    bool x_ok = x_layout != MAG_MAT_LAYOUT_TYPE_OTHER && x_batch_packed;
    bool y_ok = y_layout != MAG_MAT_LAYOUT_TYPE_OTHER && y_batch_packed;
    bool xT = x_ok && x_layout == MAG_MAT_LAYOUT_TYPE_TRANSPOSED;
    bool yT = y_ok && y_layout == MAG_MAT_LAYOUT_TYPE_TRANSPOSED;
    bool cloned_x = false;
    bool cloned_y = false;
    mag_status_t st = MAG_OK;
    if (!x_ok) {
      if (st = mag_contiguous(err, &x, x); mag_iserr(st)) return st;
      xT = false;
      cloned_x = true;
    }
    if (!y_ok) {
      if (st = mag_contiguous(err, &y, y); mag_iserr(st)) goto end;
      yT = false;
      cloned_y = true;
    }
    {
      int64_t batch_rank = r->meta.coords.rank > 2 ? r->meta.coords.rank-2 : 0;
      int64_t batch_total = std::accumulate(r->meta.coords.shape, r->meta.coords.shape + batch_rank, int64_t{1}, std::multiplies<int64_t>());
      if (int64_t xb = tensor_batch_total(x); xb != 1 && xb != batch_total) {
        mag_tensor_t *xm = nullptr;
        if (st = materialize_batch_broadcast(err, &xm, x, r); mag_iserr(st)) goto end;
        if (cloned_x) mag_tensor_decref(x);
        x = xm;
        xT = false;
        cloned_x = true;
      }
      if (int64_t yb = tensor_batch_total(y); yb != 1 && yb != batch_total) {
        mag_tensor_t *ym = nullptr;
        if (st = materialize_batch_broadcast(err, &ym, y, r); mag_iserr(st)) goto end;
        if (cloned_y) mag_tensor_decref(y);
        y = ym;
        yT = false;
        cloned_y = true;
      }
      int64_t M = x->meta.coords.rank == 1 ? 1 : x->meta.coords.shape[x->meta.coords.rank - 2];
      int64_t Kx = x->meta.coords.shape[x->meta.coords.rank - 1];
      int64_t N = y->meta.coords.rank == 1 ? 1 : y->meta.coords.shape[y->meta.coords.rank - 1];
      int64_t Ky = y->meta.coords.rank == 1 ? y->meta.coords.shape[0] : y->meta.coords.shape[y->meta.coords.rank - 2];
      mag_assert2(Kx == Ky);
      int64_t K = Kx;
      int64_t a_bs = tensor_batch_total(x) == 1 ? 0 : M*K;
      int64_t b_bs = tensor_batch_total(y) == 1 ? 0 : K*N;
      int64_t c_bs = M*N;
      auto *__restrict__ br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
      const auto *__restrict__ bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
      const auto *__restrict__ by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));
      mag_matmul_type_t mm_type = mag_matmul_type_detect(x, y);
      switch (mm_type) {
        case MAG_MATMUL_TYPE_DOT:
        case MAG_MATMUL_TYPE_BMM_DOT:
        case MAG_MATMUL_TYPE_GEMV_VEC_MAT:
        case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT:
          launch_gemv(1, N, K, batch_total, a_bs, b_bs, br, bx, by, yT, stream);
          goto end;
        case MAG_MATMUL_TYPE_GEMV_MAT_VEC:
        case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC:
          launch_gemv(1, M, K, batch_total, b_bs, a_bs, br, by, bx, !xT, stream);
          goto end;
        default: break;
      }
      if (N == 1) {
        launch_gemv(1, M, K, batch_total, b_bs, a_bs, br, by, bx, !xT, stream);
        goto end;
      }
      if (M <= GEMV_MAX_THIN_M && !xT) {
        launch_gemv(M, N, K, batch_total, a_bs, b_bs, br, bx, by, yT, stream);
        goto end;
      }
      gemm_desc d {};
      d.dtype = r->meta.dtype;
      d.a = bx;
      d.b = by;
      d.c = br;
      d.M = M;
      d.N = N;
      d.K = K;
      d.a_kmajor = !xT;
      d.lda = xT ? M : K;
      d.b_kmajor = yT;
      d.ldb = yT ? K : N;
      d.ldc = N;
      d.batch = batch_total;
      d.a_bstride = a_bs;
      d.b_bstride = b_bs;
      d.c_bstride = c_bs;
      st = gemm(err, dev, d, stream);
    }
    end:
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
