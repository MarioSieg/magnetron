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

#include "../mag_cuda_gemm.cuh"
#include "../mag_cuda_tma.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

#ifndef MAG_CUDA_ARCH_NS
#error "mag_cuda_gemm_sm100.cu must be compiled through mag_register_cuda_arch()"
#endif

namespace mag::MAG_CUDA_ARCH_NS {
  namespace {
    constexpr uint32_t BM = 128;
    constexpr uint32_t BK = 64;
    constexpr uint32_t UMMA_K = 16;
    constexpr uint32_t NUM_THREADS = 256;
    constexpr uint32_t EPILOGUE_THREADS = 128;
    constexpr uint32_t EPILOGUE_WARP0 = 4;
    constexpr uint32_t ACC_STAGES = 2;
    constexpr uint32_t GROUP_M = 16;
    constexpr uint32_t CTRL_BYTES = 1024;
    constexpr uint32_t SMEM_ALIGN_SLACK = 1024;
    constexpr uint32_t MN_CHUNK_BYTES = BK*64*2;

    struct gemm_kargs final {
      int32_t M, N, K;
      int32_t batch;
      int32_t k_batch;
      int32_t k_tiles;
      int32_t m_tiles, n_tiles;
      uint32_t tiles_per_batch;
      uint32_t num_tiles;
      int32_t a_bcast, b_bcast;
      int32_t c_vec_ok;
      int64_t ldc;
      int64_t c_bstride;
      void *c;
      const void *bias;
    };

    __device__ __forceinline__ uint32_t smem_u32(const void *p) noexcept {
      return static_cast<uint32_t>(__cvta_generic_to_shared(p));
    }

    __device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) noexcept {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count) : "memory");
    }

    __device__ __forceinline__ void fence_barrier_init() noexcept {
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }

    __device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx) noexcept {
      asm volatile(
        "{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.release.cta.shared::cta.b64 st, [%0], %1;\n}"
        :: "r"(bar), "r"(tx) : "memory"
      );
    }

    __device__ __forceinline__ void mbar_arrive(uint32_t bar) noexcept {
      asm volatile(
        "{\n.reg .b64 st;\nmbarrier.arrive.release.cta.shared::cta.b64 st, [%0];\n}"
        :: "r"(bar) : "memory"
      );
    }

    [[nodiscard]] __device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) noexcept {
      uint32_t ok;
      asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\nselp.b32 %0, 1, 0, p;\n}"
        : "=r"(ok) : "r"(bar), "r"(parity) : "memory"
      );
      return ok != 0;
    }

    __device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) noexcept {
      while (!mbar_try_wait(bar, parity)) {}
    }

    __device__ __forceinline__ void tma_prefetch_desc(const void *map) noexcept {
      asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
    }

    __device__ __forceinline__ void tma_load_3d(uint32_t dst, const void *map, uint32_t bar, int32_t c0, int32_t c1, int32_t c2) noexcept {
      asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(bar) : "memory"
      );
    }

    __device__ __forceinline__ void tmem_alloc(uint32_t dst_smem, uint32_t ncols) noexcept {
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(dst_smem), "r"(ncols) : "memory");
    }

    __device__ __forceinline__ void tmem_relinquish_alloc_permit() noexcept {
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }

    __device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t ncols) noexcept {
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(ncols) : "memory");
    }

    __device__ __forceinline__ void tc_fence_before_sync() noexcept {
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    }

    __device__ __forceinline__ void tc_fence_after_sync() noexcept {
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    }

    __device__ __forceinline__ void tc_commit(uint32_t bar) noexcept {
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(bar) : "memory");
    }

    __device__ __forceinline__ void tc_mma_f16(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, uint32_t accumulate) noexcept {
      asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}"
        :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(accumulate) : "memory"
      );
    }

    __device__ __forceinline__ void tmem_ld_32x32b_x32(uint32_t taddr, uint32_t (&v)[32]) noexcept {
      asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]),
          "=r"(v[8]), "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15]),
          "=r"(v[16]), "=r"(v[17]), "=r"(v[18]), "=r"(v[19]), "=r"(v[20]), "=r"(v[21]), "=r"(v[22]), "=r"(v[23]),
          "=r"(v[24]), "=r"(v[25]), "=r"(v[26]), "=r"(v[27]), "=r"(v[28]), "=r"(v[29]), "=r"(v[30]), "=r"(v[31])
        : "r"(taddr) : "memory"
      );
    }

    __device__ __forceinline__ void tmem_ld_wait() noexcept {
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    }

    [[nodiscard]] __device__ __forceinline__ uint64_t make_smem_desc_sw128(uint32_t saddr, uint32_t lbo_bytes, uint32_t sbo_bytes) noexcept {
      uint64_t d = static_cast<uint64_t>((saddr>>4) & 0x3fffu);
      d |= static_cast<uint64_t>((lbo_bytes>>4) & 0x3fffu)<<16;
      d |= static_cast<uint64_t>((sbo_bytes>>4) & 0x3fffu)<<32;
      d |= static_cast<uint64_t>(1)<<46;
      d |= static_cast<uint64_t>(2)<<61;
      return d;
    }

    template <typename T, bool A_KMAJOR, bool B_KMAJOR, uint32_t BN>
    [[nodiscard]] __host__ __device__ constexpr uint32_t make_instr_desc() noexcept {
      constexpr uint32_t fmt = std::is_same_v<T, __nv_bfloat16> ? 1u : 0u;
      uint32_t d = 0;
      d |= 1u<<4;
      d |= fmt<<7;
      d |= fmt<<10;
      d |= (A_KMAJOR ? 0u : 1u)<<15;
      d |= (B_KMAJOR ? 0u : 1u)<<16;
      d |= (BN>>3)<<17;
      d |= (BM>>4)<<24;
      return d;
    }

    template <typename T>
    __device__ __forceinline__ void pack2(uint32_t &dst, float a, float b) noexcept {
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        __nv_bfloat162 p = __floats2bfloat162_rn(a, b);
        dst = *reinterpret_cast<uint32_t *>(&p);
      } else {
        half2 p = __floats2half2_rn(a, b);
        dst = *reinterpret_cast<uint32_t *>(&p);
      }
    }

    template <uint32_t BN>
    __device__ __forceinline__ void decode_tile(uint32_t tile, const gemm_kargs &a, int32_t &m0, int32_t &n0, int32_t &batch) noexcept {
      batch = static_cast<int32_t>(tile / a.tiles_per_batch);
      uint32_t t = tile - static_cast<uint32_t>(batch)*a.tiles_per_batch;
      uint32_t tiles_per_group = GROUP_M*static_cast<uint32_t>(a.n_tiles);
      uint32_t group = t / tiles_per_group;
      uint32_t first_m = group*GROUP_M;
      uint32_t gsize = min(GROUP_M, static_cast<uint32_t>(a.m_tiles) - first_m);
      uint32_t in = t - group*tiles_per_group;
      uint32_t mt = first_m + in % gsize;
      uint32_t nt = in / gsize;
      m0 = static_cast<int32_t>(mt*BM);
      n0 = static_cast<int32_t>(nt*BN);
    }

    template <typename T, bool A_KMAJOR, bool B_KMAJOR, uint32_t BN, uint32_t STAGES>
    __global__ void __launch_bounds__(NUM_THREADS, 1) gemm_tcgen05_kernel(
      const __grid_constant__ CUtensorMap map_a,
      const __grid_constant__ CUtensorMap map_b,
      const __grid_constant__ gemm_kargs args
    ) {
      static_assert(std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>);
      static_assert(BN == 64 || BN == 128 || BN == 256, "UMMA N must be a multiple of 16 in [16, 256]");
      static_assert(BM == 128);
      constexpr uint32_t A_BYTES = BM*BK*sizeof(T);
      constexpr uint32_t B_BYTES = BN*BK*sizeof(T);
      constexpr uint32_t STAGE_BYTES = A_BYTES + B_BYTES;
      constexpr uint32_t TMEM_COLS = ACC_STAGES*BN < 32 ? 32 : ACC_STAGES*BN;
      constexpr uint32_t IDESC = make_instr_desc<T, A_KMAJOR, B_KMAJOR, BN>();
      constexpr uint32_t A_LBO = A_KMAJOR ? 0 : MN_CHUNK_BYTES;
      constexpr uint32_t B_LBO = B_KMAJOR ? 0 : MN_CHUNK_BYTES;
      constexpr uint32_t A_KSTEP = A_KMAJOR ? (UMMA_K*sizeof(T))>>4 : (UMMA_K*128)>>4;
      constexpr uint32_t B_KSTEP = B_KMAJOR ? (UMMA_K*sizeof(T))>>4 : (UMMA_K*128)>>4;
      static_assert((STAGE_BYTES & 1023) == 0);

      extern __shared__ uint8_t smem_raw[];
      const uint32_t smem_base = smem_u32(smem_raw);
      const uint32_t ctrl = (smem_base + 1023u) & ~1023u;
      const uint32_t full_bar = ctrl;
      const uint32_t empty_bar = full_bar + STAGES*8;
      const uint32_t tmem_full_bar = empty_bar + STAGES*8;
      const uint32_t tmem_empty_bar = tmem_full_bar + ACC_STAGES*8;
      const uint32_t tmem_ptr_addr = tmem_empty_bar + ACC_STAGES*8;
      const uint32_t tiles_base = ctrl + CTRL_BYTES;
      volatile uint32_t *tmem_ptr_slot = reinterpret_cast<volatile uint32_t *>(smem_raw + (tmem_ptr_addr - smem_base));

      const uint32_t warp = threadIdx.x>>5;
      const uint32_t lane = threadIdx.x&31;
      const uint32_t num_k_iters = static_cast<uint32_t>(args.k_tiles)*static_cast<uint32_t>(args.k_batch);

      if (warp == 0 && lane == 0) {
        tma_prefetch_desc(&map_a);
        tma_prefetch_desc(&map_b);
        #pragma unroll
        for (uint32_t s=0; s < STAGES; ++s) {
          mbar_init(full_bar + s*8, 1);
          mbar_init(empty_bar + s*8, 1);
        }
        #pragma unroll
        for (uint32_t s=0; s < ACC_STAGES; ++s) {
          mbar_init(tmem_full_bar + s*8, 1);
          mbar_init(tmem_empty_bar + s*8, EPILOGUE_THREADS);
        }
        fence_barrier_init();
      } else if (warp == 2) {
        tmem_alloc(tmem_ptr_addr, TMEM_COLS);
        tmem_relinquish_alloc_permit();
      }
      __syncthreads();
      tc_fence_after_sync();
      const uint32_t tmem_base = *tmem_ptr_slot;

      if (warp == 0) {
        /* TMA producer */
        if (lane == 0) {
          uint32_t stage = 0, phase = 0;
          for (uint32_t tile = blockIdx.x; tile < args.num_tiles; tile += gridDim.x) {
            int32_t m0, n0, batch;
            decode_tile<BN>(tile, args, m0, n0, batch);
            for (uint32_t it=0; it < num_k_iters; ++it) {
              mbar_wait(empty_bar + stage*8, phase^1);
              int32_t kb = static_cast<int32_t>(it / static_cast<uint32_t>(args.k_tiles));
              int32_t k0 = static_cast<int32_t>(it - static_cast<uint32_t>(kb)*static_cast<uint32_t>(args.k_tiles))*static_cast<int32_t>(BK);
              int32_t zsel = args.k_batch > 1 ? kb : batch;
              int32_t za = args.a_bcast ? 0 : zsel;
              int32_t zb = args.b_bcast ? 0 : zsel;
              uint32_t a_dst = tiles_base + stage*STAGE_BYTES;
              uint32_t b_dst = a_dst + A_BYTES;
              uint32_t fb = full_bar + stage*8;
              mbar_arrive_expect_tx(fb, STAGE_BYTES);
              if constexpr (A_KMAJOR) {
                tma_load_3d(a_dst, &map_a, fb, k0, m0, za);
              } else {
                #pragma unroll
                for (uint32_t i=0; i < BM/64; ++i)
                  tma_load_3d(a_dst + i*MN_CHUNK_BYTES, &map_a, fb, m0 + static_cast<int32_t>(i*64), k0, za);
              }
              if constexpr (B_KMAJOR) {
                tma_load_3d(b_dst, &map_b, fb, k0, n0, zb);
              } else {
                #pragma unroll
                for (uint32_t i=0; i < BN/64; ++i)
                  tma_load_3d(b_dst + i*MN_CHUNK_BYTES, &map_b, fb, n0 + static_cast<int32_t>(i*64), k0, zb);
              }
              if (++stage == STAGES) { stage = 0; phase ^= 1; }
            }
          }
        }
      } else if (warp == 1) {
        /* MMA issuer */
        if (lane == 0) {
          uint32_t stage = 0, phase = 0;
          uint32_t acc = 0, acc_phase = 0;
          for (uint32_t tile = blockIdx.x; tile < args.num_tiles; tile += gridDim.x) {
            mbar_wait(tmem_empty_bar + acc*8, acc_phase^1);
            tc_fence_after_sync();
            const uint32_t d_tmem = tmem_base + acc*BN;
            for (uint32_t it=0; it < num_k_iters; ++it) {
              mbar_wait(full_bar + stage*8, phase);
              tc_fence_after_sync();
              const uint32_t a_s = tiles_base + stage*STAGE_BYTES;
              const uint32_t b_s = a_s + A_BYTES;
              const uint64_t a_desc = make_smem_desc_sw128(a_s, A_LBO, 1024);
              const uint64_t b_desc = make_smem_desc_sw128(b_s, B_LBO, 1024);
              #pragma unroll
              for (uint32_t kk=0; kk < BK/UMMA_K; ++kk)
                tc_mma_f16(d_tmem, a_desc + kk*A_KSTEP, b_desc + kk*B_KSTEP, IDESC, (it|kk) != 0);
              tc_commit(empty_bar + stage*8);
              if (it+1 == num_k_iters) tc_commit(tmem_full_bar + acc*8);
              if (++stage == STAGES) { stage = 0; phase ^= 1; }
            }
            if (++acc == ACC_STAGES) { acc = 0; acc_phase ^= 1; }
          }
        }
      } else if (warp >= EPILOGUE_WARP0) {
        const uint32_t wq = warp - EPILOGUE_WARP0;
        uint32_t acc = 0, acc_phase = 0;
        const auto *bias = static_cast<const T *>(args.bias);
        for (uint32_t tile = blockIdx.x; tile < args.num_tiles; tile += gridDim.x) {
          int32_t m0, n0, batch;
          decode_tile<BN>(tile, args, m0, n0, batch);
          mbar_wait(tmem_full_bar + acc*8, acc_phase);
          tc_fence_after_sync();
          const int32_t row = m0 + static_cast<int32_t>(wq*32 + lane);
          const bool row_ok = row < args.M;
          T *c_row = static_cast<T *>(args.c) + static_cast<int64_t>(batch)*args.c_bstride + static_cast<int64_t>(row_ok ? row : 0)*args.ldc;
          const float bias_v = (bias && row_ok) ? static_cast<float>(bias[row]) : 0.0f;
          const uint32_t taddr = tmem_base + ((wq*32)<<16) + acc*BN;
          #pragma unroll
          for (uint32_t c=0; c < BN/32; ++c) {
            uint32_t v[32];
            tmem_ld_32x32b_x32(taddr + c*32, v);
            tmem_ld_wait();
            if (c+1 == BN/32) {
              tc_fence_before_sync();
              mbar_arrive(tmem_empty_bar + acc*8);
            }
            if (!row_ok) continue;
            const int32_t col0 = n0 + static_cast<int32_t>(c*32);
            if (col0 >= args.N) continue;
            float f[32];
            #pragma unroll
            for (uint32_t j=0; j < 32; ++j) f[j] = __uint_as_float(v[j]) + bias_v;
            if (args.c_vec_ok && col0 + 32 <= args.N) {
              uint32_t p[16];
              #pragma unroll
              for (uint32_t j=0; j < 16; ++j) pack2<T>(p[j], f[2*j], f[2*j+1]);
              auto *dst = reinterpret_cast<uint4 *>(c_row + col0);
              #pragma unroll
              for (uint32_t j=0; j < 4; ++j) dst[j] = make_uint4(p[4*j], p[4*j+1], p[4*j+2], p[4*j+3]);
            } else {
              #pragma unroll
              for (uint32_t j=0; j < 32; ++j)
                if (col0 + static_cast<int32_t>(j) < args.N) c_row[col0 + j] = static_cast<T>(f[j]);
            }
          }
          if (++acc == ACC_STAGES) { acc = 0; acc_phase ^= 1; }
        }
        tc_fence_before_sync();
      }

      __syncthreads();
      if (warp == 2) {
        tc_fence_after_sync();
        tmem_dealloc(tmem_base, TMEM_COLS);
      }
    }

    template <typename T, bool A_KMAJOR, bool B_KMAJOR, uint32_t BN>
    [[nodiscard]] mag_status_t launch(mag_error_t *err, const gemm_desc &d, cudaStream_t stream, uint32_t num_sms) {
      constexpr uint32_t STAGES = BN == 256 ? 4 : 6;
      constexpr uint32_t STAGE_BYTES = (BM + BN)*BK*sizeof(T);
      constexpr size_t SMEM = SMEM_ALIGN_SLACK + CTRL_BYTES + STAGES*STAGE_BYTES;
      auto *kernel = gemm_tcgen05_kernel<T, A_KMAJOR, B_KMAJOR, BN, STAGES>;
      static std::once_flag attr_once;
      static cudaError_t attr_res = cudaSuccess;
      std::call_once(attr_once, [&] { attr_res = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(SMEM)); });
      mag_cu_rt_check(err, attr_res, "tcgen05 gemm: failed to set dynamic shared memory size");

      const bool kbatched = d.k_batch > 1;
      const int64_t a_bs = kbatched ? d.a_kbstride : d.a_bstride;
      const int64_t b_bs = kbatched ? d.b_kbstride : d.b_bstride;
      const int64_t nb = kbatched ? d.k_batch : d.batch;
      const int64_t a_batch = a_bs ? nb : 1;
      const int64_t b_batch = b_bs ? nb : 1;

      CUtensorMap map_a {}, map_b {};
      bool ok_a = A_KMAJOR
        ? tma::encode_operand_3d<T>(map_a, d.a, d.M, d.K, d.lda, a_batch, a_bs, BK, BM, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
        : tma::encode_operand_3d<T>(map_a, d.a, d.K, d.M, d.lda, a_batch, a_bs, 64, BK, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
      bool ok_b = B_KMAJOR
        ? tma::encode_operand_3d<T>(map_b, d.b, d.N, d.K, d.ldb, b_batch, b_bs, BK, BN, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
        : tma::encode_operand_3d<T>(map_b, d.b, d.K, d.N, d.ldb, b_batch, b_bs, 64, BK, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
      if (mag_unlikely(!ok_a || !ok_b))
        return mag_set_error(err, MAG_ERR_BACKEND, "cuda: tcgen05 gemm: failed to encode TMA descriptors (M=%lld N=%lld K=%lld lda=%lld ldb=%lld).",
          static_cast<long long>(d.M), static_cast<long long>(d.N), static_cast<long long>(d.K), static_cast<long long>(d.lda), static_cast<long long>(d.ldb));

      gemm_kargs args {};
      args.M = static_cast<int32_t>(d.M);
      args.N = static_cast<int32_t>(d.N);
      args.K = static_cast<int32_t>(d.K);
      args.batch = static_cast<int32_t>(d.batch);
      args.k_batch = static_cast<int32_t>(d.k_batch);
      args.k_tiles = static_cast<int32_t>((d.K + BK - 1)/BK);
      args.m_tiles = static_cast<int32_t>((d.M + BM - 1)/BM);
      args.n_tiles = static_cast<int32_t>((d.N + BN - 1)/BN);
      args.tiles_per_batch = static_cast<uint32_t>(args.m_tiles)*static_cast<uint32_t>(args.n_tiles);
      args.num_tiles = args.tiles_per_batch*static_cast<uint32_t>(d.batch);
      args.a_bcast = a_batch == 1;
      args.b_bcast = b_batch == 1;
      args.c_vec_ok = !(15 & reinterpret_cast<uintptr_t>(d.c)) && !((d.ldc*sizeof(T)) & 15) && !((d.c_bstride*sizeof(T)) & 15);
      args.ldc = d.ldc;
      args.c_bstride = d.c_bstride;
      args.c = d.c;
      args.bias = d.bias;

      uint32_t grid = std::min<uint32_t>(args.num_tiles, std::max<uint32_t>(num_sms, 1));
      kernel<<<grid, NUM_THREADS, SMEM, stream>>>(map_a, map_b, args);
      mag_cu_rt_check(err, cudaGetLastError(), "tcgen05 gemm: kernel launch failed");
      return MAG_OK;
    }

    template <typename T>
    [[nodiscard]] mag_status_t launch_dtype(mag_error_t *err, const gemm_desc &d, cudaStream_t stream, uint32_t num_sms) {
      /* Wide tiles halve the  traffic per flop but need enough tiles to fill the machine. */
      uint32_t m_tiles = static_cast<uint32_t>((d.M + BM - 1)/BM);
      uint32_t tiles256 = m_tiles*static_cast<uint32_t>((d.N + 255)/256)*static_cast<uint32_t>(d.batch);
      bool wide = d.N > 128 && tiles256 >= num_sms;
      auto go = [&]<bool AK, bool BKM>() -> mag_status_t {
        return wide ? launch<T, AK, BKM, 256>(err, d, stream, num_sms) : launch<T, AK, BKM, 128>(err, d, stream, num_sms);
      };
      if (d.a_kmajor && d.b_kmajor) return go.template operator()<true, true>();
      if (d.a_kmajor && !d.b_kmajor) return go.template operator()<true, false>();
      if (!d.a_kmajor && d.b_kmajor) return go.template operator()<false, true>();
      return go.template operator()<false, false>();
    }
  }

  bool gemm_supported(const gemm_desc &d, uint32_t cc) noexcept {
#if MAG_CUDA_SM100_FAMILY
    bool arch_ok = cc/100 == 10 && cc != 1200;
#else
    bool arch_ok = cc == 1000;
#endif
    if (!arch_ok) return false;
    if (d.dtype != MAG_DTYPE_BFLOAT16 && d.dtype != MAG_DTYPE_FLOAT16) return false;
    if (!d.a || !d.b || !d.c) return false;
    if (d.M < 1 || d.N < 1 || d.K < 1 || d.batch < 1 || d.k_batch < 1) return false;
    if (d.M > INT32_MAX || d.N > INT32_MAX || d.K > INT32_MAX || d.batch > INT32_MAX || d.k_batch > INT32_MAX) return false;
    if (d.k_batch > 1 && d.batch != 1) return false;
    int64_t tiles = ((d.M + BM - 1)/BM)*((d.N + 127)/128)*d.batch;
    if (tiles > UINT32_MAX) return false;
    if (static_cast<uint64_t>(d.K)*static_cast<uint64_t>(d.k_batch) > UINT32_MAX) return false;
    if ((reinterpret_cast<uintptr_t>(d.a) & 15) || (reinterpret_cast<uintptr_t>(d.b) & 15)) return false;
    if ((d.lda & 7) || (d.ldb & 7)) return false;
    if ((d.a_bstride & 7) || (d.b_bstride & 7) || (d.a_kbstride & 7) || (d.b_kbstride & 7)) return false;
    if (d.lda < (d.a_kmajor ? d.K : d.M) || d.ldb < (d.b_kmajor ? d.K : d.N)) return false;
    if (d.bias && (d.M > INT32_MAX)) return false;
    return true;
  }

  mag_status_t gemm(mag_error_t *err, const gemm_desc &d, cudaStream_t stream, uint32_t num_sms) {
    switch (d.dtype) {
      case MAG_DTYPE_BFLOAT16: return launch_dtype<__nv_bfloat16>(err, d, stream, num_sms);
      case MAG_DTYPE_FLOAT16: return launch_dtype<half>(err, d, stream, num_sms);
      default: return mag_set_error(err, MAG_ERR_KERNEL, "cuda: tcgen05 gemm: unsupported dtype %s.", mag_type_trait(d.dtype)->name);
    }
  }
}
