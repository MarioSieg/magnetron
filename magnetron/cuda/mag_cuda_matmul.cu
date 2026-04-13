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
#include <cuda/ptx>
#include <mma.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <numeric>

#define MAG_CUDA_MATMUL_USE_WMMA 1

namespace mag {
    enum class mat_layout_t {
        packed,
        packed_transposed,
        unsupported
    };

    struct mat_layout_info_t {
        mat_layout_t layout;
        bool batch_packed;
        [[nodiscard]] static mat_layout_info_t detect(const mag_tensor_t *tensor);
    };

#if MAG_CUDA_MATMUL_USE_WMMA /* WMMA + TMA fast kernel */

    [[nodiscard]] static int64_t tensor_batch_total(const mag_tensor_t *tensor) {
        int64_t ra = tensor->coords.rank;
        if (ra <= 2) return 1;
        int64_t batch=1;
        int64_t delta=ra-2;
        for (int64_t i=0; i < delta; ++i)
            batch *= tensor->coords.shape[i];
        return batch;
    }

     [[nodiscard]] static PFN_cuTensorMapEncodeTiled_v12000 lookup_proc_address_encode_tmap() {
        static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
        if (fn) return fn;
        cudaDriverEntryPointQueryResult stat;
        auto res = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", reinterpret_cast<void **>(&fn), 12000, cudaEnableDefault, &stat);
        if (mag_unlikely(res != cudaSuccess || stat != cudaDriverEntryPointSuccess))
            throw std::runtime_error {"Failed to get address of cuTensorMapEncodeTiled: " + std::string{cudaGetErrorString(res)}};
        return fn;
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
        int64_t ra = x->coords.rank;
        int64_t batch_total = tensor_batch_total(x);
        int64_t M = ra == 1 ? 1 : x->coords.shape[ra-2];
        int64_t K = x->coords.shape[ra-1];
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
        int64_t ra = y->coords.rank;
        int64_t batch_total = tensor_batch_total(y);
        int64_t K = ra == 1 ? y->coords.shape[0] : y->coords.shape[ra-2];
        int64_t N = ra == 1 ? 1 : y->coords.shape[ra-1];
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

    template <typename T, bool TA, bool TB, int BM, int BN, int STAGES>
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

        static constexpr int BK = 16;
        static_assert(BK == 16);
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
        __shared__ uint64_t a_bar[STAGES];
        __shared__ uint64_t b_bar[STAGES];
        __shared__ uint64_t done_bar[STAGES];

        auto *a_smem = reinterpret_cast<T *>(smem_raw);
        auto *b_smem = a_smem + STAGES*A_SIZE;
        auto *c_smem = reinterpret_cast<float *>(b_smem + STAGES*B_SIZE);

        if (tid == 0) {
            #pragma unroll
            for (int s=0; s < STAGES; ++s) {
                cuda::ptx::mbarrier_init(a_bar+s, 1);
                cuda::ptx::mbarrier_init(b_bar+s, 1);
                cuda::ptx::mbarrier_init(done_bar+s, CONSUMER_WARPS);
            }
        }
        __syncthreads();
        auto issue_tma_stage = [&](int stage, int ktile) -> void {
            if (!is_producer || lane != 0) return;

            auto *a_buf = a_smem + stage * A_SIZE;
            auto *b_buf = b_smem + stage * B_SIZE;
            int32_t a_coords[3];
            int32_t b_coords[3];
            if constexpr (!TA) { // dims {K, M, batch}, box {BK, BM, 1}
                a_coords[0] = ktile * BK;
                a_coords[1] = tile_m;
                a_coords[2] = batch;
            } else { // dims {M, K, batch}, box {BM, BK, 1}
                a_coords[0] = tile_m;
                a_coords[1] = ktile * BK;
                a_coords[2] = batch;
            }
            if constexpr (!TB) { // dims {N, K, batch}, box {BN, BK, 1}
                b_coords[0] = tile_n;
                b_coords[1] = ktile * BK;
                b_coords[2] = batch;
            } else { // dims {K, N, batch}, box {BK, BN, 1}
                b_coords[0] = ktile * BK;
                b_coords[1] = tile_n;
                b_coords[2] = batch;
            }

            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_cluster,
                cuda::ptx::space_global,
                a_buf,
                &map_a,
                a_coords,
                &a_bar[stage]
            );
            cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release,
                cuda::ptx::scope_cta,
                cuda::ptx::space_shared,
                &a_bar[stage],
                sizeof(T)*A_SIZE
            );

            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_cluster,
                cuda::ptx::space_global,
                b_buf,
                &map_b,
                b_coords,
                &b_bar[stage]
            );
            cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release,
                cuda::ptx::scope_cta,
                cuda::ptx::space_shared,
                &b_bar[stage],
                sizeof(T)*B_SIZE
            );
        };

        auto wait_stage_ready = [&](int stage, int phase) -> void {
            while (!cuda::ptx::mbarrier_try_wait_parity(&a_bar[stage], phase)) {}
            while (!cuda::ptx::mbarrier_try_wait_parity(&b_bar[stage], phase)) {}
        };

        auto producer_wait_stage_reusable = [&](int stage, int phase) -> void {
            if (!is_producer || lane != 0) return;
            while (!cuda::ptx::mbarrier_try_wait_parity(&done_bar[stage], phase)) {}
        };

        auto consumer_mark_stage_done = [&](int stage) -> void {
            if (is_producer || lane != 0) return;
            cuda::ptx::mbarrier_arrive(
                cuda::ptx::sem_release,
                cuda::ptx::scope_cta,
                cuda::ptx::space_shared,
                &done_bar[stage]
            );
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
            if constexpr (!TA && !TB) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag0, a_frag1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
                const auto *a_ptr0 = a_buf + (warp_m0<<4)*BK;
                const auto *a_ptr1 = a_buf + (warp_m1<<4)*BK;
                const auto *b_ptr = b_buf + (warp_n0<<4);
                wmma::load_matrix_sync(a_frag0, a_ptr0, BK);
                wmma::load_matrix_sync(a_frag1, a_ptr1, BK);
                wmma::load_matrix_sync(b_frag, b_ptr, BN);
                wmma::mma_sync(c_frag0, a_frag0, b_frag, c_frag0);
                wmma::mma_sync(c_frag1, a_frag1, b_frag, c_frag1);
            } else if constexpr (TA && !TB) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> a_frag0, a_frag1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;
                const auto *a_ptr0 = a_buf + (warp_m0<<4);
                const auto *a_ptr1 = a_buf + (warp_m1<<4);
                const auto *b_ptr = b_buf + (warp_n0<<4);
                wmma::load_matrix_sync(a_frag0, a_ptr0, BM);
                wmma::load_matrix_sync(a_frag1, a_ptr1, BM);
                wmma::load_matrix_sync(b_frag, b_ptr, BN);
                wmma::mma_sync(c_frag0, a_frag0, b_frag, c_frag0);
                wmma::mma_sync(c_frag1, a_frag1, b_frag, c_frag1);
            } else if constexpr (!TA && TB) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag0, a_frag1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
                const auto *a_ptr0 = a_buf + (warp_m0<<4)*BK;
                const auto *a_ptr1 = a_buf + (warp_m1<<4)*BK;
                const auto *b_ptr = b_buf + (warp_n0<<4)*BK;
                wmma::load_matrix_sync(a_frag0, a_ptr0, BK);
                wmma::load_matrix_sync(a_frag1, a_ptr1, BK);
                wmma::load_matrix_sync(b_frag, b_ptr, BK);
                wmma::mma_sync(c_frag0, a_frag0, b_frag, c_frag0);
                wmma::mma_sync(c_frag1, a_frag1, b_frag, c_frag1);
            } else {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::col_major> a_frag0, a_frag1;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
                const auto *a_ptr0 = a_buf + (warp_m0<<4);
                const auto *a_ptr1 = a_buf + (warp_m1<<4);
                const auto *b_ptr = b_buf + (warp_n0<<4)*BK;
                wmma::load_matrix_sync(a_frag0, a_ptr0, BM);
                wmma::load_matrix_sync(a_frag1, a_ptr1, BM);
                wmma::load_matrix_sync(b_frag, b_ptr, BK);
                wmma::mma_sync(c_frag0, a_frag0, b_frag, c_frag0);
                wmma::mma_sync(c_frag1, a_frag1, b_frag, c_frag1);
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
    static void launch_matmul_kernel_wmma(
        int64_t M, int64_t N, int64_t K,
        int64_t batch_total,
        T *__restrict__ br,
        mag_tensor_t *x, mag_tensor_t *y,
        bool xT, bool yT
    ) {
        static_assert(std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>);

        static constexpr int BM = 128;
        static constexpr int BN = 64;
        static constexpr int BK = 16;
        static constexpr int STAGES = 2;
        static constexpr int BLOCK_THREADS = (1 + ((BM / 16) * (BN / 16)) / 2) * 32;

        int max_smem_real;
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

        size_t smem = sizeof(T)*(STAGES * (BM*BK + BN*BK)) + sizeof(float)*((BM>>4)*(BN>>4)<<8);
        mag_assert(smem <= (unsigned)max_smem_real,"Required shared memory size for matmul kernel exceeds device limit");

        auto set_kernel_smem_size = [&](auto kernel, size_t size) -> void {
            mag_assert2(size <= INT32_MAX);
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(size));
        };

        dim3 grid_dim((N + BN - 1) / BN, (M + BM - 1) / BM, batch_total);
        dim3 block_dim(BLOCK_THREADS, 1, 1);

        if (!xT && !yT) {
            CUtensorMap map_a = init_tmap_x<T, false, BM, BK>(x);
            CUtensorMap map_b = init_tmap_y<T, false, BK, BN>(y);
            auto *kernel = matmul_kernel_wmma<T, false, false, BM, BN, STAGES>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
        } else if (!xT && yT) {
            CUtensorMap map_a = init_tmap_x<T, false, BM, BK>(x);
            CUtensorMap map_b = init_tmap_y<T, true, BK, BN>(y);
            auto *kernel = matmul_kernel_wmma<T, false, true, BM, BN, STAGES>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
        } else if (xT && !yT) {
            CUtensorMap map_a = init_tmap_x<T, true, BM, BK>(x);
            CUtensorMap map_b = init_tmap_y<T, false, BK, BN>(y);
            auto *kernel = matmul_kernel_wmma<T, true, false, BM, BN, STAGES>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
        } else {
            CUtensorMap map_a = init_tmap_x<T, true, BM, BK>(x);
            CUtensorMap map_b = init_tmap_y<T, true, BK, BN>(y);
            auto *kernel = matmul_kernel_wmma<T, true, true, BM, BN, STAGES>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
        }
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
    static void launch_matmul_kernel_fallback(
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
        mag_assert(smem <= (unsigned)max_smem_real, "Required shared memory size for matmul kernel exceeds device limit");
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
    }

    template <typename T>
    [[nodiscard]] static bool tma_aligned_bytes(int64_t elems) {
        return ((elems * static_cast<int64_t>(sizeof(T))) & 15) == 0;
    }

    template <typename T>
    [[nodiscard]] static bool is_tensor_tma_compat_x(const mag_tensor_t *x, bool TA) {
        const int64_t r = x->coords.rank;
        const int64_t batch_total = tensor_batch_total(x);
        const int64_t M = (r == 1) ? 1 : x->coords.shape[r - 2];
        const int64_t K = x->coords.shape[r - 1];

        if ((15&mag_tensor_data_ptr(x))) return false;
        if (batch_total < 1) return false;

        if (!TA) {
            // Physical layout: M x K
            const int64_t row_stride = K;
            const int64_t batch_stride = M * K;
            return tma_aligned_bytes<T>(row_stride) &&
                   tma_aligned_bytes<T>(batch_stride);
        } else {
            // Physical layout: K x M
            const int64_t row_stride = M;
            const int64_t batch_stride = M * K;
            return tma_aligned_bytes<T>(row_stride) &&
                   tma_aligned_bytes<T>(batch_stride);
        }
    }

    template <typename T>
    [[nodiscard]] static bool is_tensor_tma_compat_y(const mag_tensor_t *y, bool TB) {
        const int64_t r = y->coords.rank;
        const int64_t batch_total = tensor_batch_total(y);
        const int64_t K = (r == 1) ? y->coords.shape[0] : y->coords.shape[r - 2];
        const int64_t N = (r == 1) ? 1 : y->coords.shape[r - 1];

        if (15&mag_tensor_data_ptr(y)) return false;
        if (batch_total < 1) return false;

        if (!TB) {
            const int64_t row_stride = N;
            const int64_t batch_stride = K * N;
            return tma_aligned_bytes<T>(row_stride) && tma_aligned_bytes<T>(batch_stride);
        } else {
            // Physical layout: N x K
            const int64_t row_stride = K;
            const int64_t batch_stride = K * N;
            return tma_aligned_bytes<T>(row_stride) &&
                   tma_aligned_bytes<T>(batch_stride);
        }
    }

    template <typename T>
    static void launch_matmul(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *y = cmd.in[1];

        mag_assert2(mag_tensor_is_contiguous(r));

        mat_layout_info_t xli = mat_layout_info_t::detect(x);
        mat_layout_info_t yli = mat_layout_info_t::detect(y);

        bool x_ok = xli.layout != mat_layout_t::unsupported && xli.batch_packed;
        bool y_ok = yli.layout != mat_layout_t::unsupported && yli.batch_packed;
        bool xT = x_ok && xli.layout == mat_layout_t::packed_transposed;
        bool yT = y_ok && yli.layout == mat_layout_t::packed_transposed;

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

        int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank - 2];
        int64_t Kx = x->coords.shape[x->coords.rank - 1];
        int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank - 1];
        int64_t Ky = y->coords.rank == 1 ? y->coords.shape[0] : y->coords.shape[y->coords.rank - 2];

        mag_assert2(Kx == Ky);
        int64_t K = Kx;

        int64_t batch_rank = r->coords.rank > 2 ? r->coords.rank-2 : 0;
        int64_t batch_total = std::accumulate(r->coords.shape, r->coords.shape + batch_rank, 1, std::multiplies<int64_t>());

        auto *__restrict__ br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
        const auto *__restrict__ bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        const auto *__restrict__ by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));

        #if MAG_CUDA_MATMUL_USE_WMMA
            if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>) {
                bool can_use_wmma_tma_kernel = is_tensor_tma_compat_x<T>(x, xT) && is_tensor_tma_compat_y<T>(y, yT);
                if (can_use_wmma_tma_kernel) {
                    launch_matmul_kernel_wmma(M, N, K, batch_total, br, x, y, xT, yT);
                    goto end;
                }
            }
        #endif

        launch_matmul_kernel_fallback(M, N, K, batch_total, br, bx, by, xT, yT);

        [[maybe_unused]] end:
            if (cloned_x) mag_tensor_decref(x);
            if (cloned_y) mag_tensor_decref(y);
    }

    void misc_op_matmul(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_matmul<float>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_matmul<half>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_matmul<__nv_bfloat16>(cmd); break;
            default: mag_assert(false, "matmul: unsupported dtype");
        }
    }

    mat_layout_info_t mat_layout_info_t::detect(const mag_tensor_t *tensor) {
        mat_layout_info_t info{mat_layout_t::unsupported, false};
        int64_t rank = tensor->coords.rank;
        if (rank < 2) {
            info.layout = mat_layout_t::packed;
            info.batch_packed = true;
            return info;
        }
        int64_t rows = tensor->coords.shape[rank-2];
        int64_t cols = tensor->coords.shape[rank-1];
        int64_t srow = tensor->coords.strides[rank-2];
        int64_t scol = tensor->coords.strides[rank-1];
        if (scol == 1 && srow == cols) info.layout = mat_layout_t::packed;
        else if (srow == 1 && scol == rows) info.layout = mat_layout_t::packed_transposed;
        else return info;
        int64_t expected_batch_stride = rows*cols;
        if (rank == 2) {
            info.batch_packed = true;
            return info;
        }
        int64_t running = expected_batch_stride;
        for (int64_t i=rank-3; i >= 0; --i) {
            if (tensor->coords.strides[i] != running) {
                info.layout = mat_layout_t::unsupported;
                info.batch_packed = false;
                return info;
            }
            running *= tensor->coords.shape[i];
        }
        info.batch_packed = true;
        return info;
    }
}
