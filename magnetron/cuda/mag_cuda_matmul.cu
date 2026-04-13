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

    [[nodiscard]] static PFN_cuTensorMapEncodeTiled_v12000 getProcAddress_cuTensorMapEncodeTiled() {
        static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
        if (fn) return fn;
        cudaDriverEntryPointQueryResult stat;
        auto res = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", reinterpret_cast<void **>(&fn), 12000, cudaEnableDefault, &stat);
        if (mag_unlikely(res != cudaSuccess || stat != cudaDriverEntryPointSuccess))
            throw std::runtime_error {"Failed to get address of cuTensorMapEncodeTiled: " + std::string{cudaGetErrorString(res)}};
        return fn;
    }

    template <typename T>
    [[nodiscard]] static CUtensorMap make_tma_3d_map(
        const T* base,
        uint64_t dim0,           // fastest-moving dimension
        uint64_t dim1,           // next dimension
        uint64_t dim2,           // batch dimension
        uint64_t stride1_bytes,  // stride for dim1
        uint64_t stride2_bytes,  // stride for dim2
        uint32_t box0,
        uint32_t box1,
        uint32_t box2
    ) {
        if (!base)
            throw std::invalid_argument("make_tma_3d_map: base is null");
        if (dim0 < 1 || dim1 < 1 || dim2 < 1)
            throw std::invalid_argument("make_tma_3d_map: dimensions must be >= 1");
        if ((stride1_bytes & 15) != 0 || (stride2_bytes & 15) != 0)
            throw std::invalid_argument("make_tma_3d_map: strides must be multiples of 16 for TMA");

        static constexpr uint32_t rank = 3;

        uint64_t global_dim[rank] = { dim0, dim1, dim2 };
        uint64_t global_stride[rank - 1] = { stride1_bytes, stride2_bytes };
        uint32_t box_dim[rank] = { box0, box1, box2 };
        uint32_t elem_stride[rank] = { 1, 1, 1 };

        CUtensorMap map{};
        CUtensorMapDataType dtype{};
        if constexpr (std::is_same_v<T, __nv_bfloat16>) dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        else if constexpr (std::is_same_v<T, half>) dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
        else throw std::runtime_error("unsupported dtype for TMA map");

        auto* cuTensorMapEncodeTiled = getProcAddress_cuTensorMapEncodeTiled();
        CUresult rc = (*cuTensorMapEncodeTiled)(
            &map,
            dtype,
            rank,
            const_cast<T*>(base),
            global_dim,
            global_stride,
            box_dim,
            elem_stride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        if (rc != CUDA_SUCCESS)
            throw std::runtime_error("cuTensorMapEncodeTiled failed");

        return map;
    }

    [[nodiscard]] static int64_t tensor_batch_total(const mag_tensor_t *tensor) {
        int64_t ra = tensor->coords.rank;
        if (ra <= 2) return 1;
        int64_t batch=1;
        int64_t delta=ra-2;
        for (int64_t i=0; i < delta; ++i)
            batch *= tensor->coords.shape[i];
        return batch;
    }

    template <typename T>
    [[nodiscard]] static CUtensorMap make_map_from_batched_2d_view(
        const T* base,
        int64_t rows,
        int64_t cols,
        int64_t row_stride_elems,
        int64_t batch_total,
        int64_t batch_stride_elems,
        uint32_t box_fast,
        uint32_t box_slow
    ) {
        return make_tma_3d_map<T>(
            base,
            static_cast<uint64_t>(cols),                         // dim0
            static_cast<uint64_t>(rows),                         // dim1
            static_cast<uint64_t>(batch_total),                  // dim2
            static_cast<uint64_t>(row_stride_elems) * sizeof(T), // stride1
            static_cast<uint64_t>(batch_stride_elems) * sizeof(T), // stride2
            box_fast,
            box_slow,
            1
        );
    }

    template <typename T, int BM, int BK>
    CUtensorMap make_map_A_from_tensor(
        const mag_tensor_t* x,
        const T* base
    ) {
        const int r = x->coords.rank;
        const int64_t rows = (r == 1) ? 1 : x->coords.shape[r - 2];
        const int64_t cols = x->coords.shape[r - 1];
        const int64_t row_stride = (r == 1) ? 1 : x->coords.strides[r - 2];
        const int64_t batch_total = tensor_batch_total(x);
        const int64_t batch_stride = rows * cols;

        return make_map_from_batched_2d_view<T>(
            base,
            rows,
            cols,
            row_stride,
            batch_total,
            batch_stride,
            BK,  // fast dimension tile
            BM   // slow dimension tile
        );
    }

    template <typename T, int BK, int BN>
    CUtensorMap make_map_B_from_tensor(
        const mag_tensor_t* y,
        const T* base
    ) {
        const int r = y->coords.rank;

        const int64_t rows = (r == 1) ? y->coords.shape[0] : y->coords.shape[r - 2];
        const int64_t cols = (r == 1) ? 1 : y->coords.shape[r - 1];

        const int64_t row_stride = (r == 1) ? 1 : y->coords.strides[r - 2];

        const int64_t batch_total = tensor_batch_total(y);
        const int64_t batch_stride = rows * cols;

        return make_map_from_batched_2d_view<T>(
            base,
            rows,
            cols,
            row_stride,
            batch_total,
            batch_stride,
            BN,  // fast dimension tile
            BK   // slow dimension tile
        );
    }

#if MAG_CUDA_MATMUL_USE_WMMA
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
        static_assert(BK==16);
        static_assert(!(BM&15), "BM must be multiple of 16");
        static_assert(!(BN&15), "BN must be multiple of 16");

        static constexpr int WARPS_M = BM>>4;
        static constexpr int WARPS_N = BN>>4;
        static constexpr int WARPS_PER_BLOCK = WARPS_M*WARPS_N;
        static constexpr int BLOCK_THREADS = WARPS_PER_BLOCK<<5;
        static constexpr int A_SIZE = BM*BK;
        static constexpr int B_SIZE = BK*BN;

        int batch = blockIdx.z;
        if (batch >= batch_total) return;
        int tile_m = blockIdx.y * BM;
        int tile_n = blockIdx.x * BN;
        int tid = threadIdx.x;
        int warp_id = tid>>5;

        T *__restrict__ r_batch = br + batch*M*N;

        extern __shared__ __align__(128) uint8_t smem_raw[];
        __shared__ uint64_t a_bar[STAGES];
        __shared__ uint64_t b_bar[STAGES];

        auto *a_smem = reinterpret_cast<T *>(smem_raw);
        auto *b_smem = a_smem + STAGES*A_SIZE;
        auto *c_smem = reinterpret_cast<float *>(b_smem + STAGES*B_SIZE);

        if (tid == 0) {
           for (int stage=0; stage < STAGES; ++stage) {
               cuda::ptx::mbarrier_init(a_bar+stage, 1);
               cuda::ptx::mbarrier_init(b_bar+stage, 1);
           }
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.f);

        auto issue_tma_stage = [&](int stage, int ktile) -> void {
            if (tid != 0) return;
            auto *a_buf = a_smem + stage*A_SIZE;
            auto *b_buf = b_smem + stage*B_SIZE;
            int32_t a_coords[3] = {ktile*BK, tile_m, batch};
            int32_t b_coords[3] = {tile_n, ktile*BK, batch};
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_cluster,
                cuda::ptx::space_global,
                a_buf, &map_a, a_coords, a_bar+stage
            );
            cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release,
                cuda::ptx::scope_cta,
                cuda::ptx::space_shared,
                a_bar+stage,
                sizeof(T)*A_SIZE
            );
            cuda::ptx::cp_async_bulk_tensor(
                cuda::ptx::space_cluster,
                cuda::ptx::space_global,
                b_buf, &map_b, b_coords, b_bar+stage
            );
            cuda::ptx::mbarrier_arrive_expect_tx(
                cuda::ptx::sem_release,
                cuda::ptx::scope_cta,
                cuda::ptx::space_shared,
                b_bar+stage,
                sizeof(T)*B_SIZE
            );
        };

        auto await_stage_ready = [&](int stage, int parity) -> void {
            while (!cuda::ptx::mbarrier_try_wait_parity(a_bar+stage, parity));
            while (!cuda::ptx::mbarrier_try_wait_parity(b_bar+stage, parity));
        };

        int warp_m = warp_id / WARPS_N;
        int warp_n = warp_id % WARPS_N;
        auto compute_stage = [&](int stage) -> void {
            if (warp_id >= WARPS_PER_BLOCK) return;
            auto *a_buf = a_smem + stage*A_SIZE;
            auto *b_buf = b_smem + stage*B_SIZE;

            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::row_major> b_frag;

            const auto *a_ptr = a_buf + (warp_m<<4) * BK;
            const auto *b_ptr = b_buf + (warp_n<<4);
            wmma::load_matrix_sync(a_frag, a_ptr, BK);
            wmma::load_matrix_sync(b_frag, b_ptr, BN);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        };

        int k_tiles = (K + BK - 1) / BK;
        int prefetch = k_tiles < STAGES ? k_tiles : STAGES;
        for (int s=0; s < prefetch; ++s)
            issue_tma_stage(s, s);
        __syncthreads();

        for (int kt=0; kt < k_tiles; ++kt) {
            int stage = kt % STAGES;
            int phase = 1 & (kt / STAGES);
            int next_kt = kt + STAGES;

            await_stage_ready(stage, phase);
            compute_stage(stage);
            __syncthreads();

            if (next_kt < k_tiles)
                issue_tma_stage(stage, next_kt);
            __syncthreads();
        }

        if (warp_id < WARPS_PER_BLOCK) {
            float *c_ptr = c_smem + (warp_id<<8);
            wmma::store_matrix_sync(c_ptr, c_frag, 16, wmma::mem_row_major);
        }
        __syncthreads();

        #pragma unroll
        for (int i=tid; i < BM*BN; i += BLOCK_THREADS) {
            int row = i / BN;
            int col = i % BN;
            int g_row = tile_m + row;
            int g_col = tile_n + col;
            if (g_row < M && g_col < N) {
                int warp_store_m = row>>4;
                int warp_store_n = col>>4;
                int warp_store = warp_store_m*WARPS_N + warp_store_n;
                int row_in_warp = row&15;
                int col_in_warp = col&15;
                float v = c_smem[(warp_store<<8) + (row_in_warp<<4) + col_in_warp];
                r_batch[g_row*N + g_col] = static_cast<T>(v);
            }
        }
    }

    template <typename T>
    static void launch_matmul_kernel_wmma(
        int64_t M, int64_t N, int64_t K,
        int64_t batch_total,
        T *__restrict__ br,
        const T *__restrict__ bx,
        const T *__restrict__ by,
        mag_tensor_t *x,
        mag_tensor_t *y,
        bool xT, bool yT
    ) {
        static_assert(std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>);

        static constexpr int BM = 128;
        static constexpr int BN = 64;
        static constexpr int BK = 16;
        static_assert(BK == 16);
        static constexpr int WARPS_M = BM>>4;
        static constexpr int WARPS_N = BN>>4;
        static constexpr int STAGES = 4;
        static constexpr int BLOCK_THREADS = WARPS_M * (WARPS_N<<5);
        static_assert((BM>>4) * ((BN>>4)<<5) <= 1024);

        int max_smem_real;
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        size_t smem = sizeof(T)*(STAGES*(BM*BK + BN*BK)) + sizeof(float)*(WARPS_M*WARPS_N*16*16);
        mag_assert(smem <= (unsigned)max_smem_real, "Required shared memory size for matmul kernel exceeds device limit");
        auto set_kernel_smem_size = [&](auto kernel, size_t size) -> void {
            mag_assert2(size <= INT32_MAX);
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(size));
        };

        CUtensorMap map_a = make_map_A_from_tensor<T, BM, BK>(x, bx);
        CUtensorMap map_b = make_map_B_from_tensor<T, BK, BN>(y, by);

        dim3 grid_dim((N + BN - 1)/BN, (M + BM - 1)/BM,batch_total);
        dim3 block_dim(BLOCK_THREADS, 1, 1);

        if (!xT && !yT) {
            auto *kernel = matmul_kernel_wmma<T, false, false, BM, BN, STAGES>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
        } else if (!xT && yT) {
            auto *kernel = matmul_kernel_wmma<T, false, true, BM, BN, STAGES>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
        } else if (xT && !yT) {
            auto *kernel = matmul_kernel_wmma<T, true, false, BM, BN, STAGES>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, map_a, map_b);
        } else {
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
                launch_matmul_kernel_wmma(M, N, K, batch_total, br, bx, by, x, y, xT, yT);
                goto end;
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
