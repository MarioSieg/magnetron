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

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cmath>
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

#if MAG_CUDA_MATMUL_USE_WMMA
    namespace cp_async {
        __device__ __forceinline__ void load(void *dst, const void *tma_map, uint64_t *bar, int32_t row, int32_t col) {
            static_assert(sizeof(void *) == sizeof(uint64_t));
            auto ptma = reinterpret_cast<uint64_t>(tma_map);
            auto pmbar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
            auto pdst = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];\n"
                :
                : "r"(pdst), "l"(ptma), "r"(pmbar), "r"(row), "r"(col)
                : "memory"
            );
        }

        template <const int32_t nb, typename t_dst, typename t_src>
        __device__ __forceinline__ void cg256(t_dst dst, t_src src) {
            asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        template <const int32_t nb, typename t_dst, typename t_src>
        __device__ __forceinline__ void cg64(t_dst dst, t_src src) {
            asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" :: "r"(dst), "l"(src), "n"(nb));
        }

        __device__ __forceinline__ void commit_group() {
            asm volatile("cp.async.commit_group;\n" ::: "memory");
        }

        template <const int32_t n>
        __device__ __forceinline__ void await_group() {
            asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
        }
    }

    template <typename T, bool TA, bool TB, int BM, int BN>
    __global__ static void matmul_kernel_wmma(
        int64_t M,
        int64_t N,
        int64_t K,
        int64_t batch_total,
        T *br,
        const T *bx,
        const T *by
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
        static constexpr int STAGES = 2;
        static constexpr int A_SIZE = BM*BK;
        static constexpr int B_SIZE = BK*BN;

        int batch = blockIdx.z;
        if (batch >= batch_total) return;
        int tile_m = blockIdx.y * BM;
        int tile_n = blockIdx.x * BN;
        int tid = threadIdx.x;
        int warp_id = tid>>5;

        const T *x_batch = bx + batch*M*K;
        const T *y_batch = by + batch*K*N;
        T *r_batch = br + batch*M*N;

        extern __shared__ uint8_t smem_raw[];

        auto *a_smem = reinterpret_cast<T *>(smem_raw);
        auto *b_smem = a_smem + STAGES*A_SIZE;
        auto *c_smem = reinterpret_cast<float *>(b_smem + STAGES*B_SIZE);

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        int warp_m = warp_id / WARPS_N;
        int warp_n = warp_id % WARPS_N;

        auto load_stage = [&](int stage, int k0) {
            auto *a_buf = a_smem + stage*A_SIZE;
            auto *b_buf = b_smem + stage*B_SIZE;

            #pragma unroll
            for (int i=tid; i < BM*BK; i += BLOCK_THREADS) {
                int row = i / BK;
                int col = i % BK;
                int g_m = tile_m + row;
                int g_k = k0 + col;
                a_buf[row*BK + col] = g_m < M && g_k < K ? x_batch[TA ? g_k*M + g_m : g_m*K + g_k] : T{};
            }

            #pragma unroll
            for (int i=tid; i < BK*BN; i += BLOCK_THREADS) {
                int n_local = i / BK;
                int k_local = i % BK;
                int g_k = k0 + k_local;
                int g_n = tile_n + n_local;
                b_buf[k_local + n_local*BK] = g_k < K && g_n < N ? y_batch[TB ? g_n*K + g_k : g_k*N + g_n] : T{};
            }
        };

        auto compute_stage = [&](int stage) {
            if (warp_id >= WARPS_PER_BLOCK) return;
            auto *a_buf = a_smem + stage*A_SIZE;
            auto *b_buf = b_smem + stage*B_SIZE;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
            const T *a_ptr = a_buf + (warp_m<<4)*BK;
            const T *b_ptr = b_buf + (warp_n<<4)*BK;
            wmma::load_matrix_sync(a_frag, a_ptr, BK);
            wmma::load_matrix_sync(b_frag, b_ptr, BK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
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
        const T *bx,
        const T *by,
        bool xT, bool yT
    ) {
        static_assert(std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>);

        static constexpr int BM = 64;
        static constexpr int BN = 64;
        static constexpr int BK = 16;
        static_assert(BK == 16);
        static constexpr int WARPS_M = BM>>4;
        static constexpr int WARPS_N = BN>>4;
        static constexpr int STAGES = 2;
        static constexpr int BLOCK_THREADS = WARPS_M * (WARPS_N<<5);

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

        dim3 grid_dim((N + BN - 1)/BN, (M + BM - 1)/BM,batch_total);
        dim3 block_dim(BLOCK_THREADS, 1, 1);
        if (!xT && !yT) {
            auto *kernel = matmul_kernel_wmma<T, false, false, BM, BN>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, bx, by);
        } else if (!xT && yT) {
            auto *kernel = matmul_kernel_wmma<T, false, true, BM, BN>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, bx, by);
        } else if (xT && !yT) {
            auto *kernel = matmul_kernel_wmma<T, true, false, BM, BN>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, bx, by);
        } else {
            auto *kernel = matmul_kernel_wmma<T, true, true, BM, BN>;
            set_kernel_smem_size(kernel, smem);
            kernel<<<grid_dim, block_dim, smem>>>(M, N, K, batch_total, br, bx, by);
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
        const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        const auto *by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));

        #if MAG_CUDA_MATMUL_USE_WMMA
            if constexpr (std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, half>) {
                launch_matmul_kernel_wmma(M, N, K, batch_total, br, bx, by, xT, yT);
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
