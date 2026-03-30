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

#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>

namespace mag {
    // In order
    // https://siboehm.com/articles/22/CUDA-MMM
    // https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html
    // https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog
    // https://gau-nernst.github.io/tcgen05/
    template <typename T, int BM, int BN, int BK, int TM, int TN>
    __global__ static void matmul_kernel_2d(
        int64_t M, int64_t N, int64_t K,
        int64_t batch_total,
        T *br, const T *bx, const T *by
    ) {
        static constexpr int A_SIZE = BM*BK;
        static constexpr int B_SIZE = BK*BN;

        extern __shared__ uint8_t smem[];

        // MxK @ KxN -> MxN
        // Tensor.uniform(2, 8, device='cuda') @ Tensor.uniform(8, 2, device='cuda')

        T *a_smem = reinterpret_cast<T *>(smem);
        T *b_smem = reinterpret_cast<T *>(smem) + A_SIZE;

        int batch = blockIdx.z;
        if (batch >= batch_total) return;

        bx += batch*M*K;
        by += batch*K*N;
        br += batch*M*N;

        int tile_m = blockIdx.y * BM;
        int tile_n = blockIdx.x * BN;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tid = threadIdx.y*blockDim.x + threadIdx.x;
        int nthreads = blockDim.x*blockDim.y;

        int local_m0 = ty * TM;
        int local_n0 = tx * TN;

        float acc[TM][TN] = {};
        for (int k0=0; k0 < K; k0 += BK) {
            #pragma unroll
            for (int i=tid; i < A_SIZE; i += nthreads) {
                int row = i / BK;
                int col = i % BK;
                int g_row = tile_m + row;
                int g_col = k0 + col;
                a_smem[i] = g_row < M && g_col < K ? bx[g_row*K + g_col] : T{};
            }
            #pragma unroll
            for (int i=tid; i < B_SIZE; i += nthreads) {
                int row = i / BN;
                int col = i % BN;
                int g_row = k0 + row;
                int g_col = tile_n + col;
                b_smem[i] = g_row < K && g_col < N ? by[g_row*N + g_col] : T{};
            }
            __syncthreads();
            #pragma unroll
            for (int kk=0; kk < BK; ++kk) {
                float a_frag[TM];
                float b_frag[TN];
                #pragma unroll
                for (int i=0; i < TM; ++i) {
                    a_frag[i] = static_cast<float>(a_smem[(local_m0 + i)*BK + kk]);
                }
                #pragma unroll
                for (int i=0; i < TN; ++i) {
                    b_frag[i] = static_cast<float>(b_smem[kk*BN + (local_n0 + i)]);
                }
                #pragma unroll
                for (int i=0; i < TM; ++i) {
                    #pragma unroll
                    for (int j=0; j < TN; ++j) {
                        acc[i][j] += a_frag[i] * b_frag[j];
                    }
                }
            }
            __syncthreads();
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
    static void launch_matmul(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *y = cmd.in[1];

        mag_contiguous(nullptr, &x, x);
        mag_contiguous(nullptr, &y, y);

        int64_t M = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank-2];
        int64_t N = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank-1];
        int64_t K = x->coords.shape[x->coords.rank-1];
        int64_t batch_rank = r->coords.rank > 2 ? r->coords.rank-2 : 0;
        int64_t batch_total = std::accumulate(r->coords.shape, r->coords.shape + batch_rank, 1, std::multiplies<int64_t>());

        auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
        const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        const auto *by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));

        int max_smem_real;
        int device;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&max_smem_real, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

        static constexpr int BM = 64;
        static constexpr int BN = 64;
        static constexpr int BK = 32;
        static constexpr int TM = 4;
        static constexpr int TN = 4;
        static constexpr int TRX = BN/TN;
        static constexpr int TRY = BM/TM;
        static_assert(TRX*TRY <= 1024);

        auto &kernel = matmul_kernel_2d<T, BM, BN, BK, TM, TN>;

        size_t smem_size = (BM*BK + BN*BK) * sizeof(T);
        mag_assert(smem_size <= (unsigned)max_smem_real, "Required shared memory size for matmul kernel exceeds device limit");
        cudaFuncSetAttribute(&kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
        int64_t blocks_x = (N + BN-1)/BN;
        int64_t blocks_y = (M + BM-1)/BM;
        dim3 grid_dim = dim3(blocks_x, blocks_y, batch_total);
        dim3 block_dim = dim3(TRX, TRY, 1);
        kernel<<<grid_dim, block_dim, smem_size>>>(M, N, K, batch_total, br, bx, by);

        mag_tensor_decref(x);
        mag_tensor_decref(y);
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
}
