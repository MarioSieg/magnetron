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
    template <typename T, int BM, int BN, int BK>
    __global__ static void matmul_kernel_2d(
        int64_t M, int64_t N, int64_t K,
        int64_t batch_total,
        T *br, const T *bx, const T *by
    ) {
        static constexpr int A_SIZE = BM*BK;
        static constexpr int B_SIZE = BN*BK;

        extern __shared__ uint8_t smem[];

        // MxK @ KxN -> MxN
        // Tensor.uniform(2, 8, device='cuda') @ Tensor.uniform(8, 2, device='cuda')

        T *a_smem = reinterpret_cast<T *>(smem);
        T *b_smem = reinterpret_cast<T *>(smem) + BM*BK;

        int batch = blockIdx.z;
        if (batch >= batch_total) return;

        bx += batch*M*K;
        by += batch*K*N;
        br += batch*M*N;

        int o_row = blockIdx.x*BM;
        int o_col = blockIdx.y*BN;
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tid = threadIdx.y*blockDim.x + threadIdx.x;
        int nthreads = blockDim.x*blockDim.y;

        // Make this 2D
        // GMEM -> SMEM[TILE] -> REG[TILE]
        float acc = 0.f;

        for (int k=0; k < K; k += BK) {
            #pragma unroll
            for (int i=tid; i < A_SIZE; i += nthreads) {
                int g_row = o_row + i/BK;
                int g_col = i%BK + k;
                // it goes GMEM -> REG -> SMEM
                // you can do cp.async on newerish arch ampere+(?)
                // you can load in float4
                a_smem[i] = g_row < M && g_col < K ? bx[g_row*K + g_col] : T{};
            }
            // Shared memory swizzling
            #pragma unroll
            for (int i=tid; i < B_SIZE; i += nthreads) {
                int g_row = k + i/BN;
                int g_col = i%BN + o_col;
                b_smem[i] = g_row < K && g_col < N ? by[g_row*N + g_col] : T{};
            }
            // can be double buffered
            // 2 SMEM tiles
            // You load to tile 1
            // load tile 2
            // you compute on tile 1
            // load to tile 1
            // compute on tile 2
            // loop
            __syncthreads();
            // Do ACC with tensor cores
            // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions
            // MMA not WGMMA
            for (int i=0; i < BK; i++) {
                acc += static_cast<float>(a_smem[tx*BK + i]) * static_cast<float>(b_smem[i*BN + ty]);
            }
            __syncthreads();
        }
        if (o_row + tx < M && o_col + ty < N) {
            br[(o_row+tx)*N + o_col + ty] = acc;
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

        static constexpr int BK = 32;
        static constexpr int BM = 32;
        static constexpr int BN = 32;

        size_t smem_size = (BM*BK + BN*BK) * sizeof(T);
        mag_assert(smem_size <= (unsigned)max_smem_real, "Required shared memory size for matmul kernel exceeds device limit");
        cudaFuncSetAttribute(matmul_kernel_2d<T, BM, BN, BK>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        int64_t blocks_x = (M + BM-1)/BM;
        int64_t blocks_y = (N + BN-1)/BN;
        dim3 grid_dim = dim3(blocks_x, blocks_y, batch_total);
        dim3 block_dim = dim3(BM, BN, 1);
        matmul_kernel_2d<T, BM, BN, BK><<<grid_dim, block_dim, smem_size>>>(M, N, K, batch_total, br, bx, by);

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
