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
    template <typename T, bool TA, bool TB, int BM, int BN, int BK, int TM, int TN>
    __global__ static void matmul_kernel(
        int64_t M, int64_t N, int64_t K,
        int64_t batch_total,
        T *br, const T *bx, const T *by
    ) {
        static constexpr int A_SIZE = BM*BK;
        static constexpr int B_SIZE = BK*BN;

        extern __shared__ uint8_t smem[];

        auto *a_smem = reinterpret_cast<T *>(smem);
        auto *b_smem = reinterpret_cast<T *>(smem) + A_SIZE;

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
        for (int k0=0; k0 < K; k0 += BK) {
            #pragma unroll
            for (int i=tid; i < A_SIZE; i += nthreads) {
                int row = i / BK;
                int col = i % BK;
                int g_row = tile_m + row;
                int g_col = k0 + col;
                a_smem[i] = g_row < M && g_col < K ? bx[g_row*a_row_stride + g_col*a_col_stride] : T{};
            }
            #pragma unroll
            for (int i=tid; i < B_SIZE; i += nthreads) {
                int row = i / BN;
                int col = i % BN;
                int g_row = k0 + row;
                int g_col = tile_n + col;
                b_smem[i] = g_row < K && g_col < N ? by[g_row*b_row_stride + g_col*b_col_stride] : T{};
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

    enum class mat_layout_t {
        packed,
        packed_transposed,
        unsupported
    };

    struct mat_layout_info_t {
        mat_layout_t layout;
        bool batch_packed;
    };

    [[nodiscard]] static mat_layout_info_t detect_mat_layout_info(const mag_tensor_t *tensor) {
        mat_layout_info_t info{mat_layout_t::unsupported, false};
        int64_t rank = tensor->coords.rank;
        if (rank < 2) {
            info.layout = mat_layout_t::packed;
            info.batch_packed = true;
            return info;
        }
        int64_t rows = tensor->coords.shape[rank - 2];
        int64_t cols = tensor->coords.shape[rank - 1];
        int64_t srow = tensor->coords.strides[rank - 2];
        int64_t scol = tensor->coords.strides[rank - 1];

        if (scol == 1 && srow == cols) {
            info.layout = mat_layout_t::packed;
        } else if (srow == 1 && scol == rows) {
            info.layout = mat_layout_t::packed_transposed;
        } else {
            return info;
        }
        int64_t expected_batch_stride = rows * cols;
        if (rank == 2) {
            info.batch_packed = true;
            return info;
        }
        int64_t running = expected_batch_stride;
        for (int64_t d = rank-3; d >= 0; --d) {
            if (tensor->coords.strides[d] != running) {
                info.layout = mat_layout_t::unsupported;
                info.batch_packed = false;
                return info;
            }
            running *= tensor->coords.shape[d];
        }
        info.batch_packed = true;
        return info;
    }

    template <typename T>
    static void launch_matmul(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *y = cmd.in[1];

        mag_assert2(mag_tensor_is_contiguous(r));

        mat_layout_info_t xli = detect_mat_layout_info(x);
        mat_layout_info_t yli = detect_mat_layout_info(y);

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

        int64_t M  = x->coords.rank == 1 ? 1 : x->coords.shape[x->coords.rank - 2];
        int64_t Kx = x->coords.shape[x->coords.rank - 1];
        int64_t N  = y->coords.rank == 1 ? 1 : y->coords.shape[y->coords.rank - 1];
        int64_t Ky = y->coords.rank == 1 ? y->coords.shape[0] : y->coords.shape[y->coords.rank - 2];

        mag_assert2(Kx == Ky);
        int64_t K = Kx;

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

        int64_t blocks_x = (N + BN-1)/BN;
        int64_t blocks_y = (M + BM-1)/BM;
        dim3 grid_dim = dim3(blocks_x, blocks_y, batch_total);
        dim3 block_dim = dim3(TRX, TRY, 1);

        size_t smem_size = (BM*BK + BN*BK) * sizeof(T);
        mag_assert(smem_size <= (unsigned)max_smem_real, "Required shared memory size for matmul kernel exceeds device limit");
        auto set_kernel_smem_size = [&](auto kernel) -> void {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
        };

        if (!xT && !yT) {
            auto *kernel = matmul_kernel<T, false, false, BM, BN, BK, TM, TN>;
            set_kernel_smem_size(kernel);
            kernel<<<grid_dim, block_dim, smem_size>>>(M, N, K, batch_total, br, bx, by);
        } else if (!xT && yT) {
            auto *kernel = matmul_kernel<T, false, true, BM, BN, BK, TM, TN>;
            set_kernel_smem_size(kernel);
            kernel<<<grid_dim, block_dim, smem_size>>>(M, N, K, batch_total, br, bx, by);
        } else if (xT && !yT) {
            auto *kernel = matmul_kernel<T, true, false, BM, BN, BK, TM, TN>;
            set_kernel_smem_size(kernel);
            kernel<<<grid_dim, block_dim, smem_size>>>(M, N, K, batch_total, br, bx, by);
        } else {
            auto *kernel = matmul_kernel<T, true, true, BM, BN, BK, TM, TN>;
            set_kernel_smem_size(kernel);
            kernel<<<grid_dim, block_dim, smem_size>>>(M, N, K, batch_total, br, bx, by);
        }

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
}
