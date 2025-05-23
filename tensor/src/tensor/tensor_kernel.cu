#include "tensor/tensor.h"
#include "tensor/tensor_kernel.h"
#include "dtype/dtype.h"
#include "mem/mem_util.h"
#include <cassert>
#include <iostream>

namespace {
    constexpr int TILE = 16;
    constexpr int BLOCKDIM_X=TILE, BLOCKDIM_Y=TILE;
    constexpr int THREADS = BLOCKDIM_X*BLOCKDIM_Y;
    constexpr int GRID_BOUND = 65535;

    struct LaunchConfig {
        dim3 grid;
        dim3 block;
    };
    inline LaunchConfig _1d_config(int64_t n) {
        return {{std::min((n+THREADS-1)/THREADS, (int64_t)GRID_BOUND), 1, 1},
                {THREADS, 1, 1}};
    }
    inline LaunchConfig _2d_config(int64_t m, int64_t n) {
        return {{std::min((n+(int64_t)BLOCKDIM_X-1)/BLOCKDIM_X, (int64_t)GRID_BOUND),
                 std::min((m+(int64_t)BLOCKDIM_Y-1)/BLOCKDIM_Y, (int64_t)GRID_BOUND), 1},
                {BLOCKDIM_X, BLOCKDIM_Y, 1}};
    }
    inline LaunchConfig _3d_config(int64_t b, int64_t m, int64_t n) {
        return {{std::min((m+BLOCKDIM_Y-1)/BLOCKDIM_Y, (int64_t)GRID_BOUND), 
                 std::min((n+BLOCKDIM_X-1)/BLOCKDIM_X, (int64_t)GRID_BOUND),
                 std::min(b, (int64_t)GRID_BOUND)},
                {BLOCKDIM_Y, BLOCKDIM_X, 1}};
    }
}
namespace cuda_kernel {

    template<typename Scalar>
    __global__ void add_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t n){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i<n) {
            out[i] = a[i] + b[i];
            i += gridDim.x*blockDim.x;
        }
    }

    template<typename Scalar>
    __global__ void mul_kernel(const Scalar *a, const Scalar *b, Scalar *out, const int64_t n){
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i<n) {
            out[i] = a[i] * b[i];
            i += gridDim.x * blockDim.x;
        }
    }

    template<typename Scalar>
    __global__ void eq_kernel(const Scalar *a, const Scalar *b, bool *out, const int64_t n){

        __shared__ bool cache[THREADS];
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        bool local_res = true;

        while (i<n) {
            if (a[i]!=b[i]) {
                local_res = false;
                break;
            }
            i += gridDim.x*blockDim.x;
        }
        cache[threadIdx.x] = local_res;
        __syncthreads();

        for (int s=blockDim.x/2; s>0; s>>=1) {
            if (threadIdx.x<s) {
                cache[threadIdx.x] =
                    cache[threadIdx.x] && cache[threadIdx.x+s];
            } 
            __syncthreads();
        }

        if (threadIdx.x==0) {
            atomicAnd(reinterpret_cast<int*>(out),
                      static_cast<int>(cache[0]));
        }
    }

    template<typename SrcScalar, typename DstScalar>
    __global__ void cast_kernel(const SrcScalar *src, DstScalar *dst, const int64_t n) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i<n) {
            dst[i] = static_cast<DstScalar>(src[i]);
            i += gridDim.x*blockDim.x;
        }
    }

    template<typename Scalar>
    __global__ void dot_kernel(
        const Scalar *a, const Scalar *b, Scalar *out,
        const int64_t K
    ) {
        __shared__ Scalar cache[THREADS];
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        Scalar dot = 0;
        // Grid-stride loop 
        while (i < K) {
            dot += a[i] * b[i];
            i += blockDim.x * gridDim.x;
        }
        cache[threadIdx.x] = dot;
        __syncthreads();

        // Reduction in shared memory
        for (int s=blockDim.x/2; s>0; s>>=1) {
            if (threadIdx.x < s) {
                cache[threadIdx.x] += cache[threadIdx.x+s];
            }
            __syncthreads();
        }

        if (threadIdx.x==0) {
            atomicAdd(out, cache[0]);
        }
    }

    template<typename Scalar>
    __global__ void mm_kernel(
        const Scalar *a, const Scalar *b, Scalar *out,
        const int64_t M, const int64_t K, const int64_t N
    ) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        for (int ii = i; ii < M; ii += gridDim.y * blockDim.y) {
            for (int jj = j; jj < N; jj += gridDim.x * blockDim.x) {
                Scalar s = 0;
                for (int k = 0; k < K; ++k) {
                    s += a[ii * K + k] * b[k * N + jj];
                }
                out[ii * N + jj] = s;
            }
        }
    }

    template<typename Scalar>
    __global__ void mm_kernel_tiled(
        const Scalar *a, const Scalar *b, Scalar *out,
        const int64_t M, const int64_t K, const int64_t N
    ) {
        __shared__ Scalar tile_a[TILE][TILE];
        __shared__ Scalar tile_b[TILE][TILE];

        for (int block_i=blockIdx.y; block_i*TILE<M; block_i+=gridDim.y) {
            for (int block_j=blockIdx.x; block_j*TILE<N; block_j+=gridDim.x) {
                
                int i= block_i*TILE + threadIdx.y;
                int j= block_j*TILE + threadIdx.x;

                Scalar s = 0;
                int NUM_TILE = (K+TILE-1)/TILE;
                for (int kk=0; kk<NUM_TILE; ++kk) {
                    int a_j = kk*TILE + threadIdx.x;
                    int b_i = kk*TILE + threadIdx.y;

                    if (i < M && a_j < K)
                        tile_a[threadIdx.y][threadIdx.x] = a[i * K + a_j];
                    else
                        tile_a[threadIdx.y][threadIdx.x] = 0;

                    if (j < N && b_i < K)
                        tile_b[threadIdx.y][threadIdx.x] = b[b_i * N + j];
                    else
                        tile_b[threadIdx.y][threadIdx.x] = 0;

                    __syncthreads();

                    for (int k = 0; k < TILE; ++k) {
                        s += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
                    }
                    __syncthreads();
                }
                if (i < M && j < N) {
                    out[i * N + j] = s;
                }
            }
        }
    }

    template<typename Scalar>
    __global__ void bmm_kernel(
        const Scalar *a, const Scalar *b, Scalar *out,
        const int64_t B, const int64_t M, const int64_t K, const int64_t N
    ) {
        // Grid-stride over batch
        for (int batch = blockIdx.z; batch < B; batch += gridDim.z) {
            for (int block_i = blockIdx.y; block_i * blockDim.y < M; block_i += gridDim.y) {
                for (int block_j = blockIdx.x; block_j * blockDim.x < N; block_j += gridDim.x) {

                    int i = block_i * blockDim.y + threadIdx.y;
                    int j = block_j * blockDim.x + threadIdx.x;

                    if (i >= M || j >= N) continue;

                    const Scalar* a_batch = a + batch * M * K;
                    const Scalar* b_batch = b + batch * K * N;
                    Scalar* out_batch = out + batch * M * N;

                    Scalar s = 0;
                    for (int k = 0; k < K; ++k) {
                        s += a_batch[i * K + k] * b_batch[k * N + j];
                    }

                    out_batch[i * N + j] = s;
                }
            }
        }
    }

    template <typename Scalar>
    __global__ void bmm_kernel_tiled(
        const Scalar* __restrict__ a,
        const Scalar* __restrict__ b,
        Scalar* __restrict__ out,
        const int64_t B, const int64_t M, const int64_t K, const int64_t N
    ) {
        __shared__ Scalar tile_a[TILE][TILE];
        __shared__ Scalar tile_b[TILE][TILE];

        // grid stride
        for (int64_t batch=blockIdx.z; batch<B; batch+=gridDim.z) {

            const Scalar *a_ptr = a + batch*(M*K);
            const Scalar *b_ptr = b + batch*(K*N);
            Scalar *out_ptr = out + batch*(M*N);

            for (int64_t block_i=blockIdx.y; block_i*TILE<M; block_i+=gridDim.y) {
                for (int64_t block_j=blockIdx.x; block_j*TILE<N; block_j+=gridDim.x) {

                    int i = block_i * blockDim.y + threadIdx.y;
                    int j = block_j * blockDim.x + threadIdx.x;
                    
                    Scalar s = 0;
                    int64_t NUM_TILES = (K+TILE-1)/TILE;
                    for (int64_t kk=0; kk<NUM_TILES; ++kk) {

                        int64_t a_j = kk*TILE + threadIdx.x;
                        int64_t b_i = kk*TILE + threadIdx.y;

                        if (i<M && a_j<K) {
                            tile_a[threadIdx.y][threadIdx.x] = a_ptr[i*K+a_j];
                        } else {
                            tile_a[threadIdx.y][threadIdx.x] = 0;
                        }
                        if (j<N && b_i<K) {
                            tile_b[threadIdx.y][threadIdx.x] = b_ptr[b_i*N+j];
                        } else {
                            tile_b[threadIdx.y][threadIdx.x] = 0;
                        }
                        __syncthreads();

                        for (int k=0; k<TILE; ++k) {
                            s += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
                        }
                        __syncthreads();
                    }
                    if (i<M && j<N) {
                        out_ptr[i*N+j] = s;
                    }
                }
            }
        }
    }
} // namespace anonymous

namespace tensor_kernel {
namespace cuda {
    template<typename Scalar>
    void launch_add_kernel(const Tensor &a, const Tensor &b, Tensor &out){
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();

        auto cfg = _1d_config(out.numel());      
        cuda_kernel::add_kernel<Scalar><<<cfg.grid, cfg.block>>>(a_ptr, b_ptr, out_ptr, out.numel());
        cudaDeviceSynchronize();
    }
    template<typename Scalar>
    void launch_mul_kernel(const Tensor &a, const Tensor &b, Tensor &out){
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();
        
        auto cfg = _1d_config(out.numel());
        cuda_kernel::mul_kernel<Scalar><<<cfg.grid, cfg.block>>>(a_ptr, b_ptr, out_ptr, out.numel());
        cudaDeviceSynchronize();
    }
    template<typename Scalar>
    void launch_eq_kernel(const Tensor &a, const Tensor &b, bool &out){
        
        // `out` is a variable that store at host memory

        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();

        bool *out_ptr;
        cudaMalloc(&out_ptr, sizeof(bool));
        cudaMemset(out_ptr, true, sizeof(bool));

        auto cfg = _1d_config(a.numel());
        cuda_kernel::eq_kernel<Scalar><<<cfg.grid, cfg.block>>>(a_ptr, b_ptr, out_ptr, a.numel());

        cudaMemcpy(&out, out_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(out_ptr);
        cudaDeviceSynchronize();
    }
    template<typename SrcScalar, typename DstScalar>
    void launch_cast_kernel(const Tensor &src, Tensor &dst){
        const SrcScalar *src_ptr = src.data_as<SrcScalar>();
        DstScalar *dst_ptr = dst.data_as<DstScalar>();

        auto cfg = _1d_config(src.numel());
        cuda_kernel::cast_kernel<SrcScalar, DstScalar><<<cfg.grid, cfg.block>>>(src_ptr, dst_ptr, src.numel());
        cudaDeviceSynchronize();
    }
    template<typename Scalar>
    void launch_dot_kernel(const Tensor &a, const Tensor &b, Tensor &out) {
        const int64_t K = a.size(-1); 
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();

        auto cfg = _1d_config(a.numel());
        cuda_kernel::dot_kernel<Scalar><<<cfg.grid, cfg.block>>>(a_ptr, b_ptr, out_ptr, K);
        cudaDeviceSynchronize();
    }
    template<typename Scalar>
    void launch_mm_kernel(const Tensor &a, const Tensor &b, Tensor &out) {
        const int64_t M=a.size(-2), K=a.size(-1), N=b.size(-1); 
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();

        auto cfg = _2d_config(M, N);
        cuda_kernel::mm_kernel<Scalar><<<cfg.grid, cfg.block>>>(a_ptr, b_ptr, out_ptr, M, K, N);

        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     std::cerr << "CUDA kernel launch failed: "
        //               << cudaGetErrorString(err) << std::endl;
        // }

        cudaDeviceSynchronize();
    }
    template<typename Scalar>
    void launch_bmm_kernel(const Tensor &a, const Tensor &b, Tensor &out) {
        const int64_t B=a.size(-3), M=a.size(-2), K=a.size(-1), N=b.size(-1); 
        const Scalar *a_ptr = a.data_as<Scalar>();
        const Scalar *b_ptr = b.data_as<Scalar>();
        Scalar *out_ptr = out.data_as<Scalar>();

        auto cfg = _3d_config(B, M, N);
        cuda_kernel::bmm_kernel<Scalar><<<cfg.grid, cfg.block>>>(a_ptr, b_ptr, out_ptr, B, M, K, N);
        cudaDeviceSynchronize();
    }
} // namespace cuda
} // namespace tensor_kernel

// init
#define INSTANTIATION(scalar_type, cpp_scalar, ...) \
template void tensor_kernel::cuda::launch_add_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);                       \
template void tensor_kernel::cuda::launch_mul_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);                       \
template void tensor_kernel::cuda::launch_eq_kernel<cpp_scalar>(  \
    const Tensor&, const Tensor&, bool&);                         
FOR_EACH_SCALAR_TYPE(INSTANTIATION)
#undef INSTANTIATION

#define INSTANTIATION(scalar_type, cpp_scalar, ...) \
template void tensor_kernel::cuda::launch_dot_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);                       \
template void tensor_kernel::cuda::launch_mm_kernel<cpp_scalar>(  \
    const Tensor&, const Tensor&, Tensor&);                       \
template void tensor_kernel::cuda::launch_bmm_kernel<cpp_scalar>( \
    const Tensor&, const Tensor&, Tensor&);
MATMUL_SCALAR_TYPE(INSTANTIATION)
#undef INSTANTIATION

#define INSTANTIATION(src_scalar_type, src_scalar, dst_scalar_type, dst_scalar) \
template void tensor_kernel::cuda::launch_cast_kernel<src_scalar, dst_scalar>( \
    const Tensor&, Tensor&);
FOR_EACH_SCALAR_TYPE_PAIR(INSTANTIATION)
#undef INSTANTIATION