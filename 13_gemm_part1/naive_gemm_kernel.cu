#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define TILE_SIZE 32
#define BLOCK_SIZE 32

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void gemm_naive(float* x, float* y, float* out, int m, int n, int k){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < m && col < n){
        float sum = 0.0f;
        for(int i = 0; i < k; i++){
            sum += x[row * k + i] * y[i * n + col];
        }
        out[row * n + col] = sum;
    }
}

__global__ void gemm_shared(float* x, float* y, float* out, int M, int N, int K){
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float shared_x[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_y[TILE_SIZE][TILE_SIZE];
    unsigned int length_bk = (K + TILE_SIZE - 1) / TILE_SIZE;
    float sum = 0.0f;
    for(int sblock_idx = 0; sblock_idx < length_bk; sblock_idx++){
        // load data into shared memory
        shared_x[ty][tx] = (row < M && (sblock_idx * TILE_SIZE + tx) < K) ? x[row * K + sblock_idx * TILE_SIZE + tx]:0.0f;
        shared_y[ty][tx] = (col < N && (sblock_idx * TILE_SIZE + ty) < K) ? y[(sblock_idx * TILE_SIZE + ty) * N + col]:0.0f;
        __syncthreads();
        #pragma unroll
        for(int i = 0; i < TILE_SIZE; i++){
            sum += shared_x[ty][i] * shared_y[i][tx];
        }
        __syncthreads();
    }
    if(row < M && col < N) out[row * N + col] = sum;
}



torch::Tensor gemm(torch::Tensor x, torch::Tensor y){
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    int m = x.size(0);
    int n = y.size(1);
    int k = x.size(1);
    int grid_x = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_y = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_x, grid_y);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    torch::Tensor z = torch::empty({m, n}, x.options());
    float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    float* z_ptr = z.data_ptr<float>();
    gemm_shared<<<grid, block>>>(x_ptr, y_ptr, z_ptr, m, n, k);
    return z;
}