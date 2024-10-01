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
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#define BLOCKDIM 16
#define BLOCK_DIM 16
struct __align__(16) float4 {
    float x, y, z, w;
};
__global__ void mat_transpose_naive(float* x, float* out, int cols, int rows){
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (row_idx < rows && col_idx < cols){
        unsigned int idx = col_idx + row_idx * cols;
        unsigned int out_idx = row_idx + col_idx * rows;
        out[out_idx] = x[idx];
    }
}

__global__ void mat_transpose_shared(float* x, float* out, int cols, int rows){
    unsigned int local_row_idx = threadIdx.x;
    unsigned int local_col_idx = threadIdx.y;
    unsigned int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float tile[BLOCKDIM][BLOCKDIM];
    
    if (row_idx < rows && col_idx < cols){
        unsigned int idx = col_idx + row_idx * cols;
        tile[local_col_idx][local_row_idx] = x[idx];
    }
    __syncthreads();
    if (row_idx < rows && col_idx < cols){
        unsigned int idx = row_idx + col_idx * rows;
        out[idx] = tile[local_col_idx][local_row_idx];
    }
}

__global__ void mat_transpose_shared_pack(float* x, float* out, int cols, int rows){
    unsigned int local_row_idx = threadIdx.x;
    unsigned int local_col_idx = threadIdx.y;
    unsigned int row_idx = blockIdx.x * blockDim.x / 4 + threadIdx.x;
    unsigned int col_idx = blockIdx.y * blockDim.y / 4+ threadIdx.y;
    __shared__ float tile[BLOCKDIM][BLOCKDIM];
    
    if (row_idx < rows && col_idx < cols){
        unsigned int idx = col_idx + row_idx * cols;
        tile[local_col_idx][local_row_idx] = x[idx];
    }
    __syncthreads();
    if (row_idx < rows && col_idx < cols){
        unsigned int idx = row_idx + col_idx * rows;
        out[idx] = tile[local_col_idx][local_row_idx];
    }
}


#define MAT_TRANSPOSE_CREATE(kernel_function_name) \
torch::Tensor mat_transpose_##kernel_function_name(torch::Tensor x){       \
    CHECK_INPUT(x);      \
    int rows = x.size(0);\
    int cols = x.size(1);\
    int grid_x = (cols + BLOCKDIM - 1) / BLOCKDIM;\
    int grid_y = (rows + BLOCKDIM - 1) / BLOCKDIM;\
    dim3 grid(grid_x, grid_y);\
    dim3 block(BLOCKDIM, BLOCKDIM);\
    torch::Tensor out = torch::empty({cols, rows}, x.options());\
    float* x_ptr = x.data_ptr<float>();\
    float* out_ptr = out.data_ptr<float>();\
    kernel_function_name<<<grid, block>>>(x_ptr, out_ptr, cols, rows);\
    return out;\
}
#define MAT_TRANSPOSE_CREATE_PACK(kernel_function_name) \
torch::Tensor mat_transpose_##kernel_function_name(torch::Tensor x){       \
    CHECK_INPUT(x);      \
    int rows = x.size(0);\
    int cols = x.size(1);\
    int grid_x = (cols + (BLOCKDIM*4) - 1) / (BLOCKDIM*4);\
    int grid_y = (rows + (BLOCKDIM*4) - 1) / (BLOCKDIM*4);\
    dim3 grid(grid_x, grid_y);\
    dim3 block(BLOCKDIM, BLOCKDIM);\
    torch::Tensor out = torch::empty({cols, rows}, x.options());\
    float* x_ptr = x.data_ptr<float>();\
    float* out_ptr = out.data_ptr<float>();\
    kernel_function_name<<<grid, block>>>(x_ptr, out_ptr, cols, rows);\
    return out;\
}

MAT_TRANSPOSE_CREATE(mat_transpose_naive)
MAT_TRANSPOSE_CREATE(mat_transpose_shared)
MAT_TRANSPOSE_CREATE_PACK()