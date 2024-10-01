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
#define WARP_SIZE 32
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val){
    #pragma unroll
    for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    return val;
}

template<const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float v){
    const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    v = warp_reduce_sum_f32<WARP_SIZE>(v);
    if(lane == 0) shared[wid] = v;
    __syncthreads();
    v = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    v = warp_reduce_sum_f32<NUM_WARPS>(v);
    return v;
}

template< const int NUM_THREADS=256>
__global__ void rmsnorm_f32_kernel(float* x, float* y,float* w, int N, int k){
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    int block_id = blockIdx.x;

    __shared__ float sum_sqx;
    float value = (global_tid < N * k) ? x[global_tid] : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value * value);
    if (local_tid == 0) sum_sqx = sum;
    __syncthreads();
    y[global_tid] = global_tid < N * k ? w[local_tid] * value / sqrtf(sum_sqx / k) : 0.0f;
}
torch::Tensor launch_rmsnorm_kernel_fp32(torch::Tensor x, torch::Tensor w) {
    CHECK_INPUT(x);    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor y = torch::empty_like(x, options);
    int batch_size = x.size(0);
    int seqlen = x.size(1);
    int hidden_size = x.size(2);
    int N = batch_size * seqlen;
    int K = hidden_size;
    rmsnorm_f32_kernel<768><<<N, 768>>>(x.data_ptr<float>(), y.data_ptr<float>(), w.data_ptr<float>(), N, K);
    return y;
}