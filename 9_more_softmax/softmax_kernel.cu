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

struct __align__(8) MD
{
    float m;
    float d;
}; 


template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value;
        
        value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
        value.m = bigger_m.m;
    }
    return value;
}


template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_shffl_sum(float val){
    #pragma unroll
    for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    return val;
}

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
  #pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val){
    constexpr int NUM_WARPS = (NUM_THREADS - 1 + WARP_SIZE)/ WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
    
    float t_val = warp_shffl_sum<WARP_SIZE>(val);
    if(lane == 0) shared[warp] = t_val;
    __syncthreads();

    t_val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    t_val = warp_shffl_sum<NUM_WARPS>(t_val);
    t_val = __shfl_sync(0xffffffff, t_val, 0, 32);
    return t_val;
}


template<const int NUM_THREADS=256>
__device__ float block_reduce_max_f32(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  static __shared__ float shared[NUM_WARPS];
  
  float value = warp_reduce_max_f32<WARP_SIZE>(val);
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
  value = warp_reduce_max_f32<NUM_WARPS>(value);
  // WRAN: need to broadcast value to all threads within warp
  value = __shfl_sync(0xffffffff, value, 0, 32);
  return value;
}



template<const int NUM_THREADS = 256>
__global__ void safe_softmax_kernel(const float* x, float* y, float* total,int length){

    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
    float val = global_tid < length ? x[global_tid] : -FLT_MAX;
    float max_val = block_reduce_max_f32<NUM_THREADS>(val);
    float exp_val = global_tid < length ? expf(x[global_tid] - max_val) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (local_tid == 0) {
      atomicAdd(total, exp_sum);
    }
    __threadfence(); 
    
    if (global_tid < length) y[global_tid] = exp_val / (*total); 
}


template<const int NUM_THREADS = 256 >
__global__ void online_softmax_kernel(const float* x, float* y, int length) {
  
    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
    const int WAPR_NUM = NUM_THREADS / WARP_SIZE;
    int warp_id = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;
    MD val;
    val.m = global_tid < length ? x[global_tid] : -FLT_MAX;
    val.d = global_tid < length ? 1.0f : 0.0f;

    __shared__ MD shared[ WAPR_NUM ]; 
    MD res = warp_reduce_md_op<WARP_SIZE>(val);

    if (lane_id == 0) shared[warp_id] = res; 
    __syncthreads();

    if (local_tid < WARP_SIZE) {
        MD block_res = shared[local_tid];
        block_res = warp_reduce_md_op<WAPR_NUM>(block_res); 
        if (local_tid == 0) {
            shared[0] = block_res; 
        }
    }
    __syncthreads();

    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);
    if (global_tid < length) {
        y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
    }
}



torch::Tensor launch_softmax_kernel_fp32(torch::Tensor x){
    CHECK_INPUT(x);    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    const int N = x.numel();

    auto output = torch::zeros({N}, options);
    auto total = torch::zeros({1}, options);
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    safe_softmax_kernel<256><<<grid, block>>>(x.data_ptr<float>(), output.data_ptr<float>(), total.data_ptr<float>(), N);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());
    return output;
}

torch::Tensor launch_online_softmax_kernel_fp32(torch::Tensor x){
    CHECK_INPUT(x);    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    const int N = x.numel();

    auto output = torch::zeros({N}, options);
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    online_softmax_kernel<256><<<grid, block>>>(x.data_ptr<float>(), output.data_ptr<float>(), N);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());
    return output;
}