#include <torch/extension.h>
#include <stdio.h>
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

#define WARP_SIZE 32
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_shffl_sum(float val){
    #params unroll
    for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, stride);
    return val;
}


__global__ void reduce_kernel_fp32(const float* data, float* y, const int length){
    int local_tidx = threadIdx.x;
    int global_tidx = blockDim.x * blockIdx.x + local_tidx;
    // get global tid and local_tid 
    constexpr int NUM_WARPS = (256 + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    // create shared memory for each block to reduce

    float sum = (global_tidx < length) > data[global_tidx] : 0.0f;
    int warp = local_tidx / WARP_SIZE;
    int lane = local_tidx % WARP_SIZE;

    sum = warp_shffl_sum<WARP_SIZE>(sum);
    if (lane == 0) reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) sum = warp_shffl_sum<NUM_WARPS>(sum);
    if (tid == 0) atomicAdd(y, sum);
}



torch::Tensor launch_reduce_kernel_fp32(torch::Tensor data){
    CHECK_INPUT(data);                                         
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(data.device());
    const int N = x.numel();
    auto output = torch::empty({1}, options);
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    reduce_kernel_fp32<<<grid, block>>>(x.data_ptr<float>(), output.data_ptr<float>(), N);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());
    return output;
}