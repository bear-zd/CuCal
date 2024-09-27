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
    // #params unroll
    for(int stride = kWarpSize >> 1; stride >= 1; stride >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, stride);
    return val;
}
template<const int NUM_THREADS = 256>
__global__ void dot_product(const float* x, const float* y, float* output, int length){
    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;

    constexpr int NUM_WARPS = (NUM_THREADS - 1) / WARP_SIZE + 1;
    // equals to '''int num_warps = (blockDim.x - 1 + WARP_SIZE) / WARP_SIZE; '''
    __shared__ float block_sum[NUM_WARPS];
    float v = global_tid < length ? x[global_tid] * y[global_tid] : 0.0f;

    int warp_id = local_tid / WARP_SIZE;
    int lane = local_tid % WARP_SIZE;
    v = warp_shffl_sum<WARP_SIZE>(v);

    if (lane == 0) block_sum[warp_id] = v;
    __syncthreads();
    // do shared memory operate with sync

    //use the first warp to compute
    float prod = lane < NUM_WARPS ? block_sum[lane] : 0.0f;
    if (warp_id == 0) prod = warp_shffl_sum<WARP_SIZE>(prod);
    if (local_tid == 0) atomicAdd(output, prod);
}


torch::Tensor launch_dot_product_kernel_fp32(torch::Tensor x, torch::Tensor y){
    CHECK_INPUT(x);
    CHECK_INPUT(y);                                         
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    const int N = x.numel();

    auto output = torch::empty({1}, options);
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    dot_product<256><<<grid, block>>>(x.data_ptr<float>(), y.data_ptr<float>(), output.data_ptr<float>(), N);
    CUDA_ERR(cudaGetLastError());
    CUDA_ERR(cudaDeviceSynchronize());
    return output;
}