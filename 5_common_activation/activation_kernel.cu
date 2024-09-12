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

__global__ void relu_kernel(const float* x, float* output, const int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        float t = x[idx];
        if (t < 0) 
            output[idx] = 0;
        else 
            output[idx] = t;
    }
}

__global__ void tanh_kernel(const float* x, float* output, const int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) 
        output[idx] = (expf(x[idx]) - expf(-x[idx])) / (expf(x[idx]) + expf(-x[idx]));
}


__global__ void sigmoid_kernel(const float* x, float* output, const int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) 
        output[idx] = 1 / (1 + expf(-x[idx]));
}

#define activation_function_create(activation_name)  \
torch::Tensor activation_name##_function(torch::Tensor x) {\
    CHECK_INPUT(x);                                         \
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());\
    const int N = x.numel();\
    auto output = torch::empty({N}, options);\
    dim3 block(256);\
    dim3 grid((N + block.x - 1) / block.x);\
    activation_name##_kernel<<<grid, block>>>(x.data_ptr<float>(), output.data_ptr<float>(), N);\
    CUDA_ERR(cudaGetLastError());\
    CUDA_ERR(cudaDeviceSynchronize());\
    return output;\
}

activation_function_create(relu);
activation_function_create(sigmoid);
activation_function_create(tanh);

