// #include <torch/extension.h>
// #include <stdio.h>
// #include <c10/cuda/CUDAException.h>

// #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) \
//     CHECK_CUDA(x);     \
//     CHECK_CONTIGUOUS(x)
// #define CUDA_ERR(ans)                         \
//     {                                         \
//         gpuAssert((ans), __FILE__, __LINE__); \
//     }
// inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
// {
//     if (code != cudaSuccess)
//     {
//         fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//         if (abort)
//             exit(code);
//     }
// }
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
  }

__global__ void rgb2gray(unsigned char* input, unsigned char* output, const int width, const int height)
{
    int xidx = blockDim.x * blockIdx.x + threadIdx.x;
    int yidx = blockDim.y * blockIdx.y + threadIdx.y;

    if (xidx < width && yidx < height)
    {
        // output[yidx * width + xidx] = x1 * input[yidx * width + xidx] + x2 * input[yidx * width + xidx + width * height] + x3 * input[yidx * width + xidx + 2 * width * height];
        // attention! the code above is a wrong type of computation due to the storage of data structure ! Image have been permute to (2, 0, 1) demension. So the correct code can be found as follows.

        
        int OUTPUT_OFFSET = xidx + yidx * width ;
        int INPUT_OFFSET = 3 * OUTPUT_OFFSET;
        unsigned int r = input[INPUT_OFFSET + 0];
        unsigned int g = input[INPUT_OFFSET + 1];
        unsigned int b = input[INPUT_OFFSET + 2];
        
        output[OUTPUT_OFFSET] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

torch::Tensor rgb_to_grayscale(torch::Tensor image)
{
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const int height = image.size(0);
    const int width = image.size(1);

    dim3 threads_per_block(32, 32);
    dim3 number_of_blocks(cdiv(width, threads_per_block.x), cdiv(height, threads_per_block.y));

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    rgb2gray<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(image.data_ptr<unsigned char>(), result.data_ptr<unsigned char>(), width, height);
    
    return result;
}