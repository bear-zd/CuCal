from torch.utils.cpp_extension import load_inline
import torch

cuda_source = '''
__global__ void MatAddFloat(float* c, const float* a, const float* b, int n) {
    // Calculate the row and column index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the base index for the current thread
    int idx = j * n + i ;

    if (i < n && j < n  ) 
        c[idx] = a[idx] + b[idx];
}


torch::Tensor MatAdd(torch::Tensor Mat1, torch::Tensor Mat2) {
    auto result = torch::empty_like(Mat1);
    const auto n = Mat1.size(0);
    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((n + threads_per_block.x - 1) / threads_per_block.x,
                          (n + threads_per_block.y - 1) / threads_per_block.y);

    MatAddFloat<<<number_of_blocks, threads_per_block>>>(
        result.data_ptr<float>(), Mat1.data_ptr<float>(), Mat2.data_ptr<float>(), n);

    return result;
    }
'''


cpp_source = "torch::Tensor MatAdd(torch::Tensor Mat1, torch::Tensor Mat2);"

# Load the CUDA kernel as a PyTorch extension
cpp_extension = load_inline(
    name='cpp_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['MatAdd'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    # build_directory='./load_inline_cuda',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

a = torch.rand(2**8,device='cuda',dtype=torch.float32)
b = torch.rand(2**8,device='cuda',dtype=torch.float32)

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    res = cpp_extension.MatAdd(a, b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))