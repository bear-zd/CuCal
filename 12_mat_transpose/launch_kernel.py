from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch
import time
ntest = 20

def show_time(func):
    def wrapper(*args, **kwargs):
        times = list()
        res = None
        # GPU warm up
        for _ in range(10):
            res = func(*args, **kwargs)
        for t in range(ntest):
            # sync the threads to get accurate cuda running time
            torch.cuda.synchronize(device="cuda:0")
            start_time = time.time()
            res = func(*args, **kwargs)
            torch.cuda.synchronize(device="cuda:0")
            end_time = time.time()
            times.append((end_time - start_time) * 1e6)
            if t == ntest - 1:
                print(f"time spend : {sum(times)/ntest:.2f} us")
        return res
    return wrapper

def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=[flags], verbose=verbose, name=name)


# cuda_src = Path("softmax_kernel.cu").read_text()
# cpp_src = "torch::Tensor launch_softmax_kernel_fp32(torch::Tensor x);"
# funcs = ["launch_softmax_kernel_fp32"]

cuda_src = Path("mat_transpose_kernel.cu").read_text()
cpp_src = "torch::Tensor mat_transpose_mat_transpose_naive(torch::Tensor x);\ntorch::Tensor mat_transpose_mat_transpose_shared(torch::Tensor x);"
funcs = ["mat_transpose_mat_transpose_naive", "mat_transpose_mat_transpose_shared"]


lib = load_cuda(cuda_src, cpp_src, funcs)

# 创建输入张量
x = torch.rand((1024, 1024), dtype=torch.float32).cuda()

# 使用装饰器来 profile CUDA 核函数
torch.transpose = show_time(torch.transpose)
lib.mat_transpose_mat_transpose_naive = show_time(lib.mat_transpose_mat_transpose_naive)
lib.mat_transpose_mat_transpose_shared = show_time(lib.mat_transpose_mat_transpose_shared)

# 计算正确结果
correct = torch.transpose(x, 0, 1)

# 计算 CUDA 核函数结果
cuda_res1 = lib.mat_transpose_mat_transpose_naive(x)
cuda_res2 = lib.mat_transpose_mat_transpose_shared(x)

# 打印结果
print("Input Tensor X (first 10 elements):", x[0][:2])
print("Correct Transpose Result (first 10 elements):", correct[0][:2])
print("CUDA Transpose Result (first 10 elements):", cuda_res1[0][:2])
print("CUDA Transpose Result (first 10 elements):", cuda_res2[0][:2])

# print("Correct Result Shape:", correct.shape)
# print("CUDA Result Shape:", cuda_res.shape)
# print("Correct Result Mean:", correct.mean().item())
# print("CUDA Result Mean:", cuda_res.mean().item())
# print("Correct Result Std:", correct.std().item())
# print("CUDA Result Std:", cuda_res.std().item())