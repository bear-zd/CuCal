from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch
import time
ntest = 10

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

cuda_src = Path("complex_gemm_kernel.cu").read_text()
cpp_src = "torch::Tensor gemm(torch::Tensor x, torch::Tensor y);"
funcs = ["gemm"]

lib = load_cuda(cuda_src, cpp_src, funcs)

x = torch.rand((1024, 512), dtype=torch.float32).cuda()
y = torch.rand((512, 1024), dtype=torch.float32).cuda()

lib.gemm = show_time(lib.gemm)

correct = torch.matmul(x, y)

cuda_res = lib.gemm(x, y)

# 打印结果
print("Input Tensor X (first 10 elements):", x[0][:10])
print("Input Tensor Y (first 10 elements):", y[0][:10])
print("Correct GEMM Result (first 10 elements):", correct[0][:10])
print("CUDA GEMM Result (first 10 elements):", cuda_res[0][:10])

print("Correct Result Shape:", correct.shape)
print("CUDA Result Shape:", cuda_res.shape)
print("Correct Result Mean:", correct.mean().item())
print("CUDA Result Mean:", cuda_res.mean().item())
print("Correct Result Std:", correct.std().item())
print("CUDA Result Std:", cuda_res.std().item())