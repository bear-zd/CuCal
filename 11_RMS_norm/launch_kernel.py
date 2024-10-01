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

cuda_src = Path("rmsnorm_kernel.cu").read_text()
cpp_src = "torch::Tensor launch_rmsnorm_kernel_fp32(torch::Tensor x, torch::Tensor w);"
funcs = ["launch_rmsnorm_kernel_fp32"]

lib = load_cuda(cuda_src, cpp_src, funcs)

i1 = torch.rand((1,1, 768), dtype=torch.float32).cuda()
w = torch.rand((768), dtype=torch.float32).cuda()
b = torch.rand((768), dtype=torch.float32).cuda()
torch_ln = torch.nn.LayerNorm((768), eps=1e-5, elementwise_affine=True).cuda()
torch_ln.weight.data = w
torch_ln.bias.data = b
correct = (i1 - i1.mean(dim=-1, keepdim=True)) / torch.sqrt((i1.std(dim=-1, keepdim=True) + 1e-5)) * w + b
lib.launch_rmsnorm_kernel_fp32 = show_time(lib.launch_rmsnorm_kernel_fp32)
torch_res = torch_ln(i1)
cuda_res = lib.launch_rmsnorm_kernel_fp32(i1, w)
print("Input Tensor (i1):", i1[0][0][:10])
print("Torch LayerNorm Result:", torch_res[0][0][:10])
print("CUDA LayerNorm Result:", cuda_res[0][0][:10])
print("Correct LayerNorm Result:", correct[0][0][:10])

print("Torch Result Shape:", torch_res.shape)
print("CUDA Result Shape:", cuda_res.shape)
print("Torch Result Mean:", torch_res.mean().item())
print("CUDA Result Mean:", cuda_res.mean().item())
print("Correct Result Mean:", correct.mean().item())
print("Torch Result Std:", torch_res.std().item())
print("CUDA Result Std:", cuda_res.std().item())
print("Correct Result Std:", correct.std().item())