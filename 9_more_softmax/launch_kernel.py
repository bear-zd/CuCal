from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch
import time
ntest = 1

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


cuda_src = Path("softmax_kernel.cu").read_text()
cpp_src = "torch::Tensor launch_softmax_kernel_fp32(torch::Tensor x);"
funcs = ["launch_softmax_kernel_fp32"]

lib = load_cuda(cuda_src, cpp_src, funcs)

i1 = torch.rand(1000, dtype=torch.float32).cuda()
torch.nn.Softmax = show_time(torch.nn.Softmax(dim=0))
lib.launch_softmax_kernel_fp32 = show_time(lib.launch_softmax_kernel_fp32)
torch_res = torch.nn.Softmax(i1)
cuda_res = lib.launch_softmax_kernel_fp32(i1)
print(i1[0], torch_res[0], cuda_res[0], torch.sum(torch_res), torch.sum(cuda_res))