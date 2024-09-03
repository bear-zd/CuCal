import torch
from torch.utils.cpp_extension import load
import copy
import time
import numpy as np
ntest = 10

def show_time(func):
    def wrapper(*args, **kwargs):
        times = list()
        res = None
        # GPU warm up
        for _ in range(10):
            res = func(*args, **kwargs)
        for _ in range(ntest):
            # sync the threads to get accurate cuda running time
            # torch.cuda.synchronize(device="cuda:0")
            start_time = time.time()
            res = func(*args, **kwargs)
            # torch.cuda.synchronize(device="cuda:0")
            end_time = time.time()
            times.append((end_time - start_time) * 1e6)
        return times, res
    return wrapper

cuda_module = load(
    name="gemm2",
    extra_include_paths=["include"],
    sources=["kernel/gemm2_ops.cpp", "kernel/gemm2_kernel.cu"],
    verbose=True
)
cuda_module.torch_launch_gemm2 = show_time(cuda_module.torch_launch_gemm2)
torch.mm = show_time(torch.mm)

if __name__ == "__main__":
    a = torch.rand((64, 64), dtype=torch.float32,device='cuda')
    b = torch.rand((64, 64), dtype=torch.float32,device='cuda')
    # int_a = torch.ones(2**20, dtype=torch.int8, device="cuda:0")
    cal_time, cuda_res = cuda_module.torch_launch_gemm2(a, b)
    # o = torch.zeros((1024, 1024), dtype=torch.float32,device='cuda')
    print("cuda time:  {:.3f}us".format(np.mean(cal_time)))
    # cal_time, torch_res = torch.mm(a, b, out=o) 
    print("torch time:  {:.3f}us".format(np.mean(cal_time)))
    print(cuda_res[0])
    
    # print(o[0])

    # actually result is correct. The result isn't equals sum is because the sum inplace chaging the value.
    # print(result, int_b.sum(0))