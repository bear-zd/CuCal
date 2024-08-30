import time
import argparse
import numpy as np
import torch

# c = a + b (shape: [n])
n = 2**13
a = torch.rand((n,n), device="cuda:0")
b = torch.rand((n,n), device="cuda:0")
cuda_c = torch.rand((n,n), device="cuda:0")
low = 0
high = 10
int_a = torch.randint(low, high, (n,n), device="cuda:0",dtype=torch.uint8)
int_b = torch.randint(low, high, (n,n), device="cuda:0",dtype=torch.uint8)
int_cuda_c = torch.randint(low, high, (n,n), device="cuda:0",dtype=torch.uint8)
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
            torch.cuda.synchronize(device="cuda:0")
            start_time = time.time()
            res = func(*args, **kwargs)
            torch.cuda.synchronize(device="cuda:0")
            end_time = time.time()
            times.append((end_time - start_time) * 1e6)
        return times, res
    return wrapper

@show_time
def run_cuda():
    cuda_module.torch_launch_add2float(cuda_c, a, b, n)
    return cuda_c

@show_time
def runint_cuda():
    cuda_module.torch_launch_add2int(int_cuda_c, int_a, int_b, n)
    return int_cuda_c

@show_time
def run_torch():
    c = a + b
    return c.contiguous()
@show_time
def runint_torch():
    int_c = int_a + int_b
    return int_c.contiguous()

if __name__ == "__main__":

    from torch.utils.cpp_extension import load
    cuda_module = load(name="add2",
                        extra_include_paths=["include"],
                        sources=["kernel/add2_ops.cpp", "kernel/add2_kernel.cu"],
                        verbose=True)

    cudafloat_time, cudafloat_res = run_cuda()
    print("cuda time:  {:.3f}us".format(np.mean(cudafloat_time)))
    torchfloat_time, torchfloat_res = run_torch()
    print("Torch time:  {:.3f}us".format(np.mean(cudafloat_time)))
    print(f"Sum of cuda core: {cudafloat_res.sum()}; Sum of torch: {torchfloat_res.sum()}")
    del(a, b, cuda_c)
    cudaint_time, cudaint_res = runint_cuda()
    print("cuda time:  {:.3f}us".format(np.mean(cudafloat_time)))
    torchint_time, torchint_res = runint_torch()
    
    print("Torch time:  {:.3f}us".format(np.mean(cudafloat_time)))
    print(f"Sum of cuda core: {cudaint_res.sum()}; Sum of torch: {torchint_res.sum()}")
