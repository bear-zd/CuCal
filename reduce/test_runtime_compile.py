import torch
from torch.utils.cpp_extension import load
import copy

cuda_module = load(
    name="reduce2",
    extra_include_paths=["include"],
    sources=["kernel/reduce2_ops.cpp", "kernel/reduce2_kernel.cu"],
    verbose=True
)

if __name__ == "__main__":
    int_a = torch.randint(0, 10, (2**20,), dtype=torch.int32,device='cuda')
    int_b = copy.deepcopy(int_a)
    # int_a = torch.ones(2**20, dtype=torch.int8, device="cuda:0")
    result = cuda_module.torch_launch_reduce2int(int_a)
    print(int_a)
    print(result, int_b.sum(0))