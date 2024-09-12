import load
from pathlib import Path
import torch

cuda_source = Path("histogram_kernel.cu").read_text()
# cuda_source = load.cuda_begin + "\n" + cuda_source
cpp_source = "torch::Tensor launch_histogram_kernel(torch::Tensor data);"
lib = load.load_cuda(cuda_src=cuda_source, cpp_src=cpp_source, funcs="launch_histogram_kernel")


a = torch.tensor(list(range(10))*1000, dtype=torch.int32).cuda()
h_i32 = lib.launch_histogram_kernel(a)
print("-" * 80)
for i in range(h_i32.shape[0]):
    print(f"h_i32   {i}: {h_i32[i]}")
    
