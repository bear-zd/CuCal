from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch


def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=[flags], verbose=verbose, name=name)


cuda_src = Path("dproduct_kernel.cu").read_text()
cpp_src = "torch::Tensor launch_dot_product_kernel_fp32(torch::Tensor x, torch::Tensor y);"
funcs = ["launch_dot_product_kernel_fp32"]

lib = load_cuda(cuda_src, cpp_src, funcs)

i1 = torch.rand(1000, dtype=torch.float32).cuda()
i2 = torch.rand(1000, dtype=torch.float32).cuda()
torch_res = torch.dot(i1, i2)
cuda_res = lib.launch_dot_product_kernel_fp32(i1, i2)
print(i1[0], i2[0], torch_res, cuda_res)