from torch.utils.cpp_extension import load_inline
from pathlib import Path
import torch


def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False, name=None):
    "Simple wrapper for torch.utils.cpp_extension.load_inline"
    if name is None: name = funcs[0]
    flags = "-O3 -Xptxas -O3 -Xcompiler -O3" if opt else "-O0 -Xptxas -O0 -Xcompiler -O0"
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=[flags], verbose=verbose, name=name)


cuda_src = Path("activation_kernel.cu").read_text()
cpp_src = "torch::Tensor relu_function(torch::Tensor x);\ntorch::Tensor sigmoid_function(torch::Tensor x);"
funcs = ["relu_function", "sigmoid_function"]

lib = load_cuda(cuda_src, cpp_src, funcs)

d = torch.rand(1000, dtype=torch.float32).cuda()
o_rc = lib.relu_function(d)
o_rt = torch.nn.functional.relu(d)
o_sc = lib.sigmoid_function(d)
o_st = torch.nn.functional.sigmoid(d)

print(d[0], o_rc[0], o_rt[0], o_sc[0], o_st[0])