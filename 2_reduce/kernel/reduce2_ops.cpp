#include <torch/extension.h>
#include "reduce2.h"

int64_t  torch_launch_reduce2int(torch::Tensor x) {
    int64_t n = x.numel();
    // int *y = new int[n];
    torch::TensorOptions options = torch::TensorOptions().device(torch::kCUDA);
    auto y = torch::zeros((n + 31) / 32 , options); 
    int64_t  res = launch_reduce2int((int*)x.data_ptr(), (int*)y.data_ptr(), n);
    return res;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_reduce2int", &torch_launch_reduce2int, "reduce2 int kernel wrapper");
}

TORCH_LIBRARY(add2float, m) {
    m.def("torch_launch_reduce2int", torch_launch_reduce2int);
}