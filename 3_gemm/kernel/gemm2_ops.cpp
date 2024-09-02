#include <torch/extension.h>
#include "gemm2.h"

torch::Tensor torch_launch_gemm2(torch::Tensor x, torch::Tensor y) {
    // Ensure x and y are floats
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be of type float");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be of type float");

    int64_t m = x.size(0);
    int64_t n = y.size(1);
    int64_t k = x.size(1);

    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto z = torch::empty({m, n}, options);

    // Call CUDA kernel launcher
    launch_gemm2(x.data_ptr<float>(), y.data_ptr<float>(), z.data_ptr<float>(), m, n, k);

    return z;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_gemm2", &torch_launch_gemm2, "gemm kernel wrapper");
}

TORCH_LIBRARY(add2float, m) {
    m.def("torch_launch_gemm2", torch_launch_gemm2);
}