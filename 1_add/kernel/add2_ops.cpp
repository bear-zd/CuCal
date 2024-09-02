#include <torch/extension.h>
#include "add2.h"

void torch_launch_add2float(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_add2float((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

void torch_launch_add2int(torch::Tensor &c,
                          const torch::Tensor &a,
                          const torch::Tensor &b,
                          int64_t n) {
    launch_add2int((int8_t*)c.data_ptr(),
                   (const int8_t*)a.data_ptr(),
                   (const int8_t*)b.data_ptr(),
                   n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2float", &torch_launch_add2float, "add2 float kernel wrapper");
    m.def("torch_launch_add2int", &torch_launch_add2int, "add2 int kernel wrapper");
}

TORCH_LIBRARY(add2float, m) {
    m.def("torch_launch_add2float", torch_launch_add2float);
    m.def("torch_launch_add2int", torch_launch_add2int);
} 
