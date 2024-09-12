
__global__ void histogram_kernel(int* data, int* stat, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        atomicAdd(&(stat[data[idx]]), 1);
    }
}

torch::Tensor launch_histogram_kernel(torch::Tensor data){
    int length = data.size(0);
    std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(data, 0);
    torch::Tensor max_val = std::get<0>(max_a).cpu();                              
    const int maxv = max_val.item().to<int>();
    auto stat = torch::empty({maxv+1}, torch::TensorOptions().dtype(torch::kInt32).device(data.device()));
    dim3 block(256);
    dim3 grid((length + block.x - 1 ) / block.x);
    histogram_kernel<<<block, grid>>>(data.data_ptr<int>(), stat.data_ptr<int>(), length);
    return stat;
}