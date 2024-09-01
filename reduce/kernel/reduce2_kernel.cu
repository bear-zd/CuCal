__global__ void ReduceInt(int * g_idata,int * g_odata,int64_t n) 
{
	unsigned int tid = threadIdx.x;
	if (tid >= n) return;
	int *idata = g_idata + blockIdx.x*blockDim.x;
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}

int64_t launch_reduce2int(int *x, int *y, int64_t n) {
    dim3 block(32); // increased block size for better performance
    dim3 grid((n + block.x - 1) / block.x);
    int sharedMemSize = block.x * sizeof(int);
    ReduceInt<<<grid, block, sharedMemSize>>>(x, y, n);

    cudaError_t err = cudaGetLastError();
    cudaDeviceSynchronize();
    int *h_y = new int[grid.x](); 
    cudaMemcpy(h_y, y, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

    int64_t gpu_sum = 0;
    for (int i = 0; i < grid.x; i++) {
        gpu_sum += h_y[i];
    }

    delete[] h_y;
    return gpu_sum;
}
