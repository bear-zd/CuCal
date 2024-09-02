__global__ void ReduceInt(int * x,int * y,int64_t n) 
{
	unsigned int tid = threadIdx.x;
	if (tid >= n) return;
	int *idata = x + blockIdx.x*blockDim.x;
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		if ((tid % (2 * stride)) == 0)
		{
			idata[tid] += idata[tid + stride];
		}
		__syncthreads();
	}
	if (tid == 0)
		y[blockIdx.x] = idata[0];

}
__global__ void reduceNeighboredLess(int * g_idata,int *g_odata,int64_t n)
{
	unsigned int tid = threadIdx.x;
	unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
	// convert global data pointer to the local point of this block
	int *idata = g_idata + blockIdx.x*blockDim.x;
	if (idx > n)
		return;
	//in-place reduction in global memory
	for (int stride = 1; stride < blockDim.x; stride *= 2)
	{
		//convert tid into local array index
		int index = 2 * stride *tid;
		if (index < blockDim.x)
		{
			idata[index] += idata[index + stride];
		}
		__syncthreads();
	}
	//write result for this block to global men
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];
}
__global__ void reduceUnroll2(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*2+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*2;
	if(idx+blockDim.x<n)
	{
		g_idata[idx]+=g_idata[idx+blockDim.x];

	}
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>0 ; stride >>=1)
	{
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnroll(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*8+threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*8;
    // 相当于每个block分配了8*idx个数量的数据，先对8个数据进行归约
	if(idx+7 * blockDim.x<n)
	{
		int a1=g_idata[idx];
		int a2=g_idata[idx+blockDim.x];
		int a3=g_idata[idx+2*blockDim.x];
		int a4=g_idata[idx+3*blockDim.x];
		int a5=g_idata[idx+4*blockDim.x];
		int a6=g_idata[idx+5*blockDim.x];
		int a7=g_idata[idx+6*blockDim.x];
		int a8=g_idata[idx+7*blockDim.x];
		g_idata[idx]=a1+a2+a3+a4+a5+a6+a7+a8;

	}
	__syncthreads();
    // 接下来只剩下1024个数据了，因此可以简单的进行划分，线程束数量逐步下降。
	if(iBlockSize>=1024 && tid <512)
		idata[tid]+=idata[tid+512];
	__syncthreads();
	if(iBlockSize>=512 && tid <256)
		idata[tid]+=idata[tid+256];
	__syncthreads();
	if(iBlockSize>=256 && tid <128)
		idata[tid]+=idata[tid+128];
	__syncthreads();
	if(iBlockSize>=128 && tid <64)
		idata[tid]+=idata[tid+64];
	__syncthreads();
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vmem = idata;
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];

	}

	if (tid == 0)
		g_odata[blockIdx.x] = idata[0];

}
int64_t launch_reduce2int(int *x, int *y, int64_t n) {
    dim3 block(1024); // increased block size for better performance
    dim3 grid((n + block.x - 1) / block.x);
    int sharedMemSize = block.x * sizeof(int);
    reduceUnroll2<<<grid, block, block.x>>>(x, y, n);

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
