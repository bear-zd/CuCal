#include <cuda_runtime.h>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void naiveSgemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

__global__ void shared_gemm(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {
        const int BM = 128;
        const int BN = 128;
        const int BK = 8;
        const int TM = 8;
        const int TN = 8;
        
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;
        const int tid = tx + ty * blockDim.x;

        __shared__ float s_a[BM][BK];
        __shared__ float s_b[BK][BN];

        float r_c[TM][TN] = {0.0};

        int load_a_smem_m = tid >> 1; 
        int load_a_smem_k = (tid & 1) << 2;
        /*  used in the following code:
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]);
        */
        // pick 4 block each time. because of the BK = 8 and 4 block of float is satified the data storage in memory
        int load_b_smem_k = tid >> 5;
        int load_b_smem_n = (tid & 31) << 2;
        /*  used in the following code:
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
        */
        // BK = 8, in the vertical , BK * 4 int data = 32. 

        int load_a_gmem_m = by * BM + load_a_smem_m ;
        int load_b_gmem_n = bx * BN + load_b_smem_n ;
        // only transfer the local smem index to global index (mentioned that this is only the m,n value)
        for(int bk = 0; bk < (K + BK - 1)/BK; bk++ ){
            int load_a_gmem_k = bk * BK + load_a_smem_k;
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]); 
            int load_b_gmem_k = bk * BK + load_b_smem_k;
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_smem_n, N);
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
            // copy the data from global to shared memory
            __syncthreads();
            // split the code above and below. Now consider the data in shared memory;
            #pragma unroll
            for(int k=0; k < BK; k++){
                #pragma unroll
                for(int m=0; m<TM; m++){
                    #pragma unroll
                    for(int n=0; n<TN; n++){
                        int comp_a_smem_n = ty * TM + m;
                    }
                }
            }
        }





        
}


void launch_gemm2(float* a, float* b, float* c, int M, int N, int K) {
    // Define block and grid sizes
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    naiveSgemm<<<gridSize, blockSize>>>(a, b, c, M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();

}