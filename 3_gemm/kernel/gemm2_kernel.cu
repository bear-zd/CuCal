#include <cuda_runtime.h>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

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

void launch_gemm2(float* a, float* b, float* c, int M, int N, int K) {
    // Define block and grid sizes
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    naiveSgemm<<<gridSize, blockSize>>>(a, b, c, M, N, K);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();

}