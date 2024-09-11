#include <cuda_runtime.h>
#include <stdio.h>
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
            int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
            FLOAT4(s_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(a[load_a_gmem_addr]); 
            int load_b_gmem_k = bk * BK + load_b_smem_k;
            int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
            FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(b[load_b_gmem_addr]);
            // copy the data from global to shared memory.
            __syncthreads();
            // split the code above and below. Now consider the data in shared memory;
            #pragma unroll
            for(int k=0; k < BK; k++){
                #pragma unroll
                for(int m = 0; m < TM; m++){
                    #pragma unroll
                    for(int n = 0; n < TN; n++){
                        int comp_a_smem_m = ty * TM + m;
                        int comp_b_smem_n = tx * TN + n;
                        r_c[m][n] += s_a[comp_a_smem_m][k] * s_b[k][comp_b_smem_n];
                    }
                }
            }
            // a bit too complex. Each time get a col of s_a and a row of s_b to multiple and add to the r_c;
            // after sum above. each thread store the r_c matrix in local but not write to c matrix;
            __syncthreads();
        }
        #pragma unroll
        for(int i=0; i < TM ; i+=1){
            int store_c_gmem_m = by * BM + ty * TM + i;
            #pragma unroll
            for(int j=0; j < TN; j+=4){
                int store_c_gmem_n = bx * BN + tx * TN + j;
                int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
                FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][j]);
            }
        }
        
}


__global__ void sgemm_V3(
    float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 32;
    const int BN = 32;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k    ][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2         ]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2         ]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

void launch_gemm2(float* a, float* b, float* c, int M, int N, int K) {
    // Define block and grid sizes

    int BM = 32;
    int BN = 32;
    dim3 blockSize(BN, BM); // 16x16 threads per block
    dim3 gridSize((N + BN - 1) / BN, (M + BM - 1) / BM);

    // Launch the kernel
    naiveSgemm<<<gridSize, blockSize>>>(a, b, c, M, N, K);
    cudaDeviceSynchronize();
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    // printf("%d", err);

}