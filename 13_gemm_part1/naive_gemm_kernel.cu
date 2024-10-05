#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define TILE_SIZE 8
#define BLOCK_SIZE 16

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void gemm_naive(float* x, float* y, float* out, int m, int n, int k){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < m && col < n){
        float sum = 0.0f;
        for(int i = 0; i < k; i++){
            sum += x[row * k + i] * y[i * n + col];
        }
        out[row * n + col] = sum;
    }
}

__global__ void gemm_shared(float* x, float* y, float* out, int M, int N, int K){
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float shared_x[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_y[BLOCK_SIZE][BLOCK_SIZE];
    unsigned int length_bk = (K + TILE_SIZE - 1) / TILE_SIZE;
    float sum = 0.0f;
    for(int sblock_idx = 0; sblock_idx < length_bk; sblock_idx++){
        // load data into shared memory
        shared_x[ty][tx] = (row < M && (sblock_idx * BLOCK_SIZE + tx) < K) ? x[row * K + sblock_idx * BLOCK_SIZE + tx]:0.0f;
        shared_y[ty][tx] = (col < N && (sblock_idx * BLOCK_SIZE + ty) < K) ? y[(sblock_idx * BLOCK_SIZE + ty) * N + col]:0.0f;
        __syncthreads();
        #pragma unroll
        for(int i = 0; i < BLOCK_SIZE; i++){
            // tx, ty calculate for `BLOCK_SIZE x length_bk` times
            sum += shared_x[ty][i] * shared_y[i][tx];
        }
        __syncthreads();
    }
    if(row < M && col < N) out[row * N + col] = sum;
}

template<const int BM=128, const int BN=128, const int BK=8, const int TM=8, const int TN=8>
__global__ void gemm_shared_tile(float* a, float* b, float* c, int M, int N, int K) {
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tid = threadIdx.y * blockDim.x + tx; // tid within the block
  __shared__ float s_a[BM][BK], s_b[BK][BN]; // 2*128*8*4=8KB
  
  // 0. 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=8 按行读取 A行主序
  // 对于s_a每行8个数据，每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
  int ta_y = tid / 2; // tid/2 (128/8)*(128/8)=256 threads per block, tid/2->[0,128), BM=128 0~127
  int ta_x = (tid % 2 == 0) ? 0 : 4;  // (tid%2 == 0) ? 0 : 4, col of s_a 0,4
  // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
  // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
  int tb_y = tid / 32; // tid/32, row of s_b 256/32=8 行 0~7
  int tb_x = (tid % 32) * 4;  // (tid % 32) * 4, col of s_b 0,4,...,124
  // 1. 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
  float r_c[TM][TN] = {0.0}; // 8x8
  // 2. 先对K进行分块，每块BK大小
  for (int bk = 0; bk < (K + BK - 1) / BK; ++bk) {
    // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
    FLOAT4(s_a[ta_y][ta_x]) = FLOAT4(a[(by * BM + ta_y) * K + (bk * BK + ta_x)]);
    // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
    FLOAT4(s_b[tb_y][tb_x]) = FLOAT4(b[(bk * BK + tb_y) * N + (bx * BN + tb_x)]); 
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      // 3. 每个线程负责计算BM*BN(128x128)中的TM*TN(8x8)个元素
      #pragma unroll
      for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
          // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
          r_c[m][n] += s_a[ty * TM + m][k] * s_b[k][tx * TN + n];
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int m = 0; m < TM; ++m) {
    int tc_y = by * BM + ty * TM + m;
    #pragma unroll
    for (int n = 0; n < TN; n += 4) {
      int tc_x = bx * BN + tx * TN + n;
      FLOAT4(c[tc_y * N + tc_x]) = FLOAT4(r_c[m][n]);
    }
  }
}

torch::Tensor gemm(torch::Tensor x, torch::Tensor y){
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    int m = x.size(0);
    int n = y.size(1);
    int k = x.size(1);
    int grid_x = (m + 128 - 1) / 128;
    int grid_y = (n + 128 - 1) / 128;
    dim3 grid(grid_x, grid_y);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    torch::Tensor z = torch::empty({m, n}, x.options());
    float* x_ptr = x.data_ptr<float>();
    float* y_ptr = y.data_ptr<float>();
    float* z_ptr = z.data_ptr<float>();
    sgemm_t_8x8_sliced_k_f32x4_kernel<<<grid, block>>>(x_ptr, y_ptr, z_ptr, m, n, k);
    return z;
}