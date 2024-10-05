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

template<const int BM=128, const int BN=128, const int BK=8, const int TM=8, const int TN=8, const int OFFSET=0>
__global__ void sgemm_t_8x8_sliced_k_f32x4_bcf_kernel(
  float* a, float* b, float* c, const int M, const int N, const int K) {
  
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  __shared__ float s_a[BK][BM + OFFSET];
  __shared__ float s_b[BK][BN + OFFSET];
  // __shared__ float s_a[BK][BM + 4];
  // __shared__ float s_b[BK][BN + 4];

  float r_load_a[TM/2]; // 4
  float r_load_b[TN/2]; // 4
  float r_comp_a[TM];
  float r_comp_b[TN];
  float r_c[TM][TN] = {0.0};

  // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim 
  // row major values from A matrix, and store it in COL major s_a[BK][BM].
  int load_a_smem_m = tid / 2; // tid / 2，(0,1,2,...,128)
  // (0b00000000 & 0b00000001) << 2 = 0
  // (0b00000001 & 0b00000001) << 2 = 4
  // (0b00000010 & 0b00000001) << 2 = 0
  // (0b00000011 & 0b00000001) << 2 = 4
  int load_a_smem_k = (tid & 1) << 2; // (0,4)
  // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim 
  // row major values from B matrix, and store it in ROW major s_b[BK][BN].
  int load_b_smem_k = tid / 32; // 0~8
  // (0b00000000 & 0b00011111) << 2 = 0
  // (0b00000001 & 0b00011111) << 2 = 4
  // (0b00000010 & 0b00011111) << 2 = 8
  // (0b00000011 & 0b00011111) << 2 = 12
  int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

  int load_a_gmem_m = by * BM + load_a_smem_m;
  int load_b_gmem_n = bx * BN + load_b_smem_n;

  if (load_a_gmem_m >= M || load_b_gmem_n >= N) return;

  for (int bk = 0; bk < (K + BK - 1) / BK; bk++) {
    int load_a_gmem_k = bk * BK + load_a_smem_k;
    int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
    int load_b_gmem_k = bk * BK + load_b_smem_k;
    int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
    FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
    FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

    // 0. bank layout analysis: s_a[8][128]
    // 4 bytes per bank(32 banks, total 128 bytes, 32 float values), 
    // 1 float per bank. smem banks layout for s_a[8][128]:
    // 8*(128/32)=32 bank layers, 4 layers per k-th row.
    // [k=0][m=  [0],   [1],   [2],...,    [31]]
    // layer_0   [b0],  [b1],  [b2],...,   [b31]
    // [k=0][m=  [32],  [33],  [34],...,   [63]]
    // layer_1   [b0],  [b1],  [b2],...,   [b31]
    // [k=0][m=  [64],  [65],  [66],...,   [95]]
    // layer_2   [b0],  [b1],  [b2],...,   [b31]
    // [k=0][m=  [96],  [97],  [98],...,   [127]]
    // layer_3   [b0],  [b1],  [b2],...,   [b31]
    // ...       ...               ...
    // [k=7][m=  [0],   [1],   [2],...,    [31]]
    // layer_28  [b0],  [b1],  [b2],...,   [b31]
    // [k=7][m=  [32],  [33],  [34],...,   [63]]
    // layer_29  [b0],  [b1],  [b2],...,   [b31]
    // [k=7][m=  [64],  [65],  [66],...,   [95]]
    // layer_30  [b0],  [b1],  [b2],...,   [b31]
    // [k=7][m=  [96],  [97],  [98],...,   [127]]
    // layer_31  [b0],  [b1],  [b2],...,   [b31]
    // 1. bank conficts analysis: s_a[8][128]
    // tid 0   -> m 0,   k 0 -> all access bank 0  (layer_0/4/8/12)
    // tid 1   -> m 0,   k 4 -> all access bank 0  (layer_16/20/24/28)
    // tid 2   -> m 1,   k 0 -> all access bank 1  (layer_0/4/8/12)
    // tid 3   -> m 1,   k 4 -> all access bank 1  (layer_16/20/24/28)
    // tid 4   -> m 2,   k 0 -> all access bank 2  (layer_0/4/8/12)
    // tid 5   -> m 2,   k 4 -> all access bank 2  (layer_16/20/24/28)
    // tid 6   -> m 3,   k 0 -> all access bank 3  (layer_0/4/8/12)
    // tid 7   -> m 3,   k 4 -> all access bank 3  (layer_16/20/24/28)
    // ...        ...           ...                ...
    // tid 28  -> m 14,  k 0 -> all access bank 14 (layer_0/4/8/12)
    // tid 29  -> m 14,  k 4 -> all access bank 14 (layer_16/20/24/28)
    // tid 30  -> m 15,  k 0 -> all access bank 15 (layer_0/2/4/6)
    // tid 31  -> m 15,  k 4 -> all access bank 15 (layer_16/20/24/28)
    // conclusion: we still have bank conflicts for smem_a write access, 
    // each 2 consecutive threads within warp access the same bank! 
    // thus, we still need 2 memory issues as least per warp.
    s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0]; // e.g layer_0  b0
    s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1]; // e.g layer_4  b0
    s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2]; // e.g layer_8  b0
    s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3]; // e.g layer_12 b0
    // 2. bank layout analysis: s_b[8][128] same as s_a[8][128]
    // 3. bank conficts analysis: s_b[8][128]
    // tid 0   -> k 0, n 0   -> all access bank 0~3   (layer_0)
    // tid 1   -> k 0, n 4   -> all access bank 4~7   (layer_0)
    // tid 2   -> k 0, n 8   -> all access bank 7~11  (layer_0)
    // tid 7   -> k 0, n 28  -> all access bank 28~31 (layer_0)
    // tid 8   -> k 0, n 32  -> all access bank 0~3   (layer_1)
    // ...        ...         ...                 ...
    // tid 15  -> k 0, n 60  -> all access bank 28~31 (layer_1)
    // tid 16  -> k 0, n 64  -> all access bank 0~3   (layer_2)
    // ...        ...         ...                 ...
    // tid 31  -> k 0, n 124 -> all access bank 28~31 (layer_3)
    // conclusion: we still have bank conflicts within warp, 
    // 0/8/16/24 -> bank 0~3, 1/9/17/25 -> bank 4~7, etc. 
    // thus, we still need 4 memory issues at least per warp.
    FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

    __syncthreads();

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
      // bank conflicts analysis, tx/ty 0~15, 0~7 bank 4*8=32 bytes
      // tid 0~15 access bank 0~3,  tid 16~31 access bank 4~7, etc.
      // tid 0,  tk 0 -> ty 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),   
      // tid 0,  tk 7 -> ty 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
      // tid 15, tk 0 -> ty 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),   
      // tid 15, tk 7 -> ty 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
      // tid 16, tk 0 -> ty 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),   
      // tid 16, tk 7 -> ty 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
      // tid 31, tk 0 -> ty 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),   
      // tid 31, tk 7 -> ty 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
      // tid 255,tk 0 -> ty 15 -> [0][0+60~63],[0][64+60~63] -> bank 28~31(layer_1/3),   
      // tid 255,tk 7 -> ty 15 -> [7][0+60~63],[0][64+60~63] -> bank 28~31(layer_29/31), 
      FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2         ]);
      FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
      // if (tid == < 32 && bx == 0 && by == 0) {
      //   printf("tid: %d, tx: %d, ty: %d, [%d][%d]\n", tid, tx, ty, tk, ty * TM / 2);
      //   printf("tid: %d, tx: %d, ty: %d, [%d][%d]\n", tid, tx, ty, tk, ty * TM / 2 + BM / 2);
      // }
      // conclusion: still have bank conflicts, need 16 memory issues ?

      // tid 0/8/16/24  access bank 0~3,  tid 1/9/17/25  access bank 4~7, 
      // tid 2/10/18/26 access bank 8~11, tid 7/15/23/31 access bank 28~31, etc.
      // tid 0, tk 0 -> tx 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),    
      // tid 0, tk 7 -> tx 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
      // tid 1, tk 0 -> tx 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),    
      // tid 1, tk 7 -> tx 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
      FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2         ]);
      FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
      // conclusion: still have some bank conflicts, need 4 memory issues.

      #pragma unroll
      for (int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < TN; tn++) {
          // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
          r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
        }
      }
    }
    // sync per BK.
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
  }
  #pragma unroll
  for (int i = 0; i < TM / 2; i++) {
    int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
    int store_c_gmem_n = bx * BN + tx * TN / 2;
    int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
    FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
    FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
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
    gemm_shared_tile<<<grid, block>>>(x_ptr, y_ptr, z_ptr, m, n, k);
    return z;
}