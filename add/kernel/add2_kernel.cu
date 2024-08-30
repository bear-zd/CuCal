__global__ void MatAddFloat(float* c, const float* a, const float* b, int n) {
    // Calculate the row and column index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the base index for the current thread
    int idx = j * n + i ;

    if (i < n && j < n  ) 
        c[idx] = a[idx] + b[idx];
}

__global__ void MatAddInt(int8_t* c, const int8_t* a, const int8_t* b, int n) {
    // Calculate the row and column index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the base index for the current thread
    int idx = j * n + i * 4 ;

    if ((i*4) < n && j < n  ) {
        c[idx] = a[idx] + b[idx];
        c[idx+1] = a[idx+1] + b[idx+1];
        c[idx+2] = a[idx+2] + b[idx+2];
        c[idx+3] = a[idx+3] + b[idx+3];
    }
}


void launch_add2float(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 block(32, 32);
    dim3 grid(n/block.x, n/block.y);

    MatAddFloat<<<grid, block>>>(c, a, b, n);
}


void launch_add2int(int8_t* c,
                 const int8_t* a,
                 const int8_t* b,
                 int n) {
    dim3 block(32, 32);
    dim3 grid(n/block.x, n/block.y);

    MatAddInt<<<grid, block>>>(c, a, b, n);
}