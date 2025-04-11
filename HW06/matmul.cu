#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n * n) {
        unsigned int row = idx / n;
        unsigned int col = idx % n;
        
        float sum = 0.0f;
        for (unsigned int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        
        C[row * n + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    unsigned int total_threads = n * n;
    unsigned int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);
}