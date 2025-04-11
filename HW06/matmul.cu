#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Calculate the linear index for this thread
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the index is within bounds
    if (idx < n * n) {
        // Calculate the row and column indices for this thread
        unsigned int row = idx / n;
        unsigned int col = idx % n;
        
        // Compute one element of C
        float sum = 0.0f;
        for (unsigned int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        
        // Store the result
        C[row * n + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    // Calculate grid dimensions
    unsigned int total_threads = n * n;
    unsigned int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // Launch the kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);
}