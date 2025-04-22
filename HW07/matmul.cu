#include "matmul.cuh"
#include <cuda_runtime.h>

template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    
    unsigned int block_dim = blockDim.x;
    
    extern __shared__ char shared_mem[];
    T* tile_A = (T*)shared_mem;
    T* tile_B = (T*)(shared_mem + block_dim * block_dim * sizeof(T));
    
    unsigned int row = by * block_dim + ty;
    unsigned int col = bx * block_dim + tx;
    
    T sum = 0;
    
    unsigned int num_tiles = (n + block_dim - 1) / block_dim;
    
    for (unsigned int tile = 0; tile < num_tiles; ++tile) {
        if (row < n && tile * block_dim + tx < n) {
            tile_A[ty * block_dim + tx] = A[row * n + tile * block_dim + tx];
        } else {
            tile_A[ty * block_dim + tx] = 0;
        }
        
        if (col < n && tile * block_dim + ty < n) {
            tile_B[ty * block_dim + tx] = B[(tile * block_dim + ty) * n + col];
        } else {
            tile_B[ty * block_dim + tx] = 0;
        }
        
        __syncthreads();
        
        for (unsigned int k = 0; k < block_dim; ++k) {
            sum += tile_A[ty * block_dim + k] * tile_B[k * block_dim + tx];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, 
                      unsigned int block_dim) {
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    dim3 threads(block_dim, block_dim);
    
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(int);
    
    matmul_kernel<int><<<blocks, threads, shared_mem_size>>>(A, B, C, n);
    
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, 
                      unsigned int block_dim) {
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    dim3 threads(block_dim, block_dim);
    
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);
    
    matmul_kernel<float><<<blocks, threads, shared_mem_size>>>(A, B, C, n);
    
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, 
                      unsigned int block_dim) {
    dim3 blocks((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    dim3 threads(block_dim, block_dim);
    
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(double);
    
    matmul_kernel<double><<<blocks, threads, shared_mem_size>>>(A, B, C, n);
    
    cudaDeviceSynchronize();
}