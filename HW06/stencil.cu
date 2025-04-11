#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_mem[];
    
    float* shared_mask = shared_mem;
    float* shared_image = shared_mem + (2 * R + 1);
    float* shared_output = shared_image + blockDim.x + 2 * R;
    
    unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_idx = threadIdx.x;
    
    if (local_idx < 2 * R + 1) {
        shared_mask[local_idx] = mask[local_idx];
    }
    
    int image_idx = global_idx - R;
    
    if (local_idx < blockDim.x + 2 * R) {
        int source_idx = blockIdx.x * blockDim.x + local_idx - R;
        if (source_idx < 0 || source_idx >= n) {
            shared_image[local_idx] = 1.0f;
        } else {
            shared_image[local_idx] = image[source_idx];
        }
    }
    
    __syncthreads();
    
    if (global_idx < n) {
        float sum = 0.0f;
        for (int j = -R; j <= R; j++) {
            sum += shared_image[local_idx + R + j] * shared_mask[j + R];
        }
        shared_output[local_idx] = sum;
    }
    
    __syncthreads();
    
    if (global_idx < n) {
        output[global_idx] = shared_output[local_idx];
    }
}

__host__ void stencil(const float* image, const float* mask, float* output, unsigned int n, unsigned int R, unsigned int threads_per_block) {
    unsigned int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    size_t shared_mem_size = ((2 * R + 1) + (threads_per_block + 2 * R) + threads_per_block) * sizeof(float);
    
    stencil_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(image, mask, output, n, R);
}