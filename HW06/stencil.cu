#include "stencil.cuh"

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    extern __shared__ float shared_mem[];
    
    float* shared_mask = shared_mem;
    float* shared_image = shared_mem + (2 * R + 1);
    float* shared_output = shared_image + blockDim.x + 2 * R;
    
    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tx;
    
    if (tx < 2 * R + 1) {
        shared_mask[tx] = mask[tx];
    }
    
    int load_idx = i - R;
    if (load_idx >= 0 && load_idx < n) {
        shared_image[tx + R] = image[load_idx];
    } else {
        shared_image[tx + R] = 1.0f;
    }
    
    if (tx < R) {
        int left_idx = blockIdx.x * blockDim.x - R + tx;
        if (left_idx >= 0) {
            shared_image[tx] = image[left_idx];
        } else {
            shared_image[tx] = 1.0f;
        }
        
        int right_idx = (blockIdx.x + 1) * blockDim.x + tx;
        if (right_idx < n) {
            shared_image[blockDim.x + R + tx] = image[right_idx];
        } else {
            shared_image[blockDim.x + R + tx] = 1.0f;
        }
    }
    
    __syncthreads();
    
    if (i < n) {
        float result = 0.0f;
        for (int j = -R; j <= R; j++) {
            result += shared_image[tx + R + j] * shared_mask[j + R];
        }
        shared_output[tx] = result;
    }
    
    __syncthreads();
    
    if (i < n) {
        output[i] = shared_output[tx];
    }
}

__host__ void stencil(const float* image,
                     const float* mask,
                     float* output,
                     unsigned int n,
                     unsigned int R,
                     unsigned int threads_per_block) {
    
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    int shared_mem_size = (2 * R + 1 + threads_per_block + 2 * R + threads_per_block) * sizeof(float);
    
    stencil_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(image, mask, output, n, R);
}