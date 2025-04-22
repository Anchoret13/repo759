#include "reduce.cuh"
#include <cuda_runtime.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    if (i < n) {
        sdata[tid] = g_idata[i];
        if (i + blockDim.x < n) {
            sdata[tid] += g_idata[i + blockDim.x];
        }
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N, 
                    unsigned int threads_per_block) {
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    
    float *d_in = *input;
    float *d_out = *output;
    
    unsigned int shared_mem_size = threads_per_block * sizeof(float);
    
    reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_in, d_out, N);
    
    unsigned int curr_size = blocks;
    
    while (curr_size > 1) {
        float *temp = d_in;
        d_in = d_out;
        d_out = temp;
        
        blocks = (curr_size + threads_per_block * 2 - 1) / (threads_per_block * 2);
        
        reduce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(d_in, d_out, curr_size);
        
        curr_size = blocks;
    }
    
    cudaMemcpy(*input, d_out, sizeof(float), cudaMemcpyDeviceToDevice);
    
    cudaDeviceSynchronize();
}