#include "reduce.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N threads_per_block" << std::endl;
        return 1;
    }
    
    unsigned int N = std::stoul(argv[1]);
    unsigned int threads_per_block = std::stoul(argv[2]);
    
    float *h_data = new float[N];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (unsigned int i = 0; i < N; ++i) {
        h_data[i] = dist(gen);
    }
    
    float *d_input;
    cudaMalloc(&d_input, N * sizeof(float));
    
    cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    
    unsigned int blocks = (N + threads_per_block * 2 - 1) / (threads_per_block * 2);
    
    float *d_output;
    cudaMalloc(&d_output, blocks * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    reduce(&d_input, &d_output, N, threads_per_block);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    float result;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << result << std::endl;
    std::cout << milliseconds << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_input);
    cudaFree(d_output);
    
    delete[] h_data;
    
    return 0;
}