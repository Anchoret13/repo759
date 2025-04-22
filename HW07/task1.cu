#include "matmul.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

template <typename T>
void fillMatrixRandom(T* matrix, unsigned int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0, 10.0);
    
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<T>(dist(gen));
    }
}

template <>
void fillMatrixRandom<int>(int* matrix, unsigned int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-10, 10);
    
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = dist(gen);
    }
}

template <typename T>
void testMatmul(void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int),
               unsigned int n, unsigned int block_dim) {
    T *h_A = new T[n * n];
    T *h_B = new T[n * n];
    T *h_C = new T[n * n];
    
    fillMatrixRandom<T>(h_A, n);
    fillMatrixRandom<T>(h_B, n);
    
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(T));
    cudaMalloc(&d_B, n * n * sizeof(T));
    cudaMalloc(&d_C, n * n * sizeof(T));
    
    cudaMemcpy(d_A, h_A, n * n * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(T), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    matmul_func(d_A, d_B, d_C, n, block_dim);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_C, d_C, n * n * sizeof(T), cudaMemcpyDeviceToHost);
    
    std::cout << h_C[0] << std::endl;
    std::cout << h_C[n * n - 1] << std::endl;
    std::cout << milliseconds << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n block_dim" << std::endl;
        return 1;
    }
    
    unsigned int n = std::stoi(argv[1]);
    unsigned int block_dim = std::stoi(argv[2]);
    
    testMatmul<int>(matmul_1, n, block_dim);
    testMatmul<float>(matmul_2, n, block_dim);
    testMatmul<double>(matmul_3, n, block_dim);
    
    return 0;
}