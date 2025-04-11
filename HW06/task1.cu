#include <iostream>
#include <random>
#include <chrono>
#include "matmul.cuh"

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n threads_per_block" << std::endl;
        return 1;
    }

    size_t n = std::stoi(argv[1]);
    unsigned int threads_per_block = std::stoi(argv[2]);

    if (n <= 0 || threads_per_block <= 0) {
        std::cerr << "Error: Matrix size and threads_per_block must be positive integers" << std::endl;
        return 1;
    }

    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    float* h_C = new float[n * n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < n * n; i++) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, n * n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    matmul(d_A, d_B, d_C, n, threads_per_block);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << h_C[n * n - 1] << std::endl;
    std::cout << milliseconds << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}