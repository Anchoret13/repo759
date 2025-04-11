#include <iostream>
#include <random>
#include <chrono>
#include "matmul.cuh"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n threads_per_block" << std::endl;
        return 1;
    }

    size_t n = std::stoi(argv[1]);
    unsigned int threads_per_block = std::stoi(argv[2]);

    // Validate inputs
    if (n <= 0 || threads_per_block <= 0) {
        std::cerr << "Error: Matrix size and threads_per_block must be positive integers" << std::endl;
        return 1;
    }

    // Allocate host memory for matrices
    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    float* h_C = new float[n * n];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Fill matrices with random values
    for (size_t i = 0; i < n * n; i++) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Call matrix multiplication function
    matmul(d_A, d_B, d_C, n, threads_per_block);

    // Record stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the last element of the resulting matrix
    std::cout << h_C[n * n - 1] << std::endl;
    
    // Print the execution time
    std::cout << milliseconds << std::endl;

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}