#include <iostream>
#include <random>
#include "matmul.cuh"

int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " n threads_per_block" << std::endl;
            return 1;
        }

        size_t n = std::stoi(argv[1]);
        unsigned int threads_per_block = std::stoi(argv[2]);

        std::cout << "Starting with n=" << n << ", threads_per_block=" << threads_per_block << std::endl;

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
        
        // Print a few values to verify initialization
        std::cout << "First few values of A: " << h_A[0] << ", " << h_A[1] << std::endl;
        std::cout << "First few values of B: " << h_B[0] << ", " << h_B[1] << std::endl;

        float *d_A, *d_B, *d_C;
        cudaError_t err;
        
        err = cudaMalloc(&d_A, n * n * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_A: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        
        err = cudaMalloc(&d_B, n * n * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_B: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_A);
            return 1;
        }
        
        err = cudaMalloc(&d_C, n * n * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate d_C: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_A);
            cudaFree(d_B);
            return 1;
        }

        err = cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy A to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return 1;
        }
        
        err = cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy B to device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return 1;
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        matmul(d_A, d_B, d_C, n, threads_per_block);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return 1;
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        err = cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy C from device: " << cudaGetErrorString(err) << std::endl;
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return 1;
        }

        std::cout << "Result calculation complete. Last element: " << h_C[n * n - 1] << std::endl;
        std::cout << h_C[n * n - 1] << std::endl;
        std::cout << milliseconds << std::endl;

        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
        return 1;
    }

    return 0;
}