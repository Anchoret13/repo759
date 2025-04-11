#include <iostream>
#include <random>
#include <chrono>
#include "stencil.cuh"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n R threads_per_block" << std::endl;
        return 1;
    }

    unsigned int n = std::stoi(argv[1]);
    unsigned int R = std::stoi(argv[2]);
    unsigned int threads_per_block = std::stoi(argv[3]);

    if (n <= 0 || R <= 0 || threads_per_block <= 0) {
        std::cerr << "Error: All input parameters must be positive integers" << std::endl;
        return 1;
    }

    if (threads_per_block < 2 * R + 1) {
        std::cerr << "Error: threads_per_block must be >= 2 * R + 1" << std::endl;
        return 1;
    }

    float* h_image = new float[n];
    float* h_mask = new float[2 * R + 1];
    float* h_output = new float[n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (unsigned int i = 0; i < n; i++) {
        h_image[i] = dist(gen);
    }

    for (unsigned int i = 0; i < 2 * R + 1; i++) {
        h_mask[i] = dist(gen);
    }

    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image, n * sizeof(float));
    cudaMalloc(&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << h_output[n - 1] << std::endl;
    std::cout << milliseconds << std::endl;

    delete[] h_image;
    delete[] h_mask;
    delete[] h_output;
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}