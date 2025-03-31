#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "vscale.cuh"

float random_float(float min, float max) {
    float scale = rand() / (float) RAND_MAX;
    return min + scale * (max - min);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    unsigned int n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "n must be positive" << std::endl;
        return 1;
    }

    srand(time(nullptr));

    float *h_a = new float[n];
    float *h_b = new float[n];

    for (unsigned int i = 0; i < n; i++) {
        h_a[i] = random_float(-10.0f, 10.0f);
        h_b[i] = random_float(0.0f, 1.0f);
    }

    float *d_a, *d_b;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 512;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vscale<<<blocks, threadsPerBlock>>>(d_a, d_b, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_b, d_b, n * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << milliseconds << std::endl;
    std::cout << h_b[0] << std::endl;
    std::cout << h_b[n-1] << std::endl;

    delete[] h_a;
    delete[] h_b;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}