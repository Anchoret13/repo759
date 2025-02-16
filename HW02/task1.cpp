#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include "scan.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return EXIT_FAILURE;
    }

    std::size_t n = std::stoul(argv[1]);
    if(n == 0) {
        std::cerr << "n must be a positive integer." << std::endl;
        return EXIT_FAILURE;
    }

    float* arr = new float[n];
    float* output = new float[n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    scan(arr, output, n);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << duration.count() << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[n - 1] << std::endl;

    delete[] arr;
    delete[] output;

    return EXIT_SUCCESS;
}
