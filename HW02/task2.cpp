#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include "convolution.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n m" << std::endl;
        return EXIT_FAILURE;
    }

    std::size_t n = std::stoul(argv[1]);
    std::size_t m = std::stoul(argv[2]);

    if (m % 2 == 0) {
        std::cerr << "Error: m must be an odd number." << std::endl;
        return EXIT_FAILURE;
    }

    float* image = new float[n * n];
    float* mask  = new float[m * m];
    float* output = new float[n * n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> image_dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> mask_dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = image_dist(gen);
    }

    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = mask_dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << duration.count() << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[n * n - 1] << std::endl;

    delete[] image;
    delete[] mask;
    delete[] output;

    return EXIT_SUCCESS;
}
