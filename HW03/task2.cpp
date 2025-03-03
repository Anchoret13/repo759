#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include "convolution.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n t\n";
        std::cerr << "  n: size of the square matrix\n";
        std::cerr << "  t: number of threads (1-20)\n";
        return EXIT_FAILURE;
    }

    // Parse command line arguments
    std::size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);
    
    // Validate thread count
    if (t < 1 || t > 20) {
        std::cerr << "Thread count must be between 1 and 20\n";
        return EXIT_FAILURE;
    }
    
    // Set the number of threads for OpenMP
    omp_set_num_threads(t);

    // Create and fill the image
    std::vector<float> image(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = dist(gen);
    }

    // Create the 3×3 mask
    std::vector<float> mask = {
        0.5f, 1.0f, 0.5f,
        1.0f, 2.0f, 1.0f,
        0.5f, 1.0f, 0.5f
    };

    // Allocate space for the output
    std::vector<float> output(n * n);

    // Apply the convolution and time it
    auto start = std::chrono::high_resolution_clock::now();
    convolve(image.data(), output.data(), n, mask.data(), 3);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the time in milliseconds
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Print results
    std::cout << output[0] << std::endl;
    std::cout << output[n * n - 1] << std::endl;
    std::cout << time_ms << std::endl;

    return 0;
}