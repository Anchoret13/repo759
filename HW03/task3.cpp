#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include "msort.h"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " n t ts\n";
        std::cerr << "  n: array size\n";
        std::cerr << "  t: number of threads\n";
        std::cerr << "  ts: threshold for switching to serial sort\n";
        return EXIT_FAILURE;
    }

    // Parse command line arguments
    std::size_t n = std::stoul(argv[1]);
    int t = std::stoi(argv[2]);
    std::size_t threshold = std::stoul(argv[3]);

    // Set number of threads
    omp_set_num_threads(t);

    // Create and fill array with random numbers
    std::vector<int> arr(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000, 1000);

    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = dist(gen);
    }

    // Apply msort and measure time
    auto start = std::chrono::high_resolution_clock::now();
    msort(arr.data(), n, threshold);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time in milliseconds
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Print the first and last elements
    std::cout << arr[0] << std::endl;
    std::cout << arr[n-1] << std::endl;
    std::cout << time_ms << std::endl;

    return 0;
}