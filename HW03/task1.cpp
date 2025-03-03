#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include "matmul.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n t\n";
        std::cerr << "  n: matrix dimension\n";
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
    
    // Set number of threads for OpenMP
    omp_set_num_threads(t);
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    
    // Allocate memory for matrices
    std::vector<float> A(n * n), B(n * n), C(n * n);
    
    // Fill matrices A and B with random values
    for (std::size_t i = 0; i < n * n; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }
    
    // Perform matrix multiplication and measure time
    auto start = std::chrono::high_resolution_clock::now();
    mmul(A.data(), B.data(), C.data(), n);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate elapsed time in milliseconds
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Print first element of C
    std::cout << C[0] << std::endl;
    
    // Print last element of C
    std::cout << C[n * n - 1] << std::endl;
    
    // Print time taken in milliseconds
    std::cout << time_ms << std::endl;
    
    return EXIT_SUCCESS;
}