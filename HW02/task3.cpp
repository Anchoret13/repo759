#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include "matmul.h"

// Helper function to print the timing result, first element, and last element.
void printResults(double time_ms, const double* C, unsigned int n, const std::string& label) {
    std::cout << label << " time (ms): " << time_ms << "\n";
    std::cout << label << " first element: " << C[0] << "\n";
    std::cout << label << " last element: " << C[n*n - 1] << "\n";
    std::cout << "-----------------------------------------\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " n\n";
        return EXIT_FAILURE;
    }
    unsigned int n = std::stoul(argv[1]);

    std::vector<double> A(n * n), B(n * n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    std::vector<double> C1(n * n), C2(n * n), C3(n * n), C4(n * n);

    // mmul1
    {
        auto start = std::chrono::high_resolution_clock::now();
        mmul1(A.data(), B.data(), C1.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        printResults(time_ms, C1.data(), n, "mmul1");
    }

    // mmul2
    {
        auto start = std::chrono::high_resolution_clock::now();
        mmul2(A.data(), B.data(), C2.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        printResults(time_ms, C2.data(), n, "mmul2");
    }

    // mmul3
    {
        auto start = std::chrono::high_resolution_clock::now();
        mmul3(A.data(), B.data(), C3.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        printResults(time_ms, C3.data(), n, "mmul3");
    }

    // mmul4
    {
        auto start = std::chrono::high_resolution_clock::now();
        mmul4(A, B, C4.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        printResults(time_ms, C4.data(), n, "mmul4");
    }

    return 0;
}
