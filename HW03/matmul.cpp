#include "matmul.h"

void mmul(const float* A, const float* B, float* C, const std::size_t n) {
    #pragma omp parallel for
    for (std::size_t idx = 0; idx < n * n; ++idx) {
        C[idx] = 0.0f;
    }

    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            float a_ik = A[i * n + k];
            for (std::size_t j = 0; j < n; ++j) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}
