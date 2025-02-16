#include "matmul.h"
#include <vector>
#include <cstring>

void mmul1(const double* A, const double* B, double* C, const unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (unsigned int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}


void mmul2(const double* A, const double* B, double* C, const unsigned int n)
{
    for (unsigned int idx = 0; idx < n * n; ++idx) {
        C[idx] = 0.0;
    }

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < n; ++k) {
            double a_ik = A[i * n + k];
            for (unsigned int j = 0; j < n; ++j) {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n)
{
    std::vector<double> B_T(n * n);
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            B_T[j * n + i] = B[i * n + j];
        }
    }

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (unsigned int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B_T[j * n + k];
            }
            C[i * n + j] = sum;
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, 
    double* C, const unsigned int n)
{
for (unsigned int i = 0; i < n; ++i) {
 for (unsigned int j = 0; j < n; ++j) {
     double sum = 0.0;
     for (unsigned int k = 0; k < n; ++k) {
         sum += A[i * n + k] * B[k * n + j];
     }
     C[i * n + j] = sum;
 }
}
}
