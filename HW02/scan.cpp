#include "scan.h"

void scan(const float *arr, float *output, std::size_t n) {
    if (n == 0) return;  // Handle empty array case

    float sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += arr[i];
        output[i] = sum;
    }
}