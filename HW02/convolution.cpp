#include "convolution.h"

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    std::size_t offset = (m - 1) / 2;

    for (std::size_t x = 0; x < n; ++x) {
        for (std::size_t y = 0; y < n; ++y) {
            float sum = 0.0f;
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < m; ++j) {
                    int row = static_cast<int>(x) + static_cast<int>(i) - static_cast<int>(offset);
                    int col = static_cast<int>(y) + static_cast<int>(j) - static_cast<int>(offset);
                    
                    float f_val = 0.0f;
                    if (row >= 0 && row < static_cast<int>(n) &&
                        col >= 0 && col < static_cast<int>(n)) {
                        f_val = image[row * n + col];
                    } else {
                        bool rowIn = (row >= 0 && row < static_cast<int>(n));
                        bool colIn = (col >= 0 && col < static_cast<int>(n));
                        if (rowIn || colIn)
                            f_val = 1.0f;
                        else
                            f_val = 0.0f;
                    }
                    
                    sum += mask[i * m + j] * f_val;
                }
            }
            output[x * n + y] = sum;
        }
    }
}
