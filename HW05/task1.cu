#include <cstdio>

__global__ void factorial_kernel() {
    int tid = threadIdx.x;
    int num = tid + 1;  // Numbers 1 to 8
    
    unsigned long long result = 1;
    for (int i = 1; i <= num; i++) {
        result *= i;
    }
    
    printf("%d!=%llu\n", num, result);
}

int main() {
    factorial_kernel<<<1, 8>>>();
    cudaDeviceSynchronize();
    
    return 0;
}