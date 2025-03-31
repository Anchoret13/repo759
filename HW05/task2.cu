#include <cstdio>
#include <cstdlib>
#include <ctime>

__global__ void compute_kernel(int a, int *dA) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int index = y * blockDim.x + x;
    
    dA[index] = a * x + y;
}

int main() {
    // Set random seed
    srand(time(nullptr));
    
    // Generate random value for a
    int a = rand() % 10 + 1;  // Random value between 1 and 10
    
    // Allocate device memory
    int *dA;
    cudaMalloc(&dA, 16 * sizeof(int));
    
    // Launch kernel with 2 blocks of 8 threads each
    compute_kernel<<<2, 8>>>(a, dA);
    
    // Copy results back to host
    int hA[16];
    cudaMemcpy(hA, dA, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(dA);
    
    // Print results
    for (int i = 0; i < 16; i++) {
        printf("%d ", hA[i]);
    }
    printf("\n");
    
    return 0;
}