#!/bin/bash
#SBATCH --job-name=task3
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=task3_output.txt

# Load specific CUDA version
module load nvidia/cuda/11.8.0

# Compile with the flag to handle compiler compatibility
nvcc -allow-unsupported-compiler vscale.cu task3.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

# Run with different array sizes - we'll use a smaller set to make sure it completes
echo "Running task3 with 512 threads per block"
for n in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576
do
    echo "n = $n"
    ./task3 $n
done

# Modify task3.cu to use 16 threads per block for the second part
# Note: This would require modifying your task3.cu to accept a second parameter for thread count
# If your task3.cu doesn't support this, you'll need to modify it