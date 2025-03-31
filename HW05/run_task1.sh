#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=task1_output.txt

# Load specific CUDA version
module load nvidia/cuda/11.8.0

# Compile with the flag to handle compiler compatibility
nvcc -allow-unsupported-compiler task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

# Run the executable
./task1