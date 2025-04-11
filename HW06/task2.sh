#!/bin/bash
#SBATCH --job-name=task2_benchmark
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G
#SBATCH --output=task2_benchmark.out
#SBATCH --error=task2_benchmark.err
#SBATCH --partition=research
#SBATCH --gres=gpu:gtx1080:1

# Load CUDA module
module load nvidia/cuda/11.8.0

mkdir -p ./logs/task2

nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

R=128
TPB1=1024
TPB2=256

# Reducing the range to make it more likely to complete within the time limit
for i in {10..20}; do
    n=$((2**i))
    echo "Running with n = $n, R = $R, TPB = $TPB1"
    ./task2 $n $R $TPB1 > ./logs/task2/n${n}_R${R}_TPB${TPB1}.txt
    
    echo "Running with n = $n, R = $R, TPB = $TPB2"
    ./task2 $n $R $TPB2 > ./logs/task2/n${n}_R${R}_TPB${TPB2}.txt
done

# Don't try to plot - just save the data
echo "Benchmark completed. Raw results saved in ./logs/task2/"