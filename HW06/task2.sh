#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=task2_output.txt
#SBATCH --error=task2_error.txt

rm -f task2_benchmark.out task2_benchmark.err

module load nvidia/cuda/11.8.0

rm -rf ./logs/task2
mkdir -p ./logs/task2

nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

R=128
TPB1=1024
TPB2=512

for i in {10..29}; do
    n=$((2**i))
    echo "Running with n = $n, R = $R, TPB = $TPB1"
    ./task2 $n $R $TPB1 > ./logs/task2/n${n}_R${R}_TPB${TPB1}.txt
    
    echo "Running with n = $n, R = $R, TPB = $TPB2"
    ./task2 $n $R $TPB2 > ./logs/task2/n${n}_R${R}_TPB${TPB2}.txt
done

echo "Benchmark completed. Raw results saved in ./logs/task2/"
