#!/bin/bash
#SBATCH --job-name=task2_benchmark
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --output=task2_benchmark.out
#SBATCH --error=task2_benchmark.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=cuda

mkdir -p ./logs/task2

nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

R=128
TPB1=1024
TPB2=256

for i in {10..29}; do
    n=$((2**i))
    echo "Running with n = $n, R = $R, TPB = $TPB1"
    ./task2 $n $R $TPB1 > ./logs/task2/n${n}_R${R}_TPB${TPB1}.txt
    
    echo "Running with n = $n, R = $R, TPB = $TPB2"
    ./task2 $n $R $TPB2 > ./logs/task2/n${n}_R${R}_TPB${TPB2}.txt
done

python vis_task2.py