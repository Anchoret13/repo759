#!/bin/bash
#SBATCH --job-name=matmul_benchmark
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --output=matmul_benchmark.out
#SBATCH --error=matmul_benchmark.err
#SBATCH --gres=gpu:1

mkdir -p ./logs/task1

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

THREADS_PER_BLOCK_1=1024
THREADS_PER_BLOCK_2=256

echo "n,threads_per_block,time_ms,last_element" > ./logs/task1/results.csv

for i in {5..14}
do
    n=$((2**i))
    output=$(./task1 $n $THREADS_PER_BLOCK_1)
    last_element=$(echo "$output" | head -n 1)
    time_ms=$(echo "$output" | tail -n 1)
    echo "$n,$THREADS_PER_BLOCK_1,$time_ms,$last_element" >> ./logs/task1/results.csv
    
    output=$(./task1 $n $THREADS_PER_BLOCK_2)
    last_element=$(echo "$output" | head -n 1)
    time_ms=$(echo "$output" | tail -n 1)
    echo "$n,$THREADS_PER_BLOCK_2,$time_ms,$last_element" >> ./logs/task1/results.csv
done

module load python/3.9.13
python3 vis_task1.py

echo "Benchmark completed. Results and plots saved in ./logs/task1/"