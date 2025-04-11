#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=task1_output.txt
#SBATCH --error=task1_error.txt

rm -f matmul_benchmark.out matmul_benchmark.err

module load nvidia/cuda/11.8.0

rm -rf ./logs/task1
mkdir -p ./logs/task1

nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

THREADS_PER_BLOCK_1=1024
THREADS_PER_BLOCK_2=256

echo "n,threads_per_block,time_ms,last_element" > ./logs/task1/results.csv

# Start with small matrix sizes for testing
for i in {2..14}
do
    n=$((2**i))
    echo "Running matrix size n=$n with $THREADS_PER_BLOCK_1 threads per block"
    output=$(./task1 $n $THREADS_PER_BLOCK_1 2>&1)
    echo "Raw output: $output"
    
    # Extract the last two lines only for the CSV
    last_element=$(echo "$output" | grep -v "Starting with" | grep -v "First few values" | grep -v "Result calculation" | head -n 1)
    time_ms=$(echo "$output" | grep -v "Starting with" | grep -v "First few values" | grep -v "Result calculation" | tail -n 1)
    
    echo "Extracted: last_element=$last_element, time_ms=$time_ms"
    echo "$n,$THREADS_PER_BLOCK_1,$time_ms,$last_element" >> ./logs/task1/results.csv
    
    echo "Running matrix size n=$n with $THREADS_PER_BLOCK_2 threads per block"
    output=$(./task1 $n $THREADS_PER_BLOCK_2 2>&1)
    echo "Raw output: $output"
    
    # Extract the last two lines only for the CSV
    last_element=$(echo "$output" | grep -v "Starting with" | grep -v "First few values" | grep -v "Result calculation" | head -n 1)
    time_ms=$(echo "$output" | grep -v "Starting with" | grep -v "First few values" | grep -v "Result calculation" | tail -n 1)
    
    echo "Extracted: last_element=$last_element, time_ms=$time_ms"
    echo "$n,$THREADS_PER_BLOCK_2,$time_ms,$last_element" >> ./logs/task1/results.csv
done

echo "Benchmark completed. Results saved in ./logs/task1/results.csv"