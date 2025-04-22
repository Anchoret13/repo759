#!/bin/bash
#SBATCH --job-name=task2_hw07
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=task2_output.txt
#SBATCH --error=task2_error.txt

module load nvidia/cuda

rm -rf ./logs/task2
mkdir -p ./logs/task2

rm ./task2_output.txt
rm ./task2_error.txt
rm ./task1

# Use -allow-unsupported-compiler flag to override GCC version check
nvcc task2.cu reduce.cu -allow-unsupported-compiler -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

# Define the threads per block values to test
TPB1=1024
TPB2=256

echo "N,threads_per_block,time_ms,result" > ./logs/task2/results.csv

# Run from 2^10 to 2^30
for i in {10..30}
do
    N=$((2**i))
    
    echo "Running with N = $N, threads_per_block = $TPB1"
    output=$(./task2 $N $TPB1 2>&1)
    result=$(echo "$output" | sed -n '1p')
    time_ms=$(echo "$output" | sed -n '2p')
    echo "$N,$TPB1,$time_ms,$result" >> ./logs/task2/results.csv
    
    echo "Running with N = $N, threads_per_block = $TPB2"
    output=$(./task2 $N $TPB2 2>&1)
    result=$(echo "$output" | sed -n '1p')
    time_ms=$(echo "$output" | sed -n '2p')
    echo "$N,$TPB2,$time_ms,$result" >> ./logs/task2/results.csv
done

echo "Benchmark completed. Results saved in ./logs/task2/results.csv"
echo "To generate the plot, run: python3 plot_task2.py"