#!/bin/bash
#SBATCH --job-name=msort_t_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=msort_t_benchmark.out
#SBATCH --partition=normal

# Create results directory if it doesn't exist
mkdir -p results_3

# Compile the program
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Set the optimal threshold value from the previous benchmark
# Change this value based on the results from run_ts_benchmark.sh
optimal_ts=128  # Example value - replace with your actual optimal ts

# Run benchmarks for different thread counts (t)
n=1000000

echo "threads,time" > results_3/t_benchmark.csv

for t in {1..20}
do
    echo "Running with t = $t"
    
    # Run 3 times and take the average to get more reliable measurements
    total_time=0
    for run in {1..3}
    do
        output=$(./task3 $n $t $optimal_ts)
        # Extract time from the third line of the output
        time=$(echo "$output" | sed -n '3p')
        total_time=$(echo "$total_time + $time" | bc)
    done
    avg_time=$(echo "$total_time / 3" | bc -l)
    
    echo "$t,$avg_time" >> results_3/t_benchmark.csv
done

echo "Thread benchmark results saved in results_3/t_benchmark.csv"