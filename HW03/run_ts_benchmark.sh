#!/bin/bash
#SBATCH --job-name=msort_ts_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=msort_ts_benchmark.out
#SBATCH --partition=normal

# Create results directory if it doesn't exist
mkdir -p results_3

# Compile the program
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Run benchmarks for different threshold values (ts)
n=1000000
t=8

echo "threshold,time" > results_3/ts_benchmark.csv

for i in {1..10}
do
    # Calculate 2^i
    ts=$((2**i))
    echo "Running with ts = $ts"
    
    # Run 3 times and take the average to get more reliable measurements
    total_time=0
    for run in {1..3}
    do
        output=$(./task3 $n $t $ts)
        # Extract time from the third line of the output
        time=$(echo "$output" | sed -n '3p')
        total_time=$(echo "$total_time + $time" | bc)
    done
    avg_time=$(echo "$total_time / 3" | bc -l)
    
    echo "$ts,$avg_time" >> results_3/ts_benchmark.csv
done

echo "Threshold benchmark results saved in results_3/ts_benchmark.csv"
