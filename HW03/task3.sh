#!/bin/bash

# Alternative approach without using Slurm
# This script will run the benchmarks directly on the current node

# Create results directory
mkdir -p results_3

# Compile the program
echo "Compiling the program..."
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# =====================================================
# Threshold benchmark
# =====================================================
echo "Starting threshold benchmark..."
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

echo "Threshold benchmark completed."

# =====================================================
# Find optimal threshold
# =====================================================
echo "Finding optimal threshold value..."
optimal_ts=$(awk -F, 'NR>1 {if (NR==2 || $2<min) {min=$2; opt=$1}} END {print opt}' results_3/ts_benchmark.csv)
echo "Optimal threshold value is $optimal_ts"
echo "$optimal_ts" > results_3/optimal_ts.txt

# =====================================================
# Thread count benchmark
# =====================================================
echo "Starting thread count benchmark..."
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

echo "Thread count benchmark completed."

# =====================================================
# Generate plots
# =====================================================
echo "Generating plots using Python..."
python3 plot3.py

echo "All benchmarks and plots completed."
echo "Results are in the results_3 directory."