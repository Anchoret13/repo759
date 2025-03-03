#!/bin/bash

# Extract the optimal threshold value (with minimum execution time)
optimal_ts=$(awk -F, 'NR>1 {if (NR==2 || $2<min) {min=$2; opt=$1}} END {print opt}' results_3/ts_benchmark.csv)

echo "Optimal threshold value is $optimal_ts"

# Update the second benchmark script with this value
sed -i "s/optimal_ts=.*  # Example value/optimal_ts=$optimal_ts  # Optimal value/" run_t_benchmark.sh

# Save the optimal threshold to a file for later reference
echo "$optimal_ts" > results_3/optimal_ts.txt
