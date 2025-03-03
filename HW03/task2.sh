#!/bin/bash
#SBATCH --job-name=conv_bench
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --output=conv_bench_%j.out

# Compile the program
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

# Matrix size
N=1024

# Output file
OUTPUT_FILE="results_2/benchmark_data.txt"

# Clear output file
> $OUTPUT_FILE

# Run benchmark for each thread count
for t in {1..20}
do
    echo "Running with $t threads..."
    # Run 3 times and take the average for more reliable results
    TOTAL_TIME=0
    RUNS=3
    
    for run in $(seq 1 $RUNS)
    do
        # Run the program and capture the output
        result=$(./task2 $N $t)
        # Extract the time (third line of output)
        time=$(echo "$result" | sed -n '3p')
        TOTAL_TIME=$(echo "$TOTAL_TIME + $time" | bc)
    done
    
    # Calculate average time
    AVG_TIME=$(echo "scale=2; $TOTAL_TIME / $RUNS" | bc)
    
    # Save to output file
    echo "$t $AVG_TIME" >> $OUTPUT_FILE
    echo "Average time for $t threads: $AVG_TIME ms"
done

echo "Benchmark completed. Results saved to $OUTPUT_FILE"
