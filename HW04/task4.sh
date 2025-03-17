#!/bin/bash
#SBATCH --job-name=nbody_openmp
#SBATCH --output=nbody_results_%j.out
#SBATCH --error=nbody_results_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Compile the code
g++ task3.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

echo "Compilation completed."

# Parameters
PARTICLES=1000
SIM_TIME=100

# Create results directory
mkdir -p results

# Run experiments with different scheduling policies and thread counts
for SCHEDULE in static dynamic guided; do
    RESULT_FILE="results/nbody_${SCHEDULE}.csv"
    echo "Threads,Runtime_ms" > $RESULT_FILE
    
    echo "Running experiments with $SCHEDULE scheduling..."
    
    for THREADS in 1 2 3 4 5 6 7 8; do
        echo "  Running with $THREADS threads..."
        
        # Run the simulation 3 times and calculate the average
        TOTAL_TIME=0
        RUNS=3
        
        for ((i=1; i<=$RUNS; i++)); do
            OUTPUT=$(./task3 $PARTICLES $SIM_TIME $THREADS $SCHEDULE)
            # Extract the runtime from the output
            TIME=$(echo $OUTPUT | grep -oP 'Time: \K[0-9.]+')
            TOTAL_TIME=$(echo "$TOTAL_TIME + $TIME" | bc)
            echo "    Run $i: $TIME ms"
        done
        
        # Calculate average runtime
        AVG_TIME=$(echo "scale=2; $TOTAL_TIME / $RUNS" | bc)
        echo "  Average runtime with $THREADS threads: $AVG_TIME ms"
        
        # Append to results file
        echo "$THREADS,$AVG_TIME" >> $RESULT_FILE
    done
    
    echo "Completed experiments with $SCHEDULE scheduling."
done

echo "All experiments completed."