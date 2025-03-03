#!/bin/bash
#SBATCH --job-name=mmul_benchmark
#SBATCH --output=mmul_benchmark_%j.out
#SBATCH --error=mmul_benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# Create results directory if it doesn't exist
mkdir -p results

# Compile the program
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

# Run for different thread counts and save results
echo "threads,time_ms" > results/benchmark_data.csv

for t in {1..20}
do
    echo "Running with $t threads..."
    output=$(./task1 1024 $t)
    
    # Extract the time (third line of output)
    time_ms=$(echo "$output" | sed -n '3p')
    
    # Save to CSV
    echo "$t,$time_ms" >> results/benchmark_data.csv
done

echo "Benchmark completed. Results saved to results/benchmark_data.csv"

# Generate the plot using gnuplot
gnuplot << EOF
# Output to PDF file
set terminal pdfcairo enhanced color font "Helvetica,12" size 8,5
set output "task1.pdf"

# Set up the plot style
set title "Matrix Multiplication Performance (n=1024)" font "Helvetica,14"
set xlabel "Number of Threads (t)" font "Helvetica,12"
set ylabel "Execution Time (ms)" font "Helvetica,12"
set grid
set key top right

# Set x-axis range and tics
set xrange [0:21]
set xtics 2

# Read the first data point to calculate ideal speedup
stats 'results/benchmark_data.csv' using 1:2 every ::1::1 nooutput
T1 = STATS_min_y

# Plot the data
plot 'results/benchmark_data.csv' using 1:2 with linespoints lw 2 pt 7 ps 1 title "Actual Performance", \
     '' using 1:(T1/\$1) with lines dashtype 2 lc rgb "red" title "Ideal Speedup"
EOF

echo "Plot saved as task1.pdf"
