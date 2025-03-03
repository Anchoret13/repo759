#!/bin/bash
#SBATCH --job-name=mmul_benchmark
#SBATCH --output=mmul_benchmark_%j.out
#SBATCH --error=mmul_benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=research

# Check for available modules
echo "Checking for available modules..."
module spider gcc
module spider gnuplot

# Try to load a compatible gcc module
if module -t list 2>&1 | grep -q "gcc"; then
    # If any gcc module is loaded, use that
    echo "Using currently loaded gcc module"
elif module avail gcc 2>&1 | grep -q "gcc/"; then
    # Try to load the available gcc module
    available_gcc=$(module avail gcc 2>&1 | grep "gcc/" | head -1 | awk '{print $1}')
    echo "Loading available gcc module: $available_gcc"
    module load $available_gcc
else
    echo "No gcc module found, using system compiler"
fi

# Check if gnuplot is available
if command -v gnuplot >/dev/null 2>&1; then
    echo "gnuplot is available in the system path"
else
    echo "WARNING: gnuplot is not available. Will create the script but may not be able to generate the plot."
fi

# Create results directory if it doesn't exist
mkdir -p results

# Compile the program
g++ task1.cpp matmul.cpp -Wall -O3 -std=c++17 -o task1 -fopenmp

# Create a header for the CSV file
echo "threads,time_ms" > results/benchmark_data.csv

# Run for different thread counts and save results
for t in {1..20}
do
    echo "Running with $t threads..."
    
    # Set OMP_NUM_THREADS environment variable
    export OMP_NUM_THREADS=$t
    
    # Run the program
    output=$(./task1 1024 $t)
    
    # Extract the time (third line of output)
    time_ms=$(echo "$output" | sed -n '3p')
    
    # Save to CSV
    echo "$t,$time_ms" >> results/benchmark_data.csv
    
    echo "Thread count $t completed: $time_ms ms"
done

echo "Benchmark completed. Results saved to results/benchmark_data.csv"

# Save data in a format that's easy to plot with any tool
echo "Saving benchmark data in plotting-friendly format..."

# Calculate the ideal speedup
T1=$(awk 'NR==2 {print $2}' results/benchmark_data.csv | tr -d '\r')
echo "Single-thread time: $T1 ms"

# Create a data file with actual and ideal performance
echo "# Threads  Actual_Time(ms)  Ideal_Time(ms)" > results/plot_data.txt
awk -v t1="$T1" 'NR>1 {printf "%d  %.2f  %.2f\n", $1, $2, t1/$1}' FS="," results/benchmark_data.csv >> results/plot_data.txt

echo "Plot data saved to results/plot_data.txt"
echo "To generate the plot, you can use any plotting tool with this data file"

# Try to use gnuplot if available
if command -v gnuplot >/dev/null 2>&1; then
    echo "Attempting to generate plot with gnuplot..."
    
    cat > plot_script.gp << EOF
# Output to PDF file
set terminal pdf enhanced color size 8,5
set output "task1.pdf"

# Set up the plot style
set title "Matrix Multiplication Performance (n=1024)"
set xlabel "Number of Threads (t)"
set ylabel "Execution Time (ms)"
set grid
set key top right

# Set x-axis range and tics
set xrange [0:21]
set xtics 2

# Plot the data
plot 'results/plot_data.txt' using 1:2 with linespoints lw 2 pt 7 ps 1 title "Actual Performance", \
     '' using 1:3 with lines dashtype 2 lc rgb "red" title "Ideal Speedup"
EOF

    # Run gnuplot
    gnuplot plot_script.gp && echo "Plot saved as task1.pdf" || echo "Failed to generate plot with gnuplot"
else
    echo "Gnuplot not available. Please generate the plot manually using results/plot_data.txt"
    
    # Create a simple ASCII plot as a fallback
    echo -e "\nSimple ASCII representation of the data:"
    echo "Threads | Time (ms)"
    echo "--------|----------"
    awk -F, 'NR>1 {printf "%7d | %9.2f\n", $1, $2}' results/benchmark_data.csv
fi