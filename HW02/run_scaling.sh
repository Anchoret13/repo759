#!/bin/bash
#SBATCH --job-name=task1_scaling
#SBATCH --output=task1_scaling.out
#SBATCH --error=task1_scaling.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1

# Uncomment and adjust the following line if your system requires module initialization
# source /etc/profile.d/modules.sh

# If necessary, load modules (uncomment if modules are available)
# module load g++
# module load gnuplot

# Compile the code
g++ scan.cpp task1.cpp -Wall -O3 -std=c++17 -o task1

# Remove any old results file
rm -f results.dat

# Loop over exponents 10 to 30, so that n = 2^10, 2^11, ..., 2^30
for exp in {10..30}; do
    n=$(( 2**exp ))
    # Run task1; capture the first output line (the execution time in milliseconds)
    time_ms=$(./task1 "$n" | head -n 1)
    echo "$n $time_ms" >> results.dat
done

# Check if gnuplot is available before plotting
if command -v gnuplot &> /dev/null; then
    gnuplot <<EOF
set terminal pdf
set output "task1.pdf"
set title "Scaling Analysis of Inclusive Scan"
set xlabel "Array Size (n)"
set ylabel "Time (ms)"
set logscale x 2
plot "results.dat" using 1:2 with linespoints title "Scan Time"
EOF
else
    echo "gnuplot is not available. Please plot results.dat using your preferred plotting tool."
fi
