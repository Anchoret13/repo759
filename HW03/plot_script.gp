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
plot 'results/benchmark_data.csv' using 1:2 with linespoints lw 2 pt 7 ps 1 title "Actual Performance",      '' using 1:(T1/$1) with lines dashtype 2 lc rgb "red" title "Ideal Speedup"
