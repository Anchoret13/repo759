#!/bin/bash
#SBATCH --job-name=task1_hw07
#SBATCH --partition=interactive
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=task1_output.txt
#SBATCH --error=task1_error.txt

module load nvidia/cuda

rm -rf ./logs/task1
mkdir -p ./logs/task1

# Use -allow-unsupported-compiler flag to override GCC version check
nvcc task1.cu matmul.cu -allow-unsupported-compiler -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

echo "n,block_dim,type,time_ms,first_element,last_element" > ./logs/task1/results.csv

for i in {5..14}
do
    n=$((2**i))
    
    for block_dim in 8 16 32
    do
        echo "Running matrix size n=$n with block_dim=$block_dim"
        output=$(./task1 $n $block_dim 2>&1)
        
        # Extract the values (each type has 3 lines: first element, last element, time)
        int_first=$(echo "$output" | sed -n '1p')
        int_last=$(echo "$output" | sed -n '2p')
        int_time=$(echo "$output" | sed -n '3p')
        
        float_first=$(echo "$output" | sed -n '4p')
        float_last=$(echo "$output" | sed -n '5p')
        float_time=$(echo "$output" | sed -n '6p')
        
        double_first=$(echo "$output" | sed -n '7p')
        double_last=$(echo "$output" | sed -n '8p')
        double_time=$(echo "$output" | sed -n '9p')
        
        echo "$n,$block_dim,int,$int_time,$int_first,$int_last" >> ./logs/task1/results.csv
        echo "$n,$block_dim,float,$float_time,$float_first,$float_last" >> ./logs/task1/results.csv
        echo "$n,$block_dim,double,$double_time,$double_first,$double_last" >> ./logs/task1/results.csv
    done
done

echo "Benchmark completed. Results saved in ./logs/task1/results.csv"
echo "To generate the plot, run: python3 plot_task1.py"