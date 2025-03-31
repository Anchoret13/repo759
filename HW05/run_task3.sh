#!/bin/bash
#SBATCH --job-name=task3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=task3_output.txt

module load gcc cuda

# Run with different array sizes
for n in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728 268435456 536870912
do
    echo "Running with n = $n"
    ./task3 $n
done