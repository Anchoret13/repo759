#!/bin/bash
#SBATCH --job-name=task2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=task2_output.txt

module load gcc cuda

./task2