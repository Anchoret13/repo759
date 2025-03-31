#!/bin/bash
#SBATCH --job-name=task1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=task1_output.txt

module load gcc cuda

./task1
