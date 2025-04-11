import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def get_time_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            try:
                return float(lines[1].strip())
            except:
                return None
    return None

def main():
    # Parameters
    R = 128
    TPB1 = 1024
    TPB2 = 256
    
    # Initialize data structures
    n_values = [2**i for i in range(10, 30)]
    times_tpb1 = []
    times_tpb2 = []
    
    # Read data for TPB1
    for n in n_values:
        filename = f"./logs/task2/n{n}_R{R}_TPB{TPB1}.txt"
        time = get_time_from_file(filename)
        if time is not None:
            times_tpb1.append(time)
        else:
            print(f"Warning: Could not read time from {filename}")
    
    # Read data for TPB2
    for n in n_values:
        filename = f"./logs/task2/n{n}_R{R}_TPB{TPB2}.txt"
        time = get_time_from_file(filename)
        if time is not None:
            times_tpb2.append(time)
        else:
            print(f"Warning: Could not read time from {filename}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot both lines
    if len(times_tpb1) == len(n_values):
        plt.plot(n_values, times_tpb1, 'b-o', label=f'Threads per block = {TPB1}')
    if len(times_tpb2) == len(n_values):
        plt.plot(n_values, times_tpb2, 'r-s', label=f'Threads per block = {TPB2}')
    
    # Set logarithmic scale for x-axis
    plt.xscale('log', base=2)
    
    # Add labels and title
    plt.xlabel('Input Size (n)')
    plt.ylabel('Execution Time (ms)')
    plt.title(f'Stencil Performance (R = {R})')
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Save the plot
    plt.savefig('task2.pdf')
    print("Plot saved as task2.pdf")

if __name__ == "__main__":
    main()