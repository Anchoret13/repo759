import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def get_time_from_file(filename):
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                try:
                    time_value = float(lines[1].strip())
                    print(f"File: {filename}, Time: {time_value}")
                    return time_value
                except ValueError as e:
                    print(f"Error parsing time in {filename}: {e}")
                    return None
            else:
                print(f"Not enough lines in {filename}")
                return None
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    return None

def main():
    R = 128
    TPB1 = 1024
    TPB2 = 512
    
    n_values = [2**i for i in range(10, 30)]
    times_tpb1 = []
    times_tpb2 = []
    
    valid_n_tpb1 = []
    valid_n_tpb2 = []
    
    print("Reading TPB1 data...")
    for n in n_values:
        filename = f"./logs/task2/n{n}_R{R}_TPB{TPB1}.txt"
        time = get_time_from_file(filename)
        if time is not None and time > 0:
            times_tpb1.append(time)
            valid_n_tpb1.append(n)
        else:
            print(f"Warning: Invalid time from {filename}")
    
    print("\nReading TPB2 data...")
    for n in n_values:
        filename = f"./logs/task2/n{n}_R{R}_TPB{TPB2}.txt"
        time = get_time_from_file(filename)
        if time is not None and time > 0:
            times_tpb2.append(time)
            valid_n_tpb2.append(n)
        else:
            print(f"Warning: Invalid time from {filename}")
    
    plt.figure(figsize=(12, 8))
    
    print(f"\nTPB1 data points: {len(times_tpb1)}")
    print(f"TPB2 data points: {len(times_tpb2)}")
    
    if len(times_tpb1) > 0:
        plt.plot(valid_n_tpb1, times_tpb1, 'b-o', linewidth=2, label=f'Threads per block = {TPB1}')
        print(f"Plotted TPB1 data: {list(zip(valid_n_tpb1, times_tpb1))}")
    
    if len(times_tpb2) > 0:
        plt.plot(valid_n_tpb2, times_tpb2, 'r-s', linewidth=2, label=f'Threads per block = {TPB2}')
        print(f"Plotted TPB2 data: {list(zip(valid_n_tpb2, times_tpb2))}")
    
    plt.xscale('log', base=2)
    
    if len(times_tpb1) + len(times_tpb2) > 0:
        plt.yscale('log')
    
    plt.xlabel('Input Size (n)', fontsize=14)
    plt.ylabel('Execution Time (ms)', fontsize=14)
    plt.title(f'Stencil Performance (R = {R})', fontsize=16)
    
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    
    reference_n = valid_n_tpb1 if len(valid_n_tpb1) > 0 else valid_n_tpb2
    if len(reference_n) > 0:
        scaling_factor = 1e-6
        reference_times = [n * scaling_factor for n in reference_n]
        plt.plot(reference_n, reference_times, 'k--', alpha=0.7, linewidth=2, label='O(n) reference')
    
    plt.legend()
    plt.tight_layout()
    
    try:
        plt.savefig('task2.pdf')
        print("Plot saved as task2.pdf")
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    main()