import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    results = pd.read_csv('./logs/task1/results.csv')
    print("Successfully loaded data with shape:", results.shape)
    print(results.head())
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

if results.isna().any().any():
    print("Warning: Data contains NaN values")
    results = results.dropna()
    print("After dropping NaN values, shape:", results.shape)

grouped = results.groupby('threads_per_block')

plt.figure(figsize=(12, 8))

colors = ['blue', 'red', 'green', 'purple']
markers = ['o', 's', '^', 'x']
color_idx = 0

for name, group in grouped:
    plt.plot(group['n'], group['time_ms'], 
             marker=markers[color_idx % len(markers)], 
             linestyle='-', 
             color=colors[color_idx % len(colors)],
             linewidth=2,
             label=f'threads_per_block={name}')
    color_idx += 1

plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.xlabel('Matrix Size (n)', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.title('Matrix Multiplication Performance', fontsize=16)

n_values = sorted(results['n'].unique())
scaling_factor = 1e-6
reference_times = [n**3 * scaling_factor for n in n_values]

plt.plot(n_values, reference_times, 'k--', label='O(n³) reference', alpha=0.7, linewidth=2)

plt.legend(fontsize=12)
plt.tight_layout()

try:
    plt.savefig('./task1.pdf')
    print("Plot saved to ./task1.pdf")
except Exception as e:
    print(f"Error saving plot: {e}")

plt.figure(figsize=(12, 8))

pivot_data = results.pivot(index='n', columns='threads_per_block', values='time_ms')
print("\nPivot data summary:")
print(pivot_data.describe())

reference_thread = pivot_data.mean().idxmax()
print(f"Using {reference_thread} threads as reference (slowest on average)")

speedup = pivot_data[reference_thread] / pivot_data

color_idx = 0
for thread in pivot_data.columns:
    if thread != reference_thread:
        plt.plot(speedup.index, speedup[thread], 
                marker=markers[color_idx % len(markers)], 
                linestyle='-', 
                color=colors[color_idx % len(colors)],
                linewidth=2,
                label=f'Speedup: {thread} vs {reference_thread}')
        color_idx += 1

plt.xscale('log', base=2)
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.xlabel('Matrix Size (n)', fontsize=14)
plt.ylabel(f'Speedup relative to {reference_thread} threads', fontsize=14)
plt.title('Performance Improvement by Thread Count', fontsize=16)
plt.axhline(y=1, color='r', linestyle='-', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

try:
    plt.savefig('./task1_speedup.pdf')
    print("Speedup plot saved to ./task1_speedup.pdf")
except Exception as e:
    print(f"Error saving speedup plot: {e}")