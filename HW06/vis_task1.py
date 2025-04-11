import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results = pd.read_csv('./logs/task1/results.csv')

grouped = results.groupby('threads_per_block')

plt.figure(figsize=(12, 8))

for name, group in grouped:
    plt.plot(group['n'], group['time_ms'], marker='o', linestyle='-', label=f'threads_per_block={name}')

plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.xlabel('Matrix Size (n)')
plt.ylabel('Execution Time (ms)')
plt.title('Matrix Multiplication Performance')
plt.legend()

n_values = sorted(results['n'].unique())
reference_times = [n**3/10**7 for n in n_values]
plt.plot(n_values, reference_times, 'k--', label='O(n³) reference')

plt.legend()
plt.tight_layout()
plt.savefig('./task1.pdf')
print("Plot saved to ./task1.pdf")

plt.figure(figsize=(12, 8))

pivot_data = results.pivot(index='n', columns='threads_per_block', values='time_ms')
columns = pivot_data.columns

reference_thread = columns[0]  
speedup = pivot_data[reference_thread] / pivot_data

for thread in columns:
    if thread != reference_thread:
        plt.plot(speedup.index, speedup[thread], marker='o', linestyle='-', 
                 label=f'Speedup: {thread} vs {reference_thread}')

plt.xscale('log', base=2)
plt.grid(True, which="both", ls="--")
plt.xlabel('Matrix Size (n)')
plt.ylabel(f'Speedup relative to {reference_thread} threads')
plt.title('Performance Improvement by Thread Count')
plt.axhline(y=1, color='r', linestyle='-')
plt.legend()
plt.tight_layout()
plt.savefig('./task1_speedup.pdf')
print("Speedup plot saved to ./task1_speedup.pdf")