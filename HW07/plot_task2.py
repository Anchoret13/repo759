import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    results = pd.read_csv('./logs/task2/results.csv')
    print("Successfully loaded data with shape:", results.shape)
    print(results.head())
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

if results.isna().any().any():
    print("Warning: Data contains NaN values")
    results = results.dropna()
    print("After dropping NaN values, shape:", results.shape)

plt.figure(figsize=(12, 8))

# Group by threads_per_block and plot
for tpb in sorted(results['threads_per_block'].unique()):
    tpb_data = results[results['threads_per_block'] == tpb]
    plt.plot(tpb_data['N'], tpb_data['time_ms'], 
             marker='o', linestyle='-', linewidth=2,
             label=f'threads_per_block={tpb}')

plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.xlabel('Array Size (N)', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.title('Parallel Reduction Performance', fontsize=16)

# Add O(log n) reference line
n_values = sorted(results['N'].unique())
scaling_factor = 0.1
reference_times = [np.log2(n) * scaling_factor for n in n_values]
plt.plot(n_values, reference_times, 'k--', label='O(log n) reference', alpha=0.7, linewidth=2)

plt.legend(fontsize=12)
plt.tight_layout()

try:
    plt.savefig('./task2.pdf')
    print("Plot saved to ./task2.pdf")
except Exception as e:
    print(f"Error saving plot: {e}")

# Calculate speedup between the two thread configurations
plt.figure(figsize=(12, 8))

tpb_values = sorted(results['threads_per_block'].unique())
if len(tpb_values) >= 2:
    tpb_baseline = min(tpb_values)
    tpb_compare = max(tpb_values)
    
    baseline_data = results[results['threads_per_block'] == tpb_baseline].set_index('N')
    compare_data = results[results['threads_per_block'] == tpb_compare].set_index('N')
    
    common_n = sorted(set(baseline_data.index) & set(compare_data.index))
    speedup = [baseline_data.loc[n, 'time_ms'] / compare_data.loc[n, 'time_ms'] for n in common_n]
    
    plt.plot(common_n, speedup, marker='o', linestyle='-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
    
    plt.xscale('log', base=2)
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Array Size (N)', fontsize=14)
    plt.ylabel(f'Speedup ({tpb_compare} vs {tpb_baseline} threads)', fontsize=14)
    plt.title('Parallel Reduction Speedup', fontsize=16)
    
    try:
        plt.savefig('./task2_speedup.pdf')
        print("Speedup plot saved to ./task2_speedup.pdf")
    except Exception as e:
        print(f"Error saving speedup plot: {e}")
else:
    print("Not enough thread configurations to calculate speedup")