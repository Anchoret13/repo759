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

# Create plots for different data types
plt.figure(figsize=(12, 8))

for dtype in ['int', 'float', 'double']:
    type_data = results[results['type'] == dtype]
    
    for block_dim in sorted(type_data['block_dim'].unique()):
        block_data = type_data[type_data['block_dim'] == block_dim]
        plt.plot(block_data['n'], block_data['time_ms'], 
                marker='o', linestyle='-', linewidth=2,
                label=f'{dtype}, block_dim={block_dim}')

plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.xlabel('Matrix Size (n)', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.title('Matrix Multiplication Performance', fontsize=16)

# Add O(n³) reference line
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

# Find best performing block_dim for n=2^14
n_max = 2**14
max_data = results[results['n'] == n_max]
best_results = max_data.loc[max_data.groupby('type')['time_ms'].idxmin()]
print("\nBest performing block_dim for n=2^14:")
print(best_results[['type', 'block_dim', 'time_ms']])

# Compare performance across data types
plt.figure(figsize=(12, 8))

# For the best block_dim for each data type
best_block_dims = {}
for dtype in ['int', 'float', 'double']:
    type_data = results[results['type'] == dtype]
    best_block_dim = max_data[max_data['type'] == dtype].loc[max_data[max_data['type'] == dtype]['time_ms'].idxmin()]['block_dim']
    best_block_dims[dtype] = best_block_dim
    
    # Filter data for this type and its best block_dim
    filtered_data = type_data[type_data['block_dim'] == best_block_dim]
    plt.plot(filtered_data['n'], filtered_data['time_ms'], 
            marker='o', linestyle='-', linewidth=2,
            label=f'{dtype}, best block_dim={best_block_dim}')

plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.xlabel('Matrix Size (n)', fontsize=14)
plt.ylabel('Execution Time (ms)', fontsize=14)
plt.title('Matrix Multiplication Performance by Data Type (Best Block Dim)', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()

try:
    plt.savefig('./task1_by_type.pdf')
    print("Type comparison plot saved to ./task1_by_type.pdf")
except Exception as e:
    print(f"Error saving type comparison plot: {e}")