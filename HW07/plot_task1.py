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

# Find best performing block_dim for n=2^14 if available
n_max = 2**14
if n_max in results['n'].values:
    max_data = results[results['n'] == n_max]
    if not max_data.empty:
        best_results = max_data.loc[max_data.groupby('type')['time_ms'].idxmin()]
        print("\nBest performing block_dim for n=2^14:")
        print(best_results[['type', 'block_dim', 'time_ms']])
    else:
        print(f"\nNo data available for n={n_max}")
else:
    print(f"\nNo data available for n={n_max}, largest n is {results['n'].max()}")
    # Print best results for largest available n instead
    largest_n = results['n'].max()
    if largest_n > 0:
        print(f"Best performing block_dim for the largest available n={largest_n}:")
        largest_data = results[results['n'] == largest_n]
        best_largest = largest_data.loc[largest_data.groupby('type')['time_ms'].idxmin()]
        print(best_largest[['type', 'block_dim', 'time_ms']])

# Compare performance across data types with available data
plt.figure(figsize=(12, 8))

data_by_type = {}
for dtype in ['int', 'float', 'double']:
    type_data = results[results['type'] == dtype]
    if not type_data.empty:
        # Find best block_dim for each type based on average performance
        avg_perf = type_data.groupby('block_dim')['time_ms'].mean()
        if not avg_perf.empty:
            best_block_dim = avg_perf.idxmin()
            # Filter data for this type and its best block_dim
            filtered_data = type_data[type_data['block_dim'] == best_block_dim]
            if not filtered_data.empty:
                plt.plot(filtered_data['n'], filtered_data['time_ms'], 
                        marker='o', linestyle='-', linewidth=2,
                        label=f'{dtype}, best block_dim={best_block_dim}')
                data_by_type[dtype] = filtered_data

if data_by_type:
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
else:
    print("Not enough data to create type comparison plot")