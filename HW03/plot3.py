import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Create results directory if it doesn't exist
os.makedirs('results_3', exist_ok=True)

# Plot 1: Threshold vs Time (linear-log scale)
try:
    # Read threshold benchmark data
    ts_data = pd.read_csv('results_3/ts_benchmark.csv')
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(ts_data['threshold'], ts_data['time'], 'o-', linewidth=2, markersize=8)
    plt.grid(True, which="both", ls="-")
    plt.title('msort Performance vs Threshold (n = 10^6, t = 8)', fontsize=14)
    plt.xlabel('Threshold (ts)', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    
    # Set x-axis to use powers of 2
    plt.xticks(
        [2**i for i in range(1, 11)],
        ['2^{}'.format(i) for i in range(1, 11)]
    )
    
    # Find and mark the optimal threshold
    optimal_idx = ts_data['time'].idxmin()
    optimal_ts = ts_data.loc[optimal_idx, 'threshold']
    optimal_time = ts_data.loc[optimal_idx, 'time']
    
    plt.plot(optimal_ts, optimal_time, 'r*', markersize=15)
    plt.annotate(f'Optimal: ts={optimal_ts}',
                xy=(optimal_ts, optimal_time),
                xytext=(optimal_ts*1.5, optimal_time*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results_3/task3_ts.pdf')
    print(f"Generated threshold plot: results_3/task3_ts.pdf")
except Exception as e:
    print(f"Error generating threshold plot: {e}")

# Plot 2: Thread Count vs Time (linear-linear scale)
try:
    # Read thread benchmark data
    t_data = pd.read_csv('results_3/t_benchmark.csv')
    
    # Try to read optimal threshold from file
    try:
        with open('results_3/optimal_ts.txt', 'r') as f:
            optimal_ts = int(f.read().strip())
    except:
        optimal_ts = 128  # Default if file doesn't exist
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_data['threads'], t_data['time'], 'o-', linewidth=2, markersize=8)
    plt.grid(True)
    plt.title(f'msort Performance vs Number of Threads (n = 10^6, ts = {optimal_ts})', fontsize=14)
    plt.xlabel('Number of Threads (t)', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    
    # Set integer x-axis
    plt.xticks(range(1, 21))
    
    # Find and mark the optimal thread count
    optimal_idx = t_data['time'].idxmin()
    optimal_t = t_data.loc[optimal_idx, 'threads']
    optimal_time = t_data.loc[optimal_idx, 'time']
    
    plt.plot(optimal_t, optimal_time, 'r*', markersize=15)
    plt.annotate(f'Optimal: t={optimal_t}',
                xy=(optimal_t, optimal_time),
                xytext=(optimal_t*1.1, optimal_time*0.9),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    # Calculate speedup
    serial_time = t_data.loc[t_data['threads'] == 1, 'time'].values[0]
    speedup = [serial_time / time for time in t_data['time']]
    
    # Create a second y-axis for speedup
    ax2 = plt.gca().twinx()
    ax2.plot(t_data['threads'], speedup, 'r--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Speedup (relative to 1 thread)', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add ideal speedup line for reference
    ideal_speedup = [t for t in t_data['threads']]
    ax2.plot(t_data['threads'], ideal_speedup, 'g-.', linewidth=1, alpha=0.5, label='Ideal Speedup')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('results_3/task3_t.pdf')
    print(f"Generated thread count plot: results_3/task3_t.pdf")
except Exception as e:
    print(f"Error generating thread count plot: {e}")

# Create summary report
try:
    with open('results_3/benchmark_summary.txt', 'w') as f:
        f.write("Parallel Merge Sort Benchmark Summary\n")
        f.write("====================================\n\n")
        
        # Threshold benchmark summary
        try:
            ts_data = pd.read_csv('results_3/ts_benchmark.csv')
            optimal_ts = ts_data.loc[ts_data['time'].idxmin(), 'threshold']
            f.write(f"Threshold Benchmark (n=1,000,000, t=8):\n")
            f.write(f"- Optimal threshold value: {optimal_ts}\n")
            f.write(f"- Execution times for different thresholds:\n")
            
            for _, row in ts_data.iterrows():
                f.write(f"  ts={row['threshold']}: {row['time']:.2f} ms\n")
            
            f.write("\n")
        except:
            f.write("Threshold benchmark data not available.\n\n")
            
        # Thread count benchmark summary
        try:
            t_data = pd.read_csv('results_3/t_benchmark.csv')
            optimal_t = t_data.loc[t_data['time'].idxmin(), 'threads']
            
            try:
                with open('results_3/optimal_ts.txt', 'r') as ts_file:
                    optimal_ts = int(ts_file.read().strip())
            except:
                optimal_ts = 128  # Default
                
            f.write(f"Thread Count Benchmark (n=1,000,000, ts={optimal_ts}):\n")
            f.write(f"- Optimal thread count: {optimal_t}\n")
            
            serial_time = t_data.loc[t_data['threads'] == 1, 'time'].values[0]
            optimal_time = t_data.loc[t_data['threads'] == optimal_t, 'time'].values[0]
            speedup = serial_time / optimal_time
            efficiency = speedup / optimal_t * 100
            
            f.write(f"- Speedup at optimal thread count: {speedup:.2f}x\n")
            f.write(f"- Parallel efficiency: {efficiency:.2f}%\n")
            f.write(f"- Execution times for different thread counts:\n")
            
            for _, row in t_data.iterrows():
                thread_speedup = serial_time / row['time']
                f.write(f"  t={row['threads']}: {row['time']:.2f} ms (speedup: {thread_speedup:.2f}x)\n")
                
        except:
            f.write("Thread count benchmark data not available.\n")
            
    print("Generated benchmark summary: results_3/benchmark_summary.txt")
except Exception as e:
    print(f"Error generating summary: {e}")