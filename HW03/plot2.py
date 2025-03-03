import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('results_2/benchmark_data.txt')
threads = data[:, 0]
times = data[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(threads, times, 'o-', linewidth=2, markersize=8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Number of Threads', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('Convolution Performance Scaling (n=1024)', fontsize=14)

single_thread_time = times[0]
theoretical_times = [single_thread_time/t for t in threads]
plt.plot(threads, theoretical_times, 'r--', alpha=0.7, label='Ideal Scaling')

plt.legend(fontsize=10)

max_threads_cpu = 20  
plt.axvline(x=max_threads_cpu, color='gray', linestyle=':', alpha=0.7)
plt.text(max_threads_cpu+0.2, max(times)*0.9, 'Max Available Threads', 
         rotation=90, va='top', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('results_2/task2.pdf')
