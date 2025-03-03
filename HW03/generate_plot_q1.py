import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('./results/plot_data.txt', skiprows=1)
threads = data[:, 0]
actual_time = data[:, 1]
ideal_time = data[:, 2]

plt.figure(figsize=(10, 6))

plt.plot(threads, actual_time, 'o-', linewidth=2, markersize=8, label='Actual Performance')

plt.plot(threads, ideal_time, 'r--', linewidth=2, label='Ideal Speedup')

plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Number of Threads (t)', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.title('Matrix Multiplication Performance (n=1024)', fontsize=14)

plt.xlim(0, 21)
plt.xticks(range(0, 21, 2))
plt.ylim(bottom=0)  
plt.legend(fontsize=10)

t1 = actual_time[0]
t_max = actual_time[-1]
speedup = t1 / t_max
plt.annotate(f'Maximum Speedup: {speedup:.2f}x',
             xy=(0.02, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('task1.pdf', dpi=300, format='pdf')
print('Plot saved as task1.pdf')
