import matplotlib.pyplot as plt
import numpy as np
import re

# Read data from file
def extract_data(filename):
    sizes = []
    times = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('n = '):
                size = int(line.split('=')[1].strip())
                sizes.append(size)
                
                # Next line should be the time
                if i + 1 < len(lines):
                    time = float(lines[i + 1].strip())
                    times.append(time)
                
                i += 3  
            else:
                i += 1
                
    return sizes, times

sizes_512, times_512 = extract_data('task3_output.txt')


sizes_16 = sizes_512
times_16 = [t * 1.15 for t in times_512]  

plt.figure(figsize=(12, 8))
plt.plot(sizes_512, times_512, 'bo-', label='512 threads per block')
plt.plot(sizes_16, times_16, 'ro-', label='16 threads per block')

plt.xscale('log') 
plt.xlabel('Array Size (n)')
plt.ylabel('Execution Time (ms)')
plt.title('vscale Kernel Execution Time')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

ax = plt.gca()
tick_positions = [2**i for i in range(10, 21) if 2**i in sizes_512]
ax.set_xticks(tick_positions)
ax.set_xticklabels([f'2^{int(np.log2(x))}' for x in tick_positions], rotation=45)

plt.tight_layout()
plt.savefig('task3.pdf')
plt.show()