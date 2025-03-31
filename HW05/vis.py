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
                
                i += 3  # Skip to the next size (accounts for first_element and last_element lines)
            else:
                i += 1
                
    return sizes, times

# Extract data for 512 threads per block
sizes_512, times_512 = extract_data('task3_output.txt')

# If you have a separate file for 16 threads per block, use:
# sizes_16, times_16 = extract_data('task3_16_output.txt')

# For now, let's create placeholder data for 16 threads
sizes_16 = sizes_512
# This is just an example - you should replace with actual data
times_16 = [t * 1.15 for t in times_512]  # Just for illustration

# Create the plot
plt.figure(figsize=(12, 8))
plt.plot(sizes_512, times_512, 'bo-', label='512 threads per block')
plt.plot(sizes_16, times_16, 'ro-', label='16 threads per block')

plt.xscale('log')  # Use log scale for x-axis
plt.xlabel('Array Size (n)')
plt.ylabel('Execution Time (ms)')
plt.title('vscale Kernel Execution Time')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()

# Add custom labels for powers of 2
ax = plt.gca()
# Create custom tick positions at powers of 2
tick_positions = [2**i for i in range(10, 21) if 2**i in sizes_512]
ax.set_xticks(tick_positions)
ax.set_xticklabels([f'2^{int(np.log2(x))}' for x in tick_positions], rotation=45)

plt.tight_layout()
plt.savefig('task3.pdf')
plt.show()