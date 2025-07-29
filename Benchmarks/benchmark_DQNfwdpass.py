import subprocess
import matplotlib.pyplot as plt
import re

# Define batch sizes to test
batch_sizes = [64, 128, 256, 512, 1024]

# Paths to source files (inside DQNfwdpass folder)
openmp_source = "DQNfwdpass/dqn_openmp.cpp"
cuda_source = "DQNfwdpass/dqn_cuda.cu"

# Executables (inside DQNfwdpass folder)
openmp_exec = "./DQNfwdpass/dqn_openmp"
cuda_exec = "./DQNfwdpass/dqn_cuda"

# Compile both versions first
print("Compiling OpenMP version...")
subprocess.run(["g++", "-fopenmp", "-O3", openmp_source, "-o", openmp_exec[2:]], check=True)

print("Compiling CUDA version...")
subprocess.run(["nvcc", "-O3", cuda_source, "-o", cuda_exec[2:]], check=True)

# Storage for timing data
openmp_times = []
cuda_times = []

def extract_time(output_str, label):
    match = re.search(rf"{label}.*?([\d.]+)", output_str)
    return float(match.group(1)) if match else None

# Loop over batch sizes and run tests
for batch in batch_sizes:
    print(f"\nRunning batch size {batch}")

    # Run OpenMP version
    result = subprocess.run([openmp_exec, str(batch)], capture_output=True, text=True)
    print(result.stdout.strip())
    time = extract_time(result.stdout, "OpenMP")
    openmp_times.append(time)

    # Run CUDA version
    result = subprocess.run([cuda_exec, str(batch)], capture_output=True, text=True)
    print(result.stdout.strip())
    time = extract_time(result.stdout, "CUDA")
    cuda_times.append(time)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, openmp_times, marker='o', label='OpenMP (CPU)', color='blue')
plt.plot(batch_sizes, cuda_times, marker='s', label='CUDA (GPU)', color='orange')

plt.xlabel("Batch Size")
plt.ylabel("Forward Pass Time (seconds)")
plt.title("OpenMP vs CUDA: Deep Q-Network Forward Pass Timing")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_timing_plot.png", dpi=300)
plt.show()
