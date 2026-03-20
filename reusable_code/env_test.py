import torch
import os
import subprocess

# check cuda version
command = "nvcc -V"
result = subprocess.run(command, shell=True, capture_output=True, text=True)
print("CUDA version:")
stdout = result.stdout
lines = stdout.split('\n')[:-1]
for line in lines:
    print('    ', end='')
    print(line)

# check torch version
print("torch version: ", torch.__version__)

# check if cuda is available
avail = torch.cuda.is_available()
print("cuda is available: ", avail)
if not avail:
    exit()

# check is CUDA_VISIBLE_DEVICES is set
try:
    print("visible devices: ", os.environ["CUDA_VISIBLE_DEVICES"])
except:
    print("CUDA_VISIBLE_DEVICES not set.")

# Check if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using device: ", device)
print("running matmul...")

# Define the size of the matrices
matrix_size = 100

# Generate random matrices on GPU
matrix1 = torch.randn(matrix_size, matrix_size, device=device)
matrix2 = torch.randn(matrix_size, matrix_size, device=device)

# Multiply the matrices on GPU
result = torch.mm(matrix1, matrix2)

print("matmul successful!!")