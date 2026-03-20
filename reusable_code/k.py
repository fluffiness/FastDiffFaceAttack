import torch
import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Script for wasting (and occupying) GPU resources.")

parser.add_argument("-d","--dev_no", type=int, help="GPU device number, i.e. CUDA_VISIBLE_DEVICES", default=0)
parser.add_argument("--max", type=float, help="Maximum GPU memory usage rate", default=0.5)

# Parse the arguments
args = parser.parse_args()
args_dict = vars(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.dev_no)

dev_no = os.environ.get('CUDA_VISIBLE_DEVICES')
print(f"CUDA_VISIBLE_DEVICES={dev_no}")

device = torch.device(f"cuda:{dev_no}" if torch.cuda.is_available() else "cpu")
print(f"running on device {device}")
print(f"maximum GPU memory occupancy rate = {args.max}")


def get_gpu_memory(gpu_id):
    """
    Returns the current GPU memory usage and total memory for the specified GPU ID.
    """
    # Run the nvidia-smi command to get GPU info
    cmd = 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i {}'.format(gpu_id)
    output = subprocess.check_output(cmd, shell=True).decode().strip()

    # Parse the output to extract memory usage and total memory
    memory_used, memory_total = map(int, output.split('\n')[0].split(','))

    return memory_used, memory_total

batch_size = 10
matrix_size = 5000
steps=0

while True:
    # Create some tensor to perform operations on
    tensor = torch.randn(batch_size, matrix_size, matrix_size, device=device)
    # Perform some operations on the tensor
    result = torch.bmm(tensor, tensor)
    memory_used, memory_total = get_gpu_memory(dev_no)
    if memory_used < memory_total * args.max:
        batch_size += 5
    steps = (steps + 1) % 100
    print(f"[step: {steps}] Current batch_size={batch_size}. GPU {dev_no} memory usage: {memory_used}/{memory_total} MB", end='\r')