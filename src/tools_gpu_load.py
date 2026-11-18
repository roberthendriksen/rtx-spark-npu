"""Keep the GPU saturated from a separate process (used by 09_gpu_contention.py).

Runs in its own environment (.venv-gpuload, NVIDIA's PyTorch build), because TensorRT on Windows
on Arm cannot build GPU engines. Prints READY once warm, then "IPS <unix time> <iterations/s>".
"""
import sys
import time

import torch

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
x = torch.randn(6144, 6144, device="cuda", dtype=torch.float16)
for _ in range(5):
    (x @ x).sum().item()
print("READY", torch.__version__, torch.cuda.get_device_name(0), flush=True)
t_start = t_last = time.perf_counter()
n = 0
while time.perf_counter() - t_start < seconds:
    y = x @ x
    torch.cuda.synchronize()
    n += 1
    now = time.perf_counter()
    if now - t_last >= 1.0:
        print(f"IPS {time.time():.3f} {n / (now - t_last):.2f}", flush=True)
        n, t_last = 0, now
