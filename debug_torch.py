import torch
import os

print(f"CUDA available: {torch.cuda.is_available()}")
try:
    print(f"Default device: {torch.tensor([]).device}")
except Exception as e:
    print(f"Error getting default device: {e}")

try:
    x = torch.randn(10)
    print(f"randn device: {x.device}")
except Exception as e:
    print(f"Error creating randn: {e}")

print("Environment variables:")
for k, v in os.environ.items():
    if "CUDA" in k or "TORCH" in k:
        print(f"{k}={v}")
