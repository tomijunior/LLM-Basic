import torch

print(f"Versi PyTorch: {torch.__version__}")
print(f"CUDA Tersedia (GPU): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Nama GPU: {torch.cuda.get_device_name(0)}")