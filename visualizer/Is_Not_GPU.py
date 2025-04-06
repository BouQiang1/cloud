import torch


if __name__ == '__main__':
    # 检查 PyTorch 是否能够访问 GPU
    gpu_available = torch.cuda.is_available()
    gpu_version = torch.version.cuda if gpu_available else "No CUDA"

    print(f"Is CUDA available: {gpu_available}")
    print(f"CUDA version: {gpu_version}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    y = torch.tensor([4.0, 5.0, 6.0]).to(device)
    z = x + y
    print(z)
