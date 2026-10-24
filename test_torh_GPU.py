import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch 编译的 CUDA 版本: {torch.version.cuda}") # 如果为 None，说明装的是 CPU 版

# 核心判断：GPU 是否可用
gpu_available = torch.cuda.is_available()
print(f"\nCUDA (GPU) 是否可用: {gpu_available}")

if gpu_available:
    print(f"GPU 设备数量: {torch.cuda.device_count()}")
    print(f"当前 GPU 索引: {torch.cuda.current_device()}")
    print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")
else:
    print("原因可能是：未安装 GPU 版 PyTorch、显卡不支持、或驱动/CUDA 环境有问题。")