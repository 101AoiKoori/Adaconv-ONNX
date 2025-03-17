import torch
import subprocess
import re


def get_cuda_version():
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        output = result.stdout
        match = re.search(r'release (\d+\.\d+)', output)
        if match:
            return match.group(1)
        else:
            print("无法从 nvcc 输出中提取 CUDA 版本信息。")
            return None
    except FileNotFoundError:
        print("未找到 nvcc，可能未安装 CUDA 或其未添加到系统路径中。")
        return None


def get_cudnn_version():
    try:
        import torch.backends.cudnn as cudnn
        return torch.backends.cudnn.version()
    except Exception as e:
        print(f"获取 cuDNN 版本时出错: {e}")
        return None


def get_pytorch_version():
    return torch.__version__


def check_compatibility(cuda_version, cudnn_version, pytorch_version):
    # 这里只是简单示例，实际兼容性需要参考官方文档
    if cuda_version is None or cudnn_version is None or pytorch_version is None:
        print("由于缺少版本信息，无法进行兼容性检查。")
        return
    print(f"CUDA 版本: {cuda_version}")
    print(f"cuDNN 版本: {cudnn_version}")
    print(f"PyTorch 版本: {pytorch_version}")
    # 这里可以根据官方兼容性表格添加更详细的检查逻辑
    if cuda_version.startswith('11.') and '1.10' in pytorch_version:
        print("CUDA 11.x 和 PyTorch 1.10 通常是兼容的。")
    else:
        print("请参考 PyTorch 官方文档确认版本兼容性。")


if __name__ == "__main__":
    cuda_version = get_cuda_version()
    cudnn_version = get_cudnn_version()
    pytorch_version = get_pytorch_version()
    check_compatibility(cuda_version, cudnn_version, pytorch_version)

    