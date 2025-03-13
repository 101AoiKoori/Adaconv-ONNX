import torch
import onnxruntime as ort
from datetime import datetime
import subprocess
import platform
import sys
import os

def get_system_info():
    """获取系统基本信息"""
    print("="*50)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"操作系统: {platform.platform()}")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"ONNXRuntime版本: {ort.__version__}")
    print("="*50 + "\n")

def check_nvidia_driver():
    """检查NVIDIA驱动"""
    print("[1/6] 检查NVIDIA显卡驱动...")
    try:
        result = subprocess.check_output(["nvidia-smi"], shell=True, stderr=subprocess.STDOUT)
        print("✅ 检测到NVIDIA驱动:")
        print(result.decode().strip())
        return True
    except Exception as e:
        print("❌ 未检测到NVIDIA驱动或nvidia-smi不可用")
        print(f"错误信息: {str(e)}")
        return False

def check_cuda_toolkit():
    """检查CUDA工具包"""
    print("\n[2/6] 检查CUDA工具包...")
    try:
        cuda_version = subprocess.check_output(["nvcc", "--version"]).decode()
        print("✅ 检测到CUDA工具包:")
        print(cuda_version.split("release")[-1].strip())
        return True
    except Exception as e:
        print("❌ 未检测到CUDA工具包或nvcc不可用")
        print(f"错误信息: {str(e)}")
        return False

def check_pytorch_cuda():
    """检查PyTorch CUDA支持"""
    print("\n[3/6] 检查PyTorch CUDA支持...")
    
    # 基础检查
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA可用: {'✅' if cuda_available else '❌'}")
    
    if cuda_available:
        # 设备信息
        device_count = torch.cuda.device_count()
        print(f"检测到 {device_count} 个CUDA设备")
        
        for i in range(device_count):
            print(f"\n设备 {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f} GB")
        
        # 版本信息
        print("\nCUDA版本:")
        print(f"  PyTorch内置CUDA版本: {torch.version.cuda}")
        if hasattr(torch.cuda, 'get_driver_version'):
            print(f"  当前驱动版本: {torch.cuda.get_driver_version()}")
        else:
            print("  当前驱动版本: (通过PyTorch无法获取，请直接运行`nvidia-smi`查看)")
                
        # 执行计算测试
        print("\n执行计算测试...")
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = (x @ y).mean()
            print(f"✅ 计算测试通过: 结果值={z.item():.4f}")
            return True
        except Exception as e:
            print(f"❌ 计算测试失败: {str(e)}")
            return False
    return False

def check_onnxruntime_cuda():
    """检查ONNX Runtime CUDA支持"""
    print("\n[4/6] 检查ONNX Runtime CUDA支持...")
    
    # 获取可用provider
    providers = ort.get_available_providers()
    print("可用计算后端:", providers)
    
    if 'CUDAExecutionProvider' not in providers:
        print("❌ 未检测到CUDAExecutionProvider")
        return False
    
    # 创建CUDA session
    print("\n创建CUDA会话...")
    try:
        options = ort.SessionOptions()
        session = ort.InferenceSession(
            "model.onnx" if os.path.exists("model.onnx") else "example.onnx",
            providers=['CUDAExecutionProvider'],
            sess_options=options
        )
        print("✅ CUDA会话创建成功")
        return True
    except Exception as e:
        print(f"❌ CUDA会话创建失败: {str(e)}")
        return False

def check_cudnn():
    """检查cuDNN安装"""
    print("\n[5/6] 检查cuDNN安装...")
    try:
        from torch.backends import cudnn
        print(f"PyTorch使用的cuDNN版本: {cudnn.version()}")
        
        # 执行cuDNN加速的卷积操作
        input = torch.randn(1,3,224,224).cuda()
        conv = torch.nn.Conv2d(3, 64, kernel_size=3).cuda()
        output = conv(input)
        print("✅ cuDNN计算测试通过")
        return True
    except Exception as e:
        print(f"❌ cuDNN检查失败: {str(e)}")
        return False

def check_environment_variables():
    """检查环境变量"""
    print("\n[6/6] 检查环境变量...")
    required_paths = [
        "CUDA_PATH",
        "PATH",
        "LD_LIBRARY_PATH" if platform.system() != "Windows" else ""
    ]
    
    for var in required_paths:
        if not var: continue
        value = os.environ.get(var, "")
        print(f"{var}:")
        print("  " + "\n  ".join(value.split(os.pathsep)) if value else print("  (未设置)"))

def main():
    get_system_info()
    results = {
        "nvidia_driver": check_nvidia_driver(),
        "cuda_toolkit": check_cuda_toolkit(),
        "pytorch_cuda": check_pytorch_cuda(),
        "onnxruntime_cuda": check_onnxruntime_cuda(),
        "cudnn": check_cudnn()
    }
    
    check_environment_variables()
    
    # 生成总结报告
    print("\n" + "="*50)
    print("CUDA环境验证总结:")
    for name, status in results.items():
        print(f"{name.replace('_', ' ').title():<20} : {'✅' if status else '❌'}")
    
    if all(results.values()):
        print("\n🎉 所有检查项通过！CUDA环境配置正确")
    else:
        print("\n⚠️ 存在未通过的检查项，请根据以上输出排查问题")

if __name__ == "__main__":
    main()