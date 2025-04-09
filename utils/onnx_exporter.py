import torch
import argparse
import onnx
import yaml
import os
from pathlib import Path
from typing import Tuple, Dict, Optional, Union, List, Any

from models.model import StyleTransfer
from onnx import shape_inference
from hyperparam.hyperparam import Hyperparameter

# 忽略ONNX警告
import warnings
warnings.filterwarnings(
    "ignore",
    message="Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.",
    category=UserWarning,
    module="torch.onnx"
)


def load_config(config_path: str) -> Hyperparameter:
    """
    从YAML文件加载配置并转换为Hyperparameter对象
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        解析后的Hyperparameter对象
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 转换为Hyperparameter对象
    return Hyperparameter(**config)


def export_onnx(
    model: torch.nn.Module,
    dummy_input: Tuple[torch.Tensor, torch.Tensor],
    output_path: str,
    dynamic_axes: Optional[Dict] = None,
    opset: int = 16
) -> bool:
    """
    将模型导出为ONNX格式
    
    Args:
        model: PyTorch模型
        dummy_input: 模拟输入(内容图像,风格图像)
        output_path: 输出ONNX文件路径
        dynamic_axes: 动态轴配置(可选)
        opset: ONNX操作集版本
        
    Returns:
        导出是否成功
    """
    try:
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 执行ONNX导出
        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=output_path,
            input_names=["content", "style"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False
        )
        
        # 形状推理
        model_onnx = onnx.load(output_path)
        inferred_model = shape_inference.infer_shapes(model_onnx)
        onnx.save(inferred_model, output_path)
        
        print(f"✅ ONNX导出成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 导出失败: {str(e)}")
        return False


def get_groups_config(hyper_param: Hyperparameter) -> List[int]:
    """
    根据配置获取分组卷积参数
    
    Args:
        hyper_param: 超参数配置
        
    Returns:
        分组卷积参数列表
    """
    # 计算分组参数(如果未明确设置)
    if hyper_param.groups_list:
        return hyper_param.groups_list
    elif hyper_param.groups:
        # 如果groups是单个整数,重复4次
        if isinstance(hyper_param.groups, int):
            return [hyper_param.groups] * 4
        return hyper_param.groups
    else:
        # 根据比例计算
        base_channels = [512, 256, 128, 64]
        groups_list = [
            max(1, int(c * r)) 
            for c, r in zip(base_channels, hyper_param.groups_ratios)
        ]
        print(f"自动计算的分组参数: {groups_list}")
        return groups_list


def initialize_model(hyper_param: Hyperparameter, device: str) -> torch.nn.Module:
    """
    初始化具有正确参数的模型
    
    Args:
        hyper_param: 超参数配置
        device: 设备('cuda'或'cpu')
        
    Returns:
        初始化的模型
    """
    # 获取分组配置
    groups_list = get_groups_config(hyper_param)
    
    # 创建导出配置
    export_config = {
        'export_mode': True,
        'fixed_batch_size': hyper_param.fixed_batch_size,
        'use_fixed_size': hyper_param.use_fixed_size
    }

    # 初始化模型
    model = StyleTransfer(
        image_shape=hyper_param.image_shape,
        style_dim=hyper_param.style_dim,
        style_kernel=hyper_param.style_kernel,
        groups=groups_list,
        export_config=export_config
    ).to(device)
    
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: str):
    """
    从检查点加载权重(包含兼容性处理)
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        device: 设备('cuda'或'cpu')
        
    Returns:
        加载权重后的模型
    """
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 提取状态字典(兼容不同格式)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 加载状态字典(使用strict=False允许缺失参数)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        # 打印调试信息
        if missing:
            print(f"❓ 缺失参数: {len(missing)} 个")
        if unexpected:
            print(f"⚠️ 意外参数: {len(unexpected)} 个")
                
        print(f"✅ 检查点加载成功: {checkpoint_path}")
        return model
        
    except Exception as e:
        print(f"❌ 检查点加载失败: {str(e)}")
        raise


def main(
    checkpoint_path: str,
    config_path: str,
    output_path: str,
    opset: int = 16
):
    """
    主导出函数
    
    Args:
        checkpoint_path: 检查点路径
        config_path: 配置文件路径
        output_path: 输出ONNX文件路径
        opset: ONNX操作集版本
    """
    # 加载配置
    print(f"🔧 正在加载配置: {config_path}")
    hyper_param = load_config(config_path)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 使用设备: {device.upper()}")
    
    # 初始化模型
    print(f"🔧 正在初始化模型...")
    model = initialize_model(hyper_param, device)
    
    # 加载权重
    print(f"🔧 正在加载检查点: {checkpoint_path}")
    model = load_checkpoint(model, checkpoint_path, device)
    
    # 设置为评估模式
    model.eval()
    
    # 生成模拟输入
    print(f"🔧 正在准备模拟输入...")
    batch_size = hyper_param.fixed_batch_size if hyper_param.fixed_batch_size else 1
    dummy_content = torch.randn(batch_size, 3, *hyper_param.image_shape, device=device)
    dummy_style = torch.randn(batch_size, 3, *hyper_param.image_shape, device=device)
    
    # 动态轴配置
    dynamic_axes = None
    if not hyper_param.use_fixed_size:
        dynamic_axes = {
            'content': {2: 'height', 3: 'width'},
            'style': {2: 'height', 3: 'width'},
            'output': {2: 'height', 3: 'width'}
        }
        
        # 如果fixed_batch_size未设置，则批次大小也是动态的
        if not hyper_param.fixed_batch_size:
            dynamic_axes['content'][0] = 'batch_size'
            dynamic_axes['style'][0] = 'batch_size'
            dynamic_axes['output'][0] = 'batch_size'
    
    # 执行导出
    print(f"🔧 正在导出模型...")
    success = export_onnx(
        model=model,
        dummy_input=(dummy_content, dummy_style),
        output_path=output_path,
        dynamic_axes=dynamic_axes,
        opset=opset
    )
    
    if success:
        # 获取导出文件大小
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        print(f"📊 导出文件大小: {file_size:.2f} MB")
        
        # 打印导出模式总结
        mode_str = "静态模式" if hyper_param.use_fixed_size else "动态模式"
        batch_str = f"固定批次大小: {batch_size}" if hyper_param.fixed_batch_size else "动态批次大小"
        shape_str = f"固定空间尺寸: {hyper_param.image_shape}" if hyper_param.use_fixed_size else "动态空间尺寸"
        
        print(f"✅ 导出完成: {output_path}")
        print(f"   - 模式: {mode_str}")
        print(f"   - {batch_str}")
        print(f"   - {shape_str}")
        print(f"   - ONNX操作集: {opset}")
    else:
        print(f"❌ 导出失败")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaConv ONNX导出工具")
    parser.add_argument("--checkpoint", required=True, help="PyTorch检查点路径(.pt)")
    parser.add_argument("--config", default="configs/lambda100.yaml", help="配置文件路径")
    parser.add_argument("--output", required=True, help="输出ONNX文件路径")
    parser.add_argument("--opset", type=int, default=16, help="ONNX操作集版本(默认: 16)")
    
    args = parser.parse_args()
    
    try:
        main(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            output_path=args.output,
            opset=args.opset
        )
    except Exception as e:
        import traceback
        print(f"❌ 导出过程终止，出现异常:")
        traceback.print_exc()
        exit(1)