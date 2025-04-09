"""
AdaConv 简化版ONNX导出脚本 - 兼容新版StyleTransfer

使用方法:
    python exporter.py --output model.onnx  # 使用默认设置
    python exporter.py --output model.onnx --static  # 静态模式
    python exporter.py --output model.onnx --dynamic  # 完全动态模式
    python exporter.py --output model.onnx --dynamic-batch  # 动态批次大小
"""

import argparse
import os
import yaml
import torch
from pathlib import Path
from models.model import StyleTransfer
from models.encoder import Encoder
from hyperparam.hyperparam import Hyperparameter
import warnings
warnings.filterwarnings(
        "ignore",
        message="Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.",
        category=UserWarning,
        module="torch.onnx"
    )


def get_default_paths():
    """获取默认路径和参数"""
    # 默认路径 - 根据实际情况修改
    ckpt_dir = Path("./logs/ckpts")
    config_dir = Path("./configs")
    
    # 检查目录是否存在，不存在则创建
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取最新检查点
    checkpoint_path = ckpt_dir / "last.pt"
    if not checkpoint_path.exists():
        # 查找其他检查点
        ckpts = list(ckpt_dir.glob("*.pt"))
        if ckpts:
            checkpoint_path = sorted(ckpts)[-1]  # 使用最新的检查点
        else:
            raise FileNotFoundError(f"找不到模型检查点，请确保 {ckpt_dir} 目录中存在检查点文件")
    
    # 获取配置文件
    config_path = config_dir / "lambda100.yaml"
    if not config_path.exists():
        # 查找其他配置文件
        configs = list(config_dir.glob("*.yaml"))
        if configs:
            config_path = configs[0]  # 使用第一个找到的配置
        else:
            raise FileNotFoundError(f"找不到配置文件，请确保 {config_dir} 目录中存在YAML配置文件")
    
    return str(checkpoint_path), str(config_path)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return Hyperparameter(**config)


def initialize_model(hyper_param, device, export_config=None):
    """初始化模型，兼容新的StyleTransfer构造函数"""
    # 处理groups参数
    if hyper_param.groups_list:
        groups = hyper_param.groups_list
    elif hyper_param.groups:
        groups = hyper_param.groups
    else:
        # 根据比例计算分组
        base_channels = [512, 256, 128, 64]
        groups = [max(1, int(c * r)) for c, r in zip(base_channels, hyper_param.groups_ratios)]
        print(f"自动计算的分组: {groups}")
    
    # 创建模型
    return StyleTransfer(
        image_shape=hyper_param.image_shape,
        style_dim=hyper_param.style_dim,
        style_kernel=hyper_param.style_kernel,
        groups=groups,
        export_config=export_config
    ).to(device)


def load_checkpoint(model, checkpoint_path, device):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=True)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ 检查点加载成功: {checkpoint_path}")
    return model


def export_onnx(model, dummy_input, output_path, dynamic_axes=None, opset=16):
    """导出ONNX模型"""
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=output_path,
        input_names=["content", "style"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL
    )
    print(f"✅ ONNX导出成功: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AdaConv ONNX导出工具 - 自定义版本")
    
    # 基本参数
    parser.add_argument("--output", required=True, help="输出ONNX模型路径 (例如: models/adaconv.onnx)")
    parser.add_argument("--checkpoint", help="模型检查点路径 (.pt文件)")
    parser.add_argument("--config", help="配置文件路径 (.yaml文件)")
    parser.add_argument("--opset", type=int, default=16, help="ONNX操作集版本 (默认: 16)")
    parser.add_argument("--batch-size", type=int, default=1, help="导出批次大小 (默认: 1)")
    
    # 导出模式选项 (互斥)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--static", action="store_true", help="使用完全静态模式导出 (固定批次大小和尺寸)")
    mode_group.add_argument("--dynamic", action="store_true", help="使用完全动态模式导出 (动态批次大小和尺寸)")
    mode_group.add_argument("--dynamic-batch", action="store_true", help="使用动态批次大小模式导出 (固定尺寸)")
    mode_group.add_argument("--dynamic-shape", action="store_true", help="使用动态尺寸模式导出 (固定批次大小)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取默认路径
    try:
        default_checkpoint, default_config = get_default_paths()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 使用命令行参数或默认值
    checkpoint_path = args.checkpoint or default_checkpoint
    config_path = args.config or default_config
    
    # 基于参数设置导出模式
    is_static = args.static
    dynamic_batch = args.dynamic or args.dynamic_batch
    dynamic_shape = args.dynamic or args.dynamic_shape
    
    # 打印导出信息
    print("\n" + "="*60)
    print("AdaConv ONNX导出工具 - 自定义版本")
    print("="*60)
    print(f"检查点文件: {checkpoint_path}")
    print(f"配置文件: {config_path}")
    print(f"输出文件: {args.output}")
    print(f"批次大小: {args.batch_size}")
    
    # 打印导出模式
    if is_static:
        print("导出模式: 完全静态 (固定批次大小和尺寸)")
    elif dynamic_batch and dynamic_shape:
        print("导出模式: 完全动态 (动态批次大小和尺寸)")
    elif dynamic_batch:
        print("导出模式: 动态批次大小 (固定尺寸)")
    elif dynamic_shape:
        print("导出模式: 动态尺寸 (固定批次大小)")
    else:
        print("导出模式: 默认 (只有批次大小可动态)")
        dynamic_batch = True  # 默认启用动态批次大小
    
    print("="*60 + "\n")
    
    try:
        # 加载配置
        hyper_param = load_config(config_path)
        
        # 设置设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device.upper()}")
        
        # 准备导出配置
        export_config = {
            'export_mode': True,
            'fixed_batch_size': args.batch_size if is_static or not dynamic_batch else None,
            'use_fixed_size': is_static or not dynamic_shape
        }
        
        # 初始化模型
        model = initialize_model(hyper_param, device, export_config)
        
        # 加载权重
        model = load_checkpoint(model, checkpoint_path, device)
        
        # 设置为评估模式
        model.eval()
        
        # 生成模拟输入
        input_batch_size = args.batch_size
        dummy_content = torch.randn(input_batch_size, 3, *hyper_param.image_shape, device=device)
        dummy_style = torch.randn(input_batch_size, 3, *hyper_param.image_shape, device=device)
        
        # 动态轴配置
        dynamic_axes = None
        if not is_static:
            dynamic_axes = {}
            if dynamic_batch:
                # 批次维度动态
                dynamic_axes.update({
                    'content': {0: 'batch_size'},
                    'style': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                })
            if dynamic_shape:
                # 空间维度动态
                dynamic_axes.update({
                    'content': {2: 'height', 3: 'width'},
                    'style': {2: 'height', 3: 'width'},
                    'output': {2: 'height', 3: 'width'}
                })
        
        # 执行导出
        export_onnx(
            model=model,
            dummy_input=(dummy_content, dummy_style),
            output_path=args.output,
            dynamic_axes=dynamic_axes,
            opset=args.opset
        )
        
        print(f"\n✅ 模型成功导出到: {args.output}")
        print(f"   - 批次大小: {'动态' if dynamic_batch else args.batch_size}")
        print(f"   - 尺寸模式: {'动态' if dynamic_shape else '固定'}")
        
    except Exception as e:
        import traceback
        print(f"\n❌ 导出失败: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()