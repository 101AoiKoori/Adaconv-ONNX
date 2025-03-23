import sys
import os
import yaml
from utils.onnx_validator import ONNXValidator

def main():
    # 获取当前脚本所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_model_path = os.path.join(current_dir, "model.onnx")
    yaml_file_path = os.path.join(current_dir, "configs/lambda100.yaml")

    try:
        # 加载YAML配置文件
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"错误: 未找到 {yaml_file_path} 文件。")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误: 解析 YAML 文件时出错: {e}")
        sys.exit(1)

    # 从配置文件中获取图像尺寸和固定批量大小
    image_size = config.get('image_size', 256)
    fixed_batch_size = config.get('fixed_batch_size', 1)
    style_kernel = 3  # 固定通道数为 3

    input_shape_combinations = [
        {"content": (fixed_batch_size, style_kernel, image_size, image_size), "style": (fixed_batch_size, style_kernel, image_size, image_size)},
    ]

    for input_shapes in input_shape_combinations:
        # 格式化输入形状为符合 run_inference_test 要求的格式
        dummy_input_shapes = []
        for name, shape in input_shapes.items():
            dummy_input_shapes.append(f"{name}:{','.join(map(str, shape))}")

        # 打印输入形状进行检查
        print("输入形状:", dummy_input_shapes)

        # 创建验证器实例
        validator = ONNXValidator(onnx_model_path)

        # 执行完整验证
        validator.full_validation(dummy_input_shapes)

if __name__ == "__main__":
    sys.exit(main())