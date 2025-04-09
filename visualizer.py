import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
import argparse
import math

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, img_size=256):
    """加载并预处理图片"""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def tensor_to_numpy(tensor):
    """将tensor转换为numpy图像"""
    img = tensor.clone().detach().cpu().numpy()
    img = img.squeeze(0).transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img

def process_style_transfer(model, content_img, style_img):
    """处理风格迁移"""
    with torch.no_grad():
        output = model(content_img, style_img)
    return output

def create_visualization(model_path, test_dir="./Test", output_path="style_transfer_visualization.png"):
    print(f"Loading model from {model_path}...")
    
    # 加载模型
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint

        # 尝试导入模型定义
        try:
            from models.model import StyleTransfer
            
            # 创建模型实例（使用默认参数）
            model = StyleTransfer(
                image_shape=(256, 256),
                style_dim=512,
                style_kernel=3,
                export_config={'export_mode': False}
            ).to(device)
            
            # 加载权重
            model.load_state_dict(model_state_dict)
            model.eval()
            print("Model loaded successfully!")
        except ImportError:
            print("Error: Could not import model definition. Make sure the models package is in your PYTHONPATH.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 检查测试目录
    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist!")
        return
    
    # 寻找内容和风格图像
    content_dir = test_dir / "content"
    style_dir = test_dir / "style"
    
    if not content_dir.exists() or not style_dir.exists():
        print(f"Error: Content or style directory not found in {test_dir}")
        return
    
    # 获取所有图像
    content_images = sorted([f for f in content_dir.glob("*.jpg") or content_dir.glob("*.png")])
    style_images = sorted([f for f in style_dir.glob("*.jpg") or style_dir.glob("*.png")])
    
    if not content_images:
        print("No content images found!")
        return
    
    if not style_images:
        print("No style images found!")
        return
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    
    # 确定网格大小
    num_content = len(content_images)
    num_style = len(style_images)
    
    # 加载所有图像
    content_tensors = []
    style_tensors = []
    content_names = []
    style_names = []
    
    print("Loading images...")
    for content_path in content_images:
        try:
            content_tensors.append(load_image(content_path))
            content_names.append(content_path.stem)
        except Exception as e:
            print(f"Error loading content image {content_path}: {e}")
    
    for style_path in style_images:
        try:
            style_tensors.append(load_image(style_path))
            style_names.append(style_path.stem)
        except Exception as e:
            print(f"Error loading style image {style_path}: {e}")
    
    # 创建结果图
    print("Generating style transfers...")
    results = []
    
    for content_tensor in content_tensors:
        content_results = []
        for style_tensor in style_tensors:
            try:
                output = process_style_transfer(model, content_tensor, style_tensor)
                content_results.append(tensor_to_numpy(output))
            except Exception as e:
                print(f"Error in style transfer: {e}")
                # 在错误情况下添加黑色图像
                content_results.append(np.zeros((256, 256, 3)))
        results.append(content_results)
    
    # 创建可视化网格
    print("Creating visualization grid...")
    
    # 确定最佳网格大小，使输出图像接近正方形
    total_images = (num_content + 1) * (num_style + 1)
    grid_size = math.ceil(math.sqrt(total_images))
    
    # 创建大小适当的图像
    plt.figure(figsize=(12, 12), dpi=300)
    
    # 添加标题
    plt.suptitle("AdaConv Style Transfer Visualization", fontsize=16, y=0.99)
    
    # 创建网格
    rows = num_content + 1
    cols = num_style + 1
    
    # 创建子图
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, i * cols + j + 1)
            
            # 第一行显示风格图像
            if i == 0 and j > 0:
                plt.imshow(tensor_to_numpy(style_tensors[j-1]))
                plt.title(f"Style: {style_names[j-1]}", fontsize=8)
            # 第一列显示内容图像
            elif j == 0 and i > 0:
                plt.imshow(tensor_to_numpy(content_tensors[i-1]))
                plt.title(f"Content: {content_names[i-1]}", fontsize=8)
            # 第一个单元格留空或显示模型信息
            elif i == 0 and j == 0:
                plt.text(0.5, 0.5, "AdaConv\nModel", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=plt.gca().transAxes,
                        fontsize=10)
            # 其他单元格显示结果
            else:
                plt.imshow(results[i-1][j-1])
            
            plt.axis('off')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # 保存结果
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved successfully to {output_path}")
    
    # 显示图像
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description="AdaConv Style Transfer Visualizer")
    parser.add_argument("--model", type=str, default="./logs/ckpts/last.pt", help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, default="./Test", help="Directory containing test images")
    parser.add_argument("--output", type=str, default="style_transfer_visualization.png", help="Output image path")
    
    args = parser.parse_args()
    
    create_visualization(args.model, args.test_dir, args.output)
    
if __name__ == "__main__":
    main()