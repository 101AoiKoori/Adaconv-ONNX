import os
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from model import StyleTransfer

# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

def load_image(img_path, image_size=None):
    """Load and preprocess an image."""
    img = Image.open(img_path).convert('RGB')
    
    if image_size:
        transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    return transform(img).unsqueeze(0), img.size

def tensor_to_image(tensor):
    """Convert a tensor to a numpy image."""
    img = tensor.clone().detach().cpu().squeeze(0).numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img

"""Create a grid visualization of style transfer results."""
def create_style_transfer_grid(model, content_images, style_images, device, title=None, output_path=None, image_size=None):
    """Create a grid visualization of style transfer results."""
    num_styles = len(style_images)
    num_contents = len(content_images)
    
    # Create a figure with custom grid
    fig = plt.figure(figsize=(3 * (num_contents + 1), 3 * (num_styles + 1)))
    gs = gridspec.GridSpec(num_styles + 1, num_contents + 1, figure=fig)
    
    # Load all images and transfer styles
    all_results = []
    
    # Add content images along the top row
    for i, content_path in enumerate(content_images):
        content, _ = load_image(content_path, image_size)
        content = content.to(device)  # 只对图像张量调用 to(device) 方法
        content_img = tensor_to_image(content)
        ax = plt.subplot(gs[0, i+1])
        ax.imshow(content_img)
        ax.set_title(f"Content {i+1}")
        ax.axis('off')
    
    # Add style images in the first column
    for j, style_path in enumerate(style_images):
        style, _ = load_image(style_path, image_size)
        style = style.to(device)  # 只对图像张量调用 to(device) 方法
        style_img = tensor_to_image(style)
        ax = plt.subplot(gs[j+1, 0])
        ax.imshow(style_img)
        ax.set_title(f"Style {j+1}")
        ax.axis('off')
    
    # Process style transfer for each combination
    with torch.no_grad():
        for j, style_path in enumerate(style_images):
            style, _ = load_image(style_path, image_size)
            style = style.to(device)  # 只对图像张量调用 to(device) 方法
            
            for i, content_path in enumerate(content_images):
                content, _ = load_image(content_path, image_size)
                content = content.to(device)  # 只对图像张量调用 to(device) 方法
                
                # Generate the stylized image
                output = model(content, style)
                output_img = tensor_to_image(output)
                
                # Add to the grid
                ax = plt.subplot(gs[j+1, i+1])
                ax.imshow(output_img)
                ax.axis('off')
    
    # Set overall title if provided
    if title:
        plt.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"可视化结果已保存至 {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='生成风格迁移可视化网格')
    parser.add_argument('--model_path', type=str, required=True, help='训练好的模型检查点路径')
    parser.add_argument('--content_dir', type=str, required=True, help='内容图片目录')
    parser.add_argument('--style_dir', type=str, required=True, help='风格图片目录')
    parser.add_argument('--output_path', type=str, default='style_transfer_grid.png', help='输出文件路径')
    parser.add_argument('--style_dim', type=int, default=512, help='风格维度')
    parser.add_argument('--style_kernel', type=int, default=3, help='风格核大小')
    parser.add_argument('--use_auto_size', action='store_true', help='自动检测和使用图片尺寸')
    parser.add_argument('--image_size', type=str, default=None, help='可选：图片大小，格式为WxH')

    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 获取内容和风格图片
    content_images = sorted([os.path.join(args.content_dir, f) for f in os.listdir(args.content_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    style_images = sorted([os.path.join(args.style_dir, f) for f in os.listdir(args.style_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not content_images or not style_images:
        print("错误：未找到图片文件。")
        return

    print(f"找到 {len(content_images)} 张内容图片和 {len(style_images)} 张风格图片。")

    # 自动检测图片大小或使用用户指定的大小
    image_size = None
    if args.image_size:
        width, height = map(int, args.image_size.split('x'))
        image_size = (height, width)
        print(f"使用指定的图片大小: {width}x{height}")
    elif args.use_auto_size or args.image_size is None:
        # 获取第一张内容图片的大小作为参考
        first_img = Image.open(content_images[0]).convert('RGB')
        width, height = first_img.size
        image_size = (height, width)
        print(f"自动检测到图片大小: {width}x{height}")

    # 初始化模型
    model = StyleTransfer(
        image_shape=image_size,
        style_dim=args.style_dim,
        style_kernel=args.style_kernel,
        use_fixed_size=True  # 为可视化启用固定大小处理
    )

    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    new_state_dict = model.state_dict()

    # 只加载形状匹配的参数
    for name, param in model_state_dict.items():
        if name in new_state_dict and new_state_dict[name].shape == param.shape:
            new_state_dict[name] = param

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 创建网格可视化
    create_style_transfer_grid(
        model, 
        content_images, 
        style_images, 
        device,
        title="AdaConv 风格迁移结果",
        output_path=args.output_path,
        image_size=image_size
    )

if __name__ == "__main__":
    main()