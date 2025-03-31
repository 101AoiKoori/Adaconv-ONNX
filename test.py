import os
import torch
from models.model import StyleTransfer
from PIL import Image
import matplotlib.pyplot as plt
from utils.checkepoint import create_style_transfer_grid, load_image, tensor_to_image

def demo():
    """
    Simple demonstration of the visualization script without command line arguments
    """
    # Configuration
    model_path = "./logs/ckpts/last.pt"  # Replace with your model path
    content_dir = "./Test/content"  # Replace with your content images directory
    style_dir = "./Test/style"      # Replace with your style images directory
    output_path = "./adaconv_style.png"
    style_dim = 512
    style_kernel = 3

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 自动检测图片大小
    content_images = sorted([os.path.join(content_dir, f) for f in os.listdir(content_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if content_images:
        first_img = Image.open(content_images[0]).convert('RGB')
        width, height = first_img.size
        image_size = (height, width)
        print(f"Automatically detected image size: {width}x{height}")
    else:
        print("No content images found.")
        return

    # Initialize model
    model = StyleTransfer(
        image_shape=image_size,
        style_dim=style_dim,
        style_kernel=style_kernel,
        use_fixed_size=True
    )

    # Load model weights (if available)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model from {model_path}, using initialized weights. Error: {e}")

    model.to(device)
    model.eval()

    # Get content and style images
    content_images = sorted([os.path.join(content_dir, f) for f in os.listdir(content_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    style_images = sorted([os.path.join(style_dir, f) for f in os.listdir(style_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Found {len(content_images)} content images and {len(style_images)} style images.")

    # Create the grid visualization
    create_style_transfer_grid(
        model, 
        content_images, 
        style_images, 
        device,
        output_path=output_path,
        image_size=image_size
    )

    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    demo()