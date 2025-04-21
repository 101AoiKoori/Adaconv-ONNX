import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
import argparse
import math

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, img_size=256):
    """Load and preprocess an image"""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor

def tensor_to_numpy(tensor):
    """Convert tensor to numpy image"""
    img = tensor.clone().detach().cpu().numpy()
    img = img.squeeze(0).transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img

def process_style_transfer(model, content_img, style_img):
    """Perform style transfer"""
    with torch.no_grad():
        output = model(content_img, style_img)
    return output

def create_visualization(model_path, test_dir="./Test", output_path="style_transfer_visualization.png"):
    print(f"Loading model from {model_path}...")
    
    # Load model
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
        else:
            model_state_dict = checkpoint

        # Attempt to import model definition
        try:
            from models.model import StyleTransfer
            
            # Create model instance (with default parameters)
            model = StyleTransfer(
                image_shape=(256, 256),
                style_dim=512,
                style_kernel=3,
                export_config={'export_mode': False}
            ).to(device)
            
            # Load weights
            model.load_state_dict(model_state_dict)
            model.eval()
            print("Model loaded successfully!")
        except ImportError:
            print("Error: Could not import model definition. Ensure the models package is in your PYTHONPATH.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Check test directory
    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist!")
        return
    
    # Find content and style images
    content_dir = test_dir / "content"
    style_dir = test_dir / "style"
    
    if not content_dir.exists() or not style_dir.exists():
        print(f"Error: Content or style directory not found in {test_dir}")
        return
    
    # Get all images
    content_images = sorted([f for f in content_dir.glob("*.jpg") or content_dir.glob("*.png")])
    style_images = sorted([f for f in style_dir.glob("*.jpg") or style_dir.glob("*.png")])
    
    if not content_images:
        print("No content images found!")
        return
    
    if not style_images:
        print("No style images found!")
        return
    
    print(f"Found {len(content_images)} content images and {len(style_images)} style images")
    
    # Determine grid size
    num_content = len(content_images)
    num_style = len(style_images)
    
    # Load all images
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
    
    # Create result grid
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
                # Add black image on error
                content_results.append(np.zeros((256, 256, 3)))
        results.append(content_results)
    
    # Create visualization grid
    print("Creating visualization grid...")
    
    # Determine optimal grid size for square-like output
    total_images = (num_content + 1) * (num_style + 1)
    grid_size = math.ceil(math.sqrt(total_images))
    
    # Create appropriately sized figure
    plt.figure(figsize=(12, 12), dpi=300)
    
    # Add title
    plt.suptitle("AdaConv Style Transfer Visualization", fontsize=16, y=0.99)
    
    # Create grid
    rows = num_content + 1
    cols = num_style + 1
    
    # Create subplots
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, i * cols + j + 1)
            
            # First row shows style images
            if i == 0 and j > 0:
                plt.imshow(tensor_to_numpy(style_tensors[j-1]))
                plt.title(f"Style: {style_names[j-1]}", fontsize=8)
            # First column shows content images
            elif j == 0 and i > 0:
                plt.imshow(tensor_to_numpy(content_tensors[i-1]))
                plt.title(f"Content: {content_names[i-1]}", fontsize=8)
            # First cell is empty or shows model info
            elif i == 0 and j == 0:
                plt.text(0.5, 0.5, "AdaConv\nModel", 
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=plt.gca().transAxes,
                        fontsize=10)
            # Other cells show results
            else:
                plt.imshow(results[i-1][j-1])
            
            plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save result
    print(f"Saving visualization to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Visualization saved successfully to {output_path}")
    
    # Display image
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