import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from matplotlib.figure import Figure
from PIL import Image
import io
import datetime
from pathlib import Path
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

class VisualizationManager:
    """
    Manages visualizations for style transfer training
    
    This class is responsible for:
    - Creating TensorBoard visualizations
    - Generating model structure visualizations
    - Creating feature map visualizations
    - Converting matplotlib figures to images
    """
    
    def __init__(self, writer: SummaryWriter, batch_size: int, nrow: int = None):
        """
        Initialize the VisualizationManager
        
        Args:
            writer: TensorBoard SummaryWriter instance
            batch_size: Batch size for grid visualizations
            nrow: Number of images per row in grids, defaults to half the batch size
        """
        self.writer = writer
        self.batch_size = batch_size
        self.nrow = nrow if nrow is not None else batch_size // 2
        
    def write_metrics(self, metrics: dict, step: int, prefix: str = ""):
        """
        Write numerical metrics to TensorBoard
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Current training step
            prefix: Prefix for metric names
        """
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}_{name}" if prefix else name, value, step)
    
    def write_images(self, images: dict, step: int, prefix: str = ""):
        """
        Write image grids to TensorBoard
        
        Args:
            images: Dictionary of image name -> tensor
            step: Current training step
            prefix: Prefix for image names
        """
        for name, img_tensor in images.items():
            self.writer.add_image(
                f"{prefix}_{name}" if prefix else name,
                make_grid(img_tensor, nrow=self.nrow),
                step
            )
    
    def write_hyperparameters(self, hyperparams: dict, step: int = 0):
        """
        Write hyperparameters as text to TensorBoard
        
        Args:
            hyperparams: Dictionary of hyperparameter settings
            step: Current training step
        """
        # Convert all values to strings to ensure JSON compatibility
        hyperparams_str = {k: str(v) for k, v in hyperparams.items()}
        self.writer.add_text('Hyperparameters', json.dumps(hyperparams_str, indent=4), step)
    
    def visualize_model_structure(self, model_info: dict):
        """
        Create a visualization of the model structure
        
        Args:
            model_info: Dictionary of model structure information
        
        Returns:
            Figure: Matplotlib figure with model structure visualization
        """
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=[[k, v] for k, v in model_info.items()],
            colLabels=["Parameter", "Value"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        fig.tight_layout()
        return fig
    
    def visualize_parameters_distribution(self, model_parameters):
        """
        Create a visualization of model parameter distributions
        
        Args:
            model_parameters: Dictionary of named parameters
        
        Returns:
            Figure: Matplotlib figure with parameter distributions
        """
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Collect parameter histogram data
        param_histograms = {}
        for name, param in model_parameters:
            if param.requires_grad:
                # Truncate name, keep last few components
                short_name = '/'.join(name.split('.')[-2:])
                param_histograms[short_name] = param.data.cpu().flatten().numpy()
        
        # Plot histograms for the first 10 main parameters
        for i, (name, values) in enumerate(list(param_histograms.items())[:10]):
            ax.hist(values, alpha=0.5, bins=50, label=name)
        
        ax.set_title("Parameter Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        return fig
    
    def visualize_gradients(self, model_parameters):
        """
        Create a visualization of gradient norms by layer
        
        Args:
            model_parameters: Model parameters
        
        Returns:
            Figure: Matplotlib figure with gradient visualizations
        """
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Collect gradient data
        grad_norms = []
        layer_names = []
        
        for name, param in model_parameters:
            if param.requires_grad and param.grad is not None:
                # Calculate norm of gradients for each layer
                norm = param.grad.norm().item()
                grad_norms.append(norm)
                # Truncate name, keep last few components
                short_name = '/'.join(name.split('.')[-2:])
                layer_names.append(short_name)
        
        # Only show gradients for top 15 layers
        if len(grad_norms) > 15:
            indices = np.argsort(grad_norms)[-15:]
            grad_norms = [grad_norms[i] for i in indices]
            layer_names = [layer_names[i] for i in indices]
        
        # Draw bar chart
        y_pos = range(len(layer_names))
        ax.barh(y_pos, grad_norms, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(layer_names)
        ax.invert_yaxis()  # Labels start from top
        ax.set_xlabel('Gradient Norm')
        ax.set_title('Gradient Norms by Layer')
        
        fig.tight_layout()
        return fig
    
    def create_feature_maps_visualization(self, feature_maps):
        """
        Create visualizations of feature maps
        
        Args:
            feature_maps: List of feature map tensors
        
        Returns:
            Figure or None: Matplotlib figure with feature map visualizations
        """
        if not feature_maps:
            return None
            
        # Select the last feature map for visualization
        feature_map = feature_maps[-1]
        
        # Take only first batch and limited channels
        batch_idx = 0
        num_channels = min(16, feature_map.shape[1])
        
        # Create figure
        fig = Figure(figsize=(10, 10))
        
        for i in range(num_channels):
            ax = fig.add_subplot(4, 4, i+1)
            # Extract single channel and convert to CPU
            channel_data = feature_map[batch_idx, i].detach().cpu().numpy()
            
            # Draw feature map
            im = ax.imshow(channel_data, cmap='viridis')
            ax.set_title(f'Channel {i}')
            ax.axis('off')
        
        fig.tight_layout()
        return fig
    
    def plot_to_image(self, figure):
        """
        Convert a Matplotlib figure to an image tensor for TensorBoard
        
        Args:
            figure: Matplotlib figure
        
        Returns:
            torch.Tensor: Image tensor
        """
        # Save the figure to a memory buffer
        buf = io.BytesIO()
        figure.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Convert PIL image to tensor
        image = Image.open(buf)
        image = np.array(image)
        
        # Convert to RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        
        # Convert to CHW format for PyTorch
        image = image.transpose((2, 0, 1))
        
        return torch.from_numpy(image)
    
    def add_figure_to_tensorboard(self, tag, figure, step):
        """
        Add a matplotlib figure to tensorboard
        
        Args:
            tag: Tag for the image
            figure: Matplotlib figure
            step: Current training step
        """
        image = self.plot_to_image(figure)
        self.writer.add_image(tag, image, step)
        
    def log_system_metrics(self, step, device="cuda"):
        """
        Log system metrics like GPU memory usage
        
        Args:
            step: Current training step
            device: Computing device
        """
        if device == "cuda" and torch.cuda.is_available():
            self.writer.add_scalar('System/GPU_Memory_Allocated_GB', 
                                  torch.cuda.memory_allocated() / (1024**3), 
                                  step)
            if hasattr(torch.cuda, 'max_memory_allocated'):
                self.writer.add_scalar('System/GPU_Memory_Peak_GB', 
                                      torch.cuda.max_memory_allocated() / (1024**3),
                                      step)
                                      
    def log_performance_metrics(self, batch_time, examples_per_sec, progress, step):
        """
        Log performance metrics
        
        Args:
            batch_time: Time taken for current batch
            examples_per_sec: Examples processed per second
            progress: Training progress as percentage
            step: Current training step
        """
        self.writer.add_scalar('Performance/Batch_Time_Seconds', batch_time, step)
        self.writer.add_scalar('Performance/Examples_Per_Second', examples_per_sec, step)
        self.writer.add_scalar('Performance/Progress_Percent', progress * 100, step)
        
    def log_eta(self, eta_seconds, step):
        """
        Log estimated time to completion
        
        Args:
            eta_seconds: Estimated seconds to completion
            step: Current training step
        """
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        self.writer.add_scalar('Training/ETA_Minutes', eta_seconds / 60, step)
        self.writer.add_text('Training/ETA', f"Estimated time remaining: {eta_str}", step)

    def calculate_ssim_psnr(self, real_images, fake_images, step):
        """
        Calculate SSIM and PSNR metrics and log them to TensorBoard
        
        Args:
            real_images: Real image tensors
            fake_images: Generated image tensors
            step: Current training step
        """
        # 忽略 TypedStorage 警告
        

        try:
            # Calculate SSIM and PSNR
            ssim_values = []
            psnr_values = []
            
            # Ensure images are on CPU and clamped to [0, 1] range
            real_np = real_images.detach().cpu().clamp(0, 1)
            fake_np = fake_images.detach().cpu().clamp(0, 1)
            
            # Iterate through each image in the batch
            for i in range(real_np.size(0)):
                try:
                    # Get individual images and convert to (H, W, C) format
                    real_img = real_np[i].permute(1, 2, 0).numpy()
                    fake_img = fake_np[i].permute(1, 2, 0).numpy()
                    
                    # Calculate SSIM (Structural Similarity)
                    # Use channel_axis=2 instead of multichannel=True
                    ssim_value = ssim(real_img, fake_img, channel_axis=2, data_range=1.0)
                    ssim_values.append(ssim_value)
                    
                    # Calculate PSNR (Peak Signal-to-Noise Ratio)
                    psnr_value = psnr(real_img, fake_img, data_range=1.0)
                    psnr_values.append(psnr_value)
                    
                except Exception as e:
                    print(f"Warning: Error calculating metrics for image {i}: {str(e)}")
                    # Continue with next image instead of failing the entire batch
                    continue
            
            # Calculate averages if we have any valid values
            if ssim_values:
                avg_ssim = np.mean(ssim_values)
                self.writer.add_scalar('Metrics/SSIM', avg_ssim, step)
                print(f"Average SSIM: {avg_ssim:.4f}")
            
            if psnr_values:
                avg_psnr = np.mean(psnr_values)
                self.writer.add_scalar('Metrics/PSNR', avg_psnr, step)
                print(f"Average PSNR: {avg_psnr:.4f}")
            
        except Exception as e:
            print(f"Warning: Error calculating image quality metrics: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging

    def write_training_metrics_and_images(self, train_loss, train_style_loss, train_content_loss, train_contents, train_styles, train_styled_content, step):
        """
        Write training metrics and images to TensorBoard
        
        Args:
            train_loss: Training loss
            train_style_loss: Training style loss
            train_content_loss: Training content loss
            train_contents: Training content images
            train_styles: Training style images
            train_styled_content: Training styled content images
            step: Current training step
        """
        self.write_metrics({
            "loss": train_loss,
            "style_loss": train_style_loss,
            "content_loss": train_content_loss
        }, step, prefix="train")

        self.write_images({
            "content_images": train_contents,
            "style_images": train_styles,
            "styled_content_images": train_styled_content
        }, step, prefix="train")