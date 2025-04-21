import torch
from torch import nn
from typing import Optional, Tuple, Dict, List, Union, Any

from models.blocks import GlobalStyleEncoder, KernelPredictor, AdaConv2D
from models.decoder import Decoder
from models.encoder import Encoder
from models.utils import init_weights

class StyleTransfer(nn.Module):
    def __init__(
        self, 
        image_shape: Tuple[int, int], 
        style_dim: int, 
        style_kernel: int, 
        groups: Union[int, List[int]] = None,
        export_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        AdaConv Style Transfer Model
        
        Args:
            image_shape: Input image dimensions (height, width)
            style_dim: Number of style descriptor channels
            style_kernel: Adaptive convolution kernel size
            groups: Number of groups for grouped convolution, can be single integer or list (different grouping per layer)
            export_config: Export configuration with optional fields:
                - export_mode: Whether to enable export mode (default: False)
                - fixed_batch_size: Fixed batch size for export (default: None)
                - use_fixed_size: Whether to use fixed spatial dimensions (default: False)
        """
        super().__init__()
        self.image_shape = image_shape
        self.style_dim = style_dim
        self.style_kernel = style_kernel
        self.groups = groups
        
        # Process export configuration
        self.export_config = export_config or {}
        self.export_mode = self.export_config.get('export_mode', False)
        self.fixed_batch_size = self.export_config.get('fixed_batch_size', None)
        self.use_fixed_size = self.export_config.get('use_fixed_size', False)

        # Initialize encoder (VGG)
        self.encoder = Encoder()
        self.encoder.freeze()
        encoder_scale = self.encoder.scale_factor
        
        # Calculate encoder output dimensions
        encoder_hw = None
        if self.use_fixed_size:
            encoder_hw = (
                self.image_shape[0] // encoder_scale,
                self.image_shape[1] // encoder_scale
            )

        # Initialize global style encoder
        self.global_style_encoder = GlobalStyleEncoder(
            style_feat_shape=(
                self.style_dim,
                self.image_shape[0] // encoder_scale,
                self.image_shape[1] // encoder_scale,
            ),
            style_descriptor_shape=(
                self.style_dim,
                self.style_kernel,
                self.style_kernel,
            ),
            export_mode=self.export_mode,
            fixed_batch_size=self.fixed_batch_size
        )
        
        # Initialize decoder
        self.decoder = Decoder(
            style_dim=self.style_dim,
            style_kernel=self.style_kernel,
            groups=self.groups,
            export_config={
                'export_mode': self.export_mode,
                'fixed_batch_size': self.fixed_batch_size,
                'use_fixed_size': self.use_fixed_size,
                'input_hw': encoder_hw
            }
        )

        # Initialize weights
        self.apply(init_weights)
        
        # Normalization settings
        self.freeze_normalization = False

    def set_export_mode(self, enabled: bool = True, fixed_batch_size: Optional[int] = None):
        """
        Set model export mode
        
        Args:
            enabled: Whether to enable export mode
            fixed_batch_size: Fixed batch size for export
        """
        self.export_mode = enabled
        if enabled and fixed_batch_size is not None:
            self.fixed_batch_size = fixed_batch_size
        elif not enabled:
            self.fixed_batch_size = None
        
        # Update export mode for all submodules
        for module in self.modules():
            if hasattr(module, 'export_mode'):
                module.export_mode = enabled
            if hasattr(module, 'fixed_batch_size') and enabled and fixed_batch_size is not None:
                module.fixed_batch_size = fixed_batch_size

    def freeze_normalization_layers(self):
        """Freeze all normalization layers"""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                m.eval()  # Freeze statistics computation
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad_(False)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad_(False)
        self.freeze_normalization = True

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns style transfer result
        
        Args:
            content: Content image [B, 3, H, W]
            style: Style image [B, 3, H, W]
            
        Returns:
            Stylized content image [B, 3, H, W]
        """
        # Freeze normalization layers in evaluation mode
        if not self.training and not self.freeze_normalization:
            self.freeze_normalization_layers()
            
        # Extract content and style features
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        
        # Generate style descriptor
        w = self.global_style_encoder(style_feats[-1])
        
        # Decode to generate stylized image
        x = self.decoder(content_feats[-1], w)
        
        return x

    def forward_with_features(self, content: torch.Tensor, style: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass - returns style transfer result and intermediate features (for training and loss computation)
        
        Args:
            content: Content image [B, 3, H, W]
            style: Style image [B, 3, H, W]
            
        Returns:
            (x, content_feats, style_feats, x_feats):
                x: Stylized content image [B, 3, H, W]
                content_feats: List of content features
                style_feats: List of style features
                x_feats: List of output features
        """
        # Extract content and style features
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        
        # Generate style descriptor
        w = self.global_style_encoder(style_feats[-1])
        
        # Decode to generate stylized image
        x = self.decoder(content_feats[-1], w)
        
        # Extract output features (for loss computation)
        x_feats = self.encoder(x)
        
        return x, content_feats, style_feats, x_feats