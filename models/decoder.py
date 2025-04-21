import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union, Any

from models.blocks import AdaConv2D, KernelPredictor

class DecoderBlock(nn.Module):
    """
    Decoder Block - A combination of adaptive convolution and regular convolution blocks
    
    Supports dynamic/static mode switching for ONNX export
    """
    def __init__(
        self,
        style_dim: int,
        style_kernel: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        export_config: Dict[str, Any] = None,
        convs: int = 3,
        final_block: bool = False,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        
        # Handle export configuration
        self.export_config = export_config or {}
        self.export_mode = self.export_config.get('export_mode', False)
        self.fixed_batch_size = self.export_config.get('fixed_batch_size', None) if self.export_mode else None
        self.use_fixed_size = self.export_config.get('use_fixed_size', False)
        self.input_hw = self.export_config.get('input_hw', None)
        self.output_hw = self.export_config.get('output_hw', None)
        self.scale_factor = scale_factor
        
        # Shared export settings for internal components
        self.kernel_predictor = KernelPredictor(
            style_dim=style_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            style_kernel=style_kernel,
            export_mode=self.export_mode,
            fixed_batch_size=self.fixed_batch_size
        )

        self.ada_conv = AdaConv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            export_mode=self.export_mode,
            fixed_batch_size=self.fixed_batch_size,
            fixed_hw=self.input_hw if self.use_fixed_size else None
        )

        # Build standard convolution sequence
        decoder_layers = []
        for i in range(convs):
            last_layer = i == (convs - 1)
            _out_channels = out_channels if last_layer else in_channels
            decoder_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=_out_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="zeros",
                )
            )
            # Add activation
            if not last_layer or not final_block:
                decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(nn.Sigmoid())  # Use Sigmoid for final output
        
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
        # Upsampling layer - choose implementation based on mode
        if not final_block:
            if self.use_fixed_size and self.output_hw is not None:
                # Fixed-size upsampling using predefined output dimensions
                self.upsample = self._create_fixed_upsample(self.output_hw)
            else:
                # Dynamic upsampling using scale factor
                self.upsample = self._create_scaled_upsample(scale_factor)
        else:
            # No upsampling needed for final block
            self.upsample = nn.Identity()
    
    def _create_fixed_upsample(self, output_hw: Tuple[int, int]) -> nn.Module:
        """Create fixed-size upsampling layer for ONNX export"""
        class FixedUpsample(nn.Module):
            def __init__(self, output_size: Tuple[int, int]):
                super().__init__()
                self.output_size = output_size
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.interpolate(
                    x, 
                    size=self.output_size, 
                    mode='bilinear', 
                    align_corners=False
                )
        
        return FixedUpsample(output_hw)
    
    def _create_scaled_upsample(self, scale_factor: float) -> nn.Module:
        """Create scale-factor-based upsampling for dynamic sizes"""
        class ScaledUpsample(nn.Module):
            def __init__(self, factor: float):
                super().__init__()
                self.factor = factor
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.interpolate(
                    x, 
                    scale_factor=self.factor, 
                    mode='bilinear', 
                    align_corners=False
                )
        
        return ScaledUpsample(scale_factor)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input feature [B, C, H, W]
            w: Style descriptor [B, style_dim, kernel_h, kernel_w]
            
        Returns:
            Processed feature [B, C_out, H_out, W_out]
        """
        # Predict adaptive convolution kernels
        dw_kernels, pw_kernels, biases = self.kernel_predictor(w)
        
        # Apply adaptive convolution
        x = self.ada_conv(x, dw_kernels, pw_kernels, biases)
        
        # Apply standard convolution layers
        x = self.decoder_layers(x)
        
        # Apply upsampling (if needed)
        x = self.upsample(x)
        
        return x


class Decoder(nn.Module):
    """
    Decoder - Converts content features and style descriptors into stylized images
    
    Supports dynamic/static mode switching for ONNX export
    """
    def __init__(
        self, 
        style_dim: int, 
        style_kernel: int, 
        groups: Union[int, List[int]],
        export_config: Dict[str, Any] = None
    ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.style_kernel = style_kernel
        
        # Handle export configuration
        self.export_config = export_config or {}
        self.export_mode = self.export_config.get('export_mode', False)
        self.fixed_batch_size = self.export_config.get('fixed_batch_size', None) if self.export_mode else None
        self.use_fixed_size = self.export_config.get('use_fixed_size', False)
        self.input_hw = self.export_config.get('input_hw', None)
        
        # Layer structure parameters
        self.input_channels = [512, 256, 128, 64]  # Input channels per layer
        self.output_channels = [256, 128, 64, 3]   # Output channels per layer
        self.n_convs = [1, 2, 2, 4]                # Number of convolutions per layer
        
        # Handle group parameter: int or list
        if isinstance(groups, int):
            self.groups_list = [groups] * len(self.input_channels)
        elif isinstance(groups, list):
            if len(groups) != len(self.input_channels):
                raise ValueError(f"Groups list length ({len(groups)}) must match number of decoder layers ({len(self.input_channels)})")
            self.groups_list = groups
        else:
            # Default grouping strategy: dynamic calculation based on input channels
            self.groups_list = []
            for c in self.input_channels:
                if c >= 512:
                    self.groups_list.append(c // 1)
                elif c >= 256:
                    self.groups_list.append(c // 2)
                elif c >= 128:
                    self.groups_list.append(c // 4)
                else:
                    self.groups_list.append(c // 8)
                    
            # Ensure at least 1 channel per group
            self.groups_list = [max(1, g) for g in self.groups_list]

        # Build decoder blocks
        decoder_blocks = []
        current_hw = self.input_hw
        
        for i, (Cin, Cout, Nc, Group) in enumerate(zip(
            self.input_channels, self.output_channels, self.n_convs, self.groups_list
        )):
            # Determine if final block
            final_block = (i == len(self.input_channels) - 1)
            
            # Calculate output dimensions (if using fixed size)
            output_hw = None
            if self.use_fixed_size and current_hw is not None:
                if not final_block:
                    # Non-final block: double size
                    output_hw = (current_hw[0] * 2, current_hw[1] * 2)
                else:
                    # Final block: keep size unchanged
                    output_hw = current_hw
            
            # Create export config for current block
            block_export_config = {
                'export_mode': self.export_mode,
                'fixed_batch_size': self.fixed_batch_size,
                'use_fixed_size': self.use_fixed_size,
                'input_hw': current_hw,
                'output_hw': output_hw
            }
            
            # Create decoder block
            decoder_blocks.append(
                DecoderBlock(
                    style_dim=style_dim,
                    style_kernel=style_kernel,
                    in_channels=Cin,
                    out_channels=Cout,
                    groups=Group,
                    export_config=block_export_config,
                    convs=Nc,
                    final_block=final_block,
                )
            )
            
            # Update current dimensions (for next layer)
            if self.use_fixed_size:
                current_hw = output_hw
                
        # Save decoder block sequence
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input feature [B, C, H, W]
            w: Style descriptor [B, style_dim, kernel_h, kernel_w]
            
        Returns:
            Stylized image [B, 3, H_out, W_out]
        """
        # Process through all decoder blocks
        for layer in self.decoder_blocks:
            x = layer(x, w)
        return x