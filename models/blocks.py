import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Union, Any

class GlobalStyleEncoder(nn.Module):
    """
    Global Style Encoder - Generates style descriptors from style features
    
    Supports dynamic/static mode switching for ONNX export compatibility
    """
    def __init__(
            self, 
            style_feat_shape: tuple[int], 
            style_descriptor_shape: tuple[int], 
            export_mode: bool = False,
            fixed_batch_size: Optional[int] = None
        ) -> None:
        super().__init__()
        self.style_feat_shape = style_feat_shape
        self.style_descriptor_shape = style_descriptor_shape
        self.export_mode = export_mode
        self.fixed_batch_size = fixed_batch_size if export_mode else None
        channels = style_feat_shape[0]

        self.style_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1), padding_mode="reflect"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1), padding_mode="reflect"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1), padding_mode="reflect"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
        )
        in_features = int(style_feat_shape[0] * (style_feat_shape[1] // 8) * (style_feat_shape[2] // 8))
        out_features = int(style_descriptor_shape[0] * style_descriptor_shape[1] * style_descriptor_shape[2])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get batch size - Fix dynamic batch size issue
        batch_size = self.fixed_batch_size if self.export_mode and self.fixed_batch_size is not None else x.size(0)
        
        # Feature extraction
        x = self.style_encoder(x)
        x = torch.flatten(x, start_dim=1)
        w = self.fc(x)
        
        # Reshape to descriptor
        w = w.reshape(
            batch_size,
            self.style_descriptor_shape[0],
            self.style_descriptor_shape[1],
            self.style_descriptor_shape[2]
        )
        return w


class KernelPredictor(nn.Module):
    """
    Kernel Predictor - Generates dynamic convolution kernels from style descriptors
    
    Supports dynamic/static mode switching for ONNX export compatibility
    """
    def __init__(
        self, 
        style_dim: int, 
        in_channels: int, 
        out_channels: int,
        groups: int, 
        style_kernel: int,
        export_mode: bool = False,
        fixed_batch_size: Optional[int] = None
    ):
        super().__init__()
        self.style_dim = style_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_kernel = style_kernel
        self.groups = groups
        self.export_mode = export_mode
        self.fixed_batch_size = fixed_batch_size if export_mode else None

        # Depthwise convolution kernel predictor
        self.depthwise_conv_kernel_predictor = nn.Conv2d(
            in_channels=self.style_dim,
            out_channels=self.out_channels * (self.in_channels // self.groups),
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )

        # Pointwise convolution kernel predictor
        self.pointwise_conv_kernel_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=self.out_channels * (self.out_channels // self.groups),
                kernel_size=1,
            ),
        )

        # Bias predictor
        self.bias_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get batch size - Fix dynamic batch size issue
        B = self.fixed_batch_size if self.export_mode and self.fixed_batch_size is not None else w.size(0)
        
        # Predict depthwise kernel: shape [B, C_out, C_in//groups, K, K]
        dw_kernel = self.depthwise_conv_kernel_predictor(w)
        dw_kernel = dw_kernel.reshape(
            B,
            self.out_channels,
            self.in_channels // self.groups,
            self.style_kernel,
            self.style_kernel,
        )
        
        # Predict pointwise kernel: shape [B, C_out, C_out//groups, 1, 1]
        pw_kernel = self.pointwise_conv_kernel_predictor(w)
        pw_kernel = pw_kernel.reshape(
            B,
            self.out_channels,
            self.out_channels // self.groups,
            1,
            1,
        )
        
        # Predict bias: shape [B, C_out], flatten to [B * C_out]
        bias = self.bias_predictor(w)
        bias = bias.reshape(B, self.out_channels)
        if self.export_mode and self.fixed_batch_size is not None:
            # Flatten bias in export mode to shape [B*C_out]
            bias = bias.reshape(-1)
        
        return (dw_kernel, pw_kernel, bias)


class AdaConv2D(nn.Module):
    """
    Adaptive Convolution Layer - Applies dynamically generated kernels to input features
    
    Supports dynamic/static mode switching for ONNX export compatibility
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        groups: int,
        export_mode: bool = False,
        fixed_batch_size: Optional[int] = None, 
        fixed_hw: Optional[Tuple[int, int]] = None
    ): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.export_mode = export_mode
        self.fixed_batch_size = fixed_batch_size if export_mode else None
        self.fixed_hw = fixed_hw if export_mode else None
        self._epsilon = 1e-7
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input features"""
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.std(x, dim=[2, 3], keepdim=True) + self._epsilon
        return (x - mean) / std

    def forward(
        self, 
        x: torch.Tensor, 
        dw_kernels: torch.Tensor, 
        pw_kernels: torch.Tensor, 
        biases: torch.Tensor
    ) -> torch.Tensor:
        # Check whether to use export mode
        if self.export_mode:
            # Export mode: use vectorized grouped convolution
            if self.fixed_batch_size is None:
                # Dynamic batch size mode
                return self._forward_dynamic_batch(x, dw_kernels, pw_kernels, biases)
            else:
                # Static batch size mode
                return self._forward_static(x, dw_kernels, pw_kernels, biases)
        else:
            # Dynamic mode: decide which method to use based on input
            B = x.size(0)
            if B > 1:
                # Use vectorized implementation when batch size > 1
                return self._forward_batched(x, dw_kernels, pw_kernels, biases)
            else:
                # Use simple implementation when batch size = 1 (avoid unnecessary dimension transforms)
                return self._forward_simple(x, dw_kernels, pw_kernels, biases)
    
    def _forward_dynamic_batch(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """Forward pass in dynamic batch size mode, specially optimized for ONNX export"""
        # Get current batch size
        B = x.size(0)
        
        # Normalize input
        x = self._normalize(x)
        
        # Process each sample to avoid reshape issues with dynamic batch size
        outputs = []
        for i in range(B):
            # Extract single sample's input and kernels
            x_i = x[i:i+1]
            dw_kernel_i = dw_kernels[i:i+1]
            pw_kernel_i = pw_kernels[i:i+1]
            bias_i = biases[i:i+1] if biases.dim() > 1 else biases
            
            # Apply simplified forward pass for each sample
            out_i = self._forward_simple(x_i, dw_kernel_i, pw_kernel_i, bias_i)
            outputs.append(out_i)
        
        # Concatenate all outputs
        return torch.cat(outputs, dim=0)
    
    def _forward_static(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """Forward pass in export mode with fixed batch size and spatial dimensions"""
        # Ensure batch size is fixed in static mode
        B = self.fixed_batch_size if self.fixed_batch_size is not None else x.size(0)
            
        # Normalize input
        x = self._normalize(x)
        
        # Get kernel size and calculate padding
        K = dw_kernels.shape[-1]
        padding = (K - 1) // 2
        
        # Get input size (use fixed size or get from input)
        if self.fixed_hw is not None:
            H_in, W_in = self.fixed_hw
        else:
            H_in, W_in = x.shape[2], x.shape[3]
        
        # Calculate output size - always same as input
        H_out, W_out = H_in, W_in
        
        # Prepare input - use fixed shape
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        
        # Use fixed batch size
        # Use more explicit reshape operation to avoid dynamic dimensions
        x_merged = x_padded.reshape(1, B * self.in_channels, x_padded.shape[2], x_padded.shape[3])
        
        # Provide specific shapes for kernels to avoid dynamic dimensions
        dw_kernels_merged = dw_kernels.reshape(B * self.out_channels, self.in_channels // self.groups, K, K)
        pw_kernels_merged = pw_kernels.reshape(B * self.out_channels, self.out_channels // self.groups, 1, 1)
        
        # Ensure bias is one-dimensional
        if biases.dim() > 1:
            biases = biases.reshape(-1)
            
        # Calculate total groups
        conv_groups = B * self.groups
        
        # Depthwise convolution
        depthwise_out = F.conv2d(x_merged, dw_kernels_merged, groups=conv_groups, padding=0)
        
        # Use fixed shape
        depthwise_out = depthwise_out.reshape(B, self.out_channels, H_out, W_out)
        
        # Reshape before pointwise convolution
        depthwise_merged = depthwise_out.reshape(1, B * self.out_channels, H_out, W_out)
        
        # Pointwise convolution
        output = F.conv2d(depthwise_merged, pw_kernels_merged, bias=biases, groups=conv_groups)
        
        # Final output shape
        output = output.reshape(B, self.out_channels, H_out, W_out)
        
        return output
    
    def _forward_batched(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """Forward pass in batch mode with variable batch size support"""
        # Get batch size
        B = x.size(0)
        
        # Normalize input
        x = self._normalize(x)
        
        # Get kernel size and calculate padding
        K = dw_kernels.shape[-1]
        padding = (K - 1) // 2
        
        # Prepare input and kernels
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        x_merged = x_padded.reshape(1, B * self.in_channels, x_padded.shape[2], x_padded.shape[3])
        dw_kernels_merged = dw_kernels.reshape(B * self.out_channels, self.in_channels // self.groups, K, K)
        pw_kernels_merged = pw_kernels.reshape(B * self.out_channels, self.out_channels // self.groups, 1, 1)
        
        # Calculate total groups
        conv_groups = B * self.groups
        
        # Get output size
        H_out, W_out = x.shape[2], x.shape[3]
        
        # Depthwise convolution
        depthwise_out = F.conv2d(x_merged, dw_kernels_merged, groups=conv_groups, padding=0)
        depthwise_out = depthwise_out.reshape(B, self.out_channels, H_out, W_out)
        
        # Pointwise convolution
        depthwise_merged = depthwise_out.reshape(1, B * self.out_channels, H_out, W_out)
        # Flatten bias to one dimension
        if biases.dim() > 1:
            biases = biases.reshape(-1)
        output = F.conv2d(depthwise_merged, pw_kernels_merged, bias=biases, groups=conv_groups)
        output = output.reshape(B, self.out_channels, H_out, W_out)
        
        return output
    
    def _forward_simple(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """Simple forward pass for single sample case, avoiding unnecessary dimension transforms"""
        # Normalize input
        x = self._normalize(x)
        
        # Get kernel size and calculate padding
        K = dw_kernels.shape[-1]
        padding = (K - 1) // 2
        
        # Pad input
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        
        # Extract kernels and bias for single sample
        dw_kernel = dw_kernels[0]  # [C_out, C_in//groups, K, K]
        pw_kernel = pw_kernels[0]  # [C_out, C_out//groups, 1, 1]
        bias = biases[0] if biases.dim() > 1 else biases  # [C_out]
        
        # Depthwise convolution
        depthwise_out = F.conv2d(x_padded, dw_kernel, groups=self.groups, padding=0)
        
        # Pointwise convolution
        output = F.conv2d(depthwise_out, pw_kernel, bias=bias, groups=self.groups)
        
        return output