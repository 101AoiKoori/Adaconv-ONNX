import torch
from torch import nn
from torch.nn import functional as F

# ---------------- GlobalStyleEncoder ----------------
class GlobalStyleEncoder(nn.Module):
    def __init__(
            self, style_feat_shape: tuple[int], style_descriptor_shape: tuple[int], fixed_batch_size: int = None
        ) -> None:
        super().__init__()
        self.style_feat_shape = style_feat_shape
        self.style_descriptor_shape = style_descriptor_shape
        self.fixed_batch_size = fixed_batch_size
        channels = style_feat_shape[0]

        self.style_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1), padding_mode="zeros"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1), padding_mode="zeros"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3,3), padding=(1,1), padding_mode="zeros"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
        )
        in_features = int(style_feat_shape[0] * (style_feat_shape[1] // 8) * (style_feat_shape[2] // 8))
        out_features = int(style_descriptor_shape[0] * style_descriptor_shape[1] * style_descriptor_shape[2])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the actual batch size from input or use fixed value for export
        batch_size = self.fixed_batch_size if self.fixed_batch_size is not None else x.size(0)
        
        x = self.style_encoder(x)
        x = torch.flatten(x, start_dim=1)
        w = self.fc(x)
        
        # reshape to final descriptor shape
        w = w.reshape(
            batch_size,
            self.style_descriptor_shape[0],
            self.style_descriptor_shape[1],
            self.style_descriptor_shape[2]
        )
        return w

# ---------------- KernelPredictor ----------------
class KernelPredictor(nn.Module):
    def __init__(
        self, 
        style_dim: int, 
        in_channels: int, 
        out_channels: int,
        groups: int, 
        style_kernel: int, 
        fixed_batch_size: int = None
    ):
        super().__init__()
        self.style_dim = style_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_kernel = style_kernel
        self.groups = groups
        self.fixed_batch_size = fixed_batch_size

        self.depthwise_conv_kernel_predictor = nn.Conv2d(
            in_channels=self.style_dim,
            out_channels=self.out_channels * (self.in_channels // self.groups),
            kernel_size=3,
            padding=1,
            padding_mode="zeros"
        )

        self.pointwise_conv_kernel_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=self.out_channels * (self.out_channels // self.groups),
                kernel_size=1,
            ),
        )

        self.bias_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get batch size from input or use fixed value for export
        B = self.fixed_batch_size if self.fixed_batch_size is not None else w.size(0)
        
        # Predict depthwise conv kernel: shape [B, C_out, C_in//groups, K, K]
        dw_kernel = self.depthwise_conv_kernel_predictor(w)
        dw_kernel = dw_kernel.reshape(
            B,
            self.out_channels,
            self.in_channels // self.groups,
            self.style_kernel,
            self.style_kernel,
        )
        
        # Predict pointwise conv kernel: shape [B, C_out, C_out//groups, 1, 1]
        pw_kernel = self.pointwise_conv_kernel_predictor(w)
        pw_kernel = pw_kernel.reshape(
            B,
            self.out_channels,
            self.out_channels // self.groups,
            1,
            1,
        )
        
        # Predict bias: shape [B, C_out], flattened to [B * C_out]
        bias = self.bias_predictor(w)
        bias = bias.reshape(B, self.out_channels).reshape(-1)
        
        return (dw_kernel, pw_kernel, bias)

# ---------------- AdaConv2D ----------------
class AdaConv2D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        groups: int, 
        fixed_batch_size: int = None, 
        fixed_hw: tuple[int, int] = None
    ): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.fixed_batch_size = fixed_batch_size
        
        # Use fixed dimensions if provided, otherwise get dynamically
        if fixed_hw is not None:
            self.fixed_h = fixed_hw[0]
            self.fixed_w = fixed_hw[1]
            self.use_fixed_hw = True
        else:
            self.use_fixed_hw = False
            
        self._epsilon = 1e-7

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
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
        # Get batch size from input or use fixed value
        B = self.fixed_batch_size if self.fixed_batch_size is not None else x.size(0)
        
        # Normalize input
        x = self._normalize(x)
        
        # Get kernel size and calculate padding
        K = dw_kernels.shape[-1]
        padding = (K - 1) // 2
        
        # Get input dimensions (dynamic or static)
        if self.use_fixed_hw:
            H_in, W_in = self.fixed_h, self.fixed_w
        else:
            H_in, W_in = x.shape[2], x.shape[3]
        
        # Calculate output dimensions (same as input with proper padding)
        H_out = H_in
        W_out = W_in
        
        # Prepare input and convolution kernels
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        x_merged = x_padded.reshape(1, B * self.in_channels, x_padded.shape[2], x_padded.shape[3])
        dw_kernels_merged = dw_kernels.reshape(B * self.out_channels, self.in_channels // self.groups, K, K)
        pw_kernels_merged = pw_kernels.reshape(B * self.out_channels, self.out_channels // self.groups, 1, 1)
        
        # Execute depthwise separable convolution
        output = self._depthwise_separable_conv2D(x_merged, dw_kernels_merged, pw_kernels_merged, biases, B, H_out, W_out)
        
        return output

    def _depthwise_separable_conv2D(self, x, dw_kernel, pw_kernel, bias, B, H_out, W_out):
        # Calculate total groups for depthwise convolution
        conv_groups = B * self.groups
        
        # Depthwise convolution
        depthwise_out = F.conv2d(x, dw_kernel, groups=conv_groups, padding=0)
        depthwise_out = depthwise_out.reshape(B, self.out_channels, H_out, W_out)
        
        # Pointwise convolution
        depthwise_merged = depthwise_out.reshape(1, B * self.out_channels, H_out, W_out)
        output = F.conv2d(depthwise_merged, pw_kernel, bias=bias, groups=conv_groups)
        output = output.reshape(B, self.out_channels, H_out, W_out)
        
        return output
#--------------------------------------