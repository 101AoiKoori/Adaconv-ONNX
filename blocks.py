import torch
from torch import nn
from torch.nn import functional as F


class GlobalStyleEncoder(nn.Module):
    def __init__(
        self, style_feat_shape: tuple[int], style_descriptor_shape: tuple[int]
    ) -> None:
        super().__init__()
        self.style_feat_shape = style_feat_shape
        self.style_descriptor_shape = style_descriptor_shape
        channels = self.style_feat_shape[0]

        self.style_encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="reflect",
            ),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            # Block 2
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="reflect",
            ),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            # Block 3
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                padding_mode="reflect",
            ),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
        )

        in_features = int(
            self.style_feat_shape[0]
            * (self.style_feat_shape[1] // 8)
            * (self.style_feat_shape[2] // 8)
        )
        out_features = int(
            self.style_descriptor_shape[0]
            * self.style_descriptor_shape[1]
            * self.style_descriptor_shape[2]
        )
        self.fc = nn.Linear(
            in_features=in_features,
            out_features=out_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # style encoder
        x = self.style_encoder(x)
        # fully connected
        x = torch.flatten(x, start_dim=1)
        w = self.fc(x)
        # global embeddings
        w = w.view(
            -1,
            self.style_descriptor_shape[0],
            self.style_descriptor_shape[1],
            self.style_descriptor_shape[2],
        )
        return w


class KernelPredictor(nn.Module):
    def __init__(
        self,
        style_dim: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        style_kernel: int,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.style_kernel = style_kernel

        self.depthwise_conv_kernel_predictor = nn.Conv2d(
            in_channels=self.style_dim,
            out_channels=self.out_channels * (self.in_channels // self.groups),
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )
        self.pointwise_conv_kernel_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=self.out_channels * (self.out_channels // self.groups),
                kernel_size=(1, 1),
            ),
        )

        self.bias_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=self.out_channels,  # 输出总偏置数为 batch_size * out_channels
                kernel_size=(1, 1),
            ),
        )


    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor]:
        batch_size = w.shape[0]  # 动态获取 batch_size
        # depthwise kernel
        dw_kernel = self.depthwise_conv_kernel_predictor(w)
        dw_kernel = dw_kernel.view(
            batch_size,
            self.out_channels,
            self.in_channels // self.groups,
            self.style_kernel,
            self.style_kernel,
        )
        # pointwise kernel
        pw_kernel = self.pointwise_conv_kernel_predictor(w)
        pw_kernel = pw_kernel.view(
            batch_size,
            self.out_channels,
            self.out_channels // self.groups,
            1,
            1,
        )
        # bias
        bias = self.bias_predictor(w)
        bias = bias.view(batch_size, self.out_channels)   # 形状为 [B, out_channels]
        return (dw_kernel, pw_kernel, bias)

class AdaConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int):
        super().__init__()
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._epsilon = 1e-7

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.std(x, dim=[2, 3], keepdim=True) + self._epsilon
        return (x - mean) / std

    def forward(self, x: torch.Tensor, dw_kernels: torch.Tensor, 
                pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        batch_size = x.size(0)
        padding = (dw_kernels.size(-1) - 1) // 2

        # 深度卷积（向量化实现）
        x_padded = F.pad(x, (padding,)*4, mode="constant", value=0)
        x_merged = x_padded.view(1, -1, *x_padded.shape[2:])  # [1, B*C_in, H, W]
        
        dw_kernels_merged = dw_kernels.view(-1, *dw_kernels.shape[2:])  # [B*C_out, C_in/G, K, K]
        depthwise_out = F.conv2d(
            x_merged, 
            dw_kernels_merged,
            groups=batch_size * self.groups,
            padding=0
        )  # [1, B*C_out, H, W]
        depthwise_out = depthwise_out.view(batch_size, self.out_channels, *depthwise_out.shape[2:])

        # 点卷积（向量化实现）
        pw_kernels_merged = pw_kernels.view(-1, *pw_kernels.shape[2:])  # [B*C_out, C_out/G, 1, 1]
        depthwise_merged = depthwise_out.view(1, -1, *depthwise_out.shape[2:])  # [1, B*C_out, H, W]
        
        pointwise_out = F.conv2d(
            depthwise_merged,
            pw_kernels_merged,
            bias=biases.view(-1),  # [B*C_out]
            groups=batch_size * self.groups
        )  # [1, B*C_out, H, W]
        
        output = pointwise_out.view(batch_size, self.out_channels, *pointwise_out.shape[2:])
        return output
#------------------------------------------------
