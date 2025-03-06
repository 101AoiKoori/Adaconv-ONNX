import torch
from torch import nn
from torch.nn import functional as F

# ---------------- GlobalStyleEncoder ----------------
class GlobalStyleEncoder(nn.Module):
    def __init__(self, style_feat_shape: tuple[int], style_descriptor_shape: tuple[int], fixed_batch_size: int) -> None:
        super().__init__()
        self.style_feat_shape = style_feat_shape
        self.style_descriptor_shape = style_descriptor_shape
        self.fixed_batch_size = fixed_batch_size
        
        channels = style_feat_shape[0]
        self.style_encoder = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="zeros"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="zeros"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode="zeros"),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(),
        )
        in_features = int (style_feat_shape[0] * (style_feat_shape[1] // 8) * (style_feat_shape[2] // 8))
        out_features = int (style_descriptor_shape[0] * style_descriptor_shape[1] * style_descriptor_shape[2])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.style_encoder(x)
        x = torch.flatten(x, start_dim=1)
        w = self.fc(x)
        w = w.reshape(self.fixed_batch_size, 
                      self.style_descriptor_shape[0],
                      self.style_descriptor_shape[1],
                      self.style_descriptor_shape[2])
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
        fixed_batch_size: int
    ):
        super().__init__()
        self.style_dim = style_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_kernel = style_kernel
        self.groups = groups # 使用固定groups
        self.fixed_batch_size = fixed_batch_size # 使用固定batch_size

        self.depthwise_conv_kernel_predictor = nn.Conv2d(
            in_channels=self.style_dim,
            out_channels=out_channels * (in_channels // groups),
            kernel_size=3,
            padding=1,
            padding_mode="zeros"
        )

        self.pointwise_conv_kernel_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=out_channels * (out_channels // groups),
                kernel_size=1,
            ),
        )

        self.bias_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=out_channels,  # 输出形状为 [B, C_out, 1, 1]
                kernel_size=1,
            ),
        )

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = self.fixed_batch_size  # 使用固定batch size，而非动态获取
        dw_kernel = self.depthwise_conv_kernel_predictor(w)
        dw_kernel = dw_kernel.reshape(
            B,  # 使用固定值
            self.out_channels,
            self.in_channels // self.groups,
            self.style_kernel,
            self.style_kernel,
        )
        # 预测点卷积核：形状 [B, C_out, out_channels//groups, 1, 1]
        pw_kernel = self.pointwise_conv_kernel_predictor(w)
        pw_kernel = pw_kernel.reshape(
            B,
            self.out_channels,
            self.out_channels // self.groups,
            1,
            1,
        )
        # 预测偏置：形状 [B, C_out]，拉平成 [B * C_out]
        bias = self.bias_predictor(w)
        bias = bias.reshape(B, self.out_channels).reshape(-1)
        return (dw_kernel, pw_kernel, bias)

# ---------------- AdaConv2D ----------------
class AdaConv2D(nn.Module):
    def __init__(self, 
            in_channels: int, 
            out_channels: int, 
            groups: int, 
            fixed_batch_size: int, 
            fixed_hw: tuple[int, int]
        ): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.fixed_batch_size = fixed_batch_size
        self.fixed_h = fixed_hw[0]  # 新增：固定高度
        self.fixed_w = fixed_hw[1]  # 新增：固定宽度

        self.output_h = self.fixed_h  # 根据实际卷积操作调整
        self.output_w = self.fixed_w  # 例如：H_out = H_in - K + 1

        self._epsilon = 1e-7

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.std(x, dim=[2, 3], keepdim=True) + self._epsilon
        return (x - mean) / std

    def forward(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        B = self.fixed_batch_size
        x = self._normalize(x)
        K = dw_kernels.shape[-1]
        padding = (K - 1) // 2
        
        # 静态计算输出尺寸
        H_out = self.fixed_h + 2 * padding - (K - 1)  # 新增：固定高度计算
        W_out = self.fixed_w + 2 * padding - (K - 1)  # 新增：固定宽度计算
        
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        x_merged = x_padded.reshape(1, B * self.in_channels, x_padded.shape[2], x_padded.shape[3])
        dw_kernels_merged = dw_kernels.reshape(B * self.out_channels, self.in_channels // self.groups, K, K)
        conv_groups = B * self.groups

        depthwise_out = F.conv2d(x_merged, dw_kernels_merged, groups=conv_groups, padding=0)
        depthwise_out = depthwise_out.reshape(B, self.out_channels, H_out, W_out)  # 使用固定高度和宽度

        pw_kernels_merged = pw_kernels.reshape(
            B * self.out_channels, 
            self.out_channels // self.groups, 
            1, 
            1
        )
        depthwise_merged = depthwise_out.reshape(1, B * self.out_channels, H_out, W_out)
        output = F.conv2d(depthwise_merged, pw_kernels_merged, bias=biases, groups=conv_groups)
        output = output.reshape(B, self.out_channels, H_out, W_out)
        
        return output

# ---------------- main 调试 ----------------
def main():
    # 固定 batch size、in/out 通道、组数以及图像尺寸
    B = 2
    C_in = 64
    C_out = 128
    groups = 32  # 用户在 hyperparam.yaml 中指定的 groups
    H, W = 64, 64
    K = 3  # kernel size

    # 创建 AdaConv2D 模块，传入固定 batch size 和固定组数
    adaconv = AdaConv2D(in_channels=C_in, out_channels=C_out, groups=groups, fixed_batch_size=B, fixed_hw=(H, W))  # 新增：固定高度和宽度
    
    # 构造测试输入 x
    x = torch.randn(B, C_in, H, W)
    
    # 生成深度卷积核，注意：C_in // groups = 64 // 32 = 2
    dw_kernels = torch.randn(B, C_out, C_in // groups, K, K)
    # 生成点卷积核：形状 [B, C_out, C_out//groups, 1, 1] => 128 // 32 = 4
    pw_kernels = torch.randn(B, C_out, C_out // groups, 1, 1)
    # 生成偏置：形状 [B * C_out]
    biases = torch.randn(B * C_out)

    print("==== 开始 AdaConv2D 测试 ====")
    output = adaconv(x, dw_kernels, pw_kernels, biases)
    print(f"Final output shape: {output.shape}")

if __name__ == '__main__':
    main()