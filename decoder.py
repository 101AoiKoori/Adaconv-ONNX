import torch
from blocks import AdaConv2D, KernelPredictor
from torch import nn

class DecoderBlock(nn.Module):
    def __init__(
        self,
        style_dim: int,
        style_kernel: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        fixed_batch_size: int,
        input_hw: tuple[int, int],  # 新增：输入特征图的固定尺寸
        output_hw: tuple[int, int],  # 新增：输出特征图的固定尺寸
        convs: int,
        final_block: bool = False,
    ) -> None:
        super().__init__()
        self.groups = groups
        self.fixed_batch_size = fixed_batch_size
        self.input_hw = input_hw  # 新增：输入特征图的固定尺寸
        self.output_hw = output_hw  # 新增：输出特征图的固定尺寸

        # 使用固定的 groups 构造 KernelPredictor
        self.kernel_predictor = KernelPredictor(
            style_dim=style_dim,
            in_channels=in_channels,
            out_channels=in_channels,  # 保持与原始代码一致
            groups=groups,
            style_kernel=style_kernel,
            fixed_batch_size=fixed_batch_size
        )

        # 使用固定的 groups 和 fixed_batch_size 构造 AdaConv2D
        self.ada_conv = AdaConv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            fixed_batch_size=fixed_batch_size,
            fixed_hw=input_hw  # 新增：传递固定尺寸
        )

        # 构建后续的普通卷积层、激活函数和上采样模块
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
            decoder_layers.append(
                nn.ReLU() if not last_layer or not final_block else nn.Sigmoid()
            )
        if not final_block:
            decoder_layers.append(nn.Upsample(size=output_hw, mode="nearest"))  # 修改：使用固定尺寸

        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # 预测自适应卷积核（包括深度卷积核、点卷积核和偏置）
        dw_kernels, pw_kernels, biases = self.kernel_predictor(w)
        # 调用 AdaConv2D 执行自适应卷积
        x = self.ada_conv(x, dw_kernels, pw_kernels, biases)
        # 执行后续普通卷积、激活和上采样
        x = self.decoder_layers(x)
        return x

class Decoder(nn.Module):
    def __init__(self, style_dim: int, style_kernel: int, groups: int, fixed_batch_size: int, input_hw: tuple[int, int]) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.style_kernel = style_kernel
        # 按原代码设定每个解码块的输入和输出通道数量
        self.input_channels = [512, 256, 128, 64]
        self.output_channels = [256, 128, 64, 3]
        self.n_convs = [1, 2, 2, 4]

        decoder_blocks = []
        current_hw = input_hw  # 新增：初始输入尺寸
        for i, Cin in enumerate(self.input_channels):
            Cout = self.output_channels[i]
            Nc = self.n_convs[i]
            final_block = (i == len(self.input_channels) - 1)
            output_hw = (current_hw[0] * 2, current_hw[1] * 2) if not final_block else current_hw  # 新增：计算输出尺寸
            decoder_blocks.append(
                DecoderBlock(
                    style_dim=style_dim,
                    style_kernel=style_kernel,
                    in_channels=Cin,
                    out_channels=Cout,
                    groups=groups,
                    fixed_batch_size=fixed_batch_size,
                    input_hw=current_hw,  # 新增：传递输入尺寸
                    output_hw=output_hw,  # 新增：传递输出尺寸
                    convs=Nc,
                    final_block=final_block,
                )
            )
            current_hw = output_hw  # 更新当前尺寸
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_blocks:
            x = layer(x, w)
        return x