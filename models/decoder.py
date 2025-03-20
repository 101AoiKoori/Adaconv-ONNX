import torch
from models.blocks import AdaConv2D, KernelPredictor
from torch import nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(
        self,
        style_dim: int,
        style_kernel: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        fixed_batch_size: int,
        input_hw: tuple[int, int] = None, 
        output_hw: tuple[int, int] = None, 
        convs: int = 1, # 添加卷积层数参数
        final_block: bool = False, # 添加标志以确定是否为最终块
        scale_factor: int = 2, # 添加比例因子参数
    ) -> None:
        super().__init__()
        self.groups = groups
        self.fixed_batch_size = fixed_batch_size
        self.input_hw = input_hw
        self.output_hw = output_hw
        self.scale_factor = scale_factor
        self.use_fixed_hw = input_hw is not None and output_hw is not None

        # KernelPredictor 保持不变
        self.kernel_predictor = KernelPredictor(
            style_dim=style_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            style_kernel=style_kernel,
            fixed_batch_size=fixed_batch_size
        )

        # 传递可选的固定尺寸
        self.ada_conv = AdaConv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            fixed_batch_size=fixed_batch_size,
            fixed_hw=input_hw if self.use_fixed_hw else None
        )

        # 构建后续层
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
        
        # 修改：使用更兼容的上采样方法
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
        # 将上采样从Sequential中分离出来
        if not final_block:
            if self.use_fixed_hw:
                # 使用确定的输出尺寸，通过卷积+插值替代直接使用nn.Upsample
                self.upsample = self._create_fixed_upsample(output_hw)
            else:
                # 使用比例因子，通过卷积+插值替代直接使用scale_factor参数
                self.upsample = self._create_scaled_upsample(scale_factor)
        else:
            self.upsample = nn.Identity()
    
    def _create_fixed_upsample(self, output_hw):
        """创建一个更兼容ONNX常量折叠的固定尺寸上采样层"""
        class FixedUpsample(nn.Module):
            def __init__(self, output_size):
                super().__init__()
                self.output_size = output_size
            
            def forward(self, x):
                # 使用align_corners=False和mode='nearest'以提高ONNX兼容性
                return F.interpolate(
                    x, 
                    size=self.output_size, 
                    mode='bilinear', 
                    align_corners=None
                )
        
        return FixedUpsample(output_hw)
    
    def _create_scaled_upsample(self, scale_factor):
        """创建一个更兼容ONNX常量折叠的比例上采样层"""
        class ScaledUpsample(nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor
            
            def forward(self, x):
                # 直接使用 scale_factor 进行插值
                return F.interpolate(
                    x, 
                    scale_factor=self.factor, 
                    mode='bilinear', 
                    align_corners=None
                )
        
        return ScaledUpsample(scale_factor)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # 预测自适应卷积核(包括深度卷积核、点卷积核和偏置）
        dw_kernels, pw_kernels, biases = self.kernel_predictor(w)
        # 调用 AdaConv2D 执行自适应卷积
        x = self.ada_conv(x, dw_kernels, pw_kernels, biases)
        # 执行后续普通卷积和激活
        x = self.decoder_layers(x)
        # 执行上采样（已从decoder_layers中分离）
        x = self.upsample(x)
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 style_dim: int, 
                 style_kernel: int, 
                 groups: int, 
                 fixed_batch_size: int, 
                 input_hw: tuple[int, int] = None,  # 修改为可选参数
                 use_fixed_size: bool = False  # 添加标志控制是否使用固定尺寸
                ) -> None:
        super().__init__()
        self.style_dim = style_dim
        self.style_kernel = style_kernel
        self.use_fixed_size = use_fixed_size
        
        # 设置每层的通道数
        self.input_channels = [512, 256, 128, 64]
        self.output_channels = [256, 128, 64, 3]
        self.n_convs = [1, 2, 2, 4]
        
        # 设置分组卷积参数
        if isinstance(groups, list):
            # 使用传入的分组列表
            self.groups_list = groups
        elif isinstance(groups, int):
            # 单个值，应用于所有层
            self.groups_list = [groups] * len(self.input_channels)
        else:
            # 默认按输入通道数设置分组
            self.groups_list = []
            for c in self.input_channels:
                self.groups_list.extend([c // 1, c // 2, c // 4, c // 8][:1])  # 每个通道只取一个分组值
            
            # 确保每个分组至少为1
            for i, g in enumerate(self.groups_list):
                if g < 1:
                    self.groups_list[i] = 1

        decoder_blocks = []
        current_hw = input_hw  # 初始尺寸
        
        for i, Cin in enumerate(self.input_channels):
            Cout = self.output_channels[i]
            Nc = self.n_convs[i]
            final_block = (i == len(self.input_channels) - 1)
            
            # 计算输出尺寸（如果使用固定尺寸）
            if self.use_fixed_size and current_hw is not None:
                output_hw = (current_hw[0] * 2, current_hw[1] * 2) if not final_block else current_hw
            else:
                output_hw = None
                
            decoder_blocks.append(
                DecoderBlock(
                    style_dim=style_dim,
                    style_kernel=style_kernel,
                    in_channels=Cin,
                    out_channels=Cout,
                    groups=self.groups_list[i],  # 使用对应层的分组值
                    fixed_batch_size=fixed_batch_size,
                    input_hw=current_hw if self.use_fixed_size else None,
                    output_hw=output_hw if self.use_fixed_size else None,
                    convs=Nc,
                    final_block=final_block,
                )
            )
            
            if self.use_fixed_size:
                current_hw = output_hw  # 更新当前尺寸
                
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_blocks:
            x = layer(x, w)
        return x