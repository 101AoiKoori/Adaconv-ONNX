import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union, Any

from models.blocks import AdaConv2D, KernelPredictor

class DecoderBlock(nn.Module):
    """
    解码器块 - 自适应卷积和常规卷积的组合块
    
    支持动态/静态模式切换，适配ONNX导出
    """
    def __init__(
        self,
        style_dim: int,
        style_kernel: int,
        in_channels: int,
        out_channels: int,
        groups: int,
        export_config: Dict[str, Any] = None,
        convs: int = 1,
        final_block: bool = False,
        scale_factor: int = 2,
    ) -> None:
        super().__init__()
        
        # 处理导出配置
        self.export_config = export_config or {}
        self.export_mode = self.export_config.get('export_mode', False)
        self.fixed_batch_size = self.export_config.get('fixed_batch_size', None) if self.export_mode else None
        self.use_fixed_size = self.export_config.get('use_fixed_size', False)
        self.input_hw = self.export_config.get('input_hw', None)
        self.output_hw = self.export_config.get('output_hw', None)
        self.scale_factor = scale_factor
        
        # 内部KernelPredictor和AdaConv2D共享导出设置
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

        # 构建标准卷积层序列
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
            # 添加激活函数
            if not last_layer or not final_block:
                decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(nn.Sigmoid())  # 最终输出层使用Sigmoid
        
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
        # 上采样层 - 根据模式选择实现方式
        if not final_block:
            if self.use_fixed_size and self.output_hw is not None:
                # 固定尺寸上采样 - 使用预定义输出尺寸
                self.upsample = self._create_fixed_upsample(self.output_hw)
            else:
                # 动态尺寸上采样 - 使用比例因子
                self.upsample = self._create_scaled_upsample(scale_factor)
        else:
            # 最终块不需要上采样
            self.upsample = nn.Identity()
    
    def _create_fixed_upsample(self, output_hw: Tuple[int, int]) -> nn.Module:
        """创建固定尺寸上采样层 - 适配ONNX导出"""
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
        """创建比例因子上采样层 - 适配动态尺寸"""
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
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            w: 风格描述符 [B, style_dim, kernel_h, kernel_w]
            
        Returns:
            处理后的特征 [B, C_out, H_out, W_out]
        """
        # 预测自适应卷积核
        dw_kernels, pw_kernels, biases = self.kernel_predictor(w)
        
        # 应用自适应卷积
        x = self.ada_conv(x, dw_kernels, pw_kernels, biases)
        
        # 应用标准卷积层
        x = self.decoder_layers(x)
        
        # 应用上采样（如果需要）
        x = self.upsample(x)
        
        return x


class Decoder(nn.Module):
    """
    解码器 - 将内容特征和风格描述符转换为风格化图像
    
    支持动态/静态模式切换，适配ONNX导出
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
        
        # 处理导出配置
        self.export_config = export_config or {}
        self.export_mode = self.export_config.get('export_mode', False)
        self.fixed_batch_size = self.export_config.get('fixed_batch_size', None) if self.export_mode else None
        self.use_fixed_size = self.export_config.get('use_fixed_size', False)
        self.input_hw = self.export_config.get('input_hw', None)
        
        # 设置层级结构的参数
        self.input_channels = [512, 256, 128, 64]  # 每层的输入通道数
        self.output_channels = [256, 128, 64, 3]   # 每层的输出通道数
        self.n_convs = [1, 2, 2, 4]                # 每层的卷积数量
        
        # 处理分组参数：单整数或列表
        if isinstance(groups, int):
            self.groups_list = [groups] * len(self.input_channels)
        elif isinstance(groups, list):
            if len(groups) != len(self.input_channels):
                raise ValueError(f"Groups list length ({len(groups)}) must match number of decoder layers ({len(self.input_channels)})")
            self.groups_list = groups
        else:
            # 默认分组策略：基于输入通道动态计算
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
                    
            # 确保每组至少有1个通道
            self.groups_list = [max(1, g) for g in self.groups_list]

        # 构建解码器块
        decoder_blocks = []
        current_hw = self.input_hw
        
        for i, (Cin, Cout, Nc, Group) in enumerate(zip(
            self.input_channels, self.output_channels, self.n_convs, self.groups_list
        )):
            # 确定是否为最终块
            final_block = (i == len(self.input_channels) - 1)
            
            # 计算输出尺寸（如果使用固定尺寸）
            output_hw = None
            if self.use_fixed_size and current_hw is not None:
                if not final_block:
                    # 非最终块：尺寸翻倍
                    output_hw = (current_hw[0] * 2, current_hw[1] * 2)
                else:
                    # 最终块：保持尺寸不变
                    output_hw = current_hw
            
            # 为当前块创建导出配置
            block_export_config = {
                'export_mode': self.export_mode,
                'fixed_batch_size': self.fixed_batch_size,
                'use_fixed_size': self.use_fixed_size,
                'input_hw': current_hw,
                'output_hw': output_hw
            }
            
            # 创建解码器块
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
            
            # 更新当前尺寸（用于下一层）
            if self.use_fixed_size:
                current_hw = output_hw
                
        # 保存解码器块序列
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            w: 风格描述符 [B, style_dim, kernel_h, kernel_w]
            
        Returns:
            风格化图像 [B, 3, H_out, W_out]
        """
        # 依次通过所有解码器块
        for layer in self.decoder_blocks:
            x = layer(x, w)
        return x