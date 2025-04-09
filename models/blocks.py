import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, Dict, Union, Any

class GlobalStyleEncoder(nn.Module):
    """
    全局风格编码器 - 从风格特征生成风格描述符
    
    支持动态/静态模式切换，适配ONNX导出
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
        # 获取批次大小 - 修复动态批次大小问题
        batch_size = self.fixed_batch_size if self.export_mode and self.fixed_batch_size is not None else x.size(0)
        
        # 特征提取
        x = self.style_encoder(x)
        x = torch.flatten(x, start_dim=1)
        w = self.fc(x)
        
        # 重塑为描述符
        w = w.reshape(
            batch_size,
            self.style_descriptor_shape[0],
            self.style_descriptor_shape[1],
            self.style_descriptor_shape[2]
        )
        return w


class KernelPredictor(nn.Module):
    """
    卷积核预测器 - 从风格描述符生成动态卷积核
    
    支持动态/静态模式切换，适配ONNX导出
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

        # 深度卷积核预测器
        self.depthwise_conv_kernel_predictor = nn.Conv2d(
            in_channels=self.style_dim,
            out_channels=self.out_channels * (self.in_channels // self.groups),
            kernel_size=3,
            padding=1,
            padding_mode="reflect"
        )

        # 点卷积核预测器
        self.pointwise_conv_kernel_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=self.out_channels * (self.out_channels // self.groups),
                kernel_size=1,
            ),
        )

        # 偏置预测器
        self.bias_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_channels=self.style_dim,
                out_channels=out_channels,
                kernel_size=1,
            ),
        )

    def forward(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 获取批次大小 - 修复动态批次大小问题
        B = self.fixed_batch_size if self.export_mode and self.fixed_batch_size is not None else w.size(0)
        
        # 预测深度卷积核: 形状 [B, C_out, C_in//groups, K, K]
        dw_kernel = self.depthwise_conv_kernel_predictor(w)
        dw_kernel = dw_kernel.reshape(
            B,
            self.out_channels,
            self.in_channels // self.groups,
            self.style_kernel,
            self.style_kernel,
        )
        
        # 预测点卷积核: 形状 [B, C_out, C_out//groups, 1, 1]
        pw_kernel = self.pointwise_conv_kernel_predictor(w)
        pw_kernel = pw_kernel.reshape(
            B,
            self.out_channels,
            self.out_channels // self.groups,
            1,
            1,
        )
        
        # 预测偏置: 形状 [B, C_out], 展平为 [B * C_out]
        bias = self.bias_predictor(w)
        bias = bias.reshape(B, self.out_channels)
        if self.export_mode and self.fixed_batch_size is not None:
            # 导出模式下展平偏置，使其形状为 [B*C_out]
            bias = bias.reshape(-1)
        
        return (dw_kernel, pw_kernel, bias)


class AdaConv2D(nn.Module):
    """
    自适应卷积层 - 应用动态生成的卷积核到输入特征
    
    支持动态/静态模式切换，适配ONNX导出
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
        """对输入特征进行归一化"""
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
        # 判断是否使用导出模式
        if self.export_mode:
            # 导出模式: 使用向量化的分组卷积
            if self.fixed_batch_size is None:
                # 动态批次大小模式
                return self._forward_dynamic_batch(x, dw_kernels, pw_kernels, biases)
            else:
                # 静态批次大小模式
                return self._forward_static(x, dw_kernels, pw_kernels, biases)
        else:
            # 动态模式: 根据输入决定使用哪种方法
            B = x.size(0)
            if B > 1:
                # 批次大小>1时使用向量化实现
                return self._forward_batched(x, dw_kernels, pw_kernels, biases)
            else:
                # 批次大小=1时使用简单实现(避免不必要的维度变换)
                return self._forward_simple(x, dw_kernels, pw_kernels, biases)
    
    def _forward_dynamic_batch(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """动态批次大小模式下的前向传播，特别为ONNX导出优化"""
        # 获取当前批次大小
        B = x.size(0)
        
        # 归一化输入
        x = self._normalize(x)
        
        # 处理每个样本，这样可以避免动态批次大小下的reshape问题
        outputs = []
        for i in range(B):
            # 提取单个样本的输入和卷积核
            x_i = x[i:i+1]
            dw_kernel_i = dw_kernels[i:i+1]
            pw_kernel_i = pw_kernels[i:i+1]
            bias_i = biases[i:i+1] if biases.dim() > 1 else biases
            
            # 对每个样本应用简化版的前向传播
            out_i = self._forward_simple(x_i, dw_kernel_i, pw_kernel_i, bias_i)
            outputs.append(out_i)
        
        # 连接所有输出
        return torch.cat(outputs, dim=0)
    
    def _forward_static(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """导出模式下的前向传播，使用固定批次大小和空间维度"""
        # 确保在静态模式下批次大小是固定的
        B = self.fixed_batch_size if self.fixed_batch_size is not None else x.size(0)
            
        # 归一化输入
        x = self._normalize(x)
        
        # 获取卷积核尺寸和计算填充
        K = dw_kernels.shape[-1]
        padding = (K - 1) // 2
        
        # 获取输入尺寸(使用固定尺寸或从输入获取)
        if self.fixed_hw is not None:
            H_in, W_in = self.fixed_hw
        else:
            H_in, W_in = x.shape[2], x.shape[3]
        
        # 计算输出尺寸 - 始终与输入相同
        H_out, W_out = H_in, W_in
        
        # 准备输入 - 使用固定形状
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        
        # 使用固定的批次大小
        # 这里使用更为明确的重塑操作，避免动态维度
        x_merged = x_padded.reshape(1, B * self.in_channels, x_padded.shape[2], x_padded.shape[3])
        
        # 为卷积核提供具体形状，避免动态维度
        dw_kernels_merged = dw_kernels.reshape(B * self.out_channels, self.in_channels // self.groups, K, K)
        pw_kernels_merged = pw_kernels.reshape(B * self.out_channels, self.out_channels // self.groups, 1, 1)
        
        # 确保偏置是一维的
        if biases.dim() > 1:
            biases = biases.reshape(-1)
            
        # 计算总分组数
        conv_groups = B * self.groups
        
        # 深度卷积
        depthwise_out = F.conv2d(x_merged, dw_kernels_merged, groups=conv_groups, padding=0)
        
        # 使用固定形状
        depthwise_out = depthwise_out.reshape(B, self.out_channels, H_out, W_out)
        
        # 点卷积前的重塑
        depthwise_merged = depthwise_out.reshape(1, B * self.out_channels, H_out, W_out)
        
        # 点卷积
        output = F.conv2d(depthwise_merged, pw_kernels_merged, bias=biases, groups=conv_groups)
        
        # 最终输出形状
        output = output.reshape(B, self.out_channels, H_out, W_out)
        
        return output
    
    def _forward_batched(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """批处理模式下的前向传播，支持可变批次大小"""
        # 获取批次大小
        B = x.size(0)
        
        # 归一化输入
        x = self._normalize(x)
        
        # 获取卷积核尺寸和计算填充
        K = dw_kernels.shape[-1]
        padding = (K - 1) // 2
        
        # 准备输入和卷积核
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        x_merged = x_padded.reshape(1, B * self.in_channels, x_padded.shape[2], x_padded.shape[3])
        dw_kernels_merged = dw_kernels.reshape(B * self.out_channels, self.in_channels // self.groups, K, K)
        pw_kernels_merged = pw_kernels.reshape(B * self.out_channels, self.out_channels // self.groups, 1, 1)
        
        # 计算总分组数
        conv_groups = B * self.groups
        
        # 获取输出尺寸
        H_out, W_out = x.shape[2], x.shape[3]
        
        # 深度卷积
        depthwise_out = F.conv2d(x_merged, dw_kernels_merged, groups=conv_groups, padding=0)
        depthwise_out = depthwise_out.reshape(B, self.out_channels, H_out, W_out)
        
        # 点卷积
        depthwise_merged = depthwise_out.reshape(1, B * self.out_channels, H_out, W_out)
        # 将偏置展平为一维
        if biases.dim() > 1:
            biases = biases.reshape(-1)
        output = F.conv2d(depthwise_merged, pw_kernels_merged, bias=biases, groups=conv_groups)
        output = output.reshape(B, self.out_channels, H_out, W_out)
        
        return output
    
    def _forward_simple(self, x: torch.Tensor, dw_kernels: torch.Tensor, pw_kernels: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        """单样本情况下的简单前向传播，避免不必要的维度变换"""
        # 归一化输入
        x = self._normalize(x)
        
        # 获取卷积核尺寸和计算填充
        K = dw_kernels.shape[-1]
        padding = (K - 1) // 2
        
        # 填充输入
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        
        # 提取单个样本的卷积核和偏置
        dw_kernel = dw_kernels[0]  # [C_out, C_in//groups, K, K]
        pw_kernel = pw_kernels[0]  # [C_out, C_out//groups, 1, 1]
        bias = biases[0] if biases.dim() > 1 else biases  # [C_out]
        
        # 深度卷积
        depthwise_out = F.conv2d(x_padded, dw_kernel, groups=self.groups, padding=0)
        
        # 点卷积
        output = F.conv2d(depthwise_out, pw_kernel, bias=bias, groups=self.groups)
        
        return output