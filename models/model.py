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
        AdaConv风格迁移模型
        
        Args:
            image_shape: 输入图像尺寸 (高度, 宽度)
            style_dim: 风格描述符通道数
            style_kernel: 自适应卷积核尺寸
            groups: 分组卷积的分组数，可以是单个整数或列表(每层不同分组)
            export_config: 导出配置，包含以下可选字段:
                - export_mode: 是否启用导出模式 (默认: False)
                - fixed_batch_size: 导出时使用的固定批次大小 (默认: None)
                - use_fixed_size: 是否使用固定空间尺寸 (默认: False)
        """
        super().__init__()
        self.image_shape = image_shape
        self.style_dim = style_dim
        self.style_kernel = style_kernel
        self.groups = groups
        
        # 处理导出配置
        self.export_config = export_config or {}
        self.export_mode = self.export_config.get('export_mode', False)
        self.fixed_batch_size = self.export_config.get('fixed_batch_size', None)
        self.use_fixed_size = self.export_config.get('use_fixed_size', False)

        # 初始化编码器(VGG)
        self.encoder = Encoder()
        self.encoder.freeze()
        encoder_scale = self.encoder.scale_factor
        
        # 计算编码器输出尺寸
        encoder_hw = None
        if self.use_fixed_size:
            encoder_hw = (
                self.image_shape[0] // encoder_scale,
                self.image_shape[1] // encoder_scale
            )

        # 初始化全局风格编码器
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
        
        # 初始化解码器
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

        # 初始化权重
        self.apply(init_weights)
        
        # 归一化设置
        self.freeze_normalization = False

    def set_export_mode(self, enabled: bool = True, fixed_batch_size: Optional[int] = None):
        """
        设置模型的导出模式
        
        Args:
            enabled: 是否启用导出模式
            fixed_batch_size: 导出时使用的固定批次大小
        """
        self.export_mode = enabled
        if enabled and fixed_batch_size is not None:
            self.fixed_batch_size = fixed_batch_size
        elif not enabled:
            self.fixed_batch_size = None
        
        # 更新所有子模块的导出模式
        for module in self.modules():
            if hasattr(module, 'export_mode'):
                module.export_mode = enabled
            if hasattr(module, 'fixed_batch_size') and enabled and fixed_batch_size is not None:
                module.fixed_batch_size = fixed_batch_size

    def freeze_normalization_layers(self):
        """冻结所有归一化层"""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
                m.eval()  # 冻结统计量计算
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad_(False)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad_(False)
        self.freeze_normalization = True

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 返回风格迁移结果
        
        Args:
            content: 内容图像 [B, 3, H, W]
            style: 风格图像 [B, 3, H, W]
            
        Returns:
            风格化后的内容图像 [B, 3, H, W]
        """
        # 评估模式下冻结归一化层
        if not self.training and not self.freeze_normalization:
            self.freeze_normalization_layers()
            
        # 提取内容和风格特征
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        
        # 生成风格描述符
        w = self.global_style_encoder(style_feats[-1])
        
        # 解码生成风格化图像
        x = self.decoder(content_feats[-1], w)
        
        return x

    def forward_with_features(self, content: torch.Tensor, style: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        前向传播 - 返回风格迁移结果和中间特征(用于训练和损失计算)
        
        Args:
            content: 内容图像 [B, 3, H, W]
            style: 风格图像 [B, 3, H, W]
            
        Returns:
            (x, content_feats, style_feats, x_feats):
                x: 风格化后的内容图像 [B, 3, H, W]
                content_feats: 内容特征列表
                style_feats: 风格特征列表
                x_feats: 输出特征列表
        """
        # 提取内容和风格特征
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        
        # 生成风格描述符
        w = self.global_style_encoder(style_feats[-1])
        
        # 解码生成风格化图像
        x = self.decoder(content_feats[-1], w)
        
        # 提取输出特征(用于计算损失)
        x_feats = self.encoder(x)
        
        return x, content_feats, style_feats, x_feats