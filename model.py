import torch
from blocks import GlobalStyleEncoder
from decoder import Decoder
from encoder import Encoder
from torch import nn

from utils import init_weights


class StyleTransfer(nn.Module):
    def __init__(
        self, 
        image_shape: tuple[int], 
        style_dim: int, 
        style_kernel: int, 
        groups=None,  
        fixed_batch_size=None,
        use_fixed_size=False  # 添加控制是否使用固定尺寸的标志
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.style_dim = style_dim
        self.style_kernel = style_kernel
        self.groups = groups
        self.fixed_batch_size = fixed_batch_size
        self.use_fixed_size = use_fixed_size

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

        self.global_style_encoder = GlobalStyleEncoder(
            style_feat_shape=(
                self.style_dim,
                self.image_shape[0] // self.encoder.scale_factor,
                self.image_shape[1] // self.encoder.scale_factor,
            ),
            style_descriptor_shape=(
                self.style_dim,
                self.style_kernel,
                self.style_kernel,
            ),
            fixed_batch_size=fixed_batch_size
        )
        
        self.decoder = Decoder(
            style_dim=self.style_dim,
            style_kernel=self.style_kernel,
            groups=self.groups,
            fixed_batch_size=self.fixed_batch_size,
            input_hw=encoder_hw,
            use_fixed_size=self.use_fixed_size
        )

        self.apply(init_weights)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        仅返回最终生成的 x，适合 ONNX 导出时使用静态计算图。
        """
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        w = self.global_style_encoder(style_feats[-1])
        x = self.decoder(content_feats[-1], w)
        return x

    def forward_with_features(self, content: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 x 以及中间特征(content_feats, style_feats, x_feats),供训练时使用。
        """
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        w = self.global_style_encoder(style_feats[-1])
        x = self.decoder(content_feats[-1], w)
        x_feats = self.encoder(x)
        return x, content_feats, style_feats, x_feats