import torch
from blocks import GlobalStyleEncoder
from decoder import Decoder
from encoder import Encoder
from torch import nn

from utils import init_weights


class StyleTransfer(nn.Module):
    def __init__(
        self, image_shape: tuple[int], style_dim: int, style_kernel: int, groups: int = None, fixed_batch_size: int = None
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.style_dim = style_dim
        self.style_kernel = style_kernel
        self.groups = groups  # 用户可在初始化时指定固定的 groups
        self.fixed_batch_size = fixed_batch_size  # 从 hyperparameter 中获得

        self.encoder = Encoder()
        self.encoder.freeze()
        encoder_scale = self.encoder.scale_factor
        encoder_hw = (
            self.image_shape[0] // encoder_scale,
            self.image_shape[1] // encoder_scale
        )  # 新增：计算Encoder输出尺寸

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
            fixed_batch_size=fixed_batch_size  # 新增
        )
        self.decoder = Decoder(
            style_dim=self.style_dim,
            style_kernel=self.style_kernel,
            groups=self.groups,
            fixed_batch_size=self.fixed_batch_size,  # 从 hyperparameter 中获得
            input_hw=encoder_hw  # 新增：传递Encoder输出尺寸
        )

        self.apply(init_weights)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
       
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        w = self.global_style_encoder(style_feats[-1])
        x = self.decoder(content_feats[-1], w)
        return x

    def forward_with_features(self, content: torch.Tensor, style: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        w = self.global_style_encoder(style_feats[-1])
        x = self.decoder(content_feats[-1], w)
        x_feats = self.encoder(x)
        return x, content_feats, style_feats, x_feats