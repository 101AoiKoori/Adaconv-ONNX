import torch
from blocks import GlobalStyleEncoder
from decoder import Decoder
from encoder import Encoder
from torch import nn

from utils import init_weights


class StyleTransfer(nn.Module):
    def __init__(
        self, image_shape: tuple[int], style_dim: int, style_kernel: int
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.style_dim = style_dim
        self.style_kernel = style_kernel

        self.encoder = Encoder()
        self.encoder.freeze()
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
        )
        self.decoder = Decoder(style_dim=self.style_dim, style_kernel=self.style_kernel)

        self.apply(init_weights)

    def forward(
        self, content: torch.Tensor, style: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor]:
        # encoder
        content_feats = self.encoder(content)
        style_feats = self.encoder(style)
        # global style encoder
        w = self.global_style_encoder(style_feats[-1])
        # decoder
        x = self.decoder(content_feats[-1], w)

        if return_features:
            x_feats = self.encoder(x)
            return (x, content_feats, style_feats, x_feats)
        else:
            return x
