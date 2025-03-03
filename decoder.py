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
        convs: int,
        final_block: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_predictor = KernelPredictor(
            style_dim=style_dim,
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
            style_kernel=style_kernel,
        )

        self.ada_conv = AdaConv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=groups,
        )

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
                    padding_mode="reflect",
                )
            )
            decoder_layers.append(
                nn.ReLU() if not last_layer or not final_block else nn.Sigmoid()
            )

        if not final_block:
            decoder_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))

        self.decoder_layers = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # predict kernels
        dw_kernels, pw_kernels, biases = self.kernel_predictor(w)

        # ada conv
        x = self.ada_conv(x, dw_kernels, pw_kernels, biases)

        # spatial conv + act(optional) + upsample(optional)
        x = self.decoder_layers(x)

        return x


class Decoder(nn.Module):
    def __init__(self, style_dim: int, style_kernel: int) -> None:
        super().__init__()

        self.style_dim = style_dim
        self.style_kernel = style_kernel

        self.input_channels = [512, 256, 128, 64]
        self.output_channels = [256, 128, 64, 3]
        self.n_convs = [1, 2, 2, 4]
        self.groups = [512, 256 // 2, 128 // 4, 64 // 8]

        decoder_blocks = []
        for i, (Cin, Cout, Ng, Nc) in enumerate(
            zip(self.input_channels, self.output_channels, self.groups, self.n_convs)
        ):
            final_block = i == (len(self.groups) - 1)
            decoder_blocks.append(
                DecoderBlock(
                    style_dim=self.style_dim,
                    style_kernel=self.style_kernel,
                    in_channels=Cin,
                    out_channels=Cout,
                    groups=Ng,
                    convs=Nc,
                    final_block=final_block,
                )
            )
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # decoder blocks
        for layer in self.decoder_blocks:
            # (layer inputs, style descriptor)
            x = layer(x, w)
        return x
