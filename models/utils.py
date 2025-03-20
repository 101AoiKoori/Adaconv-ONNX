import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        if m.weight.requires_grad:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        if m.weight.requires_grad:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        if m.weight.requires_grad:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None and m.bias.requires_grad:
            nn.init.constant_(m.bias, 0.0)