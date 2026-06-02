from __future__ import annotations

import torch
import torch.nn as nn


class DSConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyOMRNet(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            DSConv(16, 24),
            nn.MaxPool2d(2),
            DSConv(24, 32),
            nn.MaxPool2d(2),
            DSConv(32, 48),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class SmallOMRNet(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.20) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            DSConv(24, 32),
            DSConv(32, 48, stride=2),
            DSConv(48, 64),
            DSConv(64, 96, stride=2),
            DSConv(96, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _build_mobilenetv2(
    in_channels: int = 1,
    dropout: float = 0.20,
    width_mult: float = 0.5,
    pretrained: bool = False,
) -> nn.Module:
    try:
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    except Exception as exc:  # pragma: no cover
        raise ImportError("need torchvision to use mobilenetv2 model。") from exc

    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = mobilenet_v2(weights=weights, width_mult=width_mult)

    old_conv = model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    if pretrained and old_conv.weight.shape[1] == 3 and in_channels == 1:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_conv

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, 1),
    )
    return model


def build_model(cfg: dict) -> nn.Module:
    model_name = cfg.get("name", "small_cnn").lower()
    in_channels = int(cfg.get("in_channels", 1))
    dropout = float(cfg.get("dropout", 0.2))
    pretrained = bool(cfg.get("pretrained", False))
    if model_name == "tiny_cnn":
        return TinyOMRNet(in_channels=in_channels, dropout=dropout)
    if model_name == "small_cnn":
        return SmallOMRNet(in_channels=in_channels, dropout=dropout)
    if model_name == "mobilenetv2":
        return _build_mobilenetv2(
            in_channels=in_channels,
            dropout=dropout,
            width_mult=float(cfg.get("mobilenet_width_mult", 0.5)),
            pretrained=pretrained,
        )
    raise ValueError(f"unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "DSConv",
    "TinyOMRNet",
    "SmallOMRNet",
    "build_model",
    "count_parameters",
]
