from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherited
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherited
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherited
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = torch.nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherited
        return self.conv(x)


class GNRUNet(nn.Module):
    """UNet backbone used by the GaussMarker GNR restoration module."""

    def __init__(self, in_channels: int, out_channels: int, nf: int = 128, bilinear: bool = False) -> None:
        super().__init__()
        self.inc = DoubleConv(in_channels, nf)
        self.down1 = Down(nf, nf * 2)
        self.down2 = Down(nf * 2, nf * 4)
        self.down3 = Down(nf * 4, nf * 8)
        factor = 2 if bilinear else 1
        self.up2 = Up(nf * 8, nf * 4 // factor, bilinear)
        self.up3 = Up(nf * 4, nf * 2 // factor, bilinear)
        self.up4 = Up(nf * 2, nf, bilinear)
        self.outc = OutConv(nf, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - inherited
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class GNRRestorer:
    """Wrapper for loading and running the GaussMarker GNR restoration network."""

    def __init__(
        self,
        checkpoint_path: Path,
        in_channels: int,
        out_channels: int,
        nf: int,
        device: torch.device,
        classifier_type: int,
        base_message: Optional[torch.Tensor] = None,
    ) -> None:
        self.device = device
        self.classifier_type = classifier_type
        self.base_message = base_message.to(device) if base_message is not None else None
        self.model = GNRUNet(in_channels, out_channels, nf=nf)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()

    def restore(self, reversed_m: torch.Tensor) -> torch.Tensor:
        """Run the GNR model and return the restored watermark bits (probabilities)."""
        with torch.no_grad():
            inputs = reversed_m.to(self.device, dtype=torch.float32)
            if self.classifier_type == 1:
                if self.base_message is None:
                    raise ValueError("Base watermark message is required when classifier_type=1")
                inputs = torch.cat([self.base_message, inputs], dim=1)
            logits = self.model(inputs)
            probs = torch.sigmoid(logits)
        return probs

    def restore_binary(self, reversed_m: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convenience helper returning binarised restored watermark bits."""
        probs = self.restore(reversed_m)
        return (probs > threshold).float()
