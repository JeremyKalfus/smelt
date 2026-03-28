"""exact-upstream 1d cnn baseline."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ExactUpstreamCnnClassifier(nn.Module):
    """lightweight 1d cnn with global average pooling."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        channels: Sequence[int] = (64, 128, 256),
        kernel_size: int = 5,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("cnn channels must be non-empty")
        conv_blocks: list[nn.Module] = []
        current_channels = in_channels
        for next_channels in channels:
            padding = kernel_size // 2
            conv_blocks.append(
                nn.Conv1d(current_channels, next_channels, kernel_size=kernel_size, padding=padding)
            )
            if use_batchnorm:
                conv_blocks.append(nn.BatchNorm1d(next_channels))
            conv_blocks.append(nn.ReLU(inplace=True))
            if dropout > 0:
                conv_blocks.append(nn.Dropout(dropout))
            current_channels = next_channels
        self.conv = nn.Sequential(*conv_blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(current_channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"cnn expects (batch, time, channels), found {tuple(x.shape)}")
        features = x.transpose(1, 2)
        features = self.conv(features)
        return self.pool(features).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
