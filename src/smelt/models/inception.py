"""research-extension inception-style 1d cnn classifier."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ChannelLayerNorm(nn.Module):
    """apply layer norm over channels for channel-first conv outputs."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class GroupOrLayerNorm(nn.Module):
    """prefer groupnorm when channel groups divide cleanly, else use layernorm."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        if num_channels % 8 == 0:
            self.norm: nn.Module = nn.GroupNorm(8, num_channels)
        elif num_channels % 4 == 0:
            self.norm = nn.GroupNorm(4, num_channels)
        else:
            self.norm = ChannelLayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class InceptionBranch(nn.Module):
    """single temporal branch inside the inception-style block."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            GroupOrLayerNorm(out_channels),
            build_activation(activation_name),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class InceptionPoolBranch(nn.Module):
    """pooling branch used in the inception-style block."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        activation_name: str,
    ) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            GroupOrLayerNorm(out_channels),
            build_activation(activation_name),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class InceptionTimeBlock(nn.Module):
    """multiscale residual block for time-series classification."""

    def __init__(
        self,
        *,
        in_channels: int,
        branch_channels: int,
        bottleneck_channels: int,
        activation_name: str,
        use_residual: bool,
    ) -> None:
        super().__init__()
        if bottleneck_channels > 0:
            self.bottleneck: nn.Module = nn.Sequential(
                nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1),
                GroupOrLayerNorm(bottleneck_channels),
                build_activation(activation_name),
            )
            branch_input_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            branch_input_channels = in_channels
        self.branches = nn.ModuleList(
            [
                InceptionBranch(
                    in_channels=branch_input_channels,
                    out_channels=branch_channels,
                    kernel_size=3,
                    activation_name=activation_name,
                ),
                InceptionBranch(
                    in_channels=branch_input_channels,
                    out_channels=branch_channels,
                    kernel_size=5,
                    activation_name=activation_name,
                ),
                InceptionBranch(
                    in_channels=branch_input_channels,
                    out_channels=branch_channels,
                    kernel_size=9,
                    activation_name=activation_name,
                ),
                InceptionPoolBranch(
                    in_channels=in_channels,
                    out_channels=branch_channels,
                    activation_name=activation_name,
                ),
            ]
        )
        merged_channels = branch_channels * 4
        self.post = nn.Sequential(
            GroupOrLayerNorm(merged_channels),
            build_activation(activation_name),
        )
        self.use_residual = use_residual
        if use_residual:
            self.residual = nn.Conv1d(in_channels, merged_channels, kernel_size=1)
        else:
            self.residual = None
        self.activation = build_activation(activation_name)
        self.out_channels = merged_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck_x = self.bottleneck(x)
        outputs = [
            self.branches[0](bottleneck_x),
            self.branches[1](bottleneck_x),
            self.branches[2](bottleneck_x),
            self.branches[3](x),
        ]
        merged = self.post(torch.cat(outputs, dim=1))
        if not self.use_residual:
            return merged
        return self.activation(merged + self.residual(x))


class ExactResearchInceptionClassifier(nn.Module):
    """modest inception-style classifier for fused raw+diff windows."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        stem_channels: int = 64,
        branch_channels: int = 32,
        bottleneck_channels: int = 32,
        num_blocks: int = 6,
        residual_interval: int = 3,
        activation_name: str = "gelu",
        dropout: float = 0.1,
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, found {input_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, found {num_classes}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, found {num_blocks}")
        if residual_interval <= 0:
            raise ValueError(f"residual_interval must be positive, found {residual_interval}")

        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, stem_channels, kernel_size=1),
            GroupOrLayerNorm(stem_channels),
            build_activation(activation_name),
        )
        blocks: list[nn.Module] = []
        current_channels = stem_channels
        for block_index in range(num_blocks):
            use_residual = (block_index + 1) % residual_interval == 0
            block = InceptionTimeBlock(
                in_channels=current_channels,
                branch_channels=branch_channels,
                bottleneck_channels=bottleneck_channels,
                activation_name=activation_name,
                use_residual=use_residual,
            )
            blocks.append(block)
            current_channels = block.out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(current_channels, head_hidden_dim),
            build_activation(activation_name),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )
        self.output_dim = num_classes

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"inception classifier expects (batch, time, features), found {x.shape}"
            )
        features = x.transpose(1, 2)
        features = self.blocks(self.stem(features))
        return self.pool(features).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name!r}")


@dataclass(slots=True)
class InceptionModelSummary:
    input_dim: int
    stem_channels: int
    branch_channels: int
    bottleneck_channels: int
    num_blocks: int
    residual_interval: int
    activation_name: str
    dropout: float
    head_hidden_dim: int
