"""deep temporal resnet backbones for moonshot runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


def _resolve_group_count(num_channels: int, requested_groups: int) -> int:
    groups = max(1, min(num_channels, requested_groups))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


def _build_normalization(
    *,
    normalization: str,
    num_channels: int,
    groupnorm_groups: int,
) -> nn.Module:
    if normalization == "groupnorm":
        groups = _resolve_group_count(num_channels, groupnorm_groups)
        return nn.GroupNorm(groups, num_channels)
    if normalization == "batchnorm":
        return nn.BatchNorm1d(num_channels)
    raise ValueError(f"unsupported normalization {normalization!r}")


class SqueezeExcitation1D(nn.Module):
    """channel attention for short temporal sequences."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        reduced_channels = max(8, channels // max(reduction, 1))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, reduced_channels, kernel_size=1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv1d(reduced_channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.act(scale)
        scale = self.fc2(scale)
        return x * self.gate(scale)


class StochasticDepth1D(nn.Module):
    """sample-wise residual dropout."""

    def __init__(self, drop_probability: float) -> None:
        super().__init__()
        self.drop_probability = float(drop_probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_probability <= 0:
            return x
        keep_probability = 1.0 - self.drop_probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_probability)
        return x * random_tensor / keep_probability


class ResidualTemporalBlock1D(nn.Module):
    """basic residual block with optional se and stochastic depth."""

    expansion = 1

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        normalization: str,
        groupnorm_groups: int,
        se_reduction: int,
        stochastic_depth_probability: float,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = _build_normalization(
            normalization=normalization,
            num_channels=out_channels,
            groupnorm_groups=groupnorm_groups,
        )
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm2 = _build_normalization(
            normalization=normalization,
            num_channels=out_channels,
            groupnorm_groups=groupnorm_groups,
        )
        self.se = SqueezeExcitation1D(out_channels, reduction=se_reduction)
        self.drop_path = StochasticDepth1D(stochastic_depth_probability)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _build_normalization(
                    normalization=normalization,
                    num_channels=out_channels,
                    groupnorm_groups=groupnorm_groups,
                ),
            )
        else:
            self.shortcut = nn.Identity()
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out = self.drop_path(out)
        return self.out_act(out + residual)


@dataclass(slots=True)
class TemporalResNetArchitectureSummary:
    model_family: str
    block_type: str
    stage_depths: tuple[int, ...]
    stage_widths: tuple[int, ...]
    normalization: str
    parameter_count: int
    input_feature_count: int
    stem_width: int
    kernel_size: int
    se_reduction: int
    stochastic_depth_probability: float
    head_dropout: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "block_type": self.block_type,
            "stage_depths": list(self.stage_depths),
            "stage_widths": list(self.stage_widths),
            "normalization": self.normalization,
            "parameter_count": self.parameter_count,
            "input_feature_count": self.input_feature_count,
            "stem_width": self.stem_width,
            "kernel_size": self.kernel_size,
            "se_reduction": self.se_reduction,
            "stochastic_depth_probability": self.stochastic_depth_probability,
            "head_dropout": self.head_dropout,
        }


class DeepTemporalResNet1D(nn.Module):
    """deep 1d residual cnn for short sensor sequences."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stage_depths: Sequence[int],
        stage_widths: Sequence[int],
        stem_width: int = 64,
        kernel_size: int = 5,
        normalization: str = "groupnorm",
        groupnorm_groups: int = 8,
        se_reduction: int = 16,
        head_dropout: float = 0.2,
        stochastic_depth_probability: float = 0.05,
    ) -> None:
        super().__init__()
        if len(stage_depths) != 4 or len(stage_widths) != 4:
            raise ValueError("deep temporal resnet expects exactly four stages")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.stage_depths = tuple(int(value) for value in stage_depths)
        self.stage_widths = tuple(int(value) for value in stage_widths)
        self.stem_width = int(stem_width)
        self.kernel_size = int(kernel_size)
        self.normalization = normalization
        self.groupnorm_groups = int(groupnorm_groups)
        self.se_reduction = int(se_reduction)
        self.head_dropout = float(head_dropout)
        self.stochastic_depth_probability = float(stochastic_depth_probability)
        stem_padding = 7 // 2
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                self.stem_width,
                kernel_size=7,
                padding=stem_padding,
                bias=False,
            ),
            _build_normalization(
                normalization=normalization,
                num_channels=self.stem_width,
                groupnorm_groups=self.groupnorm_groups,
            ),
            nn.GELU(),
        )
        total_blocks = sum(self.stage_depths)
        block_index = 0
        current_channels = self.stem_width
        stages: list[nn.Module] = []
        for stage_index, (depth, width) in enumerate(
            zip(self.stage_depths, self.stage_widths, strict=True)
        ):
            blocks: list[nn.Module] = []
            for block_in_stage in range(depth):
                stride = 2 if stage_index > 0 and block_in_stage == 0 else 1
                drop_probability = 0.0
                if total_blocks > 1 and self.stochastic_depth_probability > 0:
                    drop_probability = self.stochastic_depth_probability * (
                        block_index / (total_blocks - 1)
                    )
                blocks.append(
                    ResidualTemporalBlock1D(
                        in_channels=current_channels,
                        out_channels=width,
                        kernel_size=self.kernel_size,
                        stride=stride,
                        normalization=self.normalization,
                        groupnorm_groups=self.groupnorm_groups,
                        se_reduction=self.se_reduction,
                        stochastic_depth_probability=drop_probability,
                    )
                )
                current_channels = width
                block_index += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.head_dropout) if self.head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(current_channels, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"deep temporal resnet expects (batch, time, channels), found {tuple(x.shape)}"
            )
        features = x.transpose(1, 2)
        features = self.stem(features)
        for stage in self.stages:
            features = stage(features)
        features = self.pool(features).squeeze(-1)
        return self.dropout(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def architecture_summary(
        self,
        *,
        input_feature_count: int,
    ) -> TemporalResNetArchitectureSummary:
        parameter_count = int(
            sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        )
        return TemporalResNetArchitectureSummary(
            model_family="deep_temporal_resnet",
            block_type="se_basic_residual_block",
            stage_depths=self.stage_depths,
            stage_widths=self.stage_widths,
            normalization=self.normalization,
            parameter_count=parameter_count,
            input_feature_count=int(input_feature_count),
            stem_width=self.stem_width,
            kernel_size=self.kernel_size,
            se_reduction=self.se_reduction,
            stochastic_depth_probability=self.stochastic_depth_probability,
            head_dropout=self.head_dropout,
        )
