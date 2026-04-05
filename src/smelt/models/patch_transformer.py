"""patch-based temporal transformer for moonshot ensemble diversity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .transformer import SinusoidalPositionalEncoding


@dataclass(slots=True)
class PatchTransformerArchitectureSummary:
    model_family: str
    input_feature_count: int
    patch_size: int
    patch_stride: int
    model_dim: int
    num_heads: int
    num_layers: int
    mlp_ratio: float
    dropout: float
    parameter_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "block_type": "patch_transformer_encoder",
            "input_feature_count": self.input_feature_count,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "model_dim": self.model_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "parameter_count": self.parameter_count,
        }


class TemporalPatchTransformerClassifier(nn.Module):
    """patchified temporal transformer for short sensor windows."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        patch_size: int = 8,
        patch_stride: int = 4,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or patch_stride <= 0:
            raise ValueError("patch_size and patch_stride must be positive")
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout_probability = float(dropout)

        self.patch_embed = nn.Conv1d(
            input_dim,
            self.model_dim,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            bias=False,
        )
        self.patch_norm = nn.LayerNorm(self.model_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.model_dim * self.mlp_ratio),
            dropout=self.dropout_probability,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.dropout = nn.Dropout(self.dropout_probability)
        self.head = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.model_dim // 2, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"patch transformer expects (batch, time, channels), found {tuple(x.shape)}"
            )
        tokens = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        tokens = self.patch_norm(tokens)
        encoded = self.encoder(self.positional_encoding(tokens))
        return encoded.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.forward_features(x)))

    def architecture_summary(
        self,
        *,
        input_feature_count: int,
    ) -> PatchTransformerArchitectureSummary:
        parameter_count = int(
            sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        )
        return PatchTransformerArchitectureSummary(
            model_family="patch_transformer",
            input_feature_count=input_feature_count,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout_probability,
            parameter_count=parameter_count,
        )
