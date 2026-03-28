"""exact-upstream transformer baseline model."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """sinusoidal positional encoding copied from the upstream baseline shape."""

    def __init__(self, d_model: int, max_len: int = 10_000) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)
        return x + self.pe[:sequence_length].unsqueeze(0)


class ExactUpstreamTransformerClassifier(nn.Module):
    """transformer classifier matching the upstream baseline structure."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.positional_encoding = SinusoidalPositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_proj(x)
        encoded = self.encoder(self.positional_encoding(projected))
        return encoded.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.classifier(self.dropout(features))
