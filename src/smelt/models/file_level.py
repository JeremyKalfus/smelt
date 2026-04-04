"""file-level models for frozen moonshot embeddings."""

from __future__ import annotations

import torch
from torch import nn


class AttentionDeepSetsClassifier(nn.Module):
    """attention-pooled deepsets classifier for variable-length file batches."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.dropout_probability = float(dropout)
        self.window_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"attention deepsets expects (batch, windows, features), found {tuple(x.shape)}"
            )
        return self.window_mlp(x)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2d, found {tuple(mask.shape)}")
        if mask.shape[:2] != x.shape[:2]:
            raise ValueError(
                "mask shape must match batch/windows dimensions of x, "
                f"found x={tuple(x.shape)} mask={tuple(mask.shape)}"
            )
        encoded = self.encode_windows(x)
        attention_scores = self.attention(encoded).squeeze(-1)
        masked_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=1)
        attention_weights = attention_weights.masked_fill(~mask, 0.0)
        pooled = torch.sum(encoded * attention_weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)
