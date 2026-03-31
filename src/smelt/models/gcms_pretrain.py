"""research-extension gc-ms pretraining model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .inception import ExactResearchInceptionClassifier, build_activation


class ResearchGcmsPretrainModel(nn.Module):
    """align diff-only sensor windows with explicit gc-ms anchors."""

    def __init__(
        self,
        *,
        sensor_backbone: ExactResearchInceptionClassifier,
        gcms_feature_count: int,
        projection_dim: int = 128,
        gcms_hidden_dim: int = 128,
        activation_name: str = "gelu",
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if projection_dim <= 0:
            raise ValueError(f"projection_dim must be positive, found {projection_dim}")
        if gcms_hidden_dim <= 0:
            raise ValueError(f"gcms_hidden_dim must be positive, found {gcms_hidden_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, found {temperature}")
        self.sensor_backbone = sensor_backbone
        self.sensor_projection = nn.Sequential(
            nn.Linear(sensor_backbone.feature_dim, projection_dim),
            build_activation(activation_name),
        )
        self.gcms_encoder = nn.Sequential(
            nn.Linear(gcms_feature_count, gcms_hidden_dim),
            build_activation(activation_name),
            nn.Linear(gcms_hidden_dim, projection_dim),
        )
        self.temperature = temperature
        self.register_buffer("anchor_features", torch.empty(0, gcms_feature_count))

    def set_anchor_features(self, anchor_features: torch.Tensor) -> None:
        if anchor_features.ndim != 2:
            raise ValueError(
                f"anchor_features must be 2d (anchors, features), found {anchor_features.shape}"
            )
        self.anchor_features = anchor_features.detach().clone().to(dtype=torch.float32)

    def encode_sensor(self, x: torch.Tensor) -> torch.Tensor:
        features = self.sensor_backbone.forward_features(x)
        return F.normalize(self.sensor_projection(features), dim=1)

    def encode_anchors(self) -> torch.Tensor:
        if self.anchor_features.numel() == 0:
            raise ValueError("anchor_features must be set before pretraining forward passes")
        return F.normalize(self.gcms_encoder(self.anchor_features), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sensor_embeddings = self.encode_sensor(x)
        anchor_embeddings = self.encode_anchors()
        return sensor_embeddings @ anchor_embeddings.transpose(0, 1) / self.temperature
