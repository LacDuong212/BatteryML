# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch
import torch.nn as nn

from batteryml.builders import MODELS
from batteryml.models.nn_model import NNModel


class SimpleGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        residual = x
        y = self.norm(x)

        # simple neighbor aggregation by local averaging along node axis
        left = torch.roll(y, shifts=1, dims=1)
        right = torch.roll(y, shifts=-1, dims=1)
        agg = (y + left + right) / 3.0

        agg = self.msg(agg)
        x = residual + agg
        x = x + self.update(x)
        return x


class GCNBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        input_height: int,
        input_width: int,
        dropout: float = 0.1,
        num_layers: int = 1,
    ):
        super().__init__()
        node_dim = in_channels * input_width

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            SimpleGraphBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            feature = feature.unsqueeze(1)

        # feature: [B, C, H, W]
        B, C, H, W = feature.size()

        # Treat H as graph nodes, each node has C*W features
        x = feature.permute(0, 2, 1, 3).contiguous().view(B, H, C * W)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)   # graph pooling
        return x


@MODELS.register()
class GCNRegressor(NNModel):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        input_height: int,
        input_width: int,
        dropout: float = 0.1,
        num_layers: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.backbone = GCNBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            input_height=input_height,
            input_width=input_width,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        feature: torch.Tensor,
        label: torch.Tensor = None,
        return_loss: bool = False
    ):
        x = self.backbone(feature)
        x = self.head(x).view(-1)

        if return_loss:
            return torch.mean((x - label.view(-1)) ** 2)

        return x