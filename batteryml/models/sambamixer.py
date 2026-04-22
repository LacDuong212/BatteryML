# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch
import torch.nn as nn

from batteryml.builders import MODELS
from batteryml.models.nn_model import NNModel


class TemporalMixerBlock(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.channel_norm = nn.LayerNorm(hidden_dim)

        self.token_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
            nn.Dropout(dropout),
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.token_norm(x)
        y = y.transpose(1, 2)
        y = self.token_mixer(y)
        y = y.transpose(1, 2)
        x = residual + y

        residual = x
        y = self.channel_norm(x)
        y = self.channel_mixer(y)
        x = residual + y
        return x


class SambaBackbone(nn.Module):
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

        token_dim = in_channels * input_width
        seq_len = input_height

        self.input_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            TemporalMixerBlock(seq_len=seq_len, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            feature = feature.unsqueeze(1)

        # feature: [B, C, H, W]
        B, C, H, W = feature.size()
        x = feature.permute(0, 2, 1, 3).contiguous().view(B, H, C * W)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)   # [B, hidden_dim]
        return x


@MODELS.register()
class SambaMixerRegressor(NNModel):
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

        self.backbone = SambaBackbone(
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