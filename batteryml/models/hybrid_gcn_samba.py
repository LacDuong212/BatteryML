# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import torch
import torch.nn as nn

from batteryml.builders import MODELS
from batteryml.models.nn_model import NNModel
from batteryml.models.sambamixer import SambaBackbone
from batteryml.models.gcn import GCNBackbone


@MODELS.register()
class HybridGCNSambaRegressor(NNModel):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        input_height: int,
        input_width: int,
        dropout: float = 0.1,
        samba_layers: int = 1,
        gcn_layers: int = 1,
        fusion_hidden_dim: int = 32,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.samba_backbone = SambaBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            input_height=input_height,
            input_width=input_width,
            dropout=dropout,
            num_layers=samba_layers,
        )

        self.gcn_backbone = GCNBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            input_height=input_height,
            input_width=input_width,
            dropout=dropout,
            num_layers=gcn_layers,
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(
        self,
        feature: torch.Tensor,
        label: torch.Tensor = None,
        return_loss: bool = False
    ):
        h_samba = self.samba_backbone(feature)   # [B, H]
        h_gcn = self.gcn_backbone(feature)       # [B, H]

        h = torch.cat([h_samba, h_gcn], dim=-1)  # [B, 2H]
        x = self.fusion(h).view(-1)

        if return_loss:
            return torch.mean((x - label.view(-1)) ** 2)

        return x