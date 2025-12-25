"""
Mamba-based PINN model (sequence -> perfusion).

Designed for long sequences: run Mamba over per-frame global embeddings to
produce temporal weights, then aggregate spatial feature maps with those weights.
This keeps the "Mamba blocks" benefit while avoiding the very expensive per-pixel
temporal modeling when T is large.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_blocks import MambaBlock, SpatialDecoder, SpatialEncoder


class MambaPINNNet(nn.Module):
    """
    Input: (B, T, H, W) thermal frames
    Output: (B, 1, H, W) perfusion map (positive)

    Steps:
      1) Spatial encoder per frame
      2) Global pool each frame -> (B, T, C)
      3) Mamba layers on (B, T, C)
      4) Attention weights over time
      5) Weighted sum of spatial features -> decode
    """

    def __init__(self, base_channels: int = 32, d_state: int = 16, n_mamba_layers: int = 4):
        super().__init__()

        self.base_channels = base_channels
        self.d_state = d_state
        self.n_mamba_layers = n_mamba_layers

        self.spatial_encoder = SpatialEncoder(base_channels)
        feature_dim = self.spatial_encoder.out_channels

        self.mamba_layers = nn.ModuleList(
            [MambaBlock(d_model=feature_dim, d_state=d_state) for _ in range(n_mamba_layers)]
        )
        self.attn_head = nn.Linear(feature_dim, 1)

        self.spatial_decoder = SpatialDecoder(feature_dim, base_channels)

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        B, T, H, W = x.shape

        # Late-frame normalization (helps suppress early artifacts)
        late_start = int(T * 0.7)
        late_mean = x[:, late_start:].mean(dim=[1, 2, 3], keepdim=True)
        late_std = x[:, late_start:].std(dim=[1, 2, 3], keepdim=True) + 1e-8
        x_norm = (x - late_mean) / late_std

        # Spatial features per frame
        feat = self.spatial_encoder(x_norm.reshape(B * T, 1, H, W))  # (B*T, C, H', W')
        _, C, Hf, Wf = feat.shape
        feat = feat.view(B, T, C, Hf, Wf)

        # Global embedding per frame
        emb = feat.mean(dim=[3, 4])  # (B, T, C)

        # Mamba temporal processing
        for mamba in self.mamba_layers:
            emb = mamba(emb)

        # Temporal weights
        logits = self.attn_head(emb).squeeze(-1)  # (B, T)
        attn = F.softmax(logits, dim=1)

        # Aggregate spatial features with learned weights
        agg = (feat * attn.view(B, T, 1, 1, 1)).sum(dim=1)  # (B, C, H', W')

        perf = self.spatial_decoder(agg)
        if perf.shape[-2:] != (H, W):
            perf = F.interpolate(perf, size=(H, W), mode="bilinear", align_corners=True)

        if return_debug:
            return perf, attn
        return perf
