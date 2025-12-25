from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Simplified Mamba-style selective state space block (B, T, D) -> (B, T, D)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :T]
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self._ssm(x)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return y + residual

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        A = -torch.exp(self.A_log)
        x_proj = self.x_proj(x)
        delta, B_param, C_param = x_proj.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta)

        deltaA = torch.exp(delta * A.view(1, 1, -1))

        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            x_t = x[:, t, :]
            x_state = x_t.mean(dim=-1, keepdim=True).expand(-1, self.d_state)
            h = deltaA[:, t, :] * h + B_param[:, t, :] * x_state
            gate = torch.sigmoid(C_param[:, t, :].mean(dim=-1, keepdim=True))
            ys.append(x_t * gate)

        y = torch.stack(ys, dim=1)
        y = y + self.D.view(1, 1, -1) * x
        return y


class SpatialEncoder(nn.Module):
    """Extract spatial features from each frame."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        bc = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, bc, 3, stride=2, padding=1),
            nn.InstanceNorm2d(bc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc, bc * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(bc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc * 2, bc * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(bc * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.out_channels = bc * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpatialDecoder(nn.Module):
    """Decode spatial features to perfusion map."""

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        bc = base_channels

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, bc * 4, 3, padding=1),
            nn.InstanceNorm2d(bc * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(bc * 4, bc * 2, 3, padding=1),
            nn.InstanceNorm2d(bc * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(bc * 2, bc, 3, padding=1),
            nn.InstanceNorm2d(bc),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(bc, bc // 2, 3, padding=1),
            nn.InstanceNorm2d(bc // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bc // 2, 1, 1),
            nn.Softplus(beta=5.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

