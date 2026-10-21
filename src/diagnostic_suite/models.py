from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The architectures below are intentionally stronger than toy baselines:
#   * FNO uses coordinate/timestep conditioning and all low-frequency Fourier
#     corners needed by multidimensional real FFTs, including negative modes on
#     the fully complex axes.
#   * DeepONet uses a proper branch/trunk factorization with a convolutional
#     sensor encoder, coordinate trunk, and optional arbitrary query coordinates.
#   * U-Net is a multilevel encoder/decoder with normalized residual blocks.
#   * ResNet is a residual convolutional surrogate with dilation and residual
#     field prediction.
# These are still compact enough to run inside this diagnostic suite, but they
# are no longer the deliberately weakened/minimal versions from the first draft.


def conv_cls(dim: int):
    return {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]


def max_pool_cls(dim: int):
    return {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}[dim]


def adaptive_avg_pool_cls(dim: int):
    return {1: nn.AdaptiveAvgPool1d, 2: nn.AdaptiveAvgPool2d, 3: nn.AdaptiveAvgPool3d}[dim]


def interpolate_mode(dim: int) -> str:
    return {1: "linear", 2: "bilinear", 3: "trilinear"}[dim]


def _group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def avg_pool_nd(x: torch.Tensor, dim: int, kernel: int = 3, *, padding_mode: str = "replicate") -> torch.Tensor:
    """Channelwise box filter with explicit boundary extension and unit stride."""
    if kernel < 1 or kernel % 2 != 1:
        raise ValueError("The smoothing kernel must have a positive odd size")
    if padding_mode not in {"replicate", "circular"}:
        raise ValueError("Smoothing padding must be replicate or circular")
    pad = kernel // 2
    if dim == 1:
        return F.avg_pool1d(F.pad(x, (pad, pad), mode=padding_mode), kernel, stride=1)
    if dim == 2:
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode=padding_mode), kernel, stride=1)
    if dim == 3:
        return F.avg_pool3d(F.pad(x, (pad, pad, pad, pad, pad, pad), mode=padding_mode), kernel, stride=1)
    raise ValueError(dim)


def coordinate_grid_like(x: torch.Tensor) -> torch.Tensor:
    """Return normalized coordinate channels with shape (B, dim, *spatial)."""
    spatial = tuple(int(n) for n in x.shape[2:])
    axes = [torch.linspace(0.0, 1.0, n, device=x.device, dtype=x.dtype) for n in spatial]
    mesh = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack(mesh, dim=0).unsqueeze(0)
    return grid.expand(x.shape[0], -1, *spatial)


def add_tau_channel(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    if tau.ndim == 0:
        tau = tau.view(1).expand(x.shape[0])
    if tau.ndim > 1:
        tau = tau.view(tau.shape[0])
    spatial = x.shape[2:]
    t = (
        tau.to(device=x.device, dtype=x.dtype)
        .view(x.shape[0], 1, *([1] * len(spatial)))
        .expand(x.shape[0], 1, *spatial)
    )
    return torch.cat([x, t], dim=1)


def add_tau_and_coords(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Append normalized timestep and spatial coordinate channels."""
    return torch.cat([add_tau_channel(x, tau), coordinate_grid_like(x)], dim=1)


class PointwiseMLP(nn.Module):
    """1x1 convolutional MLP acting independently at every grid point."""

    def __init__(self, dim: int, in_ch: int, hidden_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        Conv = conv_cls(dim)
        self.net = nn.Sequential(
            Conv(in_ch, hidden_ch, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            Conv(hidden_ch, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, layers: int, dropout: float = 0.0):
        super().__init__()
        if layers < 2:
            raise ValueError("MLP needs at least two layers")
        mods: List[nn.Module] = []
        d = in_dim
        for _ in range(layers - 1):
            mods += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        mods.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourierFeatures(nn.Module):
    """Deterministic Fourier features for coordinate trunks."""

    def __init__(self, in_dim: int, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.include_input = include_input
        self.register_buffer("freqs", 2.0 ** torch.arange(num_frequencies, dtype=torch.float32), persistent=False)
        self.out_dim = (in_dim if include_input else 0) + 2 * in_dim * num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=x.device, dtype=x.dtype)
        angles = 2.0 * math.pi * x.unsqueeze(-1) * freqs
        feats = [torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)]
        if self.include_input:
            feats.insert(0, x)
        return torch.cat(feats, dim=-1)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        in_ch: int,
        out_ch: int,
        hidden_ch: Optional[int] = None,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        Conv = conv_cls(dim)
        hidden_ch = hidden_ch or out_ch
        pad = dilation * (kernel_size // 2)
        self.conv1 = Conv(in_ch, hidden_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(_group_count(hidden_ch), hidden_ch)
        self.conv2 = Conv(hidden_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.skip = Conv(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.gelu(self.norm1(h))
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.gelu(h + self.skip(x))


class SpectralConvND(nn.Module):
    """
    Multidimensional spectral convolution for real-valued fields.

    torch.fft.rfftn stores the last transformed dimension with nonnegative
    frequencies only, but the preceding transformed dimensions still contain
    positive and negative frequencies. A standard multidimensional FNO therefore
    needs separate weights for every low-frequency corner of those preceding
    axes. This module implements that convention for 1D, 2D, and 3D.
    """

    def __init__(self, dim: int, in_ch: int, out_ch: int, modes: int | Sequence[int]):
        super().__init__()
        self.dim = dim
        if isinstance(modes, int):
            self.modes = tuple([int(modes)] * dim)
        else:
            if len(modes) != dim:
                raise ValueError(f"modes must have length {dim}; got {modes}")
            self.modes = tuple(int(m) for m in modes)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.sign_corners = [()] if dim == 1 else list(np.ndindex(*([2] * (dim - 1))))
        scale = 1.0 / math.sqrt(max(1, in_ch * out_ch))
        self.weights = nn.ParameterList(
            [
                nn.Parameter(scale * torch.randn(in_ch, out_ch, *self.modes, dtype=torch.cfloat))
                for _ in self.sign_corners
            ]
        )

    def _mode_counts(self, xft_shape: Sequence[int]) -> Tuple[int, ...]:
        # xft_shape is the transformed spatial shape after rfftn.
        counts: List[int] = []
        for ax in range(self.dim - 1):
            n = int(xft_shape[ax])
            counts.append(min(self.modes[ax], max(1, n // 2)))
        counts.append(min(self.modes[-1], int(xft_shape[-1])))
        return tuple(counts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        spatial = tuple(int(n) for n in x.shape[2:])
        fft_dims = tuple(range(2, x.ndim))
        xft = torch.fft.rfftn(x, dim=fft_dims, norm="ortho")
        out_ft = torch.zeros((b, self.out_ch, *xft.shape[2:]), device=x.device, dtype=torch.cfloat)
        counts = self._mode_counts(xft.shape[2:])
        if any(m <= 0 for m in counts):
            return torch.fft.irfftn(out_ft, s=spatial, dim=fft_dims, norm="ortho")

        for corner_id, signs in enumerate(self.sign_corners):
            data_slices: List[slice] = [slice(None), slice(None)]
            weight_slices: List[slice] = [slice(None), slice(None)]
            for ax, sign in enumerate(signs):
                m = counts[ax]
                data_slices.append(slice(0, m) if sign == 0 else slice(-m, None))
                weight_slices.append(slice(0, m))
            m_last = counts[-1]
            data_slices.append(slice(0, m_last))
            weight_slices.append(slice(0, m_last))
            out_ft[tuple(data_slices)] = torch.einsum(
                "bi...,io...->bo...",
                xft[tuple(data_slices)],
                self.weights[corner_id][tuple(weight_slices)],
            )
        return torch.fft.irfftn(out_ft, s=spatial, dim=fft_dims, norm="ortho")


class FNOBlock(nn.Module):
    def __init__(self, dim: int, width: int, modes: int, dropout: float = 0.0):
        super().__init__()
        Conv = conv_cls(dim)
        self.spectral = SpectralConvND(dim, width, width, modes)
        self.pointwise = Conv(width, width, 1)
        self.norm = nn.GroupNorm(_group_count(width), width)
        self.mlp = PointwiseMLP(dim, width, 2 * width, width, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.spectral(x) + self.pointwise(x)
        h = F.gelu(self.norm(h))
        h = self.mlp(h)
        return F.gelu(x + h)


class FNO(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        width: int,
        modes: int,
        layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        Conv = conv_cls(dim)
        self.dim = dim
        self.lift = Conv(in_channels + 1 + dim, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(dim, width, modes, dropout=dropout) for _ in range(layers)])
        self.proj = PointwiseMLP(dim, width, 2 * width, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        h = self.lift(add_tau_and_coords(x, tau))
        for block in self.blocks:
            h = block(h)
        return self.proj(h)


class DownBlock(nn.Module):
    def __init__(self, dim: int, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        Pool = max_pool_cls(dim)
        self.net = nn.Sequential(Pool(2), ResidualConvBlock(dim, in_ch, out_ch, dropout=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, dim: int, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.conv = ResidualConvBlock(dim, in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        mode = interpolate_mode(self.dim)
        x = F.interpolate(x, size=skip.shape[2:], mode=mode, align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetND(nn.Module):
    def __init__(
        self, dim: int, in_channels: int, out_channels: int, width: int, levels: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        if levels < 2:
            raise ValueError("UNetND needs at least two levels")
        Conv = conv_cls(dim)
        self.dim = dim
        feats = [width * (2**i) for i in range(levels)]
        self.stem = ResidualConvBlock(dim, in_channels + 1 + dim, feats[0], dropout=dropout)
        self.downs = nn.ModuleList([DownBlock(dim, feats[i], feats[i + 1], dropout=dropout) for i in range(levels - 1)])
        self.bottleneck = ResidualConvBlock(dim, feats[-1], feats[-1], dropout=dropout)
        self.ups = nn.ModuleList(
            [UpBlock(dim, feats[i], feats[i - 1], feats[i - 1], dropout=dropout) for i in range(levels - 1, 0, -1)]
        )
        self.out = Conv(feats[0], out_channels, 1)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        h = self.stem(add_tau_and_coords(x, tau))
        skips = [h]
        for down in self.downs:
            h = down(h)
            skips.append(h)
        h = self.bottleneck(h)
        # Do not use skips[-1]; it has the same resolution as the bottleneck.
        for i, up in enumerate(self.ups):
            skip = skips[-(i + 2)]
            h = up(h, skip)
        return self.out(h)


# Backwards-compatible name used by earlier drafts.
UNetSmall = UNetND


class ResNetSurrogate(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        width: int,
        blocks: int = 8,
        dropout: float = 0.0,
        predict_residual: bool = True,
    ):
        super().__init__()
        Conv = conv_cls(dim)
        self.predict_residual = predict_residual and (in_channels == out_channels)
        self.stem = ResidualConvBlock(dim, in_channels + 1 + dim, width, dropout=dropout)
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock(dim, width, width, dilation=dilations[i % len(dilations)], dropout=dropout)
                for i in range(blocks)
            ]
        )
        self.head = nn.Sequential(
            Conv(width, width, 1),
            nn.GELU(),
            Conv(width, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        h = self.stem(add_tau_and_coords(x, tau))
        for block in self.blocks:
            h = block(h)
        y = self.head(h)
        return x + y if self.predict_residual else y


class DeepONet(nn.Module):
    """
    Branch/trunk DeepONet for grid sensors with optional arbitrary query points.

    forward(x, tau) returns a full grid field of shape (B,C,*spatial_shape).
    forward(x, tau, query_coords=coords) returns point values of shape (B,N,C),
    where coords has shape (N,dim) or (B,N,dim) in normalized coordinates.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        width: int,
        latent: int,
        spatial_shape: Tuple[int, ...],
        branch_levels: int = 3,
        trunk_layers: int = 4,
        fourier_frequencies: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.out_channels = out_channels
        self.spatial_shape = tuple(int(n) for n in spatial_shape)
        self.latent = int(latent)

        branch_feats = [width * (2**i) for i in range(branch_levels)]
        self.branch_stem = ResidualConvBlock(dim, in_channels + 1 + dim, branch_feats[0], dropout=dropout)
        self.branch_downs = nn.ModuleList(
            [DownBlock(dim, branch_feats[i], branch_feats[i + 1], dropout=dropout) for i in range(branch_levels - 1)]
        )
        Pool = adaptive_avg_pool_cls(dim)
        self.branch_pool = Pool(1)
        self.branch_head = MLP(
            branch_feats[-1],
            out_channels * self.latent,
            hidden_dim=max(2 * width, self.latent),
            layers=3,
            dropout=dropout,
        )

        self.coord_features = FourierFeatures(dim + 1, num_frequencies=fourier_frequencies, include_input=True)
        self.trunk = MLP(
            self.coord_features.out_dim,
            self.latent,
            hidden_dim=max(2 * width, self.latent),
            layers=trunk_layers,
            dropout=dropout,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _default_coords(self, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        axes = [torch.linspace(0.0, 1.0, n, device=device, dtype=dtype) for n in self.spatial_shape]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, self.dim)
        return grid.view(1, grid.shape[0], self.dim).expand(b, -1, -1)

    def _prepare_coords(
        self, b: int, query_coords: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, bool]:
        if query_coords is None:
            return self._default_coords(b, device, dtype), True
        coords = query_coords.to(device=device, dtype=dtype)
        if coords.ndim == 2:
            coords = coords.view(1, coords.shape[0], coords.shape[1]).expand(b, -1, -1)
        if coords.ndim != 3 or coords.shape[0] != b or coords.shape[-1] != self.dim:
            raise ValueError(
                f"query_coords must have shape (N,{self.dim}) or (B,N,{self.dim}); got {tuple(coords.shape)}"
            )
        return coords, False

    def forward(self, x: torch.Tensor, tau: torch.Tensor, query_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = x.shape[0]
        if tau.ndim == 0:
            tau = tau.view(1).expand(b)
        if tau.ndim > 1:
            tau = tau.view(tau.shape[0])
        tau = tau.to(device=x.device, dtype=x.dtype)

        h = self.branch_stem(add_tau_and_coords(x, tau))
        for down in self.branch_downs:
            h = down(h)
        pooled = self.branch_pool(h).flatten(1)
        coeff = self.branch_head(pooled).view(b, self.out_channels, self.latent)

        coords, default_grid = self._prepare_coords(b, query_coords, x.device, x.dtype)
        tau_feat = tau.view(b, 1, 1).expand(b, coords.shape[1], 1)
        trunk_in = torch.cat([coords, tau_feat], dim=-1)
        basis = self.trunk(self.coord_features(trunk_in))
        y = torch.einsum("bcl,bnl->bnc", coeff, basis) / math.sqrt(float(self.latent))
        y = y + self.bias.view(1, 1, self.out_channels)
        if not default_grid:
            return y
        return y.reshape(b, *self.spatial_shape, self.out_channels).movedim(-1, 1).contiguous()


# Backwards-compatible name used by earlier drafts.
DeepONetLite = DeepONet


class OversmoothWrapper(nn.Module):
    """Apply a 3**dim box filter after each base-model call.

    The CLI uses circular extension for periodic tasks and replicated extension
    otherwise. Direct construction requires the boundary policy explicitly.
    Spatial resolution, channels, and the trained base weights are unchanged.
    """

    def __init__(self, base: nn.Module, dim: int, passes: int = 2, *, periodic: bool):
        super().__init__()
        if dim not in {1, 2, 3}:
            raise ValueError("Smoothing dimension must be one, two, or three")
        if not isinstance(passes, int) or passes < 0:
            raise ValueError("Smoothing passes must be a nonnegative integer")
        if not isinstance(periodic, bool):
            raise ValueError("periodic must be an explicit boolean boundary policy")
        self.base = base
        self.dim = dim
        self.passes = passes
        self.padding_mode = "circular" if periodic else "replicate"

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        y = self.base(x, tau)
        for _ in range(self.passes):
            y = avg_pool_nd(y, self.dim, 3, padding_mode=self.padding_mode)
        return y


def build_model(
    model_name: str, dim: int, channels: int, spatial_shape: Tuple[int, ...], width: int, modes: int
) -> nn.Module:
    # Keep models substantial but avoid pathological memory blowups on 3D grids.
    unet_levels = 4 if min(spatial_shape) >= 32 else 3
    if model_name == "fno":
        return FNO(dim, channels, channels, width=width, modes=modes, layers=4)
    if model_name == "unet":
        return UNetND(dim, channels, channels, width=width, levels=unet_levels)
    if model_name == "resnet":
        return ResNetSurrogate(dim, channels, channels, width=width, blocks=8)
    if model_name == "deeponet":
        return DeepONet(dim, channels, channels, width=width, latent=max(2 * width, 64), spatial_shape=spatial_shape)
    raise ValueError(f"Unknown model {model_name}")
