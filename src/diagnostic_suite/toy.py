from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def _smooth_random_field(rng: np.random.Generator, grid: int, spectral_cutoff: int = 5) -> np.ndarray:
    """Generate a smooth periodic scalar field on a square grid."""
    noise = rng.normal(size=(grid, grid)).astype(np.float32)
    coeff = np.fft.rfft2(noise)
    kx = np.fft.fftfreq(grid)[:, None]
    ky = np.fft.rfftfreq(grid)[None, :]
    radius = np.sqrt(kx * kx + ky * ky)
    filt = np.exp(-0.5 * (radius * grid / float(spectral_cutoff)) ** 4)
    field = np.fft.irfft2(coeff * filt, s=(grid, grid)).astype(np.float32)
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field


def _advdiff_trajectory(
    u0: np.ndarray,
    steps: int,
    dt: float,
    velocity: Tuple[float, float],
    diffusivity: float,
) -> np.ndarray:
    """Exact spectral trajectory for periodic linear advection-diffusion."""
    grid = int(u0.shape[0])
    kx = 2.0 * np.pi * np.fft.fftfreq(grid)[:, None]
    ky = 2.0 * np.pi * np.fft.rfftfreq(grid)[None, :]
    lap = kx * kx + ky * ky
    adv = velocity[0] * kx + velocity[1] * ky
    uhat0 = np.fft.rfft2(u0)
    frames = []
    for n in range(steps):
        t = n * dt
        phase = np.exp((-diffusivity * lap - 1j * adv) * t)
        frames.append(np.fft.irfft2(uhat0 * phase, s=(grid, grid)).astype(np.float32))
    return np.stack(frames, axis=0)[..., None]


def make_periodic_advection(
    n_traj: int = 8,
    n_time: int = 8,
    size: int = 32,
    channels: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Return an in-memory periodic advection-diffusion toy array.

    This compatibility helper returns an array with shape ``(N,T,X,Y,C)``.
    For ``channels > 1``, channels are independent smooth scalar fields.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_traj):
        chans = []
        for _c in range(channels):
            u0 = _smooth_random_field(rng, size)
            velocity = (0.75 + 0.15 * rng.normal(), -0.35 + 0.15 * rng.normal())
            traj = _advdiff_trajectory(u0, steps=n_time, dt=0.05, velocity=velocity, diffusivity=0.02)
            chans.append(traj[..., 0])
        out.append(np.stack(chans, axis=-1))
    return np.stack(out, axis=0).astype(np.float32)


def make_toy_data(
    out_root: Path | str,
    *,
    n_train: int = 24,
    n_val: int = 8,
    n_test: int = 8,
    steps: int = 8,
    grid: int = 32,
    dt: float = 0.05,
    diffusivity: float = 0.02,
    seed: int = 0,
) -> Path:
    """Create a tiny 2D advection-diffusion dataset compatible with ``toy2d``.

    Files are saved as ``train.npy``, ``val.npy``, and ``test.npy`` under
    ``<out_root>/toy_advdiff2d`` with shape ``(N, T, X, Y, C)``.
    """
    out = Path(out_root) / "toy_advdiff2d"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def build_split(n: int) -> np.ndarray:
        trajs = []
        for _ in range(n):
            u0 = _smooth_random_field(rng, grid)
            # Slight per-trajectory variation tests whether models learn a map,
            # not a single fixed sequence.
            velocity = (0.75 + 0.15 * rng.normal(), -0.35 + 0.15 * rng.normal())
            nu = max(1.0e-4, diffusivity * (1.0 + 0.15 * rng.normal()))
            trajs.append(_advdiff_trajectory(u0, steps=steps, dt=dt, velocity=velocity, diffusivity=nu))
        return np.stack(trajs, axis=0).astype(np.float32)

    np.save(out / "train.npy", build_split(n_train))
    np.save(out / "val.npy", build_split(n_val))
    np.save(out / "test.npy", build_split(n_test))
    return out
