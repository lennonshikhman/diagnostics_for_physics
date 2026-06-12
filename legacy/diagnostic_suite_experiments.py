#!/usr/bin/env python3
"""
Diagnostic consistency experiments for learned PDE time propagators.

This script is intentionally self-contained.  It was written to live in

    ~/Desktop/research_work/diagnostic_suite

and to read data from the existing COMET data directory, by default

    ~/Desktop/research_work/COMET/comet_ncs_codebase/data/raw

It trains paper-grade compact FNO, DeepONet, U-Net, and ResNet-style surrogates
in well-trained and underfit regimes, then evaluates a diagnostic panel including
relative L2,
semigroup error, generator discrepancy, generator excess/ratio relative to the
reference finite-difference generator baseline, energy-law violation,
energy-decay error, integral-balance/mass error, constraint error, perturbation
response when reference perturbations are available, and scaling-law error when
configured. Unavailable optional diagnostics are reported as NaN, never as zero.

The code is defensive about file formats because PDEBench, APEBench, and The
Well store trajectories differently.  It includes generic HDF5/NPY discovery
logic adapted from the unrelated COMET codebase's data handling conventions.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

EPS = 1.0e-8


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TaskSpec:
    name: str
    label: str
    dim: int
    globs: Dict[str, List[str]]
    max_channels: int = 8
    periodic: bool = True
    channel_layout: Dict[str, Any] = field(default_factory=dict)
    energy_type: str = "generic_l2"  # generic_l2, kinetic, shallow_water, cns, mhd
    balance_channel: Optional[int] = 0
    positivity_channels: List[int] = field(default_factory=list)
    velocity_channels: List[int] = field(default_factory=list)
    magnetic_channels: List[int] = field(default_factory=list)
    scaling: Optional[str] = None  # linear_amplitude only if exact/appropriate


def build_task_specs(data_root: Path, spatial_limit_2d: int, spatial_limit_3d: int) -> Dict[str, TaskSpec]:
    # spatial limits are passed separately to readers; stored here for labels only.
    return {
        "ns2d": TaskSpec(
            name="ns2d",
            label="2D incompressible NS / APEBench",
            dim=2,
            globs={
                "train": ["apebench/apebench_incompressible_ns_2d/data/*train.npy"],
                "val": ["apebench/apebench_incompressible_ns_2d/data/*train.npy"],
                "test": ["apebench/apebench_incompressible_ns_2d/data/*test.npy"],
            },
            max_channels=4,
            periodic=True,
            energy_type="kinetic",
            balance_channel=None,
            velocity_channels=[0, 1],
        ),
        "shallow2d": TaskSpec(
            name="shallow2d",
            label="2D shallow water / PDEBench",
            dim=2,
            globs={
                "train": ["pdebench/pdebench_shallow_water_2d/*.h5"],
                "val": ["pdebench/pdebench_shallow_water_2d/*.h5"],
                "test": ["pdebench/pdebench_shallow_water_2d/*.h5"],
            },
            max_channels=4,
            periodic=False,
            energy_type="shallow_water",
            balance_channel=0,
            positivity_channels=[0],
            velocity_channels=[1, 2],
        ),
        "active2d": TaskSpec(
            name="active2d",
            label="2D active matter / The Well",
            dim=2,
            globs={
                "train": ["the_well/datasets/active_matter/data/train/*.hdf5"],
                "val": ["the_well/datasets/active_matter/data/valid/*.hdf5"],
                "test": ["the_well/datasets/active_matter/data/test/*.hdf5"],
            },
            max_channels=8,
            periodic=True,
            energy_type="generic_l2",
            balance_channel=None,
        ),
        "cns3d": TaskSpec(
            name="cns3d",
            label="3D compressible NS / PDEBench",
            dim=3,
            globs={
                "train": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
                "val": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
                "test": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
            },
            max_channels=5,
            periodic=True,
            energy_type="cns",
            balance_channel=0,
            positivity_channels=[0, 1],
            velocity_channels=[2, 3, 4],
        ),
        "mhd3d": TaskSpec(
            name="mhd3d",
            label="3D MHD / The Well",
            dim=3,
            globs={
                "train": ["the_well/datasets/MHD_64/data/train/*.hdf5"],
                "val": ["the_well/datasets/MHD_64/data/valid/*.hdf5"],
                "test": ["the_well/datasets/MHD_64/data/test/*.hdf5"],
            },
            max_channels=8,
            periodic=True,
            energy_type="mhd",
            balance_channel=0,
            positivity_channels=[0],
            # These are best-effort defaults; verify with inspect-data output.
            velocity_channels=[1, 2, 3],
            magnetic_channels=[4, 5, 6],
        ),
    }


# =============================================================================
# Generic trajectory readers
# =============================================================================

def _numeric_h5_datasets(path: Path) -> List[Tuple[str, Tuple[int, ...], str]]:
    out: List[Tuple[str, Tuple[int, ...], str]] = []
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                try:
                    if len(obj.shape) >= 3 and np.issubdtype(obj.dtype, np.number):
                        out.append((name, tuple(int(x) for x in obj.shape), str(obj.dtype)))
                except Exception:
                    pass
        f.visititems(visit)
    return out


def _choose_h5_dataset(path: Path, dim: int) -> str:
    candidates = _numeric_h5_datasets(path)
    if not candidates:
        raise RuntimeError(f"No numeric HDF5 trajectory datasets found in {path}")
    ranked = []
    for name, shape, dtype in candidates:
        lname = name.lower()
        score = 0
        if len(shape) >= dim + 2:
            score += 50
        if len(shape) >= dim + 3:
            score += 50
        if len(shape) >= 2 and shape[0] > 1:
            score += 10
        if len(shape) >= 2 and shape[1] > 1:
            score += 10
        if any(s in lname for s in ("field", "data", "solution", "tensor", "trajectory", "states", "u")):
            score += 25
        if any(s in lname for s in ("coord", "grid", "time", "x-coordinate", "y-coordinate", "param", "mask")):
            score -= 100
        if shape[-1] <= 32:
            score += 5
        score += min(50, int(math.log10(max(1, int(np.prod(shape))))))
        ranked.append((score, name, shape))
    ranked.sort(reverse=True)
    return ranked[0][1]


def _nested_h5_trajectory_keys(path: Path, dim: int) -> List[Tuple[str, Tuple[int, ...]]]:
    out: List[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            try:
                if not np.issubdtype(obj.dtype, np.number):
                    return
                shape = tuple(int(x) for x in obj.shape)
                lname = name.lower()
                if not (lname.endswith("/data") or lname.endswith("/tensor") or lname.endswith("/solution")):
                    return
                if len(shape) in {dim + 1, dim + 2} and shape[0] > 1 and max(shape[1:]) > 8:
                    out.append((name, shape))
            except Exception:
                pass
        f.visititems(visit)
    out.sort(key=lambda x: x[0])
    return out


def _top_level_cfd_keys(path: Path, dim: int) -> Tuple[List[str], Optional[Tuple[int, ...]]]:
    preferred = [
        "density", "pressure", "Vx", "Vy", "Vz", "vx", "vy", "vz",
        "velocity_x", "velocity_y", "velocity_z", "x-velocity", "y-velocity", "z-velocity",
    ]
    with h5py.File(path, "r") as f:
        shapes: Dict[str, Tuple[int, ...]] = {}
        for k in preferred:
            if k in f and isinstance(f[k], h5py.Dataset):
                try:
                    if np.issubdtype(f[k].dtype, np.number):
                        shape = tuple(int(x) for x in f[k].shape)
                        if len(shape) == dim + 2 and shape[0] > 1 and shape[1] > 1:
                            shapes[k] = shape
                except Exception:
                    pass
        if not shapes:
            return [], None
        counts: Dict[Tuple[int, ...], int] = {}
        for s in shapes.values():
            counts[s] = counts.get(s, 0) + 1
        shape = max(counts.items(), key=lambda kv: kv[1])[0]
        keys = [k for k in preferred if shapes.get(k) == shape]
        return keys, shape


def _to_channels_first_state(arr: np.ndarray, dim: int, max_channels: int) -> torch.Tensor:
    x = np.asarray(arr, dtype=np.float32)
    while x.ndim > dim + 1 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == dim:
        x = x[..., None]
    if x.ndim > dim + 1:
        # If there are extra non-spatial axes, flatten them into channels using the
        # most common convention: spatial axes first, channels/features last.
        if x.shape[-1] <= max_channels or x.shape[-1] <= x.shape[0]:
            spatial = x.shape[:dim]
            x = x.reshape(*spatial, -1)
        elif x.shape[0] <= max_channels:
            x = x.reshape(x.shape[0], *x.shape[-dim:])
        else:
            spatial = x.shape[:dim]
            x = x.reshape(*spatial, -1)
    if x.shape[-1] <= max_channels:
        x = np.moveaxis(x, -1, 0)
    elif x.shape[0] <= max_channels:
        pass
    else:
        x = np.moveaxis(x, -1, 0)
    if x.shape[0] > max_channels:
        x = x[:max_channels]
    return torch.from_numpy(np.ascontiguousarray(x))


def _spatial_limit(x: torch.Tensor, dim: int, limit: int) -> torch.Tensor:
    if limit <= 0:
        return x.contiguous()
    slices: List[slice] = [slice(None)]
    for n in x.shape[1:]:
        if n <= limit:
            slices.append(slice(None))
        else:
            step = int(math.ceil(n / limit))
            slices.append(slice(0, n, step))
    y = x[tuple(slices)]
    slices = [slice(None)]
    for n in y.shape[1:]:
        if n <= limit:
            slices.append(slice(None))
        else:
            start = (n - limit) // 2
            slices.append(slice(start, start + limit))
    return y[tuple(slices)].contiguous()


class BaseReader:
    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int, split_within_file: bool):
        self.path = Path(path)
        self.task = task
        self.split = "val" if split == "valid" else split
        self.spatial_limit = spatial_limit
        self.split_within_file = split_within_file
        self.n_traj = 0
        self.n_time = 0

    def state(self, traj: int, t: int) -> torch.Tensor:
        raise NotImplementedError

    def traj_indices_for_split(self) -> List[int]:
        if self.n_traj <= 0:
            return []
        if not self.split_within_file:
            return list(range(self.n_traj))
        n_train = max(1, int(0.80 * self.n_traj))
        n_val = max(1, int(0.10 * self.n_traj))
        if self.split == "train":
            idx = list(range(0, n_train))
        elif self.split in {"val", "valid", "validation"}:
            idx = list(range(n_train, min(self.n_traj, n_train + n_val)))
        else:
            idx = list(range(min(self.n_traj, n_train + n_val), self.n_traj))
        return idx if idx else list(range(self.n_traj))


class NpyReader(BaseReader):
    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int, split_within_file: bool):
        super().__init__(path, task, split, spatial_limit, split_within_file)
        arr = np.load(path, allow_pickle=False)
        if isinstance(arr, np.lib.npyio.NpzFile):
            keys = sorted([k for k in arr.files if np.asarray(arr[k]).ndim >= task.dim + 2])
            if not keys:
                raise RuntimeError(f"No trajectory array found in {path}")
            arr = np.asarray(arr[keys[0]])
        else:
            arr = np.asarray(arr)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == task.dim + 2:  # (T,*spatial,C?) single trajectory
            arr = arr[None]
        self.arr = arr
        self.n_traj = int(arr.shape[0])
        self.n_time = int(arr.shape[1])

    def state(self, traj: int, t: int) -> torch.Tensor:
        return _spatial_limit(_to_channels_first_state(self.arr[traj, t], self.task.dim, self.task.max_channels), self.task.dim, self.spatial_limit)


class H5Reader(BaseReader):
    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int, split_within_file: bool):
        super().__init__(path, task, split, spatial_limit, split_within_file)
        self._h5: Optional[h5py.File] = None
        self.mode = "batched"
        self.dataset_key: Optional[str] = None
        self.dataset_keys: List[str] = []
        cfd_keys, cfd_shape = _top_level_cfd_keys(self.path, task.dim)
        nested = _nested_h5_trajectory_keys(self.path, task.dim)
        if cfd_keys and cfd_shape is not None:
            self.mode = "cfd"
            self.dataset_keys = cfd_keys[: task.max_channels]
            self.shape = cfd_shape
            self.n_traj = int(cfd_shape[0])
            self.n_time = int(cfd_shape[1])
        elif nested:
            self.mode = "nested"
            self.dataset_keys = [k for k, _ in nested]
            self.shape = nested[0][1]
            self.n_traj = len(self.dataset_keys)
            self.n_time = int(self.shape[0])
        else:
            self.mode = "batched"
            self.dataset_key = _choose_h5_dataset(self.path, task.dim)
            with h5py.File(self.path, "r") as f:
                self.shape = tuple(int(x) for x in f[self.dataset_key].shape)
            if len(self.shape) == task.dim + 1:
                self.n_traj = 1
                self.n_time = int(self.shape[0])
            else:
                self.n_traj = int(self.shape[0])
                self.n_time = int(self.shape[1]) if len(self.shape) >= task.dim + 2 else 0
        if self.n_time < 2:
            raise RuntimeError(f"HDF5 reader found too few time steps in {path}: mode={self.mode}, shape={getattr(self, 'shape', None)}")

    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def state(self, traj: int, t: int) -> torch.Tensor:
        f = self.h5()
        if self.mode == "cfd":
            chans = [np.asarray(f[k][traj, t], dtype=np.float32) for k in self.dataset_keys]
            x = torch.from_numpy(np.ascontiguousarray(np.stack(chans, axis=0)))
            return _spatial_limit(x[: self.task.max_channels], self.task.dim, self.spatial_limit)
        if self.mode == "nested":
            key = self.dataset_keys[traj % len(self.dataset_keys)]
            x = _to_channels_first_state(np.asarray(f[key][t], dtype=np.float32), self.task.dim, self.task.max_channels)
            return _spatial_limit(x, self.task.dim, self.spatial_limit)
        assert self.dataset_key is not None
        dset = f[self.dataset_key]
        if len(self.shape) == self.task.dim + 1:
            arr = np.asarray(dset[t], dtype=np.float32)
        else:
            arr = np.asarray(dset[traj, t], dtype=np.float32)
        x = _to_channels_first_state(arr, self.task.dim, self.task.max_channels)
        return _spatial_limit(x, self.task.dim, self.spatial_limit)


class TrajectoryCollection:
    def __init__(self, data_root: Path, task: TaskSpec, split: str, spatial_limit: int):
        self.data_root = Path(data_root)
        self.task = task
        self.split = "val" if split == "valid" else split
        self.spatial_limit = spatial_limit
        self.readers: List[BaseReader] = []
        for pat in task.globs[self.split]:
            for p in sorted(glob.glob(str(self.data_root / pat))):
                path = Path(p)
                low = path.as_posix().lower()
                # If the file is already in a split directory or name, use all trajectories.
                split_tokens = [f"/{self.split}/", f"_{self.split}", f"{self.split}.", f"{self.split}_"]
                if self.split == "val":
                    split_tokens += ["/valid/", "_valid", "validation"]
                force_all = any(tok in low for tok in split_tokens)
                split_within = not force_all
                try:
                    if path.suffix.lower() in {".npy", ".npz"}:
                        self.readers.append(NpyReader(path, task, self.split, spatial_limit, split_within))
                    elif path.suffix.lower() in {".h5", ".hdf5", ".hdf"}:
                        self.readers.append(H5Reader(path, task, self.split, spatial_limit, split_within))
                except Exception as e:
                    print(f"[WARN] Could not open {path}: {e}")
        if not self.readers:
            raise RuntimeError(f"No readable files for task={task.name}, split={self.split}, patterns={task.globs[self.split]}")

    def describe(self) -> List[Dict[str, Any]]:
        rows = []
        for r in self.readers:
            rows.append({
                "task": self.task.name,
                "split": self.split,
                "path": str(r.path),
                "reader": type(r).__name__,
                "mode": getattr(r, "mode", "npy"),
                "n_traj": r.n_traj,
                "n_time": r.n_time,
                "shape": str(getattr(r, "shape", getattr(r, "arr", None).shape if hasattr(r, "arr") else None)),
                "keys": ",".join(getattr(r, "dataset_keys", [])) or getattr(r, "dataset_key", ""),
            })
        return rows


class PairDataset(Dataset):
    def __init__(self, collection: TrajectoryCollection, max_pairs: int, max_delta: int, seed: int = 0):
        self.collection = collection
        self.max_delta = max_delta
        self.items: List[Tuple[int, int, int, int]] = []
        for ridx, r in enumerate(collection.readers):
            for traj in r.traj_indices_for_split():
                for t0 in range(0, r.n_time - 1):
                    for dt in range(1, min(max_delta, r.n_time - 1 - t0) + 1):
                        self.items.append((ridx, traj, t0, dt))
        rng = random.Random(seed)
        rng.shuffle(self.items)
        if max_pairs > 0:
            self.items = self.items[:max_pairs]
        if not self.items:
            raise RuntimeError(f"No training pairs for {collection.task.name}/{collection.split}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ridx, traj, t0, dt = self.items[idx]
        r = self.collection.readers[ridx]
        x = r.state(traj, t0)
        y = r.state(traj, t0 + dt)
        return {
            "x": x,
            "y": y,
            "tau": torch.tensor(float(dt) / max(1, self.max_delta), dtype=torch.float32),
            "dt_index": torch.tensor(dt, dtype=torch.long),
        }


class WindowDataset(Dataset):
    def __init__(self, collection: TrajectoryCollection, max_windows: int, horizon: int, seed: int = 0):
        self.collection = collection
        self.horizon = horizon
        self.items: List[Tuple[int, int, int]] = []
        for ridx, r in enumerate(collection.readers):
            for traj in r.traj_indices_for_split():
                for t0 in range(0, max(0, r.n_time - horizon)):
                    self.items.append((ridx, traj, t0))
        rng = random.Random(seed)
        rng.shuffle(self.items)
        if max_windows > 0:
            self.items = self.items[:max_windows]
        if not self.items:
            raise RuntimeError(f"No windows for {collection.task.name}/{collection.split}; horizon={horizon}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ridx, traj, t0 = self.items[idx]
        r = self.collection.readers[ridx]
        states = [r.state(traj, t0 + k) for k in range(self.horizon + 1)]
        return torch.stack(states, dim=0)  # (T+1,C,*spatial)


# =============================================================================
# Normalization
# =============================================================================

@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def to(self, device: torch.device) -> "Normalizer":
        return Normalizer(self.mean.to(device), self.std.to(device))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return (x - self.mean.view(*shape)) / self.std.view(*shape)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.ndim - 2)
        return x * self.std.view(*shape) + self.mean.view(*shape)


def estimate_normalizer(dataset: PairDataset, n_samples: int = 256) -> Normalizer:
    idxs = list(range(min(len(dataset), n_samples)))
    sums = None
    sums2 = None
    count = 0
    for i in idxs:
        item = dataset[i]
        for key in ("x", "y"):
            x = item[key].float()
            dims = tuple(range(1, x.ndim))
            s = x.sum(dim=dims)
            s2 = (x * x).sum(dim=dims)
            n = int(np.prod(x.shape[1:]))
            sums = s if sums is None else sums + s
            sums2 = s2 if sums2 is None else sums2 + s2
            count += n
    assert sums is not None and sums2 is not None
    mean = sums / max(1, count)
    var = sums2 / max(1, count) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return Normalizer(mean.float(), std.float())


# =============================================================================
# Models
# =============================================================================

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


def avg_pool_nd(x: torch.Tensor, dim: int, kernel: int = 3) -> torch.Tensor:
    pad = kernel // 2
    if dim == 1:
        return F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), kernel, stride=1)
    if dim == 2:
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel, stride=1)
    if dim == 3:
        return F.avg_pool3d(F.pad(x, (pad, pad, pad, pad, pad, pad), mode="replicate"), kernel, stride=1)
    raise ValueError(dim)


def coordinate_grid_like(x: torch.Tensor) -> torch.Tensor:
    """Return normalized coordinate channels with shape (B, dim, *spatial)."""
    spatial = tuple(int(n) for n in x.shape[2:])
    dim = len(spatial)
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
    t = tau.to(device=x.device, dtype=x.dtype).view(x.shape[0], 1, *([1] * len(spatial))).expand(x.shape[0], 1, *spatial)
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
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_ch, out_ch, *self.modes, dtype=torch.cfloat))
            for _ in self.sign_corners
        ])

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
    def __init__(self, dim: int, in_channels: int, out_channels: int, width: int, modes: int, layers: int = 4, dropout: float = 0.0):
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
    def __init__(self, dim: int, in_channels: int, out_channels: int, width: int, levels: int = 4, dropout: float = 0.0):
        super().__init__()
        if levels < 2:
            raise ValueError("UNetND needs at least two levels")
        Conv = conv_cls(dim)
        self.dim = dim
        feats = [width * (2 ** i) for i in range(levels)]
        self.stem = ResidualConvBlock(dim, in_channels + 1 + dim, feats[0], dropout=dropout)
        self.downs = nn.ModuleList([DownBlock(dim, feats[i], feats[i + 1], dropout=dropout) for i in range(levels - 1)])
        self.bottleneck = ResidualConvBlock(dim, feats[-1], feats[-1], dropout=dropout)
        self.ups = nn.ModuleList([
            UpBlock(dim, feats[i], feats[i - 1], feats[i - 1], dropout=dropout)
            for i in range(levels - 1, 0, -1)
        ])
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
        self.blocks = nn.ModuleList([
            ResidualConvBlock(dim, width, width, dilation=dilations[i % len(dilations)], dropout=dropout)
            for i in range(blocks)
        ])
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

        branch_feats = [width * (2 ** i) for i in range(branch_levels)]
        self.branch_stem = ResidualConvBlock(dim, in_channels + 1 + dim, branch_feats[0], dropout=dropout)
        self.branch_downs = nn.ModuleList([
            DownBlock(dim, branch_feats[i], branch_feats[i + 1], dropout=dropout)
            for i in range(branch_levels - 1)
        ])
        Pool = adaptive_avg_pool_cls(dim)
        self.branch_pool = Pool(1)
        self.branch_head = MLP(branch_feats[-1], out_channels * self.latent, hidden_dim=max(2 * width, self.latent), layers=3, dropout=dropout)

        self.coord_features = FourierFeatures(dim + 1, num_frequencies=fourier_frequencies, include_input=True)
        self.trunk = MLP(self.coord_features.out_dim, self.latent, hidden_dim=max(2 * width, self.latent), layers=trunk_layers, dropout=dropout)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _default_coords(self, b: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        axes = [torch.linspace(0.0, 1.0, n, device=device, dtype=dtype) for n in self.spatial_shape]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, self.dim)
        return grid.view(1, grid.shape[0], self.dim).expand(b, -1, -1)

    def _prepare_coords(self, b: int, query_coords: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, bool]:
        if query_coords is None:
            return self._default_coords(b, device, dtype), True
        coords = query_coords.to(device=device, dtype=dtype)
        if coords.ndim == 2:
            coords = coords.view(1, coords.shape[0], coords.shape[1]).expand(b, -1, -1)
        if coords.ndim != 3 or coords.shape[0] != b or coords.shape[-1] != self.dim:
            raise ValueError(f"query_coords must have shape (N,{self.dim}) or (B,N,{self.dim}); got {tuple(coords.shape)}")
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
    def __init__(self, base: nn.Module, dim: int, passes: int = 2):
        super().__init__()
        self.base = base
        self.dim = dim
        self.passes = passes

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        y = self.base(x, tau)
        for _ in range(self.passes):
            y = avg_pool_nd(y, self.dim, 3)
        return y


def build_model(model_name: str, dim: int, channels: int, spatial_shape: Tuple[int, ...], width: int, modes: int) -> nn.Module:
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


# =============================================================================
# Metrics
# =============================================================================

def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((pred - target).flatten(1), dim=1) / (torch.linalg.vector_norm(target.flatten(1), dim=1) + EPS)


def grad_dim(u: torch.Tensor, axis: int, periodic: bool = True) -> torch.Tensor:
    # axis is spatial axis index in [0, dim-1], corresponding to tensor dim 2+axis.
    d = 2 + axis
    if periodic:
        return 0.5 * (torch.roll(u, -1, dims=d) - torch.roll(u, 1, dims=d))
    # one-sided-ish with replicate padding via roll fallback at boundaries.
    out = 0.5 * (torch.roll(u, -1, dims=d) - torch.roll(u, 1, dims=d))
    sl0 = [slice(None)] * u.ndim
    sl1 = [slice(None)] * u.ndim
    sl0[d] = 0
    sl1[d] = 1
    out[tuple(sl0)] = u[tuple(sl1)] - u[tuple(sl0)]
    sl0[d] = -1
    sl1[d] = -2
    out[tuple(sl0)] = u[tuple(sl0)] - u[tuple(sl1)]
    return out


def divergence(uvec: torch.Tensor, dim: int, periodic: bool = True) -> torch.Tensor:
    div = 0.0
    for ax in range(dim):
        div = div + grad_dim(uvec[:, ax:ax+1], ax, periodic)
    return div


def energy(u: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    spatial = tuple(range(2, u.ndim))
    if task.energy_type == "kinetic" and len(task.velocity_channels) >= task.dim and max(task.velocity_channels[:task.dim]) < u.shape[1]:
        return 0.5 * (u[:, task.velocity_channels[:task.dim]] ** 2).mean(dim=(1, *spatial))
    if task.energy_type == "shallow_water":
        h = u[:, 0:1]
        if u.shape[1] >= 3:
            v2 = (u[:, 1:3] ** 2).sum(dim=1, keepdim=True)
            return (0.5 * h.clamp_min(0) * v2 + 0.5 * h ** 2).mean(dim=(1, *spatial))
        return (0.5 * h ** 2).mean(dim=(1, *spatial))
    if task.energy_type == "cns" and u.shape[1] >= 5:
        rho = u[:, 0:1].clamp_min(0)
        p = u[:, 1:2].clamp_min(0)
        vel = u[:, 2:5]
        return (0.5 * rho * (vel ** 2).sum(dim=1, keepdim=True) + p).mean(dim=(1, *spatial))
    if task.energy_type == "mhd" and u.shape[1] >= 7:
        vel = u[:, task.velocity_channels[:3]] if max(task.velocity_channels[:3]) < u.shape[1] else u[:, 1:4]
        mag = u[:, task.magnetic_channels[:3]] if task.magnetic_channels and max(task.magnetic_channels[:3]) < u.shape[1] else u[:, 4:7]
        return 0.5 * ((vel ** 2).sum(dim=1) + (mag ** 2).sum(dim=1)).mean(dim=spatial)
    return 0.5 * (u ** 2).mean(dim=(1, *spatial))


def positivity_error(u: torch.Tensor, channels: List[int]) -> torch.Tensor:
    vals = []
    for c in channels:
        if c < u.shape[1]:
            vals.append(torch.linalg.vector_norm(torch.relu(-u[:, c:c+1]).flatten(1), dim=1) / (torch.linalg.vector_norm(u[:, c:c+1].flatten(1), dim=1) + EPS))
    if not vals:
        return torch.full((u.shape[0],), float("nan"), device=u.device)
    return torch.stack(vals, dim=0).nanmean(dim=0)


def constraint_error(u: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    vals = []
    if task.velocity_channels and len(task.velocity_channels) >= task.dim and max(task.velocity_channels[:task.dim]) < u.shape[1]:
        vel = u[:, task.velocity_channels[:task.dim]]
        div = divergence(vel, task.dim, task.periodic)
        vals.append(torch.linalg.vector_norm(div.flatten(1), dim=1) / (torch.linalg.vector_norm(vel.flatten(1), dim=1) + EPS))
    if task.magnetic_channels and len(task.magnetic_channels) >= task.dim and max(task.magnetic_channels[:task.dim]) < u.shape[1]:
        mag = u[:, task.magnetic_channels[:task.dim]]
        divb = divergence(mag, task.dim, task.periodic)
        vals.append(torch.linalg.vector_norm(divb.flatten(1), dim=1) / (torch.linalg.vector_norm(mag.flatten(1), dim=1) + EPS))
    if task.positivity_channels:
        vals.append(positivity_error(u, task.positivity_channels))
    if not vals:
        return torch.full((u.shape[0],), float("nan"), device=u.device)
    return torch.stack(vals, dim=0).nanmean(dim=0)


def integral_balance_error(u0: torch.Tensor, pred: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    c = task.balance_channel
    if c is None or c >= pred.shape[1]:
        return torch.full((pred.shape[0],), float("nan"), device=pred.device)
    spatial = tuple(range(2, pred.ndim))
    m0 = u0[:, c:c+1].mean(dim=spatial).flatten(1)
    mp = pred[:, c:c+1].mean(dim=spatial).flatten(1)
    return (mp - m0).abs().squeeze(1) / (m0.abs().squeeze(1) + pred[:, c:c+1].abs().mean(dim=spatial).flatten(1).squeeze(1) + EPS)


def batch_tau(b: int, value: float, device: torch.device) -> torch.Tensor:
    return torch.full((b,), float(value), device=device)


METRIC_NAMES = [
    "rel_l2",
    "semigroup",
    "generator",
    "energy_violation",
    "energy_decay_error",
    "integral_balance",
    "constraint",
    "perturbation_response",
    "scaling_law",
]


def summarize_metric_rows(prefix: Dict[str, Any], rows: Dict[str, List[float]]) -> Dict[str, Any]:
    """Summarize metric samples. Empty/unavailable metrics are NaN with n=0.

    This avoids the dangerous convention of writing 0.0 for an optional
    diagnostic that was not actually computed.
    """
    out: Dict[str, Any] = dict(prefix)
    for name, vals in rows.items():
        arr = np.asarray(vals, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        out[name] = float(finite.mean()) if finite.size else float("nan")
        out[name + "_std"] = float(finite.std()) if finite.size else float("nan")
        out[name + "_n"] = int(finite.size)
    # Backward-compatible alias for earlier CSVs: `energy` means energy-law
    # violation, while `energy_decay_error` measures wrong dissipation rate.
    if "energy_violation" in out:
        out["energy"] = out["energy_violation"]
        out["energy_std"] = out.get("energy_violation_std", float("nan"))
        out["energy_n"] = out.get("energy_violation_n", 0)
    return out


def add_reference_relative_columns(rows: List[Dict[str, Any]]) -> None:
    """Add excess/ratio columns for diagnostics with nonzero reference baselines.

    Generator discrepancy is a finite-difference consistency diagnostic. Even for
    reference trajectories it need not be zero at finite stored time steps, so paper
    tables should use excess/ratio relative to the reference baseline instead of
    reading raw generator values as absolute errors.
    """
    by_task: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("model") == "reference" and row.get("variant") == "reference":
            by_task[str(row.get("task"))] = row
    for row in rows:
        ref = by_task.get(str(row.get("task")))
        if not ref:
            continue
        for metric in ["generator"]:
            val = row.get(metric, float("nan"))
            base = ref.get(metric, float("nan"))
            try:
                vf = float(val); bf = float(base)
            except Exception:
                vf = float("nan"); bf = float("nan")
            row[f"{metric}_reference"] = bf
            row[f"{metric}_excess"] = vf - bf if np.isfinite(vf) and np.isfinite(bf) else float("nan")
            row[f"{metric}_ratio_to_reference"] = vf / (bf + EPS) if np.isfinite(vf) and np.isfinite(bf) else float("nan")


@torch.no_grad()
def evaluate_diagnostics(
    model: nn.Module,
    window_loader: DataLoader,
    normalizer: Normalizer,
    task: TaskSpec,
    max_delta: int,
    device: torch.device,
    model_name: str,
    variant: str,
) -> Dict[str, float]:
    model.eval()
    norm = normalizer.to(device)
    rows: Dict[str, List[float]] = {k: [] for k in METRIC_NAMES}
    for seq in window_loader:
        seq = seq.to(device).float()  # B,T+1,C,*spatial
        bsz = seq.shape[0]
        u0 = seq[:, 0]
        # choose k=2 if possible for semigroup/generator, else k=1
        k = min(2, seq.shape[1] - 1, max_delta)
        if k < 1:
            continue
        tau_k = batch_tau(bsz, k / max_delta, device)
        pred_k_n = model(norm.encode(u0), tau_k)
        pred_k = norm.decode(pred_k_n)
        target_k = seq[:, k]
        rows["rel_l2"].extend(rel_l2(pred_k, target_k).detach().cpu().tolist())

        # semigroup with a=b=1 if k>=2
        if k >= 2:
            tau_1 = batch_tau(bsz, 1 / max_delta, device)
            mid_n = model(norm.encode(u0), tau_1)
            comp_n = model(mid_n, tau_1)
            direct_n = pred_k_n
            rows["semigroup"].extend(rel_l2(norm.decode(comp_n), norm.decode(direct_n)).detach().cpu().tolist())

            gen1 = (norm.decode(mid_n) - u0) / 1.0
            gen2 = (pred_k - u0) / float(k)
            num = torch.linalg.vector_norm((gen1 - gen2).flatten(1), dim=1)
            den = torch.linalg.vector_norm(gen1.flatten(1), dim=1) + torch.linalg.vector_norm(gen2.flatten(1), dim=1) + EPS
            rows["generator"].extend((num / den).detach().cpu().tolist())
        else:
            rows["semigroup"].append(float("nan")); rows["generator"].append(float("nan"))

        e0 = energy(u0, task)
        ep = energy(pred_k, task)
        er = energy(target_k, task)
        # Energy-law violation detects the wrong sign of energy change.
        # Energy-decay error detects wrong dissipation rate, including over-smoothing
        # that remains monotone but decays too aggressively.
        rows["energy_violation"].extend((torch.relu(ep - e0) / (e0.abs() + EPS)).detach().cpu().tolist())
        rows["energy_decay_error"].extend(((ep - er).abs() / (er.abs() + EPS)).detach().cpu().tolist())
        rows["integral_balance"].extend(integral_balance_error(u0, pred_k, task).detach().cpu().tolist())
        rows["constraint"].extend(constraint_error(pred_k, task).detach().cpu().tolist())

        # Perturbed references are not generally available in benchmark trajectory files.
        # If you later add solvers, replace this NaN with true finite-perturbation response error.
        rows["perturbation_response"].append(float("nan"))

        # Scaling-law diagnostics are only valid when an exact relation is specified.
        if task.scaling == "linear_amplitude":
            alpha = 1.25
            pred_scaled = norm.decode(model(norm.encode(alpha * u0), tau_k))
            rhs = alpha * pred_k
            rows["scaling_law"].extend(rel_l2(pred_scaled, rhs).detach().cpu().tolist())
        else:
            rows["scaling_law"].append(float("nan"))

    return summarize_metric_rows({"model": model_name, "variant": variant, "task": task.name}, rows)


@torch.no_grad()
def evaluate_reference(window_loader: DataLoader, task: TaskSpec, max_delta: int, device: torch.device) -> Dict[str, float]:
    rows: Dict[str, List[float]] = {k: [] for k in METRIC_NAMES}
    for seq in window_loader:
        seq = seq.to(device).float()
        if seq.shape[1] < 3:
            continue
        u0, u1, u2 = seq[:, 0], seq[:, 1], seq[:, 2]
        rows["rel_l2"].extend([0.0] * seq.shape[0])
        rows["semigroup"].extend([0.0] * seq.shape[0])
        gen1 = u1 - u0
        gen2 = (u2 - u0) / 2.0
        num = torch.linalg.vector_norm((gen1 - gen2).flatten(1), dim=1)
        den = torch.linalg.vector_norm(gen1.flatten(1), dim=1) + torch.linalg.vector_norm(gen2.flatten(1), dim=1) + EPS
        rows["generator"].extend((num / den).detach().cpu().tolist())
        e0, e1 = energy(u0, task), energy(u1, task)
        rows["energy_violation"].extend((torch.relu(e1 - e0) / (e0.abs() + EPS)).detach().cpu().tolist())
        rows["energy_decay_error"].extend([0.0] * seq.shape[0])
        rows["integral_balance"].extend(integral_balance_error(u0, u1, task).detach().cpu().tolist())
        rows["constraint"].extend(constraint_error(u1, task).detach().cpu().tolist())
    out = summarize_metric_rows({"model": "reference", "variant": "reference", "task": task.name}, rows)
    out["best_val"] = float("nan")
    out["best_epoch"] = float("nan")
    return out


# =============================================================================
# Training
# =============================================================================

def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: Normalizer,
    epochs: int,
    lr: float,
    device: torch.device,
    out_ckpt: Path,
) -> Dict[str, Any]:
    model.to(device)
    norm = normalizer.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = -1
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(device).float()
            y = batch["y"].to(device).float()
            tau = batch["tau"].to(device).float()
            xn = norm.encode(x)
            yn = norm.encode(y)
            pred = model(xn, tau)
            loss = F.mse_loss(pred, yn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device).float()
                y = batch["y"].to(device).float()
                tau = batch["tau"].to(device).float()
                pred = model(norm.encode(x), tau)
                val_losses.append(float(F.mse_loss(pred, norm.encode(y)).detach().cpu()))
        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        hist.append({"epoch": epoch, "train_mse": tr, "val_mse": va})
        if va < best_val:
            best_val = va
            best_epoch = epoch
            out_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "best_val": best_val, "best_epoch": best_epoch}, out_ckpt)
        print(f"    epoch {epoch:04d}/{epochs:04d} train_mse={tr:.4e} val_mse={va:.4e}")
    if out_ckpt.exists():
        ckpt = torch.load(out_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
    return {"best_val": best_val, "best_epoch": best_epoch, "history": hist}


# =============================================================================
# Main orchestration
# =============================================================================

def spatial_limit_for(task: TaskSpec, args: argparse.Namespace) -> int:
    return args.spatial_limit_2d if task.dim == 2 else args.spatial_limit_3d


def inspect_data(data_root: Path, tasks: Dict[str, TaskSpec], selected: List[str], args: argparse.Namespace, outdir: Path) -> None:
    rows = []
    for name in selected:
        task = tasks[name]
        for split in ["train", "val", "test"]:
            try:
                coll = TrajectoryCollection(data_root, task, split, spatial_limit_for(task, args))
                rows.extend(coll.describe())
            except Exception as e:
                rows.append({"task": name, "split": split, "error": str(e)})
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "data_inspection.csv", index=False)
    print(df.to_string(index=False))


def run_task(task: TaskSpec, args: argparse.Namespace, outdir: Path) -> List[Dict[str, Any]]:
    print(f"\n=== Task: {task.name} ({task.label}) ===")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    spat_lim = spatial_limit_for(task, args)
    train_coll = TrajectoryCollection(args.data_root, task, "train", spat_lim)
    val_coll = TrajectoryCollection(args.data_root, task, "val", spat_lim)
    test_coll = TrajectoryCollection(args.data_root, task, "test", spat_lim)

    train_ds = PairDataset(train_coll, args.max_train_pairs, args.max_delta, seed=args.seed + 1)
    val_ds = PairDataset(val_coll, args.max_val_pairs, args.max_delta, seed=args.seed + 2)
    test_windows = WindowDataset(test_coll, args.max_test_windows, horizon=max(2, args.max_delta), seed=args.seed + 3)

    sample = train_ds[0]["x"]
    channels = int(sample.shape[0])
    spatial_shape = tuple(int(x) for x in sample.shape[1:])
    print(f"  sample shape C,*spatial = {tuple(sample.shape)}; pairs train={len(train_ds)} val={len(val_ds)} test_windows={len(test_windows)}")

    norm = estimate_normalizer(train_ds, n_samples=min(args.norm_samples, len(train_ds)))
    (outdir / "normalizers").mkdir(parents=True, exist_ok=True)
    torch.save({"mean": norm.mean, "std": norm.std}, outdir / "normalizers" / f"{task.name}.pt")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_windows, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    rows: List[Dict[str, Any]] = []
    rows.append(evaluate_reference(test_loader, task, args.max_delta, device))

    for model_name in args.models:
        if model_name == "deeponet" and task.dim == 3 and not args.deeponet_3d:
            print(f"  [skip] deeponet on {task.name} because --deeponet-3d is not set")
            continue
        for variant, epochs, width_mult in [
            ("well", args.epochs_well, 1.0),
            ("underfit", args.epochs_underfit, args.underfit_width_mult),
        ]:
            width = max(4, int(args.width * width_mult))
            modes = min(args.modes, max(2, min(spatial_shape) // 2))
            print(f"  training {model_name}/{variant}: width={width}, modes={modes}, epochs={epochs}")
            model = build_model(model_name, task.dim, channels, spatial_shape, width=width, modes=modes)
            ckpt = outdir / "checkpoints" / task.name / f"{model_name}_{variant}.pt"
            meta = train_one(model, train_loader, val_loader, norm, epochs=epochs, lr=args.lr, device=device, out_ckpt=ckpt)
            metrics = evaluate_diagnostics(model, test_loader, norm, task, args.max_delta, device, model_name, variant)
            metrics.update({"best_val": meta["best_val"], "best_epoch": meta["best_epoch"]})
            rows.append(metrics)
            pd.DataFrame(meta["history"]).to_csv(outdir / "checkpoints" / task.name / f"{model_name}_{variant}_history.csv", index=False)

            if variant == "well" and args.eval_oversmoothed:
                smooth = OversmoothWrapper(model, task.dim, passes=args.oversmooth_passes).to(device)
                sm = evaluate_diagnostics(smooth, test_loader, norm, task, args.max_delta, device, model_name, "oversmoothed")
                sm.update({"best_val": meta["best_val"], "best_epoch": meta["best_epoch"]})
                rows.append(sm)

    add_reference_relative_columns(rows)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run diagnostic consistency experiments for learned PDE propagators.")
    p.add_argument("--data-root", type=Path, default=Path.home() / "Desktop/research_work/COMET/comet_ncs_codebase/data/raw")
    p.add_argument("--outdir", type=Path, default=Path("outputs_diagnostic_suite"))
    p.add_argument("--tasks", nargs="+", default=["ns2d", "shallow2d", "active2d", "cns3d", "mhd3d"], choices=["ns2d", "shallow2d", "active2d", "cns3d", "mhd3d"])
    p.add_argument("--models", nargs="+", default=["fno", "deeponet", "unet", "resnet"], choices=["fno", "deeponet", "unet", "resnet"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spatial-limit-2d", type=int, default=64)
    p.add_argument("--spatial-limit-3d", type=int, default=32)
    p.add_argument("--max-delta", type=int, default=4)
    p.add_argument("--max-train-pairs", type=int, default=2048)
    p.add_argument("--max-val-pairs", type=int, default=512)
    p.add_argument("--max-test-windows", type=int, default=256)
    p.add_argument("--norm-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--epochs-well", type=int, default=20)
    p.add_argument("--epochs-underfit", type=int, default=2)
    p.add_argument("--underfit-width-mult", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-oversmoothed", dest="eval_oversmoothed", action="store_true", default=True)
    p.add_argument("--no-eval-oversmoothed", dest="eval_oversmoothed", action="store_false")
    p.add_argument("--oversmooth-passes", type=int, default=2)
    p.add_argument("--deeponet-3d", action="store_true", help="Allow DeepONet on 3D data. This can be memory-heavy.")
    p.add_argument("--inspect-data", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny run for debugging data loading and training.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_train_pairs = min(args.max_train_pairs, 64)
        args.max_val_pairs = min(args.max_val_pairs, 32)
        args.max_test_windows = min(args.max_test_windows, 16)
        args.epochs_well = min(args.epochs_well, 2)
        args.epochs_underfit = 1
        args.width = min(args.width, 16)
        args.modes = min(args.modes, 6)
        args.spatial_limit_2d = min(args.spatial_limit_2d, 32)
        args.spatial_limit_3d = min(args.spatial_limit_3d, 16)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / "args.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    tasks = build_task_specs(args.data_root, args.spatial_limit_2d, args.spatial_limit_3d)
    if args.inspect_data:
        inspect_data(args.data_root, tasks, args.tasks, args, args.outdir)
        return

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for name in args.tasks:
        try:
            rows = run_task(tasks[name], args, args.outdir)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(args.outdir / "metrics_summary_partial.csv", index=False)
        except Exception as e:
            print(f"[ERROR] Task {name} failed: {e}")
            all_rows.append({"task": name, "model": "ERROR", "variant": "ERROR", "error": str(e)})
    df = pd.DataFrame(all_rows)
    df.to_csv(args.outdir / "metrics_summary.csv", index=False)
    print("\n=== Metrics summary ===")
    print(df.to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.2f} minutes. Outputs in {args.outdir}")


if __name__ == "__main__":
    main()
