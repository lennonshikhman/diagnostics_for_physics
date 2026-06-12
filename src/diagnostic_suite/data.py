from __future__ import annotations

import glob
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .tasks import TaskSpec

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


