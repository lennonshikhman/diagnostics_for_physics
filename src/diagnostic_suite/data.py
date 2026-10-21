from __future__ import annotations

import glob
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .tasks import TaskSpec

_AXIS_NAMES = ("x", "y", "z")
_COORDINATE_RTOL = 5.0e-5


def _decode_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _numeric_h5_datasets(path: Path) -> List[Tuple[str, Tuple[int, ...], str]]:
    out: List[Tuple[str, Tuple[int, ...], str]] = []
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            try:
                if len(obj.shape) >= 1 and np.issubdtype(obj.dtype, np.number):
                    out.append((name, tuple(int(x) for x in obj.shape), str(obj.dtype)))
            except (TypeError, ValueError):
                return

        handle.visititems(visit)
    return out


def _choose_h5_dataset(path: Path, dim: int) -> str:
    """Choose a generic PDEBench trajectory dataset, never a Well field.

    The Well has a strict reader below.  This fallback is retained only for older
    PDEBench files whose trajectory dataset names vary by release.
    """

    candidates = _numeric_h5_datasets(path)
    ranked: List[Tuple[int, str, Tuple[int, ...]]] = []
    for name, shape, _dtype in candidates:
        if len(shape) < dim + 1:
            continue
        lname = name.lower()
        score = 0
        if lname in {"data", "solution", "tensor", "trajectory", "states"}:
            score += 200
        if lname.endswith(("/data", "/solution", "/tensor")):
            score += 150
        if len(shape) >= dim + 2:
            score += 50
        if len(shape) >= dim + 3:
            score += 25
        if any(token in lname for token in ("coord", "grid", "time", "param", "mask")):
            score -= 500
        score += min(20, int(math.log10(max(1, int(np.prod(shape))))))
        ranked.append((score, name, shape))
    if not ranked:
        raise RuntimeError(f"No numeric trajectory dataset found in {path}")
    ranked.sort(reverse=True)
    return ranked[0][1]


def _strict_h5_dataset(path: Path, allowed_keys: Sequence[str]) -> str:
    available = {name for name, _shape, _dtype in _numeric_h5_datasets(path)}
    matches = [key for key in allowed_keys if key in available]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one configured trajectory dataset {tuple(allowed_keys)} in {path}; matches={matches}"
        )
    return matches[0]


def _nested_h5_trajectory_keys(path: Path, dim: int) -> List[Tuple[str, Tuple[int, ...]]]:
    out: List[Tuple[str, Tuple[int, ...]]] = []
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            try:
                shape = tuple(int(x) for x in obj.shape)
                lname = name.lower()
                if not np.issubdtype(obj.dtype, np.number):
                    return
                if not lname.endswith(("/data", "/tensor", "/solution")):
                    return
                if len(shape) in {dim + 1, dim + 2} and shape[0] > 1:
                    out.append((name, shape))
            except (TypeError, ValueError):
                return

        handle.visititems(visit)
    out.sort(key=lambda item: item[0])
    return out


def _top_level_cfd_keys(path: Path, dim: int) -> Tuple[List[str], Optional[Tuple[int, ...]]]:
    aliases = [
        ("density", ("density",)),
        ("pressure", ("pressure",)),
        ("velocity_x", ("Vx", "vx", "velocity_x", "x-velocity")),
        ("velocity_y", ("Vy", "vy", "velocity_y", "y-velocity")),
        ("velocity_z", ("Vz", "vz", "velocity_z", "z-velocity")),
    ]
    keys: List[str] = []
    shape: Optional[Tuple[int, ...]] = None
    with h5py.File(path, "r") as handle:
        for canonical, choices in aliases:
            key = next((candidate for candidate in choices if candidate in handle), None)
            if key is None:
                raise RuntimeError(f"Missing required CNS field {canonical!r} in {path}")
            obj = handle[key]
            if not isinstance(obj, h5py.Dataset) or not np.issubdtype(obj.dtype, np.number):
                raise RuntimeError(f"CNS field {key!r} is not a numeric dataset in {path}")
            current = tuple(int(x) for x in obj.shape)
            if len(current) != dim + 2:
                raise RuntimeError(f"Unexpected shape for CNS field {key!r}: {current}")
            if shape is None:
                shape = current
            elif shape != current:
                raise RuntimeError(f"CNS fields do not share a shape in {path}: {shape} versus {current}")
            keys.append(key)
    return keys, shape


def _to_channels_first_state(arr: np.ndarray, dim: int, max_channels: Optional[int]) -> torch.Tensor:
    x = np.asarray(arr, dtype=np.float32)
    while x.ndim > dim + 1 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == dim:
        x = x[..., None]
    if x.ndim != dim + 1:
        if x.ndim < dim:
            raise ValueError(f"State has too few axes for {dim}D data: {x.shape}")
        spatial = x.shape[:dim]
        x = x.reshape(*spatial, -1)

    cap = int(max_channels) if max_channels is not None else math.inf
    first_is_channel = x.shape[0] <= cap and all(n > cap for n in x.shape[1:])
    last_is_channel = x.shape[-1] <= cap
    if last_is_channel and not first_is_channel:
        x = np.moveaxis(x, -1, 0)
    elif first_is_channel:
        pass
    elif x.shape[-1] <= 64:
        x = np.moveaxis(x, -1, 0)
    else:
        raise ValueError(f"Cannot determine channel axis for state shape {x.shape}")
    if max_channels is not None and x.shape[0] > max_channels:
        raise ValueError(f"Found {x.shape[0]} channels, expected at most {max_channels}")
    return torch.from_numpy(np.array(x, copy=True, order="C"))


def _spatial_limit(
    x: torch.Tensor, dim: int, limit: int, *, periodic: bool = False
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if x.ndim != dim + 1:
        raise ValueError(f"Expected (C,*spatial) with dim={dim}; got {tuple(x.shape)}")
    if limit <= 0:
        return x.contiguous(), tuple(1 for _ in range(dim))
    slices: List[slice] = [slice(None)]
    steps: List[int] = []
    for n in x.shape[1:]:
        step = max(1, int(math.ceil(int(n) / limit)))
        if periodic and int(n) % step:
            # A nondividing stride leaves a shorter gap across the periodic
            # seam, invalidating the uniform spacing used by physical metrics.
            candidates = [value for value in range(step, int(n) // 2 + 1) if int(n) % value == 0]
            if not candidates:
                raise ValueError(
                    f"Cannot uniformly subsample periodic size {n} to limit {limit}; "
                    "use spatial limit 0 to retain the native grid or resample externally"
                )
            step = candidates[0]
        steps.append(step)
        slices.append(slice(0, int(n), step))
    return x[tuple(slices)].contiguous(), tuple(steps)


def _infer_h5_spacing(path: Path, spatial_shape: Sequence[int], dim: int) -> Tuple[float, ...]:
    candidates: Dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 1:
                return
            try:
                if np.issubdtype(obj.dtype, np.number):
                    candidates[name.lower()] = np.asarray(obj[:], dtype=np.float64)
            except (OSError, TypeError, ValueError):
                return

        handle.visititems(visit)
    spacings: List[float] = []
    for axis, expected in zip(_AXIS_NAMES[:dim], spatial_shape, strict=True):
        matches = [
            values
            for name, values in candidates.items()
            if values.size == int(expected)
            and (
                name.endswith(f"/{axis}")
                or name == axis
                or f"{axis}-coordinate" in name
                or f"{axis}_coordinate" in name
            )
        ]
        if not matches:
            return ()
        axis_spacings: List[float] = []
        for coordinate in matches:
            diffs = np.diff(coordinate)
            if not diffs.size or not np.all(np.isfinite(diffs)):
                return ()
            absolute_diffs = np.abs(diffs)
            spacing = float(np.median(absolute_diffs))
            # Coordinate arrays are commonly stored as float32.  On long
            # linspace grids their quantization error can slightly exceed
            # 1e-5 relative to the cell width even when the source grid is
            # uniform (for example, The Well active-matter coordinates).
            if not np.allclose(absolute_diffs, spacing, rtol=_COORDINATE_RTOL, atol=1.0e-12):
                raise RuntimeError(f"Nonuniform {axis}-coordinates in {path} are unsupported")
            axis_spacings.append(spacing)
        if not np.allclose(axis_spacings, axis_spacings[0], rtol=_COORDINATE_RTOL, atol=1.0e-12):
            raise RuntimeError(f"Inconsistent {axis}-coordinate grids in {path}")
        spacings.append(axis_spacings[0])
    return tuple(spacings)


def _infer_h5_time_step(path: Path, n_time: int) -> Optional[float]:
    with h5py.File(path, "r") as handle:
        found: List[np.ndarray] = []

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 1:
                return
            lname = name.lower()
            is_time_coordinate = (
                "time" in lname
                or lname.endswith("/t")
                or lname == "t"
                or "t-coordinate" in lname
                or "t_coordinate" in lname
            )
            if not is_time_coordinate:
                return
            try:
                values = np.asarray(obj[:], dtype=np.float64)
                # Some PDEBench CFD files store time-cell boundaries, giving
                # one more coordinate than stored states.  Either convention
                # determines the same step on a uniform grid.
                if values.size in {n_time, n_time + 1}:
                    found.append(values)
            except (OSError, TypeError, ValueError):
                return

        handle.visititems(visit)
    if not found:
        return None
    time_steps: List[float] = []
    for coordinate in found:
        diffs = np.diff(coordinate)
        if not diffs.size or not np.all(np.isfinite(diffs)):
            return None
        absolute_diffs = np.abs(diffs)
        time_step = float(np.median(absolute_diffs))
        if not np.allclose(absolute_diffs, time_step, rtol=_COORDINATE_RTOL, atol=1.0e-12):
            raise RuntimeError(f"Nonuniform time coordinates in {path} are unsupported")
        time_steps.append(time_step)
    if not np.allclose(time_steps, time_steps[0], rtol=_COORDINATE_RTOL, atol=1.0e-12):
        raise RuntimeError(f"Inconsistent time coordinates in {path}")
    return time_steps[0]


def _partition_indices(n_traj: int, split: str, strategy: str, seed: int) -> List[int]:
    if n_traj <= 0:
        return []
    if strategy == "explicit":
        return list(range(n_traj))
    indices = list(range(n_traj))
    random.Random(seed).shuffle(indices)
    if strategy == "train_val":
        if split == "test":
            return list(range(n_traj))
        if n_traj < 2:
            raise RuntimeError("A disjoint train/validation split requires at least two trajectories")
        train_end = min(n_traj - 1, max(1, int(math.floor(0.9 * n_traj))))
        selected = indices[:train_end] if split == "train" else indices[train_end:]
        return sorted(selected)
    if strategy == "train_val_test":
        if n_traj < 3:
            raise RuntimeError("A disjoint train/validation/test split requires at least three trajectories")
        train_end = min(n_traj - 2, max(1, int(math.floor(0.8 * n_traj))))
        val_end = min(n_traj - 1, max(train_end + 1, int(math.floor(0.9 * n_traj))))
        if split == "train":
            return sorted(indices[:train_end])
        if split == "val":
            return sorted(indices[train_end:val_end])
        return sorted(indices[val_end:])
    raise ValueError(f"Unknown split strategy {strategy!r}")


class BaseReader:
    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int):
        self.path = Path(path)
        self.task = task
        self.split = "val" if split in {"valid", "validation"} else split
        self.spatial_limit = spatial_limit
        self.n_traj = 0
        self.n_time = 0
        self.channel_names: Tuple[str, ...] = ()
        self.spatial_shape: Tuple[int, ...] = ()
        self.grid_spacing: Tuple[float, ...] = ()
        self.time_step: Optional[float] = task.time_step
        self.grid_spacing_source = (
            "configured" if task.grid_spacing else "domain_extent" if task.domain_extent else "default_unit"
        )
        self.time_step_source = "configured" if task.time_step is not None else "missing"

    def _raw_state(self, traj: int, time_index: int) -> torch.Tensor:
        raise NotImplementedError

    def _base_grid_spacing(self, raw_spatial_shape: Sequence[int]) -> Tuple[float, ...]:
        if self.task.grid_spacing:
            return tuple(float(x) for x in self.task.grid_spacing)
        if self.task.domain_extent:
            spacing: List[float] = []
            for extent, n in zip(self.task.domain_extent, raw_spatial_shape, strict=True):
                denom = int(n) if self.task.periodic else max(1, int(n) - 1)
                spacing.append(float(extent) / denom)
            return tuple(spacing)
        return tuple(1.0 for _ in raw_spatial_shape)

    def _finish_metadata(self) -> None:
        if self.n_traj < 1 or self.n_time < 2:
            raise RuntimeError(f"Reader found an invalid trajectory shape in {self.path}")
        raw = self._raw_state(0, 0)
        limited, steps = _spatial_limit(raw, self.task.dim, self.spatial_limit, periodic=self.task.periodic)
        self.spatial_shape = tuple(int(n) for n in limited.shape[1:])
        if not self.channel_names:
            if self.task.channel_names and len(self.task.channel_names) == limited.shape[0]:
                self.channel_names = tuple(self.task.channel_names)
            else:
                self.channel_names = tuple(f"channel_{i}" for i in range(int(limited.shape[0])))
        if len(self.channel_names) != limited.shape[0]:
            raise RuntimeError(
                f"Channel schema mismatch in {self.path}: names={self.channel_names}, "
                f"state shape={tuple(limited.shape)}"
            )
        self.task.validate_state_channels(int(limited.shape[0]))
        base = self._base_grid_spacing(raw.shape[1:])
        self.grid_spacing = tuple(float(dx) * step for dx, step in zip(base, steps, strict=True))

    def state(self, traj: int, time_index: int) -> torch.Tensor:
        state, _steps = _spatial_limit(
            self._raw_state(traj, time_index),
            self.task.dim,
            self.spatial_limit,
            periodic=self.task.periodic,
        )
        return state

    def traj_indices_for_split(self) -> List[int]:
        return _partition_indices(
            self.n_traj,
            self.split,
            self.task.split_strategy,
            self.task.split_seed,
        )


class NpyReader(BaseReader):
    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int):
        super().__init__(path, task, split, spatial_limit)
        loaded = np.load(path, allow_pickle=False, mmap_mode="r")
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = sorted(key for key in loaded.files if np.asarray(loaded[key]).ndim >= task.dim + 2)
            if not keys:
                raise RuntimeError(f"No trajectory array found in {path}")
            arr = np.asarray(loaded[keys[0]])
            loaded.close()
        else:
            arr = np.asarray(loaded)
        self.arr = arr
        if self.arr.ndim == task.dim + 2:
            self.arr = self.arr[None]
        if self.arr.ndim < task.dim + 3:
            raise RuntimeError(f"Expected (N,T,*spatial,C) or (N,T,C,*spatial), got {self.arr.shape}")
        self.n_traj = int(self.arr.shape[0])
        self.n_time = int(self.arr.shape[1])
        self.channel_names = tuple(task.channel_names)
        self._finish_metadata()

    def _raw_state(self, traj: int, time_index: int) -> torch.Tensor:
        return _to_channels_first_state(
            self.arr[traj, time_index],
            self.task.dim,
            self.task.max_channels,
        )


class H5Reader(BaseReader):
    """Strict reader for PDEBench-style generic or top-level CFD HDF5 files."""

    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int):
        super().__init__(path, task, split, spatial_limit)
        self._h5: Optional[h5py.File] = None
        self.mode = "batched"
        self.dataset_key: Optional[str] = None
        self.dataset_keys: List[str] = []
        self.shape: Tuple[int, ...]

        if task.reader_kind == "cfd":
            self.mode = "cfd"
            keys, shape = _top_level_cfd_keys(self.path, task.dim)
            assert shape is not None
            self.dataset_keys = keys
            self.shape = shape
            self.n_traj, self.n_time = int(shape[0]), int(shape[1])
            self.channel_names = tuple(task.channel_names)
        else:
            nested = _nested_h5_trajectory_keys(self.path, task.dim)
            if nested:
                self.mode = "nested"
                self.dataset_keys = [key for key, _shape in nested]
                shapes = {shape for _key, shape in nested}
                if len(shapes) != 1:
                    raise RuntimeError(f"Nested trajectories have inconsistent shapes in {path}: {shapes}")
                self.shape = nested[0][1]
                self.n_traj = len(self.dataset_keys)
                self.n_time = int(self.shape[0])
            else:
                self.dataset_key = (
                    _strict_h5_dataset(self.path, task.trajectory_dataset_keys)
                    if task.trajectory_dataset_keys
                    else _choose_h5_dataset(self.path, task.dim)
                )
                with h5py.File(self.path, "r") as handle:
                    self.shape = tuple(int(x) for x in handle[self.dataset_key].shape)
                if len(self.shape) == task.dim + 1:
                    self.n_traj, self.n_time = 1, int(self.shape[0])
                else:
                    self.n_traj, self.n_time = int(self.shape[0]), int(self.shape[1])
            self.channel_names = tuple(task.channel_names)

        raw = self._raw_state(0, 0)
        inferred_spacing = _infer_h5_spacing(self.path, raw.shape[1:], task.dim)
        if inferred_spacing:
            self.task = replace(task, grid_spacing=inferred_spacing)
            self.grid_spacing_source = "file_coordinates"
        inferred_dt = _infer_h5_time_step(self.path, self.n_time)
        if inferred_dt is not None:
            self.time_step = inferred_dt
            self.time_step_source = "file_coordinates"
        self._finish_metadata()

    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def _raw_state(self, traj: int, time_index: int) -> torch.Tensor:
        handle = self.h5()
        if self.mode == "cfd":
            arrays = [np.asarray(handle[key][traj, time_index], dtype=np.float32) for key in self.dataset_keys]
            return torch.from_numpy(np.ascontiguousarray(np.stack(arrays, axis=0)))
        if self.mode == "nested":
            key = self.dataset_keys[traj]
            arr = np.asarray(handle[key][time_index], dtype=np.float32)
        else:
            assert self.dataset_key is not None
            dataset = handle[self.dataset_key]
            if len(self.shape) == self.task.dim + 1:
                arr = np.asarray(dataset[time_index], dtype=np.float32)
            else:
                arr = np.asarray(dataset[traj, time_index], dtype=np.float32)
        return _to_channels_first_state(arr, self.task.dim, self.task.max_channels)

    def __del__(self) -> None:
        handle = getattr(self, "_h5", None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, TypeError):
                pass


class WellH5Reader(BaseReader):
    """Schema-aware reader for the official The Well HDF5 format.

    It assembles all configured scalar/vector/tensor fields instead of choosing a
    single numeric dataset.  Field names and component ordering become explicit
    channel metadata used by the physical diagnostics.
    """

    _GROUPS = ("t0_fields", "t1_fields", "t2_fields")

    def __init__(self, path: Path, task: TaskSpec, split: str, spatial_limit: int):
        super().__init__(path, task, split, spatial_limit)
        self._h5: Optional[h5py.File] = None
        self.mode = "well"
        self.fields: List[Tuple[str, str]] = []
        self._sample_varying: Dict[Tuple[str, str], bool] = {}
        self._time_varying: Dict[Tuple[str, str], bool] = {}

        with h5py.File(path, "r") as handle:
            for group_name in self._GROUPS:
                if group_name not in handle:
                    if group_name in task.well_fields:
                        raise RuntimeError(f"Missing required Well group {group_name!r} in {path}")
                    continue
                group = handle[group_name]
                if not isinstance(group, h5py.Group):
                    raise RuntimeError(f"Expected HDF5 group {group_name!r} in {path}")
                if "field_names" not in group.attrs:
                    raise RuntimeError(f"Missing official Well field_names metadata in {path}/{group_name}")
                declared = _decode_strings(group.attrs["field_names"])
                requested = list(task.well_fields.get(group_name, ()))
                selected = requested or declared
                missing = [name for name in selected if name not in group]
                if missing:
                    raise RuntimeError(f"Missing required Well fields {missing} in {path}/{group_name}")
                for field_name in selected:
                    dataset = group[field_name]
                    if not isinstance(dataset, h5py.Dataset) or not np.issubdtype(dataset.dtype, np.number):
                        raise RuntimeError(f"Well field {group_name}/{field_name} is not numeric")
                    missing_attributes = [
                        name for name in ("sample_varying", "time_varying") if name not in dataset.attrs
                    ]
                    if missing_attributes:
                        raise RuntimeError(
                            f"Well field {group_name}/{field_name} is missing metadata {missing_attributes}"
                        )
                    key = (group_name, field_name)
                    self.fields.append(key)
                    self._sample_varying[key] = bool(dataset.attrs["sample_varying"])
                    self._time_varying[key] = bool(dataset.attrs["time_varying"])

            if not self.fields:
                raise RuntimeError(f"No time-varying Well fields found in {path}")

            n_traj_values: List[int] = []
            n_time_values: List[int] = []
            for key in self.fields:
                dataset = handle[key[0]][key[1]]
                cursor = 0
                if self._sample_varying[key]:
                    n_traj_values.append(int(dataset.shape[cursor]))
                    cursor += 1
                if self._time_varying[key]:
                    n_time_values.append(int(dataset.shape[cursor]))
            self.n_traj = n_traj_values[0] if n_traj_values else 1
            self.n_time = n_time_values[0] if n_time_values else 1
            if any(value != self.n_traj for value in n_traj_values):
                raise RuntimeError(f"Well fields disagree on trajectory count in {path}: {n_traj_values}")
            if any(value != self.n_time for value in n_time_values):
                raise RuntimeError(f"Well fields disagree on time count in {path}: {n_time_values}")

        self.channel_names = self._discover_channel_names()
        if task.channel_names and tuple(task.channel_names) != self.channel_names:
            raise RuntimeError(
                f"Well channel schema mismatch in {path}: expected {task.channel_names}, found {self.channel_names}"
            )
        if task.max_channels is not None and len(self.channel_names) > task.max_channels:
            raise RuntimeError(
                f"Well state has {len(self.channel_names)} channels, configured maximum is {task.max_channels}"
            )
        raw = self._raw_state(0, 0)
        inferred_spacing = _infer_h5_spacing(self.path, raw.shape[1:], task.dim)
        if inferred_spacing:
            self.task = replace(task, grid_spacing=inferred_spacing)
            self.grid_spacing_source = "file_coordinates"
        inferred_dt = _infer_h5_time_step(self.path, self.n_time)
        if inferred_dt is not None:
            self.time_step = inferred_dt
            self.time_step_source = "file_coordinates"
        self._finish_metadata()

    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def _field_array(self, group_name: str, field_name: str, traj: int, time_index: int) -> np.ndarray:
        key = (group_name, field_name)
        dataset = self.h5()[group_name][field_name]
        index: List[Any] = []
        if self._sample_varying[key]:
            index.append(traj)
        if self._time_varying[key]:
            index.append(time_index)
        index.extend(slice(None) for _ in range(dataset.ndim - len(index)))
        return np.asarray(dataset[tuple(index)], dtype=np.float32)

    def _field_channels(self, group_name: str, field_name: str, traj: int, time_index: int) -> torch.Tensor:
        arr = self._field_array(group_name, field_name, traj, time_index)
        if arr.ndim < self.task.dim:
            raise RuntimeError(f"Well field {group_name}/{field_name} has invalid state shape {arr.shape}")
        spatial = arr.shape[: self.task.dim]
        components = int(np.prod(arr.shape[self.task.dim :])) if arr.ndim > self.task.dim else 1
        reshaped = arr.reshape(*spatial, components)
        return torch.from_numpy(np.ascontiguousarray(np.moveaxis(reshaped, -1, 0)))

    def _discover_channel_names(self) -> Tuple[str, ...]:
        names: List[str] = []
        for group_name, field_name in self.fields:
            count = int(self._field_channels(group_name, field_name, 0, 0).shape[0])
            if count == 1:
                names.append(field_name)
                continue
            if group_name == "t1_fields" and count == self.task.dim:
                names.extend(f"{field_name}_{axis}" for axis in _AXIS_NAMES[: self.task.dim])
                continue
            if group_name == "t2_fields" and count == self.task.dim * self.task.dim:
                names.extend(
                    f"{field_name}_{a}{b}" for a in _AXIS_NAMES[: self.task.dim] for b in _AXIS_NAMES[: self.task.dim]
                )
                continue
            names.extend(f"{field_name}_{index}" for index in range(count))
        return tuple(names)

    def _raw_state(self, traj: int, time_index: int) -> torch.Tensor:
        channels = [self._field_channels(group, field, traj, time_index) for group, field in self.fields]
        spatial_shapes = {tuple(tensor.shape[1:]) for tensor in channels}
        if len(spatial_shapes) != 1:
            raise RuntimeError(f"Well fields have incompatible spatial shapes in {self.path}: {spatial_shapes}")
        return torch.cat(channels, dim=0)

    def __del__(self) -> None:
        handle = getattr(self, "_h5", None)
        if handle is not None:
            try:
                handle.close()
            except (OSError, TypeError):
                pass


class TrajectoryCollection:
    def __init__(self, data_root: Path, task: TaskSpec, split: str, spatial_limit: int):
        self.data_root = Path(data_root)
        self.task = task
        self.split = "val" if split in {"valid", "validation"} else split
        self.spatial_limit = spatial_limit
        self.readers: List[BaseReader] = []
        matched: List[Path] = []
        for pattern in task.globs[self.split]:
            matched.extend(Path(path) for path in sorted(glob.glob(str(self.data_root / pattern))))
        if not matched:
            raise RuntimeError(f"No files for task={task.name}, split={self.split}, patterns={task.globs[self.split]}")
        for path in matched:
            suffix = path.suffix.lower()
            if task.reader_kind == "well":
                reader: BaseReader = WellH5Reader(path, task, self.split, spatial_limit)
            elif suffix in {".npy", ".npz"}:
                reader = NpyReader(path, task, self.split, spatial_limit)
            elif suffix in {".h5", ".hdf5", ".hdf"}:
                reader = H5Reader(path, task, self.split, spatial_limit)
            else:
                raise RuntimeError(f"Unsupported trajectory file {path}")
            self.readers.append(reader)
        self._validate_reader_metadata()

    def _validate_reader_metadata(self) -> None:
        channel_schemas = {reader.channel_names for reader in self.readers}
        spatial_shapes = {reader.spatial_shape for reader in self.readers}
        if len(channel_schemas) != 1:
            raise RuntimeError(f"Files disagree on channel schema for {self.task.name}: {channel_schemas}")
        if len(spatial_shapes) != 1:
            raise RuntimeError(f"Files disagree on spatial shape for {self.task.name}: {spatial_shapes}")
        reference_spacing = self.readers[0].grid_spacing
        reference_time_step = self.readers[0].time_step
        for reader in self.readers:
            if not reader.traj_indices_for_split():
                raise RuntimeError(f"Empty {self.split} partition for {reader.path}")
            if not np.allclose(
                reader.grid_spacing,
                reference_spacing,
                rtol=1.0e-6,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"Files disagree on grid spacing for {self.task.name}: "
                    f"{reference_spacing} versus {reader.grid_spacing}"
                )
            if reader.time_step is None or reference_time_step is None:
                if reader.time_step != reference_time_step:
                    raise RuntimeError(f"Files disagree on time step for {self.task.name}")
            elif not np.isclose(
                reader.time_step,
                reference_time_step,
                rtol=1.0e-6,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"Files disagree on time step for {self.task.name}: {reference_time_step} versus {reader.time_step}"
                )

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return self.readers[0].channel_names

    @property
    def grid_spacing(self) -> Tuple[float, ...]:
        return self.readers[0].grid_spacing

    @property
    def time_step(self) -> Optional[float]:
        return self.readers[0].time_step

    def runtime_task(self) -> TaskSpec:
        return self.task.with_discretization(
            grid_spacing=self.grid_spacing,
            time_step=self.time_step,
            channel_names=self.channel_names,
        )

    def split_identities(self) -> set[Tuple[str, int]]:
        return {
            (str(reader.path.resolve()), trajectory)
            for reader in self.readers
            for trajectory in reader.traj_indices_for_split()
        }

    def describe(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for reader in self.readers:
            selected = reader.traj_indices_for_split()
            rows.append(
                {
                    "task": self.task.name,
                    "split": self.split,
                    "path": str(reader.path),
                    "reader": type(reader).__name__,
                    "mode": getattr(reader, "mode", "npy"),
                    "n_traj_total": reader.n_traj,
                    "n_traj_selected": len(selected),
                    "split_strategy": self.task.split_strategy,
                    "split_seed": self.task.split_seed,
                    "trajectory_indices": ",".join(str(index) for index in selected),
                    "n_time": reader.n_time,
                    "spatial_shape": str(reader.spatial_shape),
                    "channel_names": ",".join(reader.channel_names),
                    "grid_spacing": str(reader.grid_spacing),
                    "grid_spacing_source": reader.grid_spacing_source,
                    "time_step": reader.time_step,
                    "time_step_source": reader.time_step_source,
                    "dataset_keys": ",".join(getattr(reader, "dataset_keys", []))
                    or str(getattr(reader, "dataset_key", "")),
                }
            )
        return rows


def assert_disjoint_collections(
    train: TrajectoryCollection,
    validation: TrajectoryCollection,
    test: TrajectoryCollection,
) -> None:
    """Fail before training if any physical trajectory crosses split boundaries."""

    identities = {
        "train": train.split_identities(),
        "validation": validation.split_identities(),
        "test": test.split_identities(),
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = identities[left] & identities[right]
        if overlap:
            preview = sorted(overlap)[:5]
            raise RuntimeError(
                f"Data leakage for task={train.task.name}: {left}/{right} share "
                f"{len(overlap)} trajectories; examples={preview}"
            )


class CachedDataset(Dataset):
    """Materialize only an already-selected dataset without changing its order.

    Selection and leakage checks belong to the original collection. Dictionary
    values and trajectory windows retain their tensor values and metadata.
    """

    def __init__(self, dataset: Dataset):
        self.values = [dataset[index] for index in range(len(dataset))]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        return self.values[index]


class PairDataset(Dataset):
    def __init__(self, collection: TrajectoryCollection, max_pairs: int, max_delta: int, seed: int = 0):
        self.collection = collection
        self.max_delta = max_delta
        self.items: List[Tuple[int, int, int, int]] = []
        for reader_index, reader in enumerate(collection.readers):
            for trajectory in reader.traj_indices_for_split():
                for start in range(0, reader.n_time - 1):
                    max_valid_delta = min(max_delta, reader.n_time - 1 - start)
                    for delta in range(1, max_valid_delta + 1):
                        self.items.append((reader_index, trajectory, start, delta))
        rng = random.Random(seed)
        rng.shuffle(self.items)
        if max_pairs > 0:
            self.items = self.items[:max_pairs]
        if not self.items:
            raise RuntimeError(f"No training pairs for {collection.task.name}/{collection.split}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        reader_index, trajectory, start, delta = self.items[index]
        reader = self.collection.readers[reader_index]
        physical_dt = float(delta) * float(reader.time_step or 1.0)
        return {
            "x": reader.state(trajectory, start),
            "y": reader.state(trajectory, start + delta),
            "tau": torch.tensor(float(delta) / max(1, self.max_delta), dtype=torch.float32),
            "dt_index": torch.tensor(delta, dtype=torch.long),
            "physical_dt": torch.tensor(physical_dt, dtype=torch.float32),
        }


class WindowDataset(Dataset):
    def __init__(self, collection: TrajectoryCollection, max_windows: int, horizon: int, seed: int = 0):
        self.collection = collection
        self.horizon = horizon
        self.items: List[Tuple[int, int, int]] = []
        for reader_index, reader in enumerate(collection.readers):
            if reader.n_time <= horizon:
                raise RuntimeError(
                    f"Reader {reader.path} has {reader.n_time} states but horizon={horizon} "
                    f"requires at least {horizon + 1}"
                )
            for trajectory in reader.traj_indices_for_split():
                for start in range(0, max(0, reader.n_time - horizon)):
                    self.items.append((reader_index, trajectory, start))
        rng = random.Random(seed)
        rng.shuffle(self.items)
        if max_windows > 0:
            self.items = self.items[:max_windows]
        if not self.items:
            raise RuntimeError(f"No windows for {collection.task.name}/{collection.split}; horizon={horizon}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> torch.Tensor:
        reader_index, trajectory, start = self.items[index]
        reader = self.collection.readers[reader_index]
        states = [reader.state(trajectory, start + offset) for offset in range(self.horizon + 1)]
        return torch.stack(states, dim=0)


__all__ = [
    "BaseReader",
    "H5Reader",
    "NpyReader",
    "PairDataset",
    "TrajectoryCollection",
    "WellH5Reader",
    "WindowDataset",
    "assert_disjoint_collections",
]
