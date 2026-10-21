from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from diagnostic_suite.data import (
    TrajectoryCollection,
    WellH5Reader,
    _infer_h5_spacing,
    assert_disjoint_collections,
)
from diagnostic_suite.tasks import TaskSpec, build_task_specs


def _write_npy(path: Path, trajectories: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((trajectories, 4, 6, 6, 1), dtype=np.float32)
    for trajectory in range(trajectories):
        values[trajectory] = trajectory
    np.save(path, values)


def test_shared_files_are_partitioned_into_disjoint_train_validation_test_sets(
    tmp_path: Path,
) -> None:
    _write_npy(tmp_path / "shared.npy", trajectories=10)
    task = TaskSpec(
        name="shared",
        label="shared",
        dim=2,
        globs={
            "train": ["shared.npy"],
            "val": ["shared.npy"],
            "test": ["shared.npy"],
        },
        reader_kind="npy",
        split_strategy="train_val_test",
        max_channels=1,
        channel_names=("scalar",),
    )
    train = TrajectoryCollection(tmp_path, task, "train", spatial_limit=0)
    validation = TrajectoryCollection(tmp_path, task, "val", spatial_limit=0)
    test = TrajectoryCollection(tmp_path, task, "test", spatial_limit=0)
    assert len(train.split_identities()) == 8
    assert len(validation.split_identities()) == 1
    assert len(test.split_identities()) == 1
    assert_disjoint_collections(train, validation, test)


def test_explicit_same_file_across_splits_is_rejected_as_leakage(tmp_path: Path) -> None:
    _write_npy(tmp_path / "shared.npy", trajectories=3)
    task = TaskSpec(
        name="leaky",
        label="leaky",
        dim=2,
        globs={
            "train": ["shared.npy"],
            "val": ["shared.npy"],
            "test": ["shared.npy"],
        },
        reader_kind="npy",
        split_strategy="explicit",
        max_channels=1,
        channel_names=("scalar",),
    )
    collections = [TrajectoryCollection(tmp_path, task, split, spatial_limit=0) for split in ("train", "val", "test")]
    with pytest.raises(RuntimeError, match="Data leakage"):
        assert_disjoint_collections(*collections)


def test_advection_training_and_validation_partitions_are_disjoint(tmp_path: Path) -> None:
    _write_npy(
        tmp_path / "apebench/apebench_incompressible_ns_2d/data/2d_diff_adv__num_spatial_dims=2_train.npy",
        trajectories=10,
    )
    _write_npy(
        tmp_path / "apebench/apebench_incompressible_ns_2d/data/2d_diff_adv__num_spatial_dims=2_test.npy",
        trajectories=4,
    )
    task = build_task_specs(tmp_path, 64, 32)["adv2d"]
    train = TrajectoryCollection(tmp_path, task, "train", spatial_limit=0)
    validation = TrajectoryCollection(tmp_path, task, "val", spatial_limit=0)
    test = TrajectoryCollection(tmp_path, task, "test", spatial_limit=0)
    assert train.split_identities().isdisjoint(validation.split_identities())
    assert_disjoint_collections(train, validation, test)


def _write_cns(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (10, 3, 4, 4, 4)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("density", data=np.ones(shape, dtype=np.float32))
        handle.create_dataset("pressure", data=np.ones(shape, dtype=np.float32))
        handle.create_dataset("Vx", data=np.zeros(shape, dtype=np.float32))
        handle.create_dataset("Vy", data=np.zeros(shape, dtype=np.float32))
        handle.create_dataset("Vz", data=np.zeros(shape, dtype=np.float32))
        handle.create_dataset("x-coordinate", data=np.linspace(0.0, 0.75, 4))
        handle.create_dataset("y-coordinate", data=np.linspace(0.0, 0.75, 4))
        handle.create_dataset("z-coordinate", data=np.linspace(0.0, 0.75, 4))
        # PDEBench CFD releases may store n_time + 1 time-cell boundaries.
        handle.create_dataset("t-coordinate", data=np.array([0.0, 0.1, 0.2, 0.3]))


def test_cns_task_uses_disjoint_held_out_trajectory_indices(tmp_path: Path) -> None:
    path = tmp_path / "pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1_fixture_Train.hdf5"
    _write_cns(path)
    task = build_task_specs(tmp_path, 64, 32)["cns3d"]
    train = TrajectoryCollection(tmp_path, task, "train", spatial_limit=0)
    validation = TrajectoryCollection(tmp_path, task, "val", spatial_limit=0)
    test = TrajectoryCollection(tmp_path, task, "test", spatial_limit=0)
    assert len(train.split_identities()) == 8
    assert len(validation.split_identities()) == 1
    assert len(test.split_identities()) == 1
    assert_disjoint_collections(train, validation, test)
    assert train.grid_spacing == (0.25, 0.25, 0.25)
    assert np.isclose(train.time_step, 0.1)


def _write_shallow(path: Path, dataset_key: str = "data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.zeros((10, 3, 5, 6, 1), dtype=np.float32)
    values[..., 0] = 1.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset(dataset_key, data=values)
        handle.create_dataset("x", data=np.linspace(0.0, 1.0, 5))
        handle.create_dataset("y", data=np.linspace(0.0, 2.0, 6))
        handle.create_dataset("time", data=np.array([0.0, 0.2, 0.4]))


def test_shallow_reader_uses_configured_dataset_and_channel_contract(tmp_path: Path) -> None:
    path = tmp_path / "pdebench/pdebench_shallow_water_2d/fixture.h5"
    _write_shallow(path)
    task = build_task_specs(tmp_path, 64, 32)["shallow2d"]
    collection = TrajectoryCollection(tmp_path, task, "train", spatial_limit=0)
    state = collection.readers[0].state(collection.readers[0].traj_indices_for_split()[0], 0)
    assert state.shape == (1, 5, 6)
    assert collection.channel_names == ("height",)
    assert collection.grid_spacing == (0.25, 0.4)
    assert np.isclose(collection.time_step, 0.2)
    assert task.energy_type == "none"
    assert task.balance is None
    assert task.balance_channel == 0


def test_shallow_reader_rejects_unconfigured_numeric_dataset(tmp_path: Path) -> None:
    path = tmp_path / "pdebench/pdebench_shallow_water_2d/fixture.h5"
    _write_shallow(path, dataset_key="ambiguous_physics_array")
    task = build_task_specs(tmp_path, 64, 32)["shallow2d"]
    with pytest.raises(RuntimeError, match="configured trajectory dataset"):
        TrajectoryCollection(tmp_path, task, "train", spatial_limit=0)


def test_float32_linspace_quantization_is_accepted_as_uniform(tmp_path: Path) -> None:
    path = tmp_path / "coordinates.hdf5"
    with h5py.File(path, "w") as handle:
        dimensions = handle.create_group("dimensions")
        dimensions.create_dataset("x", data=np.linspace(0.0, 10.0, 256, dtype=np.float32))
        dimensions.create_dataset("y", data=np.linspace(0.0, 10.0, 256, dtype=np.float32))
    spacing = _infer_h5_spacing(path, (256, 256), dim=2)
    assert np.allclose(spacing, (10.0 / 255.0, 10.0 / 255.0), rtol=5.0e-5)


def test_materially_nonuniform_coordinates_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "coordinates.hdf5"
    x = np.linspace(0.0, 1.0, 8)
    x[4:] += 0.01
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x", data=x)
        handle.create_dataset("y", data=np.linspace(0.0, 1.0, 8))
    with pytest.raises(RuntimeError, match="Nonuniform x-coordinates"):
        _infer_h5_spacing(path, (8, 8), dim=2)


def _write_well_mhd(path: Path, *, include_magnetic: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        dimensions = handle.create_group("dimensions")
        dimensions.create_dataset("x", data=np.linspace(0.0, 3.0, 4))
        dimensions.create_dataset("y", data=np.linspace(0.0, 3.0, 4))
        dimensions.create_dataset("z", data=np.linspace(0.0, 3.0, 4))
        dimensions.create_dataset("time", data=np.array([0.0, 0.25, 0.5]))

        scalars = handle.create_group("t0_fields")
        scalars.attrs["field_names"] = np.asarray([b"density"])
        density = scalars.create_dataset(
            "density",
            data=np.ones((2, 3, 4, 4, 4), dtype=np.float32),
        )
        density.attrs["sample_varying"] = True
        density.attrs["time_varying"] = True

        vectors = handle.create_group("t1_fields")
        names = [b"velocity"] + ([b"magnetic_field"] if include_magnetic else [])
        vectors.attrs["field_names"] = np.asarray(names)
        velocity_values = np.zeros((2, 3, 4, 4, 4, 3), dtype=np.float32)
        velocity_values[..., 0] = 2.0
        velocity = vectors.create_dataset("velocity", data=velocity_values)
        velocity.attrs["sample_varying"] = True
        velocity.attrs["time_varying"] = True
        if include_magnetic:
            magnetic_values = np.zeros((2, 3, 4, 4, 4, 3), dtype=np.float32)
            magnetic_values[..., 1] = 3.0
            magnetic = vectors.create_dataset("magnetic_field", data=magnetic_values)
            magnetic.attrs["sample_varying"] = True
            magnetic.attrs["time_varying"] = True


def test_well_reader_assembles_all_declared_mhd_fields_in_physical_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mhd.hdf5"
    _write_well_mhd(path)
    task = build_task_specs(tmp_path, 64, 32)["mhd3d"]
    reader = WellH5Reader(path, task, "train", spatial_limit=0)
    state = reader.state(0, 0)
    assert state.shape == (7, 4, 4, 4)
    assert reader.channel_names == (
        "density",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "magnetic_field_x",
        "magnetic_field_y",
        "magnetic_field_z",
    )
    assert np.isclose(float(state[0].mean()), 1.0)
    assert np.isclose(float(state[1].mean()), 2.0)
    assert np.isclose(float(state[5].mean()), 3.0)
    assert reader.grid_spacing == (1.0, 1.0, 1.0)
    assert np.isclose(reader.time_step, 0.25)


def test_well_reader_rejects_missing_required_physical_field(tmp_path: Path) -> None:
    path = tmp_path / "mhd_missing_field.hdf5"
    _write_well_mhd(path, include_magnetic=False)
    task = build_task_specs(tmp_path, 64, 32)["mhd3d"]
    with pytest.raises(RuntimeError, match="Missing required Well fields"):
        WellH5Reader(path, task, "train", spatial_limit=0)


def test_periodic_reduction_keeps_uniform_spacing_across_the_seam(tmp_path: Path) -> None:
    from diagnostic_suite.data import NpyReader

    x = np.arange(160, dtype=np.float32) / 160
    path = tmp_path / "periodic.npy"
    np.save(path, np.broadcast_to(x[None, None, :, None], (2, 3, 160, 1)))
    task = TaskSpec(
        name="periodic",
        label="periodic",
        dim=1,
        globs={"train": ["periodic.npy"], "val": [], "test": []},
        periodic=True,
        domain_extent=(1.0,),
        max_channels=1,
        channel_names=("scalar",),
    )
    reader = NpyReader(path, task, "train", spatial_limit=64)
    nodes = reader.state(0, 0).numpy()[0]
    circular_gaps = np.diff(np.r_[nodes, nodes[0] + 1.0])
    assert len(nodes) == 40
    np.testing.assert_allclose(circular_gaps, reader.grid_spacing[0], atol=1e-7)
    assert len(nodes) * reader.grid_spacing[0] == pytest.approx(1.0)
    np.save(path, np.zeros((2, 3, 127, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="uniformly subsample"):
        NpyReader(path, task, "train", spatial_limit=64)
