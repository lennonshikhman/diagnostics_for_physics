from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

TensorCallback = Callable[..., Any]


@dataclass(frozen=True)
class ConstraintSpec:
    """One explicitly named physical admissibility constraint.

    ``kind`` is one of ``positivity``, ``divergence``, or ``custom``.  A
    divergence constraint is never inferred merely because vector channels exist;
    it must be declared here because compressible velocity is not solenoidal.
    """

    name: str
    kind: str
    channels: Tuple[int, ...] = ()
    callback: Optional[TensorCallback] = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Constraint name cannot be empty")
        if self.kind not in {"positivity", "divergence", "custom"}:
            raise ValueError(f"Unknown constraint kind {self.kind!r}")
        if self.kind in {"positivity", "divergence"} and not self.channels:
            raise ValueError(f"Constraint {self.name!r} requires channels")
        if self.kind == "custom" and self.callback is None:
            raise ValueError(f"Custom constraint {self.name!r} requires a callback")


@dataclass(frozen=True)
class BalanceSpec:
    """Configuration for a discrete control-volume balance diagnostic."""

    name: str = "mass_balance"
    quantity_channel: Optional[int] = None
    velocity_channels: Tuple[int, ...] = ()
    conserved_quantity: Optional[TensorCallback] = None
    flux: Optional[TensorCallback] = None
    source: Optional[TensorCallback] = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Balance name cannot be empty")
        has_builtin_mass = self.quantity_channel is not None and bool(self.velocity_channels)
        has_custom = self.conserved_quantity is not None and self.flux is not None
        if not (has_builtin_mass or has_custom):
            raise ValueError(
                "BalanceSpec needs quantity_channel plus velocity_channels, or conserved_quantity plus flux callbacks"
            )


@dataclass
class TaskSpec:
    name: str
    label: str
    dim: int
    globs: Dict[str, List[str]]

    # Reader and split contracts are explicit.  They replace filename heuristics
    # that previously allowed validation/test trajectories to enter training.
    reader_kind: str = "auto"  # auto, npy, pdebench, cfd, well
    split_strategy: str = "explicit"  # explicit, train_val, train_val_test
    split_seed: int = 0
    max_channels: Optional[int] = None
    channel_names: Tuple[str, ...] = ()
    trajectory_dataset_keys: Tuple[str, ...] = ()
    well_fields: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    periodic: bool = True
    grid_spacing: Tuple[float, ...] = ()
    domain_extent: Tuple[float, ...] = ()
    time_step: Optional[float] = None

    # Equation-aware diagnostics.
    energy_type: str = "generic_l2"
    energy_law: str = "reference"  # conservative, dissipative, reference, none
    energy_callback: Optional[TensorCallback] = None
    gamma: float = 1.4
    gravity: float = 1.0
    sound_speed: float = 1.0
    density_channel: Optional[int] = None
    pressure_channel: Optional[int] = None
    height_channel: Optional[int] = None
    vorticity_channel: Optional[int] = None
    velocity_channels: List[int] = field(default_factory=list)
    magnetic_channels: List[int] = field(default_factory=list)
    constraints: Tuple[ConstraintSpec, ...] = ()
    balance: Optional[BalanceSpec] = None

    # Backward-compatible fields.  New code should use ``constraints`` and
    # ``balance``; these remain accepted for downstream users of version 0.1.
    channel_layout: Dict[str, Any] = field(default_factory=dict)
    balance_channel: Optional[int] = None
    positivity_channels: List[int] = field(default_factory=list)
    scaling: Optional[str] = None

    scaling_transform: Optional[TensorCallback] = None
    scaling_time_factor: Optional[TensorCallback] = None
    custom_constraint_callbacks: Mapping[str, TensorCallback] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dim not in {1, 2, 3}:
            raise ValueError("dim must be one, two, or three")
        missing_splits = {"train", "val", "test"} - set(self.globs)
        if missing_splits:
            raise ValueError(f"globs is missing split keys: {sorted(missing_splits)}")
        if self.reader_kind not in {"auto", "npy", "pdebench", "cfd", "well"}:
            raise ValueError(f"Unknown reader kind {self.reader_kind!r}")
        if self.split_strategy not in {"explicit", "train_val", "train_val_test"}:
            raise ValueError(f"Unknown split strategy {self.split_strategy!r}")
        if self.split_seed < 0:
            raise ValueError("split_seed must be nonnegative")
        if (
            self.energy_type
            not in {
                "none",
                "generic_l2",
                "kinetic_from_vorticity",
                "kinetic",
                "shallow_water",
                "cns",
                "mhd",
            }
            and self.energy_callback is None
        ):
            raise ValueError(f"Unknown energy type {self.energy_type!r}")
        if self.energy_law not in {"conservative", "dissipative", "reference", "none"}:
            raise ValueError(f"Unknown energy law {self.energy_law!r}")
        if self.grid_spacing and len(self.grid_spacing) != self.dim:
            raise ValueError("grid_spacing must contain one value per spatial dimension")
        if self.domain_extent and len(self.domain_extent) != self.dim:
            raise ValueError("domain_extent must contain one value per spatial dimension")
        if any(not math.isfinite(value) or value <= 0 for value in self.grid_spacing):
            raise ValueError("grid_spacing values must be positive")
        if any(not math.isfinite(value) or value <= 0 for value in self.domain_extent):
            raise ValueError("domain_extent values must be positive")
        if self.time_step is not None and (not math.isfinite(self.time_step) or self.time_step <= 0):
            raise ValueError("time_step must be positive")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names must be unique")
        if len(set(self.trajectory_dataset_keys)) != len(self.trajectory_dataset_keys):
            raise ValueError("trajectory_dataset_keys must be unique")
        if self.max_channels is not None and self.channel_names:
            if self.max_channels != len(self.channel_names):
                raise ValueError("max_channels must equal the declared channel schema length")
        if self.channel_names:
            configured_indices = [
                channel
                for channel in (
                    self.density_channel,
                    self.pressure_channel,
                    self.height_channel,
                    self.vorticity_channel,
                    self.balance_channel,
                )
                if channel is not None
            ]
            configured_indices.extend(self.velocity_channels)
            configured_indices.extend(self.magnetic_channels)
            configured_indices.extend(self.positivity_channels)
            configured_indices.extend(channel for constraint in self.constraints for channel in constraint.channels)
            if self.balance is not None:
                if self.balance.quantity_channel is not None:
                    configured_indices.append(self.balance.quantity_channel)
                configured_indices.extend(self.balance.velocity_channels)
            invalid = [channel for channel in configured_indices if channel < 0 or channel >= len(self.channel_names)]
            if invalid:
                raise ValueError(
                    f"Configured channel indices {sorted(set(invalid))} exceed schema {self.channel_names}"
                )
        constraint_names = [constraint.name for constraint in self.constraints]
        if len(set(constraint_names)) != len(constraint_names):
            raise ValueError("Constraint names must be unique")
        reserved_metric_names = {
            "rel_l2",
            "semigroup",
            "generator",
            "energy_violation",
            "energy_reference_error",
            "integral_balance",
            "global_conserved_drift",
            "constraint",
            "perturbation_response",
            "scaling_law",
        }
        reserved_overlap = (set(constraint_names) | set(self.custom_constraint_callbacks)) & reserved_metric_names
        if reserved_overlap:
            raise ValueError(f"Constraint names collide with built-in diagnostics: {sorted(reserved_overlap)}")
        overlap = set(constraint_names) & set(self.custom_constraint_callbacks)
        if overlap:
            raise ValueError(f"Constraint callback names are duplicated: {sorted(overlap)}")
        for constraint in self.constraints:
            if any(channel < 0 for channel in constraint.channels):
                raise ValueError(f"Constraint {constraint.name!r} has a negative channel")
            if constraint.kind == "divergence" and len(constraint.channels) != self.dim:
                raise ValueError(f"Divergence constraint {constraint.name!r} needs {self.dim} components")
        if self.balance is not None:
            if self.balance.quantity_channel is not None and self.balance.quantity_channel < 0:
                raise ValueError("Balance quantity channel cannot be negative")
            if self.balance.quantity_channel is not None and len(self.balance.velocity_channels) != self.dim:
                raise ValueError(f"Built-in mass balance needs {self.dim} velocity components")
        if self.scaling not in {None, "linear_amplitude"}:
            raise ValueError(f"Unknown legacy scaling relation {self.scaling!r}")

    def validate_state_channels(self, channels: int) -> None:
        """Check physical requirements against an actual state, including custom API input."""

        if channels < 1:
            raise ValueError("A state must contain at least one channel")
        if self.channel_names and channels != len(self.channel_names):
            raise ValueError(f"Expected channel schema {self.channel_names}; received {channels} channels")
        required = [
            value
            for value in (
                self.density_channel,
                self.pressure_channel,
                self.height_channel,
                self.vorticity_channel,
                self.balance_channel,
            )
            if value is not None
        ]
        required.extend(self.velocity_channels)
        required.extend(self.magnetic_channels)
        required.extend(self.positivity_channels)
        required.extend(value for spec in self.constraints for value in spec.channels)
        if self.balance is not None:
            if self.balance.quantity_channel is not None:
                required.append(self.balance.quantity_channel)
            required.extend(self.balance.velocity_channels)
        invalid = [value for value in required if value < 0 or value >= channels]
        if invalid:
            raise ValueError(
                f"Configured channel indices {sorted(set(invalid))} are unavailable in {channels} channels"
            )
        if self.energy_callback is not None:
            return
        if self.energy_type == "none" and self.energy_law in {"conservative", "dissipative"}:
            raise ValueError("An energy law requires an energy functional")
        if self.energy_type in {"kinetic", "shallow_water", "cns", "mhd"}:
            if len(self.velocity_channels) != self.dim:
                raise ValueError(f"{self.energy_type} energy requires {self.dim} velocity channels")
        if self.energy_type in {"cns", "mhd"} and self.density_channel is None:
            raise ValueError(f"{self.energy_type} energy requires a density channel")
        if self.energy_type == "cns" and self.pressure_channel is None:
            raise ValueError("Compressible-flow energy requires a pressure channel")
        if self.energy_type == "mhd" and len(self.magnetic_channels) != self.dim:
            raise ValueError(f"MHD energy requires {self.dim} magnetic channels")
        if self.energy_type == "kinetic_from_vorticity" and (self.dim != 2 or not self.periodic):
            raise ValueError("Vorticity inversion requires a two-dimensional periodic state")
        if self.energy_type in {"cns", "mhd"} and self.pressure_channel is not None:
            if not math.isfinite(self.gamma) or self.gamma <= 1:
                raise ValueError("Energy with internal pressure requires finite gamma greater than one")

    def with_discretization(
        self,
        *,
        grid_spacing: Sequence[float],
        time_step: Optional[float],
        channel_names: Sequence[str],
    ) -> "TaskSpec":
        """Return a runtime task carrying reader-validated metadata."""

        return replace(
            self,
            grid_spacing=tuple(float(x) for x in grid_spacing),
            time_step=float(time_step) if time_step is not None else self.time_step,
            channel_names=tuple(str(x) for x in channel_names),
        )


def build_task_specs(data_root: Path, spatial_limit_2d: int, spatial_limit_3d: int) -> Dict[str, TaskSpec]:
    # ``data_root`` and limits are accepted for API compatibility.  Readers use
    # them at runtime and attach the effective discretization after subsampling.
    del data_root, spatial_limit_2d, spatial_limit_3d
    return {
        "toy2d": TaskSpec(
            name="toy2d",
            label="2D periodic advection-diffusion toy dataset",
            dim=2,
            globs={
                "train": ["toy_advdiff2d/train.npy"],
                "val": ["toy_advdiff2d/val.npy"],
                "test": ["toy_advdiff2d/test.npy"],
            },
            reader_kind="npy",
            split_strategy="explicit",
            max_channels=1,
            channel_names=("scalar",),
            periodic=True,
            grid_spacing=(1.0, 1.0),
            time_step=0.05,
            energy_type="generic_l2",
            energy_law="dissipative",
            balance_channel=0,
            scaling="linear_amplitude",
        ),
        "adv2d": TaskSpec(
            name="adv2d",
            label="2D scalar linear advection / APEBench",
            dim=2,
            globs={
                "train": ["apebench/apebench_incompressible_ns_2d/data/2d_diff_adv__num_spatial_dims=2_train.npy"],
                "val": ["apebench/apebench_incompressible_ns_2d/data/2d_diff_adv__num_spatial_dims=2_train.npy"],
                "test": ["apebench/apebench_incompressible_ns_2d/data/2d_diff_adv__num_spatial_dims=2_test.npy"],
            },
            reader_kind="npy",
            split_strategy="train_val",
            max_channels=1,
            channel_names=("scalar",),
            periodic=True,
            domain_extent=(1.0, 1.0),
            # The legacy directory name does not identify the equation. Upstream
            # diff_adv is difficulty-interface scalar advection, not fluid flow.
            energy_type="generic_l2",
            energy_law="conservative",
            balance_channel=0,
            scaling="linear_amplitude",
            constraints=(),
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
            reader_kind="pdebench",
            split_strategy="train_val_test",
            # The public PDEBench radial-dam-break file stores water height
            # only.  Do not invent velocity channels or apply diagnostics that
            # require them.
            max_channels=1,
            channel_names=("height",),
            trajectory_dataset_keys=("data", "solution", "tensor"),
            periodic=False,
            energy_type="none",
            energy_law="none",
            height_channel=0,
            constraints=(
                ConstraintSpec(
                    name="height_positivity",
                    kind="positivity",
                    channels=(0,),
                    description="Negative shallow-water height relative to height norm.",
                ),
            ),
            # Global mean height is still a useful mass-drift proxy.  A
            # flux/source control-volume balance is unavailable without the
            # velocity field.
            balance_channel=0,
            positivity_channels=[0],
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
            reader_kind="well",
            split_strategy="explicit",
            max_channels=None,
            # Empty selections mean: load every field in the format-declared order.
            well_fields={},
            periodic=True,
            domain_extent=(10.0, 10.0),
            time_step=0.25,
            energy_type="generic_l2",
            energy_law="reference",
            balance_channel=None,
            constraints=(),
        ),
        "cns3d": TaskSpec(
            name="cns3d",
            label="3D compressible Navier-Stokes / PDEBench",
            dim=3,
            globs={
                "train": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
                "val": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
                "test": ["pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5"],
            },
            reader_kind="cfd",
            split_strategy="train_val_test",
            max_channels=5,
            channel_names=("density", "pressure", "velocity_x", "velocity_y", "velocity_z"),
            periodic=True,
            energy_type="cns",
            energy_law="conservative",
            gamma=1.4,
            density_channel=0,
            pressure_channel=1,
            velocity_channels=[2, 3, 4],
            constraints=(
                ConstraintSpec("density_positivity", "positivity", (0,)),
                ConstraintSpec("pressure_positivity", "positivity", (1,)),
            ),
            balance=BalanceSpec(
                name="mass_balance",
                quantity_channel=0,
                velocity_channels=(2, 3, 4),
                description="Discrete periodic compressible-flow mass balance.",
            ),
            balance_channel=0,
            positivity_channels=[0, 1],
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
            reader_kind="well",
            split_strategy="explicit",
            max_channels=7,
            channel_names=(
                "density",
                "velocity_x",
                "velocity_y",
                "velocity_z",
                "magnetic_field_x",
                "magnetic_field_y",
                "magnetic_field_z",
            ),
            well_fields={
                "t0_fields": ("density",),
                "t1_fields": ("velocity", "magnetic_field"),
            },
            periodic=True,
            domain_extent=(64.0, 64.0, 64.0),
            time_step=0.01,
            energy_type="mhd",
            # The public MHD_64 state has no pressure/internal-energy channel.
            # Compare kinetic+magnetic energy to the reference instead of claiming
            # a one-sided total-energy law.
            energy_law="reference",
            density_channel=0,
            velocity_channels=[1, 2, 3],
            magnetic_channels=[4, 5, 6],
            constraints=(
                ConstraintSpec("density_positivity", "positivity", (0,)),
                ConstraintSpec("magnetic_divergence", "divergence", (4, 5, 6)),
            ),
            balance=BalanceSpec(
                name="mass_balance",
                quantity_channel=0,
                velocity_channels=(1, 2, 3),
                description="Discrete periodic MHD mass balance.",
            ),
            balance_channel=0,
            positivity_channels=[0],
        ),
    }


def apply_file_selection(tasks: Dict[str, TaskSpec], path: Path | str) -> Dict[str, TaskSpec]:
    """Restrict benchmark inputs using an explicit, archived relative-path map.

    This supports fixed-parameter subsets without interpreting a filename as a
    PDE identity or changing any physical metric definition or split policy.
    """
    selections = json.loads(Path(path).read_text())
    if not isinstance(selections, dict):
        raise ValueError("File selection must map task names to split paths")
    output = dict(tasks)
    for name, splits in selections.items():
        if name not in tasks:
            raise ValueError(f"Unknown file-selection task {name!r}")
        if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
            raise ValueError(f"File selection for {name} requires train, val, and test")
        for patterns in splits.values():
            if not isinstance(patterns, list) or not patterns:
                raise ValueError("Each selected split requires a nonempty path list")
            if any(
                not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts
                for value in patterns
            ):
                raise ValueError("Selected paths must stay relative to the data root")
        output[name] = replace(tasks[name], globs=splits)
    return output


def build_all_task_specs(
    data_root: Path | str,
    spatial_limit_2d: int = 64,
    spatial_limit_3d: int = 32,
) -> Dict[str, TaskSpec]:
    """Backward-compatible alias for :func:`build_task_specs`."""

    return build_task_specs(Path(data_root), spatial_limit_2d, spatial_limit_3d)


__all__ = [
    "BalanceSpec",
    "ConstraintSpec",
    "TaskSpec",
    "build_all_task_specs",
    "build_task_specs",
]
