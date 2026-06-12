from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "toy2d": TaskSpec(
            name="toy2d",
            label="2D periodic advection-diffusion toy dataset",
            dim=2,
            globs={
                "train": ["toy_advdiff2d/train.npy"],
                "val": ["toy_advdiff2d/val.npy"],
                "test": ["toy_advdiff2d/test.npy"],
            },
            max_channels=1,
            periodic=True,
            energy_type="generic_l2",
            balance_channel=0,
            scaling="linear_amplitude",
        ),
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




def build_all_task_specs(data_root: Path | str, spatial_limit_2d: int = 64, spatial_limit_3d: int = 32) -> Dict[str, TaskSpec]:
    """Backward-compatible alias for :func:`build_task_specs`."""
    return build_task_specs(Path(data_root), spatial_limit_2d, spatial_limit_3d)
