"""Export independent manufactured balance/constraint checks for the manuscript."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import torch

from diagnostic_suite import BalanceSpec, ConstraintSpec
from diagnostic_suite.metrics import constraint_errors, integral_balance_error
from diagnostic_suite.runner import _source_fingerprint
from diagnostic_suite.tasks import TaskSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("PAPER/revision_evidence/balance_constraints"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dim in (1, 2, 3):
        grid, dt = 8, 0.125
        x = (torch.arange(grid, dtype=torch.float64) + 0.5) / grid
        coordinates = torch.stack(torch.meshgrid(*([x] * dim), indexing="ij"))[None]
        initial = torch.full((2, 1, *([grid] * dim)), 10.0, dtype=torch.float64)
        task = TaskSpec(
            name="manufactured",
            label="manufactured",
            dim=dim,
            globs={"train": [], "val": [], "test": []},
            periodic=False,
            grid_spacing=(1 / grid,) * dim,
            balance=BalanceSpec(
                conserved_quantity=lambda state, task: state,
                flux=lambda state, task, coordinate=coordinates, shape=([grid] * dim): coordinate.expand(
                    state.shape[0], -1, *shape
                ),
                source=lambda state, task: torch.full_like(state, 2.0),
            ),
        )
        for subvolume in (False, True):
            volume = (slice(2, 6),) * dim if subvolume else None
            value = integral_balance_error(initial, initial + (2 - dim) * dt, task, delta_t=dt, control_volume=volume)
            rows.append(
                dict(
                    case="linear_flux_source",
                    dim=dim,
                    grid=grid,
                    horizon=dt,
                    subvolume=subvolume,
                    residual=value.max().item(),
                )
            )
    for grid in (16, 32, 64, 128):
        coordinate = ((torch.arange(grid, dtype=torch.float64) + 0.5) / grid)[None, None]
        initial = torch.full_like(coordinate, 10.0)
        task = TaskSpec(
            name="cubic_flux",
            label="cubic flux",
            dim=1,
            globs={"train": [], "val": [], "test": []},
            periodic=False,
            grid_spacing=(1 / grid,),
            balance=BalanceSpec(
                conserved_quantity=lambda state, task: state, flux=lambda state, task, x=coordinate: x**3
            ),
        )
        value = integral_balance_error(initial, initial - 0.3 * coordinate.square(), task, delta_t=0.1)
        rows.append(
            dict(case="spatial_refinement", dim=1, grid=grid, horizon=0.1, subvolume=False, residual=value.item())
        )
    for dt in (0.1, 0.05, 0.025, 0.0125):
        initial = torch.ones((1, 1, 16), dtype=torch.float64)
        task = TaskSpec(
            name="exponential_source",
            label="exponential source",
            dim=1,
            globs={"train": [], "val": [], "test": []},
            periodic=True,
            grid_spacing=(1 / 16,),
            balance=BalanceSpec(
                conserved_quantity=lambda state, task: state,
                flux=lambda state, task: torch.zeros_like(state),
                source=lambda state, task: state,
            ),
        )
        value = integral_balance_error(initial, initial * math.exp(dt), task, delta_t=dt)
        rows.append(
            dict(case="temporal_refinement", dim=1, grid=16, horizon=dt, subvolume=False, residual=value.item())
        )
    grid = 32
    x = 2 * math.pi * torch.arange(grid, dtype=torch.float64) / grid
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    fields = {
        "solenoidal": torch.stack((torch.sin(yy), torch.sin(xx)))[None],
        "nonsolenoidal": torch.stack((torch.sin(xx), torch.zeros_like(xx)))[None],
    }
    task = TaskSpec(
        name="magnetic",
        label="magnetic",
        dim=2,
        globs={"train": [], "val": [], "test": []},
        periodic=True,
        grid_spacing=(2 * math.pi / grid,) * 2,
        constraints=(ConstraintSpec("magnetic_divergence", "divergence", (0, 1)),),
    )
    constraint_rows = [
        {"case": name, "magnetic_divergence": constraint_errors(value, task)["magnetic_divergence"].item()}
        for name, value in fields.items()
    ]
    table = pd.DataFrame(rows)
    spatial = table.loc[table.case.eq("spatial_refinement"), "residual"].to_numpy()
    temporal = table.loc[table.case.eq("temporal_refinement"), "residual"].to_numpy()
    checks = {
        "flux_source_mass_cancel_in_1d_2d_3d_and_subvolumes": bool(
            table.loc[table.case.eq("linear_flux_source"), "residual"].max() < 1e-14
        ),
        "spatial_residual_refines_at_second_order": bool(((spatial[:-1] / spatial[1:]) > 3.8).all()),
        "endpoint_trapezoid_residual_is_third_order_per_step": bool(((temporal[:-1] / temporal[1:]) > 7.5).all()),
        "solenoidal_field_satisfies_discrete_constraint": constraint_rows[0]["magnetic_divergence"] < 1e-14,
        "known_gradient_field_has_unit_constraint_residual": abs(constraint_rows[1]["magnetic_divergence"] - 1) < 1e-8,
    }
    table.to_csv(args.outdir / "balance.csv", index=False)
    pd.DataFrame(constraint_rows).to_csv(args.outdir / "constraints.csv", index=False)
    result = dict(
        checks=checks,
        source_sha256=_source_fingerprint(),
        spatial_refinement_factors=(spatial[:-1] / spatial[1:]).tolist(),
        temporal_refinement_factors=(temporal[:-1] / temporal[1:]).tolist(),
    )
    (args.outdir / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise AssertionError("Manufactured balance/constraint verification failed")


if __name__ == "__main__":
    main()
