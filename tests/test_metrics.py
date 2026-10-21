from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from diagnostic_suite.metrics import (
    add_reference_relative_columns,
    constraint_errors,
    divergence,
    energy,
    energy_law_error,
    generator_discrepancy,
    global_conserved_quantity_drift,
    integral_balance_error,
    perturbation_response_error,
    rel_l2,
    scaling_law_error,
    semigroup_error,
)
from diagnostic_suite.tasks import BalanceSpec, ConstraintSpec, TaskSpec, build_task_specs


def _task(dim: int = 2, **kwargs: object) -> TaskSpec:
    return TaskSpec(
        name="test",
        label="test",
        dim=dim,
        globs={"train": [], "val": [], "test": []},
        **kwargs,
    )


def test_relative_semigroup_and_generator_kernels() -> None:
    initial = torch.randn(3, 2, 8, 8)
    direction = torch.full_like(initial, 0.2)
    one = initial + direction
    two = initial + 2.0 * direction

    assert torch.allclose(rel_l2(initial, initial), torch.zeros(3), atol=1.0e-7)
    assert torch.allclose(semigroup_error(two, two), torch.zeros(3), atol=1.0e-7)
    assert torch.allclose(
        generator_discrepancy(initial, one, two, 0.5, 1.0),
        torch.zeros(3),
        atol=1.0e-6,
    )
    assert torch.all(semigroup_error(two, two + 0.1) > 0)


def test_equation_specific_energy_definitions() -> None:
    generic = torch.ones(2, 1, 4, 4)
    assert torch.allclose(energy(generic, _task()), torch.full((2,), 0.5))

    shallow = _task(
        energy_type="shallow_water",
        height_channel=0,
        velocity_channels=[1, 2],
        gravity=1.0,
    )
    shallow_state = torch.zeros(1, 3, 3, 3)
    shallow_state[:, 0] = 2.0
    shallow_state[:, 1] = 3.0
    shallow_state[:, 2] = 4.0
    assert torch.allclose(energy(shallow_state, shallow), torch.tensor([27.0]))

    cns = _task(
        dim=3,
        energy_type="cns",
        density_channel=0,
        pressure_channel=1,
        velocity_channels=[2, 3, 4],
        gamma=1.4,
    )
    cns_state = torch.zeros(1, 5, 2, 2, 2)
    cns_state[:, 0] = 2.0
    cns_state[:, 1] = 3.0
    cns_state[:, 2] = 1.0
    cns_state[:, 3] = 2.0
    cns_state[:, 4] = 2.0
    assert torch.allclose(energy(cns_state, cns), torch.tensor([16.5]), atol=1.0e-6)

    mhd = _task(
        dim=3,
        energy_type="mhd",
        density_channel=0,
        velocity_channels=[1, 2, 3],
        magnetic_channels=[4, 5, 6],
    )
    mhd_state = torch.zeros(1, 7, 2, 2, 2)
    mhd_state[:, 0] = 2.0
    mhd_state[:, 1] = 1.0
    mhd_state[:, 5] = 2.0
    assert torch.allclose(energy(mhd_state, mhd), torch.tensor([3.0]))


def test_vorticity_energy_is_finite_and_amplitude_quadratic() -> None:
    n = 32
    x = torch.arange(n) * (2.0 * math.pi / n)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    omega = (torch.sin(xx) + torch.cos(yy))[None, None]
    task = _task(
        energy_type="kinetic_from_vorticity",
        vorticity_channel=0,
        periodic=True,
        grid_spacing=(2.0 * math.pi / n, 2.0 * math.pi / n),
    )
    base = energy(omega, task)
    doubled = energy(2.0 * omega, task)
    assert torch.isfinite(base).all()
    assert torch.allclose(doubled, 4.0 * base, rtol=1.0e-5, atol=1.0e-7)


def test_energy_law_distinguishes_conservative_and_dissipative_tasks() -> None:
    initial = torch.ones(1, 1, 4, 4)
    smaller = 0.5 * initial
    larger = 2.0 * initial
    conservative = _task(energy_law="conservative")
    dissipative = _task(energy_law="dissipative")
    assert energy_law_error(initial, smaller, conservative).item() > 0
    assert energy_law_error(initial, smaller, dissipative).item() == 0
    assert energy_law_error(initial, larger, dissipative).item() > 0


def test_divergence_uses_physical_grid_spacing_and_boundary_rules() -> None:
    nx, ny = 9, 7
    dx, dy = 0.25, 0.5
    x = torch.arange(nx) * dx
    y = torch.arange(ny) * dy
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    vector = torch.stack((xx, 2.0 * yy), dim=0).unsqueeze(0)
    value = divergence(vector, 2, periodic=False, grid_spacing=(dx, dy))
    assert torch.allclose(value, torch.full_like(value, 3.0), atol=1.0e-6)


def test_constraints_are_equation_specific_and_not_automatically_divergence_free() -> None:
    tasks = build_task_specs(".", 64, 32)
    shallow_names = set(constraint_errors(torch.ones(1, 1, 6, 6), tasks["shallow2d"]))
    cns_names = set(constraint_errors(torch.ones(1, 5, 4, 4, 4), tasks["cns3d"]))
    mhd_names = set(constraint_errors(torch.ones(1, 7, 4, 4, 4), tasks["mhd3d"]))
    assert shallow_names == {"height_positivity"}
    assert cns_names == {"density_positivity", "pressure_positivity"}
    assert mhd_names == {"density_positivity", "magnetic_divergence"}
    assert "velocity_divergence" not in shallow_names | cns_names | mhd_names


def test_magnetic_divergence_known_solenoidal_and_non_solenoidal_fields() -> None:
    n = 16
    spacing = 2.0 * math.pi / n
    x = torch.arange(n) * spacing
    xx, yy, zz = torch.meshgrid(x, x, x, indexing="ij")
    state = torch.ones(1, 7, n, n, n)
    state[:, 4] = torch.sin(yy)
    state[:, 5] = torch.sin(zz)
    state[:, 6] = torch.sin(xx)
    task = _task(
        dim=3,
        periodic=True,
        grid_spacing=(spacing, spacing, spacing),
        constraints=(ConstraintSpec("div_B", "divergence", (4, 5, 6)),),
    )
    exact = constraint_errors(state, task)["div_B"]
    state[:, 4] = torch.sin(xx)
    violated = constraint_errors(state, task)["div_B"]
    assert exact.item() < 1.0e-6
    assert violated.item() > 0.2


def test_control_volume_balance_uses_flux_and_source_and_global_drift_is_separate() -> None:
    base = torch.zeros(1, 3, 5, 5)
    base[:, 0] = 1.0
    task = _task(
        periodic=False,
        grid_spacing=(1.0, 1.0),
        time_step=1.0,
        balance=BalanceSpec(quantity_channel=0, velocity_channels=(1, 2)),
        balance_channel=0,
    )
    assert torch.allclose(integral_balance_error(base, base, task), torch.zeros(1))

    # A nonzero outward boundary flux produces a nonzero residual even though
    # the stored global quantity itself is unchanged.
    flux_state = base.clone()
    flux_state[:, 1] = torch.arange(5).view(5, 1)
    assert integral_balance_error(flux_state, flux_state, task).item() > 0
    assert global_conserved_quantity_drift(flux_state, flux_state, task).item() == 0

    def unit_source(state: torch.Tensor, _task_spec: TaskSpec) -> torch.Tensor:
        return torch.ones_like(state[:, :1])

    source_task = _task(
        periodic=False,
        grid_spacing=(1.0, 1.0),
        balance=BalanceSpec(
            quantity_channel=0,
            velocity_channels=(1, 2),
            source=unit_source,
        ),
    )
    increased = base.clone()
    increased[:, 0] = 2.0
    assert integral_balance_error(base, increased, source_task, delta_t=1.0).item() < 1.0e-7

    proxy_only = _task(balance_channel=0)
    assert torch.isnan(integral_balance_error(base[:, :1], base[:, :1], proxy_only)).all()
    assert global_conserved_quantity_drift(base[:, :1], base[:, :1], proxy_only).item() == 0


def test_control_volume_balance_rejects_missing_physical_discretization() -> None:
    task = _task(
        balance=BalanceSpec(quantity_channel=0, velocity_channels=(1, 2)),
        time_step=1.0,
    )
    state = torch.ones(1, 3, 4, 4)
    with pytest.raises(ValueError, match="grid_spacing or domain_extent"):
        integral_balance_error(state, state, task)


def test_perturbation_and_scaling_metrics_have_direct_positive_negative_cases() -> None:
    base = torch.randn(2, 1, 4, 4)
    reference_perturbed = base + 0.1
    exact = perturbation_response_error(base, reference_perturbed, 2 * base, 2 * base + 0.1)
    wrong = perturbation_response_error(base, reference_perturbed, 2 * base, 2 * base)
    assert torch.allclose(exact, torch.zeros(2), atol=1.0e-6)
    assert torch.all(wrong > 0.9)
    assert torch.allclose(scaling_law_error(1.25 * base, 1.25 * base), torch.zeros(2))


def test_reference_relative_columns_are_seed_and_horizon_specific() -> None:
    rows = [
        {
            "task": "toy",
            "model": "reference",
            "variant": "reference",
            "seed": 1,
            "horizon_steps": 2,
            "generator": 0.25,
        },
        {
            "task": "toy",
            "model": "fno",
            "variant": "well",
            "seed": 1,
            "horizon_steps": 2,
            "generator": 0.5,
        },
    ]
    add_reference_relative_columns(rows)
    assert np.isclose(rows[1]["generator_reference"], 0.25)
    assert np.isclose(rows[1]["generator_excess"], 0.25)
    assert np.isclose(rows[1]["generator_ratio_to_reference"], 2.0, rtol=1.0e-5)
