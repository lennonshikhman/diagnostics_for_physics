"""Deterministic known-positive and known-negative diagnostic demonstrations."""

from __future__ import annotations

import math
from typing import Dict

import torch

from .api import DiagnosticConfig, DiagnosticPanel
from .metrics import constraint_errors, integral_balance_error
from .propagators import CallablePropagator, SavedPredictionAdapter
from .tasks import BalanceSpec, ConstraintSpec, TaskSpec


def _scale_state(state: torch.Tensor, scale: float) -> torch.Tensor:
    return float(scale) * state


def run_self_check(*, verbose: bool = False) -> Dict[str, float]:
    """Run analytic checks that do not depend on trained models or external data.

    Raises
    ------
    AssertionError
        If a diagnostic fails to distinguish an exact construction from a known
        violation.  A successful return is suitable for installation/CI smoke
        testing but is not a replacement for benchmark reproduction.
    """

    torch.manual_seed(7)
    initial = torch.randn(2, 1, 8, 8)
    direction = torch.full_like(initial, 0.125)
    translation = CallablePropagator(
        lambda state, horizon: state + float(horizon) * direction,
        name="exact_translation",
    )
    references = {1.0: initial + direction, 2.0: initial + 2.0 * direction}
    panel = DiagnosticPanel(DiagnosticConfig(dim=2, energy_law="reference"))
    exact = panel.evaluate_propagator(initial, references, translation)

    broken = CallablePropagator(
        lambda state, horizon: state + float(horizon) * direction + 0.2 * float(horizon) ** 2 * direction,
        name="non_semigroup_translation",
    )
    violated = panel.evaluate_propagator(initial, references, broken)

    decay_panel = DiagnosticPanel(
        DiagnosticConfig(
            dim=2,
            energy_law="dissipative",
            scaling_transform=_scale_state,
        )
    )
    decay = CallablePropagator(
        lambda state, horizon: math.exp(-float(horizon)) * state,
        name="linear_decay",
    )
    decay_references = {
        1.0: math.exp(-1.0) * initial,
        2.0: math.exp(-2.0) * initial,
    }
    scaling = decay_panel.evaluate_propagator(
        initial,
        decay_references,
        decay,
        scale=1.7,
    )

    perturbation_exact = panel.perturbation_response(
        2.0 * initial,
        2.0 * (initial + 0.1),
        2.0 * initial,
        2.0 * (initial + 0.1),
    )
    perturbation_wrong = panel.perturbation_response(
        2.0 * initial,
        2.0 * (initial + 0.1),
        2.0 * initial,
        2.0 * initial,
    )

    saved_without_composition = SavedPredictionAdapter(decay_references)
    saved_report = decay_panel.evaluate_propagator(
        initial,
        decay_references,
        saved_without_composition,
    )

    physical_task = TaskSpec(
        name="deterministic_balance",
        label="deterministic_balance",
        dim=2,
        globs={"train": [], "val": [], "test": []},
        periodic=False,
        grid_spacing=(1.0, 1.0),
        time_step=1.0,
        channel_names=("height", "velocity_x", "velocity_y"),
        constraints=(ConstraintSpec("height_positivity", "positivity", (0,)),),
        balance=BalanceSpec(
            name="mass_balance",
            quantity_channel=0,
            velocity_channels=(1, 2),
        ),
    )
    admissible = torch.zeros(2, 3, 6, 6)
    admissible[:, 0] = 1.0
    negative = admissible.clone()
    negative[:, 0, 2:4, 2:4] = -1.0
    changed_mass = admissible.clone()
    changed_mass[:, 0] = 1.5

    values = {
        "exact_semigroup": exact.means["semigroup"],
        "exact_generator": exact.means["generator"],
        "broken_semigroup": violated.means["semigroup"],
        "scaling_exact": scaling.means["scaling_law"],
        "perturbation_exact": perturbation_exact.means["perturbation_response"],
        "perturbation_wrong": perturbation_wrong.means["perturbation_response"],
        "positive_constraint": float(constraint_errors(admissible, physical_task)["height_positivity"].mean()),
        "negative_constraint": float(constraint_errors(negative, physical_task)["height_positivity"].mean()),
        "unchanged_balance": float(integral_balance_error(admissible, admissible, physical_task).mean()),
        "changed_balance": float(integral_balance_error(admissible, changed_mass, physical_task).mean()),
    }

    tolerance = 1.0e-6
    assert values["exact_semigroup"] < tolerance
    assert values["exact_generator"] < tolerance
    assert values["broken_semigroup"] > 1.0e-3
    assert values["scaling_exact"] < tolerance
    assert values["perturbation_exact"] < tolerance
    assert values["perturbation_wrong"] > 0.5
    assert values["positive_constraint"] < tolerance
    assert values["negative_constraint"] > 0.1
    assert values["unchanged_balance"] < tolerance
    assert values["changed_balance"] > 0.1
    assert not saved_report.applicability["semigroup"].available

    if verbose:
        print("Deterministic diagnostic self-check: PASS")
        for name, value in values.items():
            print(f"  {name}: {value:.6e}")
        print("  saved_prediction_semigroup: correctly unavailable without composed endpoints")
    return values


__all__ = ["run_self_check"]
