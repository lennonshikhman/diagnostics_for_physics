from __future__ import annotations

import numpy as np
import torch

from diagnostic_suite.metrics import (
    add_reference_relative_columns,
    constraint_error,
    energy,
    integral_balance_error,
    rel_l2,
)
from diagnostic_suite.tasks import TaskSpec


def test_rel_l2_zero_for_identical_fields() -> None:
    x = torch.randn(3, 2, 8, 8)
    assert torch.allclose(rel_l2(x, x), torch.zeros(3), atol=1.0e-7)


def test_energy_generic_l2_matches_half_mean_square() -> None:
    task = TaskSpec(name="toy", label="toy", dim=2, globs={"train": [], "val": [], "test": []})
    x = torch.ones(2, 1, 4, 4)
    assert torch.allclose(energy(x, task), torch.full((2,), 0.5))


def test_integral_balance_zero_for_unchanged_conserved_channel() -> None:
    task = TaskSpec(name="toy", label="toy", dim=2, globs={"train": [], "val": [], "test": []}, balance_channel=0)
    x = torch.randn(4, 1, 8, 8)
    assert torch.allclose(integral_balance_error(x, x, task), torch.zeros(4), atol=1.0e-7)


def test_constraint_error_detects_negative_positivity_channel() -> None:
    task = TaskSpec(
        name="pos",
        label="pos",
        dim=2,
        globs={"train": [], "val": [], "test": []},
        positivity_channels=[0],
    )
    x = torch.ones(1, 1, 4, 4)
    y = -torch.ones(1, 1, 4, 4)
    assert constraint_error(x, task).item() == 0.0
    assert constraint_error(y, task).item() > 0.9


def test_reference_relative_columns() -> None:
    rows = [
        {"task": "toy", "model": "reference", "variant": "reference", "generator": 0.25},
        {"task": "toy", "model": "fno", "variant": "well", "generator": 0.5},
    ]
    add_reference_relative_columns(rows)
    assert np.isclose(rows[1]["generator_reference"], 0.25)
    assert np.isclose(rows[1]["generator_excess"], 0.25)
    assert np.isclose(rows[1]["generator_ratio_to_reference"], 2.0, rtol=1.0e-5)
