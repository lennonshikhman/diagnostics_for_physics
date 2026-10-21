import json
import math

import pandas as pd
import pytest
import torch

from diagnostic_suite.metrics import generator_discrepancy
from diagnostic_suite.sensitivity import heat_mode_discrepancy, run_sensitivity


def test_exact_heat_discrepancy_has_first_order_horizon_bias():
    rate, norm = 2.0, 8.0
    first = heat_mode_discrepancy(rate, 1e-4, norm)
    second = heat_mode_discrepancy(rate, 5e-5, norm)
    assert first / second == pytest.approx(2.0, rel=1e-3)
    assert first == pytest.approx(rate * 1e-4 / 4, rel=1e-3)
    state = torch.ones((1, 1, 64), dtype=torch.float64)
    h = 0.1
    actual = generator_discrepancy(state, state * math.exp(-rate * h), state * math.exp(-2 * rate * h), h, 2 * h)
    assert actual.item() == pytest.approx(heat_mode_discrepancy(rate, h, norm), abs=1e-14)


def test_sensitivity_distinguishes_accuracy_from_generator_ratio(tmp_path):
    report = run_sensitivity(tmp_path, quick=True)
    assert report["checks_passed"] == report["checks_total"]
    table = pd.read_csv(tmp_path / "reference_ratios.csv")
    frozen = table.loc[table.case.eq("frozen") & table.generator_ratio_available]
    assert frozen.generator_ratio.eq(0).all()
    assert frozen.known_generator_error.min() > 0.99
    correct = table.loc[table.case.eq("correct") & table.generator_ratio_available]
    assert correct.generator_ratio.max() <= 1
    assert correct.state_error.eq(0).all()
    assert all(json.loads((tmp_path / "checks.json").read_text()).values())
