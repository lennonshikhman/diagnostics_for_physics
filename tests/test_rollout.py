import numpy as np
import pytest
import torch

from diagnostic_suite.rollout import (
    bounded_chain,
    crossed_draws,
    crossed_means,
    percentile_summary,
    spearman_rows,
)


def test_chaining_reaches_four_horizons_without_out_of_range_queries():
    class Translation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.queries = []

        def forward(self, state, tau):
            self.queries.extend(tau.tolist())
            return state + tau[:, None, None]

    model = Translation()
    result = bounded_chain(model, torch.zeros(3, 1, 8), 0.5, 8)
    assert torch.equal(result[2], torch.ones(3, 1, 8))
    assert torch.equal(result[8], torch.full((3, 1, 8), 4.0))
    assert set(model.queries) == {0.5}
    for fraction in (0, -1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="query"):
            bounded_chain(model, torch.zeros(1, 1, 8), fraction, 2)


def test_chaining_retains_nonfinite_failure_instead_of_a_successful_subset():
    class Fails(torch.nn.Module):
        def forward(self, state, tau):
            return state * float("nan")

    assert torch.isnan(bounded_chain(Fails(), torch.ones(1, 1, 8), 1, 2)[2]).all()
    draws = crossed_draws(2, 3, replicates=40)
    values = np.ones((2, 3))
    values[0, 0] = np.nan
    summary = percentile_summary(np.nan, crossed_means(values, draws))
    assert summary["status"] == "undefined_or_failed"
    assert summary["undefined_replicates"] == 40


def test_crossed_bootstrap_resamples_both_axes_and_preserves_pairing():
    values = np.arange(12, dtype=float).reshape(3, 4)
    draws = crossed_draws(3, 4, replicates=500)
    first = crossed_means(values, draws)
    second = crossed_means(values + 2, draws)
    assert np.allclose(second - first, 2)
    assert first.std() > 0
    constant = percentile_summary(2.0, second - first)
    assert constant["low"] == pytest.approx(2)
    assert constant["high"] == pytest.approx(2)
    with pytest.raises(ValueError, match="matrix"):
        crossed_means(values.T, draws)


def test_spearman_handles_ties_padding_and_constant_vectors():
    first = np.array([[1, 1, 3, np.nan], [4, 4, 4, np.nan]], dtype=float)
    second = np.array([[2, 2, 6, np.nan], [1, 2, 3, np.nan]], dtype=float)
    result = spearman_rows(first, second)
    assert result[0] == pytest.approx(1)
    assert np.isnan(result[1])
    assert spearman_rows([1, 2, 3], [3, 2, 1])[0] == pytest.approx(-1)


def test_spearman_accepts_readonly_pandas_exports(monkeypatch):
    import pandas as pd

    export = pd.DataFrame.to_numpy

    def export_with_readonly_views(frame, *args, **kwargs):
        values = export(frame, *args, **kwargs)
        if not kwargs.get("copy", False):
            values.setflags(write=False)
        return values

    monkeypatch.setattr(pd.DataFrame, "to_numpy", export_with_readonly_views)
    first = np.array([1.0, 1.0, 3.0])
    second = np.array([6.0, 6.0, 2.0])
    first.setflags(write=False)
    second.setflags(write=False)
    np.testing.assert_allclose(spearman_rows(first, second), [-1.0])
    np.testing.assert_array_equal(first, [1.0, 1.0, 3.0])
    np.testing.assert_array_equal(second, [6.0, 6.0, 2.0])
