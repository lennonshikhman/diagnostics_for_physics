from __future__ import annotations

import numpy as np
import pytest
import torch

from diagnostic_suite import (
    AutoregressivePropagator,
    BalanceSpec,
    CallablePropagator,
    ConstraintSpec,
    DiagnosticConfig,
    DiagnosticPanel,
    SavedPredictionAdapter,
)


def _scale(state: torch.Tensor, factor: float) -> torch.Tensor:
    return factor * state


@pytest.mark.parametrize("reference_scale", [0.0, 1.0e-8, 0.2])
def test_generator_reference_ratio_rejects_epsilon_dominated_scale(reference_scale: float) -> None:
    state = np.ones((2, 1, 8), dtype=np.float64)
    panel = DiagnosticPanel(DiagnosticConfig(dim=1))
    result = panel.generator_discrepancy(state, 0.9 * state, 0.7 * state, reference_scale=reference_scale)
    if reference_scale <= 1.0e-8:
        assert not result.applicability["generator_ratio_to_reference"].available
        assert result.sample_counts["generator_ratio_to_reference"] == 0
        assert np.isnan(result.means["generator_ratio_to_reference"])
    else:
        assert result.applicability["generator_ratio_to_reference"].available
        assert result.means["generator_ratio_to_reference"] == pytest.approx(result.means["generator"] / (0.2 + 1e-8))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        panel.generator_discrepancy(state, state, state, reference_scale=-1)


def test_pair_api_reports_applicability_metadata_and_separate_constraints() -> None:
    panel = DiagnosticPanel(
        DiagnosticConfig(
            name="shallow_custom",
            dim=2,
            grid_spacing=(0.5, 0.5),
            time_step=0.1,
            periodic=False,
            energy_type="shallow_water",
            energy_law="reference",
            height_channel=0,
            velocity_channels=(1, 2),
            constraints=(ConstraintSpec("height_positivity", "positivity", (0,)),),
            balance=BalanceSpec(quantity_channel=0, velocity_channels=(1, 2)),
            metadata={"model": "external_surrogate", "version": "fixture"},
        )
    )
    initial = torch.zeros(2, 3, 6, 6)
    initial[:, 0] = 1.0
    prediction = initial.clone()
    prediction[:, 0, 0, 0] = -0.1
    result = panel.evaluate_pair(initial, initial, prediction, delta_t=0.1)
    report = result.as_report()

    assert "height_positivity" in result.samples
    assert result.means["height_positivity"] > 0
    assert not result.applicability["energy_violation"].available
    assert result.applicability["integral_balance"].available
    assert report["metadata"]["model"] == "external_surrogate"
    assert report["metadata"]["grid_spacing"] == (0.5, 0.5)


def test_multiple_constraints_are_not_collapsed_into_a_composite_score() -> None:
    panel = DiagnosticPanel(
        DiagnosticConfig(
            dim=2,
            constraints=(
                ConstraintSpec("density_positivity", "positivity", (0,)),
                ConstraintSpec("pressure_positivity", "positivity", (1,)),
            ),
        )
    )
    state = torch.ones(2, 2, 4, 4)
    result = panel.evaluate_pair(state, state, state)
    assert result.applicability["density_positivity"].available
    assert result.applicability["pressure_positivity"].available
    assert not result.applicability["constraint"].available
    assert result.sample_counts["constraint"] == 0


def test_callable_propagator_audits_all_pair_and_operator_metrics() -> None:
    initial = torch.randn(2, 1, 5, 5)
    direction = torch.full_like(initial, 0.1)
    references = {1.0: initial + direction, 2.0: initial + 2.0 * direction}
    propagator = CallablePropagator(
        lambda state, horizon: state + horizon * direction,
        name="translation",
    )
    result = DiagnosticPanel(DiagnosticConfig(dim=2)).evaluate_propagator(
        initial,
        references,
        propagator,
    )
    assert result.means["rel_l2"] < 1.0e-7
    assert result.means["semigroup"] < 1.0e-7
    assert result.means["generator"] < 1.0e-6
    assert "energy_reference_error" in result.samples
    assert result.metadata["propagator"] == "translation"


def test_saved_predictions_do_not_claim_semigroup_without_composed_endpoints() -> None:
    initial = torch.randn(1, 1, 4, 4)
    predictions = {1.0: initial + 1.0, 2.0: initial + 2.0}
    panel = DiagnosticPanel(DiagnosticConfig(dim=2))

    direct_only = panel.evaluate_propagator(
        initial,
        predictions,
        SavedPredictionAdapter(predictions),
    )
    assert not direct_only.applicability["semigroup"].available
    assert direct_only.sample_counts["semigroup"] == 0

    with_composition = panel.evaluate_propagator(
        initial,
        predictions,
        SavedPredictionAdapter(predictions, composed_predictions=predictions),
    )
    assert with_composition.applicability["semigroup"].available
    assert with_composition.means["semigroup"] == 0.0
    with pytest.raises(ValueError, match="Saved direct endpoints"):
        panel.evaluate_propagator(
            initial,
            predictions,
            SavedPredictionAdapter(predictions),
            perturbed_initial=initial + 0.1,
            perturbed_references={1.0: predictions[1.0] + 0.1, 2.0: predictions[2.0] + 0.1},
        )


def test_autoregressive_adapter_and_shape_validation() -> None:
    initial = np.zeros((1, 1, 3, 3), dtype=np.float32)
    adapter = AutoregressivePropagator(lambda state: state + 1.0, step_size=0.5)
    assert np.allclose(adapter(initial, 1.5), 3.0)

    saved = SavedPredictionAdapter({1.0: np.zeros((1, 1, 4, 4), dtype=np.float32)})
    try:
        saved(initial, 1.0)
    except ValueError as error:
        assert "shape" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("SavedPredictionAdapter accepted a shape mismatch")


def test_public_perturbation_and_configurable_scaling() -> None:
    initial = torch.randn(2, 1, 4, 4)
    panel = DiagnosticPanel(DiagnosticConfig(dim=2, scaling_transform=_scale, energy_law="dissipative"))
    perturbation = panel.perturbation_response(
        2 * initial,
        2 * (initial + 0.1),
        2 * initial,
        2 * (initial + 0.1),
    )
    assert perturbation.means["perturbation_response"] == 0.0

    decay = CallablePropagator(lambda state, horizon: np.exp(-horizon) * state)
    references = {1.0: np.exp(-1.0) * initial, 2.0: np.exp(-2.0) * initial}
    result = panel.evaluate_propagator(initial, references, decay, scale=1.3)
    assert result.applicability["scaling_law"].available
    assert result.means["scaling_law"] < 1.0e-6


def test_custom_pde_energy_and_constraint_callbacks_require_no_source_edits() -> None:
    def custom_energy(state: torch.Tensor, _task: object) -> torch.Tensor:
        return state[:, 0].abs().flatten(1).mean(dim=1)

    def upper_bound(state: torch.Tensor, _task: object) -> torch.Tensor:
        excess = torch.relu(state[:, 0] - 1.0)
        return excess.flatten(1).mean(dim=1)

    panel = DiagnosticPanel(
        DiagnosticConfig(
            dim=2,
            energy_type="user_defined",
            energy_law="conservative",
            energy_callback=custom_energy,
            custom_constraint_callbacks={"unit_upper_bound": upper_bound},
        )
    )
    initial = torch.ones(2, 1, 4, 4)
    prediction = 1.5 * initial
    result = panel.evaluate_pair(initial, initial, prediction)
    assert result.means["energy_violation"] > 0
    assert result.means["unit_upper_bound"] == 0.5
    assert result.applicability["unit_upper_bound"].available
