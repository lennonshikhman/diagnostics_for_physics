from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import diagnostic_suite.runner as runner
from diagnostic_suite import (
    AutoregressivePropagator,
    BalanceSpec,
    CallablePropagator,
    ConstraintSpec,
    DiagnosticConfig,
    DiagnosticPanel,
)
from diagnostic_suite.cli import parse_args
from diagnostic_suite.metrics import integral_balance_error, summarize_metric_rows
from diagnostic_suite.runner import ExperimentRunError, apply_smoke_config, run_experiment
from diagnostic_suite.tables import _degradation_ratio_table, _representative_disagreements
from diagnostic_suite.tasks import TaskSpec, build_task_specs
from diagnostic_suite.toy import make_toy_data


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_failed_prediction_is_not_averaged_away(bad_value: float) -> None:
    initial = np.ones((2, 1, 4, 4), dtype=np.float64)
    prediction = initial.copy()
    prediction[1] = bad_value
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        DiagnosticPanel(DiagnosticConfig(dim=2)).evaluate_pair(initial, initial, prediction)
    with pytest.raises(FloatingPointError, match="finite samples"):
        summarize_metric_rows({}, {"rel_l2": [0.0, bad_value]}, {"rel_l2": (True, "")})


def test_applicable_callback_failure_and_missing_channel_raise() -> None:
    state = np.ones((1, 1, 4, 4))
    config = DiagnosticConfig(dim=2, constraints=(ConstraintSpec("density", "positivity", (9,)),))
    with pytest.raises(ValueError, match="unavailable"):
        DiagnosticPanel(config).evaluate_pair(state, state, state)
    config = DiagnosticConfig(dim=2, energy_callback=lambda state, task: state[:, 0, 0, 0] * float("nan"))
    with pytest.raises(FloatingPointError, match="energy_reference_error"):
        DiagnosticPanel(config).evaluate_pair(state, state, state)
    summary = summarize_metric_rows({}, {"missing": [np.nan]}, {"missing": (False, "No physical contract")})
    assert summary["missing_status"] == "unavailable"
    assert summary["missing_total"] == 1
    assert summary["missing_n"] == summary["missing_invalid"] == 0


def test_double_precision_signal_and_callable_input_are_preserved() -> None:
    state = np.ones((1, 1, 4, 4), dtype=np.float64)
    panel = DiagnosticPanel(DiagnosticConfig(dim=2))
    actual = panel.evaluate_pair(state, state, state + 1e-8).means["rel_l2"]
    assert actual == pytest.approx(1e-8, rel=1e-6)

    def propagate(value: torch.Tensor, horizon: float) -> torch.Tensor:
        assert value.dtype == torch.float64
        return value * np.exp(-horizon)

    result = panel.evaluate_propagator(
        state, {0.1: state * np.exp(-0.1), 0.2: state * np.exp(-0.2)}, CallablePropagator(propagate)
    )
    assert result.means["rel_l2"] == 0
    assert result.means["semigroup"] < 1e-14


def test_autoregressive_audit_preserves_horizon_specific_availability() -> None:
    state = np.ones((1, 1, 4, 4))
    adapter = AutoregressivePropagator(lambda x: x + 1, step_size=0.5)
    panel = DiagnosticPanel(DiagnosticConfig(dim=2))
    result = panel.evaluate_propagator(state, {0.5: state + 1, 1.0: state + 2, 1.5: state + 3}, adapter)
    assert not result.by_horizon[0.5].applicability["semigroup"].available
    assert result.by_horizon[1.5].means["semigroup"] == 0
    assert "by construction" in result.by_horizon[1.5].applicability["semigroup"].reason
    assert result.by_horizon[1.5].metadata["generator_horizons"] == [0.5, 1.5]
    assert result.sample_counts["semigroup"] == 2
    assert set(result.as_report()["by_horizon"]) == {"0.5", "1.0", "1.5"}


@pytest.mark.parametrize("step_size", [0.0, -1.0, float("nan"), float("inf")])
def test_autoregressive_rejects_invalid_step_size(step_size: float) -> None:
    with pytest.raises(ValueError, match="step_size"):
        AutoregressivePropagator(lambda x: x, step_size=step_size)


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("subvolume", [False, True])
def test_cell_balance_cancels_manufactured_flux_source_and_mass_change(dim: int, subvolume: bool) -> None:
    n, dt = 8, 0.125
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    coordinates = torch.stack(torch.meshgrid(*([x] * dim), indexing="ij"))[None]
    initial = torch.full((2, 1, *([n] * dim)), 10.0, dtype=torch.float64)
    prediction = initial + (2 - dim) * dt
    task = TaskSpec(
        name="manufactured",
        label="manufactured",
        dim=dim,
        globs={"train": [], "val": [], "test": []},
        periodic=False,
        grid_spacing=(1 / n,) * dim,
        balance=BalanceSpec(
            conserved_quantity=lambda state, task: state[:, :1],
            flux=lambda state, task: coordinates.expand(state.shape[0], -1, *([n] * dim)),
            source=lambda state, task: torch.full_like(state[:, :1], 2.0),
        ),
    )
    volume = (slice(2, 6),) * dim if subvolume else None
    residual = integral_balance_error(initial, prediction, task, delta_t=dt, control_volume=volume)
    assert torch.max(residual) < 1e-14


def test_cell_balance_converges_for_smooth_nonconstant_flux() -> None:
    errors = []
    for n in (16, 32, 64):
        x = ((torch.arange(n, dtype=torch.float64) + 0.5) / n)[None, None]
        initial = torch.full_like(x, 10.0)
        prediction = initial - 0.1 * 3 * x.square()
        task = TaskSpec(
            name="cubic",
            label="cubic",
            dim=1,
            globs={"train": [], "val": [], "test": []},
            periodic=False,
            grid_spacing=(1 / n,),
            balance=BalanceSpec(
                conserved_quantity=lambda state, task: state, flux=lambda state, task, coordinate=x: coordinate**3
            ),
        )
        errors.append(integral_balance_error(initial, prediction, task, delta_t=0.1).item())
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


def test_installed_layout_fingerprint_is_nonempty_and_changes_with_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "site-packages" / "diagnostic_suite"
    package.mkdir(parents=True)
    path = package / "runner.py"
    path.write_text("VALUE = 1\n")
    monkeypatch.setattr(runner, "__file__", str(path))
    before = runner._source_fingerprint()
    assert before != hashlib.sha256(b"").hexdigest()
    path.write_text("VALUE = 2\n")
    assert runner._source_fingerprint() != before
    path.unlink()
    with pytest.raises(RuntimeError, match="empty"):
        runner._source_fingerprint()


def _resume_fixture(tmp_path: Path):
    make_toy_data(tmp_path / "data", n_train=4, n_val=2, n_test=2, steps=4, grid=8, seed=19)
    args = parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--outdir",
            str(tmp_path / "output"),
            "--tasks",
            "toy2d",
            "--models",
            "resnet",
            "--device",
            "cpu",
            "--max-delta",
            "2",
            "--eval-horizons",
            "2",
            "--width",
            "4",
            "--max-train-pairs",
            "4",
            "--max-val-pairs",
            "2",
            "--max-test-windows",
            "2",
            "--smoke",
            "--resume",
        ]
    )
    apply_smoke_config(args)
    tasks = build_task_specs(args.data_root, 64, 32)
    run_experiment(args, tasks)
    return args, tasks


@pytest.mark.parametrize("change", ["data", "task"])
def test_resume_rejects_changed_inputs_without_overwriting_original_evidence(tmp_path: Path, change: str) -> None:
    args, tasks = _resume_fixture(tmp_path)
    protected = ["metrics_summary.csv", "provenance.json", "data_contracts/toy2d/seed_0_files.csv"]
    original = {name: (args.outdir / name).read_bytes() for name in protected}
    if change == "data":
        path = args.data_root / "toy_advdiff2d/test.npy"
        np.save(path, np.load(path) * 10)
    else:
        tasks["toy2d"].energy_law = "conservative"
    with pytest.raises(ExperimentRunError, match="failed"):
        run_experiment(args, tasks)
    assert all((args.outdir / name).read_bytes() == value for name, value in original.items())
    failures = list((args.outdir / "invocations").glob("*/errors.json"))
    assert len(failures) == 1
    assert "signature mismatch" in json.loads(failures[0].read_text())[0]["error"]


def test_generator_ratio_is_not_ranked_as_a_monotone_error() -> None:
    rows = [
        dict(
            task="adv2d",
            model="resnet",
            variant=variant,
            rel_l2=0.1,
            semigroup=0.1,
            generator_ratio_to_reference=ratio,
            energy_violation=0.1,
        )
        for variant, ratio in [("well", 0.6), ("underfit", 0.8)]
    ]
    ratios = _degradation_ratio_table(pd.DataFrame(rows), ["adv2d"])
    assert ratios.iloc[0]["generator_ratio_to_reference_ratio"] == pytest.approx(4 / 3)
    assert ratios.iloc[0]["max_structural_ratio"] == 1.0
    assert _representative_disagreements(ratios).empty


@pytest.mark.parametrize("grid", [16, 64])
def test_referee_pde_demo_has_independent_references_and_strict_json(tmp_path: Path, grid: int) -> None:
    from diagnostic_suite.referee import run_referee_demo

    manifest = run_referee_demo(tmp_path, grid=grid)
    assert manifest["checks_passed"] == manifest["checks_total"] == 16

    def reject_constant(value: str):
        raise AssertionError(f"Nonstandard JSON constant {value}")

    report = json.loads((tmp_path / "reports.json").read_text(), parse_constant=reject_constant)
    assert report["correct"]["by_horizon"]["0.2"]["metadata"]["generator_horizons"] == [0.05, 0.2]
    assert len(pd.read_csv(tmp_path / "metrics.csv")) == 15
