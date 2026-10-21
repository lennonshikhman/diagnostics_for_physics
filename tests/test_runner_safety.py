from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import diagnostic_suite.runner as runner
from diagnostic_suite.cli import parse_args
from diagnostic_suite.evaluate import evaluate_reference
from diagnostic_suite.runner import ExperimentRunError, apply_smoke_config, run_experiment
from diagnostic_suite.tasks import TaskSpec, build_task_specs
from diagnostic_suite.toy import make_toy_data


def test_missing_data_causes_non_successful_run_after_error_artifacts_are_saved(
    tmp_path: Path,
) -> None:
    args = parse_args(
        [
            "--data-root",
            str(tmp_path / "missing"),
            "--outdir",
            str(tmp_path / "output"),
            "--tasks",
            "toy2d",
            "--models",
            "resnet",
            "--device",
            "cpu",
        ]
    )
    tasks = build_task_specs(args.data_root, 64, 32)
    with pytest.raises(ExperimentRunError, match="failed"):
        run_experiment(args, tasks)
    assert (args.outdir / "errors.json").is_file()
    table = pd.read_csv(args.outdir / "metrics_summary.csv")
    assert table.loc[0, "model"] == "ERROR"


def test_toy_smoke_run_writes_horizon_and_reproducibility_artifacts(tmp_path: Path) -> None:
    make_toy_data(
        tmp_path / "data",
        n_train=6,
        n_val=3,
        n_test=3,
        steps=5,
        grid=8,
        seed=9,
    )
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
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--smoke",
        ]
    )
    apply_smoke_config(args)
    table = run_experiment(args, build_task_specs(args.data_root, 64, 32))
    assert set(table["horizon_steps"]) == {2}
    assert {"reference", "well", "underfit", "oversmoothed"} <= set(table["variant"])
    assert (args.outdir / "args.json").is_file()
    assert (args.outdir / "provenance.json").is_file()
    assert (args.outdir / "metrics_across_seeds.csv").is_file()
    assert (args.outdir / "data_contracts/toy2d/seed_0.csv").is_file()
    assert (args.outdir / "data_contracts/toy2d/seed_0_files.csv").is_file()


def test_resume_skips_a_completed_task_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_toy_data(
        tmp_path / "data",
        n_train=6,
        n_val=3,
        n_test=3,
        steps=5,
        grid=8,
        seed=10,
    )
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
            "--smoke",
            "--resume",
        ]
    )
    apply_smoke_config(args)
    tasks = build_task_specs(args.data_root, 64, 32)
    first = run_experiment(args, tasks)
    assert (args.outdir / "progress/toy2d/seed_0/complete.json").is_file()

    def unexpected_training(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a completed resume run must not train again")

    monkeypatch.setattr(runner, "train_one", unexpected_training)
    second = run_experiment(args, tasks)
    pd.testing.assert_frame_equal(first, second, check_dtype=False)


def test_reference_reports_unavailable_energy_as_nan() -> None:
    task = TaskSpec(
        name="height_only",
        label="height only",
        dim=2,
        globs={"train": ["unused"], "val": ["unused"], "test": ["unused"]},
        channel_names=("height",),
        max_channels=1,
        energy_type="none",
        energy_law="none",
    )
    sequence = torch.ones(3, 1, 4, 4)
    row = evaluate_reference(
        DataLoader([sequence], batch_size=1),
        task,
        max_delta=2,
        device=torch.device("cpu"),
        eval_horizon=2,
        seed=0,
    )
    assert row["energy_reference_error_available"] is False
    assert row["energy_reference_error_n"] == 0
    assert pd.isna(row["energy_reference_error"])
