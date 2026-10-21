from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from diagnostic_suite.tables import TASK_METRICS, TASK_ORDER, build_paper_tables


def _synthetic_metrics() -> pd.DataFrame:
    rows = []
    for task_index, task in enumerate(TASK_ORDER, start=1):
        reference = {
            "task": task,
            "model": "reference",
            "variant": "reference",
            "seed": 0,
            "horizon_steps": 2,
            "rel_l2": 0.0,
            "semigroup": 0.0,
            "generator": 0.1 * task_index,
        }
        rows.append(reference)
        for variant, scale in (("well", 1.0), ("underfit", 1.2), ("oversmoothed", 1.4)):
            row = {
                "task": task,
                "model": "resnet",
                "variant": variant,
                "seed": 0,
                "horizon_steps": 2,
            }
            for metric in TASK_METRICS[task]:
                row[metric] = 0.1 * scale
            rows.append(row)
    dataframe = pd.DataFrame(rows)
    dataframe["metric_schema_version"] = 2
    for metric in (
        "rel_l2",
        "semigroup",
        "generator",
        "generator_ratio_to_reference",
        "energy_violation",
        "energy_reference_error",
        "integral_balance",
        "global_conserved_drift",
        "constraint",
        "height_positivity",
        "density_positivity",
        "pressure_positivity",
        "magnetic_divergence",
        "perturbation_response",
        "scaling_law",
    ):
        if metric not in dataframe:
            dataframe[metric] = np.nan
        dataframe[f"{metric}_available"] = dataframe[metric].notna()
    return dataframe


def test_table_builder_preserves_equation_specific_applicability(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    outdir = tmp_path / "tables"
    _synthetic_metrics().to_csv(metrics, index=False)
    written = build_paper_tables(metrics, outdir)

    assert outdir / "table_manifest.json" in written
    assert (outdir / "table_4_reference_sanity.tex").is_file()
    shallow = pd.read_csv(outdir / "table_6_shallow2d_controlled_degradation.csv")
    assert "global_conserved_drift_mean" in shallow
    assert "height_positivity_mean" in shallow
    assert "energy_violation_mean" not in shallow

    cns = pd.read_csv(outdir / "table_8_cns3d_controlled_degradation.csv")
    assert "density_positivity_mean" in cns
    assert "pressure_positivity_mean" in cns
    assert "constraint_mean" not in cns

    mhd = pd.read_csv(outdir / "table_9_mhd3d_controlled_degradation.csv")
    assert "magnetic_divergence_mean" in mhd
    assert (outdir / "table_11_representative_disagreements.csv").is_file()


def test_table_builder_rejects_invalid_or_incomplete_evidence(tmp_path: Path) -> None:
    import pytest

    metrics = tmp_path / "metrics.csv"
    original = _synthetic_metrics()
    cases = [
        original.drop(index=2),
        pd.concat([original, original.iloc[[1]]]),
        original.assign(metric_schema_version=1),
    ]
    corrupt = original.copy()
    corrupt.loc[1, "rel_l2"] = np.nan
    cases.append(corrupt)
    for i, case in enumerate(cases):
        case.to_csv(metrics, index=False)
        output = tmp_path / f"tables_{i}"
        with pytest.raises(ValueError):
            build_paper_tables(metrics, output)
        assert not output.exists()
