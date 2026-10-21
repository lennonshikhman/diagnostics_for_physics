"""Validate complete external coverage and generate descriptive manuscript tables."""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ("adv2d", "shallow2d", "active2d", "cns3d", "mhd3d")
MODELS = ("fno", "deeponet", "unet", "resnet")
VARIANTS = ("well", "underfit", "oversmoothed")
LABELS = dict(
    adv2d="ADV2D",
    shallow2d="SW2D",
    active2d="Active2D",
    cns3d="CNS3D",
    mhd3d="MHD3D",
    fno="FNO",
    deeponet="DeepONet",
    unet="U-Net",
    resnet="ResNet",
    well="Standard",
    underfit="Short",
    oversmoothed="Smoothed",
)


def entry(frame, metric, scale=1):
    values = frame[metric]
    available = frame.get(metric + "_available", pd.Series(True, index=frame.index)).fillna(False).astype(bool)
    if not available.any():
        return "---"
    if not available.all() or len(values) != 3 or not np.isfinite(values).all():
        raise ValueError(f"Inconsistent availability or incomplete seeds: {metric}")
    return rf"\({values.mean() * scale:.3g}\pm{values.std(ddof=1) * scale:.2g}\)"


def tabular(headers, rows, *, long=False, caption="", label=""):
    environment = "longtable" if long else "tabular"
    text = [rf"\begin{{{environment}}}{{{'l' * 2 + 'r' * (len(headers) - 2)}}}"]
    if long:
        text += [
            rf"\caption{{{caption}}}\label{{{label}}}\\",
            r"\toprule",
            " & ".join(headers) + r"\\",
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            " & ".join(headers) + r"\\",
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endfoot",
        ]
    else:
        text += [r"\toprule", " & ".join(headers) + r"\\", r"\midrule"]
    text += [" & ".join(row) + r"\\" for row in rows]
    if not long:
        text += [r"\bottomrule"]
    return "\n".join(text + [rf"\end{{{environment}}}"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs_revision_20261021/external"))
    parser.add_argument("--paper", type=Path, default=Path("PAPER"))
    args = parser.parse_args()
    frame = pd.read_csv(args.root / "metrics_summary.csv")
    if "error" in frame and frame.error.fillna("").str.len().any():
        raise ValueError("External results contain failed task/seed records")
    keys = ("task", "model", "variant", "seed", "horizon_steps")
    observed = set(frame.loc[:, keys].itertuples(index=False, name=None))
    expected = {
        (task, model, variant, seed, horizon)
        for task, model, variant, seed, horizon in itertools.product(TASKS, MODELS, VARIANTS, range(3), (1, 2, 4))
        if not (task in ("cns3d", "mhd3d") and model == "deeponet")
    }
    expected |= {
        (task, "reference", "reference", seed, horizon)
        for task, seed, horizon in itertools.product(TASKS, range(3), (1, 2, 4))
    }
    if observed != expected or len(frame) != len(expected) or len(frame) != 531:
        raise ValueError(
            f"Coverage mismatch: {len(frame)} rows, missing={expected - observed}, extra={observed - expected}"
        )
    for metric in (
        "rel_l2",
        "semigroup",
        "generator",
        "energy_violation",
        "energy_reference_error",
        "integral_balance",
        "constraint",
        "density_positivity",
        "pressure_positivity",
        "magnetic_divergence",
        "scaling_law",
    ):
        available = frame[metric + "_available"].fillna(False).astype(bool)
        if not np.isfinite(frame.loc[available, metric]).all() or (frame[metric + "_invalid"].fillna(0) != 0).any():
            raise ValueError(f"Invalid applicable values: {metric}")
    evidence = args.paper / "revision_evidence/external"
    generated = args.paper / "generated"
    evidence.mkdir(parents=True, exist_ok=True)
    generated.mkdir(exist_ok=True)
    for name in ("metrics_summary.csv", "metrics_across_seeds.csv", "args.json", "provenance.json"):
        shutil.copy2(args.root / name, evidence / name)
    for directory in ("data_contracts", "progress"):
        shutil.copytree(args.root / directory, evidence / directory, dirs_exist_ok=True)
    final = frame.loc[frame.horizon_steps.eq(2) & frame.model.ne("reference")]
    rows_main, rows_all, rows_constraints = [], [], []
    for task, model, variant in itertools.product(TASKS, MODELS, VARIANTS):
        group = final.loc[final.task.eq(task) & final.model.eq(model) & final.variant.eq(variant)]
        if group.empty:
            continue
        label = LABELS[task] + " / " + LABELS[model]
        values = [
            entry(group, metric, scale)
            for metric, scale in (
                ("rel_l2", 100),
                ("semigroup", 100),
                ("energy_reference_error", 1),
                ("integral_balance", 1),
                ("generator_ratio_to_reference", 1),
                ("scaling_law", 1),
            )
        ]
        rows_all.append([label, LABELS[variant]] + values)
        if model == "fno":
            rows_main.append([LABELS[task], LABELS[variant]] + values)
        if task in ("cns3d", "mhd3d"):
            rows_constraints.append(
                [label, LABELS[variant]]
                + [entry(group, m) for m in ("density_positivity", "pressure_positivity", "magnetic_divergence")]
            )
    headers = [
        "Task/model",
        "Budget/filter",
        r"State (\%)",
        r"SG (\%)",
        r"Functional",
        r"Balance",
        r"\(R_{\rm gen}\)",
        "Scaling",
    ]
    (generated / "external_main.tex").write_text(tabular(headers, rows_main))
    (generated / "external_all.tex").write_text(
        tabular(
            headers,
            rows_all,
            long=True,
            label="tab:external_all",
            caption="All supported architectures and variants at horizon two, mean and sample standard deviation across three seeds. State and SG are percentages; other columns are dimensionless. Standard and short budgets do not assert convergence. Raw records preserve all three horizons.",
        )
    )
    (generated / "external_constraints.tex").write_text(
        tabular(
            [
                "Task/model",
                "Budget/filter",
                r"Density positivity",
                r"Pressure positivity",
                r"Magnetic divergence",
            ],
            rows_constraints,
            long=True,
            label="tab:external_constraints",
            caption="Separate constraint components at horizon two, with three-seed descriptive spread. A dash means unavailable. SW2D height positivity is zero in all 108 learned-model seed/horizon records and is not repeated as rows of zeros here.",
        )
    )
    ranges = {}
    for task in TASKS:
        group = frame.loc[frame.task.eq(task) & frame.model.ne("reference")]
        ranges[task] = {
            "rel_l2_min": float(group.rel_l2.min()),
            "rel_l2_max": float(group.rel_l2.max()),
            "window_counts": sorted(group.rel_l2_n.unique().astype(int).tolist()),
            "standard_training_seconds_total": float(
                group.loc[group.variant.eq("well") & group.horizon_steps.eq(2), "training_seconds"].sum()
            ),
            "short_training_seconds_total": float(
                group.loc[group.variant.eq("underfit") & group.horizon_steps.eq(2), "training_seconds"].sum()
            ),
        }
    interpretation = (
        r"The regenerated application contains 531 per-seed/horizon records: 486 learned-model/variant records and 45 reference records. "
        r"Every declared task, supported architecture, variant, seed, and horizon is present; applicable numerical values are finite. "
        r"These bounded training runs include appreciable errors and are used as software applications, not as the low-error evidence for temporal inconsistency. "
        r"That evidence comes from Section~\ref{sec:learned}. "
        r"The appendix reports all supported architectures at horizon two; the archived CSV retains horizons one, two, and four. "
        r"Reference generator, balance, and constraint residuals accompany each comparison. A reduced divergence or generator discrepancy can also arise from smoothing and is not, by itself, improved physical accuracy."
        "\n"
    )
    (generated / "external_interpretation.tex").write_text(interpretation)
    (evidence / "summary.json").write_text(json.dumps(ranges, indent=2) + "\n")
    (evidence / "checks.json").write_text(
        json.dumps(
            {
                "complete_expected_coverage": True,
                "rows": len(frame),
                "no_failed_task_seeds": True,
                "all_applicable_metrics_finite": True,
                "all_invalid_counts_zero": True,
                "independent_training_seeds": 3,
                "external_window_uncertainty": "Descriptive seed spread, not independent-trajectory confidence intervals.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(ranges, indent=2))


if __name__ == "__main__":
    main()
