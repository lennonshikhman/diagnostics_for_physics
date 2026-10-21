"""Build manuscript-ready benchmark tables from corrected metric artifacts.

The builder never edits a manuscript. It writes auditable CSV and LaTeX table
fragments whose columns follow each dataset's actual diagnostic applicability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

TASK_ORDER = ("adv2d", "shallow2d", "active2d", "cns3d", "mhd3d")
TASK_LABELS = {
    "adv2d": "ADV2D",
    "shallow2d": "SW2D",
    "active2d": "AM2D",
    "cns3d": "CNS3D",
    "mhd3d": "MHD3D",
}
MODEL_ORDER = ("fno", "deeponet", "unet", "resnet")
MODEL_LABELS = {
    "fno": "FNO",
    "deeponet": "DeepONet",
    "unet": "U-Net",
    "resnet": "ResNet",
}
VARIANT_ORDER = ("well", "underfit", "oversmoothed")
TABLE_NUMBER = {
    "adv2d": 5,
    "shallow2d": 6,
    "active2d": 7,
    "cns3d": 8,
    "mhd3d": 9,
}

# Tables intentionally use equation-specific columns. In particular, the
# height-only SW2D release has neither velocity-dependent total energy nor a
# flux/source balance, and CNS/MHD constraints are never averaged together.
TASK_METRICS: Dict[str, Tuple[str, ...]] = {
    "adv2d": (
        "rel_l2",
        "semigroup",
        "generator_ratio_to_reference",
        "energy_violation",
    ),
    "shallow2d": (
        "rel_l2",
        "semigroup",
        "generator_ratio_to_reference",
        "global_conserved_drift",
        "height_positivity",
    ),
    "active2d": (
        "rel_l2",
        "semigroup",
        "generator_ratio_to_reference",
        "energy_reference_error",
    ),
    "cns3d": (
        "rel_l2",
        "semigroup",
        "generator_ratio_to_reference",
        "energy_violation",
        "integral_balance",
        "density_positivity",
        "pressure_positivity",
    ),
    "mhd3d": (
        "rel_l2",
        "semigroup",
        "generator_ratio_to_reference",
        "energy_reference_error",
        "integral_balance",
        "density_positivity",
        "magnetic_divergence",
    ),
}

REFERENCE_METRICS = (
    "rel_l2",
    "semigroup",
    "generator",
    "energy_violation",
    "energy_reference_error",
    "integral_balance",
    "global_conserved_drift",
    "height_positivity",
    "density_positivity",
    "pressure_positivity",
    "magnetic_divergence",
)

APPLICABILITY_METRICS = (
    "rel_l2",
    "semigroup",
    "generator",
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
)


def _require_columns(dataframe: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"Metrics table is missing required columns: {missing}")


def _validate_metrics(dataframe: pd.DataFrame, metrics_path: Path) -> None:
    """Reject failed, duplicate, obsolete, or incomplete evidence before export."""

    _require_columns(dataframe, ("metric_schema_version",))
    if not pd.to_numeric(dataframe["metric_schema_version"], errors="coerce").eq(2).all():
        raise ValueError("Tables require metric schema 2; regenerate results with the current diagnostics")
    keys = ["task", "model", "variant", "seed", "horizon_steps"]
    if dataframe.duplicated(keys).any() or dataframe[keys].isna().any().any():
        raise ValueError("Metric identity columns contain duplicates or missing values")
    for column in dataframe:
        if not column.endswith("_available"):
            continue
        metric = column.removesuffix("_available")
        if metric not in dataframe:
            continue
        flags = dataframe[column].astype(str).str.lower().map({"true": True, "false": False})
        # Constraints on other equation families have no flag in their rows.
        if (flags.isna() & dataframe[column].notna()).any():
            raise ValueError(f"Invalid availability flag in {column}")
        values = pd.to_numeric(dataframe[metric], errors="coerce")
        if (flags.eq(True) & ~np.isfinite(values)).any():
            raise ValueError(f"Applicable metric {metric} contains failed numerical values")
        if (flags.eq(False) & values.notna()).any():
            raise ValueError(f"Unavailable metric {metric} contains numerical values")
    if not np.isfinite(pd.to_numeric(dataframe["rel_l2"], errors="coerce")).all():
        raise ValueError("State errors must be finite")
    launch = metrics_path.parent / "args.json"
    arguments = json.loads(launch.read_text()) if launch.is_file() else {}
    seeds = arguments.get("seeds") or ([arguments["seed"]] if "seed" in arguments else sorted(dataframe.seed.unique()))
    for task, task_rows in dataframe.groupby("task"):
        observed = set(task_rows.loc[~task_rows.model.eq("reference"), "model"])
        models = set(arguments.get("models", observed))
        if task in {"cns3d", "mhd3d"} and not arguments.get("deeponet_3d", False):
            models.discard("deeponet")
        if not models:
            raise ValueError(f"No learned-model rows for {task}")
        variants = {"well", "underfit"}
        if arguments.get("eval_oversmoothed", "oversmoothed" in set(task_rows.variant)):
            variants.add("oversmoothed")
        expected = {(seed, "reference", "reference") for seed in seeds}
        expected.update((seed, model, variant) for seed in seeds for model in models for variant in variants)
        actual = set(task_rows[["seed", "model", "variant"]].itertuples(index=False, name=None))
        if actual != expected:
            raise ValueError(
                f"Incomplete or unexpected seed/model/variant coverage for {task}: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )


def _available_tasks(dataframe: pd.DataFrame) -> List[str]:
    present = set(str(value) for value in dataframe["task"].dropna().unique())
    return [task for task in TASK_ORDER if task in present]


def _numeric(dataframe: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    output = dataframe.copy()
    for column in columns:
        if column not in output:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _aggregate(
    dataframe: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    prepared = _numeric(dataframe, metrics)
    grouped = prepared.groupby(list(group_columns), dropna=False, sort=False)[list(metrics)].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def _sort_models(dataframe: pd.DataFrame) -> pd.DataFrame:
    output = dataframe.copy()
    output["model"] = pd.Categorical(output["model"], MODEL_ORDER, ordered=True)
    output["variant"] = pd.Categorical(output["variant"], VARIANT_ORDER, ordered=True)
    output = output.sort_values(["model", "variant"]).reset_index(drop=True)
    output["model"] = output["model"].map(MODEL_LABELS).astype(str)
    output["variant"] = output["variant"].astype(str)
    return output


def _write_table(dataframe: pd.DataFrame, stem: str, outdir: Path) -> List[Path]:
    csv_path = outdir / f"{stem}.csv"
    tex_path = outdir / f"{stem}.tex"
    dataframe.to_csv(csv_path, index=False)
    latex = dataframe.to_latex(
        index=False,
        na_rep="--",
        float_format=lambda value: f"{value:.4e}",
        escape=True,
    )
    tex_path.write_text(latex, encoding="utf-8")
    return [csv_path, tex_path]


def _table_3_applicability(dataframe: pd.DataFrame, tasks: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for task in tasks:
        task_rows = dataframe[dataframe["task"].eq(task)]
        row: Dict[str, object] = {"task": TASK_LABELS[task]}
        for metric in APPLICABILITY_METRICS:
            column = f"{metric}_available"
            if column not in task_rows:
                row[metric] = False
                continue
            values = task_rows[column].dropna()
            if values.dtype == object:
                parsed = values.astype(str).str.lower().map({"true": True, "false": False})
                values = parsed.dropna()
            row[metric] = bool(values.astype(bool).any()) if not values.empty else False
        rows.append(row)
    return pd.DataFrame(rows)


def _table_4_reference(dataframe: pd.DataFrame, tasks: Sequence[str]) -> pd.DataFrame:
    reference = dataframe[dataframe["model"].eq("reference") & dataframe["task"].isin(tasks)]
    reference = reference.copy()
    for metric in REFERENCE_METRICS:
        available_column = f"{metric}_available"
        if metric in reference and available_column in reference:
            available = reference[available_column]
            if available.dtype == object:
                available = available.astype(str).str.lower().map({"true": True, "false": False})
            reference.loc[~available.fillna(False).astype(bool), metric] = np.nan
    output = _aggregate(reference, ["task"], REFERENCE_METRICS)
    output["task"] = output["task"].map(TASK_LABELS)
    return output


def _controlled_degradation_table(dataframe: pd.DataFrame, task: str) -> pd.DataFrame:
    selected = dataframe[dataframe["task"].eq(task) & ~dataframe["model"].eq("reference")]
    output = _aggregate(selected, ["model", "variant"], TASK_METRICS[task])
    return _sort_models(output)


def _best_well_table(dataframe: pd.DataFrame, tasks: Sequence[str]) -> pd.DataFrame:
    selected = dataframe[dataframe["variant"].eq("well") & dataframe["task"].isin(tasks)]
    all_metrics = tuple(dict.fromkeys(metric for task in tasks for metric in TASK_METRICS[task]))
    aggregated = _aggregate(selected, ["task", "model"], all_metrics)
    best_indices = aggregated.groupby("task", sort=False)["rel_l2_mean"].idxmin()
    output = aggregated.loc[best_indices].copy()
    task_rank = {task: index for index, task in enumerate(TASK_ORDER)}
    output["_rank"] = output["task"].map(task_rank)
    output = output.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)
    output["task"] = output["task"].map(TASK_LABELS)
    output["model"] = output["model"].map(MODEL_LABELS)
    return output


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1.0e-12:
        return float("nan")
    return float(numerator / denominator)


def _degradation_ratio_table(dataframe: pd.DataFrame, tasks: Sequence[str]) -> pd.DataFrame:
    aggregated_by_task = {
        task: _aggregate(
            dataframe[dataframe["task"].eq(task) & ~dataframe["model"].eq("reference")],
            ["model", "variant"],
            TASK_METRICS[task],
        )
        for task in tasks
    }
    rows: List[Dict[str, object]] = []
    for task, table in aggregated_by_task.items():
        for model in MODEL_ORDER:
            baseline_rows = table[table["model"].eq(model) & table["variant"].eq("well")]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows.iloc[0]
            for variant in ("underfit", "oversmoothed"):
                degraded_rows = table[table["model"].eq(model) & table["variant"].eq(variant)]
                if degraded_rows.empty:
                    continue
                degraded = degraded_rows.iloc[0]
                row: Dict[str, object] = {
                    "task": TASK_LABELS[task],
                    "model": MODEL_LABELS[model],
                    "variant": variant,
                }
                ratios: List[float] = []
                for metric in TASK_METRICS[task]:
                    ratio = _safe_ratio(float(degraded[f"{metric}_mean"]), float(baseline[f"{metric}_mean"]))
                    row[f"{metric}_ratio"] = ratio
                    # R_gen is centered on one; increasing it can approach or
                    # depart from the reference and is not a monotone error.
                    if metric not in {"rel_l2", "generator_ratio_to_reference"} and np.isfinite(ratio):
                        ratios.append(ratio)
                state_ratio = float(row.get("rel_l2_ratio", float("nan")))
                structural_worsening = max(ratios) if ratios else float("nan")
                row["max_structural_ratio"] = structural_worsening
                row["disagreement_score"] = _safe_ratio(structural_worsening, state_ratio)
                rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    return output.sort_values("disagreement_score", ascending=False, na_position="last").reset_index(drop=True)


def _representative_disagreements(all_ratios: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    if all_ratios.empty:
        return all_ratios
    eligible = all_ratios[
        all_ratios["rel_l2_ratio"].le(1.5)
        & all_ratios["max_structural_ratio"].gt(1.1)
        & np.isfinite(all_ratios["disagreement_score"])
    ]
    return eligible.head(limit).reset_index(drop=True)


def build_paper_tables(
    metrics_path: Path | str,
    outdir: Path | str,
    *,
    horizon: int = 2,
    require_all_tasks: bool = True,
) -> List[Path]:
    """Export validated benchmark tables using the legacy table filenames."""

    metrics_path = Path(metrics_path)
    outdir = Path(outdir)
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    dataframe = pd.read_csv(metrics_path)
    _require_columns(dataframe, ("task", "model", "variant", "seed", "horizon_steps", "rel_l2"))
    if dataframe["model"].eq("ERROR").any():
        failures = dataframe.loc[dataframe["model"].eq("ERROR"), ["task", "seed"]].to_dict("records")
        raise RuntimeError(f"Cannot build paper tables from failed benchmark rows: {failures}")
    dataframe = dataframe[pd.to_numeric(dataframe["horizon_steps"], errors="coerce").eq(int(horizon))].copy()
    if dataframe.empty:
        raise ValueError(f"No metric rows found for horizon {horizon}")
    tasks = _available_tasks(dataframe)
    missing_tasks = [task for task in TASK_ORDER if task not in tasks]
    if require_all_tasks and missing_tasks:
        raise ValueError(f"Metrics table is missing benchmark tasks: {missing_tasks}")
    if not tasks:
        raise ValueError("Metrics table contains no recognized benchmark tasks")
    _validate_metrics(dataframe, metrics_path)

    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    written.extend(_write_table(_table_3_applicability(dataframe, tasks), "table_3_applicability", outdir))
    written.extend(_write_table(_table_4_reference(dataframe, tasks), "table_4_reference_sanity", outdir))
    for task in tasks:
        table_number = TABLE_NUMBER[task]
        stem = f"table_{table_number}_{task}_controlled_degradation"
        written.extend(_write_table(_controlled_degradation_table(dataframe, task), stem, outdir))
    written.extend(_write_table(_best_well_table(dataframe, tasks), "table_10_best_well", outdir))
    ratios = _degradation_ratio_table(dataframe, tasks)
    written.extend(_write_table(ratios, "table_11_degradation_ratios_all", outdir))
    written.extend(
        _write_table(
            _representative_disagreements(ratios),
            "table_11_representative_disagreements",
            outdir,
        )
    )

    digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    manifest = {
        "source_metrics": str(metrics_path.resolve()),
        "source_metrics_sha256": digest,
        "horizon_steps": int(horizon),
        "tasks": tasks,
        "seed_values": sorted(int(value) for value in dataframe["seed"].dropna().unique()),
        "files": [path.name for path in written],
    }
    manifest_path = outdir / "table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written.append(manifest_path)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build corrected manuscript tables from benchmark metrics.")
    parser.add_argument("--metrics", type=Path, required=True, help="Completed metrics_summary.csv")
    parser.add_argument("--outdir", type=Path, required=True, help="Destination for CSV/LaTeX tables")
    parser.add_argument("--horizon", type=int, default=2, help="Discrete evaluation horizon used by the tables")
    parser.add_argument(
        "--allow-partial-tasks",
        action="store_true",
        help="Build tables from a task subset (intended only for smoke validation).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    written = build_paper_tables(
        args.metrics,
        args.outdir,
        horizon=args.horizon,
        require_all_tasks=not args.allow_partial_tasks,
    )
    print(f"Wrote {len(written)} table artifacts to {args.outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["build_paper_tables", "main", "parse_args"]
