"""Audit existing controlled checkpoints; never train or select on rollout error.

The two partitions use T-sized and T/2-sized calls. There is no one-call query
at 2T or 4T: every network time input is in (0, 1]. All selections and failed
accuracy matches remain in the sample files, tables, and paired contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostic_suite import __version__
from diagnostic_suite.controlled import PROBLEMS, ControlledFNO, _torch_load
from diagnostic_suite.metrics import rel_l2
from diagnostic_suite.rollout import (
    bounded_chain,
    crossed_draws,
    crossed_means,
    percentile_summary,
    spearman_rows,
)
from diagnostic_suite.runner import _source_fingerprint

NAMES = {"heat1d": "Heat", "advection1d": "Advection", "burgers1d": "Burgers"}
ARMS = {"supervised": "Supervised", "pino": "PINO-style", "composition": "Composition"}
METRICS = [
    "direct_T",
    "composed_T",
    "semigroup_T",
    "full_2T",
    "full_4T",
    "half_2T",
    "half_4T",
    "full_2T_minus_direct_T",
    "full_4T_minus_direct_T",
    "partition_4T",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_sha(checkpoint):
    digest = hashlib.sha256(json.dumps(checkpoint["architecture"], sort_keys=True).encode())
    for key, value in sorted(checkpoint["model"].items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def matrix(frame, metric, seeds, count):
    pivot = frame.pivot(index="seed", columns="test_initial_state", values=metric)
    pivot = pivot.reindex(index=seeds, columns=range(count))
    if len(frame) != len(seeds) * count:
        raise ValueError("Incomplete/duplicate seed by initial-condition audit coverage")
    return pivot.to_numpy()


def metric_summary(values, draws):
    return percentile_summary(float(values.mean()), crossed_means(values, draws))


def ratio_summary(first, second, draws):
    a, b = crossed_means(first, draws), crossed_means(second, draws)
    samples = np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)
    return percentile_summary(float(first.mean() / second.mean()) if second.mean() > 0 else np.nan, samples)


@torch.no_grad()
def audit(root, outdir, problems, seeds, *, device="cpu", grid=64):
    rows, manifests = [], []
    device = torch.device(device)
    for name in problems:
        problem = PROBLEMS[name]
        coefficient_hash = None
        for seed in seeds:
            folder = root / name / f"seed_{seed}"
            coefficient_file = folder / "split_coefficients.npz"
            values = np.load(coefficient_file)["test"]
            current_hash = hashlib.sha256(values.tobytes()).hexdigest()
            if coefficient_hash is not None and current_hash != coefficient_hash:
                raise ValueError("Crossed inference requires the same test initial states across seeds")
            coefficient_hash = current_hash
            coefficients64 = torch.as_tensor(values, dtype=torch.float64)
            coefficients = coefficients64.to(device=device, dtype=torch.float32)
            initial = problem.reference(coefficients, 0, grid)
            references = {k: problem.reference(coefficients64, k * problem.final_time, grid).to(device) for k in (2, 4)}
            for arm in ARMS:
                selections = ["best_validation"] if arm == "supervised" else ["best_validation", "matched_accuracy"]
                for selection in selections:
                    checkpoint_path = folder / arm / f"{selection}.pt"
                    array_path = folder / arm / f"test_{selection}.npz"
                    original_hash, array_hash = sha(checkpoint_path), sha(array_path)
                    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
                    network = ControlledFNO(**checkpoint["architecture"]).to(device)
                    network.load_state_dict(checkpoint["model"])
                    network.eval()
                    full = bounded_chain(network, initial, 1.0, 4)
                    half = bounded_chain(network, initial, 0.5, 8)
                    with np.load(array_path) as saved:
                        direct = torch.as_tensor(saved["1.0_prediction"], device=device)
                        composed = torch.as_tensor(saved["1.0_composed"], device=device)
                        reference = torch.as_tensor(saved["1.0_reference"], device=device)
                        sg = torch.as_tensor(saved["1.0_semigroup"], device=device)
                    # Existing endpoint arrays anchor the old study. Check that
                    # the checkpoint reproduces them, allowing backend roundoff.
                    disagreement = max((full[1] - direct).abs().max().item(), (half[2] - composed).abs().max().item())
                    if disagreement > 2e-5:
                        raise ValueError(f"Checkpoint does not reproduce its saved predictions: {checkpoint_path}")
                    metrics = {
                        "direct_T": rel_l2(direct.double(), reference.double()),
                        "composed_T": rel_l2(composed.double(), reference.double()),
                        "semigroup_T": sg,
                        "partition_4T": rel_l2(half[8].double(), full[4].double()),
                    }
                    for k in (2, 4):
                        metrics[f"full_{k}T"] = rel_l2(full[k].double(), references[k])
                        metrics[f"half_{k}T"] = rel_l2(half[2 * k].double(), references[k])
                        metrics[f"full_{k}T_minus_direct_T"] = metrics[f"full_{k}T"] - metrics["direct_T"]
                    identity = dict(
                        problem=name,
                        seed=seed,
                        arm=arm,
                        selection=selection,
                        checkpoint_model_sha256=model_sha(checkpoint),
                    )
                    arrays = {key: value.detach().cpu().numpy() for key, value in metrics.items()}
                    for index in range(len(initial)):
                        rows.append(
                            dict(
                                **identity,
                                test_initial_state=index,
                                **{key: float(value[index]) for key, value in arrays.items()},
                            )
                        )
                    if sha(checkpoint_path) != original_hash or sha(array_path) != array_hash:
                        raise RuntimeError("The checkpoint-only audit modified an input artifact")
                    manifests.append(
                        dict(
                            **identity,
                            checkpoint=str(checkpoint_path.relative_to(root)),
                            checkpoint_sha256=original_hash,
                            saved_arrays_sha256=array_hash,
                            test_coefficients_sha256=current_hash,
                            training_source_sha256=checkpoint.get("source_sha256"),
                            selected_step=checkpoint["step"],
                            validation_rel_l2=checkpoint["validation_rel_l2"],
                            reproduction_maximum_absolute_difference=disagreement,
                        )
                    )
                    print(f"Audited {name} seed={seed} {arm}/{selection}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "rollout_samples.csv", index=False)
    (outdir / "checkpoint_manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    return frame


def associations(frame, seeds, count, replicates):
    # Aliases of identical model weights are retained in tables but counted only
    # once in an association. This is not deletion of an unfavorable selection.
    unique = frame.drop_duplicates(["problem", "seed", "checkpoint_model_sha256", "test_initial_state"])
    rng = np.random.default_rng(20261022)
    metrics = ["semigroup_T", "full_2T", "full_4T", "half_2T", "half_4T"]
    points, bootstrap = {}, {}
    for problem, group in unique.groupby("problem", sort=False):
        ids = {seed: list(group.loc[group.seed.eq(seed), "checkpoint_model_sha256"].unique()) for seed in seeds}
        slots = max(map(len, ids.values()))
        values = np.full((len(seeds), slots, count, len(metrics)), np.nan)
        for s, seed in enumerate(seeds):
            for slot, key in enumerate(ids[seed]):
                records = group.loc[group.seed.eq(seed) & group.checkpoint_model_sha256.eq(key)].sort_values(
                    "test_initial_state"
                )
                if len(records) != count or not np.isfinite(records[metrics]).all().all():
                    raise ValueError("A failed checkpoint cannot silently disappear from a rank association")
                values[s, slot] = records[metrics].to_numpy()
        averaged = values.mean(2).reshape(-1, len(metrics))
        points[problem] = averaged
        seed_indices = rng.integers(len(seeds), size=(replicates, len(seeds)))
        state_weights = rng.multinomial(count, np.full(count, 1 / count), size=replicates) / count
        estimates = np.einsum("bi,smik->bsmk", state_weights, values, optimize=True)
        estimates = estimates[np.arange(replicates)[:, None], seed_indices]
        bootstrap[problem] = estimates.reshape(replicates, -1, len(metrics))
    points["pooled_equations"] = np.concatenate(list(points.values()))
    bootstrap["pooled_equations"] = np.concatenate(list(bootstrap.values()), axis=1)
    output = []
    for problem in points:
        for index, metric in enumerate(metrics[1:], 1):
            a = points[problem]
            b = bootstrap[problem]
            summary = percentile_summary(
                spearman_rows(a[:, 0], a[:, index])[0], spearman_rows(b[:, :, 0], b[:, :, index])
            )
            output.append(
                dict(problem=problem, rollout=metric, unique_checkpoints=int(np.isfinite(a[:, 0]).sum()), **summary)
            )
    return output


def formatted(summary, scale=100):
    if summary["status"] != "available":
        return r"\text{failed/undefined}"
    return f"{scale * summary['mean']:.3f}\\,[{scale * summary['low']:.3f},{scale * summary['high']:.3f}]"


def report(frame, outdir, generated, seeds, replicates):
    count = frame.test_initial_state.nunique()
    draws = crossed_draws(len(seeds), count, replicates)
    summaries, contrasts = [], []
    for (problem, arm, selection), group in frame.groupby(["problem", "arm", "selection"], sort=False):
        summaries.append(
            dict(
                problem=problem,
                arm=arm,
                selection=selection,
                metrics={key: metric_summary(matrix(group, key, seeds, count), draws) for key in METRICS},
            )
        )
    for problem in frame.problem.unique():
        baseline = frame.loc[frame.problem.eq(problem) & frame.arm.eq("supervised")]
        comparison = frame.loc[
            frame.problem.eq(problem) & frame.arm.eq("composition") & frame.selection.eq("matched_accuracy")
        ]
        contrasts.append(
            dict(
                problem=problem,
                pooling="all_training_seeds",
                training_seeds=len(seeds),
                ratios_supervised_over_composition={
                    key: ratio_summary(
                        matrix(baseline, key, seeds, count), matrix(comparison, key, seeds, count), draws
                    )
                    for key in ["composed_T", "full_2T", "full_4T", "half_2T", "half_4T"]
                },
            )
        )
    association = associations(frame, seeds, count, replicates)
    for name, value in [
        ("rollout_summary", summaries),
        ("rollout_contrasts", contrasts),
        ("rollout_associations", association),
    ]:
        (outdir / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Equation & Training/selection & Direct \(T\) & Composed \(T\) & Chain \(2T\) & Chain \(4T\)\\",
        r"\midrule",
    ]
    half_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Equation & Training/selection & Half-step \(2T\) & Half-step \(4T\) & \(4T-T\) gap (pp) & Partition gap\\",
        r"\midrule",
    ]
    for row in summaries:
        label = ARMS[row["arm"]] + ("/best" if row["selection"] == "best_validation" else "/match")
        prefix = f"{NAMES[row['problem']]} & {label} & "
        lines.append(
            prefix
            + " & ".join(
                r"\(" + formatted(row["metrics"][key]) + r"\)"
                for key in ["direct_T", "composed_T", "full_2T", "full_4T"]
            )
            + r"\\"
        )
        half_lines.append(
            prefix
            + " & ".join(
                r"\(" + formatted(row["metrics"][key]) + r"\)"
                for key in ["half_2T", "half_4T", "full_4T_minus_direct_T", "partition_4T"]
            )
            + r"\\"
        )
    for filename, content in [("controlled_rollout.tex", lines), ("controlled_rollout_half.tex", half_lines)]:
        (generated / filename).write_text("\n".join(content + [r"\bottomrule", r"\end{tabular}"]) + "\n")
    adv = [row for row in summaries if row["problem"] == "advection1d"]
    if adv:
        a = next(r for r in adv if r["arm"] == "supervised")["metrics"]
        b = next(r for r in adv if r["arm"] == "composition" and r["selection"] == "matched_accuracy")["metrics"]
        ratio = next(r for r in contrasts if r["problem"] == "advection1d")["ratios_supervised_over_composition"][
            "full_4T"
        ]
        text = (
            f"For advection, the supervised and matched-composition models have composed-at-\\(T\\) "
            f"state errors {100 * a['composed_T']['mean']:.3f}\\% and {100 * b['composed_T']['mean']:.3f}\\%, respectively. "
            f"Their full-step \\(4T\\) rollout errors are {100 * a['full_4T']['mean']:.3f}\\% and "
            f"{100 * b['full_4T']['mean']:.3f}\\%; the supervised/composition ratio is \\({formatted(ratio, 1)}\\). "
            "The composition penalty lowers disagreement between time partitions while increasing both "
            "composed-state and longer-rollout error. The supervised arm gives the more accurate forecast.\n"
        )
        (generated / "rollout_interpretation.tex").write_text(text)
        (generated / "rollout_response.tex").write_text(text)
    assoc_lines = []
    for row in association:
        if row["rollout"] == "full_4T":
            label = NAMES.get(row["problem"], "Pooled equations")
            assoc_lines.append(f"{label}: \\(\\rho={formatted(row, 1)}\\)")
    (generated / "rollout_association_text.tex").write_text(
        "Spearman correlations between semigroup error and full-step \\(4T\\) rollout error are "
        + "; ".join(assoc_lines)
        + ". The heat and Burgers intervals cross zero. The larger pooled coefficient combines "
        "within-equation association with differences between equations. These descriptive correlations "
        "summarize the audited checkpoints; prediction for unseen models would require a separate test. "
        "All \\(2T\\), \\(4T\\), and half-step coefficients are archived.\n"
    )
    return summaries, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs_revision_20261021/controlled"))
    parser.add_argument("--paper", type=Path, default=Path("PAPER"))
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--replicates", type=int, default=10_000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    outdir = args.paper / "revision_evidence/rollout"
    generated = args.paper / "generated"
    outdir.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    frame = audit(args.root, outdir, args.problems, args.seeds, device=args.device, grid=args.grid)
    report(frame, outdir, generated, args.seeds, args.replicates)
    (outdir / "manifest.json").write_text(
        json.dumps(
            dict(
                package_version=__version__,
                analysis_source_sha256=_source_fingerprint(),
                analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                samples=len(frame),
                selected_checkpoint_labels=len(frame.drop_duplicates(["problem", "seed", "arm", "selection"])),
                maximum_queried_fraction=1.0,
                checkpoints_and_saved_predictions_unchanged=True,
                protocol="Supplementary out-of-horizon audit; no retraining or selection using rollout metrics.",
                one_call_comparison="Only at T; excess long-rollout error is measured relative to the one-call error at T, not a forbidden one-call query beyond T.",
                uncertainty="Paired crossed bootstrap over training seeds and shared test initial states; fixed equation strata for pooled correlations.",
                all_numerical_values_finite=bool(np.isfinite(frame[METRICS]).all().all()),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
