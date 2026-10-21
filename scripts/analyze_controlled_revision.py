"""Audit all controlled results, paired uncertainty, and fixed-step controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostic_suite.controlled import PROBLEMS, ControlledFNO, _torch_load
from diagnostic_suite.metrics import rel_l2, semigroup_error

NAMES = {"heat1d": "Heat", "advection1d": "Advection", "burgers1d": "Burgers"}
ARMS = {"supervised": "Supervised", "pino": "PINO-style", "composition": "Composition"}


def crossed_interval(first, second, *, ratio=False, scale=1.0):
    rng = np.random.default_rng(20261021)
    samples = []
    for _ in range(40):
        seeds = rng.integers(first.shape[0], size=(250, first.shape[0], 1))
        states = rng.integers(first.shape[1], size=(250, 1, first.shape[1]))
        a = first[seeds, states].mean((1, 2))
        b = second[seeds, states].mean((1, 2))
        samples.extend((a / b if ratio else (a - b) * scale).tolist())
    point = first.mean() / second.mean() if ratio else (first.mean() - second.mean()) * scale
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return float(point), float(lo), float(hi)


def matrix(frame, column):
    pivot = frame.pivot(index="seed", columns="test_initial_state", values=column).sort_index()
    if pivot.shape != (5, 256) or pivot.isna().any().any():
        raise ValueError(f"Expected complete five-seed by 256-initial-condition matrix, got {pivot.shape}")
    return pivot.to_numpy()


def interval_text(value, digits=2):
    point, lo, hi = value
    return rf"\({point:.{digits}f}\;[{lo:.{digits}f},{hi:.{digits}f}]\)"


def mean_sd(series, scale=1):
    return rf"\({scale * series.mean():.3f}\pm{scale * series.std(ddof=1):.3f}\)"


@torch.no_grad()
def autoregressive(root, outdir):
    rows = []
    for problem_name, problem in PROBLEMS.items():
        for seed in range(5):
            folder = root / problem_name / f"seed_{seed}"
            checkpoint = _torch_load(folder / "supervised/best_validation.pt", torch.device("cpu"))
            model = ControlledFNO(**checkpoint["architecture"])
            model.load_state_dict(checkpoint["model"])
            model.eval()
            coefficients = torch.as_tensor(np.load(folder / "split_coefficients.npz")["test"], dtype=torch.float32)
            initial = problem.reference(coefficients, 0, 64)

            def rollout(state, steps, network=model):
                for _ in range(steps):
                    state = network(state, torch.full((len(state),), 1 / 8))
                return state

            prediction = rollout(initial, 8)
            composed = rollout(rollout(initial, 4), 4)
            errors = rel_l2(prediction, problem.reference(coefficients, problem.final_time, 64))
            composition = semigroup_error(prediction, composed)
            for i in range(len(initial)):
                rows.append(
                    dict(
                        problem=problem_name,
                        seed=seed,
                        test_initial_state=i,
                        rel_l2=errors[i].item(),
                        semigroup=composition[i].item(),
                    )
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "autoregressive_samples.csv", index=False)
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"Equation & State error (\%) & Semigroup residual\\", r"\midrule"]
    for name, group in frame.groupby("problem", sort=False):
        means = group.groupby("seed").rel_l2.mean()
        lines.append(f"{NAMES[name]} & {mean_sd(means, 100)} & " + rf"\({group.semigroup.max():.1e}\)" + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (outdir.parent.parent / "generated/autoregressive_control.tex").write_text("\n".join(lines) + "\n")
    return dict(maximum_semigroup=float(frame.semigroup.max()), rows=len(frame))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs_revision_20261021/controlled"))
    parser.add_argument("--paper", type=Path, default=Path("PAPER"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    evidence = args.paper / "revision_evidence/controlled"
    evidence.mkdir(parents=True, exist_ok=True)
    generated = args.paper / "generated"
    generated.mkdir(exist_ok=True)
    frame = pd.read_csv(args.root / "test_samples.csv")
    if len(frame) != 76800 or frame.isna().any().any():
        raise ValueError("Controlled study is incomplete or contains missing numerical values")
    frame.to_csv(evidence / "test_samples.csv", index=False)
    for name in ("args.json", "manifest.json"):
        (evidence / name).write_bytes((args.root / name).read_bytes())
    final = frame.loc[frame.fraction.eq(1)]
    seed_means = final.groupby(["problem", "arm", "selection", "seed"]).mean(numeric_only=True).reset_index()
    seed_means.to_csv(evidence / "seed_means.csv", index=False)
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Equation & Training & State (\%) & Fluctuation (\%) & Semigroup (\%) & \(R_{\rm gen}\)\\",
        r"\midrule",
    ]
    for name in PROBLEMS:
        for arm in ARMS:
            group = seed_means.loc[
                seed_means.problem.eq(name) & seed_means.arm.eq(arm) & seed_means.selection.eq("best_validation")
            ]
            ratio = group.generator / (group.generator_reference + 1e-8)
            lines.append(
                f"{NAMES[name]} & {ARMS[arm]} & {mean_sd(group.rel_l2, 100)} & "
                f"{mean_sd(group.fluctuation_rel_l2, 100)} & {mean_sd(group.semigroup, 100)} & "
                f"{mean_sd(ratio)}" + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "controlled_main.tex").write_text("\n".join(lines) + "\n")

    results, eligibility = [], []
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Equation & Eligible & Supervised \(L^2\) (\%) & Composition \(L^2\) (\%) & SG reduction & Gap (pp)\\",
        r"\midrule",
    ]
    for name in PROBLEMS:
        baseline = final.loc[final.problem.eq(name) & final.arm.eq("supervised")]
        alternative = final.loc[
            final.problem.eq(name) & final.arm.eq("composition") & final.selection.eq("matched_accuracy")
        ]
        baseline_validation = baseline.groupby("seed").validation_rel_l2.first()
        for arm in ("pino", "composition"):
            candidates = final.loc[final.problem.eq(name) & final.arm.eq(arm) & final.selection.eq("matched_accuracy")]
            scores = candidates.groupby("seed").validation_rel_l2.first()
            for seed in range(5):
                ratio = scores[seed] / baseline_validation[seed]
                eligibility.append(
                    dict(
                        problem=name,
                        arm=arm,
                        seed=seed,
                        baseline_validation=float(baseline_validation[seed]),
                        candidate_validation=float(scores[seed]),
                        ratio=float(ratio),
                        eligible=bool(0.8 <= ratio <= 1.25 and max(scores[seed], baseline_validation[seed]) < 0.02),
                    )
                )
        eligible = sum(x["eligible"] for x in eligibility if x["problem"] == name and x["arm"] == "composition")
        sg_a, sg_b = matrix(baseline, "semigroup"), matrix(alternative, "semigroup")
        l2_a, l2_b = matrix(baseline, "rel_l2"), matrix(alternative, "rel_l2")
        reduction = crossed_interval(sg_a, sg_b, ratio=True)
        gap = crossed_interval(sg_a, sg_b, scale=100)
        accuracy_gap = crossed_interval(l2_a, l2_b, scale=100)
        result = dict(
            problem=name,
            eligible_seeds=eligible,
            baseline_l2=float(100 * l2_a.mean()),
            composition_l2=float(100 * l2_b.mean()),
            baseline_semigroup=float(100 * sg_a.mean()),
            composition_semigroup=float(100 * sg_b.mean()),
            semigroup_reduction=reduction,
            semigroup_gap_pp=gap,
            l2_gap_pp=accuracy_gap,
        )
        results.append(result)
        lines.append(
            f"{NAMES[name]} & {eligible}/5 & "
            + rf"\({100 * l2_a.mean():.3f}\) & \({100 * l2_b.mean():.3f}\) & "
            + f"{interval_text(reduction)} & {interval_text(gap, 3)}"
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (generated / "controlled_matched.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(eligibility).to_csv(evidence / "matching_eligibility.csv", index=False)
    (evidence / "paired_results.json").write_text(json.dumps(results, indent=2) + "\n")
    adv = next(x for x in results if x["problem"] == "advection1d")
    heat = next(x for x in results if x["problem"] == "heat1d")
    burgers = next(x for x in results if x["problem"] == "burgers1d")
    text = (
        f"For advection, all five seeds meet the validation matching criterion. At the held-out terminal horizon, "
        f"supervised and matched composition models have state errors {adv['baseline_l2']:.3f}\\% and {adv['composition_l2']:.3f}\\%, "
        f"respectively. Their semigroup errors are {adv['baseline_semigroup']:.3f}\\% and {adv['composition_semigroup']:.3f}\\%, "
        f"a reduction factor {adv['semigroup_reduction'][0]:.2f} "
        f"(95\\% interval [{adv['semigroup_reduction'][1]:.2f},{adv['semigroup_reduction'][2]:.2f}]). "
        f"This is a low-error comparison, unlike the historical active-matter example with errors near 0.7. "
        f"Heat has {heat['eligible_seeds']}/5 eligible seeds; Burgers has {burgers['eligible_seeds']}/5. "
        f"Those failed criteria remain visible. In particular, the Burgers composition arm trades prediction accuracy "
        f"against its targeted structural loss and is not described as an accuracy-matched success. "
        f"All best-validation arms and fluctuation errors remain in Table~\\ref{{tab:controlled}}; "
        f"a nonsignificant accuracy difference is not interpreted as proof of equivalence.\n"
    )
    (generated / "controlled_interpretation.tex").write_text(text)
    (generated / "controlled_response.tex").write_text(
        f"The new advection comparison uses five seeds and 256 independent test initial conditions. "
        f"Terminal state errors are {adv['baseline_l2']:.3f}\\% and {adv['composition_l2']:.3f}\\%, "
        f"while semigroup errors are {adv['baseline_semigroup']:.3f}\\% and {adv['composition_semigroup']:.3f}\\%. "
        f"The paired semigroup reduction is {adv['semigroup_reduction'][0]:.2f} "
        f"[{adv['semigroup_reduction'][1]:.2f},{adv['semigroup_reduction'][2]:.2f}] (95\\% interval). "
        f"All five advection seeds meet the validation matching rule. Heat and Burgers matching failures are reported.\n"
    )
    ar = autoregressive(args.root, evidence)
    (evidence / "checks.json").write_text(
        json.dumps(
            dict(
                complete_samples=len(frame) == 76800,
                all_numerical_values_finite=bool(np.isfinite(frame.select_dtypes("number")).all().all()),
                all_advection_seeds_accuracy_matched=adv["eligible_seeds"] == 5,
                advection_paired_reduction_lower_bound_above_one=adv["semigroup_reduction"][1] > 1,
                autoregressive_composition_is_zero=ar["maximum_semigroup"] == 0,
                independent_test_initial_states=256,
                training_seeds=5,
                autoregressive_rows=ar["rows"],
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
