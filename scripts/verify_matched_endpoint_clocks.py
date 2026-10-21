"""Supplementary designed contrast: identical accurate endpoints, distinct clocks.

This is a post hoc verification on trained supervised FNOs, not a second
confirmatory training study. Validation states calibrate the clock error.
No semigroup metric or held-out state is used in calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_controlled_revision import crossed_interval, matrix

from diagnostic_suite.controlled import PROBLEMS, ControlledFNO, _torch_load
from diagnostic_suite.metrics import rel_l2, semigroup_error


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs_revision_20261021/controlled"))
    parser.add_argument("--outdir", type=Path, default=Path("PAPER/revision_evidence/matched_clocks"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows, calibrations = [], []
    for name in ("heat1d", "advection1d"):
        problem = PROBLEMS[name]
        horizon = problem.final_time / 2
        for seed in range(5):
            folder = args.root / name / f"seed_{seed}"
            path = folder / "supervised/best_validation.pt"
            checkpoint = _torch_load(path, torch.device("cpu"))
            model = ControlledFNO(**checkpoint["architecture"])
            model.load_state_dict(checkpoint["model"])
            model.eval()
            splits = np.load(folder / "split_coefficients.npz")
            validation = torch.as_tensor(splits["validation"], dtype=torch.float32)
            initial = problem.reference(validation, 0, 64)
            reference = problem.reference(validation, horizon, 64)

            def predict(state, time, network=model, final_time=problem.final_time):
                return network(state, torch.full((len(state),), time / final_time))

            def validation_error(factor, base=initial, target=reference, time=horizon, predictor=predict):
                return rel_l2(predictor(base, factor * time), target).mean().item()

            low, high = 1.0, 2.0
            if not validation_error(low) < 0.01 < validation_error(high):
                raise ValueError("The predeclared one-percent calibration target is not bracketed")
            for _ in range(24):
                midpoint = (low + high) / 2
                if validation_error(midpoint) < 0.01:
                    low = midpoint
                else:
                    high = midpoint
            factor = (low + high) / 2
            calibrations.append(
                dict(
                    problem=name,
                    seed=seed,
                    factor=factor,
                    validation_state_error=validation_error(factor),
                    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
            test = torch.as_tensor(splits["test"], dtype=torch.float32)
            u0 = problem.reference(test, 0, 64)
            truth = problem.reference(test, horizon, 64)
            direct_constant = predict(u0, factor * horizon)
            quadratic_time = horizon * (1 + (factor - 1) * horizon / horizon)
            direct_quadratic = predict(u0, quadratic_time)
            if not torch.equal(direct_constant, direct_quadratic):
                raise ValueError("Designed clocks do not have identical endpoint predictions")
            half_constant = factor * horizon / 2
            half_quadratic = horizon / 2 * (1 + (factor - 1) / 2)
            composed_constant = predict(predict(u0, half_constant), half_constant)
            composed_quadratic = predict(predict(u0, half_quadratic), half_quadratic)
            endpoint = rel_l2(direct_constant, truth)
            sg_constant = semigroup_error(direct_constant, composed_constant)
            sg_quadratic = semigroup_error(direct_quadratic, composed_quadratic)
            for i in range(len(test)):
                rows.append(
                    dict(
                        problem=name,
                        seed=seed,
                        test_initial_state=i,
                        rel_l2=endpoint[i].item(),
                        semigroup_constant=sg_constant[i].item(),
                        semigroup_quadratic=sg_quadratic[i].item(),
                        endpoint_max_difference=0.0,
                    )
                )
            np.savez_compressed(
                args.outdir / f"{name}_seed_{seed}.npz",
                initial=u0.numpy(),
                reference=truth.numpy(),
                direct_constant=direct_constant.numpy(),
                direct_quadratic=direct_quadratic.numpy(),
                composed_constant=composed_constant.numpy(),
                composed_quadratic=composed_quadratic.numpy(),
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.outdir / "test_samples.csv", index=False)
    pd.DataFrame(calibrations).to_csv(args.outdir / "validation_calibration.csv", index=False)
    summaries = frame.groupby("problem")[["rel_l2", "semigroup_constant", "semigroup_quadratic"]].mean()
    contrasts = {}
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Equation & Identical state (\%) & Constant SG (\%) & Quadratic SG (\%) & SG factor [95\% interval]\\",
        r"\midrule",
    ]
    for name, label in (("heat1d", "Heat"), ("advection1d", "Advection")):
        group = frame.loc[frame.problem.eq(name)]
        contrast = crossed_interval(
            matrix(group, "semigroup_quadratic"), matrix(group, "semigroup_constant"), ratio=True
        )
        contrasts[name] = contrast
        values = summaries.loc[name]
        lines.append(
            f"{label} & "
            + " & ".join(rf"\({100 * values[c]:.3f}\)" for c in ("rel_l2", "semigroup_constant", "semigroup_quadratic"))
            + rf" & \({contrast[0]:.2f}\;[{contrast[1]:.2f},{contrast[2]:.2f}]\)"
            + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    generated = args.outdir.parent.parent / "generated"
    generated.mkdir(exist_ok=True)
    (generated / "matched_clocks.tex").write_text("\n".join(lines) + "\n")
    (args.outdir / "checks.json").write_text(
        json.dumps(
            dict(
                rows=len(frame),
                expected_rows=2560,
                identical_held_out_endpoints=True,
                calibration_uses="Validation state error only; target 0.01; bracket [1,2]; 24 bisections",
                scope="Post hoc designed clock perturbations of trained models, not independent training interventions.",
                summary=summaries.to_dict(orient="index"),
                paired_sg_factors=contrasts,
            ),
            indent=2,
        )
        + "\n"
    )
    print(summaries.to_string())


if __name__ == "__main__":
    main()
