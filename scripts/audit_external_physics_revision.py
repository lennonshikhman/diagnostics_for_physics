"""Re-audit saved CNS/MHD checkpoints on their original test windows at k=2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostic_suite import __version__
from diagnostic_suite.controlled import _torch_load
from diagnostic_suite.data import TrajectoryCollection, WindowDataset
from diagnostic_suite.evaluate import _predict
from diagnostic_suite.metrics import energy, energy_reference_error, rel_l2
from diagnostic_suite.models import OversmoothWrapper, build_model
from diagnostic_suite.normalization import Normalizer
from diagnostic_suite.physical_audit import cns_energy_components, physical_samples
from diagnostic_suite.runner import _source_fingerprint
from diagnostic_suite.tasks import apply_file_selection, build_task_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs_revision_20261021/external"))
    parser.add_argument("--data-root", type=Path, default=Path("outputs_revision_20261021/raw"))
    parser.add_argument("--paper", type=Path, default=Path("PAPER"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Explicit IEEE float32 settings reproduce the archived FNO metrics;
    # do not rely on a runtime-dependent cuDNN precision default.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    configuration = json.loads((args.root / "args.json").read_text())
    tasks = apply_file_selection(build_task_specs(args.data_root, 32, 16), Path(configuration["file_selection"]))
    device = torch.device(args.device)
    out = args.paper / "revision_evidence/external_physics"
    out.mkdir(parents=True, exist_ok=True)
    samples, smoothing, provenance = [], [], []
    for name in ("cns3d", "mhd3d"):
        collection = TrajectoryCollection(args.data_root, tasks[name], "test", configuration["spatial_limit_3d"])
        task = collection.runtime_task()
        cache = {}
        for seed in configuration["seeds"]:
            windows = WindowDataset(
                collection,
                configuration["max_test_windows"],
                horizon=max(configuration["eval_horizons"]),
                seed=seed + 3,
            )
            pairs = []
            for reader_index, trajectory, start in windows.items:
                reader = collection.readers[reader_index]
                states = []
                for index in (start, start + 2):
                    key = (reader_index, trajectory, index)
                    if key not in cache:
                        cache[key] = reader.state(trajectory, index)
                    states.append(cache[key])
                pairs.append(states)
            initial = torch.stack([pair[0] for pair in pairs]).to(device)
            reference = torch.stack([pair[1] for pair in pairs]).to(device)
            volume = tuple(slice(0, size // 2) for size in initial.shape[2:])
            norm_path = args.root / "normalizers" / name / f"seed_{seed}.pt"
            norm_saved = _torch_load(norm_path, torch.device("cpu"))
            normalizer = Normalizer(norm_saved["mean"], norm_saved["std"]).to(device)

            def append(
                model_name,
                variant,
                prediction,
                *,
                initial=initial,
                reference=reference,
                task=task,
                volume=volume,
                windows=windows,
                name=name,
                seed=seed,
            ):
                values = physical_samples(
                    initial, reference, prediction, task, delta_t=2 * task.time_step, control_volume=volume
                )
                arrays = {key: tensor.detach().cpu().numpy() for key, tensor in values.items()}
                if not all(np.isfinite(a).all() for a in arrays.values()):
                    raise ValueError("Nonfinite applicable external physical audit")
                for index, identity in enumerate(windows.items):
                    samples.append(
                        dict(
                            task=name,
                            seed=seed,
                            model=model_name,
                            variant=variant,
                            window=index,
                            trajectory=identity[1],
                            start=identity[2],
                            **{key: float(a[index]) for key, a in arrays.items()},
                        )
                    )

            append("reference", "reference", reference)
            for model_name in ("fno", "unet", "resnet"):
                for variant in ("well", "underfit"):
                    path = args.root / "checkpoints" / name / f"seed_{seed}" / f"{model_name}_{variant}.pt"
                    before = hashlib.sha256(path.read_bytes()).hexdigest()
                    checkpoint = _torch_load(path, torch.device("cpu"))
                    model = build_model(
                        model_name,
                        3,
                        initial.shape[1],
                        tuple(initial.shape[2:]),
                        width=configuration["width"],
                        modes=configuration["modes"],
                    ).to(device)
                    model.load_state_dict(checkpoint["model"])
                    model.eval()
                    predictions = []
                    smooth_predictions = []
                    smoothed = OversmoothWrapper(
                        model, 3, passes=configuration["oversmooth_passes"], periodic=task.periodic
                    )
                    with torch.no_grad():
                        for start in range(0, len(initial), 16):
                            state = initial[start : start + 16]
                            predictions.append(
                                _predict(model, normalizer, state, 2, configuration["max_delta"], device)
                            )
                            if variant == "well":
                                smooth_predictions.append(
                                    _predict(smoothed, normalizer, state, 2, configuration["max_delta"], device)
                                )
                    prediction = torch.cat(predictions)
                    append(model_name, variant, prediction)
                    if variant == "well":
                        filtered = torch.cat(smooth_predictions)
                        append(model_name, "oversmoothed", filtered)
                        if name == "cns3d":
                            parts = cns_energy_components(reference, task)
                            difference = rel_l2(filtered.double(), prediction.double())
                            total = parts["kinetic"] + parts["internal"]
                            data = {
                                "prediction_difference": difference,
                                "reference_kinetic": parts["kinetic"],
                                "reference_internal": parts["internal"],
                                "reference_internal_fraction": parts["internal"] / total,
                                "standard_functional": energy(prediction.double(), task),
                                "smoothed_functional": energy(filtered.double(), task),
                                "standard_functional_error": energy_reference_error(
                                    reference.double(), prediction.double(), task
                                ),
                                "smoothed_functional_error": energy_reference_error(
                                    reference.double(), filtered.double(), task
                                ),
                            }
                            for index in range(len(initial)):
                                smoothing.append(
                                    dict(
                                        seed=seed,
                                        model=model_name,
                                        window=index,
                                        **{key: float(tensor[index]) for key, tensor in data.items()},
                                    )
                                )
                    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
                        raise RuntimeError("External checkpoint changed during read-only audit")
                    provenance.append(
                        dict(
                            task=name,
                            seed=seed,
                            model=model_name,
                            variant=variant,
                            checkpoint=str(path.relative_to(args.root)),
                            sha256=before,
                        )
                    )
                    print(f"Audited {name} seed={seed} {model_name}/{variant}", flush=True)
    frame = pd.DataFrame(samples)
    smooth = pd.DataFrame(smoothing)
    frame.to_csv(out / "subvolume_samples.csv", index=False)
    smooth.to_csv(out / "cns_smoothing_samples.csv", index=False)
    if not (smooth.prediction_difference > 0).all():
        raise ValueError("A nominally smoothed CNS prediction did not differ from the base prediction")
    seed_means = frame.groupby(["task", "model", "variant", "seed"]).mean(numeric_only=True).reset_index()
    seed_means.to_csv(out / "subvolume_seed_means.csv", index=False)
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Task/model & Variant & Full-domain balance & Octant balance\\",
        r"\midrule",
    ]
    labels = dict(well="Standard", underfit="Short", oversmoothed="Smoothed", reference="Reference")
    for (task_name, model_name, variant), group in seed_means.groupby(["task", "model", "variant"], sort=False):

        def entry(key, group=group):
            return rf"\({group[key].mean():.4g}\pm{group[key].std(ddof=1):.2g}\)"

        lines.append(
            f"{task_name.upper()}/{model_name} & {labels[variant]} & {entry('full_balance')} & "
            f"{entry('subvolume_balance')}" + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (args.paper / "generated/external_subvolumes.tex").write_text("\n".join(lines) + "\n")
    reference_rows = smooth.loc[smooth.model.eq("fno")]
    internal_fraction = reference_rows.reference_internal.sum() / (
        reference_rows.reference_internal.sum() + reference_rows.reference_kinetic.sum()
    )
    fno = smooth.loc[smooth.model.eq("fno")]
    standard, filtered = fno.standard_functional_error.mean(), fno.smoothed_functional_error.mean()
    summary = dict(
        package_version=__version__,
        source_sha256=_source_fingerprint(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        samples=len(frame),
        smoothing_pairs=len(smooth),
        all_smoothed_states_differ=True,
        arithmetic="Float32 saved-checkpoint inference; float64 physical kernels. Primary tables retain original float32 audit records.",
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        reference_internal_fraction=float(internal_fraction),
        reference_kinetic_fraction=float(1 - internal_fraction),
        fno_standard_functional_error=float(standard),
        fno_smoothed_functional_error=float(filtered),
        fno_same_at_three_significant_digits=f"{standard:.3g}" == f"{filtered:.3g}",
        minimum_state_change=float(smooth.prediction_difference.min()),
        maximum_state_change=float(smooth.prediction_difference.max()),
        octant_slices="First half of each of the three grid axes, [0,N/2)^3.",
        uncertainty="Descriptive standard deviation over three seeds; overlapping trajectory windows.",
        checkpoints_unchanged=True,
        checkpoints=provenance,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    explanation = (
        f"For CNS3D, reference internal energy contributes {100 * internal_fraction:.5f}\\% of the retained-grid total. "
        f"All {len(smooth):,} saved-checkpoint smoothing pairs change the predicted state; "
        f"the FNO reference-functional errors are {standard:.7f} (standard) and {filtered:.7f} (smoothed). "
        "The pressure contribution is linear and its periodic spatial mean is preserved by filtering; "
        "the small kinetic contribution changes too little to alter the displayed total-functional precision. "
        "Full-domain mass balance is likewise insensitive to this mass-preserving filter.\n"
    )
    (args.paper / "generated/cns_smoothing_interpretation.tex").write_text(explanation)
    print(json.dumps({key: value for key, value in summary.items() if key != "checkpoints"}, indent=2))


if __name__ == "__main__":
    main()
