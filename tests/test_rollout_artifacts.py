import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostic_suite.controlled import PROBLEMS, ControlledFNO
from diagnostic_suite.metrics import semigroup_error


def test_checkpoint_only_audit_retains_selection_aliases_and_input_bytes(tmp_path):
    root, paper = tmp_path / "saved", tmp_path / "paper"
    coefficients = PROBLEMS["heat1d"].coefficients(8, 31).numpy()
    inputs = {}
    for seed in range(2):
        torch.manual_seed(seed)
        architecture = dict(width=4, modes=3, layers=1)
        model = ControlledFNO(**architecture).eval()
        folder = root / "heat1d" / f"seed_{seed}"
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "split_coefficients.npz", test=coefficients)
        state = PROBLEMS["heat1d"].reference(torch.tensor(coefficients).float(), 0, 16)
        reference = PROBLEMS["heat1d"].reference(torch.tensor(coefficients).float(), 0.2, 16)
        with torch.no_grad():
            direct = model(state, torch.ones(8))
            composed = model(model(state, torch.full((8,), 0.5)), torch.full((8,), 0.5))
        for arm in ("supervised", "pino", "composition"):
            for selection in ["best_validation"] if arm == "supervised" else ["best_validation", "matched_accuracy"]:
                target = folder / arm
                target.mkdir(exist_ok=True)
                path = target / f"{selection}.pt"
                torch.save(
                    dict(
                        architecture=architecture,
                        model=model.state_dict(),
                        step=1,
                        validation_rel_l2=0.01,
                        source_sha256="fixture",
                    ),
                    path,
                )
                inputs[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                np.savez_compressed(
                    target / f"test_{selection}.npz",
                    **{
                        "1.0_prediction": direct.numpy(),
                        "1.0_composed": composed.numpy(),
                        "1.0_reference": reference.numpy(),
                        "1.0_semigroup": semigroup_error(direct, composed).numpy(),
                    },
                )
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_rollout_revision.py"
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--paper",
            str(paper),
            "--problems",
            "heat1d",
            "--seeds",
            "0",
            "1",
            "--grid",
            "16",
            "--replicates",
            "100",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = paper / "revision_evidence/rollout"
    rows = pd.read_csv(evidence / "rollout_samples.csv")
    assert len(rows) == 80
    assert rows.groupby(["seed", "checkpoint_model_sha256"]).ngroups == 2
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert manifest["selected_checkpoint_labels"] == 10
    assert manifest["maximum_queried_fraction"] == 1
    assert manifest["checkpoints_and_saved_predictions_unchanged"]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in inputs.items())
    summaries = json.loads((evidence / "rollout_summary.json").read_text())
    assert len(summaries) == 5
    assert (paper / "generated/controlled_rollout.tex").exists()
