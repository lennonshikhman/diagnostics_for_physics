import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from diagnostic_suite import __version__


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("invalid_scaling", [False, True])
def test_external_export_reports_existing_scaling_and_rejects_invalid_values(tmp_path, monkeypatch, invalid_scaling):
    module = load_script("analyze_external_revision")
    root, paper = tmp_path / "run", tmp_path / "paper"
    root.mkdir()
    for name in ("data_contracts", "progress"):
        (root / name).mkdir()
    for name in ("args", "provenance"):
        (root / f"{name}.json").write_text("{}")
    pd.DataFrame().to_csv(root / "metrics_across_seeds.csv", index=False)
    combinations = [
        (task, model, variant, seed, horizon)
        for task, model, variant, seed, horizon in itertools.product(
            module.TASKS, module.MODELS, module.VARIANTS, range(3), (1, 2, 4)
        )
        if not (task in ("cns3d", "mhd3d") and model == "deeponet")
    ]
    combinations += [
        (task, "reference", "reference", seed, horizon)
        for task, seed, horizon in itertools.product(module.TASKS, range(3), (1, 2, 4))
    ]
    rows = []
    for task, model, variant, seed, horizon in combinations:
        row = dict(
            task=task, model=model, variant=variant, seed=seed, horizon_steps=horizon, rel_l2_n=128, training_seconds=1
        )
        for metric in [
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
            "generator_ratio_to_reference",
            "scaling_law",
        ]:
            available = metric != "scaling_law" or task == "adv2d"
            row[metric] = 0.125 if available else np.nan
            row[metric + "_available"] = available
            row[metric + "_invalid"] = 0
        rows.append(row)
    if invalid_scaling:
        rows[0]["scaling_law"] = np.nan
    pd.DataFrame(rows).to_csv(root / "metrics_summary.csv", index=False)
    monkeypatch.setattr("sys.argv", ["export", "--root", str(root), "--paper", str(paper)])
    if invalid_scaling:
        with pytest.raises(ValueError, match="scaling_law"):
            module.main()
    else:
        module.main()
        table = (paper / "generated/external_main.tex").read_text()
        assert "Scaling" in table
        adv = next(line for line in table.splitlines() if line.startswith("ADV2D & Standard"))
        assert adv.endswith(r"\(0.125\pm0\)\\")
        cns = next(line for line in table.splitlines() if line.startswith("CNS3D & Standard"))
        assert cns.endswith(r"---\\")
        assert json.loads((paper / "revision_evidence/external/checks.json").read_text())["rows"] == 531


def test_release_version_matches_package_and_citation_metadata():
    root = Path(__file__).resolve().parents[1]
    assert f'version = "{__version__}"' in (root / "pyproject.toml").read_text()
    assert f"version: {__version__}" in (root / "CITATION.cff").read_text()
    assert json.loads((root / ".zenodo.json").read_text())["version"] == __version__
