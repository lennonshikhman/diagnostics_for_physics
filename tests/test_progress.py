from __future__ import annotations

import json
from pathlib import Path

from diagnostic_suite.progress import collect_progress


def test_progress_counts_applicable_model_units(tmp_path: Path) -> None:
    outdir = tmp_path / "run"
    outdir.mkdir()
    (outdir / "args.json").write_text(
        json.dumps(
            {
                "tasks": ["adv2d", "cns3d"],
                "models": ["fno", "deeponet"],
                "seeds": [0],
                "seed": 0,
                "deeponet_3d": False,
            }
        ),
        encoding="utf-8",
    )
    progress_dir = outdir / "progress" / "adv2d" / "seed_0"
    progress_dir.mkdir(parents=True)
    (progress_dir / "reference.json").write_text("{}", encoding="utf-8")
    (progress_dir / "fno.json").write_text("{}", encoding="utf-8")

    progress = collect_progress(outdir)
    assert progress["task_seeds_expected"] == 2
    assert progress["models_expected"] == 3
    assert progress["models_complete"] == 1
    assert progress["references_complete"] == 1
