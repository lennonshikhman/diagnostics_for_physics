"""Concise progress reporting for resumable benchmark campaigns."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

TASK_DIMS = {
    "toy2d": 2,
    "adv2d": 2,
    "shallow2d": 2,
    "active2d": 2,
    "cns3d": 3,
    "mhd3d": 3,
}


def _service_state(unit: str | None) -> str:
    if not unit:
        return "not_requested"
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=ActiveState,SubState", "--value"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return "/".join(values) if values else "not_found"


def _expected_units(arguments: Dict[str, Any]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int, str]]]:
    tasks = [str(task) for task in arguments.get("tasks", [])]
    configured_seeds = arguments.get("seeds")
    seeds = [int(seed) for seed in configured_seeds] if configured_seeds else [int(arguments.get("seed", 0))]
    models = [str(model) for model in arguments.get("models", [])]
    deeponet_3d = bool(arguments.get("deeponet_3d", False))
    task_seed_units = [(task, seed) for task in tasks for seed in seeds]
    model_units = [
        (task, seed, model)
        for task, seed in task_seed_units
        for model in models
        if not (model == "deeponet" and TASK_DIMS.get(task) == 3 and not deeponet_3d)
    ]
    return task_seed_units, model_units


def _latest_artifact(outdir: Path) -> Dict[str, Any] | None:
    candidates = [
        path
        for pattern in ("checkpoints/**/*.pt", "checkpoints/**/*_history.csv", "progress/**/*.json")
        for path in outdir.glob(pattern)
        if path.is_file()
    ]
    if not candidates:
        return None
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    modified = path.stat().st_mtime
    return {
        "path": str(path.relative_to(outdir)),
        "modified_unix_time": modified,
        "age_seconds": max(0.0, time.time() - modified),
    }


def collect_progress(outdir: Path | str, *, unit: str | None = None) -> Dict[str, Any]:
    outdir = Path(outdir)
    args_path = outdir / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError(f"Run arguments not found: {args_path}")
    arguments = json.loads(args_path.read_text(encoding="utf-8"))
    task_seed_units, model_units = _expected_units(arguments)

    completed_task_seeds = [
        (task, seed)
        for task, seed in task_seed_units
        if (outdir / "progress" / task / f"seed_{seed}" / "complete.json").is_file()
    ]
    completed_models = [
        (task, seed, model)
        for task, seed, model in model_units
        if (outdir / "progress" / task / f"seed_{seed}" / f"{model}.json").is_file()
    ]
    completed_references = [
        (task, seed)
        for task, seed in task_seed_units
        if (outdir / "progress" / task / f"seed_{seed}" / "reference.json").is_file()
    ]

    failures: List[Dict[str, Any]] = []
    errors_path = outdir / "errors.json"
    if errors_path.is_file():
        loaded = json.loads(errors_path.read_text(encoding="utf-8"))
        failures = loaded if isinstance(loaded, list) else [{"error": "invalid errors.json"}]

    expected_model_count = len(model_units)
    complete_model_count = len(completed_models)
    return {
        "outdir": str(outdir.resolve()),
        "service": _service_state(unit),
        "task_seeds_complete": len(completed_task_seeds),
        "task_seeds_expected": len(task_seed_units),
        "references_complete": len(completed_references),
        "models_complete": complete_model_count,
        "models_expected": expected_model_count,
        "percent_complete": 100.0 * complete_model_count / expected_model_count if expected_model_count else 0.0,
        "failures": failures,
        "latest_artifact": _latest_artifact(outdir),
        "tables_complete": (outdir / "paper_tables" / "table_manifest.json").is_file(),
    }


def _format_progress(progress: Dict[str, Any]) -> str:
    latest = progress["latest_artifact"]
    latest_text = "none"
    if latest is not None:
        latest_text = f"{latest['path']} ({latest['age_seconds'] / 60.0:.1f} min ago)"
    return "\n".join(
        [
            f"Run: {progress['outdir']}",
            f"Service: {progress['service']}",
            (
                f"Progress: {progress['models_complete']}/{progress['models_expected']} models "
                f"({progress['percent_complete']:.1f}%), "
                f"{progress['task_seeds_complete']}/{progress['task_seeds_expected']} task-seeds"
            ),
            f"References: {progress['references_complete']}/{progress['task_seeds_expected']}",
            f"Latest: {latest_text}",
            f"Failures: {len(progress['failures'])}",
            f"Tables: {'complete' if progress['tables_complete'] else 'pending'}",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report resumable benchmark progress.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--unit", default=None, help="Optional user systemd unit name")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    progress = collect_progress(args.outdir, unit=args.unit)
    print(json.dumps(progress, indent=2) if args.json else _format_progress(progress))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["collect_progress", "main", "parse_args"]
