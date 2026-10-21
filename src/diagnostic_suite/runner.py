"""Reproducible benchmark orchestration with fail-fast scientific checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import __version__
from .data import (
    CachedDataset,
    PairDataset,
    TrajectoryCollection,
    WindowDataset,
    assert_disjoint_collections,
)
from .evaluate import evaluate_diagnostics, evaluate_reference
from .metrics import add_reference_relative_columns
from .models import OversmoothWrapper, build_model
from .normalization import estimate_normalizer
from .tasks import TaskSpec
from .training import train_one


class ExperimentRunError(RuntimeError):
    """Raised after partial results are saved when one or more tasks fail."""


def spatial_limit_for(task: TaskSpec, args: argparse.Namespace) -> int:
    return args.spatial_limit_2d if task.dim == 2 else args.spatial_limit_3d


def _selected_seeds(args: argparse.Namespace) -> List[int]:
    configured = getattr(args, "seeds", None)
    seeds = [int(value) for value in configured] if configured else [int(args.seed)]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Seeds must be unique; received {seeds}")
    return seeds


def _selected_horizons(args: argparse.Namespace) -> List[int]:
    configured = getattr(args, "eval_horizons", None)
    if configured:
        horizons = [int(value) for value in configured]
    else:
        horizons = list(dict.fromkeys([1, min(2, int(args.max_delta)), int(args.max_delta)]))
    if not horizons or any(value < 1 or value > int(args.max_delta) for value in horizons):
        raise ValueError(f"Evaluation horizons must be unique integers in [1, {args.max_delta}]; received {horizons}")
    if len(set(horizons)) != len(horizons):
        raise ValueError(f"Evaluation horizons must be unique; received {horizons}")
    return sorted(horizons)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_fingerprint() -> str:
    """Hash the executable package source used by a run.

    A Git revision alone is insufficient while publication repairs are present
    in a dirty worktree.  This fingerprint also prevents resume caches from
    being reused after any package source file changes.
    """

    source_root = Path(__file__).resolve().parent
    paths = sorted(source_root.rglob("*.py"))
    if not paths:
        raise RuntimeError(f"Cannot fingerprint an empty package source directory: {source_root}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _provenance(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "created_unix_time": time.time(),
        "git_revision": _git_revision(),
        "git_dirty": _git_dirty(),
        "source_sha256": _source_fingerprint(),
        "package_version": __version__,
        "h5py": h5py.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_requested": args.device,
        "command": sys.argv,
    }


def _jsonable_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _resume_signature(
    args: argparse.Namespace,
    *,
    task: TaskSpec | None = None,
    data_manifest: Sequence[Dict[str, Any]] = (),
    split_contracts: Sequence[Dict[str, Any]] = (),
    seed: int | None = None,
) -> str:
    arguments = _jsonable_arguments(args)
    # These control artifact location/behavior rather than scientific content.
    arguments.pop("outdir", None)
    arguments.pop("resume", None)
    payload = {
        "arguments": arguments,
        "source_sha256": _source_fingerprint(),
        "package_version": __version__,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "pandas": pd.__version__,
        },
        "task": asdict(task) if task is not None else None,
        "seed": seed,
        "data": [{key: row[key] for key in ("relative_path", "size_bytes", "sha256")} for row in data_manifest],
        "splits": list(split_contracts),
    }
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except TypeError as error:
        raise ValueError("Resumable benchmark tasks require a serializable, declarative configuration") from error
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_progress(path: Path, signature: str, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "created_unix_time": time.time(),
        "rows": list(rows),
    }
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_progress(path: Path, signature: str) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_signature = payload.get("signature")
    if actual_signature != signature:
        raise RuntimeError(
            f"Resume artifact signature mismatch in {path}; code, data, task, environment, or arguments changed. "
            "Use a new --outdir for the revised computation"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Invalid resume artifact rows in {path}")
    return rows


def _validate_collection_contracts(collections: Sequence[TrajectoryCollection]) -> TaskSpec:
    first = collections[0]
    for collection in collections[1:]:
        if collection.channel_names != first.channel_names:
            raise RuntimeError(
                f"Split channel mismatch for {first.task.name}: {first.channel_names} versus {collection.channel_names}"
            )
        if not np.allclose(
            collection.grid_spacing,
            first.grid_spacing,
            rtol=1.0e-6,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"Split grid-spacing mismatch for {first.task.name}: "
                f"{first.grid_spacing} versus {collection.grid_spacing}"
            )
        if collection.time_step is None or first.time_step is None:
            if collection.time_step != first.time_step:
                raise RuntimeError(f"Split time-step mismatch for {first.task.name}")
        elif not np.isclose(collection.time_step, first.time_step, rtol=1.0e-6, atol=1.0e-12):
            raise RuntimeError(
                f"Split time-step mismatch for {first.task.name}: {first.time_step} versus {collection.time_step}"
            )
    runtime_task = first.runtime_task()
    requires_physical_grid = runtime_task.balance is not None or any(
        constraint.kind == "divergence" for constraint in runtime_task.constraints
    )
    if requires_physical_grid:
        defaulted = [
            str(reader.path)
            for collection in collections
            for reader in collection.readers
            if reader.grid_spacing_source == "default_unit"
        ]
        if defaulted:
            raise RuntimeError(
                "Physical balance/divergence diagnostics require configured domain "
                f"metadata or file coordinates; unit spacing was only a fallback for {defaulted}"
            )
    if runtime_task.balance is not None and runtime_task.time_step is None:
        raise RuntimeError(f"Physical balance for {runtime_task.name} requires a configured or file-derived time step")
    return runtime_task


def inspect_data(
    data_root: Path,
    tasks: Dict[str, TaskSpec],
    selected: List[str],
    args: argparse.Namespace,
    outdir: Path,
) -> None:
    """Inspect data contracts and include an explicit split-overlap audit."""

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for name in selected:
        task = tasks[name]
        collections: Dict[str, TrajectoryCollection] = {}
        for split in ["train", "val", "test"]:
            try:
                collection = TrajectoryCollection(
                    data_root,
                    task,
                    split,
                    spatial_limit_for(task, args),
                )
                collections[split] = collection
                rows.extend(collection.describe())
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                rows.append({"task": name, "split": split, "error": message})
                errors.append(f"{name}/{split}: {message}")
        if len(collections) == 3:
            try:
                assert_disjoint_collections(collections["train"], collections["val"], collections["test"])
                _validate_collection_contracts(list(collections.values()))
                rows.append(
                    {
                        "task": name,
                        "split": "AUDIT",
                        "split_disjoint": True,
                        "schema_consistent": True,
                    }
                )
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                rows.append(
                    {
                        "task": name,
                        "split": "AUDIT",
                        "split_disjoint": False,
                        "schema_consistent": False,
                        "error": message,
                    }
                )
                errors.append(f"{name}/AUDIT: {message}")
    dataframe = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(outdir / "data_inspection.csv", index=False)
    print(dataframe.to_string(index=False))
    if errors:
        raise ExperimentRunError("Data inspection found errors; see data_inspection.csv. " + " | ".join(errors))


def _loader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _model_seed(seed: int, model_name: str, variant: str) -> int:
    model_offsets = {"fno": 10_000, "deeponet": 20_000, "unet": 30_000, "resnet": 40_000}
    del variant
    # Match initialization and minibatch streams between training-budget arms.
    return seed + model_offsets[model_name]


@lru_cache(maxsize=None)
def _sha256(path_text: str, size_bytes: int, modified_ns: int, changed_ns: int) -> str:
    del size_bytes, modified_ns, changed_ns
    path = Path(path_text)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_manifest(
    collections: Sequence[TrajectoryCollection],
    data_root: Path,
    *,
    include_hashes: bool,
) -> List[Dict[str, Any]]:
    root = data_root.resolve()
    paths = sorted({reader.path.resolve() for collection in collections for reader in collection.readers})
    rows: List[Dict[str, Any]] = []
    for path in paths:
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = str(path)
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": stat.st_size,
                "modified_unix_time": stat.st_mtime,
                "sha256": _sha256(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if include_hashes
                else "not_requested",
            }
        )
    return rows


def run_task(
    task: TaskSpec,
    args: argparse.Namespace,
    outdir: Path,
    *,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    seed = int(args.seed if seed is None else seed)
    progress_dir = outdir / "progress" / task.name / f"seed_{seed}"
    completed_progress = progress_dir / "complete.json"
    if args.scaling_factor <= 0 or np.isclose(args.scaling_factor, 1.0):
        raise ValueError("--scaling-factor must be positive and nontrivial (not 1.0)")
    _set_seed(seed)
    print(f"\n=== Task: {task.name} ({task.label}); seed={seed} ===")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    spatial_limit = spatial_limit_for(task, args)
    train_collection = TrajectoryCollection(args.data_root, task, "train", spatial_limit)
    validation_collection = TrajectoryCollection(args.data_root, task, "val", spatial_limit)
    test_collection = TrajectoryCollection(args.data_root, task, "test", spatial_limit)
    assert_disjoint_collections(train_collection, validation_collection, test_collection)
    runtime_task = _validate_collection_contracts([train_collection, validation_collection, test_collection])

    split_rows: List[Dict[str, Any]] = []
    for collection in (train_collection, validation_collection, test_collection):
        split_rows.extend(collection.describe())
    resume = bool(getattr(args, "resume", False))
    manifest = _data_manifest(
        [train_collection, validation_collection, test_collection],
        args.data_root,
        include_hashes=resume or bool(getattr(args, "hash_data", False)),
    )
    signature = (
        _resume_signature(
            args,
            task=runtime_task,
            data_manifest=manifest,
            split_contracts=split_rows,
            seed=seed,
        )
        if resume
        else ""
    )
    if resume:
        # Validate every reusable artifact before replacing any original contract.
        for artifact in sorted(progress_dir.glob("*.json")):
            _read_progress(artifact, signature)
        if completed_progress.is_file():
            print(f"  [resume] verified code/data/contracts for {task.name}, seed={seed}")
            return _read_progress(completed_progress, signature)
    split_dir = outdir / "data_contracts" / runtime_task.name
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(split_dir / f"seed_{seed}.csv", index=False)
    pd.DataFrame(manifest).to_csv(split_dir / f"seed_{seed}_files.csv", index=False)

    train_dataset = PairDataset(
        train_collection,
        args.max_train_pairs,
        args.max_delta,
        seed=seed + 1,
    )
    validation_dataset = PairDataset(
        validation_collection,
        args.max_val_pairs,
        args.max_delta,
        seed=seed + 2,
    )
    horizons = _selected_horizons(args)
    test_windows = WindowDataset(
        test_collection,
        args.max_test_windows,
        horizon=max(horizons),
        seed=seed + 3,
    )
    if bool(getattr(args, "cache_data", False)):
        print("  caching the selected physical pairs and test windows", flush=True)
        train_dataset = CachedDataset(train_dataset)
        validation_dataset = CachedDataset(validation_dataset)
        test_windows = CachedDataset(test_windows)

    sample = train_dataset[0]["x"]
    channels = int(sample.shape[0])
    spatial_shape = tuple(int(value) for value in sample.shape[1:])
    print(
        f"  sample shape C,*spatial={tuple(sample.shape)}; "
        f"pairs train={len(train_dataset)} val={len(validation_dataset)} "
        f"test_windows={len(test_windows)}; horizons={horizons}"
    )

    normalizer = estimate_normalizer(
        train_dataset,
        n_samples=min(args.norm_samples, len(train_dataset)),
    )
    normalizer_dir = outdir / "normalizers" / runtime_task.name
    normalizer_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mean": normalizer.mean,
            "std": normalizer.std,
            "channel_names": runtime_task.channel_names,
            "grid_spacing": runtime_task.grid_spacing,
            "time_step": runtime_task.time_step,
            "seed": seed,
        },
        normalizer_dir / f"seed_{seed}.pt",
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_windows,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows: List[Dict[str, Any]] = []
    reference_progress = progress_dir / "reference.json"
    if bool(getattr(args, "resume", False)) and reference_progress.is_file():
        print(f"  [resume] reference diagnostics for {runtime_task.name}, seed={seed}")
        rows.extend(_read_progress(reference_progress, signature))
    else:
        reference_rows: List[Dict[str, Any]] = []
        for horizon in horizons:
            reference_rows.append(
                evaluate_reference(
                    test_loader,
                    runtime_task,
                    args.max_delta,
                    device,
                    eval_horizon=horizon,
                    seed=seed,
                )
            )
        rows.extend(reference_rows)
        if bool(getattr(args, "resume", False)):
            _write_progress(reference_progress, signature, reference_rows)

    for model_name in args.models:
        if model_name == "deeponet" and runtime_task.dim == 3 and not args.deeponet_3d:
            print(f"  [skip] deeponet on {runtime_task.name}; use --deeponet-3d to enable")
            continue
        model_progress = progress_dir / f"{model_name}.json"
        if bool(getattr(args, "resume", False)) and model_progress.is_file():
            print(f"  [resume] {model_name} on {runtime_task.name}, seed={seed}")
            rows.extend(_read_progress(model_progress, signature))
            continue
        model_rows: List[Dict[str, Any]] = []
        for variant, epochs, width_multiplier in [
            ("well", args.epochs_well, 1.0),
            ("underfit", args.epochs_underfit, args.underfit_width_mult),
        ]:
            training_seed = _model_seed(seed, model_name, variant)
            _set_seed(training_seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                generator=_loader_generator(training_seed),
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            width = max(4, int(args.width * width_multiplier))
            modes = min(args.modes, max(2, min(spatial_shape) // 2))
            print(f"  training {model_name}/{variant}: width={width}, modes={modes}, epochs={epochs}")
            model = build_model(
                model_name,
                runtime_task.dim,
                channels,
                spatial_shape,
                width=width,
                modes=modes,
            )
            parameter_count = _parameter_count(model)
            checkpoint_dir = outdir / "checkpoints" / runtime_task.name / f"seed_{seed}"
            checkpoint = checkpoint_dir / f"{model_name}_{variant}.pt"
            training_started = time.perf_counter()
            metadata = train_one(
                model,
                train_loader,
                validation_loader,
                normalizer,
                epochs=epochs,
                lr=args.lr,
                device=device,
                out_ckpt=checkpoint,
            )
            training_seconds = time.perf_counter() - training_started
            pd.DataFrame(metadata["history"]).to_csv(
                checkpoint_dir / f"{model_name}_{variant}_history.csv",
                index=False,
            )

            for horizon in horizons:
                evaluation_started = time.perf_counter()
                metrics = evaluate_diagnostics(
                    model,
                    test_loader,
                    normalizer,
                    runtime_task,
                    args.max_delta,
                    device,
                    model_name,
                    variant,
                    eval_horizon=horizon,
                    seed=seed,
                    scaling_factor=args.scaling_factor,
                )
                metrics.update(
                    {
                        "best_val": metadata["best_val"],
                        "best_epoch": metadata["best_epoch"],
                        "parameter_count": parameter_count,
                        "model_seed": training_seed,
                        "training_seconds": training_seconds,
                        "evaluation_seconds": time.perf_counter() - evaluation_started,
                    }
                )
                model_rows.append(metrics)

                if variant == "well" and args.eval_oversmoothed:
                    smoothed = OversmoothWrapper(
                        model,
                        runtime_task.dim,
                        passes=args.oversmooth_passes,
                        periodic=runtime_task.periodic,
                    ).to(device)
                    smooth_metrics = evaluate_diagnostics(
                        smoothed,
                        test_loader,
                        normalizer,
                        runtime_task,
                        args.max_delta,
                        device,
                        model_name,
                        "oversmoothed",
                        eval_horizon=horizon,
                        seed=seed,
                        scaling_factor=args.scaling_factor,
                    )
                    smooth_metrics.update(
                        {
                            "best_val": metadata["best_val"],
                            "best_epoch": metadata["best_epoch"],
                            "parameter_count": parameter_count,
                            "model_seed": training_seed,
                            "training_seconds": training_seconds,
                        }
                    )
                    model_rows.append(smooth_metrics)

        rows.extend(model_rows)
        if bool(getattr(args, "resume", False)):
            _write_progress(model_progress, signature, model_rows)

    add_reference_relative_columns(rows)
    if bool(getattr(args, "resume", False)):
        _write_progress(completed_progress, signature, rows)
    return rows


def apply_smoke_config(args: argparse.Namespace) -> argparse.Namespace:
    """Reduce a run configuration to a quick, representative workload."""

    args.max_train_pairs = min(args.max_train_pairs, 64)
    args.max_val_pairs = min(args.max_val_pairs, 32)
    args.max_test_windows = min(args.max_test_windows, 16)
    args.epochs_well = min(args.epochs_well, 2)
    args.epochs_underfit = min(args.epochs_underfit, 1)
    args.width = min(args.width, 16)
    args.modes = min(args.modes, 6)
    args.spatial_limit_2d = min(args.spatial_limit_2d, 32)
    args.spatial_limit_3d = min(args.spatial_limit_3d, 16)
    if getattr(args, "eval_horizons", None):
        args.eval_horizons = [value for value in args.eval_horizons if value <= args.max_delta]
    return args


def _write_seed_summary(dataframe: pd.DataFrame, path: Path) -> None:
    if dataframe.empty or "seed" not in dataframe:
        pd.DataFrame().to_csv(path, index=False)
        return
    group_columns = ["task", "model", "variant", "horizon_steps", "horizon_time"]
    excluded = set(group_columns) | {"seed", "best_epoch"}
    numeric_columns = [
        column
        for column in dataframe.select_dtypes(include=[np.number]).columns
        if column not in excluded and not column.endswith("_n")
    ]
    if not numeric_columns:
        pd.DataFrame().to_csv(path, index=False)
        return
    summary = dataframe.groupby(group_columns, dropna=False)[numeric_columns].agg(["mean", "std", "count"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary.copy().reset_index().to_csv(path, index=False)


def run_experiment(args: argparse.Namespace, tasks: Dict[str, TaskSpec]) -> pd.DataFrame:
    """Run all task/seed combinations and fail nonzero after saving any errors."""

    args.outdir.mkdir(parents=True, exist_ok=True)
    resume = bool(getattr(args, "resume", False))
    if (args.outdir / "args.json").exists() and not resume:
        raise FileExistsError("Output directory already contains a run; use --resume or a new --outdir")
    attempt_dir = args.outdir / "invocations" / str(time.time_ns())
    attempt_dir.mkdir(parents=True)
    arguments = json.dumps(_jsonable_arguments(args), indent=2) + "\n"
    provenance = json.dumps(_provenance(args), indent=2) + "\n"
    (attempt_dir / "args.json").write_text(arguments, encoding="utf-8")
    (attempt_dir / "provenance.json").write_text(provenance, encoding="utf-8")
    if not (args.outdir / "args.json").exists():
        (args.outdir / "args.json").write_text(arguments, encoding="utf-8")
        (args.outdir / "provenance.json").write_text(provenance, encoding="utf-8")
    # A rejected resume must leave the previous numerical results and provenance intact.
    artifact_dir = attempt_dir if resume and (args.outdir / "metrics_summary.csv").exists() else args.outdir

    all_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    started = time.time()
    for name in args.tasks:
        if name not in tasks:
            errors.append({"task": name, "seed": None, "error": f"Unknown task {name!r}"})
            continue
        for seed in _selected_seeds(args):
            try:
                all_rows.extend(run_task(tasks[name], args, args.outdir, seed=seed))
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                print(f"[ERROR] Task {name}, seed {seed} failed: {message}", file=sys.stderr)
                failure = {
                    "task": name,
                    "model": "ERROR",
                    "variant": "ERROR",
                    "seed": seed,
                    "error": message,
                }
                all_rows.append(failure)
                errors.append(failure)
            pd.DataFrame(all_rows).to_csv(
                artifact_dir / "metrics_summary_partial.csv",
                index=False,
            )

    dataframe = pd.DataFrame(all_rows)
    destination = artifact_dir if errors else args.outdir
    dataframe.to_csv(destination / "metrics_summary.csv", index=False)
    _write_seed_summary(
        dataframe[dataframe.get("model", pd.Series(dtype=object)) != "ERROR"],
        destination / "metrics_across_seeds.csv",
    )
    elapsed_minutes = (time.time() - started) / 60.0
    print("\n=== Metrics summary ===")
    display_columns = [
        column
        for column in (
            "task",
            "model",
            "variant",
            "seed",
            "horizon_steps",
            "rel_l2",
            "semigroup",
            "generator_ratio_to_reference",
            "energy_violation",
            "energy_reference_error",
            "integral_balance",
            "global_conserved_drift",
        )
        if column in dataframe
    ]
    print(dataframe[display_columns].to_string(index=False))
    print(f"\nFinished in {elapsed_minutes:.2f} minutes. Outputs: {args.outdir}")
    if errors:
        (destination / "errors.json").write_text(
            json.dumps(errors, indent=2) + "\n",
            encoding="utf-8",
        )
        raise ExperimentRunError(
            f"{len(errors)} task/seed run(s) failed. Attempt artifacts: {destination}; see errors.json."
        )
    errors_path = args.outdir / "errors.json"
    if errors_path.exists():
        errors_path.write_text("[]\n", encoding="utf-8")
    return dataframe


__all__ = [
    "ExperimentRunError",
    "apply_smoke_config",
    "inspect_data",
    "run_experiment",
    "run_task",
    "spatial_limit_for",
]
