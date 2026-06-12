from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import PairDataset, TrajectoryCollection, WindowDataset
from .evaluate import evaluate_diagnostics, evaluate_reference
from .metrics import add_reference_relative_columns
from .models import OversmoothWrapper, build_model
from .normalization import estimate_normalizer
from .tasks import TaskSpec
from .training import train_one

def spatial_limit_for(task: TaskSpec, args: argparse.Namespace) -> int:
    return args.spatial_limit_2d if task.dim == 2 else args.spatial_limit_3d


def inspect_data(data_root: Path, tasks: Dict[str, TaskSpec], selected: List[str], args: argparse.Namespace, outdir: Path) -> None:
    rows = []
    for name in selected:
        task = tasks[name]
        for split in ["train", "val", "test"]:
            try:
                coll = TrajectoryCollection(data_root, task, split, spatial_limit_for(task, args))
                rows.extend(coll.describe())
            except Exception as e:
                rows.append({"task": name, "split": split, "error": str(e)})
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "data_inspection.csv", index=False)
    print(df.to_string(index=False))


def run_task(task: TaskSpec, args: argparse.Namespace, outdir: Path) -> List[Dict[str, Any]]:
    print(f"\n=== Task: {task.name} ({task.label}) ===")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    spat_lim = spatial_limit_for(task, args)
    train_coll = TrajectoryCollection(args.data_root, task, "train", spat_lim)
    val_coll = TrajectoryCollection(args.data_root, task, "val", spat_lim)
    test_coll = TrajectoryCollection(args.data_root, task, "test", spat_lim)

    train_ds = PairDataset(train_coll, args.max_train_pairs, args.max_delta, seed=args.seed + 1)
    val_ds = PairDataset(val_coll, args.max_val_pairs, args.max_delta, seed=args.seed + 2)
    test_windows = WindowDataset(test_coll, args.max_test_windows, horizon=max(2, args.max_delta), seed=args.seed + 3)

    sample = train_ds[0]["x"]
    channels = int(sample.shape[0])
    spatial_shape = tuple(int(x) for x in sample.shape[1:])
    print(f"  sample shape C,*spatial = {tuple(sample.shape)}; pairs train={len(train_ds)} val={len(val_ds)} test_windows={len(test_windows)}")

    norm = estimate_normalizer(train_ds, n_samples=min(args.norm_samples, len(train_ds)))
    (outdir / "normalizers").mkdir(parents=True, exist_ok=True)
    torch.save({"mean": norm.mean, "std": norm.std}, outdir / "normalizers" / f"{task.name}.pt")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
    test_loader = DataLoader(test_windows, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    rows: List[Dict[str, Any]] = []
    rows.append(evaluate_reference(test_loader, task, args.max_delta, device))

    for model_name in args.models:
        if model_name == "deeponet" and task.dim == 3 and not args.deeponet_3d:
            print(f"  [skip] deeponet on {task.name} because --deeponet-3d is not set")
            continue
        for variant, epochs, width_mult in [
            ("well", args.epochs_well, 1.0),
            ("underfit", args.epochs_underfit, args.underfit_width_mult),
        ]:
            width = max(4, int(args.width * width_mult))
            modes = min(args.modes, max(2, min(spatial_shape) // 2))
            print(f"  training {model_name}/{variant}: width={width}, modes={modes}, epochs={epochs}")
            model = build_model(model_name, task.dim, channels, spatial_shape, width=width, modes=modes)
            ckpt = outdir / "checkpoints" / task.name / f"{model_name}_{variant}.pt"
            meta = train_one(model, train_loader, val_loader, norm, epochs=epochs, lr=args.lr, device=device, out_ckpt=ckpt)
            metrics = evaluate_diagnostics(model, test_loader, norm, task, args.max_delta, device, model_name, variant)
            metrics.update({"best_val": meta["best_val"], "best_epoch": meta["best_epoch"]})
            rows.append(metrics)
            pd.DataFrame(meta["history"]).to_csv(outdir / "checkpoints" / task.name / f"{model_name}_{variant}_history.csv", index=False)

            if variant == "well" and args.eval_oversmoothed:
                smooth = OversmoothWrapper(model, task.dim, passes=args.oversmooth_passes).to(device)
                sm = evaluate_diagnostics(smooth, test_loader, norm, task, args.max_delta, device, model_name, "oversmoothed")
                sm.update({"best_val": meta["best_val"], "best_epoch": meta["best_epoch"]})
                rows.append(sm)

    add_reference_relative_columns(rows)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run diagnostic consistency experiments for learned PDE propagators.")
    p.add_argument("--data-root", type=Path, default=Path.home() / "Desktop/research_work/COMET/comet_ncs_codebase/data/raw")


def apply_smoke_config(args: argparse.Namespace) -> argparse.Namespace:
    """Reduce a run configuration to a quick debugging workload."""
    args.max_train_pairs = min(args.max_train_pairs, 64)
    args.max_val_pairs = min(args.max_val_pairs, 32)
    args.max_test_windows = min(args.max_test_windows, 16)
    args.epochs_well = min(args.epochs_well, 2)
    args.epochs_underfit = min(args.epochs_underfit, 1)
    args.width = min(args.width, 16)
    args.modes = min(args.modes, 6)
    args.spatial_limit_2d = min(args.spatial_limit_2d, 32)
    args.spatial_limit_3d = min(args.spatial_limit_3d, 16)
    return args


def run_experiment(args: argparse.Namespace, tasks: Dict[str, TaskSpec]) -> pd.DataFrame:
    """Run the configured benchmark and return the metrics table."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    with open(args.outdir / "args.json", "w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for name in args.tasks:
        try:
            rows = run_task(tasks[name], args, args.outdir)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(args.outdir / "metrics_summary_partial.csv", index=False)
        except Exception as e:  # keep long multi-task runs from failing entirely
            print(f"[ERROR] Task {name} failed: {e}")
            all_rows.append({"task": name, "model": "ERROR", "variant": "ERROR", "error": str(e)})
    df = pd.DataFrame(all_rows)
    df.to_csv(args.outdir / "metrics_summary.csv", index=False)
    print("\n=== Metrics summary ===")
    print(df.to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.2f} minutes. Outputs in {args.outdir}")
    return df
