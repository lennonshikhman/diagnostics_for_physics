from __future__ import annotations

import argparse
from pathlib import Path

from . import __version__
from .toy import make_toy_data

TASK_CHOICES = ["toy2d", "adv2d", "shallow2d", "active2d", "cns3d", "mhd3d"]
MODEL_CHOICES = ["fno", "deeponet", "unet", "resnet"]


def _task_name(value: str) -> str:
    if value == "ns2d":
        raise argparse.ArgumentTypeError(
            "The former ns2d configuration selected APEBench scalar advection. "
            "Use adv2d with regenerated results; old NS2D-labelled evidence cannot be reused."
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="diagnostic-suite",
        description="Run diagnostic consistency experiments for learned PDE time propagators.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Root containing benchmark datasets or toy_advdiff2d."
    )
    p.add_argument("--outdir", type=Path, default=Path("outputs_diagnostic_suite"))
    p.add_argument("--file-selection", type=Path, help="JSON mapping task names to explicit train/val/test file lists.")
    p.add_argument(
        "--cache-data", action="store_true", help="Cache the selected preprocessed pairs/windows in host memory."
    )
    p.add_argument("--tasks", nargs="+", default=["toy2d"], choices=TASK_CHOICES, type=_task_name)
    p.add_argument("--models", nargs="+", default=["fno", "unet", "resnet"], choices=MODEL_CHOICES)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Independent benchmark seeds. Overrides --seed when supplied.",
    )
    p.add_argument("--spatial-limit-2d", type=int, default=64)
    p.add_argument("--spatial-limit-3d", type=int, default=32)
    p.add_argument("--max-delta", type=int, default=4)
    p.add_argument(
        "--eval-horizons",
        nargs="+",
        type=int,
        default=None,
        help="Discrete horizons reported separately (default: 1, 2, and --max-delta).",
    )
    p.add_argument("--max-train-pairs", type=int, default=2048)
    p.add_argument("--max-val-pairs", type=int, default=512)
    p.add_argument("--max-test-windows", type=int, default=256)
    p.add_argument("--norm-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--epochs-well", type=int, default=20)
    p.add_argument("--epochs-underfit", type=int, default=2)
    p.add_argument("--underfit-width-mult", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-oversmoothed", dest="eval_oversmoothed", action="store_true", default=True)
    p.add_argument("--no-eval-oversmoothed", dest="eval_oversmoothed", action="store_false")
    p.add_argument("--oversmooth-passes", type=int, default=2)
    p.add_argument(
        "--scaling-factor",
        type=float,
        default=1.25,
        help="Positive amplitude/parameter factor used for configured benchmark scaling tests.",
    )
    p.add_argument("--deeponet-3d", action="store_true", help="Allow DeepONet on 3D data. This can be memory-heavy.")
    p.add_argument("--inspect-data", action="store_true", help="Inspect available data and exit.")
    p.add_argument(
        "--hash-data",
        action="store_true",
        help="Record SHA-256 checksums for every consumed dataset file (extra I/O).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Reuse progress after verifying code, environment, task, split and data hashes (extra I/O).",
    )
    p.add_argument("--smoke", action="store_true", help="Tiny run for debugging data loading and training.")
    p.add_argument(
        "--self-check",
        action="store_true",
        help="Run deterministic known-positive/known-negative diagnostic checks and exit.",
    )

    p.add_argument(
        "--make-toy", action="store_true", help="Generate a tiny toy_advdiff2d dataset under --data-root and exit."
    )
    p.add_argument("--toy-train", type=int, default=24)
    p.add_argument("--toy-val", type=int, default=8)
    p.add_argument("--toy-test", type=int, default=8)
    p.add_argument("--toy-steps", type=int, default=8)
    p.add_argument("--toy-grid", type=int, default=32)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_check:
        from .self_check import run_self_check

        run_self_check(verbose=True)
        return
    if args.make_toy:
        out = make_toy_data(
            args.data_root,
            n_train=args.toy_train,
            n_val=args.toy_val,
            n_test=args.toy_test,
            steps=args.toy_steps,
            grid=args.toy_grid,
            seed=args.seed,
        )
        print(f"Wrote toy dataset to {out}")
        return

    # Heavy torch imports are delayed until a run/inspection is actually requested.
    from .runner import apply_smoke_config, inspect_data, run_experiment
    from .tasks import apply_file_selection, build_task_specs

    if args.smoke:
        args = apply_smoke_config(args)

    tasks = build_task_specs(args.data_root, args.spatial_limit_2d, args.spatial_limit_3d)
    if args.file_selection:
        tasks = apply_file_selection(tasks, args.file_selection)
    if args.inspect_data:
        inspect_data(args.data_root, tasks, args.tasks, args, args.outdir)
        return
    run_experiment(args, tasks)


if __name__ == "__main__":  # pragma: no cover
    main()
