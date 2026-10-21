"""Reproducible low-error learned-propagator and PINO-style comparisons.

References are closed-form Fourier solutions or the Cole--Hopf solution of
periodic viscous Burgers. Test coefficients never enter fitting or selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from . import __version__
from .constants import EPS
from .metrics import energy_law_error, energy_reference_error, generator_discrepancy, rel_l2, semigroup_error
from .models import SpectralConvND
from .runner import _source_fingerprint
from .tasks import TaskSpec


@dataclass(frozen=True)
class ExactProblem:
    name: str
    viscosity: float
    speed: float = 0.0
    final_time: float = 0.2
    mean: float = 0.25
    modes: int = 6

    @property
    def nonlinear(self) -> bool:
        return self.name == "burgers1d"

    def coefficients(self, count: int, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        values = torch.randn((count, 2, self.modes), dtype=torch.float64, generator=generator)
        frequency = torch.arange(1, self.modes + 1, dtype=torch.float64)
        values = values / frequency.pow(1.5)
        if self.nonlinear:
            # phi_0 >= 0.4 everywhere, so Cole--Hopf remains nonsingular.
            values = 0.6 * values / values.abs().sum((1, 2), keepdim=True)
        else:
            rms = values.square().sum((1, 2), keepdim=True).sqrt() / math.sqrt(2)
            amplitude = 0.25 + 0.25 * torch.rand((count, 1, 1), dtype=torch.float64, generator=generator)
            values = values / rms * amplitude
        return values

    def reference(self, coefficients: torch.Tensor, horizon: torch.Tensor | float, grid: int = 64) -> torch.Tensor:
        dtype, device = coefficients.dtype, coefficients.device
        t = torch.as_tensor(horizon, dtype=dtype, device=device)
        t = t.expand(coefficients.shape[0]).reshape(-1, 1, 1)
        x = torch.arange(grid, dtype=dtype, device=device).reshape(1, 1, -1) / grid
        wave = 2 * math.pi * torch.arange(1, self.modes + 1, dtype=dtype, device=device).reshape(1, -1, 1)
        coordinate = x - (self.mean if self.nonlinear else self.speed) * t
        phase = wave * coordinate
        decay = torch.exp(-self.viscosity * wave.square() * t)
        a, b = coefficients[:, 0, :, None], coefficients[:, 1, :, None]
        oscillation = a * torch.cos(phase) + b * torch.sin(phase)
        if not self.nonlinear:
            return self.mean + (decay * oscillation).sum(1, keepdim=True)
        potential = 1 + (decay * oscillation).sum(1, keepdim=True)
        derivative = (decay * wave * (-a * torch.sin(phase) + b * torch.cos(phase))).sum(1, keepdim=True)
        return self.mean - 2 * self.viscosity * derivative / potential

    def rhs(self, state: torch.Tensor) -> torch.Tensor:
        grid = state.shape[-1]
        wave = 2 * math.pi * torch.fft.rfftfreq(grid, d=1 / grid, device=state.device, dtype=state.dtype)
        spectrum = torch.fft.rfft(state, dim=-1)
        first = torch.fft.irfft(1j * wave * spectrum, n=grid, dim=-1)
        second = torch.fft.irfft(-wave.square() * spectrum, n=grid, dim=-1)
        return self.viscosity * second - (state if self.nonlinear else self.speed) * first


PROBLEMS = {
    "heat1d": ExactProblem("heat1d", viscosity=0.01),
    "advection1d": ExactProblem("advection1d", viscosity=0.0, speed=0.3),
    "burgers1d": ExactProblem("burgers1d", viscosity=0.02, modes=4),
}


class ControlledFNO(nn.Module):
    """Periodic time-conditioned FNO with an exact identity at horizon zero.

    No PDE coefficients or analytic propagators appear in this architecture.
    The same architecture and initial weights are used by all training arms.
    """

    def __init__(self, width: int = 32, modes: int = 12, layers: int = 4, *, mean: float = 0.25, scale: float = 0.4):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("scale", torch.tensor(scale))
        self.lift = nn.Conv1d(2, width, 1)
        self.spectral = nn.ModuleList([SpectralConvND(1, width, width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(), nn.Conv1d(2 * width, 1, 1))

    def forward(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.reshape(-1, 1, 1).expand(state.shape[0], 1, state.shape[-1])
        h = self.lift(torch.cat(((state - self.mean) / self.scale, tau), dim=1))
        for spectral, local in zip(self.spectral, self.local, strict=True):
            h = F.gelu(spectral(h) + local(h))
        return state + self.scale * tau * self.project(h)


def _relative_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).square().flatten(1).mean(1) / (target.square().flatten(1).mean(1) + EPS)).mean()


@torch.no_grad()
def validation_error(model: nn.Module, problem: ExactProblem, coefficients: torch.Tensor, grid: int) -> float:
    initial = problem.reference(coefficients, 0, grid)
    errors = []
    for fraction in (0.125, 0.25, 0.5, 1.0):
        tau = torch.full((len(initial),), fraction, device=initial.device)
        errors.append(rel_l2(model(initial, tau), problem.reference(coefficients, fraction * problem.final_time, grid)))
    return torch.cat(errors).mean().item()


def _checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), **metadata), path)


def _torch_load(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def train_problem(outdir: Path, problem: ExactProblem, args: argparse.Namespace, seed: int) -> dict:
    directory = outdir / problem.name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: getattr(args, key)
        for key in (
            "grid",
            "n_train",
            "n_validation",
            "n_test",
            "dataset_seed",
            "width",
            "modes",
            "layers",
            "batch_size",
            "warmup_steps",
            "continuation_steps",
            "validation_every",
            "lr",
            "physics_weight",
            "composition_weight",
        )
    }
    completed = directory / "training_manifest.json"
    if completed.exists():
        prior = json.loads(completed.read_text())
        if not args.resume:
            raise FileExistsError(f"Completed study exists at {directory}; use --resume to verify and reuse it")
        if prior.get("training_configuration") != configuration or prior["source_sha256"] != _source_fingerprint():
            raise RuntimeError(f"Controlled-study configuration/source mismatch at {directory}")
        for relative, digest in prior["file_sha256"].items():
            if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Controlled-study artifact changed: {directory / relative}")
        print(f"verified completed training {problem.name} seed={seed}", flush=True)
        return prior
    device = torch.device(args.device)
    data = {
        split: problem.coefficients(count, args.dataset_seed + offset)
        for split, count, offset in (
            ("train", args.n_train, 0),
            ("validation", args.n_validation, 1),
            ("test", args.n_test, 2),
        )
    }
    hashes = {split: hashlib.sha256(value.numpy().tobytes()).hexdigest() for split, value in data.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        a = {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in data[left]}
        b = {hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in data[right]}
        if a & b:
            raise RuntimeError("A coefficient identity appears in two splits")
    np.savez_compressed(directory / "split_coefficients.npz", **{key: value.numpy() for key, value in data.items()})
    train = data["train"].to(device=device, dtype=torch.float32)
    validation = data["validation"].to(device=device, dtype=torch.float32)
    initial = problem.reference(train, 0, args.grid)
    high_initial = problem.reference(train, 0, 2 * args.grid)
    physical_scale = problem.rhs(high_initial).square().mean().clamp_min(1e-6).detach()
    scale = float(initial.std().item())
    architecture = dict(width=args.width, modes=args.modes, layers=args.layers, mean=problem.mean, scale=scale)
    torch.manual_seed(seed)
    model = ControlledFNO(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    history, best = [], float("inf")
    warmup_path = directory / "warmup.pt"
    started = time.perf_counter()

    def train_stage(arm: str, steps: int, target_error: float | None = None):
        nonlocal model, optimizer, best
        generator = torch.Generator(device=device).manual_seed(seed + (10000 if arm != "warmup" else 0))
        best, closest = float("inf"), float("inf")
        stage_started = time.perf_counter()
        for step in range(1, steps + 1):
            model.train()
            indices = torch.randint(len(train), (args.batch_size,), generator=generator, device=device)
            # Keep central PDE-difference calls inside the training time range.
            tau = 0.01 + 0.98 * torch.rand(args.batch_size, generator=generator, device=device)
            x = initial[indices]
            target = problem.reference(train[indices], tau * problem.final_time, args.grid)
            prediction = model(x, tau)
            data_loss = _relative_mse(prediction, target)
            auxiliary = torch.zeros((), device=device)
            if arm == "pino":
                eps = 1e-3
                x_high = high_initial[indices]
                center = model(x_high, tau)
                derivative = (model(x_high, tau + eps) - model(x_high, tau - eps)) / (2 * eps * problem.final_time)
                auxiliary = (derivative - problem.rhs(center)).square().mean() / physical_scale
                loss = data_loss + args.physics_weight * auxiliary
            elif arm == "composition":
                midpoint = model(x, tau / 2)
                composed = model(midpoint, tau / 2)
                auxiliary = _relative_mse(composed, prediction)
                loss = data_loss + args.composition_weight * auxiliary
            else:
                loss = data_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {arm} training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Equal update schedule in continuation arms; warmup is shared.
            lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(1, steps))))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if step % args.validation_every == 0 or step == steps:
                model.eval()
                score = validation_error(model, problem, validation, args.grid)
                metadata = dict(
                    arm=arm,
                    step=step,
                    validation_rel_l2=score,
                    architecture=architecture,
                    seed=seed,
                    problem=problem.__dict__,
                    split_hashes=hashes,
                    source_sha256=_source_fingerprint(),
                    package_version=__version__,
                )
                if score < best:
                    best = score
                    _checkpoint(
                        warmup_path if arm == "warmup" else directory / arm / "best_validation.pt",
                        model,
                        optimizer,
                        metadata,
                    )
                if target_error is not None:
                    distance = abs(math.log(score / target_error))
                    if distance < closest:
                        closest = distance
                        _checkpoint(directory / arm / "matched_accuracy.pt", model, optimizer, metadata)
                history.append(
                    dict(
                        arm=arm,
                        step=step,
                        data_loss=data_loss.item(),
                        auxiliary_loss=auxiliary.item(),
                        validation_rel_l2=score,
                        stage_seconds=time.perf_counter() - stage_started,
                    )
                )
                pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
                print(f"{problem.name} seed={seed} {arm} {step}/{steps}: validation L2={score:.5g}", flush=True)
        return best

    train_stage("warmup", args.warmup_steps)
    warmup = _torch_load(warmup_path, device)
    arm_results = {}
    supervised_target = None
    for arm in ("supervised", "pino", "composition"):
        model = ControlledFNO(**architecture).to(device)
        model.load_state_dict(warmup["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
        optimizer.load_state_dict(copy.deepcopy(warmup["optimizer"]))
        score = train_stage(arm, args.continuation_steps, supervised_target)
        if arm == "supervised":
            supervised_target = score
        arm_results[arm] = dict(best_validation_rel_l2=score)
    manifest = dict(
        problem=problem.__dict__,
        seed=seed,
        architecture=architecture,
        split_hashes=hashes,
        training_initial_states=args.n_train,
        validation_initial_states=args.n_validation,
        test_initial_states=args.n_test,
        parameter_count=sum(p.numel() for p in model.parameters()),
        warmup_selected_step=warmup["step"],
        warmup_steps=args.warmup_steps,
        continuation_steps=args.continuation_steps,
        arms=arm_results,
        elapsed_seconds=time.perf_counter() - started,
        source_sha256=_source_fingerprint(),
        package_version=__version__,
        selection="Minimum mean validation relative L2 over four horizons; matched checkpoints minimize log accuracy distance only.",
        test_used_for_selection=False,
        training_configuration=configuration,
        file_sha256={
            str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*")
            if path.is_file() and path.suffix in {".pt", ".npz", ".csv"}
        },
    )
    (directory / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@torch.no_grad()
def evaluate_problem(outdir: Path, problem: ExactProblem, args: argparse.Namespace, seed: int) -> list[dict]:
    directory = outdir / problem.name / f"seed_{seed}"
    device = torch.device(args.device)
    values = np.load(directory / "split_coefficients.npz")
    coefficients = torch.as_tensor(values["test"], dtype=torch.float32, device=device)
    initial = problem.reference(coefficients, 0, args.grid)
    mean = initial.mean()
    task = TaskSpec(
        name=problem.name,
        label=problem.name,
        dim=1,
        globs={"train": [], "val": [], "test": []},
        grid_spacing=(1 / args.grid,),
        energy_type="generic_l2",
        energy_law="dissipative" if problem.viscosity else "conservative",
    )
    rows = []
    for arm in ("supervised", "pino", "composition"):
        for selection in ("best_validation",) if arm == "supervised" else ("best_validation", "matched_accuracy"):
            checkpoint = _torch_load(directory / arm / f"{selection}.pt", device)
            model = ControlledFNO(**checkpoint["architecture"]).to(device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            arrays = {}
            for fraction in (0.125, 0.25, 0.5, 1.0):
                tau = torch.full((len(initial),), fraction, device=device)
                horizon = fraction * problem.final_time
                reference = problem.reference(coefficients, horizon, args.grid)
                prediction = model(initial, tau)
                composed = model(model(initial, tau / 2), tau / 2)
                first = model(initial, tau / 2)
                ref_first = problem.reference(coefficients, horizon / 2, args.grid)
                reference_gen = generator_discrepancy(initial, ref_first, reference, horizon / 2, horizon)
                generator = generator_discrepancy(initial, first, prediction, horizon / 2, horizon)
                metrics = {
                    "rel_l2": rel_l2(prediction, reference),
                    "fluctuation_rel_l2": rel_l2(prediction - mean, reference - mean),
                    "semigroup": semigroup_error(prediction, composed),
                    "generator": generator,
                    "generator_reference": reference_gen,
                    "energy_violation": energy_law_error(initial, prediction, task),
                    "energy_reference_error": energy_reference_error(reference, prediction, task),
                }
                for key, value in metrics.items():
                    arrays[f"{fraction}_{key}"] = value.cpu().numpy()
                arrays[f"{fraction}_prediction"] = prediction.cpu().numpy()
                arrays[f"{fraction}_reference"] = reference.cpu().numpy()
                arrays[f"{fraction}_composed"] = composed.cpu().numpy()
                for i in range(len(initial)):
                    rows.append(
                        dict(
                            problem=problem.name,
                            seed=seed,
                            arm=arm,
                            selection=selection,
                            horizon=horizon,
                            fraction=fraction,
                            test_initial_state=i,
                            validation_rel_l2=checkpoint["validation_rel_l2"],
                            selected_step=checkpoint["step"],
                            **{key: value[i].item() for key, value in metrics.items()},
                        )
                    )
            np.savez_compressed(directory / arm / f"test_{selection}.npz", **arrays)
    pd.DataFrame(rows).to_csv(directory / "test_samples.csv", index=False)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs_controlled"))
    parser.add_argument("--problems", nargs="+", choices=list(PROBLEMS), default=list(PROBLEMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-validation", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--dataset-seed", type=int, default=20261021)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--continuation-steps", type=int, default=2000)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--physics-weight", type=float, default=0.05)
    parser.add_argument("--composition-weight", type=float, default=0.2)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    arguments_path = args.outdir / ("last_invocation.json" if (args.outdir / "args.json").exists() else "args.json")
    arguments_path.write_text(json.dumps(arguments, indent=2) + "\n")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = []
    for name in args.problems:
        for seed in args.seeds:
            if not args.evaluate_only:
                train_problem(args.outdir, PROBLEMS[name], args, seed)
            if not args.train_only:
                rows.extend(evaluate_problem(args.outdir, PROBLEMS[name], args, seed))
            pd.DataFrame(rows).to_csv(args.outdir / "test_samples.csv", index=False)
    (args.outdir / "manifest.json").write_text(
        json.dumps(
            dict(
                package_version=__version__,
                source_sha256=_source_fingerprint(),
                python=platform.python_version(),
                torch=str(torch.__version__),
                numpy=np.__version__,
                device=args.device,
                gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
                test_sample_rows=len(rows),
                arguments=arguments,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
