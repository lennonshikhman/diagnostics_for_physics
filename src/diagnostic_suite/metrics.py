from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .constants import EPS
from .tasks import TaskSpec

def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((pred - target).flatten(1), dim=1) / (torch.linalg.vector_norm(target.flatten(1), dim=1) + EPS)


def grad_dim(u: torch.Tensor, axis: int, periodic: bool = True) -> torch.Tensor:
    # axis is spatial axis index in [0, dim-1], corresponding to tensor dim 2+axis.
    d = 2 + axis
    if periodic:
        return 0.5 * (torch.roll(u, -1, dims=d) - torch.roll(u, 1, dims=d))
    # one-sided-ish with replicate padding via roll fallback at boundaries.
    out = 0.5 * (torch.roll(u, -1, dims=d) - torch.roll(u, 1, dims=d))
    sl0 = [slice(None)] * u.ndim
    sl1 = [slice(None)] * u.ndim
    sl0[d] = 0
    sl1[d] = 1
    out[tuple(sl0)] = u[tuple(sl1)] - u[tuple(sl0)]
    sl0[d] = -1
    sl1[d] = -2
    out[tuple(sl0)] = u[tuple(sl0)] - u[tuple(sl1)]
    return out


def divergence(uvec: torch.Tensor, dim: int, periodic: bool = True) -> torch.Tensor:
    div = 0.0
    for ax in range(dim):
        div = div + grad_dim(uvec[:, ax:ax+1], ax, periodic)
    return div


def energy(u: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    spatial = tuple(range(2, u.ndim))
    if task.energy_type == "kinetic" and len(task.velocity_channels) >= task.dim and max(task.velocity_channels[:task.dim]) < u.shape[1]:
        return 0.5 * (u[:, task.velocity_channels[:task.dim]] ** 2).mean(dim=(1, *spatial))
    if task.energy_type == "shallow_water":
        h = u[:, 0:1]
        if u.shape[1] >= 3:
            v2 = (u[:, 1:3] ** 2).sum(dim=1, keepdim=True)
            return (0.5 * h.clamp_min(0) * v2 + 0.5 * h ** 2).mean(dim=(1, *spatial))
        return (0.5 * h ** 2).mean(dim=(1, *spatial))
    if task.energy_type == "cns" and u.shape[1] >= 5:
        rho = u[:, 0:1].clamp_min(0)
        p = u[:, 1:2].clamp_min(0)
        vel = u[:, 2:5]
        return (0.5 * rho * (vel ** 2).sum(dim=1, keepdim=True) + p).mean(dim=(1, *spatial))
    if task.energy_type == "mhd" and u.shape[1] >= 7:
        vel = u[:, task.velocity_channels[:3]] if max(task.velocity_channels[:3]) < u.shape[1] else u[:, 1:4]
        mag = u[:, task.magnetic_channels[:3]] if task.magnetic_channels and max(task.magnetic_channels[:3]) < u.shape[1] else u[:, 4:7]
        return 0.5 * ((vel ** 2).sum(dim=1) + (mag ** 2).sum(dim=1)).mean(dim=spatial)
    return 0.5 * (u ** 2).mean(dim=(1, *spatial))


def positivity_error(u: torch.Tensor, channels: List[int]) -> torch.Tensor:
    vals = []
    for c in channels:
        if c < u.shape[1]:
            vals.append(torch.linalg.vector_norm(torch.relu(-u[:, c:c+1]).flatten(1), dim=1) / (torch.linalg.vector_norm(u[:, c:c+1].flatten(1), dim=1) + EPS))
    if not vals:
        return torch.full((u.shape[0],), float("nan"), device=u.device)
    return torch.stack(vals, dim=0).nanmean(dim=0)


def constraint_error(u: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    vals = []
    if task.velocity_channels and len(task.velocity_channels) >= task.dim and max(task.velocity_channels[:task.dim]) < u.shape[1]:
        vel = u[:, task.velocity_channels[:task.dim]]
        div = divergence(vel, task.dim, task.periodic)
        vals.append(torch.linalg.vector_norm(div.flatten(1), dim=1) / (torch.linalg.vector_norm(vel.flatten(1), dim=1) + EPS))
    if task.magnetic_channels and len(task.magnetic_channels) >= task.dim and max(task.magnetic_channels[:task.dim]) < u.shape[1]:
        mag = u[:, task.magnetic_channels[:task.dim]]
        divb = divergence(mag, task.dim, task.periodic)
        vals.append(torch.linalg.vector_norm(divb.flatten(1), dim=1) / (torch.linalg.vector_norm(mag.flatten(1), dim=1) + EPS))
    if task.positivity_channels:
        vals.append(positivity_error(u, task.positivity_channels))
    if not vals:
        return torch.full((u.shape[0],), float("nan"), device=u.device)
    return torch.stack(vals, dim=0).nanmean(dim=0)


def integral_balance_error(u0: torch.Tensor, pred: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    c = task.balance_channel
    if c is None or c >= pred.shape[1]:
        return torch.full((pred.shape[0],), float("nan"), device=pred.device)
    spatial = tuple(range(2, pred.ndim))
    m0 = u0[:, c:c+1].mean(dim=spatial).flatten(1)
    mp = pred[:, c:c+1].mean(dim=spatial).flatten(1)
    return (mp - m0).abs().squeeze(1) / (m0.abs().squeeze(1) + pred[:, c:c+1].abs().mean(dim=spatial).flatten(1).squeeze(1) + EPS)


def batch_tau(b: int, value: float, device: torch.device) -> torch.Tensor:
    return torch.full((b,), float(value), device=device)


METRIC_NAMES = [
    "rel_l2",
    "semigroup",
    "generator",
    "energy_violation",
    "energy_decay_error",
    "integral_balance",
    "constraint",
    "perturbation_response",
    "scaling_law",
]


def summarize_metric_rows(prefix: Dict[str, Any], rows: Dict[str, List[float]]) -> Dict[str, Any]:
    """Summarize metric samples. Empty/unavailable metrics are NaN with n=0.

    This avoids the dangerous convention of writing 0.0 for an optional
    diagnostic that was not actually computed.
    """
    out: Dict[str, Any] = dict(prefix)
    for name, vals in rows.items():
        arr = np.asarray(vals, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        out[name] = float(finite.mean()) if finite.size else float("nan")
        out[name + "_std"] = float(finite.std()) if finite.size else float("nan")
        out[name + "_n"] = int(finite.size)
    # Backward-compatible alias for earlier CSVs: `energy` means energy-law
    # violation, while `energy_decay_error` measures wrong dissipation rate.
    if "energy_violation" in out:
        out["energy"] = out["energy_violation"]
        out["energy_std"] = out.get("energy_violation_std", float("nan"))
        out["energy_n"] = out.get("energy_violation_n", 0)
    return out


def add_reference_relative_columns(rows: List[Dict[str, Any]]) -> None:
    """Add excess/ratio columns for diagnostics with nonzero reference baselines.

    Generator discrepancy is a finite-difference consistency diagnostic. Even for
    reference trajectories it need not be zero at finite stored time steps, so paper
    tables should use excess/ratio relative to the reference baseline instead of
    reading raw generator values as absolute errors.
    """
    by_task: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("model") == "reference" and row.get("variant") == "reference":
            by_task[str(row.get("task"))] = row
    for row in rows:
        ref = by_task.get(str(row.get("task")))
        if not ref:
            continue
        for metric in ["generator"]:
            val = row.get(metric, float("nan"))
            base = ref.get(metric, float("nan"))
            try:
                vf = float(val); bf = float(base)
            except Exception:
                vf = float("nan"); bf = float("nan")
            row[f"{metric}_reference"] = bf
            row[f"{metric}_excess"] = vf - bf if np.isfinite(vf) and np.isfinite(bf) else float("nan")
            row[f"{metric}_ratio_to_reference"] = vf / (bf + EPS) if np.isfinite(vf) and np.isfinite(bf) else float("nan")


