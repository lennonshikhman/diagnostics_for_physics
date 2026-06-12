from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .constants import EPS
from .normalization import Normalizer
from .tasks import TaskSpec
from .metrics import (
    METRIC_NAMES, batch_tau, constraint_error, energy, integral_balance_error,
    rel_l2, summarize_metric_rows,
)

@torch.no_grad()
def evaluate_diagnostics(
    model: nn.Module,
    window_loader: DataLoader,
    normalizer: Normalizer,
    task: TaskSpec,
    max_delta: int,
    device: torch.device,
    model_name: str,
    variant: str,
) -> Dict[str, float]:
    model.eval()
    norm = normalizer.to(device)
    rows: Dict[str, List[float]] = {k: [] for k in METRIC_NAMES}
    for seq in window_loader:
        seq = seq.to(device).float()  # B,T+1,C,*spatial
        bsz = seq.shape[0]
        u0 = seq[:, 0]
        # choose k=2 if possible for semigroup/generator, else k=1
        k = min(2, seq.shape[1] - 1, max_delta)
        if k < 1:
            continue
        tau_k = batch_tau(bsz, k / max_delta, device)
        pred_k_n = model(norm.encode(u0), tau_k)
        pred_k = norm.decode(pred_k_n)
        target_k = seq[:, k]
        rows["rel_l2"].extend(rel_l2(pred_k, target_k).detach().cpu().tolist())

        # semigroup with a=b=1 if k>=2
        if k >= 2:
            tau_1 = batch_tau(bsz, 1 / max_delta, device)
            mid_n = model(norm.encode(u0), tau_1)
            comp_n = model(mid_n, tau_1)
            direct_n = pred_k_n
            rows["semigroup"].extend(rel_l2(norm.decode(comp_n), norm.decode(direct_n)).detach().cpu().tolist())

            gen1 = (norm.decode(mid_n) - u0) / 1.0
            gen2 = (pred_k - u0) / float(k)
            num = torch.linalg.vector_norm((gen1 - gen2).flatten(1), dim=1)
            den = torch.linalg.vector_norm(gen1.flatten(1), dim=1) + torch.linalg.vector_norm(gen2.flatten(1), dim=1) + EPS
            rows["generator"].extend((num / den).detach().cpu().tolist())
        else:
            rows["semigroup"].append(float("nan")); rows["generator"].append(float("nan"))

        e0 = energy(u0, task)
        ep = energy(pred_k, task)
        er = energy(target_k, task)
        # Energy-law violation detects the wrong sign of energy change.
        # Energy-decay error detects wrong dissipation rate, including over-smoothing
        # that remains monotone but decays too aggressively.
        rows["energy_violation"].extend((torch.relu(ep - e0) / (e0.abs() + EPS)).detach().cpu().tolist())
        rows["energy_decay_error"].extend(((ep - er).abs() / (er.abs() + EPS)).detach().cpu().tolist())
        rows["integral_balance"].extend(integral_balance_error(u0, pred_k, task).detach().cpu().tolist())
        rows["constraint"].extend(constraint_error(pred_k, task).detach().cpu().tolist())

        # Perturbed references are not generally available in benchmark trajectory files.
        # If you later add solvers, replace this NaN with true finite-perturbation response error.
        rows["perturbation_response"].append(float("nan"))

        # Scaling-law diagnostics are only valid when an exact relation is specified.
        if task.scaling == "linear_amplitude":
            alpha = 1.25
            pred_scaled = norm.decode(model(norm.encode(alpha * u0), tau_k))
            rhs = alpha * pred_k
            rows["scaling_law"].extend(rel_l2(pred_scaled, rhs).detach().cpu().tolist())
        else:
            rows["scaling_law"].append(float("nan"))

    return summarize_metric_rows({"model": model_name, "variant": variant, "task": task.name}, rows)


@torch.no_grad()
def evaluate_reference(window_loader: DataLoader, task: TaskSpec, max_delta: int, device: torch.device) -> Dict[str, float]:
    rows: Dict[str, List[float]] = {k: [] for k in METRIC_NAMES}
    for seq in window_loader:
        seq = seq.to(device).float()
        if seq.shape[1] < 3:
            continue
        u0, u1, u2 = seq[:, 0], seq[:, 1], seq[:, 2]
        rows["rel_l2"].extend([0.0] * seq.shape[0])
        rows["semigroup"].extend([0.0] * seq.shape[0])
        gen1 = u1 - u0
        gen2 = (u2 - u0) / 2.0
        num = torch.linalg.vector_norm((gen1 - gen2).flatten(1), dim=1)
        den = torch.linalg.vector_norm(gen1.flatten(1), dim=1) + torch.linalg.vector_norm(gen2.flatten(1), dim=1) + EPS
        rows["generator"].extend((num / den).detach().cpu().tolist())
        e0, e1 = energy(u0, task), energy(u1, task)
        rows["energy_violation"].extend((torch.relu(e1 - e0) / (e0.abs() + EPS)).detach().cpu().tolist())
        rows["energy_decay_error"].extend([0.0] * seq.shape[0])
        rows["integral_balance"].extend(integral_balance_error(u0, u1, task).detach().cpu().tolist())
        rows["constraint"].extend(constraint_error(u1, task).detach().cpu().tolist())
    out = summarize_metric_rows({"model": "reference", "variant": "reference", "task": task.name}, rows)
    out["best_val"] = float("nan")
    out["best_epoch"] = float("nan")
    return out

