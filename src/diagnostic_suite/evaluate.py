"""Evaluation loops shared by the command-line benchmark runner.

The public tensor/callable API lives in :mod:`diagnostic_suite.api`.  This module
adds model normalization and trajectory-window bookkeeping without changing the
scientific definitions in :mod:`diagnostic_suite.metrics`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import (
    BASE_METRIC_NAMES,
    batch_tau,
    constraint_errors,
    energy_law_error,
    energy_reference_error,
    generator_discrepancy,
    global_conserved_quantity_drift,
    integral_balance_error,
    metric_names_for_task,
    rel_l2,
    scaling_law_error,
    semigroup_error,
    summarize_metric_rows,
)
from .normalization import Normalizer
from .tasks import TaskSpec

Applicability = Dict[str, Tuple[bool, str]]


def _empty_rows(task: TaskSpec) -> Dict[str, List[float]]:
    return {name: [] for name in metric_names_for_task(task)}


def _append(rows: Dict[str, List[float]], name: str, values: torch.Tensor) -> None:
    rows.setdefault(name, []).extend(values.detach().cpu().reshape(-1).tolist())


def _append_unavailable(rows: Dict[str, List[float]], name: str, batch_size: int) -> None:
    rows.setdefault(name, []).extend([float("nan")] * batch_size)


def _base_applicability(task: TaskSpec) -> Applicability:
    constraint_names = [name for name in metric_names_for_task(task) if name not in BASE_METRIC_NAMES]
    energy_defined = task.energy_type != "none" or task.energy_callback is not None
    energy_law_defined = task.energy_law in {"conservative", "dissipative"}
    balance_defined = task.balance is not None
    global_quantity_defined = balance_defined or task.balance_channel is not None
    scaling_defined = task.scaling_transform is not None or task.scaling == "linear_amplitude"
    output: Applicability = {
        "rel_l2": (True, ""),
        "semigroup": (True, ""),
        "generator": (True, ""),
        "energy_violation": (
            energy_law_defined,
            "" if energy_law_defined else "No conservative or dissipative energy law is configured.",
        ),
        "energy_reference_error": (
            energy_defined,
            "" if energy_defined else "No energy functional is configured.",
        ),
        "integral_balance": (
            balance_defined,
            "" if balance_defined else "No conserved quantity, flux, and source contract is configured.",
        ),
        "global_conserved_drift": (
            global_quantity_defined,
            "" if global_quantity_defined else "No conserved quantity is configured.",
        ),
        "constraint": (
            len(constraint_names) == 1,
            ""
            if len(constraint_names) == 1
            else "No physical constraint is configured."
            if not constraint_names
            else "Multiple constraints are reported separately; no heterogeneous composite is defined.",
        ),
        "perturbation_response": (
            False,
            "Paired perturbed reference trajectories are not present in this benchmark dataset.",
        ),
        "scaling_law": (
            scaling_defined,
            "" if scaling_defined else "No exact scaling transformation is configured.",
        ),
    }
    for name in constraint_names:
        output[name] = (True, "")
    return output


def _predict(
    model: nn.Module,
    normalizer: Normalizer,
    state: torch.Tensor,
    horizon: int,
    max_delta: int,
    device: torch.device,
) -> torch.Tensor:
    tau = batch_tau(state.shape[0], float(horizon) / float(max_delta), device)
    return normalizer.decode(model(normalizer.encode(state), tau))


def _validate_horizon(sequence_length: int, horizon: Optional[int], max_delta: int) -> int:
    selected = min(2, sequence_length - 1, max_delta) if horizon is None else int(horizon)
    if selected < 1:
        raise ValueError("Evaluation horizon must be positive")
    if selected > max_delta:
        raise ValueError(f"Evaluation horizon {selected} exceeds max_delta={max_delta}")
    if selected >= sequence_length:
        raise ValueError(f"Evaluation horizon {selected} requires {selected + 1} states; window has {sequence_length}")
    return selected


def _physical_horizon(task: TaskSpec, horizon: int) -> float:
    return float("nan") if task.time_step is None else float(horizon) * float(task.time_step)


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
    *,
    eval_horizon: Optional[int] = None,
    seed: Optional[int] = None,
    scaling_factor: float = 1.25,
) -> Dict[str, Any]:
    """Evaluate one model at one explicit horizon.

    Keeping one horizon per table row avoids blending short- and long-horizon
    behavior into an uninterpretable average.  The runner calls this function for
    every requested horizon.
    """

    model.eval()
    norm = normalizer.to(device)
    rows = _empty_rows(task)
    applicability = _base_applicability(task)
    selected_horizon: Optional[int] = None

    for sequence in window_loader:
        sequence = sequence.to(device).float()
        if not torch.isfinite(sequence).all():
            raise FloatingPointError("Reference trajectories contain NaN or Inf")
        batch_size = sequence.shape[0]
        horizon = _validate_horizon(sequence.shape[1], eval_horizon, max_delta)
        if selected_horizon is None:
            selected_horizon = horizon
        elif selected_horizon != horizon:
            raise RuntimeError("A single evaluation row cannot mix horizons")

        initial = sequence[:, 0]
        reference = sequence[:, horizon]
        prediction = _predict(model, norm, initial, horizon, max_delta, device)
        _append(rows, "rel_l2", rel_l2(prediction, reference))

        if horizon >= 2:
            first = horizon // 2
            second = horizon - first
            midpoint = _predict(model, norm, initial, first, max_delta, device)
            composed = _predict(model, norm, midpoint, second, max_delta, device)
            _append(rows, "semigroup", semigroup_error(prediction, composed))

            one_step = _predict(model, norm, initial, 1, max_delta, device)
            physical_step = float(task.time_step or 1.0)
            _append(
                rows,
                "generator",
                generator_discrepancy(
                    initial,
                    one_step,
                    prediction,
                    physical_step,
                    horizon * physical_step,
                ),
            )
        else:
            _append_unavailable(rows, "semigroup", batch_size)
            _append_unavailable(rows, "generator", batch_size)
            applicability["semigroup"] = (False, "Semigroup evaluation requires a horizon of at least two steps.")
            applicability["generator"] = (False, "Generator evaluation requires two distinct positive horizons.")

        _append(rows, "energy_violation", energy_law_error(initial, prediction, task))
        _append(rows, "energy_reference_error", energy_reference_error(reference, prediction, task))
        delta_t = horizon * float(task.time_step or 1.0)
        _append(
            rows,
            "integral_balance",
            integral_balance_error(initial, prediction, task, delta_t=delta_t),
        )
        _append(
            rows,
            "global_conserved_drift",
            global_conserved_quantity_drift(initial, prediction, task),
        )
        components = constraint_errors(prediction, task)
        for name, value in components.items():
            _append(rows, name, value)
        if len(components) == 1:
            _append(rows, "constraint", next(iter(components.values())))
        else:
            _append_unavailable(rows, "constraint", batch_size)

        # Learned-trajectory benchmark files do not contain paired simulations
        # from perturbed initial conditions.  The public API implements the
        # metric when those four states are supplied.
        _append_unavailable(rows, "perturbation_response", batch_size)

        scale = float(scaling_factor)
        if scale <= 0 or torch.isclose(torch.tensor(scale), torch.tensor(1.0)):
            raise ValueError("scaling_factor must be positive and nontrivial (not 1.0)")
        if task.scaling_transform is not None:
            scaled_initial = task.scaling_transform(initial, scale)
            time_factor = float(task.scaling_time_factor(scale)) if task.scaling_time_factor is not None else 1.0
            scaled_horizon = float(horizon) * time_factor
            scaled_index = int(round(scaled_horizon))
            if scaled_index < 1 or not torch.isclose(
                torch.tensor(scaled_horizon), torch.tensor(float(scaled_index)), atol=1.0e-8
            ):
                _append_unavailable(rows, "scaling_law", batch_size)
                applicability["scaling_law"] = (
                    False,
                    "Configured time scaling does not map this discrete horizon to an integer step.",
                )
            elif scaled_index > max_delta:
                _append_unavailable(rows, "scaling_law", batch_size)
                applicability["scaling_law"] = (
                    False,
                    "Configured time scaling exceeds the trained maximum horizon.",
                )
            else:
                prediction_of_scaled = _predict(model, norm, scaled_initial, horizon, max_delta, device)
                base_prediction = _predict(model, norm, initial, scaled_index, max_delta, device)
                transformed_prediction = task.scaling_transform(base_prediction, scale)
                _append(
                    rows,
                    "scaling_law",
                    scaling_law_error(prediction_of_scaled, transformed_prediction),
                )
        elif task.scaling == "linear_amplitude":
            prediction_of_scaled = _predict(model, norm, scale * initial, horizon, max_delta, device)
            _append(
                rows,
                "scaling_law",
                scaling_law_error(prediction_of_scaled, scale * prediction),
            )
        else:
            _append_unavailable(rows, "scaling_law", batch_size)

    if selected_horizon is None:
        raise RuntimeError("The evaluation loader produced no windows")

    prefix: Dict[str, Any] = {
        "model": model_name,
        "variant": variant,
        "task": task.name,
        "seed": seed,
        "horizon_steps": selected_horizon,
        "horizon_time": _physical_horizon(task, selected_horizon),
    }
    return summarize_metric_rows(prefix, rows, applicability)


@torch.no_grad()
def evaluate_reference(
    window_loader: DataLoader,
    task: TaskSpec,
    max_delta: int,
    device: torch.device,
    *,
    eval_horizon: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate the stored numerical reference at one explicit horizon."""

    rows = _empty_rows(task)
    applicability = _base_applicability(task)
    applicability["perturbation_response"] = (
        False,
        "Paired perturbed reference trajectories are not present in this benchmark dataset.",
    )
    applicability["scaling_law"] = (
        False,
        "Scaled reference trajectories are not present in this benchmark dataset.",
    )
    selected_horizon: Optional[int] = None

    for sequence in window_loader:
        sequence = sequence.to(device).float()
        if not torch.isfinite(sequence).all():
            raise FloatingPointError("Reference trajectories contain NaN or Inf")
        batch_size = sequence.shape[0]
        horizon = _validate_horizon(sequence.shape[1], eval_horizon, max_delta)
        selected_horizon = horizon
        initial = sequence[:, 0]
        reference = sequence[:, horizon]

        _append(rows, "rel_l2", torch.zeros(batch_size, device=device))
        if horizon >= 2:
            _append_unavailable(rows, "semigroup", batch_size)
            applicability["semigroup"] = (
                False,
                "Stored trajectories alone do not provide an independent restarted reference evolution.",
            )
            physical_step = float(task.time_step or 1.0)
            _append(
                rows,
                "generator",
                generator_discrepancy(
                    initial,
                    sequence[:, 1],
                    reference,
                    physical_step,
                    horizon * physical_step,
                ),
            )
        else:
            _append_unavailable(rows, "semigroup", batch_size)
            _append_unavailable(rows, "generator", batch_size)
            applicability["semigroup"] = (False, "Semigroup evaluation requires a horizon of at least two steps.")
            applicability["generator"] = (False, "Generator evaluation requires two distinct positive horizons.")

        _append(rows, "energy_violation", energy_law_error(initial, reference, task))
        if applicability["energy_reference_error"][0]:
            _append(rows, "energy_reference_error", torch.zeros(batch_size, device=device))
        else:
            _append_unavailable(rows, "energy_reference_error", batch_size)
        _append(
            rows,
            "integral_balance",
            integral_balance_error(
                initial,
                reference,
                task,
                delta_t=horizon * float(task.time_step or 1.0),
            ),
        )
        _append(
            rows,
            "global_conserved_drift",
            global_conserved_quantity_drift(initial, reference, task),
        )
        components = constraint_errors(reference, task)
        for name, value in components.items():
            _append(rows, name, value)
        if len(components) == 1:
            _append(rows, "constraint", next(iter(components.values())))
        else:
            _append_unavailable(rows, "constraint", batch_size)
        _append_unavailable(rows, "perturbation_response", batch_size)
        _append_unavailable(rows, "scaling_law", batch_size)

    if selected_horizon is None:
        raise RuntimeError("The reference loader produced no windows")

    output = summarize_metric_rows(
        {
            "model": "reference",
            "variant": "reference",
            "task": task.name,
            "seed": seed,
            "horizon_steps": selected_horizon,
            "horizon_time": _physical_horizon(task, selected_horizon),
        },
        rows,
        applicability,
    )
    output["best_val"] = float("nan")
    output["best_epoch"] = float("nan")
    return output


__all__ = ["evaluate_diagnostics", "evaluate_reference"]
