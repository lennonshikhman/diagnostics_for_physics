"""Physical-state checks used by the checkpoint-only external re-audit."""

from __future__ import annotations

import torch

from .metrics import energy, energy_reference_error, integral_balance_error, rel_l2
from .tasks import TaskSpec


def cns_energy_components(state: torch.Tensor, task: TaskSpec) -> dict[str, torch.Tensor]:
    """Per-sample spatial means of kinetic and internal energy, in float64."""
    if task.energy_type != "cns":
        raise ValueError("Kinetic/internal decomposition requires a CNS energy contract")
    task.validate_state_channels(state.shape[1])
    state = state.double()
    rho = state[:, task.density_channel]
    velocity = state[:, task.velocity_channels]
    kinetic = (0.5 * rho * velocity.square().sum(1)).flatten(1).mean(1)
    internal = (state[:, task.pressure_channel] / (task.gamma - 1)).flatten(1).mean(1)
    return {"kinetic": kinetic, "internal": internal}


def physical_samples(initial, reference, prediction, task, *, delta_t, control_volume):
    """Full-domain and genuine subvolume residuals with reference baselines."""
    initial, reference, prediction = initial.double(), reference.double(), prediction.double()
    return {
        "state_error": rel_l2(prediction, reference),
        "functional": energy_reference_error(reference, prediction, task),
        "total_functional": energy(prediction, task),
        "full_balance": integral_balance_error(initial, prediction, task, delta_t=delta_t),
        "subvolume_balance": integral_balance_error(
            initial, prediction, task, delta_t=delta_t, control_volume=control_volume
        ),
        "reference_subvolume_balance": integral_balance_error(
            initial, reference, task, delta_t=delta_t, control_volume=control_volume
        ),
    }
