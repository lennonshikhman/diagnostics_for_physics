"""Small public API for post-hoc diagnostic panels.

The CLI reproduces the paper experiments. This module is the lightweight API for
users who already have arrays/tensors from their own learned PDE simulator and
want to compute diagnostic quantities directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from .constants import EPS
from .tasks import TaskSpec
from .metrics import (
    constraint_error,
    energy,
    integral_balance_error,
    rel_l2,
)


@dataclass(frozen=True)
class DiagnosticConfig:
    """Configuration for equation-dependent diagnostics.

    Parameters
    ----------
    dim:
        Number of spatial dimensions.
    energy_type:
        One of ``generic_l2``, ``kinetic``, ``shallow_water``, ``cns``, or
        ``mhd``. These choices mirror the paper implementation.
    balance_channel:
        Channel used for global integral-balance/mass diagnostic. Use ``None``
        to disable it.
    positivity_channels:
        Channels constrained to be nonnegative, such as shallow-water height or
        compressible-flow density/pressure.
    velocity_channels:
        Velocity component channels, used by kinetic energy and divergence
        constraints when available.
    magnetic_channels:
        Magnetic-field component channels, used by MHD energy/divergence
        diagnostics when available.
    periodic:
        Whether finite-difference divergence diagnostics use periodic wrapping.
    name:
        Descriptive task name included in downstream metadata.
    """

    dim: int
    energy_type: str = "generic_l2"
    balance_channel: Optional[int] = 0
    positivity_channels: Sequence[int] = field(default_factory=tuple)
    velocity_channels: Sequence[int] = field(default_factory=tuple)
    magnetic_channels: Sequence[int] = field(default_factory=tuple)
    periodic: bool = True
    name: str = "custom"

    def to_task_spec(self) -> TaskSpec:
        return TaskSpec(
            name=self.name,
            label=self.name,
            dim=int(self.dim),
            globs={},
            periodic=bool(self.periodic),
            energy_type=self.energy_type,
            balance_channel=self.balance_channel,
            positivity_channels=list(self.positivity_channels),
            velocity_channels=list(self.velocity_channels),
            magnetic_channels=list(self.magnetic_channels),
        )


@dataclass
class DiagnosticResult:
    """Container returned by :class:`DiagnosticPanel`.

    ``samples`` stores per-batch/per-trajectory metric samples as NumPy arrays.
    ``means`` stores finite-sample means, with unavailable metrics represented as
    ``nan`` rather than zero.
    """

    samples: Dict[str, np.ndarray]

    @property
    def means(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in self.samples.items():
            arr = np.asarray(value, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            out[key] = float(finite.mean()) if finite.size else float("nan")
        return out

    def as_dict(self) -> Dict[str, float]:
        """Return finite-sample means as a plain dictionary."""
        return self.means


class DiagnosticPanel:
    """Post-hoc diagnostic panel for learned PDE predictions.

    The panel expects tensors with shape ``(batch, channels, *spatial)``. Inputs
    without an explicit batch dimension are accepted as ``(channels, *spatial)``
    and promoted to batch size one.
    """

    def __init__(self, config: DiagnosticConfig | TaskSpec):
        if isinstance(config, TaskSpec):
            self.task = config
        else:
            self.task = config.to_task_spec()

    def evaluate_pair(
        self,
        initial: torch.Tensor | np.ndarray,
        reference: torch.Tensor | np.ndarray,
        prediction: torch.Tensor | np.ndarray,
    ) -> DiagnosticResult:
        """Evaluate state, energy, balance, and constraint diagnostics.

        Parameters
        ----------
        initial:
            State at the start of the prediction interval.
        reference:
            Reference future state.
        prediction:
            Learned-model future prediction.
        """
        u0 = self._ensure_state(initial)
        ref = self._ensure_state(reference, like=u0)
        pred = self._ensure_state(prediction, like=u0)

        e0 = energy(u0, self.task)
        ep = energy(pred, self.task)
        er = energy(ref, self.task)

        samples = {
            "rel_l2": self._detach(rel_l2(pred, ref)),
            "energy_violation": self._detach(torch.relu(ep - e0) / (e0.abs() + EPS)),
            "energy_decay_error": self._detach((ep - er).abs() / (er.abs() + EPS)),
            "integral_balance": self._detach(integral_balance_error(u0, pred, self.task)),
            "constraint": self._detach(constraint_error(pred, self.task)),
        }
        # Backward-compatible alias used in the paper tables.
        samples["energy"] = samples["energy_violation"]
        return DiagnosticResult(samples)

    def semigroup_error(
        self,
        direct_prediction: torch.Tensor | np.ndarray,
        composed_prediction: torch.Tensor | np.ndarray,
    ) -> DiagnosticResult:
        """Compare direct and composed learned predictions for the same horizon."""
        direct = self._ensure_state(direct_prediction)
        composed = self._ensure_state(composed_prediction, like=direct)
        return DiagnosticResult({"semigroup": self._detach(rel_l2(composed, direct))})

    def generator_discrepancy(
        self,
        initial: torch.Tensor | np.ndarray,
        prediction_a: torch.Tensor | np.ndarray,
        prediction_b: torch.Tensor | np.ndarray,
        delta_a: float = 1.0,
        delta_b: float = 2.0,
        reference_scale: Optional[float] = None,
    ) -> DiagnosticResult:
        """Finite-difference generator discrepancy for two short horizons.

        ``prediction_a`` and ``prediction_b`` are predictions from the same
        initial state at horizons ``delta_a`` and ``delta_b``. If
        ``reference_scale`` is supplied, an additional ``generator_ratio`` sample
        is returned.
        """
        u0 = self._ensure_state(initial)
        pa = self._ensure_state(prediction_a, like=u0)
        pb = self._ensure_state(prediction_b, like=u0)
        gen_a = (pa - u0) / float(delta_a)
        gen_b = (pb - u0) / float(delta_b)
        num = torch.linalg.vector_norm((gen_a - gen_b).flatten(1), dim=1)
        den = torch.linalg.vector_norm(gen_a.flatten(1), dim=1) + torch.linalg.vector_norm(gen_b.flatten(1), dim=1) + EPS
        gen = num / den
        samples = {"generator": self._detach(gen)}
        if reference_scale is not None:
            samples["generator_ratio_to_reference"] = self._detach(gen / (float(reference_scale) + EPS))
        return DiagnosticResult(samples)

    def _ensure_state(self, x: torch.Tensor | np.ndarray, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            y = x.detach().clone() if not x.requires_grad else x
            y = y.float()
        else:
            y = torch.as_tensor(x, dtype=torch.float32)
        expected_no_batch = self.task.dim + 1
        expected_with_batch = self.task.dim + 2
        if y.ndim == expected_no_batch:
            y = y.unsqueeze(0)
        if y.ndim != expected_with_batch:
            raise ValueError(
                f"Expected state shape (B,C,*spatial) or (C,*spatial) for dim={self.task.dim}; "
                f"received shape {tuple(y.shape)}."
            )
        if like is not None and tuple(y.shape) != tuple(like.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(like.shape)}, received {tuple(y.shape)}.")
        return y

    @staticmethod
    def _detach(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy().astype(float, copy=False)


__all__ = ["DiagnosticConfig", "DiagnosticPanel", "DiagnosticResult"]
