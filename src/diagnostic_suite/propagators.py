from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, runtime_checkable

import numpy as np
import torch

State = torch.Tensor | np.ndarray


@runtime_checkable
class Propagator(Protocol):
    """Minimal architecture-independent learned-propagator contract."""

    def __call__(self, state: State, horizon: float) -> State:
        """Return the state predicted at ``horizon`` from ``state``."""


@dataclass
class CallablePropagator:
    """Adapt a ``callable(state, horizon)`` to the public propagator contract."""

    function: Callable[[State, float], State]
    name: str = "callable"

    def __call__(self, state: State, horizon: float) -> State:
        return self.function(state, float(horizon))


@dataclass
class AutoregressivePropagator:
    """Adapt a one-step callable by repeated composition."""

    one_step: Callable[[State], State]
    step_size: float = 1.0
    name: str = "autoregressive"

    def __post_init__(self) -> None:
        if not np.isfinite(self.step_size) or self.step_size <= 0:
            raise ValueError("step_size must be finite and positive")

    def _steps(self, horizon: float) -> int:
        if not np.isfinite(horizon):
            raise ValueError("Horizon must be finite")
        ratio = float(horizon) / float(self.step_size)
        steps = int(round(ratio))
        if steps < 0 or not np.isclose(ratio, steps, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"Horizon {horizon} is not a nonnegative multiple of step_size={self.step_size}")
        return steps

    def __call__(self, state: State, horizon: float) -> State:
        steps = self._steps(horizon)
        output = state
        for _ in range(steps):
            output = self.one_step(output)
        return output

    def composed(self, state: State, horizon: float) -> State:
        """Compose integer steps; equality is by construction for deterministic maps."""

        steps = self._steps(horizon)
        if steps < 2:
            raise NotImplementedError("Two positive integer steps are required for autoregressive composition")
        first = steps // 2
        return self(self(state, first * self.step_size), (steps - first) * self.step_size)


@dataclass
class SavedPredictionAdapter:
    """Expose saved batched predictions through the propagator contract.

    This adapter is intended for post-hoc audits where each value already contains
    predictions for the same batch represented by ``state``.  It deliberately
    rejects shape mismatches rather than silently broadcasting or reordering data.
    """

    predictions: Mapping[float, State]
    composed_predictions: Optional[Mapping[float, State]] = None
    name: str = "saved_predictions"
    horizon_tolerance: float = 1.0e-8

    def _key(self, horizon: float) -> float:
        matches = [
            key
            for key in self.predictions
            if np.isclose(float(key), float(horizon), rtol=0.0, atol=self.horizon_tolerance)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected exactly one saved prediction for horizon={horizon}; available={list(self.predictions)}"
            )
        return matches[0]

    def __call__(self, state: State, horizon: float) -> State:
        prediction = self.predictions[self._key(horizon)]
        if tuple(np.shape(prediction)) != tuple(np.shape(state)):
            raise ValueError(
                f"Saved prediction shape {np.shape(prediction)} does not match state shape {np.shape(state)}"
            )
        return prediction

    def composed(self, state: State, horizon: float) -> State:
        """Return an explicitly saved two-stage composition.

        A direct saved endpoint cannot be reapplied to an arbitrary intermediate
        state.  Semigroup consistency is therefore unavailable unless the caller
        supplies the actual composed endpoint separately.
        """

        if self.composed_predictions is None:
            raise KeyError(
                "No composed_predictions were supplied; saved direct endpoints "
                "alone cannot establish semigroup consistency"
            )
        matches = [
            key
            for key in self.composed_predictions
            if np.isclose(float(key), float(horizon), rtol=0.0, atol=self.horizon_tolerance)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected one saved composed prediction for horizon={horizon}; "
                f"available={list(self.composed_predictions)}"
            )
        prediction = self.composed_predictions[matches[0]]
        if tuple(np.shape(prediction)) != tuple(np.shape(state)):
            raise ValueError(
                f"Saved composed prediction shape {np.shape(prediction)} does not match state shape {np.shape(state)}"
            )
        return prediction


__all__ = [
    "AutoregressivePropagator",
    "CallablePropagator",
    "Propagator",
    "SavedPredictionAdapter",
    "State",
]
