"""Checkpoint-only rollout and paired crossed-bootstrap audit helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def bounded_chain(model, initial: torch.Tensor, step_fraction: float, steps: int) -> dict[int, torch.Tensor]:
    """Restart a direct predictor using only fractions in its trained interval.

    Keys are numbers of calls, not a request for a larger one-call horizon.
    A nonfinite state is retained so an audit can report a failed rollout.
    """
    if not 0 < step_fraction <= 1 or not np.isfinite(step_fraction):
        raise ValueError("Every model query must have a fraction in (0, 1]")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    state = initial
    tau = torch.full((len(initial),), step_fraction, device=initial.device, dtype=initial.dtype)
    states = {}
    for index in range(1, steps + 1):
        state = model(state, tau)
        if state.shape != initial.shape:
            raise ValueError("A restarted predictor changed the physical-state shape")
        states[index] = state
    return states


def crossed_draws(seeds: int, states: int, replicates: int = 10_000, random_seed: int = 20261021):
    """Independent empirical seed and shared-initial-condition weights."""
    if min(seeds, states, replicates) < 1:
        raise ValueError("Positive seed, initial-state and replicate counts are required")
    rng = np.random.default_rng(random_seed)
    seed_weights = rng.multinomial(seeds, np.full(seeds, 1 / seeds), size=replicates) / seeds
    state_weights = rng.multinomial(states, np.full(states, 1 / states), size=replicates) / states
    return seed_weights, state_weights


def crossed_means(values: np.ndarray, draws) -> np.ndarray:
    """Paired bootstrap means, without dropping failed/nonfinite samples."""
    values = np.asarray(values, dtype=np.float64)
    seed_weights, state_weights = draws
    if values.ndim != 2 or values.shape != (seed_weights.shape[1], state_weights.shape[1]):
        raise ValueError("Expected a seed by initial-condition matrix matching the bootstrap draws")
    if not np.isfinite(values).all():
        return np.full(len(seed_weights), np.nan)
    return np.einsum("bs,si,bi->b", seed_weights, values, state_weights, optimize=True)


def percentile_summary(point: float, samples: np.ndarray) -> dict:
    """An unavailable interval is explicit; undefined replicates are counted."""
    samples = np.asarray(samples)
    finite = np.isfinite(samples)
    result = {
        "mean": float(point) if np.isfinite(point) else None,
        "replicates": len(samples),
        "undefined_replicates": int((~finite).sum()),
    }
    if not finite.all() or not np.isfinite(point):
        result.update(low=None, high=None, status="undefined_or_failed")
    else:
        low, high = np.quantile(samples, (0.025, 0.975))
        result.update(low=float(low), high=float(high), status="available")
    return result


def spearman_rows(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Average-rank Spearman coefficients, preserving ties and missing slots.

    Missing slots only pad unequal numbers of unique checkpoints. Nonfinite
    scientific observations must be checked before calling this function.
    A constant vector has an undefined correlation, never a zero correlation.
    """
    first, second = np.atleast_2d(first), np.atleast_2d(second)
    if first.shape != second.shape:
        raise ValueError("Rank-correlation arrays must have equal shapes")
    if not np.array_equal(np.isnan(first), np.isnan(second)):
        raise ValueError("Checkpoint padding must agree between paired arrays")
    a = pd.DataFrame(first).rank(axis=1, method="average").to_numpy(copy=True)
    b = pd.DataFrame(second).rank(axis=1, method="average").to_numpy(copy=True)
    a -= np.nanmean(a, axis=1, keepdims=True)
    b -= np.nanmean(b, axis=1, keepdims=True)
    denominator = np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1))
    return np.divide(np.nansum(a * b, axis=1), denominator, out=np.full(len(a), np.nan), where=denominator > 0)
