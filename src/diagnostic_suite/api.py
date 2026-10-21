"""Public, architecture-independent APIs for post-hoc diagnostic audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from .metrics import (
    constraint_errors,
    energy_law_error,
    energy_reference_error,
    generator_discrepancy,
    global_conserved_quantity_drift,
    integral_balance_error,
    perturbation_response_error,
    rel_l2,
    scaling_law_error,
    semigroup_error,
    summarize_metric_rows,
)
from .propagators import AutoregressivePropagator, Propagator, SavedPredictionAdapter
from .tasks import BalanceSpec, ConstraintSpec, TaskSpec, TensorCallback


def _linear_amplitude(state: torch.Tensor, factor: float) -> torch.Tensor:
    return state * factor


@dataclass(frozen=True)
class DiagnosticApplicability:
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class DiagnosticConfig:
    """Equation and discretization metadata for public diagnostic APIs.

    Custom PDEs can supply callbacks for energy, constraints, conserved quantity,
    flux, source, and scaling without modifying package source.  Constraint types
    must be explicit: vector channels alone never imply incompressibility.
    """

    dim: int
    name: str = "custom"
    grid_spacing: Sequence[float] = field(default_factory=tuple)
    time_step: Optional[float] = None
    periodic: bool = True

    energy_type: str = "generic_l2"
    energy_law: str = "reference"
    energy_callback: Optional[TensorCallback] = None
    gamma: float = 1.4
    gravity: float = 1.0
    sound_speed: float = 1.0
    density_channel: Optional[int] = None
    pressure_channel: Optional[int] = None
    height_channel: Optional[int] = None
    vorticity_channel: Optional[int] = None
    velocity_channels: Sequence[int] = field(default_factory=tuple)
    magnetic_channels: Sequence[int] = field(default_factory=tuple)

    constraints: Sequence[ConstraintSpec] = field(default_factory=tuple)
    positivity_channels: Sequence[int] = field(default_factory=tuple)
    balance: Optional[BalanceSpec] = None
    balance_channel: Optional[int] = None
    custom_constraint_callbacks: Mapping[str, TensorCallback] = field(default_factory=dict)

    scaling_transform: Optional[TensorCallback] = None
    scaling_time_factor: Optional[TensorCallback] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_task_spec(self) -> TaskSpec:
        constraints = list(self.constraints)
        existing_names = {spec.name for spec in constraints}
        for channel in self.positivity_channels:
            name = f"positivity_channel_{channel}"
            if name not in existing_names:
                constraints.append(ConstraintSpec(name, "positivity", (int(channel),)))
        return TaskSpec(
            name=self.name,
            label=self.name,
            dim=int(self.dim),
            globs={"train": [], "val": [], "test": []},
            periodic=bool(self.periodic),
            grid_spacing=tuple(float(value) for value in self.grid_spacing),
            time_step=self.time_step,
            energy_type=self.energy_type,
            energy_law=self.energy_law,
            energy_callback=self.energy_callback,
            gamma=float(self.gamma),
            gravity=float(self.gravity),
            sound_speed=float(self.sound_speed),
            density_channel=self.density_channel,
            pressure_channel=self.pressure_channel,
            height_channel=self.height_channel,
            vorticity_channel=self.vorticity_channel,
            velocity_channels=[int(value) for value in self.velocity_channels],
            magnetic_channels=[int(value) for value in self.magnetic_channels],
            constraints=tuple(constraints),
            positivity_channels=[int(value) for value in self.positivity_channels],
            balance=self.balance,
            balance_channel=self.balance_channel,
            scaling_transform=self.scaling_transform,
            scaling_time_factor=self.scaling_time_factor,
            custom_constraint_callbacks=dict(self.custom_constraint_callbacks),
        )


@dataclass
class DiagnosticResult:
    """Samples plus explicit applicability and provenance metadata."""

    samples: Dict[str, np.ndarray]
    applicability: Dict[str, DiagnosticApplicability] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    by_horizon: Dict[float, DiagnosticResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in self.samples:
            self.applicability.setdefault(name, DiagnosticApplicability(True))
        self._summary()

    def _summary(self) -> Dict[str, Any]:
        return summarize_metric_rows(
            {},
            self.samples,
            {name: (status.available, status.reason) for name, status in self.applicability.items()},
        )

    @property
    def means(self) -> Dict[str, float]:
        summary = self._summary()
        return {name: summary[name] for name in self.samples}

    @property
    def standard_deviations(self) -> Dict[str, float]:
        summary = self._summary()
        return {name: summary[f"{name}_std"] for name in self.samples}

    @property
    def sample_counts(self) -> Dict[str, int]:
        summary = self._summary()
        return {name: summary[f"{name}_n"] for name in self.samples}

    @property
    def total_counts(self) -> Dict[str, int]:
        return {name: int(np.asarray(value).size) for name, value in self.samples.items()}

    @property
    def execution_status(self) -> Dict[str, str]:
        summary = self._summary()
        return {name: summary[f"{name}_status"] for name in self.samples}

    def as_dict(self) -> Dict[str, float]:
        """Backward-compatible dictionary of finite-sample means."""

        return self.means

    def as_report(self) -> Dict[str, Any]:
        return {
            "means": self.means,
            "standard_deviations": self.standard_deviations,
            "sample_counts": self.sample_counts,
            "total_counts": self.total_counts,
            "execution_status": self.execution_status,
            "applicability": {
                name: {"available": status.available, "reason": status.reason}
                for name, status in self.applicability.items()
            },
            "metadata": dict(self.metadata),
            "by_horizon": {str(horizon): result.as_report() for horizon, result in self.by_horizon.items()},
        }


class DiagnosticPanel:
    """Post-hoc audit panel for tensors shaped ``(B,C,*spatial)``."""

    def __init__(self, config: DiagnosticConfig | TaskSpec):
        if isinstance(config, TaskSpec):
            self.task = config
            self.metadata: Dict[str, Any] = {}
        else:
            self.task = config.to_task_spec()
            self.metadata = dict(config.metadata)

    def evaluate_pair(
        self,
        initial: torch.Tensor | np.ndarray,
        reference: torch.Tensor | np.ndarray,
        prediction: torch.Tensor | np.ndarray,
        *,
        delta_t: Optional[float] = None,
        control_volume: Optional[Sequence[slice]] = None,
    ) -> DiagnosticResult:
        initial_tensor = self._ensure_state(initial)
        reference_tensor = self._ensure_state(reference, like=initial_tensor)
        prediction_tensor = self._ensure_state(prediction, like=initial_tensor)

        samples: Dict[str, np.ndarray] = {
            "rel_l2": self._detach(rel_l2(prediction_tensor, reference_tensor)),
            "energy_violation": self._detach(energy_law_error(initial_tensor, prediction_tensor, self.task)),
            "energy_reference_error": self._detach(
                energy_reference_error(reference_tensor, prediction_tensor, self.task)
            ),
            "integral_balance": self._detach(
                integral_balance_error(
                    initial_tensor,
                    prediction_tensor,
                    self.task,
                    delta_t=delta_t,
                    control_volume=control_volume,
                )
            ),
            "global_conserved_drift": self._detach(
                global_conserved_quantity_drift(initial_tensor, prediction_tensor, self.task)
            ),
        }
        constraint_components = constraint_errors(prediction_tensor, self.task)
        for name, value in constraint_components.items():
            samples[name] = self._detach(value)
        samples["constraint"] = (
            self._detach(next(iter(constraint_components.values())))
            if len(constraint_components) == 1
            else np.full(initial_tensor.shape[0], np.nan)
        )
        applicability = {
            "rel_l2": DiagnosticApplicability(True),
            "energy_violation": DiagnosticApplicability(
                self.task.energy_law in {"conservative", "dissipative"},
                "Energy law is reference-only or disabled."
                if self.task.energy_law not in {"conservative", "dissipative"}
                else "",
            ),
            "energy_reference_error": DiagnosticApplicability(
                self.task.energy_type != "none" or self.task.energy_callback is not None,
                "No energy functional is configured."
                if self.task.energy_type == "none" and self.task.energy_callback is None
                else "",
            ),
            "integral_balance": DiagnosticApplicability(
                self.task.balance is not None,
                "No conserved-quantity and flux contract is configured." if self.task.balance is None else "",
            ),
            "global_conserved_drift": DiagnosticApplicability(
                self.task.balance is not None or self.task.balance_channel is not None,
                "No conserved quantity is configured."
                if self.task.balance is None and self.task.balance_channel is None
                else "",
            ),
            "constraint": DiagnosticApplicability(
                len(constraint_components) == 1,
                ""
                if len(constraint_components) == 1
                else "No physical constraint is configured."
                if not constraint_components
                else "Multiple constraints are reported separately; no heterogeneous composite is defined.",
            ),
        }
        applicability.update({name: DiagnosticApplicability(True) for name in constraint_components})
        return DiagnosticResult(samples, applicability, self._report_metadata())

    def semigroup_error(
        self,
        direct_prediction: torch.Tensor | np.ndarray,
        composed_prediction: torch.Tensor | np.ndarray,
    ) -> DiagnosticResult:
        direct = self._ensure_state(direct_prediction)
        composed = self._ensure_state(composed_prediction, like=direct)
        return DiagnosticResult(
            {"semigroup": self._detach(semigroup_error(direct, composed))},
            {"semigroup": DiagnosticApplicability(True)},
            self._report_metadata(),
        )

    def generator_discrepancy(
        self,
        initial: torch.Tensor | np.ndarray,
        prediction_a: torch.Tensor | np.ndarray,
        prediction_b: torch.Tensor | np.ndarray,
        delta_a: float = 1.0,
        delta_b: float = 2.0,
        reference_scale: Optional[float] = None,
    ) -> DiagnosticResult:
        initial_tensor = self._ensure_state(initial)
        prediction_a_tensor = self._ensure_state(prediction_a, like=initial_tensor)
        prediction_b_tensor = self._ensure_state(prediction_b, like=initial_tensor)
        discrepancy = generator_discrepancy(
            initial_tensor,
            prediction_a_tensor,
            prediction_b_tensor,
            delta_a,
            delta_b,
        )
        samples = {"generator": self._detach(discrepancy)}
        applicability = {"generator": DiagnosticApplicability(True)}
        if reference_scale is not None:
            if not np.isfinite(reference_scale) or reference_scale < 0:
                raise ValueError("reference_scale must be finite and nonnegative")
            if reference_scale <= 1.0e-8:
                samples["generator_ratio_to_reference"] = np.full_like(samples["generator"], np.nan)
                applicability["generator_ratio_to_reference"] = DiagnosticApplicability(
                    False, "Reference scale is no larger than epsilon."
                )
            else:
                samples["generator_ratio_to_reference"] = self._detach(discrepancy / (float(reference_scale) + 1.0e-8))
                applicability["generator_ratio_to_reference"] = DiagnosticApplicability(True)
        return DiagnosticResult(
            samples,
            applicability,
            self._report_metadata(),
        )

    def perturbation_response(
        self,
        reference_base: torch.Tensor | np.ndarray,
        reference_perturbed: torch.Tensor | np.ndarray,
        model_base: torch.Tensor | np.ndarray,
        model_perturbed: torch.Tensor | np.ndarray,
    ) -> DiagnosticResult:
        reference_base_tensor = self._ensure_state(reference_base)
        reference_perturbed_tensor = self._ensure_state(reference_perturbed, like=reference_base_tensor)
        model_base_tensor = self._ensure_state(model_base, like=reference_base_tensor)
        model_perturbed_tensor = self._ensure_state(model_perturbed, like=reference_base_tensor)
        value = perturbation_response_error(
            reference_base_tensor,
            reference_perturbed_tensor,
            model_base_tensor,
            model_perturbed_tensor,
        )
        return DiagnosticResult(
            {"perturbation_response": self._detach(value)},
            {"perturbation_response": DiagnosticApplicability(True)},
            self._report_metadata(),
        )

    def scaling_error(
        self,
        prediction_of_scaled_state: torch.Tensor | np.ndarray,
        transformed_prediction: torch.Tensor | np.ndarray,
    ) -> DiagnosticResult:
        scaled = self._ensure_state(prediction_of_scaled_state)
        transformed = self._ensure_state(transformed_prediction, like=scaled)
        value = scaling_law_error(scaled, transformed)
        return DiagnosticResult(
            {"scaling_law": self._detach(value)},
            {"scaling_law": DiagnosticApplicability(True)},
            self._report_metadata(),
        )

    def evaluate_propagator(
        self,
        initial: torch.Tensor | np.ndarray,
        references: Mapping[float, torch.Tensor | np.ndarray],
        propagator: Propagator,
        *,
        perturbed_initial: Optional[torch.Tensor | np.ndarray] = None,
        perturbed_references: Optional[Mapping[float, torch.Tensor | np.ndarray]] = None,
        scale: Optional[float] = None,
    ) -> DiagnosticResult:
        """Audit a callable or saved-prediction adapter over multiple horizons.

        Direct saved endpoints support state, energy, balance, constraint, and
        generator metrics.  Semigroup consistency additionally requires either a
        genuinely callable propagator or explicitly supplied composed endpoints.
        """

        initial_tensor = self._ensure_state(initial)
        horizons = sorted(float(value) for value in references)
        if not horizons or any(not np.isfinite(value) or value <= 0 for value in horizons):
            raise ValueError("references must contain finite positive horizons")
        if (perturbed_initial is None) != (perturbed_references is None):
            raise ValueError("perturbed_initial and perturbed_references must be supplied together")
        if isinstance(propagator, SavedPredictionAdapter) and (
            perturbed_initial is not None or perturbed_references is not None or scale is not None
        ):
            raise ValueError(
                "Saved direct endpoints cannot be reused for perturbed or scaled initial states; "
                "supply those arrays to perturbation_response() or scaling_error() explicitly"
            )
        if scale is not None and (not np.isfinite(scale) or scale <= 0 or scale == 1):
            raise ValueError("scale must be finite, positive, and different from one")

        predictions: Dict[float, torch.Tensor] = {}
        per_horizon: Dict[float, DiagnosticResult] = {}
        batch_size = initial_tensor.shape[0]
        for horizon in horizons:
            reference = self._ensure_state(references[horizon], like=initial_tensor)
            prediction = self._ensure_state(propagator(initial_tensor, horizon), like=initial_tensor)
            predictions[horizon] = prediction
            pair = self.evaluate_pair(initial_tensor, reference, prediction, delta_t=horizon)
            samples = dict(pair.samples)
            applicability = dict(pair.applicability)
            metadata = {**pair.metadata, "horizon": horizon, "dtype": str(prediction.dtype)}

            def unavailable(name: str, reason: str, sample_rows=samples, statuses=applicability) -> None:
                sample_rows[name] = np.full(batch_size, np.nan)
                statuses[name] = DiagnosticApplicability(False, reason)

            try:
                if hasattr(propagator, "composed"):
                    composed = self._ensure_state(propagator.composed(initial_tensor, horizon), like=initial_tensor)
                else:
                    midpoint = self._ensure_state(propagator(initial_tensor, horizon / 2), like=initial_tensor)
                    composed = self._ensure_state(propagator(midpoint, horizon / 2), like=initial_tensor)
                samples["semigroup"] = self._detach(semigroup_error(prediction, composed))
                reason = (
                    "Integer-step composition is equal by construction for a deterministic one-step map."
                    if isinstance(propagator, AutoregressivePropagator)
                    else ""
                )
                applicability["semigroup"] = DiagnosticApplicability(True, reason)
            except (KeyError, NotImplementedError) as error:
                unavailable("semigroup", str(error))

            first = horizons[0]
            if horizon > first:
                samples["generator"] = self._detach(
                    generator_discrepancy(
                        initial_tensor,
                        predictions[first],
                        prediction,
                        first,
                        horizon,
                    )
                )
                applicability["generator"] = DiagnosticApplicability(True)
                metadata["generator_horizons"] = [first, horizon]
            else:
                unavailable("generator", "Two distinct positive horizons are required.")

            if perturbed_initial is not None and perturbed_references is not None:
                perturbed_state = self._ensure_state(perturbed_initial, like=initial_tensor)
                perturbed_reference = self._ensure_state(perturbed_references[horizon], like=initial_tensor)
                perturbed_prediction = self._ensure_state(propagator(perturbed_state, horizon), like=initial_tensor)
                samples["perturbation_response"] = self._detach(
                    perturbation_response_error(
                        reference,
                        perturbed_reference,
                        prediction,
                        perturbed_prediction,
                    )
                )
                applicability["perturbation_response"] = DiagnosticApplicability(True)
            else:
                unavailable("perturbation_response", "Paired perturbed initial/reference states were not supplied.")

            transform = self.task.scaling_transform
            if transform is None and self.task.scaling == "linear_amplitude":
                transform = _linear_amplitude
            if scale is not None and transform is not None:
                factor = float(self.task.scaling_time_factor(scale)) if self.task.scaling_time_factor else 1.0
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("scaling_time_factor must return a finite positive value")
                scaled_initial = self._ensure_state(transform(initial_tensor, scale), like=initial_tensor)
                scaled_prediction = self._ensure_state(propagator(scaled_initial, horizon), like=initial_tensor)
                base_prediction = self._ensure_state(propagator(initial_tensor, factor * horizon), like=initial_tensor)
                transformed = self._ensure_state(transform(base_prediction, scale), like=initial_tensor)
                samples["scaling_law"] = self._detach(scaling_law_error(scaled_prediction, transformed))
                applicability["scaling_law"] = DiagnosticApplicability(True)
            else:
                unavailable("scaling_law", "A scale and scaling transformation are required.")
            per_horizon[horizon] = DiagnosticResult(samples, applicability, metadata)

        combined: Dict[str, np.ndarray] = {}
        applicability = {}
        for name in next(iter(per_horizon.values())).samples:
            applicable = [result for result in per_horizon.values() if result.applicability[name].available]
            selected = applicable or list(per_horizon.values())
            combined[name] = np.concatenate([result.samples[name].reshape(-1) for result in selected])
            reasons = list(
                dict.fromkeys(
                    result.applicability[name].reason
                    for result in per_horizon.values()
                    if result.applicability[name].reason
                )
            )
            applicability[name] = DiagnosticApplicability(bool(applicable), "; ".join(reasons))
        metadata = {
            **self._report_metadata(),
            "horizons": horizons,
            "propagator": getattr(propagator, "name", type(propagator).__name__),
            "aggregation": "Pooled applicable samples; by_horizon preserves all requested horizons and availability.",
        }
        return DiagnosticResult(combined, applicability, metadata, per_horizon)

    def _report_metadata(self) -> Dict[str, Any]:
        return {
            **self.metadata,
            "task": self.task.name,
            "dim": self.task.dim,
            "grid_spacing": tuple(self.task.grid_spacing),
            "time_step": self.task.time_step,
            "periodic": self.task.periodic,
            "channel_names": tuple(self.task.channel_names),
        }

    def _ensure_state(
        self,
        value: torch.Tensor | np.ndarray,
        like: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone()
        else:
            tensor = torch.as_tensor(np.asarray(value).copy())
        if tensor.is_complex():
            raise ValueError("Diagnostic states must be real-valued")
        if tensor.dtype not in {torch.float32, torch.float64}:
            tensor = tensor.to(torch.float32)
        if like is not None:
            tensor = tensor.to(device=like.device)
        expected_without_batch = self.task.dim + 1
        expected_with_batch = self.task.dim + 2
        if tensor.ndim == expected_without_batch:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != expected_with_batch:
            raise ValueError(
                f"Expected (B,C,*spatial) or (C,*spatial) for dim={self.task.dim}; received {tuple(tensor.shape)}"
            )
        if like is not None and tuple(tensor.shape) != tuple(like.shape):
            raise ValueError(f"Shape mismatch: expected {tuple(like.shape)}, got {tuple(tensor.shape)}")
        if any(size < 1 for size in tensor.shape):
            raise ValueError("Diagnostic states cannot have empty dimensions")
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("Diagnostic state contains NaN or Inf; the audit cannot omit failed predictions")
        self.task.validate_state_channels(int(tensor.shape[1]))
        return tensor

    @staticmethod
    def _detach(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().astype(float, copy=False)


__all__ = [
    "DiagnosticApplicability",
    "DiagnosticConfig",
    "DiagnosticPanel",
    "DiagnosticResult",
]
