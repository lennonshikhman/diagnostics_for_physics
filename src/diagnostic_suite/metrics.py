from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import EPS
from .tasks import BalanceSpec, ConstraintSpec, TaskSpec


def rel_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative Euclidean state error."""

    numerator = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(target.flatten(1), dim=1)
    return numerator / (denominator + EPS)


def semigroup_error(direct_prediction: torch.Tensor, composed_prediction: torch.Tensor) -> torch.Tensor:
    """Relative discrepancy between direct and composed predictions."""

    return rel_l2(composed_prediction, direct_prediction)


def generator_discrepancy(
    initial: torch.Tensor,
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    delta_a: float,
    delta_b: float,
) -> torch.Tensor:
    """Compare two finite-difference estimates of the local generator."""

    if not all(math.isfinite(value) and value > 0 for value in (delta_a, delta_b)):
        raise ValueError("Generator horizons must be finite and positive")
    if delta_a == delta_b:
        raise ValueError("Generator horizons must be distinct")
    generator_a = (prediction_a - initial) / float(delta_a)
    generator_b = (prediction_b - initial) / float(delta_b)
    numerator = torch.linalg.vector_norm((generator_a - generator_b).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(generator_a.flatten(1), dim=1) + torch.linalg.vector_norm(
        generator_b.flatten(1), dim=1
    )
    return numerator / (denominator + EPS)


def grad_dim(
    field: torch.Tensor,
    axis: int,
    *,
    spacing: float = 1.0,
    periodic: bool = True,
) -> torch.Tensor:
    """Centered first derivative along a spatial axis.

    ``field`` is shaped ``(batch, channels, *spatial)``.  Nonperiodic boundary
    points use first-order one-sided differences; interior points remain centered.
    """

    if spacing <= 0:
        raise ValueError("Grid spacing must be positive")
    tensor_dim = 2 + axis
    if tensor_dim >= field.ndim:
        raise ValueError(f"Spatial axis {axis} is invalid for shape {tuple(field.shape)}")
    derivative = (torch.roll(field, -1, dims=tensor_dim) - torch.roll(field, 1, dims=tensor_dim)) / (
        2.0 * float(spacing)
    )
    if periodic:
        return derivative

    low = [slice(None)] * field.ndim
    low_next = [slice(None)] * field.ndim
    high = [slice(None)] * field.ndim
    high_prev = [slice(None)] * field.ndim
    low[tensor_dim] = 0
    low_next[tensor_dim] = 1
    high[tensor_dim] = -1
    high_prev[tensor_dim] = -2
    derivative[tuple(low)] = (field[tuple(low_next)] - field[tuple(low)]) / float(spacing)
    derivative[tuple(high)] = (field[tuple(high)] - field[tuple(high_prev)]) / float(spacing)
    return derivative


def divergence(
    vector_field: torch.Tensor,
    dim: int,
    *,
    periodic: bool = True,
    grid_spacing: Sequence[float] = (),
) -> torch.Tensor:
    if vector_field.shape[1] < dim:
        raise ValueError(f"Need {dim} vector components, received {vector_field.shape[1]}")
    spacing = tuple(float(value) for value in grid_spacing) or tuple(1.0 for _ in range(dim))
    if len(spacing) != dim:
        raise ValueError("grid_spacing must have one entry per spatial dimension")
    result = torch.zeros_like(vector_field[:, 0:1])
    for axis in range(dim):
        result = result + grad_dim(
            vector_field[:, axis : axis + 1],
            axis,
            spacing=spacing[axis],
            periodic=periodic,
        )
    return result


def gradient_norm(
    vector_field: torch.Tensor,
    dim: int,
    *,
    periodic: bool = True,
    grid_spacing: Sequence[float] = (),
) -> torch.Tensor:
    """Frobenius norm of the discrete vector-field Jacobian per batch item."""

    spacing = tuple(float(value) for value in grid_spacing) or tuple(1.0 for _ in range(dim))
    squared = torch.zeros(vector_field.shape[0], device=vector_field.device, dtype=vector_field.dtype)
    for component in range(min(dim, vector_field.shape[1])):
        for axis in range(dim):
            derivative = grad_dim(
                vector_field[:, component : component + 1],
                axis,
                spacing=spacing[axis],
                periodic=periodic,
            )
            squared = squared + torch.linalg.vector_norm(derivative.flatten(1), dim=1).square()
    return torch.sqrt(squared)


def _effective_grid_spacing(
    state: torch.Tensor,
    task: TaskSpec,
    *,
    cell_centered: bool = False,
) -> Tuple[float, ...]:
    if task.grid_spacing:
        return tuple(float(value) for value in task.grid_spacing)
    if task.domain_extent:
        spacing: List[float] = []
        for extent, size in zip(task.domain_extent, state.shape[2:], strict=True):
            denominator = int(size) if task.periodic or cell_centered else max(1, int(size) - 1)
            spacing.append(float(extent) / denominator)
        return tuple(spacing)
    raise ValueError(f"Task {task.name!r} requires physical grid_spacing or domain_extent for this diagnostic")


def _spatial_mean(field: torch.Tensor) -> torch.Tensor:
    return field.flatten(1).mean(dim=1)


def _kinetic_from_vorticity(vorticity: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    if task.dim != 2:
        raise ValueError("Vorticity-to-velocity kinetic energy is implemented for 2D periodic grids")
    if not task.periodic:
        raise ValueError("Vorticity-to-velocity inversion requires a periodic task")
    spacing = task.grid_spacing or (1.0, 1.0)
    omega = vorticity[:, 0]
    nx, ny = int(omega.shape[-2]), int(omega.shape[-1])
    kx = 2.0 * math.pi * torch.fft.fftfreq(nx, d=float(spacing[0]), device=omega.device)
    ky = 2.0 * math.pi * torch.fft.fftfreq(ny, d=float(spacing[1]), device=omega.device)
    kx_grid = kx.view(1, nx, 1)
    ky_grid = ky.view(1, 1, ny)
    wave_number_squared = kx_grid.square() + ky_grid.square()
    omega_hat = torch.fft.fftn(omega, dim=(-2, -1))
    safe_denominator = torch.where(
        wave_number_squared > 0,
        wave_number_squared,
        torch.ones_like(wave_number_squared),
    )
    stream_hat = omega_hat / safe_denominator
    stream_hat = torch.where(wave_number_squared > 0, stream_hat, torch.zeros_like(stream_hat))
    velocity_x = torch.fft.ifftn(1j * ky_grid * stream_hat, dim=(-2, -1)).real
    velocity_y = torch.fft.ifftn(-1j * kx_grid * stream_hat, dim=(-2, -1)).real
    return 0.5 * (velocity_x.square() + velocity_y.square()).flatten(1).mean(dim=1)


def energy(state: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    """Equation-aware energy functional in physical (decoded) variables."""

    task.validate_state_channels(int(state.shape[1]))
    if task.energy_callback is not None:
        value = task.energy_callback(state, task)
        return torch.as_tensor(value, device=state.device, dtype=state.dtype).reshape(state.shape[0])

    if task.energy_type == "none":
        return torch.full((state.shape[0],), float("nan"), device=state.device)
    if task.energy_type == "generic_l2":
        return 0.5 * state.square().flatten(1).mean(dim=1)
    if task.energy_type == "kinetic_from_vorticity":
        channel = task.vorticity_channel if task.vorticity_channel is not None else 0
        return _kinetic_from_vorticity(state[:, channel : channel + 1], task)
    if task.energy_type == "kinetic":
        channels = tuple(task.velocity_channels[: task.dim])
        if len(channels) != task.dim or max(channels) >= state.shape[1]:
            return torch.full((state.shape[0],), float("nan"), device=state.device)
        density = 0.5 * state[:, channels].square().sum(dim=1)
        return _spatial_mean(density)
    if task.energy_type == "shallow_water":
        height_channel = task.height_channel if task.height_channel is not None else 0
        velocity_channels = tuple(task.velocity_channels[: task.dim])
        if max((height_channel, *velocity_channels), default=-1) >= state.shape[1]:
            return torch.full((state.shape[0],), float("nan"), device=state.device)
        height = state[:, height_channel]
        speed_squared = state[:, velocity_channels].square().sum(dim=1)
        density = 0.5 * height * speed_squared + 0.5 * float(task.gravity) * height.square()
        return _spatial_mean(density)
    if task.energy_type == "cns":
        if task.density_channel is None or task.pressure_channel is None:
            return torch.full((state.shape[0],), float("nan"), device=state.device)
        velocity_channels = tuple(task.velocity_channels[: task.dim])
        required = (task.density_channel, task.pressure_channel, *velocity_channels)
        if len(velocity_channels) != task.dim or max(required) >= state.shape[1]:
            return torch.full((state.shape[0],), float("nan"), device=state.device)
        if task.gamma <= 1.0:
            raise ValueError("Compressible-flow gamma must exceed one")
        rho = state[:, task.density_channel]
        pressure = state[:, task.pressure_channel]
        speed_squared = state[:, velocity_channels].square().sum(dim=1)
        density = 0.5 * rho * speed_squared + pressure / (float(task.gamma) - 1.0)
        return _spatial_mean(density)
    if task.energy_type == "mhd":
        if task.density_channel is None:
            return torch.full((state.shape[0],), float("nan"), device=state.device)
        velocity_channels = tuple(task.velocity_channels[: task.dim])
        magnetic_channels = tuple(task.magnetic_channels[: task.dim])
        required = (task.density_channel, *velocity_channels, *magnetic_channels)
        if len(velocity_channels) != task.dim or len(magnetic_channels) != task.dim or max(required) >= state.shape[1]:
            return torch.full((state.shape[0],), float("nan"), device=state.device)
        rho = state[:, task.density_channel]
        speed_squared = state[:, velocity_channels].square().sum(dim=1)
        magnetic_squared = state[:, magnetic_channels].square().sum(dim=1)
        density = 0.5 * rho * speed_squared + 0.5 * magnetic_squared
        if task.pressure_channel is not None and task.pressure_channel < state.shape[1]:
            if task.gamma <= 1.0:
                raise ValueError("MHD gamma must exceed one")
            density = density + state[:, task.pressure_channel] / (float(task.gamma) - 1.0)
        return _spatial_mean(density)
    raise ValueError(f"Unknown energy type {task.energy_type!r}")


def energy_law_error(initial: torch.Tensor, prediction: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    initial_energy = energy(initial, task)
    predicted_energy = energy(prediction, task)
    if task.energy_law == "conservative":
        return (predicted_energy - initial_energy).abs() / (initial_energy.abs() + EPS)
    if task.energy_law == "dissipative":
        return torch.relu(predicted_energy - initial_energy) / (initial_energy.abs() + EPS)
    return torch.full_like(initial_energy, float("nan"))


def energy_reference_error(reference: torch.Tensor, prediction: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    reference_energy = energy(reference, task)
    predicted_energy = energy(prediction, task)
    return (predicted_energy - reference_energy).abs() / (reference_energy.abs() + EPS)


def positivity_error(state: torch.Tensor, channels: Sequence[int]) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for channel in channels:
        if channel >= state.shape[1]:
            continue
        field = state[:, channel : channel + 1]
        numerator = torch.linalg.vector_norm(torch.relu(-field).flatten(1), dim=1)
        denominator = torch.linalg.vector_norm(field.flatten(1), dim=1)
        values.append(numerator / (denominator + EPS))
    if not values:
        return torch.full((state.shape[0],), float("nan"), device=state.device)
    return torch.stack(values, dim=0).nanmean(dim=0)


def _legacy_constraint_specs(task: TaskSpec) -> Tuple[ConstraintSpec, ...]:
    specs: List[ConstraintSpec] = []
    for channel in task.positivity_channels:
        specs.append(ConstraintSpec(f"positivity_channel_{channel}", "positivity", (channel,)))
    return tuple(specs)


def constraint_errors(state: torch.Tensor, task: TaskSpec) -> Dict[str, torch.Tensor]:
    """Return explicitly named physical constraints without mixing their meanings."""

    task.validate_state_channels(int(state.shape[1]))
    specs = task.constraints or _legacy_constraint_specs(task)
    output: Dict[str, torch.Tensor] = {}
    for spec in specs:
        if spec.kind == "positivity":
            output[spec.name] = positivity_error(state, spec.channels)
        elif spec.kind == "divergence":
            if len(spec.channels) != task.dim or max(spec.channels) >= state.shape[1]:
                output[spec.name] = torch.full((state.shape[0],), float("nan"), device=state.device)
                continue
            vector = state[:, spec.channels]
            div = divergence(
                vector,
                task.dim,
                periodic=task.periodic,
                grid_spacing=_effective_grid_spacing(state, task),
            )
            numerator = torch.linalg.vector_norm(div.flatten(1), dim=1)
            denominator = gradient_norm(
                vector,
                task.dim,
                periodic=task.periodic,
                grid_spacing=_effective_grid_spacing(state, task),
            )
            output[spec.name] = numerator / (denominator + EPS)
        else:
            assert spec.callback is not None
            value = spec.callback(state, task)
            output[spec.name] = torch.as_tensor(
                value,
                device=state.device,
                dtype=state.dtype,
            ).reshape(state.shape[0])
    for name, callback in task.custom_constraint_callbacks.items():
        value = callback(state, task)
        output[name] = torch.as_tensor(value, device=state.device, dtype=state.dtype).reshape(state.shape[0])
    return output


def constraint_error(state: torch.Tensor, task: TaskSpec) -> torch.Tensor:
    """Backward-compatible aggregate of explicitly required constraints.

    New reports should use :func:`constraint_errors` and retain each component.
    """

    components = list(constraint_errors(state, task).values())
    if not components:
        return torch.full((state.shape[0],), float("nan"), device=state.device)
    return torch.stack(components, dim=0).nanmean(dim=0)


def _callback_field(value: Any, state: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=state.device, dtype=state.dtype)
    if channels == 1 and tuple(tensor.shape) == (state.shape[0], *state.shape[2:]):
        tensor = tensor.unsqueeze(1)
    if tuple(tensor.shape) != (state.shape[0], channels, *state.shape[2:]):
        raise ValueError(f"{name} callback must return shape (B, {channels}, *spatial)")
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} callback returned NaN or Inf")
    return tensor


def _quantity_field(state: torch.Tensor, balance: BalanceSpec, task: TaskSpec) -> torch.Tensor:
    if balance.conserved_quantity is not None:
        value = balance.conserved_quantity(state, task)
        return _callback_field(value, state, 1, "Conserved quantity")
    assert balance.quantity_channel is not None
    if balance.quantity_channel >= state.shape[1]:
        raise ValueError(f"Balance channel {balance.quantity_channel} is unavailable")
    return state[:, balance.quantity_channel : balance.quantity_channel + 1]


def _flux_field(state: torch.Tensor, balance: BalanceSpec, task: TaskSpec) -> torch.Tensor:
    if balance.flux is not None:
        value = balance.flux(state, task)
        return _callback_field(value, state, task.dim, "Flux")
    quantity = _quantity_field(state, balance, task)
    channels = tuple(balance.velocity_channels[: task.dim])
    if len(channels) != task.dim or max(channels) >= state.shape[1]:
        raise ValueError("Mass-balance velocity channels are unavailable")
    return quantity * state[:, channels]


def _source_field(state: torch.Tensor, balance: BalanceSpec, task: TaskSpec) -> torch.Tensor:
    if balance.source is None:
        return torch.zeros_like(_quantity_field(state, balance, task))
    value = balance.source(state, task)
    return _callback_field(value, state, 1, "Source")


def _face_flux(component: torch.Tensor, axis: int, index: int, periodic: bool) -> torch.Tensor:
    """Linear reconstruction from cell centers to a control-volume face."""

    dimension = axis + 1
    size = component.shape[dimension]
    if 0 < index < size:
        return 0.5 * (component.select(dimension, index - 1) + component.select(dimension, index))
    if periodic:
        return 0.5 * (component.select(dimension, -1) + component.select(dimension, 0))
    if size < 2:
        raise ValueError("Nonperiodic flux reconstruction requires at least two cells per axis")
    edge, neighbor = (0, 1) if index == 0 else (-1, -2)
    return 1.5 * component.select(dimension, edge) - 0.5 * component.select(dimension, neighbor)


def _normalize_control_volume(
    spatial_shape: Sequence[int],
    control_volume: Optional[Sequence[slice]],
) -> Tuple[slice, ...]:
    if control_volume is None:
        return tuple(slice(0, int(size)) for size in spatial_shape)
    if len(control_volume) != len(spatial_shape):
        raise ValueError("control_volume needs one slice per spatial dimension")
    normalized: List[slice] = []
    for item, size in zip(control_volume, spatial_shape, strict=True):
        start = 0 if item.start is None else int(item.start)
        stop = int(size) if item.stop is None else int(item.stop)
        if item.step not in {None, 1} or start < 0 or stop > int(size) or start >= stop:
            raise ValueError(f"Invalid control-volume slice {item} for axis length {size}")
        normalized.append(slice(start, stop))
    return tuple(normalized)


def integral_balance_error(
    initial: torch.Tensor,
    prediction: torch.Tensor,
    task: TaskSpec,
    *,
    delta_t: Optional[float] = None,
    control_volume: Optional[Sequence[slice]] = None,
) -> torch.Tensor:
    """Discrete control-volume residual including boundary flux and source.

    States are uniform cell-center samples (or cell averages). Volume integrals
    use cell-volume weights. Face fluxes use linear center-to-face reconstruction;
    nonperiodic boundary faces use linear extrapolation. Flux and source terms
    use a trapezoidal-in-time approximation between the
    initial and predicted states.  For a full periodic domain, the net boundary
    flux vanishes exactly.  A bare ``balance_channel`` is deliberately not enough:
    that weaker statistic is reported separately by
    :func:`global_conserved_quantity_drift`.
    """

    if task.balance is None:
        return torch.full((prediction.shape[0],), float("nan"), device=prediction.device)

    balance = task.balance
    task.validate_state_channels(int(initial.shape[1]))
    spacing = _effective_grid_spacing(initial, task, cell_centered=True)
    if delta_t is None and task.time_step is None:
        raise ValueError("A physical balance diagnostic requires delta_t or task.time_step")
    dt = float(delta_t if delta_t is not None else task.time_step)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("delta_t must be finite and positive")

    volume_slices = _normalize_control_volume(initial.shape[2:], control_volume)
    tensor_slices = (slice(None), slice(None), *volume_slices)
    quantity_initial = _quantity_field(initial, balance, task)[tensor_slices]
    quantity_prediction = _quantity_field(prediction, balance, task)[tensor_slices]
    cell_volume = float(np.prod(spacing))
    mass_initial = quantity_initial.flatten(1).sum(dim=1) * cell_volume
    mass_prediction = quantity_prediction.flatten(1).sum(dim=1) * cell_volume

    source = 0.5 * (
        _source_field(initial, balance, task)[tensor_slices] + _source_field(prediction, balance, task)[tensor_slices]
    )
    source_integral = source.flatten(1).sum(dim=1) * cell_volume

    full_domain = all(
        item.start == 0 and item.stop == int(size) for item, size in zip(volume_slices, initial.shape[2:], strict=True)
    )
    if task.periodic and full_domain:
        boundary_integral = torch.zeros_like(mass_initial)
    else:
        flux = 0.5 * (_flux_field(initial, balance, task) + _flux_field(prediction, balance, task))
        boundary_integral = torch.zeros_like(mass_initial)
        for axis, item in enumerate(volume_slices):
            other_slices = list(volume_slices)
            other_slices[axis] = slice(item.start, item.stop)
            component = flux[:, axis]
            low_index = int(item.start)
            high_index = int(item.stop)
            low = _face_flux(component, axis, low_index, task.periodic)
            high = _face_flux(component, axis, high_index, task.periodic)
            remaining_slices = tuple(other_slices[index] for index in range(task.dim) if index != axis)
            if remaining_slices:
                low = low[(slice(None), *remaining_slices)]
                high = high[(slice(None), *remaining_slices)]
            face_area = float(np.prod([spacing[index] for index in range(task.dim) if index != axis]))
            boundary_integral = (
                boundary_integral
                + (high.reshape(high.shape[0], -1).sum(dim=1) - low.reshape(low.shape[0], -1).sum(dim=1)) * face_area
            )

    residual = mass_prediction - mass_initial + dt * boundary_integral - dt * source_integral
    scale = mass_initial.abs() + mass_prediction.abs() + dt * boundary_integral.abs() + dt * source_integral.abs()
    return residual.abs() / (scale + EPS)


def global_conserved_quantity_drift(
    initial: torch.Tensor,
    prediction: torch.Tensor,
    task: TaskSpec,
) -> torch.Tensor:
    """Global drift proxy, reported separately from control-volume balance."""

    if task.balance is not None:
        initial_quantity = _quantity_field(initial, task.balance, task)
        predicted_quantity = _quantity_field(prediction, task.balance, task)
    elif task.balance_channel is not None and task.balance_channel < initial.shape[1]:
        initial_quantity = initial[:, task.balance_channel : task.balance_channel + 1]
        predicted_quantity = prediction[:, task.balance_channel : task.balance_channel + 1]
    else:
        return torch.full((prediction.shape[0],), float("nan"), device=prediction.device)
    initial_integral = initial_quantity.flatten(1).sum(dim=1)
    predicted_integral = predicted_quantity.flatten(1).sum(dim=1)
    content_scale = initial_quantity.abs().flatten(1).sum(dim=1) + predicted_quantity.abs().flatten(1).sum(dim=1)
    return (predicted_integral - initial_integral).abs() / (content_scale + EPS)


def perturbation_response_error(
    reference_base: torch.Tensor,
    reference_perturbed: torch.Tensor,
    model_base: torch.Tensor,
    model_perturbed: torch.Tensor,
) -> torch.Tensor:
    """Relative error in a model's response to a controlled perturbation."""

    reference_response = reference_perturbed - reference_base
    model_response = model_perturbed - model_base
    numerator = torch.linalg.vector_norm((model_response - reference_response).flatten(1), dim=1)
    denominator = torch.linalg.vector_norm(reference_response.flatten(1), dim=1)
    return numerator / (denominator + EPS)


def scaling_law_error(
    prediction_of_scaled_state: torch.Tensor,
    transformed_prediction: torch.Tensor,
) -> torch.Tensor:
    """Relative commutation error for a configured scaling transformation."""

    return rel_l2(prediction_of_scaled_state, transformed_prediction)


def batch_tau(batch_size: int, value: float, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), float(value), device=device)


BASE_METRIC_NAMES = [
    "rel_l2",
    "semigroup",
    "generator",
    "energy_violation",
    "energy_reference_error",
    "integral_balance",
    "global_conserved_drift",
    "constraint",
    "perturbation_response",
    "scaling_law",
]

# Public compatibility alias.
METRIC_NAMES = BASE_METRIC_NAMES


def metric_names_for_task(task: TaskSpec) -> List[str]:
    names = list(BASE_METRIC_NAMES)
    names.extend(spec.name for spec in (task.constraints or _legacy_constraint_specs(task)))
    names.extend(name for name in task.custom_constraint_callbacks if name not in names)
    return names


def summarize_metric_rows(
    prefix: Dict[str, Any],
    rows: Mapping[str, Sequence[float]],
    applicability: Optional[Mapping[str, Tuple[bool, str]]] = None,
) -> Dict[str, Any]:
    """Summarize metric samples with explicit availability and reasons."""

    output: Dict[str, Any] = dict(prefix)
    output["metric_schema_version"] = 2
    for name, values in rows.items():
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        output[name] = float(finite.mean()) if finite.size else float("nan")
        output[f"{name}_std"] = float(finite.std()) if finite.size else float("nan")
        output[f"{name}_n"] = int(finite.size)
        default_available = bool(array.size)
        available, reason = (
            applicability.get(name, (default_available, "")) if applicability is not None else (default_available, "")
        )
        if available and (not array.size or finite.size != array.size):
            raise FloatingPointError(
                f"Applicable diagnostic {name!r} failed: {finite.size}/{array.size} finite samples; "
                "non-finite predictions or callback results cannot be omitted from an audit"
            )
        if not available and finite.size:
            raise ValueError(f"Unavailable diagnostic {name!r} unexpectedly contains finite samples")
        output[f"{name}_total"] = int(array.size)
        output[f"{name}_invalid"] = int(array.size - finite.size) if available else 0
        output[f"{name}_status"] = "ok" if available else "unavailable"
        output[f"{name}_available"] = bool(available)
        output[f"{name}_reason"] = str(reason)
    return output


def add_reference_relative_columns(rows: List[Dict[str, Any]]) -> None:
    """Add excess and ratio columns for diagnostics with reference baselines."""

    by_task_seed_and_horizon: Dict[Tuple[str, Any, Any], Dict[str, Any]] = {}
    for row in rows:
        if row.get("model") == "reference" and row.get("variant") == "reference":
            by_task_seed_and_horizon[(str(row.get("task")), row.get("seed"), row.get("horizon_steps"))] = row
    for row in rows:
        key = (str(row.get("task")), row.get("seed"), row.get("horizon_steps"))
        reference = by_task_seed_and_horizon.get(key)
        if reference is None:
            # Version 0.1 tables did not carry a seed or horizon.
            reference = by_task_seed_and_horizon.get((str(row.get("task")), None, None))
        if reference is None:
            continue
        named_constraints = [
            name
            for name in reference
            if name not in BASE_METRIC_NAMES
            and f"{name}_available" in reference
            and not name.endswith(("_reference", "_ratio_to_reference", "_excess"))
        ]
        compared_metrics = [
            "generator",
            "integral_balance",
            "global_conserved_drift",
            "constraint",
            *named_constraints,
        ]
        for metric in dict.fromkeys(compared_metrics):
            try:
                value = float(row.get(metric, float("nan")))
                baseline = float(reference.get(metric, float("nan")))
            except (TypeError, ValueError):
                value = baseline = float("nan")
            row[f"{metric}_reference"] = baseline
            row[f"{metric}_excess"] = value - baseline if np.isfinite(value) and np.isfinite(baseline) else float("nan")
            ratio_available = bool(np.isfinite(value) and np.isfinite(baseline) and abs(baseline) > EPS)
            ratio_name = f"{metric}_ratio_to_reference"
            row[ratio_name] = value / (baseline + EPS) if ratio_available else float("nan")
            row[f"{ratio_name}_available"] = ratio_available
            row[f"{ratio_name}_reason"] = (
                "" if ratio_available else "Reference scale is missing or no larger than epsilon."
            )


__all__ = [
    "BASE_METRIC_NAMES",
    "METRIC_NAMES",
    "add_reference_relative_columns",
    "batch_tau",
    "constraint_error",
    "constraint_errors",
    "divergence",
    "energy",
    "energy_law_error",
    "energy_reference_error",
    "generator_discrepancy",
    "global_conserved_quantity_drift",
    "grad_dim",
    "gradient_norm",
    "integral_balance_error",
    "metric_names_for_task",
    "perturbation_response_error",
    "positivity_error",
    "rel_l2",
    "scaling_law_error",
    "semigroup_error",
    "summarize_metric_rows",
]
