"""Exact-solution verification and generator sensitivity for the revision.

These experiments evaluate the public kernels, not surrogate-model rankings.
All grids sample the same unit periodic interval and resolved physical modes.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import __version__
from .constants import EPS
from .metrics import generator_discrepancy, perturbation_response_error, rel_l2, scaling_law_error
from .runner import _source_fingerprint


def heat_mode_discrepancy(rate: float, horizon: float, state_norm: float) -> float:
    """Exact scalar result for the implemented (h, 2h) normalized discrepancy.

    expm1 avoids subtracting nearly equal exponentials. The epsilon has exactly
    the same units and placement as in the public finite-difference kernel.
    """
    increment = -math.expm1(-rate * horizon)
    a = increment / horizon
    b = -math.expm1(-2 * rate * horizon) / (2 * horizon)
    return abs(a - b) * state_norm / ((abs(a) + abs(b)) * state_norm + EPS)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_sensitivity(outdir: Path | str, *, quick: bool = False) -> dict:
    """Run time, grid, spectrum, precision, and controlled-response sweeps."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    viscosity = 0.01
    grids = (32, 64, 128, 256) if not quick else (32, 64)
    modes = (1, 2, 4, 8, 12) if not quick else (1, 4)
    horizons = np.logspace(-8, -0.7, 25 if not quick else 7)
    rows = []
    for grid in grids:
        for mode in modes:
            for dtype in (torch.float32, torch.float64):
                x = (torch.arange(grid, dtype=dtype) + 0.5) / grid
                initial = torch.sin(2 * math.pi * mode * x)[None, None]
                norm = torch.linalg.vector_norm(initial).double().item()
                continuum_rate = viscosity * (2 * math.pi * mode) ** 2
                for reference_kind in ("continuum", "centered_laplacian"):
                    rate = (
                        continuum_rate
                        if reference_kind == "continuum"
                        else 4 * viscosity * grid**2 * math.sin(math.pi * mode / grid) ** 2
                    )
                    for h in horizons:
                        first = initial * math.exp(-rate * h)
                        second = initial * math.exp(-rate * 2 * h)
                        value = generator_discrepancy(initial, first, second, h, 2 * h).item()
                        expected = heat_mode_discrepancy(rate, h, norm)
                        continuum_expected = heat_mode_discrepancy(continuum_rate, h, norm)
                        rows.append(
                            dict(
                                grid=grid,
                                mode=mode,
                                dtype=str(dtype).split(".")[-1],
                                reference_kind=reference_kind,
                                horizon=float(h),
                                nondimensional_horizon=rate * h,
                                generator=value,
                                expected=expected,
                                absolute_arithmetic_error=abs(value - expected),
                                continuum_expected=continuum_expected,
                                discretization_difference=abs(expected - continuum_expected),
                                reference_above_epsilon=value > EPS,
                            )
                        )
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "single_mode.csv", index=False)

    mixed_rows = []
    for grid in grids:
        x = ((torch.arange(grid, dtype=torch.float64) + 0.5) / grid)[None, None]
        # Keep the physical spectrum fixed across grids; no new high modes are
        # introduced by refining the grid. These are smooth finite sums, not a
        # claim about an infinite-series Sobolev regularity class.
        frequency = torch.arange(1, 13, dtype=torch.float64)[:, None]
        basis = torch.sin(2 * math.pi * frequency * x.reshape(1, -1) + 0.31 * frequency)
        for decay in (1.0, 2.0, 4.0):
            coefficients = frequency.pow(-decay)
            initial = (coefficients * basis).sum(0)[None, None]
            rate = viscosity * (2 * math.pi * frequency) ** 2
            for h in horizons:
                first = (coefficients * torch.exp(-rate * h) * basis).sum(0)[None, None]
                second = (coefficients * torch.exp(-rate * 2 * h) * basis).sum(0)[None, None]
                mixed_rows.append(
                    dict(
                        grid=grid,
                        spectral_decay=decay,
                        horizon=float(h),
                        generator=generator_discrepancy(initial, first, second, h, 2 * h).item(),
                    )
                )
    pd.DataFrame(mixed_rows).to_csv(outdir / "mixed_modes.csv", index=False)

    ratio_rows = []
    grid, mode = 64, 4
    x = (torch.arange(grid, dtype=torch.float64) + 0.5) / grid
    initial = torch.sin(2 * math.pi * mode * x)[None, None]
    rate = viscosity * (2 * math.pi * mode) ** 2
    exact_generator = -rate * initial
    rng = torch.Generator().manual_seed(20261021)
    noise = torch.randn((2, *initial.shape), dtype=torch.float64, generator=rng)
    for h in horizons:
        reference_a = initial * math.exp(-rate * h)
        reference_b = initial * math.exp(-rate * 2 * h)
        reference = generator_discrepancy(initial, reference_a, reference_b, h, 2 * h).item()
        for case in ("correct", "frozen", "double_diffusion", "clock_defect", "noisy_1e-7"):
            if case == "frozen":
                a, b = initial.clone(), initial.clone()
            elif case == "double_diffusion":
                a, b = initial * math.exp(-2 * rate * h), initial * math.exp(-4 * rate * h)
            elif case == "clock_defect":
                a = initial * math.exp(-rate * (h + 0.75 * h**2))
                b = initial * math.exp(-rate * (2 * h + 0.75 * (2 * h) ** 2))
            elif case == "noisy_1e-7":
                a, b = reference_a + 1e-7 * noise[0], reference_b + 1e-7 * noise[1]
            else:
                a, b = reference_a, reference_b
            model = generator_discrepancy(initial, a, b, h, 2 * h).item()
            ratio_rows.append(
                dict(
                    case=case,
                    horizon=float(h),
                    reference_generator=reference,
                    model_generator=model,
                    generator_ratio=model / (reference + EPS) if reference > EPS else None,
                    generator_ratio_available=reference > EPS,
                    state_error=rel_l2(b, reference_b).item(),
                    known_generator_error=rel_l2((a - initial) / h, exact_generator).item(),
                )
            )
    pd.DataFrame(ratio_rows).to_csv(outdir / "reference_ratios.csv", index=False)

    # Independent base and perturbed Fourier solutions, without interpolation.
    response_rows = []
    base = 1 + 0.2 * torch.sin(2 * math.pi * x)[None, None]
    direction = torch.cos(6 * math.pi * x)[None, None]
    for h in (0.05, 0.1, 0.2):
        target = 1 + (base - 1) * math.exp(-viscosity * (2 * math.pi) ** 2 * h)
        for amplitude in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
            target_perturbed = target + amplitude * direction * math.exp(-viscosity * (6 * math.pi) ** 2 * h)
            initial_perturbed = base + amplitude * direction
            for case in ("correct", "frozen", "source_leak"):
                prediction = target if case == "correct" else base if case == "frozen" else target + 0.4 * h
                perturbed = (
                    target_perturbed
                    if case == "correct"
                    else initial_perturbed
                    if case == "frozen"
                    else target_perturbed + 0.4 * h
                )
                for scale in (0.5, 1.25, 2.0):
                    scaled = scale * prediction if case != "source_leak" else scale * target + 0.4 * h
                    response_rows.append(
                        dict(
                            case=case,
                            horizon=h,
                            amplitude=amplitude,
                            scale=scale,
                            perturbation_response=perturbation_response_error(
                                target, target_perturbed, prediction, perturbed
                            ).item(),
                            scaling_law=scaling_law_error(scaled, scale * prediction).item(),
                        )
                    )
    response = pd.DataFrame(response_rows)
    response.to_csv(outdir / "response_scaling.csv", index=False)

    accurate = table.loc[table.dtype.eq("float64") & table.horizon.ge(1e-4)]
    selected_h = float(horizons[np.argmin(np.abs(np.log(horizons / 0.01)))])
    spatial = table.loc[
        table.dtype.eq("float64")
        & table.reference_kind.eq("centered_laplacian")
        & table["mode"].eq(1)
        & table.horizon.eq(selected_h)
    ].sort_values("grid")
    refinement = (
        spatial.discretization_difference.to_numpy()[:-1] / spatial.discretization_difference.to_numpy()[1:]
    ).tolist()
    small_h = 1e-4
    temporal_ratio = heat_mode_discrepancy(viscosity * (2 * math.pi) ** 2, small_h, 8) / heat_mode_discrepancy(
        viscosity * (2 * math.pi) ** 2, small_h / 2, 8
    )
    mixed = pd.DataFrame(mixed_rows)
    invariant = mixed.loc[mixed.horizon.eq(selected_h)].groupby("spectral_decay").generator.agg(["min", "max"])
    grid_difference = float((invariant["max"] - invariant["min"]).max())
    checks = {
        "double_precision_agrees_with_closed_form": bool(accurate.absolute_arithmetic_error.max() < 1e-10),
        "exact_finite_horizon_discrepancy_is_nonzero": bool(accurate.generator.gt(0).all()),
        "correct_perturbation_all_amplitudes": bool(
            response.loc[response.case.eq("correct"), "perturbation_response"].max() < 1e-8
        ),
        "correct_scaling_all_factors": bool(response.loc[response.case.eq("correct"), "scaling_law"].max() < 1e-13),
        "frozen_response_is_detected": bool(
            response.loc[response.case.eq("frozen"), "perturbation_response"].min() > 0.1
        ),
        "source_breaks_scaling": bool(response.loc[response.case.eq("source_leak"), "scaling_law"].min() > 1e-3),
        "first_order_finite_horizon_bias": abs(temporal_ratio - 2) < 1e-3,
        "second_order_centered_laplacian_refinement": all(3.9 < value < 4.1 for value in refinement),
        "resolved_fixed_spectrum_is_grid_invariant": grid_difference < 1e-10,
    }
    _write_json(outdir / "checks.json", checks)
    manifest = dict(
        package_version=__version__,
        source_sha256=_source_fingerprint(),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=str(torch.__version__),
        device="cpu",
        viscosity=viscosity,
        generator_pair="(h, 2h)",
        epsilon=EPS,
        single_mode_rows=len(table),
        mixed_mode_rows=len(mixed_rows),
        ratio_rows=len(ratio_rows),
        response_rows=len(response),
        checks_passed=sum(checks.values()),
        checks_total=len(checks),
        elapsed_seconds=time.perf_counter() - started,
        quick=quick,
        selected_spatial_horizon=selected_h,
        centered_laplacian_refinement_factors=refinement,
        temporal_halving_factor=temporal_ratio,
        fixed_spectrum_grid_difference=grid_difference,
    )
    _write_json(outdir / "manifest.json", manifest)
    if not all(checks.values()):
        raise AssertionError(f"Sensitivity verification failed: {checks}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs_sensitivity"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_sensitivity(args.outdir, quick=args.quick), indent=2))


if __name__ == "__main__":
    main()
