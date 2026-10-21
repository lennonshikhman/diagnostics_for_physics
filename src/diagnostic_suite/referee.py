"""Reproducible PDE-based installation demonstration with analytic references."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import __version__
from .api import DiagnosticConfig, DiagnosticPanel
from .constants import EPS
from .propagators import CallablePropagator, SavedPredictionAdapter
from .runner import _source_fingerprint
from .tasks import BalanceSpec, ConstraintSpec


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _number(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if value == 0:
        return "$0$"
    if 0.01 <= abs(value) < 100:
        return f"${value:.3f}$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def run_referee_demo(outdir: Path | str, *, grid: int = 64) -> dict[str, Any]:
    """Compare FFT propagators with independent closed-form Fourier-mode references.

    The test problem is u_t + c u_x = nu u_xx on the unit periodic interval,
    with c=0.3, nu=0.02 and four smooth, positive initial states. All arithmetic
    is float64 on CPU. The five audit cases use the same references and horizons.
    """
    if grid < 16:
        raise ValueError("The referee demonstration needs at least 16 grid points")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    speed, viscosity = 0.3, 0.02
    horizons = (0.05, 0.1, 0.2)
    x = (torch.arange(grid, dtype=torch.float64) / grid)[None, None]
    phase = torch.arange(4, dtype=torch.float64)[:, None, None] * 0.4

    def reference(horizon: float, *, perturbation: bool = False) -> torch.Tensor:
        coordinate = x - speed * horizon
        state = (
            1.0
            + 0.2 * math.exp(-viscosity * (2 * math.pi) ** 2 * horizon) * torch.sin(2 * math.pi * coordinate + phase)
            + 0.1 * math.exp(-viscosity * (4 * math.pi) ** 2 * horizon) * torch.cos(4 * math.pi * coordinate - phase)
        )
        if perturbation:
            state = state + 0.001 * math.exp(-viscosity * (6 * math.pi) ** 2 * horizon) * torch.cos(
                6 * math.pi * coordinate
            )
        return state

    wave_number = 2 * math.pi * torch.fft.rfftfreq(grid, d=1 / grid, dtype=torch.float64)

    def fft_step(state: torch.Tensor, horizon: float, *, diffusion: float = viscosity) -> torch.Tensor:
        multiplier = torch.exp((-1j * speed * wave_number - diffusion * wave_number.square()) * horizon)
        return torch.fft.irfft(torch.fft.rfft(state, dim=-1) * multiplier, n=grid, dim=-1)

    def flux(state: torch.Tensor, task: Any) -> torch.Tensor:
        derivative = torch.fft.irfft(1j * wave_number * torch.fft.rfft(state, dim=-1), n=grid, dim=-1)
        return speed * state - viscosity * derivative

    panel = DiagnosticPanel(
        DiagnosticConfig(
            dim=1,
            name="periodic_advection_diffusion",
            grid_spacing=(1 / grid,),
            energy_type="generic_l2",
            energy_law="dissipative",
            balance=BalanceSpec(conserved_quantity=lambda state, task: state, flux=flux),
            constraints=(ConstraintSpec("scalar_positivity", "positivity", (0,)),),
            scaling_transform=lambda state, factor: factor * state,
            metadata={"state_units": "dimensionless", "reference": "closed_form_fourier_modes"},
        )
    )
    initial = reference(0.0)
    references = {h: reference(h) for h in horizons}
    cases = {
        "correct": fft_step,
        "frozen": lambda state, horizon: state,
        "clock_defect": lambda state, horizon: fft_step(state, horizon + 0.75 * horizon * horizon),
        "overdiffusive": lambda state, horizon: fft_step(state, horizon, diffusion=4 * viscosity),
        "source_leak": lambda state, horizon: fft_step(state, horizon) + 0.4 * horizon,
    }
    reports = {}
    rows = []
    for name, function in cases.items():
        result = panel.evaluate_propagator(
            initial,
            references,
            CallablePropagator(function, name=name),
            perturbed_initial=reference(0.0, perturbation=True),
            perturbed_references={h: reference(h, perturbation=True) for h in horizons},
            scale=1.25,
        )
        reports[name] = result.as_report()
        for horizon, local in result.by_horizon.items():
            row = {"case": name, "horizon": horizon, **local.means}
            row.update({f"{key}_n": value for key, value in local.sample_counts.items()})
            rows.append(row)
    table = pd.DataFrame(rows)
    baselines = table.loc[table["case"].eq("correct")].set_index("horizon")["generator"]
    table["generator_reference"] = table["horizon"].map(baselines)
    table["generator_ratio_to_reference"] = table["generator"] / (table["generator_reference"] + EPS)
    selected = table[table["horizon"].eq(horizons[-1])].set_index("case")
    checks: dict[str, bool] = {
        "fft_matches_independent_closed_form": bool(selected.loc["correct", "rel_l2"] < 1e-12),
        "exact_composition": bool(selected.loc["correct", "semigroup"] < 1e-12),
        "nonzero_finite_time_reference_generator": bool(selected.loc["correct", "generator"] > 1e-4),
        "paired_perturbation_matches": bool(selected.loc["correct", "perturbation_response"] < 1e-10),
        "exact_linear_scaling": bool(selected.loc["correct", "scaling_law"] < 1e-12),
        "positive_reference_states": bool(selected.loc["correct", "scalar_positivity"] == 0),
        "periodic_mass_conservation": bool(selected.loc["correct", "integral_balance"] < 1e-12),
        "clock_defect_detected": bool(selected.loc["clock_defect", "semigroup"] > 1e-4),
        "overdiffusion_obeys_dissipation": bool(selected.loc["overdiffusive", "energy_violation"] < 1e-12),
        "overdiffusion_changes_reference_energy": bool(selected.loc["overdiffusive", "energy_reference_error"] > 1e-3),
        "unconfigured_source_detected": bool(selected.loc["source_leak", "integral_balance"] > 1e-2),
        "frozen_map_is_inaccurate_despite_zero_semigroup": bool(
            selected.loc["frozen", "rel_l2"] > 1e-2
            and selected.loc["frozen", "semigroup"] == 0
            and selected.loc["frozen", "generator"] == 0
        ),
    }
    saved = panel.evaluate_propagator(initial, references, SavedPredictionAdapter(references))
    checks["saved_endpoints_do_not_certify_composition"] = not saved.applicability["semigroup"].available
    invalid = initial.clone()
    invalid[0] = float("nan")
    try:
        panel.evaluate_pair(initial, references[horizons[-1]], invalid)
    except FloatingPointError:
        checks["invalid_prediction_rejected"] = True
    else:
        checks["invalid_prediction_rejected"] = False
    tiny = panel.evaluate_pair(initial, initial, initial + 1e-8, delta_t=horizons[-1]).means["rel_l2"]
    checks["double_precision_signal_retained"] = 1e-9 < tiny < 1e-7
    negative = panel.evaluate_pair(initial, initial, -initial, delta_t=horizons[-1]).means["scalar_positivity"]
    checks["negative_state_detected"] = negative > 0.99

    table.to_csv(outdir / "metrics.csv", index=False)
    (outdir / "reports.json").write_text(json.dumps(_json_safe(reports), indent=2, allow_nan=False) + "\n")
    (outdir / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    manifest = {
        "package_version": __version__,
        "source_sha256": _source_fingerprint(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float64",
        "torch_threads": torch.get_num_threads(),
        "grid": grid,
        "batch_size": 4,
        "speed": speed,
        "viscosity": viscosity,
        "horizons": horizons,
        "elapsed_seconds": time.perf_counter() - started,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "benchmark_training": False,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    labels = {
        "correct": "Correct FFT",
        "frozen": "Frozen map",
        "clock_defect": "Clock defect",
        "overdiffusive": "Excess diffusion",
        "source_leak": "Source leak",
    }
    metrics = (
        "rel_l2",
        "semigroup",
        "generator_ratio_to_reference",
        "energy_reference_error",
        "integral_balance",
        "perturbation_response",
    )
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Propagator & $E_{L^2}$ & $E_{\rm sg}$ & $R_{\rm gen}$ & $E_{\rm ref}$ & $E_{\rm bal}$ & $E_{\rm pert}$ \\",
        r"\midrule",
    ]
    for name in cases:
        lines.append(" & ".join([labels[name]] + [_number(float(selected.loc[name, m])) for m in metrics]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (outdir / "diagnostics.tex").write_text("\n".join(lines) + "\n")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Referee demonstration checks failed: {failed}; see {outdir}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs_referee"))
    parser.add_argument("--grid", type=int, default=64)
    args = parser.parse_args(argv)
    manifest = run_referee_demo(args.outdir, grid=args.grid)
    print(f"Referee demonstration: {manifest['checks_passed']}/{manifest['checks_total']} checks passed. {args.outdir}")


if __name__ == "__main__":
    main()
