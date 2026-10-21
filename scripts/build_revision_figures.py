"""Render manuscript figures directly from the archived revision CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


COLORS = ("#225ea8", "#b07d18", "#b54c22", "#6b7834", "#7b5ba7")


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(figure, outdir, name):
    figure.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(outdir / f"{name}.png", bbox_inches="tight")
    plt.close(figure)


def sensitivity(root: Path, outdir: Path):
    single = pd.read_csv(root / "sensitivity/single_mode.csv")
    mixed = pd.read_csv(root / "sensitivity/mixed_modes.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), constrained_layout=True)
    exact = single.loc[single.dtype.eq("float64") & single.reference_kind.eq("continuum") & single.grid.eq(64)]
    for color, mode in zip(COLORS, (1, 4, 12), strict=False):
        frame = exact.loc[exact["mode"].eq(mode)]
        axes[0, 0].loglog(frame.horizon, frame.generator, "o-", color=color, markersize=3, label=f"mode {mode}")
    axes[0, 0].set(
        xlabel=r"first horizon $h$", ylabel=r"reference $E_{\rm gen}(h,2h)$", title="(a) Exact finite-horizon variation"
    )
    axes[0, 0].legend(frameon=False)
    selected_h = single.horizon.unique()[np.argmin(np.abs(np.log(single.horizon.unique() / 0.01)))]
    discrete = single.loc[
        single.dtype.eq("float64") & single.reference_kind.eq("centered_laplacian") & single.horizon.eq(selected_h)
    ]
    for color, mode in zip(COLORS, (1, 4, 12), strict=False):
        frame = discrete.loc[discrete["mode"].eq(mode)].sort_values("grid")
        axes[0, 1].loglog(
            frame.grid, frame.discretization_difference, "o-", color=color, markersize=3, label=f"mode {mode}"
        )
    axes[0, 1].set(
        xlabel="spatial grid size",
        ylabel=r"$|E_{\rm gen}^{\Delta x}-E_{\rm gen}|$",
        title=f"(b) Laplacian discretization, h={selected_h:.3g}",
    )
    axes[0, 1].set_xticks((32, 64, 128, 256), ("32", "64", "128", "256"))
    axes[0, 1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    precision = single.loc[single.reference_kind.eq("continuum") & single.grid.eq(64) & single["mode"].eq(4)]
    for color, dtype in zip(COLORS, ("float32", "float64"), strict=False):
        frame = precision.loc[precision.dtype.eq(dtype)]
        axes[1, 0].loglog(
            frame.horizon,
            frame.absolute_arithmetic_error.clip(lower=1e-18),
            "o-",
            color=color,
            markersize=3,
            label=dtype,
        )
    axes[1, 0].set(
        xlabel=r"first horizon $h$", ylabel="absolute arithmetic error", title="(c) Cancellation at short horizons"
    )
    axes[1, 0].legend(frameon=False)
    for color, decay in zip(COLORS, (1, 2, 4), strict=False):
        frame = mixed.loc[mixed.grid.eq(64) & mixed.spectral_decay.eq(decay)]
        axes[1, 1].loglog(
            frame.horizon, frame.generator, "o-", color=color, markersize=3, label=rf"amplitude $k^{{-{decay}}}$"
        )
    axes[1, 1].set(xlabel=r"first horizon $h$", ylabel=r"reference $E_{\rm gen}$", title="(d) Fixed 12-mode spectra")
    axes[1, 1].legend(frameon=False)
    save(fig, outdir, "generator_sensitivity")

    ratios = pd.read_csv(root / "sensitivity/reference_ratios.csv")
    response = pd.read_csv(root / "sensitivity/response_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    cases = ("correct", "frozen", "double_diffusion", "noisy_1e-7", "clock_defect")
    for color, case, marker in zip(COLORS, cases, ("o", "s", "^", "x", "D"), strict=True):
        frame = ratios.loc[ratios.case.eq(case) & ratios.generator_ratio_available]
        axes[0].semilogx(
            frame.horizon, frame.generator_ratio, marker=marker, markersize=3, color=color, label=case.replace("_", " ")
        )
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set(
        xlabel=r"first horizon $h$",
        ylabel=r"$R_{\rm gen}$",
        title="(a) Reference ratios and controlled defects",
    )
    axes[0].axhline(1, color="0.3", linestyle="--", linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=7)
    for color, case, marker in zip(COLORS, cases, ("o", "s", "^", "x", "D"), strict=True):
        frame = ratios.loc[ratios.case.eq(case)]
        axes[1].loglog(
            frame.horizon, frame.known_generator_error.clip(lower=1e-17), marker=marker, color=color, markersize=3
        )
    axes[1].set(
        xlabel=r"first horizon $h$",
        ylabel="relative error against known generator",
        title="(b) Independent generator accuracy",
    )
    save(fig, outdir, "reference_ratios")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)
    for color, case in zip(COLORS, ("correct", "frozen", "source_leak"), strict=False):
        frame = response.loc[response.case.eq(case) & response.horizon.eq(0.2) & response.scale.eq(1.25)]
        axes[0].loglog(
            frame.amplitude,
            frame.perturbation_response.clip(lower=1e-17),
            "o-",
            color=color,
            markersize=3,
            label=case.replace("_", " "),
        )
    axes[0].set(xlabel="perturbation amplitude", ylabel="response error", title="(a) Paired exact responses at t=0.2")
    axes[0].legend(frameon=False, fontsize=8)
    for color, case in zip(COLORS, ("correct", "frozen", "source_leak"), strict=False):
        frame = response.loc[response.case.eq(case) & response.horizon.eq(0.2) & response.amplitude.eq(1e-3)]
        axes[1].semilogy(
            frame.scale,
            frame.scaling_law.clip(lower=1e-17),
            "o-",
            color=color,
            markersize=3,
            label=case.replace("_", " "),
        )
    axes[1].set(
        xlabel=r"amplitude scale $\lambda$", ylabel="scaling error", title="(b) Exact linear amplitude symmetry"
    )
    save(fig, outdir, "response_scaling")
    (outdir / "figure_sources.json").write_text(
        json.dumps(
            {
                "generator_sensitivity": ["sensitivity/single_mode.csv", "sensitivity/mixed_modes.csv"],
                "reference_ratios": ["sensitivity/reference_ratios.csv"],
                "response_scaling": ["sensitivity/response_scaling.csv"],
                "zero_display_policy": "Logarithmic displays floor exact zeros at the annotated plotting floor; CSVs retain zeros.",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("PAPER/revision_evidence"))
    parser.add_argument("--outdir", type=Path, default=Path("PAPER/figures"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    style()
    sensitivity(args.evidence_root, args.outdir)


if __name__ == "__main__":
    main()
