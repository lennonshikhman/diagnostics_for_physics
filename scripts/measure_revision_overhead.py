"""Measure audit costs on explicit periodic advection--diffusion states.

Latency includes public-API validity checks and report construction. CPU RSS is
the total process high-water mark, not an incremental tensor-memory estimate.
GPU memory is peak additional PyTorch allocator usage after warmup. Calls are
counted directly. This is an audit-cost study, not a solver speed comparison.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diagnostic_suite import BalanceSpec, CallablePropagator, ConstraintSpec, DiagnosticConfig, DiagnosticPanel
from diagnostic_suite.metrics import rel_l2
from diagnostic_suite.runner import _source_fingerprint


def measure_case(dim: int, grid: int, batch: int, device: str, dtype: torch.dtype, repeats: int) -> list[dict]:
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device).manual_seed(20261021)
    initial = 1 + 0.05 * torch.randn(
        (batch, 1, *([grid] * dim)), device=target_device, dtype=dtype, generator=generator
    )
    axes = [2 * np.pi * torch.fft.fftfreq(grid, d=1 / grid, device=target_device, dtype=dtype) for _ in range(dim - 1)]
    axes.append(2 * np.pi * torch.fft.rfftfreq(grid, d=1 / grid, device=target_device, dtype=dtype))
    mesh = torch.meshgrid(*axes, indexing="ij")
    multiplier = -0.01 * sum(k.square() for k in mesh) - 0.1j * sum(mesh)
    spatial_dims = tuple(range(2, dim + 2))
    calls = [0]

    def propagate(state, horizon):
        calls[0] += 1
        return torch.fft.irfftn(
            torch.fft.rfftn(state, dim=spatial_dims) * torch.exp(multiplier * horizon),
            s=state.shape[2:],
            dim=spatial_dims,
        )

    def flux(state, task):
        spectrum = torch.fft.rfftn(state, dim=spatial_dims)
        return torch.cat(
            [
                0.1 * state - 0.01 * torch.fft.irfftn(1j * k * spectrum, s=state.shape[2:], dim=spatial_dims)
                for k in mesh
            ],
            dim=1,
        )

    horizons = (0.05, 0.1)
    references = {h: propagate(initial, h) for h in horizons}
    perturbed = initial + 1e-3 * torch.randn(initial.shape, device=target_device, dtype=dtype, generator=generator)
    perturbed_references = {h: propagate(perturbed, h) for h in horizons}
    panel = DiagnosticPanel(
        DiagnosticConfig(
            dim=dim,
            grid_spacing=(1 / grid,) * dim,
            energy_type="generic_l2",
            energy_law="dissipative",
            balance=BalanceSpec(conserved_quantity=lambda state, task: state, flux=flux),
            constraints=(ConstraintSpec("scalar_positivity", "positivity", (0,)),),
            scaling_transform=lambda state, factor: factor * state,
        )
    )
    adapter = CallablePropagator(propagate)

    def predict_endpoints():
        return {h: propagate(initial, h) for h in horizons}

    def pair_panel():
        return panel.evaluate_pair(initial, references[0.1], references[0.1], delta_t=0.1)

    def callable_panel():
        return panel.evaluate_propagator(
            initial,
            references,
            adapter,
            perturbed_initial=perturbed,
            perturbed_references=perturbed_references,
            scale=1.25,
        )

    operations = {
        "state_error_kernel": lambda: rel_l2(references[0.1], references[0.1]),
        "saved_endpoint_panel": pair_panel,
        "predict_two_endpoints": predict_endpoints,
        "callable_two_horizon_panel": callable_panel,
    }

    def synchronize():
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)

    rows = []
    for name, function in operations.items():
        for _ in range(5):
            function()
        synchronize()
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
            allocated = torch.cuda.memory_allocated(target_device)
        else:
            allocated = None
        calls[0] = 0
        samples = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter_ns()
            function()
            synchronize()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        rows.append(
            dict(
                dim=dim,
                grid=grid,
                batch=batch,
                channels=1,
                device=device,
                dtype=str(dtype).split(".")[-1],
                operation=name,
                median_ms=float(np.median(samples)),
                q25_ms=float(np.quantile(samples, 0.25)),
                q75_ms=float(np.quantile(samples, 0.75)),
                repeats=repeats,
                warmup=5,
                propagator_calls_per_audit=calls[0] / repeats,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                additional_peak_cuda_mib=(torch.cuda.max_memory_allocated(target_device) - allocated) / 2**20
                if allocated is not None
                else None,
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("outputs_overhead"))
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for device in args.devices:
        for dtype in (torch.float32, torch.float64):
            for dim, grid in ((2, 32), (2, 64), (2, 128), (3, 16), (3, 32), (3, 64)):
                rows.extend(measure_case(dim, grid, 4, device, dtype, args.repeats))
                pd.DataFrame(rows).to_csv(args.outdir / "latency.csv", index=False)
                print(f"{device} {dtype} dim={dim} grid={grid}: measured", flush=True)
    manifest = dict(
        source_sha256=_source_fingerprint(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        cpu_threads=torch.get_num_threads(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        memory_scope="CPU total process high-water RSS; CUDA additional PyTorch allocator peak after warmup.",
        timing_scope="Synchronized latency including public API validation, copies, and report construction; FFT propagator.",
        records=len(rows),
        script_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
