"""Generate representative timing tables from all 96 archived measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=Path("PAPER"))
    args = parser.parse_args()
    evidence = args.paper / "revision_evidence"
    frame = pd.concat(
        [pd.read_csv(evidence / f"overhead_{device}/latency.csv") for device in ("cpu", "gpu")], ignore_index=True
    )
    if len(frame) != 96 or frame.duplicated(["device", "dim", "grid", "dtype", "operation"]).any():
        raise ValueError("Expected 96 distinct overhead configurations")
    expected_calls = dict(
        state_error_kernel=0, saved_endpoint_panel=0, predict_two_endpoints=2, callable_two_horizon_panel=12
    )
    if not frame.propagator_calls_per_audit.eq(frame.operation.map(expected_calls)).all():
        raise ValueError("Measured model calls do not match documented audit scope")
    combined = evidence / "overhead"
    combined.mkdir(exist_ok=True)
    frame.to_csv(combined / "latency.csv", index=False)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Representative synchronized median latencies in milliseconds, batch size four. Two endpoint predictions use two callable evaluations; the full configured two-horizon audit uses 12. The saved endpoint panel makes no model calls. CUDA memory is additional allocator peak for the full panel, in MiB. All grids, precisions, quartiles, CPU peak RSS, and raw 31-repeat summaries are archived.}",
        r"\label{tab:overhead}",
        r"\small",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Device & Grid & Precision & State only & Saved panel & Two predictions & Full panel & CUDA MiB\\",
        r"\midrule",
    ]
    for device in ("cpu", "cuda"):
        for dim, grid in ((2, 64), (3, 32)):
            for dtype in ("float32", "float64"):
                group = frame.loc[
                    frame.device.eq(device) & frame.dim.eq(dim) & frame.grid.eq(grid) & frame.dtype.eq(dtype)
                ].set_index("operation")
                medians = [f"{group.loc[key, 'median_ms']:.3f}" for key in expected_calls]
                memory = (
                    f"{group.loc['callable_two_horizon_panel', 'additional_peak_cuda_mib']:.2f}"
                    if device == "cuda"
                    else "---"
                )
                lines.append(" & ".join([device.upper(), rf"\({grid}^{dim}\)", dtype] + medians + [memory]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (args.paper / "generated/overhead_table.tex").write_text("\n".join(lines) + "\n")
    summary = dict(
        records=len(frame),
        measured_calls=expected_calls,
        cpu_total_peak_rss_mib=float(frame.loc[frame.device.eq("cpu"), "process_peak_rss_mib"].max()),
        cuda_max_additional_allocator_mib=float(frame.additional_peak_cuda_mib.max()),
        cpu_manifest=json.loads((evidence / "overhead_cpu/manifest.json").read_text()),
        cuda_manifest=json.loads((evidence / "overhead_gpu/manifest.json").read_text()),
    )
    (combined / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
