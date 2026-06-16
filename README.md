# Diagnostics for Physics

Production-facing Python package for diagnostic consistency testing of learned PDE time propagators.

`diagnostic_suite` evaluates learned PDE simulators as approximate **time-propagation operators**, not only as predictors of discretized state fields. The package provides a lightweight public API, a command-line interface, toy dataset generation, benchmark readers, neural surrogate baselines, diagnostic metrics, tests, and reproducibility scripts.

The software is designed for post hoc auditing of learned PDE simulators, including neural operators, convolutional surrogates, autoregressive emulators, custom callable propagators, and saved prediction arrays.

Implemented diagnostics include:

- relative `L2` state error;
- semigroup consistency;
- finite-difference generator discrepancy;
- reference-normalized generator ratio;
- energy-law violation and energy-decay error;
- global integral-balance error;
- admissibility or constraint error;
- perturbation-response error when paired perturbed references are supplied;
- scaling-law error when an exact scaling relation is configured.

Unavailable diagnostics are reported as `NaN`, not zero. This is intentional: the package distinguishes between a diagnostic that is meaningfully satisfied and one that is mathematically or physically inapplicable for the available data.

## Repository layout

```text
.
├── src/diagnostic_suite/
│   ├── api.py              # Lightweight public DiagnosticPanel API
│   ├── cli.py              # Command-line entry point
│   ├── data.py             # NPY/HDF5 trajectory readers and datasets
│   ├── evaluate.py         # Diagnostic evaluation loops
│   ├── metrics.py          # Post hoc diagnostic metrics
│   ├── models.py           # FNO, DeepONet, U-Net, ResNet surrogates
│   ├── normalization.py    # Training-set normalization
│   ├── runner.py           # Experiment orchestration
│   ├── tasks.py            # Benchmark/task definitions
│   └── toy.py              # Synthetic advection-diffusion data generator
├── tests/                  # Unit and smoke tests
├── scripts/                # Reproducibility helpers
├── docs/                   # Software notes and manuscript-facing material
├── legacy/                 # Archived research scripts
├── pyproject.toml
├── LICENSE
└── CITATION.cff
```

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

On resource-constrained machines, use:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest
```

## Lightweight API usage

Use `DiagnosticPanel` when you already have predicted and reference states from your own surrogate. States should be shaped as `(batch, channels, *spatial)` or `(channels, *spatial)`.

```python
import numpy as np

from diagnostic_suite import DiagnosticConfig, DiagnosticPanel

u0 = np.random.randn(1, 2, 32, 32).astype("float32")
uref = u0 + 0.05 * np.random.randn(1, 2, 32, 32).astype("float32")
pred = u0 + 0.07 * np.random.randn(1, 2, 32, 32).astype("float32")

panel = DiagnosticPanel(
    DiagnosticConfig(
        dim=2,
        energy_type="kinetic",
        velocity_channels=(0, 1),
        balance_channel=None,
    )
)

result = panel.evaluate_pair(u0, uref, pred)
print(result.as_dict())
```

The API also exposes `semigroup_error` and `generator_discrepancy` for users who have direct/composed predictions or predictions at two short horizons.

## Quick start without external data

Generate a tiny periodic advection-diffusion dataset:

```bash
diagnostic-suite --make-toy --data-root data --toy-grid 24 --toy-steps 6
```

Inspect the generated dataset:

```bash
diagnostic-suite --data-root data --tasks toy2d --inspect-data
```

Run a small smoke experiment:

```bash
diagnostic-suite \
  --data-root data \
  --tasks toy2d \
  --models fno unet resnet \
  --smoke \
  --device cpu \
  --outdir outputs_toy_smoke
```

The final metrics table is written to:

```text
outputs_toy_smoke/metrics_summary.csv
```

## Benchmark data layout

The package uses the following default relative paths under `--data-root`:

```text
apebench/apebench_incompressible_ns_2d/data/*train.npy
apebench/apebench_incompressible_ns_2d/data/*test.npy
pdebench/pdebench_shallow_water_2d/*.h5
pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5
the_well/datasets/active_matter/data/{train,valid,test}/*.hdf5
the_well/datasets/MHD_64/data/{train,valid,test}/*.hdf5
```

Inspect detected files and HDF5 keys before running a full benchmark:

```bash
diagnostic-suite \
  --data-root /path/to/data/raw \
  --tasks ns2d shallow2d active2d cns3d mhd3d \
  --inspect-data
```

## Reproducing benchmark tables

A practical multi-dataset run:

```bash
diagnostic-suite \
  --data-root /path/to/data/raw \
  --tasks ns2d shallow2d active2d cns3d mhd3d \
  --models fno deeponet unet resnet \
  --epochs-well 25 \
  --epochs-underfit 2 \
  --max-train-pairs 4096 \
  --max-val-pairs 1024 \
  --max-test-windows 512 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --outdir outputs_diagnostic_suite
```

By default, DeepONet is skipped on 3D datasets to avoid excessive memory use. Use `--deeponet-3d` only with a small 3D spatial limit, for example:

```bash
diagnostic-suite \
  --data-root /path/to/data/raw \
  --tasks cns3d mhd3d \
  --models deeponet \
  --spatial-limit-3d 16 \
  --deeponet-3d \
  --outdir outputs_deeponet_3d
```

## Outputs

Important output files:

- `metrics_summary.csv`: final metrics table;
- `metrics_summary_partial.csv`: incremental table written after each task;
- `data_inspection.csv`: available when `--inspect-data` is used;
- `checkpoints/<task>/<model>_<variant>.pt`: saved model checkpoints;
- `checkpoints/<task>/<model>_<variant>_history.csv`: training histories;
- `normalizers/<task>.pt`: training-set normalization statistics.

## Interpretation notes

The diagnostics are numerical tests relative to the available reference discretization. They are not formal correctness certificates.

Several diagnostics require equation-specific information:

- energy diagnostics require a specified conserved or dissipated functional;
- integral-balance diagnostics require identifiable conserved variables, fluxes, or sources;
- constraint diagnostics require a known admissible state set;
- perturbation-response diagnostics require paired reference trajectories under controlled perturbations;
- scaling-law diagnostics require a valid scaling relation.

When this information is unavailable, the corresponding diagnostic is marked unavailable rather than reported as a misleading numerical value.

Integral balance is implemented as a global conserved-channel drift diagnostic. Full finite-volume flux/source balance requires explicit fluxes, source terms, or solver access.

## Citation

Please cite the accompanying manuscript using the following placeholder BibTeX entry until formal publication metadata is available:

```bibtex
@misc{shikhman2026diagnosticsuite,
  author = {Shikhman, Lennon J.},
  title = {A Diagnostic Software Suite for Auditing Learned PDE Simulators},
  year = {2026},
  note = {Manuscript},
  url = {https://github.com/lennonshikhman/diagnostics_for_physics}
}
```

If you use the software directly, please also cite the repository release through the metadata in `CITATION.cff`.

## License

MIT. See `LICENSE`.
