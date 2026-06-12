# Diagnostics for Physics

Production-facing Python package for diagnostic consistency testing of learned PDE time propagators.

This repository reorganizes the original experiment harness for the manuscript **Diagnostic Metrics for Learned PDE Simulators Beyond Relative Error** into an installable package with a command-line interface, toy dataset generation, benchmark readers, neural surrogate baselines, diagnostic metrics, tests, and reproducibility scripts.

The package evaluates learned PDE simulators as approximate **time-propagation operators**, not only as predictors of discretized state fields. Implemented diagnostics include:

- relative `L2` state error;
- semigroup consistency;
- finite-difference generator discrepancy and reference-normalized generator ratio;
- energy-law violation and energy-decay error;
- global integral-balance error;
- admissibility/constraint error;
- perturbation-response placeholder, reported as `NaN` unless paired perturbed references are supplied;
- scaling-law error when an exact scaling relation is configured.

## Repository layout

```text
.
├── src/diagnostic_suite/
│   ├── api.py              # lightweight public DiagnosticPanel API
│   ├── cli.py              # command-line entry point
│   ├── data.py             # NPY/HDF5 trajectory readers and datasets
│   ├── evaluate.py         # diagnostic evaluation loops
│   ├── metrics.py          # post hoc diagnostic metrics
│   ├── models.py           # FNO, DeepONet, U-Net, ResNet surrogates
│   ├── normalization.py    # training-set normalization
│   ├── runner.py           # experiment orchestration
│   ├── tasks.py            # benchmark/task definitions
│   └── toy.py              # synthetic advection-diffusion data generator
├── tests/                  # unit and smoke tests
├── scripts/                # reproducibility helpers
├── docs/                   # CPC/software-facing notes
├── legacy/                 # original single-file experiment script
├── pyproject.toml
├── LICENSE
└── CITATION.cff
```

## Install

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

Use `DiagnosticPanel` when you already have predicted and reference states from your own surrogate. States should be shaped `(batch, channels, *spatial)` or `(channels, *spatial)`.

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

print(panel.evaluate_pair(u0, uref, pred).as_dict())
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

The final metrics table will be written to:

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

## Reproducing the manuscript-style benchmark table

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

By default, DeepONet is skipped on 3D datasets to avoid excessive memory use. Use `--deeponet-3d` only with a small 3D spatial limit, for example `--spatial-limit-3d 16 --deeponet-3d`.

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

Unavailable optional diagnostics are reported as `NaN`, not zero. In particular, perturbation-response error requires paired reference trajectories under controlled perturbations, and scaling-law error is only evaluated when a task explicitly configures an exact scaling relation.

Integral balance is implemented as a global conserved-channel drift diagnostic. Full finite-volume flux/source balance requires explicit fluxes, sources, or solver access.

## License

MIT. See `LICENSE`.
