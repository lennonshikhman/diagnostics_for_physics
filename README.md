# PDECheck

Python package for diagnostic consistency testing of learned PDE time propagators. Version 0.3.3 accompanies the revised Computer Physics Communications manuscript.

`diagnostic_suite` evaluates learned PDE simulators as approximate **time-propagation operators**, not only as predictors of discretized state fields. The package provides a lightweight public API, a command-line interface, toy dataset generation, benchmark readers, neural surrogate baselines, diagnostic metrics, tests, and reproducibility scripts.

The software is designed for post hoc auditing of learned PDE simulators, including neural operators, convolutional surrogates, autoregressive emulators, custom callable propagators, and saved prediction arrays.

Implemented diagnostics include:

- relative `L2` state error;
- semigroup consistency;
- finite-difference generator discrepancy;
- reference-normalized generator ratio;
- conservative/dissipative energy-law violation and reference-energy error;
- discrete control-volume balance with conserved quantity, boundary flux, source,
  physical grid spacing, and elapsed time;
- a separately named global conserved-quantity drift proxy;
- individually named equation-specific constraints (for example,
  `height_positivity`, `density_positivity`, and `magnetic_divergence`);
- perturbation-response error from paired base/perturbed references and predictions;
- configurable scaling-law commutation error.

Unavailable diagnostics are reported as `NaN`, not zero. This is intentional: the package distinguishes between a diagnostic that is meaningfully satisfied and one that is mathematically or physically inapplicable for the available data.
Every result also carries an availability flag and a human-readable reason.
Nonfinite input states and failed applicable metrics raise an error; they cannot
disappear from an average. Float64 inputs retain their precision.

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
│   ├── propagators.py      # Callable, autoregressive, saved-output adapters
│   ├── runner.py           # Experiment orchestration
│   ├── self_check.py       # Analytic positive/negative diagnostic checks
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

The distribution name is `pdecheck`. Install a published release with:

```bash
python -m pip install pdecheck
pdecheck --self-check
```

Python imports continue to use `diagnostic_suite`. The `diagnostic-suite` and
`diagnostics-for-physics` commands remain available. See
[publishing instructions](docs/publishing.md) for release configuration.

Requires Python 3.10 or newer, NumPy 1.23.5+, pandas 1.5.3+, h5py 3.8+,
and PyTorch 2.0.1+. A GPU is optional. From the repository root:

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

Run the solver- and dataset-independent scientific checks:

```bash
diagnostic-suite --self-check
```

Run the reproducible PDE demonstration used for the revised manuscript:

```bash
OMP_NUM_THREADS=1 diagnostic-suite-referee --outdir outputs_referee
```

This audits five propagators for periodic advection–diffusion against independent
closed-form references, using float64 on CPU. It writes a 15-row metrics CSV,
per-horizon JSON reports, 16 pass/fail checks, a provenance manifest, and a LaTeX
table. No external dataset, pretrained weights, or GPU is needed. These checks
validate the diagnostic implementation; the separate training smoke run below
checks neural-model integration. The controlled learned-model study and external
application are separate experiments described below. See [the user guide](docs/user_guide.md).

For a release installation, build with `python -m build`, inspect with
`python scripts/check_distributions.py dist`, and install the generated wheel.
CI runs tests and examples from outside the checkout using that wheel. Both
distributions exclude the private manuscript directory.

On resource-constrained machines, use:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest
```

## Lightweight API usage

Use `DiagnosticPanel` when you already have predicted and reference states from your own surrogate. States should be shaped as `(batch, channels, *spatial)` or `(channels, *spatial)`.

```python
import numpy as np

from diagnostic_suite import ConstraintSpec, DiagnosticConfig, DiagnosticPanel

u0 = np.random.randn(1, 2, 32, 32).astype("float32")
uref = u0 + 0.05 * np.random.randn(1, 2, 32, 32).astype("float32")
pred = u0 + 0.07 * np.random.randn(1, 2, 32, 32).astype("float32")

panel = DiagnosticPanel(
    DiagnosticConfig(
        dim=2,
        energy_type="kinetic",
        energy_law="reference",
        velocity_channels=(0, 1),
        constraints=(
            ConstraintSpec("velocity_divergence", "divergence", (0, 1)),
        ),
        grid_spacing=(1.0 / 32.0, 1.0 / 32.0),
        metadata={"model": "my_surrogate", "checkpoint": "run-17"},
    )
)

result = panel.evaluate_pair(u0, uref, pred, delta_t=0.1)
print(result.as_report())
```

The API exposes direct tensor methods for semigroup, generator, perturbation,
and scaling diagnostics. `CallablePropagator`, `AutoregressivePropagator`, and
`SavedPredictionAdapter` support multi-horizon audits. Saved endpoints do not
claim semigroup applicability unless actual composed endpoints are also supplied.
Custom PDEs can provide energy, constraint, conserved-quantity, flux, source,
and scaling callbacks without editing package source. See
[`docs/user_guide.md`](docs/user_guide.md).

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
apebench/apebench_incompressible_ns_2d/data/2d_diff_adv__num_spatial_dims=2_train.npy
apebench/apebench_incompressible_ns_2d/data/2d_diff_adv__num_spatial_dims=2_test.npy
pdebench/pdebench_shallow_water_2d/*.h5
pdebench/pdebench_compressible_ns_3d/3D_CFD_Rand_M0.1*_Train.hdf5
the_well/datasets/active_matter/data/{train,valid,test}/*.hdf5
the_well/datasets/MHD_64/data/{train,valid,test}/*.hdf5
```

`adv2d` is **scalar linear advection**, matching APEBench's `diff_adv` scenario.
The directory name `apebench_incompressible_ns_2d` is a legacy local path, not an
equation identifier. Version 0.2 rejects the old `ns2d` task name. The configured
state is a scalar; no fluid velocity or kinetic energy is inferred. See the
[upstream scenario documentation](https://tum-pbs.github.io/apebench/extending_apebench/)
and [data contracts](docs/benchmark_data.md).

The split policy is explicit and is audited before training:

| Task | Policy |
|---|---|
| ADV2D | Deterministic seeded 90/10 trajectory partition for training/validation; external test file |
| SW2D | Deterministic 80/10/10 trajectory partition of the shared height-only file |
| CNS3D | Deterministic 80/10/10 trajectory partition of the shared training file |
| Active2D/MHD3D | Official, physically separate train/validation/test directories |

Any trajectory identity shared across splits causes a nonzero run failure before
normalization or training. The Well reader assembles all configured scalar and
vector fields and verifies the resulting channel names rather than selecting a
single numeric dataset heuristically.

Inspect detected files and HDF5 keys before running a full benchmark:

```bash
diagnostic-suite \
  --data-root /path/to/data/raw \
  --tasks adv2d shallow2d active2d cns3d mhd3d \
  --inspect-data
```

## Reproducing the October revision

The revision adds three distinct evidence levels:

- Analytic/manufactured verification: heat-generator sensitivity to horizon,
  frequency, grid and precision; paired perturbation and exact scaling; flux
  quadrature and explicitly normalized constraints.
- Controlled trained models: heat, advection and viscous Burgers, five seeds,
  common FNO architecture and warmup, supervised/PINO-style/composition
  continuations, validation-only selection and accuracy matching. Matching
  failures are retained. Crossed bootstrap resamples seeds and independent test
  initial-state identities. A separate post hoc clock construction has identical
  direct endpoints and different compositions; it is not an independent training
  comparison.
- External application: five configured PDE families, three seeds, compact
  time-conditioned baselines, standard/short/smoothed variants, and three
  retained horizons. These bounded coarse-grid runs are not a leaderboard.

Exact commands and frozen choices are in
[the reproduction script](scripts/reproduce_revision.sh),
[the protocol](configs/revision_protocol.json), and
[the revision guide](docs/revision_reproduction.md).
The source distribution includes those files. Third-party raw data are acquired
separately and are not redistributed.

~~~bash
diagnostic-suite-sensitivity --outdir outputs_sensitivity
python scripts/verify_balance_and_constraints.py --outdir outputs_balance
bash scripts/reproduce_revision.sh controlled
REVISION_RAW_ROOT=/path/to/data/raw bash scripts/reproduce_revision.sh external
~~~

## Running external benchmarks

The commands below illustrate the general benchmark workflow. The exact revision
uses fixed-parameter Well file selections and the budgets in the reproduction
script. Its numerical records are regenerated with corrected task identities,
splits, metric definitions and report schema; historical tables are not reused.

A practical multi-dataset run:

```bash
diagnostic-suite \
  --data-root /path/to/data/raw \
  --tasks adv2d shallow2d active2d cns3d mhd3d \
  --models fno deeponet unet resnet \
  --seeds 0 1 2 \
  --eval-horizons 1 2 4 \
  --epochs-well 25 \
  --epochs-underfit 2 \
  --max-train-pairs 4096 \
  --max-val-pairs 1024 \
  --max-test-windows 512 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --hash-data \
  --resume \
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
- `metrics_across_seeds.csv`: mean, standard deviation, and count across seeds;
- `paper_tables/table_3_*.{csv,tex}` through
  `paper_tables/table_11_*.{csv,tex}`: benchmark exports retaining legacy filenames;
- `metrics_summary_partial.csv`: incremental table written after each task;
- `progress/<task>/seed_<seed>/*.json`: atomic, source/configuration-signed
  progress records used by `--resume`;
- `data_inspection.csv`: available when `--inspect-data` is used;
- `data_contracts/<task>/seed_<seed>.csv`: exact files, trajectory indices,
  channel schema, spacing, and time step used by each split;
- `data_contracts/<task>/seed_<seed>_files.csv`: sizes and optional SHA-256 hashes;
- `checkpoints/<task>/seed_<seed>/<model>_<variant>.pt`: checkpoints;
- `checkpoints/<task>/seed_<seed>/<model>_<variant>_history.csv`: histories;
- `normalizers/<task>/seed_<seed>.pt`: training-only normalization statistics;
- `args.json` and `provenance.json`: configuration, software, hardware, and Git revision;
- `errors.json`: emitted when any task fails; the command exits nonzero.

`--resume` verifies the installed package source, numerical dependency versions,
full task definition, exact split membership, and input SHA-256 hashes before
reusing progress. It hashes inputs even without `--hash-data`, which can cost
time for large files. A mismatch aborts and preserves the original evidence.
Every invocation has a separate record under `invocations/`. An existing run
directory requires `--resume`; use a fresh directory for a changed experiment.

Rebuild the result tables without editing the manuscript:

```bash
diagnostic-suite-tables \
  --metrics outputs_diagnostic_suite/metrics_summary.csv \
  --outdir outputs_diagnostic_suite/paper_tables \
  --horizon 2
```

The table exporter requires schema-2 metrics and rejects failed values, duplicate
rows, and incomplete seed/model/variant coverage. Keep `args.json` beside the
metrics to check coverage against the requested experiment. Without that file,
completely absent models or seeds cannot be inferred from the CSV alone.
Generator ratios are interpreted relative to one and are excluded from monotone
deterioration rankings. No disagreement is reported when no comparison qualifies.

For a concise status readout during a resumable campaign:

```bash
diagnostic-suite-progress \
  --outdir outputs_diagnostic_suite \
  --unit diagnostics-corrected-tables
```

## Interpretation notes

The diagnostics are numerical tests relative to the available reference discretization. They are not formal correctness certificates.

Several diagnostics require equation-specific information:

- energy diagnostics require a specified conserved or dissipated functional;
- integral-balance diagnostics require identifiable conserved variables, fluxes, or sources;
- constraint diagnostics require a known admissible state set;
- perturbation-response diagnostics require paired reference trajectories under controlled perturbations;
- scaling-law diagnostics require a valid scaling relation.

When this information is unavailable, the corresponding diagnostic is marked unavailable rather than reported as a misleading numerical value.

Control-volume balance and global drift are intentionally different columns.
`integral_balance` is available only when a conserved quantity plus flux/source
contract is configured. `global_conserved_drift` is the weaker whole-domain
proxy. Velocity divergence is never inferred from the presence of velocity
channels: it is configured only for equations that require incompressibility.

The benchmark files do not contain paired perturbed trajectories, so their
`perturbation_response` column remains unavailable with an explicit reason. The
implemented public API computes it when the required four states are supplied.
MHD energy is reference-compared because the configured public MHD state omits a
pressure/internal-energy channel; the package does not label that partial energy
as a conserved total energy.

## Citation

Please cite the accompanying manuscript using the following placeholder BibTeX entry until formal publication metadata is available:

```bibtex
@misc{shikhman2026diagnosticsuite,
  author = {Shikhman, Lennon J.},
  title = {PDECheck: A Diagnostic Software Suite for Auditing Learned PDE Simulators},
  year = {2026},
  note = {Manuscript},
  url = {https://github.com/lennonshikhman/diagnostics_for_physics}
}
```

If you use the software directly, please also cite the repository release through the metadata in `CITATION.cff`.

## License

MIT. See `LICENSE`.
