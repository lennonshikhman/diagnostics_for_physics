# User guide

## State and propagator contracts

Public diagnostics accept NumPy arrays or PyTorch tensors shaped
`(batch, channels, *spatial)` or `(channels, *spatial)`. Spatial dimensionality
must match `DiagnosticConfig.dim`. Values must be decoded physical variables;
normalization belongs inside the model adapter, not inside a physical metric.
The API preserves float32 and float64 input precision and rejects complex or
nonfinite states. Configured channel indices must exist in the actual state;
an out-of-range density, velocity, or constraint channel raises an error.

A callable propagator implements:

```python
prediction = propagator(state, horizon)
```

`CallablePropagator` wraps that signature. `AutoregressivePropagator` repeatedly
applies a one-step callable. `SavedPredictionAdapter` exposes stored endpoints;
because an endpoint array cannot be reapplied to an arbitrary midpoint, semigroup
consistency is marked unavailable unless `composed_predictions` is supplied.
Saved direct endpoints also cannot be silently reused for a perturbed or scaled
initial state; pass the corresponding arrays to `perturbation_response()` or
`scaling_error()` explicitly.

An autoregressive audit uses integer multiples of its positive `step_size`.
Composition is unavailable at one step. At two or more steps, the adapter splits
the requested integer rollout into two nonempty parts. Equality then follows
from using the same one-step map repeatedly and is reported as true by
construction; it is not independent evidence of physical accuracy. Stored
reference endpoints likewise cannot establish a reference composition error.

## Equation metadata

`DiagnosticConfig` makes physical assumptions explicit:

- `grid_spacing` has one physical spacing per spatial dimension;
- `time_step` records the stored reference spacing;
- `periodic` controls derivative and balance boundaries;
- `energy_type` and `energy_law` distinguish a functional from whether it should
  be conserved, dissipated, or only compared with the reference;
- physical channels are named by `density_channel`, `pressure_channel`,
  `height_channel`, `vorticity_channel`, `velocity_channels`, and
  `magnetic_channels`;
- every `ConstraintSpec` names exactly one required condition;
- `BalanceSpec` supplies a conserved quantity, flux, and optional source.

Velocity channels never imply zero divergence. Add a divergence constraint only
for a system whose physics requires it. For example, compressible Navier--Stokes
uses density and pressure positivity, while MHD uses density positivity and
magnetic-field divergence.

## Custom PDE callbacks

Callbacks receive decoded batched state tensors and the runtime `TaskSpec`.
Energy and custom-constraint callbacks return one scalar per batch item. A
conserved-quantity or source callback returns `(B, 1, *spatial)`. A flux callback
returns `(B, dim, *spatial)`.

```python
import torch

from diagnostic_suite import BalanceSpec, DiagnosticConfig, DiagnosticPanel


def quantity(state, task):
    return state[:, :1]


def flux(state, task):
    q = quantity(state, task)
    return q * state[:, 1 : 1 + task.dim]


def source(state, task):
    return torch.zeros_like(state[:, :1])


panel = DiagnosticPanel(
    DiagnosticConfig(
        dim=2,
        grid_spacing=(0.01, 0.01),
        time_step=0.001,
        balance=BalanceSpec(
            conserved_quantity=quantity,
            flux=flux,
            source=source,
        ),
    )
)
```

A scaling callback has signature `transform(state, scale)`. An optional
`scaling_time_factor(scale)` maps the base horizon used on the transformed right
hand side.

## Metric meanings

- `rel_l2`: state error relative to the stored reference state.
- `semigroup`: direct propagation versus actual two-stage composition.
- `generator`: disagreement between two finite-difference generator estimates.
- `energy_violation`: wrong conservative change or wrong sign of dissipative
  change, according to `energy_law`.
- `energy_reference_error`: predicted energy versus the reference trajectory's
  energy; this captures over-dissipation without calling it a law violation.
- `integral_balance`: normalized discrete control-volume residual including
  boundary flux and source.
- `global_conserved_drift`: weaker whole-domain quantity drift, kept separate.
- named constraint columns: one interpretable physical requirement per column.
- `constraint`: compatibility alias only when exactly one named constraint is
  configured. With multiple constraints it is unavailable, preventing unlike
  requirements from being collapsed into an uninterpretable composite.
- `perturbation_response`: model response versus reference response to the same
  controlled initial perturbation.
- `scaling_law`: failure of propagation to commute with a configured exact
  scaling transformation.

`evaluate_propagator` reports each requested horizon in `result.by_horizon` and
in `result.as_report()["by_horizon"]`, including time, dtype, counts, and reasons.
The generator uses the first requested positive horizon and each later horizon;
it is unavailable at the first horizon. The combined summary pools available
samples across horizons for compatibility; use the per-horizon reports for
scientific comparisons.

Applicable metrics must have finite values for every sample. A failed input,
callback, or numerical result raises an exception instead of disappearing from
the denominator of an average. Unavailable diagnostics contain `NaN` values and
an explicit reason, with execution status `unavailable`; successful diagnostics
have status `ok`. Reports preserve finite counts and total counts. Standard JSON
exports from the referee demonstration encode unavailable values as `null`.

The reference-normalized generator ratio is available only for a finite
reference discrepancy greater than epsilon. It is compared with one, not with
zero: a frozen, inaccurate map can have generator discrepancy zero. It is not
used as a monotonically increasing error in table disagreement rankings.

## Discrete balance convention

Balance callbacks return cell-centered values or uniform-grid cell averages.
The volume integral uses cell volumes; interior face fluxes average neighboring
cell centers. Periodic faces wrap; nonperiodic boundary faces use linear
extrapolation and need at least two cells. Face areas are the product of the
tangential spacings. Flux and source are integrated in time by the endpoint
trapezoid rule. For a volume with integrated quantities `M0`, `M1`, time-averaged
outward flux `F`, and time-averaged source `S`, the reported residual is
`abs(M1-M0+dt*F-dt*S)/(abs(M0)+abs(M1)+dt*abs(F)+dt*abs(S)+1e-8)`.
Whole-domain periodic flux cancels; subvolumes retain their boundary flux.
Exact nonlinear solutions can have a nonzero quadrature residual. Compare a
reference baseline and refinement behavior before interpreting that residual.

Derivative-based divergence uses centered differences, with one-sided
nonperiodic boundary derivatives. These derivative samples and the cell-based
balance quadrature have different boundary conventions; supply matching physical
spacing and cell-centered fields when using both in an audit.

## Benchmark safety and reproduction

Run a data audit before an expensive benchmark:

```bash
diagnostic-suite \
  --data-root /path/to/data \
  --tasks adv2d shallow2d active2d cns3d mhd3d \
  --inspect-data
```

The audit verifies channel schemas and trajectory-level split disjointness.
Training aborts before normalization if either check fails. Full runs should use
multiple seeds and explicit horizons:

```bash
diagnostic-suite \
  --data-root /path/to/data \
  --tasks adv2d shallow2d active2d cns3d mhd3d \
  --models fno deeponet unet resnet \
  --seeds 0 1 2 \
  --eval-horizons 1 2 4 \
  --hash-data \
  --outdir outputs_diagnostic_suite
```

The runner saves partial tables after every task/seed. It may continue to gather
independent failures, but exits nonzero if any task failed and writes
`errors.json`; a missing dataset can no longer appear to be a successful run.

## Installation check versus benchmark reproduction

`diagnostic-suite --self-check` runs analytic positive/negative cases without
external data or learned models. It validates metric wiring and adapters. It does
not reproduce benchmark evidence. Benchmark tables require the external datasets
and should be accompanied by `args.json`, `provenance.json`, split contracts,
data hashes, checkpoints, training histories, and both per-seed and across-seed
metric tables.
