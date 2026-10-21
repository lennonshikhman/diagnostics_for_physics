# Software statement

This repository provides diagnostic consistency tests for learned PDE time
propagators. Version 0.3.2 accompanies the manuscript revision. Independent
analytic/manufactured checks, a five-seed controlled learned-model study, and
regenerated fixed-parameter external applications provide separate evidence.
Historical tables are not reassigned to changed implementations.

## Scope

The software is an implementation and reproducibility package for a diagnostic
framework. It is intended to support a computational physics manuscript and to
help users audit learned PDE simulators against structural diagnostics. It is
not presented as a solver library and does not replace the benchmark datasets or
reference numerical solvers.

## Implemented components

- Dataset discovery/readers for selected APEBench, PDEBench, and The Well files.
- Compact neural surrogate baselines: FNO, DeepONet-lite, U-Net, and ResNet.
- Post-hoc diagnostics:
  - relative L2 error;
  - semigroup consistency;
  - finite-difference generator discrepancy;
  - energy-law violation;
  - reference-energy error;
  - control-volume flux/source balance and separately reported global drift;
  - explicitly named equation-specific constraint/admissibility errors;
  - perturbation-response error from paired perturbed simulations;
  - configurable scaling-law error.
- Training, evaluation, checkpointing, and CSV summary generation.
- Lightweight public API for users with their own arrays/tensors.

## Reproducibility policy

Unavailable optional diagnostics are written as `NaN` instead of `0.0`. This is
important because a zero value would incorrectly imply the diagnostic was
computed and passed.

The generator discrepancy is a finite-time diagnostic. Reference trajectories
can have nonzero generator discrepancy at finite stored time spacing, so the
analysis reports reference-normalized generator ratios when comparing learned
models.

Shared benchmark files are partitioned by deterministic trajectory indices, and
the runner proves train/validation/test disjointness before fitting a normalizer
or model. Channel names, grid spacing, time step, split identities, run arguments,
Git revision, dependency versions, and optional dataset hashes are persisted with
each result package.

Resume always checks full data hashes before reuse, as well as installed source,
task metadata, split membership, and numerical dependencies. A failed applicable
metric aborts the audit. Only explicitly unavailable diagnostics may contain
missing values. Float64 API inputs are preserved.
