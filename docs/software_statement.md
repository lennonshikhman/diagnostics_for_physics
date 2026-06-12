# Software statement

This repository provides the implementation used to evaluate diagnostic
consistency tests for learned PDE time propagators.

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
  - energy-decay error;
  - integral-balance error;
  - constraint/admissibility error;
  - scaling-law hook where an exact relation is configured.
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
