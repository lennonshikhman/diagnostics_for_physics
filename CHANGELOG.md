# Changelog

## 0.3.3

- Name the software PDECheck and use `pdecheck` as the distribution name and
  primary command. Preserve `diagnostic_suite` imports and both existing main
  command aliases; verify all three installed commands in regression tests.
- Add a GitHub trusted-publishing workflow and PyPI account setup instructions.
- Update the manuscript title, software metadata, and citation metadata to use
  PDECheck. Diagnostic kernels and experimental results are unchanged.

## 0.3.2

- Copy ranked arrays before centering so Spearman analysis works with pandas
  copy-on-write exports; add a read-only-array regression. Numerical results
  are unchanged.
- Tighten generated manuscript prose while retaining the reported numerical
  results, failed matching cases, and supplementary rollout outcomes.

## 0.3.1

- Require an explicit periodic boundary policy when constructing a smoothing
  wrapper; reject omitted and nonboolean policies.
- Add a checkpoint-only composed-state and out-of-horizon rollout audit with
  paired crossed uncertainty, unique-checkpoint rank associations, and retained
  selection aliases and failed matching cases. No retraining is required.
- Report external amplitude-scaling diagnostics and verify smoothed CNS state,
  energy components, and subvolume mass balance against reference baselines.
- Clarify all-seed pooling, reference-energy interpretation, and the limits of
  temporal self-consistency as evidence for forecasting accuracy.

## 0.3.0

- Added exact heat-generator horizon/frequency/grid/precision sensitivity and
  independently known generator comparisons.
- Added paired perturbation/scaling verification, including passing defective
  maps that show the limits of individual diagnostics.
- Added reproducible heat/advection/Burgers FNO training with common warmup,
  supervised, PINO-style and composition continuations, validation-only matching,
  per-initial-state samples and crossed uncertainty analysis.
- Added fixed-parameter file selection and checked caching for external data,
  retaining native metadata, split identities and source/data hashes.
- Matched standard/short-budget initialization, width and minibatch streams.
- Specified and tested two-pass box smoothing with equation-aware boundaries.
- Added CPU/CUDA audit-cost measurement, architecture/API documentation,
  standalone scientific figures, and exact revision reproduction commands.

## 0.2.0 (revision candidate)

- Corrected the APEBench `diff_adv` task to scalar advection (`adv2d`); reject
  the old `ns2d` name instead of assigning it fluid diagnostics.
- Preserve float64 API states, reject nonfinite states and applicable metric
  failures, validate physical channels and callback fields, and report counts
  and execution status for each horizon.
- Fixed integer-step autoregressive composition and documented when it holds
  by construction. Stored reference endpoints no longer certify composition.
- Corrected cell-face balance quadrature, including one-dimensional boundaries;
  added manufactured balance and refinement regressions.
- Preserve uniform periodic spacing when subsampling benchmark grids; reject
  reductions that cannot retain a nontrivial uniform periodic grid.
- Make resume verify actual installed source, full data hashes, task metadata,
  and split membership before reusing results; preserve original run evidence
  when a resumed invocation fails.
- Require schema-2 benchmark exports with valid coverage. Do not rank a smaller
  generator ratio as inherently better, or fabricate disagreement examples.
- Added a self-contained analytic PDE demonstration with generated CSV, JSON,
  LaTeX, provenance, and 16 automated checks; test built wheels outside checkout.
- Exclude the private manuscript directory from Git and distribution archives.

- Made all shared-file train/validation/test partitions deterministic and disjoint,
  with a pre-training leakage audit and persisted trajectory identities.
- Added schema-aware The Well field assembly and runtime channel/grid/time metadata.
- Replaced inferred velocity-divergence constraints with explicitly named,
  equation-specific constraints.
- Added physical grid spacing to derivatives; corrected compressible-flow and MHD
  energy definitions and separated energy-law from reference-energy diagnostics.
- Implemented discrete control-volume flux/source balance and retained global
  conserved-quantity drift as a separately named proxy.
- Added perturbation-response, configurable scaling, callable/autoregressive/saved
  adapters, applicability reasons, custom PDE callbacks, and result metadata.
- Added per-horizon, multi-seed evaluation; fail-fast run semantics; provenance and
  data manifests; analytic self-checks; expanded tests; and continuous integration.
- Removed ambiguous benchmark aliases that labeled every reference-energy error
  as energy decay, and disabled heterogeneous composite constraint scores.

## 0.1.0

- Initial package layout.
- Added public `DiagnosticPanel` API.
- Added CLI entry points for benchmark experiments and toy data generation.
- Added tests, citation metadata, license, and packaging metadata.
