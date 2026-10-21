**CPiP software review — 12 September 2026**

The main revision priorities are reliable diagnostic failure reporting, correct dataset identity, and agreement between the archived program and the manuscript. The package already has useful foundations: a public API, explicit physics metadata, a user guide, deterministic split checks, a synthetic dataset, analytic self-checks, and reproducibility outputs.

This review covers the current working copy, including its pre-existing modified and untracked files, and the local 33-page manuscript `A_Diagnostic_Software_Suite_for_Auditing_Learned_PDE_Simulators (2).pdf`. It does not establish that these are identical to the files submitted to CPC. No implementation, tests, or manuscript files were changed during this review.

The local Git HEAD was `eb6bfb7`. The reviewed source fingerprint, computed over package Python files and `pyproject.toml` using the runner's source-checkout algorithm, was `7c95c15e822c9f10494b93d97da8bbd108f989a542cb95d674bba39e28fdc4ed`.

**Verification performed**

| Check | Result and boundary |
|---|---|
| Existing tests against the source copy | 46 passed in 2.25 seconds |
| Source distribution and wheel build | Both built successfully in build isolation |
| Existing tests against the installed wheel | 46 passed in 2.26 seconds |
| Installed CLI scientific self-check | Passed |
| Both shipped Python examples | Ran successfully against the installed package |
| CPU toy benchmark | FNO, DeepONet, U-Net, and ResNet; eight training jobs, derived oversmoothed variants, horizons 1/2/4; 39 result rows and eight checkpoints |
| Ruff lint | Passed with installed Ruff 0.16.6 |
| Ruff formatting check | Failed for `src/diagnostic_suite/tasks.py` and `tests/test_data_contracts.py` |
| Full five-dataset benchmark | Not rerun; third-party dataset contents and the original numerical table artifacts were not verified |

The package tests used Python 3.10.19, NumPy 1.24.4, pandas 2.3.3, h5py 3.14.0, and PyTorch 2.11.0.dev20260106+cu128 on CPU. The temporary wheel environment inherited existing dependencies through `--system-site-packages`; this was not a clean resolution of the minimum supported dependencies. No GPU benchmark campaign was launched.

Builds, logs, rendered manuscript pages, and toy artifacts are in `/tmp/diagnostic-suite-cpc-review-33zd2blf/`. That directory is temporary; the evidence and reproductions below are the durable record.

**1. High priority: correct the identity of the NS2D benchmark.**

[Task configuration](../src/diagnostic_suite/tasks.py#L255) labels the task incompressible Navier–Stokes, assigns a vorticity channel, and computes kinetic energy from that vorticity. Its selected files are `2d_diff_adv__num_spatial_dims=2_train.npy` and the corresponding test file.

APEBench's own registry maps `diff_adv` to `Advection`, and the corresponding class generates the scenario name `2d_diff_adv`. The `diff` prefix refers to the difficulty interface; it does not establish Navier–Stokes dynamics. Sources: [APEBench scenario registry](https://raw.githubusercontent.com/tum-pbs/apebench/main/apebench/scenarios/difficulty/__init__.py), [Advection implementation](https://raw.githubusercontent.com/tum-pbs/apebench/main/apebench/scenarios/difficulty/_linear.py).

Verify the generating recipe and manifest of the exact submitted files. If they are the standard APEBench advection release, rename the task and choose appropriate scalar-advection diagnostics, or replace the data with an explicitly identified Navier–Stokes dataset and rerun affected results. Update the abstract, Section 6.1.1, task documentation, and tables together. A directory name containing `incompressible_ns` is insufficient provenance. The default configuration mismatch is verified; the contents of the historical benchmark files were not inspected in this review.

**2. High priority: distinguish an unavailable diagnostic from a failed numerical evaluation.**

[Public result aggregation](../src/diagnostic_suite/api.py#L117) and [benchmark aggregation](../src/diagnostic_suite/metrics.py#L518) silently remove every nonfinite sample. In a two-sample audit with one correct prediction and one entirely NaN prediction, the public report returned `rel_l2 = 0.0`, `sample_counts = 1`, `available = True`, and an empty reason. A reader inspecting the mean and availability would miss the failed prediction.

A configured positivity constraint on nonexistent channel 9 also returned `available = True` with no finite samples or explanatory reason. [Applicability assignment](../src/diagnostic_suite/api.py#L240) checks whether a constraint was configured rather than whether the input fulfills its contract.

Add separate applicability and execution status, total/valid/invalid sample counts, and a documented policy for NaN/Inf input or callback output. The default should raise or explicitly mark the audit failed when an applicable diagnostic cannot be evaluated. Validate configured channel indices against actual input shapes. Keep intentionally unavailable metrics separate from failed metrics throughout API reports, CSVs, and tables.

Minimal reproductions:

```python
import numpy as np
from diagnostic_suite import ConstraintSpec, DiagnosticConfig, DiagnosticPanel

u = np.ones((2, 1, 4, 4), dtype=np.float32)
p = u.copy()
p[1] = np.nan
r = DiagnosticPanel(DiagnosticConfig(dim=2)).evaluate_pair(u, u, p)
print(r.means['rel_l2'], r.sample_counts['rel_l2'], r.applicability['rel_l2'])
# Current: 0.0, 1, available=True with an empty reason.

panel = DiagnosticPanel(DiagnosticConfig(
    dim=2, constraints=(ConstraintSpec('missing', 'positivity', (9,)),)
))
r = panel.evaluate_pair(u, u, u)
print(r.means['missing'], r.sample_counts['missing'], r.applicability['missing'])
# Current: nan, 0, available=True with an empty reason.
```

**3. High priority: fix installed-package provenance and resume validation.**

There are two independently verified problems in [the runner](../src/diagnostic_suite/runner.py#L104).

First, `_source_fingerprint()` assumes the package resides under a checkout's `src/diagnostic_suite`. After a normal wheel installation it finds no files and returns the SHA-256 of empty input, `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. Changes to installed package code therefore cannot be detected by the advertised source signature.

Second, [the resume signature](../src/diagnostic_suite/runner.py#L149) contains arguments and source identity but not the consumed dataset identity. [Completed tasks return before data inspection](../src/diagnostic_suite/runner.py#L365). In the isolated toy experiment, multiplying `test.npy` by ten and rerunning the identical command with `--resume --hash-data` exited successfully, reused the completed task, and produced byte-identical metrics. Its data manifest retained the old checksum. The scratch test file was restored afterward.

Hash the actual installed package resources and record the distribution version; treat an empty source set as an error. Validate input manifests, effective task configuration, and split identities before reusing cached results. Preserve the provenance of the computations being reused rather than replacing it with only the resume invocation's environment. Add installed-wheel and modified-input resume regression cases.

Installed-wheel reproduction, from outside a source checkout:

```python
import hashlib
from diagnostic_suite.runner import _source_fingerprint
print(_source_fingerprint() == hashlib.sha256(b'').hexdigest())
# Current installed wheel: True.
```

**4. High priority: preserve numerical precision in the public diagnostic API.**

[`_ensure_state`](../src/diagnostic_suite/api.py#L485) converts every NumPy or tensor input to float32. An all-ones float64 reference and a prediction offset by `1e-8` produce a public relative error of exactly zero; the direct float64 kernel returns approximately `9.9999999e-9`. This matters for numerical-solver audits, small perturbations, and short-time generator tests.

Preserve floating input precision or provide an explicit diagnostic dtype policy. Keep any model-specific float32 conversion in its adapter. Document epsilon and near-zero-reference behavior, and add tests spanning float32/float64, small perturbations, and small horizons.

```python
import numpy as np
from diagnostic_suite import DiagnosticConfig, DiagnosticPanel
u = np.ones((1, 1, 4, 4), dtype=np.float64)
print(DiagnosticPanel(DiagnosticConfig(dim=2)).evaluate_pair(u, u, u + 1e-8).means['rel_l2'])
# Current: 0.0.
```

**5. High priority: make the advertised autoregressive adapter work through the audit panel.**

[`evaluate_propagator`](../src/diagnostic_suite/api.py#L376) always attempts an equal half-horizon split. With `AutoregressivePropagator(step_size=1)` and valid requested horizons 1 and 2, it requests horizon 0.5 and raises `ValueError`. The existing test exercises the adapter directly, which misses this integration failure.

Expose valid time increments/composition through the adapter, use integer-step splits where meaningful, and report one-step composition as unavailable. For repeated application of the same deterministic one-step map, also explain that semigroup consistency on integer steps is satisfied by construction and provides limited independent evidence.

```python
import numpy as np
from diagnostic_suite import AutoregressivePropagator, DiagnosticConfig, DiagnosticPanel
u = np.ones((1, 1, 4, 4), dtype=np.float32)
prop = AutoregressivePropagator(lambda x: x + 1, step_size=1)
DiagnosticPanel(DiagnosticConfig(dim=2)).evaluate_propagator(
    u, {1.0: u + 1, 2.0: u + 2}, prop
)
# Current: ValueError about horizon 0.5.
```

**6. High priority: correct the direction used to select generator-disagreement examples.**

[Table 11 selection](../src/diagnostic_suite/tables.py#L260) treats an increase in every structural quantity as worsening. This includes `generator_ratio_to_reference`, although the manuscript explicitly explains that values below one are not automatically better.

A synthetic pair with all other metrics unchanged and generator ratio moving from 0.6 to 0.8 was classified as structural worsening by a factor of 1.333 and selected as a representative disagreement. The ratio actually moved closer to the reference value of one. Conversely, moving farther below one can be missed.

Exclude the raw generator ratio from a monotone error ranking, or explicitly define and justify a separate deviation-from-reference measure. Retain signed direction and the raw ratio in reports. Add examples approaching and departing from one on both sides. Inspect the automatically selected Table 11 cases again after this correction.

**7. Medium priority: repair 1D boundary-flux balance and strengthen its validation.**

The public configuration accepts one spatial dimension. In [the boundary integral](../src/diagnostic_suite/metrics.py#L424), selecting a 1D boundary leaves shape `(B,)`; calling `flatten(1)` then raises `IndexError`. This was reproduced with a nonperiodic `(1, 2, 4)` state, spacing 0.25, a quantity channel, and one velocity channel. A 1D subvolume follows the same failing branch.

Handle zero-dimensional faces explicitly and test 1D, 2D, and 3D control volumes. Add manufactured solutions with nonzero flux and source cancellation, plus mesh refinement, to establish quadrature accuracy. Document whether state samples are cell-centered or nodal and how face fluxes and volume integrals are approximated. Existing balance tests demonstrate some nonzero responses, but do not establish convergence of the flux quadrature.

**8. High priority for the revision: synchronize the manuscript, metric definitions, and archived output.**

The local PDF and current working tree have concrete differences:

- PDF Table 3 marks NS2D constraints and SW2D energy/balance as applicable. Current tasks configure no NS2D constraint and disable SW2D energy and flux balance for height-only input.
- PDF Table 6 reports numerical SW2D energy-decay values. The current configuration reports that energy diagnostic as unavailable.
- PDF Tables 8 and 9 report a single constraint column. The current code intentionally separates density/pressure positivity and magnetic divergence.
- PDF Appendix A.4 describes a predicted-versus-reference global-quantity mismatch. The current `global_conserved_drift` compares the initial and predicted states, while `integral_balance` is a separate flux/source residual.
- PDF Section 5.5 and the current balance kernel use different normalization denominators.
- PDF Section 5.8 normalizes scaling error by the prediction of the scaled initial state; the current kernel normalizes by the transformed prediction.
- `CITATION.cff` references a different manuscript title from the PDF and README.

Decide and document the intended numerical definitions, then create one pinned release, reproduce the affected artifacts, and generate the manuscript tables from those artifacts. Extend `docs/paper_mapping.md` to include equations, input requirements, normalization, unavailable cases, and a known-value example for each metric. Do not transfer old table numbers to revised kernels without re-evaluation. The exact submission package should be checked before concluding these local discrepancies also exist in the files under review.

**9. CPiP requirement: add the formal Program Summary.**

The local PDF has a software metadata table but no formal Program Summary after the abstract. The publisher's [CPiP template](https://legacyfileshare.elsevier.com/promis_misc/CPCtemplateTEupdate.tex) explicitly requires a Program Summary for CPiP articles, with program identification, licensing, language, nature of the problem, solution method, and restrictions/additional comments. Follow the template and leave editor-assigned archive fields to the technical editor.

The publisher describes CPiP software as being archived in the CPC Program Library on Mendeley Data: [official journal scope](https://shop.elsevier.com/journals/computer-physics-communications/0010-4655). Ensure the reviewed source, examples, data-access instructions, and the exact version used for tables are represented in that archive. The existing MIT license already supplies an explicit license; this review did not verify the status of any existing deposit.

**10. Medium priority: turn installation checks into release checks.**

The current CI tests an editable checkout and then builds distributions. Add execution of the installed wheel from outside the repository, including self-check, examples, and a tiny toy run. This directly catches the source-fingerprint problem. Run a supported stable dependency configuration as well as a current configuration; the permissive dependency minima are not validated by this review.

Fix the two formatting failures, ensure the currently untracked CI configuration is included in the intended release workflow, and record a tested environment for exact reproduction. Keep the portable installation test independent of external benchmark data.

**Useful additions after those repairs**

- Provide a compact referee demonstration with one command, expected outputs and tolerances, a known correct PDE evolution, and intentionally broken alternatives. Extend the existing analytic self-check and toy generator rather than requiring a fresh training campaign to demonstrate every diagnostic.
- Publish exact benchmark release identifiers, acquisition commands, generating configurations for NPY exports, file hashes, channel order, boundaries, grid spacing, and split policy. The existing glob paths and manifests are helpful but do not tell an external user how to obtain or regenerate each expected file.
- Preserve per-horizon and per-sample records in the public API. It currently flattens samples from different horizons into one mean, unlike the CLI's separate horizon rows. Record which horizon pairs define a generator comparison and which requested compositions were unavailable.
- Add a short runtime and peak-memory table for auditing saved predictions and a callable at representative 2D/3D sizes. The package records training/evaluation durations, but that alone does not describe the cost of using the diagnostic layer with an existing simulator.
- Make table export readable directly: concise quantity labels and formatted mean/variation/count fields, with detailed machine columns retained in CSV. Require the intended task/model/variant/seed coverage before labeling tables complete.
- Consider memory-mapped NPY access for large inputs. The current reader loads complete arrays, and shared train/validation files are loaded separately. Validate that any performance change preserves split identities and values.

The first revision should repair correctness and provenance, align the paper with the validated release, and package the referee demonstration. Improvements to richer reporting and performance can follow that stable release boundary.
