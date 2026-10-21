# Reproducing the CPC revision

Install the software source distribution or wheel and use the scripts/configs
from the source archive. Run shell scripts from its root. Python 3.10 or newer
is required; plotting additionally needs Matplotlib. Tests require pytest,
release building requires build, and static checks use the pinned Ruff version.
Install the optional revision dependencies with pip install '.[revision]' to run
the figure builders and PDF checks.

scripts/reproduce_revision.sh separates analytic checks, the controlled study,
the external application, and overhead. Its default output root is
outputs_revision_reproduced, so the archived evidence is not overwritten.
Set REVISION_OUT_ROOT to change it and REVISION_DEVICE=cpu for CPU training.
GPU training uses much less wall time for the full experiment.

The analytic checks require no external data. The controlled study generates
disjoint coefficient identities and exact reference states itself. It uses
three equations, five seeds, one shared warmup and three continuations per
equation/seed. Pilots used another dataset seed and validation-only outputs;
they are excluded from the reported test results. Matching uses validation
accuracy only, and unsuccessful matching remains in the exported report.
The later identical-endpoint clock construction is explicitly post hoc.

External data must be acquired from APEBench, PDEBench and The Well under their
original terms. Consult docs/benchmark_data.md and the staged file manifest.
scripts/stage_revision_data.py verifies source/destination SHA-256 while
copying from a read-only archive. The fixed Well parameter selection is in
configs/revision_file_selection.json; do not silently expand it to multiple
unobserved physical parameter settings. Grid limits, windows and budgets in the
script are the reported bounds, not native-resolution or full-dataset claims.
Some exports lack historical generator/release metadata; hashes identify the
actual input bytes without supplying missing provenance.

All external baselines are time-conditioned direct predictors. Integer-step
composition uses restarted predictions at smaller requested horizons. Repeated
powers of a fixed-step map have zero composition defect by construction.
External temporal windows can overlap and are not independent trajectories;
their spread is not converted into independent-sample confidence intervals.

The recorded GPU runtime inherited an OpenFOAM LD_LIBRARY_PATH that selected
an old system cuBLASLt alongside the wheel's newer cuBLAS. Dense GEMM failed
before affected DeepONet training could finish. The failed invocation log is
retained, and the exact same scientific configuration was resumed using
env -u LD_LIBRARY_PATH, which selects the wheel's matching CUDA libraries.
An isolated GEMM/DeepONet forward/backward check verified this resolution.
The reproduction script applies this per-command environment isolation.

Checkpoints retain the training source hash. The evidence bundle preserves
that exact source snapshot separately from the final release hash; release
validation and any later edge-case API repairs are identified separately.
Resume rejects changed source, data, tasks, numerical dependencies or splits.
Use a fresh output directory when intentionally changing the implementation.

Scientific figure/table builders read archived CSVs; they do not hand-enter
numbers. The manuscript source archive supplies these generated inputs and
vector figures, so LaTeX compilation does not require model training. Its marked
source highlights changed paragraphs/environments in blue, RGB 0,70,170,
against the archived reviewed manuscript. Clean and marked text are checked
for equivalence after compilation.

The supplementary version 0.3.1 analysis reads the original checkpoints without
retraining. Extract the controlled and external training artifacts, then run:

    python scripts/analyze_rollout_revision.py --root /path/to/controlled --device cuda
    python scripts/audit_external_physics_revision.py --root /path/to/external --data-root /path/to/raw --device cuda
    python scripts/analyze_external_revision.py --root /path/to/external

The rollout audit retains all 75 selection labels, including identical-weight
aliases and unsuccessful accuracy matches. Rank associations count identical
weights once within an equation and training seed. Every network query is at
T or T/2, including rollouts to 2T and 4T; no out-of-range time input is used.
The direct error baseline is at T, not a single query at 2T or 4T. Paired crossed
uncertainty resamples training seeds and shared test initial conditions.

The physical audit reconstructs the original horizon-two test windows and emits
per-window CNS smoothing/energy records and genuine CNS/MHD octant balances,
including reference baselines. Added physical kernels use float64 on float32
predictions. The primary external tables retain their original audit records.
No third-party raw files are included in the software release.
