# CPC software-facing note

This package is intended to support a **Computational Physics Paper** submission rather than to claim full maturity as a CPC Program Library artifact.

The paper-facing contribution is the diagnostic consistency framework for learned PDE time propagators. The software-facing contribution is a reproducible implementation with:

- benchmark trajectory readers for APEBench, PDEBench, and The Well layouts;
- model wrappers for FNO, DeepONet, U-Net, and ResNet-style surrogates;
- post hoc diagnostic evaluators;
- reproducibility scripts and CSV output;
- a synthetic toy dataset so reviewers can run the package without external benchmark downloads.

Before a formal CPC software/program-library submission, add a versioned public Git repository, a Zenodo DOI, CI, and more comprehensive tests across supported benchmark data layouts.
