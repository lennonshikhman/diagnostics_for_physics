# Benchmark data contracts

The package does not redistribute third-party benchmark data. Files are located
relative to `--data-root`, and each run records the concrete matches, sizes,
trajectory indices, optional SHA-256 checksums, channel schema, spatial spacing,
and time step.

| Task | Reader | Required state channels | Split contract | Physics diagnostics |
|---|---|---|---|---|
| Toy2D | NPY | scalar | separate train/validation/test files | dissipative generic energy, global scalar drift, linear-amplitude scaling |
| ADV2D | NPY | scalar | disjoint 90/10 train/validation indices plus external test file | conservative scalar squared norm, global scalar drift, linear-amplitude scaling |
| SW2D | PDEBench HDF5 | height | deterministic 80/10/10 trajectory indices | height positivity and global mass drift; velocity-dependent energy and flux balance unavailable |
| Active2D | The Well HDF5 | all format-declared fields | official train/valid/test directories | reference generic energy only unless extended by the user |
| CNS3D | PDEBench HDF5 | density, pressure, velocity x/y/z | deterministic 80/10/10 trajectory indices | density/pressure positivity, total energy, mass balance |
| MHD3D | The Well HDF5 | density, velocity x/y/z, magnetic field x/y/z | official train/valid/test directories | density positivity, magnetic divergence, mass balance, reference kinetic-plus-magnetic energy |

The public PDEBench radial-dam-break file stores only water height. The package
therefore does not fabricate shallow-water velocity channels or report total
shallow-water energy and flux-balance diagnostics for this file.

For CNS3D, the stored state is interpreted as primitive variables. Total energy
is `rho*|u|^2/2 + p/(gamma-1)` with configured `gamma=1.4`. For MHD3D, the
configured public state has no pressure/internal-energy channel, so
`rho*|u|^2/2 + |B|^2/2` is compared with the reference and is not asserted to be
the conserved total energy.

The Well reader uses `t0_fields`, `t1_fields`, and `t2_fields` declarations and
their `sample_varying` and `time_varying` attributes. MHD requires the exact
seven-channel schema above. Missing fields, mismatched field sizes, ambiguous
channels, empty partitions, or inconsistent split metadata are fatal errors.

Dataset release identifiers are external provenance and cannot be inferred from
array contents. Record them alongside the generated SHA-256 manifest when
archiving a reproduction package.

## Source identification and acquisition

Obtain data from the originating projects: [APEBench](https://tum-pbs.github.io/apebench/),
[PDEBench](https://github.com/pdebench/PDEBench), and
[The Well](https://github.com/PolymathicAI/the_well). Follow their acquisition and
licensing instructions, and arrange the selected files under the relative paths
in the README. The package does not download data implicitly.

The APEBench recipe matches `2d_diff_adv__num_spatial_dims=2_{train,test}.npy`.
Upstream documents `diff_adv` as scalar advection, distinct from advection–diffusion
and incompressible flow. The legacy parent directory contains `ns` in its name;
that label must not be used as evidence of Navier–Stokes dynamics. The recipe
uses a unit periodic domain, treats time in stored-step units because physical
time spacing is unavailable in these NPY files, and does not configure a flux
without the generating advection coefficients. Record the generating configuration
and upstream revision when supplying a different export. Old `ns2d` result tables
are not compatible with the corrected `adv2d` task.

Resume always verifies full input hashes, including when `--hash-data` is omitted.
NPY readers memory-map the source file and copy only requested states before
preprocessing, avoiding a full-array float32 allocation.

Spatial limits are upper bounds, not promised output sizes. Periodic subsampling
uses a stride that divides the native size so the gap across the periodic seam
matches interior spacing. For example, a 160-point axis limited to 64 retains
40 points with stride four. If no nontrivial uniform subsample exists, use limit
zero to retain the native grid or supply a separately resampled dataset. This
operation is point subsampling, not a claim of conservative or alias-free
coarse graining.
