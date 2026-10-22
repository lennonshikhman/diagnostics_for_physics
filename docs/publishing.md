# Publishing PDECheck

The PyPI distribution is `pdecheck`; Python imports remain `diagnostic_suite`.
The repository remains `lennonshikhman/diagnostics_for_physics`.

## One-time PyPI account setup

Sign in to <https://pypi.org/manage/account/publishing/> and add a pending
publisher with these exact values:

| Field | Value |
| --- | --- |
| PyPI project name | `pdecheck` |
| GitHub owner | `lennonshikhman` |
| Repository name | `diagnostics_for_physics` |
| Workflow filename | `pypi.yml` |
| Environment name | `pypi` |

The first successful upload creates the project under the PyPI account that
registered this publisher. A pending publisher does not reserve the name.
No PyPI token or password is stored in GitHub.

See the [PyPI instructions for creating a project with trusted publishing](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

## Release workflow

Build and validate a new version, commit it, and create its matching `vX.Y.Z`
tag. Publishing the GitHub release triggers `.github/workflows/pypi.yml`.
The workflow checks the tag against the distribution metadata, builds both
distributions, checks their metadata and contents, tests the installed wheel
outside the checkout, and publishes with the `pypi` environment identity.

For an existing tag, run the **Publish PDECheck to PyPI** workflow manually and
enter that tag. This also supports retrying the first publication after the
pending publisher has been configured.

After publication, verify <https://pypi.org/project/pdecheck/> and install into
a clean environment with `python -m pip install pdecheck`, then run
`pdecheck --version` and `pdecheck --self-check`.
