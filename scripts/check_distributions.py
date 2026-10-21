"""Reject incomplete source archives and accidental manuscript inclusion."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


def check_distributions(directory: Path) -> None:
    wheels, sources = list(directory.glob("*.whl")), list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one source distribution and one wheel")
    with tarfile.open(sources[0]) as archive:
        source_names = [Path(*Path(name).parts[1:]).as_posix() for name in archive.getnames()]
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_names = archive.namelist()
    for name in source_names + wheel_names:
        if "PAPER" in Path(name).parts or name.lower().endswith((".tex", ".pdf")):
            raise ValueError(f"Private manuscript material in distribution: {name}")
    for required in [
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "AUTHORS.md",
        "docs/user_guide.md",
        "examples/callable_propagator_demo.py",
        "scripts/check_distributions.py",
        "scripts/reproduce_revision.sh",
        "scripts/analyze_controlled_revision.py",
        "scripts/analyze_external_revision.py",
        "scripts/check_revision_pdfs.py",
        "scripts/validate_revision_release.py",
        "configs/revision_protocol.json",
        "configs/revision_file_selection.json",
        "docs/revision_reproduction.md",
        "tests/test_revision_contracts.py",
    ]:
        if required not in source_names:
            raise ValueError(f"Source distribution is missing {required}")
    for name in ("referee", "sensitivity", "controlled", "api", "propagators"):
        if f"diagnostic_suite/{name}.py" not in wheel_names:
            raise ValueError(f"Wheel is missing {name}")
    print("Source archive and wheel verified; PAPER is excluded.")


if __name__ == "__main__":
    check_distributions(Path(sys.argv[1]))
