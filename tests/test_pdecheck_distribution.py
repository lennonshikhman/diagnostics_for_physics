from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

from diagnostic_suite import __version__


def test_installed_distribution_matches_import_version() -> None:
    distribution = importlib.metadata.distribution("pdecheck")
    assert distribution.metadata["Name"] == "pdecheck"
    assert distribution.version == __version__


@pytest.mark.parametrize("command", ["pdecheck", "diagnostic-suite", "diagnostics-for-physics"])
def test_new_and_existing_installed_commands(command: str, tmp_path: Path) -> None:
    executable = Path(sys.executable).parent / command
    result = subprocess.run([str(executable), "--version"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"pdecheck {__version__}"
