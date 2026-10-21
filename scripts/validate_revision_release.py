"""Exercise an installed wheel outside the checkout and preserve command logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def hashes(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.pt"))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    root, outdir, executable = args.source_root.resolve(), args.outdir.resolve(), str(args.python.absolute())
    if outdir.is_relative_to(root):
        raise ValueError("Installed-wheel validation must execute outside the checkout")
    outdir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    commands = []

    def run(name, arguments):
        command = [executable, *map(str, arguments)]
        result = subprocess.run(
            command,
            cwd=outdir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (outdir / f"{name}.log").write_text(result.stdout)
        commands.append(dict(name=name, command=command, exit_code=result.returncode))
        print(f"{name}: exit {result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Validation failed: {name}; see {outdir / name}.log")
        return result.stdout

    installation = json.loads(
        run(
            "installation",
            [
                "-c",
                "import json,platform,torch,numpy,pandas,h5py,diagnostic_suite;"
                "from diagnostic_suite.runner import _source_fingerprint;"
                "print(json.dumps(dict(package_path=diagnostic_suite.__file__,version=diagnostic_suite.__version__,"
                "python=platform.python_version(),torch=str(torch.__version__),numpy=numpy.__version__,"
                "pandas=pandas.__version__,h5py=h5py.__version__,source_sha256=_source_fingerprint())))",
            ],
        )
    )
    if Path(installation["package_path"]).is_relative_to(root):
        raise RuntimeError("Validation imported the checkout rather than the installed wheel")
    run("pytest", ["-m", "pytest", root / "tests", "-c", root / "pyproject.toml"])
    run("version", ["-m", "diagnostic_suite.cli", "--version"])
    run("self_check", ["-m", "diagnostic_suite.cli", "--self-check"])
    run("referee", ["-m", "diagnostic_suite.referee", "--outdir", outdir / "referee"])
    run("sensitivity", ["-m", "diagnostic_suite.sensitivity", "--outdir", outdir / "sensitivity"])
    run(
        "balance_constraints",
        [root / "scripts/verify_balance_and_constraints.py", "--outdir", outdir / "balance_constraints"],
    )
    run("api_example", [root / "examples/api_demo.py"])
    run("callable_example", [root / "examples/callable_propagator_demo.py"])
    run(
        "make_toy",
        [
            "-m",
            "diagnostic_suite.cli",
            "--make-toy",
            "--data-root",
            outdir / "data",
            "--toy-grid",
            "16",
            "--toy-steps",
            "6",
        ],
    )
    command = [
        "-m",
        "diagnostic_suite.cli",
        "--data-root",
        outdir / "data",
        "--tasks",
        "toy2d",
        "--models",
        "fno",
        "deeponet",
        "unet",
        "resnet",
        "--smoke",
        "--device",
        "cpu",
        "--resume",
        "--outdir",
        outdir / "training",
    ]
    run("training", command)
    before = hashes(outdir / "training")
    if not before:
        raise RuntimeError("Training did not produce checkpoints")
    run("resume", command)
    if hashes(outdir / "training") != before:
        raise RuntimeError("Resume changed completed checkpoint bytes")
    result = dict(
        passed=True,
        installation=installation,
        commands=commands,
        checkpoint_count=len(before),
        resume_preserved_checkpoint_bytes=True,
    )
    (outdir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
