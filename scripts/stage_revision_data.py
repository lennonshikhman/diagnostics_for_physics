"""Copy configured raw benchmarks locally with streamed SHA-256 verification.

The source can be mounted read-only. No benchmark values or split definitions
are changed. Completed tasks are explicitly listed in staging_manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from diagnostic_suite.tasks import build_task_specs


def checksum(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["adv2d", "shallow2d", "active2d", "cns3d", "mhd3d"])
    args = parser.parse_args()
    args.destination_root.mkdir(parents=True, exist_ok=True)
    path = args.destination_root / "staging_manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {"files": {}, "completed_tasks": []}
    manifest["source_root"] = str(args.source_root.resolve())
    manifest["destination_root"] = str(args.destination_root.resolve())
    specs = build_task_specs(args.source_root, 64, 32)

    def persist():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)

    for task in args.tasks:
        names = sorted(
            {
                p
                for patterns in specs[task].globs.values()
                for pattern in patterns
                for p in args.source_root.glob(pattern)
            }
        )
        if not names:
            raise FileNotFoundError(f"No source files for {task}")
        for source in names:
            relative = source.relative_to(args.source_root)
            target = args.destination_root / relative
            stat = source.stat()
            key = relative.as_posix()
            prior = manifest["files"].get(key)
            if (
                prior
                and target.is_file()
                and stat.st_size == prior["size"]
                and stat.st_mtime_ns == prior["source_mtime_ns"]
            ):
                if checksum(target) != prior["sha256"]:
                    raise RuntimeError(f"Previously staged input changed: {target}")
                print(f"verified {task}: {relative}", flush=True)
                continue
            if target.exists():
                raise FileExistsError(f"Unverified destination already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            started = time.perf_counter()
            digest = hashlib.sha256()
            copied, last = 0, started
            with source.open("rb") as incoming, temporary.open("wb") as outgoing:
                for block in iter(lambda: incoming.read(8 * 1024**2), b""):
                    outgoing.write(block)
                    digest.update(block)
                    copied += len(block)
                    now = time.perf_counter()
                    if now - last > 30:
                        print(
                            f"copying {task} {source.name}: {copied / 2**30:.2f}/{stat.st_size / 2**30:.2f} GiB",
                            flush=True,
                        )
                        last = now
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError(f"Source changed while copying: {source}")
            actual = checksum(temporary)
            if actual != digest.hexdigest():
                raise RuntimeError(f"Staged bytes fail SHA-256 comparison: {source}")
            temporary.replace(target)
            manifest["files"][key] = dict(
                size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns,
                sha256=actual,
                copy_seconds=time.perf_counter() - started,
            )
            persist()
            print(f"staged {task}: {relative} ({stat.st_size / 2**30:.2f} GiB)", flush=True)
        if task not in manifest["completed_tasks"]:
            manifest["completed_tasks"].append(task)
        persist()
        print(f"READY {task}", flush=True)


if __name__ == "__main__":
    main()
