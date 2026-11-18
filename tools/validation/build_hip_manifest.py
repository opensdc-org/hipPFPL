#!/usr/bin/env python3
"""Build the HIP CLI and record the artifacts used by the equivalence tests."""

import argparse
from pathlib import Path
import subprocess

from reference_suite import DTYPES, MODES, file_record, save_json


ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    return {path.relative_to(ROOT).as_posix(): file_record(path)["sha256"]
            for path in sorted((ROOT / "src").rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    bin_dir = output / "bin"
    command = ["make", "-j1", "hip-test-build", f"HIP_BIN_DIR={bin_dir}"]
    log_path = output / "build.log"
    with log_path.open("wb") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"HIP build failed; see {log_path}")
    if source_hashes() != sources:
        raise SystemExit("HIP source changed during the build")
    binaries = {
        f"{dtype}_{mode}_{operation}": file_record(bin_dir / dtype / "hip" / f"{dtype}_{mode}_{action}_hip")
        for dtype in DTYPES for mode in MODES
        for operation, action in (("comp", "compress"), ("decomp", "decompress"))
    }
    path = output / "manifest.json"
    save_json(path, {
        "build_returncode": result.returncode, "command": command, "build_log": file_record(log_path),
        "binaries": binaries, "source_sha256": sources,
        "runtime_probe_binary": file_record(bin_dir / "validation/hip_runtime_probe"),
        "runtime_probe_source": file_record(ROOT / "tools/validation/hip_runtime_probe.hip"),
    })
    print(f"HIP manifest: {path}")


if __name__ == "__main__":
    main()
