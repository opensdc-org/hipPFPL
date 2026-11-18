#!/usr/bin/env python3
"""Build the unchanged, pinned PFPL CPU reference in a fresh artifact directory."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time


BASELINE = "36f5aaef42744ed78b4d1525b03a7fb2168b363c"
REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, cwd, env, timeout):
    begin = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    return {"command": command, "returncode": process.returncode,
            "timed_out": timed_out, "elapsed_seconds": time.monotonic() - begin,
            "stdout": stdout, "stderr": stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh directory; existing directories are never overwritten")
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    output = args.output.resolve()
    compiler = shutil.which(args.compiler)
    if not compiler:
        parser.error("CPU compiler not found")
    compiler = str(Path(compiler).resolve())
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    logs = output / "logs"
    logs.mkdir()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    env = dict(os.environ, OMP_NUM_THREADS="2", OMP_THREAD_LIMIT="2",
               OMP_DYNAMIC="FALSE", LC_ALL="C")
    git = ["git", "-c", f"safe.directory={REPO}", "-C", str(REPO)]
    archive = subprocess.check_output(git + ["archive", BASELINE, "src", "makefile", "LICENSE"])
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            target = source / member.name
            if not target.resolve().is_relative_to(source):
                raise RuntimeError("Unexpected archive path")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
            else:
                raise RuntimeError("Reference archive contains an unsupported entry")

    source_hashes = {str(path.relative_to(source)): sha256(path)
                     for path in sorted(source.rglob("*")) if path.is_file()}
    version = subprocess.check_output([compiler, "--version"], text=True).strip()
    manifest = {
        "schema_version": 1, "baseline_sha": BASELINE,
        "source_directory": str(source), "source_sha256": source_hashes,
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "compiler": compiler, "compiler_version": version,
        "machine": platform.machine(), "byteorder": sys.byteorder,
        "environment": {k: env[k] for k in
                        ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC", "LC_ALL")},
        "original_num_runs": 9, "build_parallelism": 1,
        "binaries": {"serial": {}, "openmp": {}}, "builds": [],
    }
    manifest_path = output / "manifest.json"
    for backend in ("serial", "openmp"):
        binary_dir = output / "bin" / backend
        binary_dir.mkdir(parents=True)
        for dtype in ("f32", "f64"):
            for mode in ("abs", "rel", "noa"):
                for direction in ("comp", "decomp"):
                    name = f"{dtype}_{mode}_{direction}"
                    src = source / "src" / f"{name}_cpu.cpp"
                    binary = binary_dir / name
                    # Preserve the upstream makefile's CPU flags and source NDEBUG.
                    command = [compiler, "-O3", "-march=native"]
                    if backend == "openmp":
                        command.append("-fopenmp")
                    command += ["-mno-fma", "-ffp-contract=off", "-I" + str(source / "src"),
                                "-std=c++17", "-o", str(binary), str(src)]
                    result = run(command, source, env, args.timeout)
                    log_name = f"{backend}-{name}"
                    for stream in ("stdout", "stderr"):
                        log = logs / f"{log_name}.{stream}.txt"
                        log.write_text(result.pop(stream))
                        result[stream + "_path"] = str(log)
                        result[stream + "_sha256"] = sha256(log)
                    manifest["builds"].append(result)
                    if result["returncode"] == 0 and not result["timed_out"]:
                        manifest["binaries"][backend][name] = {
                            "path": str(binary), "sha256": sha256(binary),
                            "source": str(src), "source_sha256": sha256(src)}
                        print(f"built {backend}/{name}", flush=True)
                    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    if result["returncode"] != 0 or result["timed_out"]:
                        raise SystemExit(f"Reference build failed: {backend}/{name}; see {manifest_path}")
    print(f"Reference manifest: {manifest_path}")


if __name__ == "__main__":
    main()
