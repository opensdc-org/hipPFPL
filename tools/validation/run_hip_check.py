#!/usr/bin/env python3
"""Run one bounded, device-checked HIP validation executable and retain evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from reference_suite import child_limits, file_record, save_json, stop_owned_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--expected-pci", required=True)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.timeout <= 600:
        parser.error("Invalid device or timeout")
    binary = args.binary.resolve(strict=True)
    source = args.source.resolve(strict=True)
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if not (output.is_relative_to(root) or output.is_relative_to(Path("/tmp"))):
        parser.error("Output must stay inside this checkout or /tmp")
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(root), "rev-parse", "HEAD"],
        cwd=root, text=True).strip()
    env = os.environ.copy()
    removed = ["ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
               "OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY"]
    for name in removed:
        env.pop(name, None)
    selected = {"HIP_VISIBLE_DEVICES": str(args.device), "OMP_NUM_THREADS": "2",
                "OMP_THREAD_LIMIT": "2", "OMP_DYNAMIC": "FALSE"}
    env.update(selected)
    command = [str(binary), "--expected-pci", args.expected_pci]
    stdout = output / "stdout.json"
    stderr = output / "stderr.log"
    metadata = {
        "revision": revision, "started_utc": datetime.now(timezone.utc).isoformat(),
        "binary": file_record(binary), "test_source": file_record(source),
        "production_sources": [file_record(p) for p in sorted((root / "src").rglob("*"))
                               if p.is_file()],
        "runner": file_record(Path(__file__).resolve()),
        "command": command, "cwd": str(output), "environment": selected,
        "environment_removed": removed, "expected_pci": args.expected_pci,
        "timeout_seconds": args.timeout,
    }
    save_json(output / "invocation.json", metadata)
    timed_out = False
    started = time.monotonic()
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(command, cwd=output, env=env, stdout=out, stderr=err,
                                   start_new_session=True, preexec_fn=child_limits)
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_owned_process(process)
        except BaseException:
            stop_owned_process(process)
            raise
    errors = []
    result = None
    try:
        result = json.loads(stdout.read_text())
        if not isinstance(result, dict):
            raise ValueError("Result is not a JSON object")
        if result.get("status") != "passed":
            errors.append("Executable did not report passed")
        count = result.get("case_count")
        if type(count) is not int or count <= 0:
            errors.append("Missing positive case_count")
        if result.get("failures") not in (0, []):
            errors.append("Missing or nonzero failures")
        cases = result.get("cases")
        if not isinstance(cases, list) or len(cases) != count:
            errors.append("Case list does not match case_count")
        elif any(not isinstance(case, dict) or case.get("status") != "passed"
                 for case in cases):
            errors.append("One or more cases did not pass")
        # Each executable verifies the expected PCI before its first allocation.
        # Keep its complete device report, rather than inferring identity from an index.
        if not result.get("device") or not result.get("pci"):
            errors.append("Missing device/PCI report")
        elif result["pci"].lower() != args.expected_pci.lower():
            errors.append("Reported PCI does not match selected device")
        if result.get("errors"):
            errors.append("Executable reported errors")
    except (ValueError, OSError) as exc:
        errors.append("Invalid result JSON: " + str(exc))
    if process.returncode != 0 or timed_out:
        errors.append("Process failed or timed out")
    metadata.update({
        "returncode": process.returncode, "timeout": timed_out,
        "wall_seconds": time.monotonic() - started,
        "stdout": file_record(stdout), "stderr": file_record(stderr),
        "validation_errors": errors, "result": result,
        "status": "passed" if not errors else "failed",
    })
    save_json(output / "result.json", metadata)
    print(json.dumps({"status": metadata["status"], "result": str(output / "result.json"),
                      "returncode": process.returncode, "errors": errors}))
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
