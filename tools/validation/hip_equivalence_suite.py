#!/usr/bin/env python3
"""Compare HIP file streams with the original PFPL CPU reference."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import reference_suite as reference


ROOT = Path(__file__).resolve().parents[2]
REMOVED_ENV = ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
               "OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY")
PATHS = ("serial_to_hip", "openmp_to_hip", "hip_to_hip", "hip_to_serial", "hip_to_openmp")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pci_address(value):
    match = re.fullmatch(r"([0-9a-fA-F]{1,4}):([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])", value)
    require(match is not None, "Expected a PCI address DDDD:BB:DD.F")
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    require(device <= 31, "Invalid PCI device number")
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def verify_file(record, executable=False):
    path = Path(record["path"])
    require(path.is_absolute() and path.is_file(), "Missing absolute file: " + str(path))
    actual = reference.file_record(path)
    require(record.get("sha256") == actual["sha256"], "SHA256 mismatch: " + str(path))
    size = record.get("size_bytes", record.get("bytes"))
    require(size is None or size == actual["size_bytes"], "Size mismatch: " + str(path))
    require(not executable or os.access(path, os.X_OK), "Not executable: " + str(path))
    return actual


def verify_records(value, verified):
    """Check all saved inputs, outputs and logs, including independent repeats."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            actual = verify_file(value)
            verified[actual["path"]] = actual
        for child in value.values():
            verify_records(child, verified)
    elif isinstance(value, list):
        for child in value:
            verify_records(child, verified)


def capacity(size):
    chunks = (size + reference.CHUNK_BYTES - 1) // reference.CHUNK_BYTES
    worst = 16 + 2 * chunks + size
    allocated = 12 + 2 * chunks + chunks * reference.CHUNK_BYTES
    return {"worst_case_encoded_bytes": worst, "original_allocation_bytes": allocated,
            "safe_upper_bound": worst <= allocated}


def load_hip_manifest(path, report):
    hip = json.loads(path.read_text())
    report["hip_manifest"] = reference.file_record(path)
    require(hip.get("build_returncode") == 0, "HIP build manifest did not succeed")
    expected_keys = {f"{dtype}_{mode}_{operation}" for dtype in reference.DTYPES
                     for mode in reference.MODES for operation in ("comp", "decomp")}
    require(set(hip["binaries"]) == expected_keys, "Expected the twelve HIP codec binaries")
    report["hip_binaries"] = {key: verify_file(item, executable=True)
                              for key, item in hip["binaries"].items()}
    actual_sources = {path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*")
                      if path.is_file()}
    require(hip["source_sha256"] and actual_sources == set(hip["source_sha256"]),
            "HIP source inventory differs from its build")
    report["production_sources"] = [verify_file({"path": str(ROOT / name), "sha256": digest})
                                    for name, digest in sorted(hip["source_sha256"].items())]
    report["probe_binary"] = verify_file(hip["runtime_probe_binary"], executable=True)
    report["probe_source"] = verify_file(hip["runtime_probe_source"])
    return hip


def select_cases(cases, args):
    if args.smoke:
        chosen = [case for case in cases if case["id"].endswith("_staircase_half_chunk")]
        expected = {(f"{dtype}_{mode}_staircase_half_chunk", dtype, mode)
                    for dtype in reference.DTYPES for mode in reference.MODES}
    else:
        chosen = [case for case in cases if case["dtype"] == args.dtype and case["mode"] == args.mode]
        expected = {(f"{args.dtype}_{args.mode}_{pattern}_{length}", args.dtype, args.mode)
                    for pattern, length, _ in reference.case_fixtures(args.dtype, False)}
    require(len(chosen) == len(expected) and
            {(case["id"], case["dtype"], case["mode"]) for case in chosen} == expected,
            "CPU reference results lack the complete selected case matrix")
    return chosen


def prepare(args, report):
    report["revision"] = subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(ROOT), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = json.loads(args.reference_results.read_text())
    require(source.get("baseline_sha") == reference.BASELINE, "Wrong S1 baseline")
    require(source.get("internal_runs_unchanged") == 9, "S1 must retain nine internal runs")
    require(not source["summary"]["failures"] and not source["summary"]["harness_errors"],
            "S1 contains failures or harness errors")
    report["reference_results"] = reference.file_record(args.reference_results)
    report["reference_harness"] = verify_file(source["harness"])
    require(report["reference_harness"]["sha256"] == reference.file_record(Path(reference.__file__))["sha256"],
            "Imported reference harness differs from the S1 harness")
    report["reference_manifest"] = verify_file(source["manifest"])
    manifest, cpu_binaries = reference.validate_manifest(Path(source["manifest"]["path"]))
    require(manifest == source["reference_manifest"], "Embedded and saved S1 manifests differ")
    report["cpu_binaries"] = cpu_binaries
    report["cpu_sources"] = [verify_file({"path": str(Path(manifest["source_directory"]) / name),
                                          "sha256": digest})
                             for name, digest in sorted(manifest["source_sha256"].items())]
    load_hip_manifest(args.hip_manifest, report)
    chosen = select_cases(source["cases"], args)
    verified = {}
    for old in chosen:
        dtype, mode = old["dtype"], old["mode"]
        verify_records(old, verified)
        original = Path(old["input"]["path"]).read_bytes()
        require(len(original) == old["elements"] * reference.DTYPES[dtype][1], "Input count mismatch")
        header = reference.expected_header_parameter(original, dtype, mode, old["error_bound_token"])
        require(header == old["expected_header_parameter"], "S1 expected header contract differs")
        preflight = capacity(len(original))
        require(preflight == old["encoder_capacity_preflight"], "S1 encoder preflight differs")
        case = {key: old[key] for key in ("id", "dtype", "mode", "pattern", "elements", "input",
                                         "error_bound_token", "threshold_token", "expected_header_parameter")}
        case.update({"status": "not_run", "reference_status": old["status"],
                     "encoder_capacity_preflight": preflight, "compressions": {}, "decodings": {},
                     "stream_comparisons": {}, "issues": [], "failures": [],
                     "five_way_reconstruction_bitwise_equal": False})
        report["cases"].append(case)
        if not preflight["safe_upper_bound"]:
            require(old["status"] == "blocked_by_upstream" and
                    any(issue["id"] == "U001" for issue in old["issues"]), "Unrecorded unsafe S1 input")
            case["status"] = "blocked_by_upstream"
            case["issues"].append({"id": "U001", "status": "blocked_by_upstream",
                                   "detail": "Original allocation cannot contain the worst-case stream; encoder not invoked."})
            case["compressions"]["hip"] = {"status": "blocked_by_upstream", "issue": "U001"}
            case["decodings"] = {label: {"status": "blocked_by_upstream", "issue": "U001"} for label in PATHS}
            continue
        require(old["status"] in ("passed", "observed_upstream_not_raw_equal"),
                "Selected S1 case has no complete usable CPU oracle: " + old["id"])
        require(old["four_way_reconstruction_bitwise_equal"] and not old["failures"], "S1 CPU oracle disagrees")
        decoded_reference = Path(old["decodings"]["serial_to_serial"]["output"]["path"]).read_bytes()
        for encoder in reference.VARIANTS:
            invocation = old["compressions"][encoder]
            require(invocation["ok"], "Unsuccessful S1 encode")
            data = Path(invocation["output"]["path"]).read_bytes()
            layout = reference.parse_stream(data, dtype, len(original))
            require(layout == invocation["layout"] and layout["stored_error_bound_hex"] == header["hex"],
                    "S1 stream layout/header differs from saved evidence")
            for decoder in reference.VARIANTS:
                decoded = old["decodings"][encoder + "_to_" + decoder]
                require(decoded["ok"] and decoded["status"] == "passed", "Unsuccessful S1 decode")
                data = Path(decoded["output"]["path"]).read_bytes()
                require(data == decoded_reference, "S1 decoded bytes disagree")
                numeric = reference.check_numerics(original, data, dtype, mode,
                                                   layout["stored_error_bound"], threshold(case))
                require(numeric["ok"], "S1 reconstruction fails its numerical contract")
    report["verified_reference_files"] = list(verified.values())
    return {case["id"]: case for case in chosen}


def threshold(case):
    return float(case["threshold_token"]) if case["threshold_token"] is not None else None


def invoke(binary, arguments, inputs, directory, label, operation, args, report, destination):
    """Run one owned process; persist its invocation even when interrupted."""
    verify_file(binary, executable=True)
    for item in inputs:
        verify_file(item)
    command = [binary["path"], *map(str, arguments)]
    selected = {**reference.THREAD_ENV, "HIP_VISIBLE_DEVICES": str(args.device), "LC_ALL": "C"}
    env = os.environ.copy()
    for key in REMOVED_ENV:
        env.pop(key, None)
    env.update(selected)
    stdout, stderr = directory / (label + ".stdout.log"), directory / (label + ".stderr.log")
    invocation = {"command": command, "cwd": str(directory), "binary": binary, "inputs": inputs,
                  "environment": selected, "environment_removed": list(REMOVED_ENV),
                  "started_utc": datetime.now(timezone.utc).isoformat(), "timeout_seconds": args.timeout,
                  "timeout": False, "returncode": None, "signal": None, "ok": False}
    destination[label] = invocation
    started = time.monotonic()
    process = None
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            process = subprocess.Popen(command, cwd=directory, env=env, stdout=out, stderr=err,
                                       start_new_session=True, preexec_fn=reference.child_limits)
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                invocation["timeout"] = True
                reference.stop_owned_process(process)
            except BaseException:
                reference.stop_owned_process(process)
                raise
    finally:
        invocation["wall_seconds"] = time.monotonic() - started
        if process is not None:
            invocation["returncode"] = process.returncode
            invocation["signal"] = -process.returncode if process.returncode is not None and process.returncode < 0 else None
        for name, path in (("stdout", stdout), ("stderr", stderr)):
            if path.exists():
                invocation[name] = reference.file_record(path)
    text = stdout.read_text(errors="replace")
    samples = [float(value) for value in re.findall(r"lc " + (operation or "probe") + r" ecltime,\s*([0-9.eE+-]+)", text)]
    invocation["internal_timing_samples_seconds"] = samples
    invocation["expected_internal_runs"] = 9 if operation else None
    invocation["ok"] = (invocation["returncode"] == 0 and not invocation["timeout"] and
                         (operation is None or len(samples) == 9 and all(math.isfinite(x) and x >= 0 for x in samples)))
    require(invocation["ok"], label + " failed, timed out, or did not report nine internal runs")
    for item in inputs:
        verify_file(item)
    return invocation


def runtime_preflight(args, report):
    invocation = invoke(report["probe_binary"], ["--expected-pci", args.expected_pci], [], args.output,
                        "runtime_probe", None, args, report, report["probe"])
    probe = json.loads(Path(invocation["stdout"]["path"]).read_text())
    invocation["result"] = probe
    require(probe.get("status") == "passed" and probe.get("errors") == [] and
            probe.get("visible_device_count") == 1 and probe.get("logical_device") == 0 and
            pci_address(probe["actual_pci"]) == args.expected_pci and
            all(probe.get("checks", {}).get(key) is True for key in
                ("pci_matches", "copy_exact", "kernel_exact", "event_synchronized", "device_synchronized")),
            "GPU/PCI runtime preflight did not pass")
    return probe


def stream_compare(case, label, left, right, layouts):
    comparison = reference.compare_streams(left, layouts[0], right, layouts[1])
    case["stream_comparisons"][label] = comparison
    reference.record_stream_difference(case, comparison, label)
    require(not case["failures"], case["failures"][-1] if case["failures"] else "Stream comparison failed")


def run_case(case, old, args, report):
    directory = args.output / case["id"]
    directory.mkdir()
    if case["status"] == "blocked_by_upstream":
        reference.save_json(directory / "case.json", case)
        return
    dtype, mode = case["dtype"], case["mode"]
    key = dtype + "_" + mode
    original = Path(case["input"]["path"]).read_bytes()
    target = Path(old["decodings"]["serial_to_serial"]["output"]["path"]).read_bytes()
    streams = {name: Path(old["compressions"][name]["output"]["path"]).read_bytes() for name in reference.VARIANTS}
    records = {name: old["compressions"][name]["output"] for name in reference.VARIANTS}
    layouts = {name: reference.parse_stream(data, dtype, len(original)) for name, data in streams.items()}
    try:
        for label in ("hip", "hip_repeat") if case["id"].endswith("staircase_half_chunk") else ("hip",):
            path = directory / (label + ".pfpl")
            arguments = [case["input"]["path"], path, case["error_bound_token"]]
            if case["threshold_token"] is not None:
                arguments.append(case["threshold_token"])
            invocation = invoke(report["hip_binaries"][key + "_comp"], arguments, [case["input"]],
                                directory, label, "comp", args, report, case["compressions"])
            invocation["output"] = reference.file_record(path)
            data = path.read_bytes()
            layout = reference.parse_stream(data, dtype, len(original))
            invocation["layout"] = layout
            invocation["header_parameter_matches_expected"] = layout["stored_error_bound_hex"] == case["expected_header_parameter"]["hex"]
            require(invocation["header_parameter_matches_expected"], "HIP header violates the CLI/input error-bound contract")
            streams[label], layouts[label], records[label] = data, layout, invocation["output"]
            if label == "hip":
                for cpu in reference.VARIANTS:
                    stream_compare(case, "hip_vs_" + cpu, data, streams[cpu], (layout, layouts[cpu]))
            else:
                stream_compare(case, "hip_independent_repeat", streams["hip"], data, (layouts["hip"], layout))
        for label in PATHS:
            encoder, decoder = label.split("_to_")
            if len(streams[encoder]) > len(original):
                case["decodings"][label] = {"status": "blocked_by_upstream", "issue": "U004",
                                             "encoded_size_bytes": len(streams[encoder]), "original_size_bytes": len(original)}
                case["issues"].append({"id": "U004", "status": "blocked_by_upstream", "comparison": label,
                                       "detail": "Original CPU and GPU decoder host-input allocations are smaller than the encoded stream."})
                continue
            binary = (report["hip_binaries"][key + "_decomp"] if decoder == "hip"
                      else report["cpu_binaries"][decoder][key + "_decomp"])
            path = directory / (label + ".decoded.bin")
            invocation = invoke(binary, [records[encoder]["path"], path], [records[encoder]], directory,
                                label, "decomp", args, report, case["decodings"])
            invocation["output"] = reference.file_record(path)
            reconstructed = path.read_bytes()
            invocation["reference_comparison"] = reference.compare_bytes(target, reconstructed)
            invocation["numeric_check"] = reference.check_numerics(original, reconstructed, dtype, mode,
                                                                   layouts[encoder]["stored_error_bound"], threshold(case))
            invocation["status"] = "passed" if invocation["reference_comparison"]["equal"] and invocation["numeric_check"]["ok"] else "failed"
            require(invocation["status"] == "passed", label + " differs from S1 bytes or fails the numerical contract")
        case["five_way_reconstruction_bitwise_equal"] = (len(case["decodings"]) == 5 and
            all(item.get("status") == "passed" for item in case["decodings"].values()))
        if any(item.get("status") == "blocked_by_upstream" for item in case["decodings"].values()):
            case["status"] = "blocked_by_upstream"
        else:
            require(case["five_way_reconstruction_bitwise_equal"], "Incomplete five-way decoder comparison")
            case["status"] = "observed_upstream_not_raw_equal" if case["issues"] else "passed"
    except BaseException as exc:
        case["status"] = "failed"
        if str(exc) not in case["failures"]:
            case["failures"].append(str(exc) or type(exc).__name__)
        raise
    finally:
        reference.save_json(directory / "case.json", case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--hip-manifest", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Check one half-chunk case per dtype and mode")
    parser.add_argument("--dtype", choices=reference.DTYPES)
    parser.add_argument("--mode", choices=reference.MODES)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--expected-pci", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, choices=(20, 30), default=30)
    args = parser.parse_args()
    if (args.smoke and (args.dtype or args.mode)) or (not args.smoke and not (args.dtype and args.mode)):
        parser.error("Use --smoke or specify both --dtype and --mode")
    try:
        require(args.device >= 0 and sys.byteorder == "little", "Require a nonnegative GPU index and little-endian host")
        args.expected_pci = pci_address(args.expected_pci)
        args.reference_results = args.reference_results.resolve(strict=True)
        args.hip_manifest = args.hip_manifest.resolve(strict=True)
        args.output = args.output.resolve()
        require(args.output.is_relative_to(ROOT) or args.output.is_relative_to(Path("/tmp")),
                "Output must remain inside this checkout or /tmp")
        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    expected_cases = (len(reference.DTYPES) * len(reference.MODES) if args.smoke
                      else len(reference.case_fixtures(args.dtype, False)))
    expected_repeats = expected_cases if args.smoke else 1
    report = {"schema_version": 1, "baseline_sha": reference.BASELINE, "dtype": args.dtype, "mode": args.mode,
              "smoke": args.smoke,
              "suite_command": [sys.executable, *sys.argv],
              "harness": reference.file_record(Path(__file__).resolve()), "cases": [], "errors": [],
              "expected_case_count": expected_cases, "internal_runs_unchanged": 9, "timeout_seconds": args.timeout,
              "device": args.device, "expected_pci": args.expected_pci, "probe": {},
              "started_utc": datetime.now(timezone.utc).isoformat()}
    interrupted = False
    try:
        old_cases = prepare(args, report)
        reference.save_json(args.output / "preflight.json", report)
        runtime_preflight(args, report)
        for case in report["cases"]:
            run_case(case, old_cases[case["id"]], args, report)
    except KeyboardInterrupt:
        interrupted = True
        report["errors"].append("Interrupted; no further GPU processes were started")
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    comparisons = [item for case in report["cases"] for item in case["stream_comparisons"].values()]
    summary = {"expected_cases": expected_cases, "case_count": len(report["cases"]),
               "passed_cases": sum(case["status"] == "passed" for case in report["cases"]),
               "blocked_cases": sum(case["status"] == "blocked_by_upstream" for case in report["cases"]),
               "not_run_cases": sum(case["status"] == "not_run" for case in report["cases"]),
               "failed_cases": sum(case["status"] == "failed" for case in report["cases"]),
               "observed_upstream_not_raw_equal": sum(case["status"] == "observed_upstream_not_raw_equal" for case in report["cases"]),
               "raw_stream_comparisons": len(comparisons),
               "raw_stream_equal": sum(item["equal"] for item in comparisons),
               "raw_stream_mismatches": sum(not item["equal"] for item in comparisons),
               "u002_observations": sum(issue["id"] == "U002" for case in report["cases"] for issue in case["issues"]),
               "five_way_bitwise_equal_cases": sum(case["five_way_reconstruction_bitwise_equal"] for case in report["cases"]),
               "independent_repeat_compressions": sum("hip_repeat" in case["compressions"] for case in report["cases"]),
               "expected_independent_repeat_compressions": expected_repeats, "harness_errors": len(report["errors"])}
    failed = bool(report["errors"] or summary["failed_cases"] or summary["not_run_cases"] or
                  summary["case_count"] != expected_cases or
                  summary["independent_repeat_compressions"] != expected_repeats)
    status = "failed" if failed else "blocked_by_upstream" if summary["blocked_cases"] else (
        "complete_with_upstream_raw_mismatches" if summary["raw_stream_mismatches"] else "passed")
    report["status"] = summary["status"] = "interrupted" if interrupted else status
    report["summary"] = summary
    reference.save_json(args.output / "results.json", report)
    print(json.dumps({"results": str(args.output / "results.json"), "summary": summary}, sort_keys=True))
    return 130 if interrupted else 1 if failed else 2 if summary["blocked_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
