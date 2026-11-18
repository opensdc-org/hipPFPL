#!/usr/bin/env python3
"""Record the unchanged PFPL CPU reference and its serial/OpenMP agreement."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import struct
import subprocess
import sys
import time


BASELINE = "36f5aaef42744ed78b4d1525b03a7fb2168b363c"
CHUNK_BYTES = 16384
TIMEOUT_SECONDS = 20
SEED = 20261027
THREAD_ENV = {
    "OMP_NUM_THREADS": "2",
    "OMP_THREAD_LIMIT": "2",
    "OMP_DYNAMIC": "FALSE",
    "OMP_MAX_ACTIVE_LEVELS": "1",
    "OMP_WAIT_POLICY": "PASSIVE",
}
VARIANTS = ("serial", "openmp")
DTYPES = {"f32": ("f", 4), "f64": ("d", 8)}
MODES = ("abs", "rel", "noa")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    data = path.read_bytes()
    return {"path": str(path), "size_bytes": len(data), "sha256": sha256(data)}


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rounded(value, dtype):
    code, _ = DTYPES[dtype]
    return struct.unpack("<" + code, struct.pack("<" + code, value))[0]


def case_fixtures(dtype, smoke):
    chunk_elements = CHUNK_BYTES // DTYPES[dtype][1]
    sizes = (("half_chunk", chunk_elements // 2), ("chunk_plus_one", chunk_elements + 1),
             ("three_chunks_plus_three", 3 * chunk_elements + 3))
    patterns = ("staircase", "constant", "threshold")
    if smoke:
        sizes = sizes[:1]
        patterns = patterns[:1]
    return [(pattern, length, count) for pattern in patterns for length, count in sizes]


def make_input(dtype, pattern, count):
    code, size = DTYPES[dtype]
    if pattern == "staircase":
        values = [((index % 257) - 128) / 32 for index in range(count)]
    elif pattern == "constant":
        values = [3.25] * count
    else:
        integer_code = "I" if size == 4 else "Q"
        eight_bits = struct.unpack("<" + integer_code, struct.pack("<" + code, 8.0))[0]
        below = struct.unpack("<" + code, struct.pack("<" + integer_code, eight_bits - 1))[0]
        above = struct.unpack("<" + code, struct.pack("<" + integer_code, eight_bits + 1))[0]
        palette = [0.0, 2.0, -2.0, below, 8.0, above, -below, -8.0, -above]
        state = SEED
        # Repeat a seeded small alphabet to exercise the threshold without
        # intentionally invoking the original incompressible-capacity defect.
        tile = []
        for _ in range(64):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            tile.append(palette[state % len(palette)])
        tile[:len(palette)] = palette
        values = [tile[index % len(tile)] for index in range(count)]
    return struct.pack("<" + str(count) + code, *values)


def expected_header_parameter(original, dtype, mode, error_token):
    code, _ = DTYPES[dtype]
    parsed_eb = rounded(float(error_token), dtype)
    value = parsed_eb
    source_range = None
    if mode == "noa":
        values = [item[0] for item in struct.iter_unpack("<" + code, original)]
        source_range = rounded(max(values) - min(values), dtype)
        value = rounded(source_range * parsed_eb, dtype)
    return {
        "value": value, "hex": struct.pack("<" + code, value).hex(),
        "input_range_after_dtype_subtraction": source_range,
        "derivation": "dtype(dtype(max(input)-min(input))*dtype(atof(token)))" if mode == "noa"
                      else "dtype(atof(token))",
    }


def child_limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def stop_owned_process(process):
    """The process has its own session; never signal a pre-existing process."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def run_cli(command, case_dir, label, operation):
    stdout_path = case_dir / (label + ".stdout.log")
    stderr_path = case_dir / (label + ".stderr.log")
    env = os.environ.copy()
    env.update(THREAD_ENV)
    # Inherited affinity/team-size settings must not defeat the fixed policy.
    for key in ("OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY"):
        env.pop(key, None)
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("xb") as out, stderr_path.open("xb") as err:
        process = subprocess.Popen(
            command, cwd=case_dir, env=env, stdout=out, stderr=err,
            start_new_session=True, preexec_fn=child_limits,
        )
        try:
            process.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_owned_process(process)
        except BaseException:
            stop_owned_process(process)
            raise
    elapsed = time.monotonic() - started
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    samples = re.findall(r"lc " + operation + r" ecltime,\s*([0-9.eE+-]+)", text)
    return {
        "command": command, "cwd": str(case_dir), "environment": THREAD_ENV,
        "returncode": process.returncode,
        "signal": -process.returncode if process.returncode < 0 else None,
        "timeout": timed_out, "timeout_seconds": TIMEOUT_SECONDS,
        "wall_seconds": elapsed,
        "stdout": file_record(stdout_path), "stderr": file_record(stderr_path),
        "internal_timing_samples_seconds": [float(sample) for sample in samples],
        "expected_internal_runs": 9,
        "ok": process.returncode == 0 and not timed_out and len(samples) == 9,
    }


def parse_stream(data, dtype, expected_input_size):
    if len(data) < 16:
        raise ValueError("Compressed file is shorter than the 16-byte header")
    raw_size = struct.unpack_from("<q", data)[0]
    if raw_size != expected_input_size:
        raise ValueError("Header decoded size does not match input size")
    chunks = (raw_size + CHUNK_BYTES - 1) // CHUNK_BYTES
    payload_start = 16 + 2 * chunks
    if len(data) < payload_start:
        raise ValueError("Truncated chunk-length table")
    lengths = list(struct.unpack_from("<" + str(chunks) + "H", data, 16))
    for index, length in enumerate(lengths):
        original_chunk = min(CHUNK_BYTES, raw_size - index * CHUNK_BYTES)
        if not 0 < length <= original_chunk:
            raise ValueError("Invalid compressed chunk length at index " + str(index))
    if payload_start + sum(lengths) != len(data):
        raise ValueError("Chunk-length table does not account for the entire file")
    code, width = DTYPES[dtype]
    effective_eb = struct.unpack_from("<" + code, data, 8)[0]
    if not math.isfinite(effective_eb) or effective_eb < 0:
        raise ValueError("Unexpected nonfinite or negative effective error bound")
    return {
        "decoded_size_bytes": raw_size, "chunk_count": chunks,
        "header_hex": data[:16].hex(),
        "stored_error_bound": effective_eb,
        "stored_error_bound_hex": data[8:8 + width].hex(),
        "reserved_f32_bytes_hex": data[12:16].hex() if dtype == "f32" else None,
        "chunk_lengths": lengths, "payload_offset": payload_start,
        "raw_fallback_chunks": [i for i, length in enumerate(lengths)
                                if length == min(CHUNK_BYTES, raw_size - i * CHUNK_BYTES)],
        "sections": {
            "defined_header": sha256(data[:12] if dtype == "f32" else data[:16]),
            "chunk_length_table": sha256(data[16:payload_start]),
            "payload": sha256(data[payload_start:]),
        },
    }


def compare_bytes(left, right):
    offsets = [index for index, (a, b) in enumerate(zip(left, right)) if a != b]
    different = len(offsets) + abs(len(left) - len(right))
    return {
        "equal": different == 0, "different_bytes": different,
        "first_difference_offset": offsets[0] if offsets else (
            min(len(left), len(right)) if len(left) != len(right) else None),
        "first_differing_offsets": offsets[:16],
        "left_size_bytes": len(left), "right_size_bytes": len(right),
    }


def compare_streams(left, left_layout, right, right_layout):
    comparison = compare_bytes(left, right)
    comparison["sections_equal"] = {
        section: left_layout["sections"][section] == right_layout["sections"][section]
        for section in ("defined_header", "chunk_length_table", "payload")
    }
    comparison["effective_stream_equal"] = all(comparison["sections_equal"].values())
    comparison["reserved_bytes_equal"] = (
        left_layout["reserved_f32_bytes_hex"] == right_layout["reserved_f32_bytes_hex"])
    return comparison


def record_stream_difference(case, comparison, label):
    if comparison["equal"]:
        return
    reserved_only = (case["dtype"] == "f32" and comparison["effective_stream_equal"]
                     and comparison["left_size_bytes"] == comparison["right_size_bytes"])
    if reserved_only:
        case["issues"].append({
            "id": "U002", "status": "observed_upstream_not_raw_equal", "comparison": label,
            "detail": "Raw files differ; defined sections agree. Files were not modified.",
        })
    else:
        case["failures"].append(label + " streams differ outside the known f32 reserved field")


def check_numerics(original, decoded, dtype, mode, stored_eb, threshold):
    code, width = DTYPES[dtype]
    if len(original) != len(decoded):
        return {"ok": False, "reason": "reconstructed_length_mismatch"}
    errors = 0
    protected_errors = 0
    nonfinite = 0
    max_abs_error = 0.0
    max_source_precision_error = 0.0
    first_errors = []
    factor = rounded(1.0 + stored_eb, dtype)
    originals = struct.iter_unpack("<" + code, original)
    reconstructions = struct.iter_unpack("<" + code, decoded)
    for index, ((x,), (y,)) in enumerate(zip(originals, reconstructions)):
        finite = math.isfinite(y)
        if not finite:
            nonfinite += 1
        absolute_error = abs(x - y) if finite else None
        source_precision_error = abs(rounded(x - y, dtype)) if finite else None
        if finite:
            max_abs_error = max(max_abs_error, absolute_error)
            max_source_precision_error = max(max_source_precision_error, source_precision_error)
        if mode == "rel":
            lower = rounded(abs(x) / factor, dtype)
            upper = rounded(abs(x) * factor, dtype)
            bounded = finite and lower <= abs(y) <= upper
            if x != 0 and y != 0 and math.copysign(1, x) != math.copysign(1, y):
                bounded = False
        else:
            bounded = finite and source_precision_error <= stored_eb
        protected = threshold is not None and abs(x) >= threshold
        bit_equal = original[index * width:(index + 1) * width] == decoded[index * width:(index + 1) * width]
        if protected and not bit_equal:
            protected_errors += 1
        if not bounded:
            errors += 1
            if len(first_errors) < 8:
                first_errors.append({"index": index, "input": repr(x), "decoded": repr(y),
                                     "absolute_error_f64_diagnostic": absolute_error,
                                     "source_precision_absolute_error": source_precision_error})
    return {
        "ok": errors == 0 and protected_errors == 0,
        "stored_error_bound": stored_eb,
        "rule": "magnitude interval with dtype-rounded endpoints and sign check" if mode == "rel"
                else "abs(dtype-rounded input-minus-reconstruction) <= stored header bound; no tolerance multiplier",
        "error_bound_violations": errors, "protected_value_bit_mismatches": protected_errors,
        "nonfinite_reconstructions": nonfinite,
        "max_absolute_error_f64_diagnostic": max_abs_error,
        "max_source_precision_absolute_error": max_source_precision_error,
        "first_errors": first_errors,
    }


def run_case(binaries, output, dtype, mode, pattern, length_name, count):
    name = "_".join((dtype, mode, pattern, length_name))
    case_dir = output / name
    case_dir.mkdir()
    original = make_input(dtype, pattern, count)
    input_path = case_dir / "input.bin"
    input_path.write_bytes(original)
    token = "0.01"
    threshold = 8.0 if pattern == "threshold" else None
    case = {
        "id": name, "dtype": dtype, "mode": mode, "pattern": pattern,
        "seed": SEED if pattern == "threshold" else None,
        "elements": count, "input": file_record(input_path),
        "error_bound_token": token, "error_bound_after_cli_cast": rounded(float(token), dtype),
        "expected_header_parameter": expected_header_parameter(original, dtype, mode, token),
        "threshold_token": "8" if threshold is not None else None,
        "noa_constant_range": mode == "noa" and pattern == "constant",
        "compressions": {}, "repeat_compressions": {}, "decodings": {}, "issues": [], "failures": [],
        "four_way_reconstruction_bitwise_equal": False,
    }
    chunks = (len(original) + CHUNK_BYTES - 1) // CHUNK_BYTES
    required_worst_case = 16 + 2 * chunks + len(original)
    original_allocation = 12 + 2 * chunks + chunks * CHUNK_BYTES
    case["encoder_capacity_preflight"] = {
        "worst_case_encoded_bytes": required_worst_case,
        "original_allocation_bytes": original_allocation,
        "safe_upper_bound": required_worst_case <= original_allocation,
    }
    if required_worst_case > original_allocation:
        reason = ("U001: worst-case output exceeds the original allocation. "
                  "Compressibility is not used to waive the pre-execution safety check.")
        case["issues"].append({"id": "U001", "status": "blocked_by_upstream", "detail": reason})
        for variant in VARIANTS:
            case["compressions"][variant] = {"status": "blocked_by_upstream", "issue": "U001", "reason": reason}
            for decoder in VARIANTS:
                case["decodings"][variant + "_to_" + decoder] = {
                    "status": "blocked_by_upstream", "issue": "U001",
                    "reason": "Encoder was not invoked because its capacity upper bound is unsafe.",
                }
        case["status"] = "blocked_by_upstream"
        save_json(case_dir / "case.json", case)
        return case
    streams = {}
    layouts = {}
    for variant in VARIANTS:
        encoded_path = case_dir / (variant + ".pfpl")
        command = [binaries[variant][dtype + "_" + mode + "_comp"]["path"],
                   str(input_path), str(encoded_path), token]
        if threshold is not None:
            command.append("8")
        invocation = run_cli(command, case_dir, variant + "_comp", "comp")
        case["compressions"][variant] = invocation
        if not invocation["ok"] or not encoded_path.is_file():
            case["failures"].append(variant + " compression failed or did not report 9 runs")
            continue
        invocation["output"] = file_record(encoded_path)
        data = encoded_path.read_bytes()
        try:
            layout = parse_stream(data, dtype, len(original))
        except ValueError as exc:
            case["failures"].append(variant + ": " + str(exc))
            continue
        invocation["layout"] = layout
        invocation["header_parameter_matches_expected"] = (
            layout["stored_error_bound_hex"] == case["expected_header_parameter"]["hex"])
        if not invocation["header_parameter_matches_expected"]:
            case["failures"].append(variant + " header error-bound bits do not match the CLI/input contract")
            continue
        streams[variant], layouts[variant] = data, layout

    if len(streams) == 2:
        comparison = compare_streams(streams["serial"], layouts["serial"], streams["openmp"], layouts["openmp"])
        case["raw_stream_comparison"] = comparison
        record_stream_difference(case, comparison, "serial/OpenMP")

    # NUM_RUNS=9 overwrites one output buffer. A separate process invocation
    # is needed to observe whether the saved raw stream is reproducible.
    if pattern == "staircase" and length_name == "half_chunk":
        for variant in streams:
            repeated_path = case_dir / (variant + ".repeat.pfpl")
            command = [binaries[variant][dtype + "_" + mode + "_comp"]["path"],
                       str(input_path), str(repeated_path), token]
            invocation = run_cli(command, case_dir, variant + "_repeat_comp", "comp")
            case["repeat_compressions"][variant] = invocation
            if not invocation["ok"] or not repeated_path.is_file():
                case["failures"].append(variant + " independent repeat compression failed")
                continue
            invocation["output"] = file_record(repeated_path)
            repeated = repeated_path.read_bytes()
            try:
                repeated_layout = parse_stream(repeated, dtype, len(original))
            except ValueError as exc:
                case["failures"].append(variant + " independent repeat: " + str(exc))
                continue
            invocation["layout"] = repeated_layout
            invocation["header_parameter_matches_expected"] = (
                repeated_layout["stored_error_bound_hex"] == case["expected_header_parameter"]["hex"])
            if not invocation["header_parameter_matches_expected"]:
                case["failures"].append(variant + " independent repeat header error-bound bits do not match the CLI/input contract")
                continue
            comparison = compare_streams(streams[variant], layouts[variant], repeated, repeated_layout)
            invocation["raw_stream_comparison"] = comparison
            record_stream_difference(case, comparison, variant + " independent repeat")

    decoded_data = {}
    for encoder, data in streams.items():
        for decoder in VARIANTS:
            label = encoder + "_to_" + decoder
            if len(data) > len(original):
                case["decodings"][label] = {
                    "status": "blocked_by_upstream", "issue": "U004",
                    "encoded_size_bytes": len(data), "original_size_bytes": len(original),
                    "reason": "Original CPU decoder allocates original_size bytes for the encoded input.",
                }
                continue
            decoded_path = case_dir / (label + ".decoded.bin")
            command = [binaries[decoder][dtype + "_" + mode + "_decomp"]["path"],
                       str(case_dir / (encoder + ".pfpl")), str(decoded_path)]
            invocation = run_cli(command, case_dir, label + "_decomp", "decomp")
            case["decodings"][label] = invocation
            if not invocation["ok"] or not decoded_path.is_file():
                invocation["status"] = "failed"
                case["failures"].append(label + " decode failed or did not report 9 runs")
                continue
            invocation["output"] = file_record(decoded_path)
            reconstructed = decoded_path.read_bytes()
            decoded_data[label] = reconstructed
            invocation["numeric_check"] = check_numerics(
                original, reconstructed, dtype, mode, layouts[encoder]["stored_error_bound"], threshold)
            invocation["status"] = "passed" if invocation["numeric_check"]["ok"] else "failed"
            if not invocation["numeric_check"]["ok"]:
                case["failures"].append(label + " reference numerical behavior requires investigation")

    reference = decoded_data.get("serial_to_serial")
    case["reconstruction_comparisons"] = {
        label: compare_bytes(reference, data) for label, data in decoded_data.items()
    } if reference is not None else {}
    case["four_way_reconstruction_bitwise_equal"] = (
        len(decoded_data) == 4 and reference is not None
        and all(item["equal"] for item in case["reconstruction_comparisons"].values()))
    if reference is not None and any(data != reference for data in decoded_data.values()):
        case["failures"].append("cross-decoder reconstructed bytes differ")
    blocked = any(item.get("status") == "blocked_by_upstream" for item in case["decodings"].values())
    if case["failures"]:
        case["status"] = "failed"
    elif blocked:
        case["status"] = "blocked_by_upstream"
    elif not case["four_way_reconstruction_bitwise_equal"]:
        case["status"] = "failed"
        case["failures"].append("four-way reconstruction comparison is incomplete")
    elif case["issues"]:
        case["status"] = "observed_upstream_not_raw_equal"
    else:
        case["status"] = "passed"
    save_json(case_dir / "case.json", case)
    return case


def validate_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("baseline_sha") != BASELINE:
        raise ValueError("Manifest must identify the immutable original PFPL baseline")
    binaries = manifest["binaries"]
    verified = {}
    for variant in VARIANTS:
        verified[variant] = {}
        for dtype in DTYPES:
            for mode in MODES:
                for operation in ("comp", "decomp"):
                    key = "_".join((dtype, mode, operation))
                    entry = binaries[variant][key]
                    executable = Path(entry["path"])
                    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
                        raise ValueError("Missing executable for " + variant + "/" + key)
                    actual = file_record(executable)
                    if entry.get("sha256") is not None and entry["sha256"] != actual["sha256"]:
                        raise ValueError("Binary hash mismatch for " + variant + "/" + key)
                    verified[variant][key] = actual
    return manifest, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, non-existing output directory")
    parser.add_argument("--smoke", action="store_true", help="One half-chunk staircase per precision and mode")
    args = parser.parse_args()
    if sys.byteorder != "little":
        parser.error("The unchanged reference format is tested here only on a little-endian host")
    manifest_path = args.manifest.resolve()
    output = args.output.resolve()
    project_root = Path(__file__).resolve().parents[2]
    if not (output.is_relative_to(project_root) or output.is_relative_to(Path("/tmp"))):
        parser.error("Output must stay inside this project or /tmp")
    try:
        manifest, binaries = validate_manifest(manifest_path)
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    report = {
        "schema_version": 1, "baseline_sha": BASELINE,
        "smoke": args.smoke,
        "manifest": file_record(manifest_path), "reference_manifest": manifest,
        "verified_binaries": binaries, "harness": file_record(Path(__file__).resolve()),
        "python": sys.version, "byteorder": sys.byteorder, "environment": THREAD_ENV,
        "internal_runs_unchanged": 9, "timeout_seconds": TIMEOUT_SECONDS,
        "unsafe_input_policy": [{"issue": "U001", "reason":
            "Skip encoding if the worst-case compressed size exceeds the original allocation."}],
        "scope": "Finite CPU reference behavior only; no HIP/CUDA execution or performance claim.",
        "cases": [],
    }
    fixtures = {dtype: case_fixtures(dtype, args.smoke) for dtype in DTYPES}
    expected_cases = len(MODES) * sum(len(cases) for cases in fixtures.values())
    interrupted = False
    try:
        for dtype, cases in fixtures.items():
            for mode in MODES:
                for pattern, length_name, count in cases:
                    report["cases"].append(run_case(binaries, output, dtype, mode, pattern, length_name, count))
    except KeyboardInterrupt:
        interrupted = True
        report["interrupted"] = True
    except Exception as exc:
        report["harness_error"] = {"type": type(exc).__name__, "message": str(exc)}
    cases = report["cases"]
    summary = {
        "expected_cases": expected_cases, "completed_cases": len(cases),
        "executed_cases": sum(any("command" in item for item in case["compressions"].values()) for case in cases),
        "actual_passes": sum(case["status"] == "passed" for case in cases),
        "observed_upstream_not_raw_equal": sum(case["status"] == "observed_upstream_not_raw_equal" for case in cases),
        "raw_stream_mismatches": sum(case.get("raw_stream_comparison", {}).get("equal") is False for case in cases),
        "independent_repeat_compressions": sum(len(case["repeat_compressions"]) for case in cases),
        "expected_independent_repeat_compressions": 12,
        "raw_repeat_mismatches": sum(item.get("raw_stream_comparison", {}).get("equal") is False
                                     for case in cases for item in case["repeat_compressions"].values()),
        "blocked_cases": sum(case["status"] == "blocked_by_upstream" for case in cases),
        "blocked_encodes": sum(item.get("status") == "blocked_by_upstream"
                               for case in cases for item in case["compressions"].values()),
        "blocked_decodes": sum(item.get("status") == "blocked_by_upstream"
                               for case in cases for item in case["decodings"].values()),
        "failures": sum(case["status"] == "failed" for case in cases),
        "four_way_bitwise_equal_cases": sum(case["four_way_reconstruction_bitwise_equal"] for case in cases),
        "harness_errors": int("harness_error" in report),
    }
    if interrupted:
        summary["status"], exit_code = "interrupted", 130
    elif summary["failures"] or summary["harness_errors"] or len(cases) != expected_cases:
        summary["status"], exit_code = "failed", 1
    elif summary["blocked_cases"]:
        summary["status"], exit_code = "blocked_by_upstream", 2
    elif summary["observed_upstream_not_raw_equal"]:
        summary["status"], exit_code = "complete_with_upstream_raw_mismatches", 0
    else:
        summary["status"], exit_code = "passed", 0
    report["summary"] = summary
    save_json(output / "results.json", report)
    print(json.dumps({"results": str(output / "results.json"), "summary": summary}, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
