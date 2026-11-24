#!/usr/bin/env python3
"""Validate extended PFPL file cases, including optional quantum inputs."""

import argparse
from array import array
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import struct
import subprocess
import sys

import hip_equivalence_suite as equivalence
import reference_suite as reference


ROOT = equivalence.ROOT
SEED = 20261027
MIN_NORMAL = {"f32": "1.1754943508222875e-38", "f64": "2.2250738585072014e-308"}
QUANTUM_BOUNDS = {"qaoa": "9.15483724384103e-8", "ghz_iqft": "4.1429989323660266e-6"}


def make_case(profile, dtype, mode, dataset=None):
    bound = "0.01"
    threshold = None
    if profile == "mixed":
        bound = ({"f32": "1e36", "f64": "1e300"}[dtype] if mode == "abs"
                 else "0" if mode == "noa" else "0.01")
        threshold = MIN_NORMAL[dtype] if mode != "noa" else None
    if dataset and mode == "abs":
        bound = QUANTUM_BOUNDS[dataset]
    return {"id": "_".join((dataset or profile, dtype, mode)), "profile": profile, "dtype": dtype, "mode": mode,
            "dataset": dataset,
            "error_bound_token": bound, "threshold_token": threshold, "status": "not_run",
            "compressions": {}, "decodings": {}, "stream_comparisons": {}, "issues": [], "failures": [],
            "five_way_reconstruction_bitwise_equal": False}


def make_input(profile, dtype):
    code, width = reference.DTYPES[dtype]
    if profile == "large":
        count = (8 * 1024 * 1024) // width + 64
        return array(code, (((index % 257) - 128) / 32 for index in range(count))).tobytes()
    constant = struct.pack("<" + code, 3.25)
    data = bytearray(constant * (reference.CHUNK_BYTES // width))
    generator = random.Random(SEED)
    mantissa_bits, exponent_max = (23, 252) if dtype == "f32" else (52, 2044)
    integer_code = "I" if dtype == "f32" else "Q"
    for _ in range(2 * reference.CHUNK_BYTES // width):
        exponent = generator.randint(1, exponent_max)
        sign = generator.getrandbits(1)
        mantissa = generator.getrandbits(mantissa_bits)
        bits = (sign << (8 * width - 1)) | (exponent << mantissa_bits) | mantissa
        data.extend(struct.pack("<" + integer_code, bits))
    data.extend(constant * 64)
    return bytes(data)


def preflight(args, report):
    report["revision"] = subprocess.check_output(
        ["git", "-c", "safe.directory=" + str(ROOT), "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    equivalence.load_hip_manifest(args.hip_manifest, report)
    if args.reference_manifest is None:
        return
    manifest, _ = reference.validate_manifest(args.reference_manifest)
    equivalence.require(manifest.get("original_num_runs") == 9, "CPU manifest must retain nine internal runs")
    report["reference_manifest"] = reference.file_record(args.reference_manifest)
    report["cpu_binaries"] = {variant: {key: equivalence.verify_file(item, executable=True)
                                         for key, item in entries.items()}
                              for variant, entries in manifest["binaries"].items()}
    report["cpu_sources"] = [equivalence.verify_file({"path": str(Path(manifest["source_directory"]) / name),
                                                      "sha256": digest})
                             for name, digest in sorted(manifest["source_sha256"].items())]


def invoke(binary, arguments, inputs, directory, label, operation, args, report, destination):
    invocation = equivalence.invoke(binary, arguments, inputs, directory, label, operation, args, report, destination)
    if operation:
        samples = invocation["internal_timing_samples_seconds"]
        invocation["timing_summary_seconds"] = {"median": statistics.median(samples), "min": min(samples),
                                                 "sample_count": len(samples), "discarded_samples": 0}
    return invocation


def payload_coverage(layout, input_size):
    offset = layout["payload_offset"]
    covered = []
    for index, length in enumerate(layout["chunk_lengths"]):
        if length >= 128:
            covered.append({"chunk": index, "length": length, "file_offset": offset, "mod4": offset % 4})
        offset += length
    return {"full_chunk_raw_fallback_indices": [index for index in layout["raw_fallback_chunks"]
                                                if (index + 1) * reference.CHUNK_BYTES <= input_size],
            "chunks_ge128_count": len(covered), "file_payload_offset_mod4_ge128": sorted({row["mod4"] for row in covered}),
            "chunks_ge128": covered}


def blocked(case, issue, detail):
    case["status"] = "blocked_by_upstream"
    case["issues"].append({"id": issue, "status": "blocked_by_upstream", "detail": detail})
    case["decodings"] = {label: {"status": "blocked_by_upstream", "issue": issue}
                          for label in ("serial_to_serial", *equivalence.PATHS)}


def run_hip_case(case, args, report, directory):
    mode = case["mode"]
    packet = directory / "hip.pfpl"
    decoded = directory / "hip_to_hip.decoded.bin"
    oversized = case["profile"] == "oversized"
    if oversized:
        # A legal packet with one full raw chunk and a raw tail, plus the directory overhead.
        tail = struct.pack("<ff", 1.25, -2.5)
        original = tail * (reference.CHUNK_BYTES // len(tail) + 1)
    else:
        input_path = getattr(args, case["dataset"])
        original = input_path.read_bytes()
        equivalence.require(len(original) > reference.CHUNK_BYTES and len(original) % 8 == 0,
                            "Expected a multi-chunk, interleaved FP32 complex state vector")
        case["input"] = reference.file_record(input_path)
        invoke(report["hip_binaries"]["f32_" + mode + "_comp"],
               [input_path, packet, case["error_bound_token"]], [case["input"]], directory,
               "hip", "comp", args, report, case["compressions"])

    expected = reference.expected_header_parameter(original, "f32", mode, case["error_bound_token"])
    if oversized:
        packet.write_bytes(struct.pack("<qf4xHH", len(original), expected["value"],
                                       reference.CHUNK_BYTES, len(tail)) + original)
    case["packet"] = reference.file_record(packet)
    layout = reference.parse_stream(packet.read_bytes(), "f32", len(original))
    case["layout"] = layout
    equivalence.require(layout["stored_error_bound_hex"] == expected["hex"],
                        "HIP header violates the error-bound contract")
    if oversized:
        equivalence.require(case["packet"]["size_bytes"] > len(original) and layout["raw_fallback_chunks"] == [0, 1],
                            "Expected an oversized packet containing only raw chunks")
    else:
        case["compressions"]["hip"]["output"] = case["packet"]
    invocation = invoke(report["hip_binaries"]["f32_" + mode + "_decomp"], [packet, decoded],
                        [case["packet"]], directory, "hip_to_hip", "decomp", args, report, case["decodings"])
    invocation["output"] = reference.file_record(decoded)
    reconstructed = decoded.read_bytes()
    if oversized:
        equivalence.require(reconstructed == original, "Raw chunks must be reconstructed bit for bit")
    else:
        invocation["numeric_check"] = reference.check_numerics(
            original, reconstructed, "f32", mode, layout["stored_error_bound"], None)
        equivalence.require(invocation["numeric_check"]["ok"], "HIP reconstruction violates the error bound")
    invocation["status"] = case["status"] = "passed"


def run_case(case, args, report):
    directory = args.output / case["id"]
    directory.mkdir()
    dtype, mode, profile = case["dtype"], case["mode"], case["profile"]
    key = dtype + "_" + mode
    try:
        if profile in ("oversized", "quantum"):
            run_hip_case(case, args, report, directory)
            return
        original = make_input(profile, dtype)
        input_path = directory / "input.bin"
        input_path.write_bytes(original)
        case["input"] = reference.file_record(input_path)
        case["elements"] = len(original) // reference.DTYPES[dtype][1]
        case["chunk_count"] = (len(original) + reference.CHUNK_BYTES - 1) // reference.CHUNK_BYTES
        equivalence.require(case["chunk_count"] == (4 if profile == "mixed" else 513), "Unexpected fixture chunk count")
        case["encoder_capacity_preflight"] = equivalence.capacity(len(original))
        if not case["encoder_capacity_preflight"]["safe_upper_bound"]:
            blocked(case, "U001", "Original encoder allocation cannot contain the worst-case stream")
            return
        case["expected_header_parameter"] = reference.expected_header_parameter(original, dtype, mode, case["error_bound_token"])
        case["generator"] = ({"name": "constant/random-normal-bits/constant", "seed": SEED,
                               "random_exponent_inclusive": [1, 252 if dtype == "f32" else 2044],
                               "random_chunks": [1, 2], "constant": 3.25, "tail_elements": 64}
                              if profile == "mixed" else {"name": "staircase", "formula": "((i % 257) - 128) / 32",
                                                          "base_bytes": 8 * 1024 * 1024, "tail_elements": 64})
        streams, layouts, records = {}, {}, {}
        for backend in ("serial", "openmp", "hip"):
            binary = report["hip_binaries"][key + "_comp"] if backend == "hip" else report["cpu_binaries"][backend][key + "_comp"]
            output = directory / (backend + ".pfpl")
            arguments = [input_path, output, case["error_bound_token"]]
            if case["threshold_token"] is not None:
                arguments.append(case["threshold_token"])
            invocation = invoke(binary, arguments, [case["input"]], directory, backend, "comp", args, report, case["compressions"])
            invocation["output"] = reference.file_record(output)
            stream = output.read_bytes()
            layout = reference.parse_stream(stream, dtype, len(original))
            invocation["layout"] = layout
            invocation["header_parameter_matches_expected"] = layout["stored_error_bound_hex"] == case["expected_header_parameter"]["hex"]
            equivalence.require(invocation["header_parameter_matches_expected"], backend + " violates the header parameter contract")
            invocation["encoded_not_larger_than_original"] = len(stream) <= len(original)
            invocation["compression_ratio"] = len(original) / len(stream)
            invocation["coverage"] = payload_coverage(layout, len(original))
            streams[backend], layouts[backend], records[backend] = stream, layout, invocation["output"]
            if backend == "openmp":
                equivalence.stream_compare(case, "serial_vs_openmp", streams["serial"], stream, (layouts["serial"], layout))
            elif backend == "hip":
                for cpu in reference.VARIANTS:
                    equivalence.stream_compare(case, "hip_vs_" + cpu, stream, streams[cpu], (layout, layouts[cpu]))

        if profile == "mixed":
            coverage = case["compressions"]["hip"]["coverage"]
            equivalence.require(coverage["full_chunk_raw_fallback_indices"], "HIP did not exercise a complete raw-fallback chunk")
            if dtype == "f32" and mode == "abs":
                equivalence.require(3 in coverage["file_payload_offset_mod4_ge128"],
                                    "HIP did not exercise the expected offset mod4=3 store path")

        oracle = None
        for label in ("serial_to_serial", *equivalence.PATHS):
            encoder, decoder = label.split("_to_")
            if len(streams[encoder]) > len(original) and (decoder != "hip" or dtype != "f32"):
                case["decodings"][label] = {"status": "blocked_by_upstream", "issue": "U004"}
                case["issues"].append({"id": "U004", "status": "blocked_by_upstream", "comparison": label,
                                       "detail": "An oversized packet exceeds the legacy decoder's input buffer."})
                continue
            binary = report["hip_binaries"][key + "_decomp"] if decoder == "hip" else report["cpu_binaries"][decoder][key + "_decomp"]
            output = directory / (label + ".decoded.bin")
            invocation = invoke(binary, [records[encoder]["path"], output], [records[encoder]], directory,
                                label, "decomp", args, report, case["decodings"])
            invocation["output"] = reference.file_record(output)
            data = output.read_bytes()
            if oracle is None:
                invocation["numeric_check"] = reference.check_numerics(
                    original, data, dtype, mode, layouts[encoder]["stored_error_bound"],
                    float(case["threshold_token"]) if case["threshold_token"] is not None else None)
                equivalence.require(invocation["numeric_check"]["ok"], label + " fails the numerical contract")
                if label == "serial_to_serial":
                    oracle = data
                if profile == "mixed":
                    case["mixed_reconstruction_equals_input"] = reference.compare_bytes(original, data)
                    equivalence.require(case["mixed_reconstruction_equals_input"]["equal"], "Mixed threshold/zero-bound contract did not preserve input bits")
            else:
                invocation["reference_comparison"] = reference.compare_bytes(oracle, data)
                equivalence.require(invocation["reference_comparison"]["equal"], label + " differs from the independent CPU oracle")
                invocation["numeric_check"] = {"ok": True, "reused_from": "serial_to_serial",
                                                "reason": "Exact oracle bytes and identical independently validated error-bound bits"}
            invocation["status"] = "passed"
        if any(item["status"] == "blocked_by_upstream" for item in case["decodings"].values()):
            case["status"] = "blocked_by_upstream"
            return
        case["five_way_reconstruction_bitwise_equal"] = (oracle is not None and len(case["decodings"]) == 6 and
            all(case["decodings"][label].get("status") == "passed" for label in equivalence.PATHS))
        equivalence.require(case["five_way_reconstruction_bitwise_equal"], "Incomplete six-decoder/oracle comparison")
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
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--hip-manifest", type=Path, required=True)
    parser.add_argument("--profile", choices=("mixed", "large", "oversized", "quantum", "all"), default="all")
    parser.add_argument("--qaoa", type=Path, help="Raw QAOA FP32 state vector for the quantum profile")
    parser.add_argument("--ghz-iqft", type=Path, help="Raw GHZ + inverse QFT FP32 state vector for the quantum profile")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--expected-pci", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, choices=(20, 30), default=30)
    args = parser.parse_args()
    if args.profile in ("mixed", "large", "all") and args.reference_manifest is None:
        parser.error("This profile requires --reference-manifest")
    if args.profile == "quantum" and (args.qaoa is None or args.ghz_iqft is None):
        parser.error("The quantum profile requires --qaoa and --ghz-iqft")
    try:
        equivalence.require(args.device >= 0 and sys.byteorder == "little", "Require nonnegative GPU index and little-endian host")
        args.expected_pci = equivalence.pci_address(args.expected_pci)
        if args.reference_manifest is not None:
            args.reference_manifest = args.reference_manifest.resolve(strict=True)
        if args.profile == "quantum":
            args.qaoa = args.qaoa.resolve(strict=True)
            args.ghz_iqft = args.ghz_iqft.resolve(strict=True)
            equivalence.require(not args.qaoa.samefile(args.ghz_iqft), "Supply two distinct quantum inputs")
        args.hip_manifest = args.hip_manifest.resolve(strict=True)
        args.output = args.output.resolve()
        equivalence.require(args.output.is_relative_to(ROOT) or args.output.is_relative_to(Path("/tmp")),
                            "Output must remain inside this checkout or /tmp")
        args.output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.profile == "quantum":
        cases = [make_case("quantum", "f32", mode, dataset) for dataset in QUANTUM_BOUNDS for mode in reference.MODES]
    else:
        profiles = ("mixed", "large", "oversized") if args.profile == "all" else (args.profile,)
        cases = [make_case(profile, dtype, mode) for profile in profiles
                 for dtype in (("f32",) if profile == "oversized" else reference.DTYPES) for mode in reference.MODES]
    report = {"schema_version": 1, "baseline_sha": reference.BASELINE, "profile": args.profile,
              "suite_command": [sys.executable, *sys.argv], "python": sys.version,
              "harnesses": [reference.file_record(Path(module).resolve()) for module in (__file__, reference.__file__, equivalence.__file__)],
              "device": args.device, "expected_pci": args.expected_pci, "probe": {}, "cases": cases, "errors": [],
              "expected_case_count": len(cases), "internal_runs_unchanged": 9,
              "timing_scope": "File CLI phase timings, measured separately from device-resident round trips",
              "timeout_seconds": args.timeout, "started_utc": datetime.now(timezone.utc).isoformat()}
    interrupted = False
    try:
        preflight(args, report)
        reference.save_json(args.output / "preflight.json", report)
        probe = equivalence.runtime_preflight(args, report)
        report["production_launch_geometry"] = {
            "threads_per_block": 512,
            "logical_warp_size": 32,
            "device_properties": probe["device_properties"],
            "note": "The selected codec and execution path determine grid size and shared memory usage.",
        }
        for case in cases:
            print("BEGIN " + case["id"], file=sys.stderr, flush=True)
            run_case(case, args, report)
            print(case["status"].upper() + " " + case["id"], file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        interrupted = True
        report["errors"].append("Interrupted; no further GPU processes were started")
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + ": " + str(exc))
    comparisons = [item for case in cases for item in case["stream_comparisons"].values()]
    summary = {"expected_cases": len(cases), "case_count": len(cases),
               "passed_cases": sum(case["status"] == "passed" for case in cases),
               "failed_cases": sum(case["status"] == "failed" for case in cases),
               "blocked_cases": sum(case["status"] == "blocked_by_upstream" for case in cases),
               "not_run_cases": sum(case["status"] == "not_run" for case in cases),
               "observed_upstream_not_raw_equal": sum(case["status"] == "observed_upstream_not_raw_equal" for case in cases),
               "raw_stream_comparisons": len(comparisons), "raw_stream_mismatches": sum(not item["equal"] for item in comparisons),
               "u002_observations": sum(issue["id"] == "U002" for case in cases for issue in case["issues"]),
               "independent_cpu_oracles": sum(case["decodings"].get("serial_to_serial", {}).get("status") == "passed" for case in cases),
               "five_way_bitwise_equal_cases": sum(case["five_way_reconstruction_bitwise_equal"] for case in cases),
               "harness_errors": len(report["errors"])}
    failed = bool(report["errors"] or summary["failed_cases"] or summary["not_run_cases"])
    status = "failed" if failed else "blocked_by_upstream" if summary["blocked_cases"] else (
        "complete_with_upstream_raw_mismatches" if summary["raw_stream_mismatches"] else "passed")
    report["status"] = summary["status"] = "interrupted" if interrupted else status
    report["summary"] = summary
    reference.save_json(args.output / "results.json", report)
    print(json.dumps({"results": str(args.output / "results.json"), "summary": summary}, sort_keys=True))
    return 130 if interrupted else 1 if failed else 2 if summary["blocked_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
