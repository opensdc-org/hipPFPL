# Tests

Use Linux with Python 3.9 or later. See [`make check`](../../README.md#tests) for the routine checks and reference fetch. The commands below run individual checks from the repository root. Use fresh output directories.

## Build and check the upstream CPU reference

After fetching the reference commit, run the original serial and OpenMP programs against the same inputs:

```sh
python3 -B tools/validation/build_reference.py --output artifacts/cpu-reference
python3 -B tools/validation/reference_suite.py \
  --manifest artifacts/cpu-reference/manifest.json \
  --output artifacts/cpu-suite
```

The fixtures use partial final chunks. See `artifacts/cpu-suite/results.json` for individual outcomes.

## Check HIP primitives, components, and carry

Load the DTK environment and set `device` and `pci` as in [Tests](../../README.md#tests). Run any of these three checks:

```bash
mkdir -p artifacts/hip-tests

check_hip() {
  local source_file="tools/validation/hip_${1}_test.hip"
  local binary="artifacts/hip-tests/$1"
  "${HIPCC:-hipcc}" "$2" -std=c++17 --offload-arch="${HIP_ARCH:-gfx936}" \
    -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffp-contract=off -Isrc \
    "$source_file" -o "$binary" &&
  python3 -B tools/validation/run_hip_check.py \
    --binary "$binary" --source "$source_file" --output "$binary-run" \
    --device "$device" --expected-pci "$pci" --timeout 60
}

check_hip primitives -O2
check_hip components -O2
check_hip carry -O3
```

Each check reports its status and verifies the selected device.

## Extended file comparisons

Build the CPU reference and run its checks above. Then build the HIP programs and generate their manifest:

```sh
python3 -B tools/validation/build_hip_manifest.py --output artifacts/hip-build
```

In Bash, run any of these comparisons:

```bash
hip_args=(--hip-manifest artifacts/hip-build/manifest.json
          --device "$device" --expected-pci "$pci")
cpu_args=(--reference-manifest artifacts/cpu-reference/manifest.json)

python3 -B tools/validation/hip_equivalence_suite.py "${hip_args[@]}" \
  --reference-results artifacts/cpu-suite/results.json \
  --dtype f32 --mode abs --output artifacts/hip-equivalence
python3 -B tools/validation/extended_equivalence_suite.py "${hip_args[@]}" "${cpu_args[@]}" \
  --profile mixed --output artifacts/hip-mixed
```

The comparisons check defined header fields, chunk lengths, payload and reconstructed values. FP32 header bytes 12–15 are unassigned by PFPL and reported separately. For a CUDA reference build, use the [upstream instructions](https://github.com/burtscher/PFPL/blob/36f5aaef42744ed78b4d1525b03a7fb2168b363c/README.md#installation).
