# hipPFPL

HIP port of [PFPL](https://github.com/burtscher/PFPL) for error-bounded compression of FP32 and FP64 arrays on Hygon DCUs. It retains the original command-line interface and file format.

[Build](#build) · [Quick start](#quick-start) · [Tests](#tests) · [Credits and license](#credits-and-license)

## Build

| Target | Toolchain |
| --- | --- |
| Hygon BW150 | Linux, DTK 26.04, HIP compiler and HIP Thrust |
| CPU (serial or OpenMP) | Linux, GCC with C++17 and OpenMP |

Load the DTK environment for your installation, then run:

```sh
source /opt/dtk/env.sh
make -j1 hip
```

The [makefile](makefile) defaults to `HIP_ARCH=gfx936`. HIP executables go in `bin/f32/hip/` and `bin/f64/hip/`. Use `make -j1 serial openmp` for the CPU executables.

## Quick start

Use a nonempty, raw little-endian IEEE 754 array of complete, finite FP32 or FP64 values. Use inputs with a partial final 16 KiB chunk so uncompressed fallback fits the compressor buffer. Before decoding, check that the compressed stream is no larger than the original array. The HIP programs use the first visible device.

```sh
mkdir -p artifacts &&
sample=$(mktemp -d artifacts/quickstart.XXXXXX) &&
python3 - "$sample/input.f32" <<'PY' &&
import struct
import sys

values = [((i % 257) - 128) / 32 for i in range(2048)]
with open(sys.argv[1], "wb") as output:
    output.write(struct.pack("<2048f", *values))
PY
bin/f32/hip/f32_abs_compress_hip "$sample/input.f32" "$sample/compressed.pfpl" 0.01 &&
test -s "$sample/compressed.pfpl" &&
test "$(wc -c < "$sample/compressed.pfpl")" -le "$(wc -c < "$sample/input.f32")" &&
bin/f32/hip/f32_abs_decompress_hip "$sample/compressed.pfpl" "$sample/reconstructed.f32"
```

Use the `f64` programs for FP64 input. The compressor takes `INPUT COMPRESSED ERROR_BOUND [THRESHOLD]`. The matching decoder takes `COMPRESSED OUTPUT`.

| Mode or parameter | Meaning |
| --- | --- |
| `abs` | Absolute error bound on each value. |
| `rel` | Pointwise relative error using PFPL's multiplicative magnitude interval. |
| `noa` | Absolute error bound scaled by the input range. The packet stores the resulting absolute bound. |
| `THRESHOLD` | Optional positive, normal magnitude cutoff for values to preserve exactly. In NOA it must exceed the scaled bound. |

## Tests

Load the DTK environment and set `device` and `pci` to your assigned device index and PCI bus ID. Fetch the upstream reference once, then run the checks:

```sh
git fetch --no-tags https://github.com/burtscher/PFPL.git 36f5aaef42744ed78b4d1525b03a7fb2168b363c
make check HIP_TEST_DEVICE="$device" HIP_TEST_PCI="$pci"
```

The checks cover FP32 and FP64 in all three modes. The default output directory is `artifacts/check`. Use `CHECK_DIR` to select a fresh directory inside this checkout or under `/tmp`. See the [test instructions](tools/validation/README.md) for individual checks.

## Credits and license

This port is based on [PFPL at `36f5aae`](https://github.com/burtscher/PFPL/tree/36f5aaef42744ed78b4d1525b03a7fb2168b363c) and is distributed under the [BSD 3-Clause license](LICENSE).

Please cite Alex Fallin, Noushin Azami, Sheng Di, Franck Cappello, and Martin Burtscher, *Fast and Effective Lossy Compression on GPUs and CPUs with Guaranteed Error Bounds*, IPDPS 2025. [DOI](https://doi.org/10.1109/IPDPS64566.2025.00083).
