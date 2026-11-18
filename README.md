# hipPFPL

**HIP-based error-bounded lossy compression for scientific data**

[![BSD 3-Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](LICENSE)

hipPFPL compresses FP32 and FP64 arrays on Hygon DCUs. It supports absolute, pointwise relative, and range-relative error bounds, and retains PFPL's command-line interface and file format. The repository also provides serial and OpenMP CPU builds. The HIP implementation is based on [PFPL](https://github.com/burtscher/PFPL).

[Build](#build) · [Quick start](#quick-start) · [Error-bound modes](#error-bound-modes) · [Tests](#tests) · [Credits and license](#credits-and-license)

## Build

| Hardware | Toolchain |
| --- | --- |
| Hygon BW150 | Linux, DTK 26.04, HIP compiler and HIP Thrust |
| CPU (serial or OpenMP) | Linux, GCC with C++17 and OpenMP |

Clone the OpenSDC repository and load the DTK environment for your installation. The example uses `/opt/dtk/env.sh`.

```bash
git clone https://github.com/opensdc-org/hipPFPL.git
cd hipPFPL
source /opt/dtk/env.sh
make -j1 hip
```

The [makefile](makefile) defaults to `HIP_ARCH=gfx936` for the Hygon BW150. HIP executables are written to `bin/f32/hip/` and `bin/f64/hip/`. Build the CPU executables with `make -j1 serial openmp`.

## Quick start

The CLI reads a nonempty raw little-endian IEEE 754 array of complete, finite FP32 or FP64 values. For a quick round trip, generate a small FP32 input and compress it with an absolute error bound of `0.01`. The HIP executables use the first visible device.

Use an input with a partial final 16 KiB chunk so the uncompressed fallback fits the compressor buffer. The example checks the stream size against the input size before decoding.

```bash
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

Use the `f64` executables for FP64 input. The compressor takes `INPUT COMPRESSED ERROR_BOUND [THRESHOLD]`. The matching decoder takes `COMPRESSED OUTPUT`.

## Error-bound modes

| Mode or parameter | Meaning |
| --- | --- |
| `abs` | Absolute error bound on each value. |
| `rel` | Pointwise relative error using PFPL's multiplicative magnitude interval. |
| `noa` | Absolute error bound scaled by the input range. The packet stores the resulting absolute bound. |
| `THRESHOLD` | Optional positive, normal magnitude cutoff for values to preserve exactly. In NOA it must exceed the scaled bound. |

## Tests

Load the DTK environment and set `device` and `pci` to your assigned device index and PCI bus ID. Fetch the upstream reference once, then run the checks:

```bash
git fetch --no-tags https://github.com/burtscher/PFPL.git 36f5aaef42744ed78b4d1525b03a7fb2168b363c
make check HIP_TEST_DEVICE="$device" HIP_TEST_PCI="$pci"
```

The checks cover FP32 and FP64 in all three modes. The default output directory is `artifacts/check`. Use `CHECK_DIR` to select a fresh directory inside this checkout or under `/tmp`. See the [validation guide](tools/validation/README.md) for individual checks and the [HIP porting notes](docs/porting/hip-adaptation.md) for the implementation details.

## Credits and license

hipPFPL is maintained in the [OpenSDC community](https://github.com/opensdc-org). For bug reports or integration questions, [open an issue](https://github.com/opensdc-org/hipPFPL/issues).

For the underlying algorithm, cite Alex Fallin, Noushin Azami, Sheng Di, Franck Cappello, and Martin Burtscher, *Fast and Effective Lossy Compression on GPUs and CPUs with Guaranteed Error Bounds*, IPDPS 2025. [DOI: 10.1109/IPDPS64566.2025.00083](https://doi.org/10.1109/IPDPS64566.2025.00083).

hipPFPL is based on [PFPL at `36f5aae`](https://github.com/burtscher/PFPL/tree/36f5aaef42744ed78b4d1525b03a7fb2168b363c) and distributed under the [BSD 3-Clause License](LICENSE).
