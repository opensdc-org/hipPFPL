# hipPFPL

**HIP-based error-bounded lossy compression for scientific data**

[![BSD 3-Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](LICENSE)

hipPFPL compresses FP32 and FP64 arrays on Hygon DCUs. It supports absolute, pointwise relative, and range-relative error bounds, and retains PFPL's command-line interface and file format. The repository also provides serial and OpenMP CPU builds. The HIP implementation is based on [PFPL](https://github.com/burtscher/PFPL).

[Build](#build) · [Quick start](#quick-start) · [Error-bound modes](#error-bound-modes) · [Measuring throughput](#measuring-throughput) · [Tests](#tests) · [Credits and license](#credits-and-license)

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

The FP32 HIP programs support whole 16 KiB chunks and partial final chunks. A packet containing raw fallback chunks can be larger than the input because it also stores a header and a chunk length table.

```bash
mkdir -p artifacts &&
sample=$(mktemp -d artifacts/quickstart.XXXXXX) &&
python3 - "$sample/input.f32" <<'PY' &&
import struct
import sys

values = [((i % 257) - 128) / 32 for i in range(4096)]
with open(sys.argv[1], "wb") as output:
    output.write(struct.pack("<4096f", *values))
PY
bin/f32/hip/f32_abs_compress_hip "$sample/input.f32" "$sample/compressed.pfpl" 0.01 &&
test -s "$sample/compressed.pfpl" &&
bin/f32/hip/f32_abs_decompress_hip "$sample/compressed.pfpl" "$sample/reconstructed.f32"
```

Use the `f64` executables for FP64 input. The compressor takes `INPUT COMPRESSED ERROR_BOUND [THRESHOLD]`. The matching decoder takes `COMPRESSED OUTPUT`.

The legacy FP64 and CPU file paths have buffer limits for raw fallback packets. For those paths, use a partial final chunk and check that the packet is at most the size of the original array before passing it to the decoder.

## Error-bound modes

| Mode or parameter | Meaning |
| --- | --- |
| `abs` | Absolute error bound on each value. |
| `rel` | Pointwise relative error using PFPL's multiplicative magnitude interval. |
| `noa` | Absolute error bound scaled by the input range. The packet stores the resulting absolute bound. |
| `THRESHOLD` | Optional positive, normal magnitude cutoff for values to preserve exactly. In NOA it must exceed the scaled bound. |

## Measuring throughput

The file executables print nine phase timings in seconds under `lc comp ecltime` or `lc decomp ecltime`. Divide the original input byte count by the phase time and by `10^9` to report decimal GB/s. The HIP NOA compression interval includes the input min/max reduction.

For device-resident round-trip throughput, time compression followed by decompression of the generated packet in device memory. Count the original input bytes once and state whether the interval includes synchronization and packet-size readbacks.

### FP32 results on BW150

These measurements use one Hygon BW150 (`gfx936`). Each throughput cell lists **compression / decompression / round trip** in decimal GB/s.

| Dataset | ABS error bound (rounded) | ABS | NOA | REL |
| --- | ---: | ---: | ---: | ---: |
| QAOA (32 qubits) | `9.15e-8` | 146.48 / 157.99 / 76.14 | 111.86 / 136.53 / 61.63 | 111.90 / 105.18 / 54.19 |
| GHZ + inverse QFT (21 qubits) | `4.14e-6` | 150.17 / 160.68 / 77.66 | 110.49 / 138.57 / 61.45 | 116.80 / 108.52 / 56.37 |
| CESM-ATM surface temperature | `0.08` | 206.06 / 212.40 / 104.61 | 147.97 / 188.36 / 82.99 | 157.69 / 139.62 / 74.06 |

- **QAOA.** A state-vector sample captured during a 32-qubit QAOA simulation on a 3-regular graph with one QAOA layer (`p = 1`).
- **GHZ + inverse QFT.** The 21-qubit state vector obtained by applying the inverse quantum Fourier transform to a GHZ state. The amplitudes were computed on the CPU and stored in natural basis order.
- **CESM-ATM.** The `TS_1_1800_3600.dat` surface temperature field from the cleared CESM-ATM dataset. It contains a 1,800 × 3,600 grid of little-endian FP32 values, or 25,920,000 bytes. [SDRBench dataset description](https://sdrbench.github.io/datasets.html#cesm-atm)

The two quantum inputs are about 16 MiB each. The ABS results are from the post-cleanup build. The NOA and REL results are from earlier runs using `0.01` as the CLI error parameter.

Throughput uses median steady-state timings with data resident on the device. Each phase includes synchronization and a 4-byte length readback. NOA compression includes the input min/max reduction.

## Tests

Load the DTK environment and set `device` and `pci` to your assigned device index and PCI bus ID. Fetch the upstream reference once, then run the checks:

```bash
git fetch --no-tags https://github.com/burtscher/PFPL.git 36f5aaef42744ed78b4d1525b03a7fb2168b363c
make check HIP_TEST_DEVICE="$device" HIP_TEST_PCI="$pci"
```

The smoke suite covers FP32 and FP64 in all three modes. The default output directory is `artifacts/check`. Use `CHECK_DIR` to select a fresh directory inside this checkout or under `/tmp`. See the [validation guide](tools/validation/README.md) for extended and component checks and the [HIP porting notes](docs/porting/hip-adaptation.md) for the implementation details.

## Credits and license

hipPFPL is maintained in the [OpenSDC community](https://github.com/opensdc-org). For bug reports or integration questions, [open an issue](https://github.com/opensdc-org/hipPFPL/issues).

For the underlying algorithm, cite Alex Fallin, Noushin Azami, Sheng Di, Franck Cappello, and Martin Burtscher, *Fast and Effective Lossy Compression on GPUs and CPUs with Guaranteed Error Bounds*, IPDPS 2025. [DOI: 10.1109/IPDPS64566.2025.00083](https://doi.org/10.1109/IPDPS64566.2025.00083).

hipPFPL is based on [PFPL at `36f5aae`](https://github.com/burtscher/PFPL/tree/36f5aaef42744ed78b4d1525b03a7fb2168b363c) and distributed under the [BSD 3-Clause License](LICENSE).
