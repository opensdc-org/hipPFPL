# HIP porting notes

The port replaces CUDA runtime calls with HIP and retains PFPL's CPU code, quantization, coding, and file format. See the [build instructions](../../README.md#build).

PFPL groups lanes in sets of 32. On wave64 devices, the [HIP adapter](../../src/include/hip_warp32.h) maps each physical wave to two logical groups and keeps shuffles and votes within each group. The original cross-block carry protocol is unchanged.
