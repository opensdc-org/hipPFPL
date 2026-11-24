# HIP porting notes

The port replaces CUDA runtime calls with HIP and retains PFPL's quantization rules and file format. The FP32 HIP paths share the block transforms and scheduling helpers across ABS, NOA and REL. See the [build instructions](../../README.md#build).

PFPL groups lanes in sets of 32. On wave64 devices, the [HIP adapter](../../src/include/hip_warp32.h) maps each physical wave to two logical groups and keeps shuffles and votes within each group.

FP32 compression stages up to 4,096 chunks per window. For a single window, ABS, NOA, and REL use the same slot-offset calculation in their fused scan and pack kernels. For larger packets, they use a shared pack kernel and reuse the staging buffer in stream order. When staging allocation runs out of device memory, compression uses the cross-block carry path. The FP64 compressor continues to use that protocol.

The encoders share a generic RZE finalizer. Full FP32 staged chunks can finish the upper bitmap layers once the lower bitmap is ready. ABS publishes the second bitmap at the block synchronization before scattering the payload, so the upper bitmap encoder can run while the other waves write the lower payload. ABS also combines the threshold and NaN checks in a comparison of the original magnitude bits.

FP32 NOA reduces the input minimum and maximum into device scratch. Encoding consumes those values on the same stream. The compressor uses the Thrust reduction if allocating range scratch runs out of device memory.

FP32 decompression checks the packet directory and chunk boundaries on the host before transfer. On the device, it scans the chunk lengths once and uses the resulting offsets as blocks claim work. Full compressed chunks expand their bytes into the shared-memory layout used by the inverse bit transpose. The decoder copies raw fallback chunks directly to the output.

In FP32 ABS, each thread block starts with the chunk at its block index. Subsequent chunks are claimed from a device-wide counter initialized to the grid size during the offset scan. On `gfx936`, ABS packets with at most 2,048 chunks use the wave64 prefix scan to compute offsets within one tile. Larger packets use the tiled scan. For a full compressed chunk, the rank scan recovers upper RZE bitmap bytes on demand. Each lane then passes its reconstructed mask directly to byte expansion in a register, saving another shared-memory round trip and block-wide synchronization.
