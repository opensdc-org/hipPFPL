/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2025, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#define NDEBUG

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#define WS 32


#include <string>
#include <cmath>
#include <climits>
#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include "include/macros.h"
#include "include/sum_reduction.h"
#include "include/chunk_offsets.h"
#include "components/d_BIT_4.h"
#define PFPL_F32_ABS_RZE_ENCODE
#include "components/d_RZE_1.h"
#undef PFPL_F32_ABS_RZE_ENCODE
#define PFPL_ABS_EARLY_RZE_UPPER
#include "include/f32_encode_common.h"
#undef PFPL_ABS_EARLY_RZE_UPPER
#include "include/f32_file_io.h"


const int NUM_RUNS = 9;


// Merge the magnitude threshold and the raw-NaN check into one comparison.
static __device__ __forceinline__ unsigned int d_QABS_4_cutoff(float threshold)
{
  constexpr unsigned int infinity = 0x7f800000u;
  const unsigned int bits = __float_as_uint(threshold);
  const unsigned int magnitude = bits & 0x7fffffffu;
  if (magnitude > infinity) return infinity + 1;
  if ((bits >> 31) != 0) return 0;
  // A device that compares subnormals as zero already rejects every value.
  if (magnitude < 0x00800000u && threshold == 0.0f) return 0;
  return magnitude;
}


static __device__ __forceinline__ unsigned int d_QABS_4_word(
    unsigned int original_bits, float errorbound, unsigned int cutoff, float eb2, float inv_eb2)
{
  const int maxbin = 1 << 22;  // Leave one mantissa bit for sign.
  const float original = __uint_as_float(original_bits);
  const float scaled = original * inv_eb2;
  const int bin = static_cast<int>(roundf(scaled));
  const float reconstructed = bin * eb2;

  if ((bin >= maxbin) || (bin <= -maxbin) || ((original_bits & 0x7fffffffu) >= cutoff) ||
      (fabsf(original - reconstructed) > errorbound)) {
    return original_bits;
  }
  const unsigned int sign = 0u - static_cast<unsigned int>(bin < 0);
  return (static_cast<unsigned int>(bin) << 1) ^ sign;
}


static __device__ inline void d_QABS_4_in_place(int csize, byte* slab, float errorbound, float threshold)
{
  const int words = csize / sizeof(float);
  const float eb2 = 2 * errorbound;
  const float inv_eb2 = 0.5f / errorbound;
  const unsigned int cutoff = d_QABS_4_cutoff(threshold);
  auto* values = reinterpret_cast<unsigned int*>(slab);

  for (int i = threadIdx.x; i < words; i += TPB) {
    values[i] = d_QABS_4_word(values[i], errorbound, cutoff, eb2, inv_eb2);
  }
}


// Full chunks already have the pair alignment used by load_raw_chunk.
static __device__ inline void load_quantized_full_chunk(const byte* source, byte* slab,
                                                         float errorbound, float threshold)
{
  using Pair = unsigned long long;
  constexpr int pairs_per_thread = CS / (sizeof(Pair) * TPB);
  static_assert(sizeof(Pair) == 8 && sizeof(unsigned int) == 4 && CS % (sizeof(Pair) * TPB) == 0);
  const auto* input_pairs = reinterpret_cast<const Pair*>(source);
  auto* output_pairs = reinterpret_cast<Pair*>(slab);
  const int tid = threadIdx.x;
  const float eb2 = 2 * errorbound;
  const float inv_eb2 = 0.5f / errorbound;
  const unsigned int cutoff = d_QABS_4_cutoff(threshold);

  if ((reinterpret_cast<std::size_t>(source) & 15u) == 0) {
    constexpr int words_per_vector = 4;
    constexpr int vectors_per_thread = CS / (sizeof(uint4) * TPB);
    static_assert(sizeof(uint4) == words_per_vector * sizeof(unsigned int) && CS % (sizeof(uint4) * TPB) == 0);
    const auto* vectors = reinterpret_cast<const uint4*>(source);
    #pragma unroll
    for (int row = 0; row < vectors_per_thread; ++row) {
      const int i = tid + row * TPB;
      const uint4 raw = vectors[i];
      const unsigned int a = d_QABS_4_word(raw.x, errorbound, cutoff, eb2, inv_eb2);
      const unsigned int b = d_QABS_4_word(raw.y, errorbound, cutoff, eb2, inv_eb2);
      const unsigned int c = d_QABS_4_word(raw.z, errorbound, cutoff, eb2, inv_eb2);
      const unsigned int d = d_QABS_4_word(raw.w, errorbound, cutoff, eb2, inv_eb2);
      output_pairs[2 * i] = (static_cast<Pair>(b) << 32) | a;
      output_pairs[2 * i + 1] = (static_cast<Pair>(d) << 32) | c;
    }
    return;
  }

  #pragma unroll
  for (int row = 0; row < pairs_per_thread; ++row) {
    const int i = tid + row * TPB;
    const Pair original = input_pairs[i];
    const unsigned int low = d_QABS_4_word(static_cast<unsigned int>(original), errorbound, cutoff, eb2, inv_eb2);
    const unsigned int high = d_QABS_4_word(static_cast<unsigned int>(original >> 32), errorbound, cutoff, eb2, inv_eb2);
    output_pairs[i] = (static_cast<Pair>(high) << 32) | low;
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode_legacy(const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int* const __restrict__ outsize, int* const __restrict__ fullcarry, int* const next_chunk_counter,
                     const float errorbound, const float threshold)
{
  __shared__ long long chunk[CS / sizeof(long long)];
  __shared__ int scratch[rze_scratch_bytes / sizeof(int)];
  __shared__ int next_chunk;

  auto* slab = reinterpret_cast<byte*>(chunk);
  auto* scratch_data = reinterpret_cast<byte*>(scratch);
  const int tid = threadIdx.x;
  const int chunks = 1 + (insize - 1) / CS;
  auto* head_out = reinterpret_cast<long long*>(output);
  auto* size_out = reinterpret_cast<unsigned short*>(&head_out[2]);
  auto* data_out = reinterpret_cast<byte*>(&size_out[chunks]);

  while (true) {
    if (tid == 0) next_chunk = atomicAdd(next_chunk_counter, 1);
    __syncthreads();
    const int chunk_id = next_chunk;
    if (chunk_id >= chunks) break;
    const int base = chunk_id * CS;
    const int raw_size = min(CS, insize - base);
    const byte* source = &input[base];

    load_raw_chunk(source, slab, raw_size);
    __syncthreads();
    d_QABS_4_in_place(raw_size, slab, errorbound, threshold);
    __syncthreads();
    d_DIFFNB_BIT_4_in_place(raw_size, slab);
    __syncthreads();

    int csize = raw_size;
    const bool good = d_RZE_1_in_place(csize, slab, scratch_data);
    __syncthreads();

    if (!good || (csize >= raw_size)) csize = raw_size;
    propagate_carry(csize, chunk_id, fullcarry, scratch);

    if (tid == 0) size_out[chunk_id] = csize;
    if (csize == raw_size) load_raw_chunk(source, slab, raw_size);
    __syncthreads();

    const int offset = (chunk_id == 0) ? 0 : scratch[0];
    s2g(&data_out[offset], slab, csize);

    if (tid == 0 && chunk_id == chunks - 1) {
      head_out[0] = static_cast<long long>(insize);
      reinterpret_cast<float*>(&head_out[1])[0] = errorbound;
      reinterpret_cast<unsigned int*>(output)[3] = 0;
      const long long total = &data_out[fullcarry[chunk_id]] - output;
      *outsize = (total <= INT_MAX) ? static_cast<int>(total) : 0;
    }
  }
}


static __global__ __launch_bounds__(TPB, 2)
void d_encode_staged(const byte* const __restrict__ input, const int insize,
                     const int first, byte* const __restrict__ staging,
                     unsigned short* const __restrict__ lengths,
                     const float errorbound, const float threshold)
{
  __shared__ uint4 chunk[CS / sizeof(uint4)];
  __shared__ int scratch[rze_scratch_bytes / sizeof(int)];
  auto* slab = reinterpret_cast<byte*>(chunk);

  const int slot = blockIdx.x;
  const int chunk_id = first + slot;
  const int base = chunk_id * CS;
  const int raw_size = min(CS, insize - base);
  int csize = raw_size;
  bool good;
  if (raw_size == CS) {
    load_quantized_full_chunk(&input[base], slab, errorbound, threshold);
    __syncthreads();
    good = d_DIFFNB_BIT_ZE_4_full_chunk(csize, slab, scratch);
  } else {
    load_raw_chunk(&input[base], slab, raw_size);
    __syncthreads();
    d_QABS_4_in_place(raw_size, slab, errorbound, threshold);
    __syncthreads();
    d_DIFFNB_BIT_4_in_place(raw_size, slab);
    __syncthreads();
    good = d_RZE_1_in_place(csize, slab, reinterpret_cast<byte*>(scratch));
    __syncthreads();
  }
  if (!good || csize >= raw_size) csize = raw_size;
  if (threadIdx.x == 0) lengths[chunk_id] = static_cast<unsigned short>(csize);

  // Raw fallback is packed from the original input, so its slot is unused.
  if (csize < raw_size) {
    byte* slot_data = &staging[slot * CS];
    // Both the shared chunk and the hipMalloc staging slots are 16-byte aligned.
    static_assert(sizeof(uint4) == 16 && CS % sizeof(uint4) == 0);
    constexpr int width = sizeof(uint4);
    const auto* source = reinterpret_cast<const uint4*>(slab);
    auto* destination = reinterpret_cast<uint4*>(slot_data);
    const int tid = threadIdx.x;
    const int vectors = csize / width;
    for (int i = tid; i < vectors; i += TPB) destination[i] = source[i];

    // Pad only through the last word that the subsequent pack can load.
    const int tail = vectors * width + tid;
    const int padded = (csize + 3) & ~3;
    if (tail < padded) slot_data[tail] = (tail < csize) ? slab[tail] : 0;
  }
}


static __global__ __launch_bounds__(TPB)
void d_scan_staged_offsets(const unsigned short* const __restrict__ lengths,
                           const int first, const int count, const int chunks,
                           const int insize, const float errorbound,
                           int* const __restrict__ offsets, byte* const output,
                           int* const __restrict__ outsize)
{
  const int directory_bytes = 16 + 2 * chunks;
  // A zero from an earlier window is sticky. A prior submission is ignored
  // when starting a new packet, so no outsize reset kernel is required.
  if (first != 0 && *outsize <= 0) return;
  int payload_bytes = (first == 0) ? 0 : *outsize - directory_bytes;

  static_assert(STAGED_CHUNK_LIMIT % 4 == 0);
  __shared__ unsigned int scan_scratch[WS];
  __shared__ unsigned int tile_total;
  payload_bytes = scan_chunk_offsets<true>(lengths + first, count, offsets, payload_bytes,
                                           scan_scratch, tile_total);

  if (threadIdx.x == 0) {
    if (first == 0) {
      reinterpret_cast<long long*>(output)[0] = insize;
      reinterpret_cast<float*>(output)[2] = errorbound;
      reinterpret_cast<unsigned int*>(output)[3] = 0;
    }
    const long long total = static_cast<long long>(directory_bytes) + payload_bytes;
    *outsize = (total <= INT_MAX) ? static_cast<int>(total) : 0;
  }
}


// A block sums the completed lengths preceding its slots independently.
// This adds bounded reads of the directory and needs no inter-block ordering.
static __global__ __launch_bounds__(TPB)
void d_scan_pack_staged(const byte* const __restrict__ input, const int insize,
                        const int chunks, const byte* const __restrict__ staging,
                        const unsigned short* const __restrict__ lengths,
                        byte* const output, int* const __restrict__ outsize, float errorbound)
{
  constexpr int lengths_per_word = 4;
  static_assert(TPB % PACK_THREADS_PER_SLOT == 0);
  static_assert(PACK_SLOTS_PER_BLOCK % lengths_per_word == 0 && PACK_SLOTS_PER_BLOCK <= WS);
  static_assert((static_cast<long long>(CS) + 2) * STAGED_CHUNK_LIMIT + 16 <= INT_MAX);
  __shared__ unsigned int reduce_scratch[WS];
  __shared__ int local_offsets[PACK_SLOTS_PER_BLOCK + 1];

  const int tid = threadIdx.x;
  const int block_start = blockIdx.x * PACK_SLOTS_PER_BLOCK;
  const auto* length_words = reinterpret_cast<const unsigned long long*>(lengths);
  unsigned int sum = 0;
  for (int i = tid; i < block_start / lengths_per_word; i += TPB) {
    const unsigned long long word = length_words[i];
    sum += (word & 0xffffu) + ((word >> 16) & 0xffffu) +
           ((word >> 32) & 0xffffu) + (word >> 48);
  }

  write_local_pack_offsets(tid, block_start, chunks, lengths, local_offsets);
  // The reduction also publishes the first warp's local offsets to every slot.
  const int payload_base = static_cast<int>(fused_pack_block_sum(sum, reduce_scratch));
  const int directory_bytes = 16 + 2 * chunks;
  if (tid == 0) {
    if (block_start == 0) {
      reinterpret_cast<long long*>(output)[0] = insize;
      reinterpret_cast<float*>(output)[2] = errorbound;
      reinterpret_cast<unsigned int*>(output)[3] = 0;
    }
    if (block_start + PACK_SLOTS_PER_BLOCK >= chunks) {
      *outsize = directory_bytes + payload_base + local_offsets[PACK_SLOTS_PER_BLOCK];
    }
  }

  const int local_slot = tid / PACK_THREADS_PER_SLOT;
  const int slot = block_start + local_slot;
  if (slot >= chunks) return;
  const int base = slot * CS;
  const int raw_size = min(CS, insize - base);
  const int length = lengths[slot];
  const bool raw = (length == raw_size);
  const byte* source = raw ? &input[base] : &staging[base];
  byte* destination = output + directory_bytes + payload_base + local_offsets[local_slot];
  copy_staged_packet<PACK_THREADS_PER_SLOT>(destination, source, length);
}


// staging and offsets hold min(number of chunks, STAGED_CHUNK_LIMIT) slots.
// output has room for the 16-byte header, a u16 per chunk, and insize bytes.
static void submit_staged_encode(const byte* input, int insize, byte* output, int* outsize,
                                 byte* staging, int* offsets, float errorbound, float threshold,
                                 hipStream_t stream = nullptr)
{
  const int chunks = 1 + (insize - 1) / CS;
  auto* lengths = reinterpret_cast<unsigned short*>(output + 16);
  if (chunks <= STAGED_CHUNK_LIMIT) {
    const int blocks = (chunks + PACK_SLOTS_PER_BLOCK - 1) / PACK_SLOTS_PER_BLOCK;
    d_encode_staged<<<chunks, TPB, 0, stream>>>(input, insize, 0, staging, lengths, errorbound, threshold);
    d_scan_pack_staged<<<blocks, TPB, 0, stream>>>(input, insize, chunks, staging, lengths,
                                                 output, outsize, errorbound);
    return;
  }
  for (int first = 0; first < chunks; first += STAGED_CHUNK_LIMIT) {
    const int remaining = chunks - first;
    const int count = (remaining < STAGED_CHUNK_LIMIT) ? remaining : STAGED_CHUNK_LIMIT;
    d_encode_staged<<<count, TPB, 0, stream>>>(input, insize, first, staging, lengths, errorbound, threshold);
    d_scan_staged_offsets<<<1, TPB, 0, stream>>>(lengths, first, count, chunks, insize, errorbound,
                                                offsets, output, outsize);
    d_pack_staged<<<count, TPB, 0, stream>>>(input, insize, first, chunks, staging, offsets, lengths,
                                          output, outsize);
  }
}


static void submit_legacy_encode(const byte* input, int insize, byte* output, int* outsize,
                                 int* carry, int blocks, float errorbound, float threshold,
                                 hipStream_t stream = nullptr)
{
  const int chunks = 1 + (insize - 1) / CS;
  check_compression_hip(hipMemsetAsync(carry, 0, static_cast<std::size_t>(chunks + 1) * sizeof(int), stream),
                        "initialize fallback compression scratch");
  d_encode_legacy<<<blocks, TPB, 0, stream>>>(input, insize, output, outsize, carry, carry + chunks,
                                            errorbound, threshold);
}


static void submit_encode(const byte* input, int insize, byte* output, int* outsize,
                          const EncodeScratch& scratch, int legacy_blocks,
                          float errorbound, float threshold, hipStream_t stream = nullptr)
{
  if (scratch.staging_bytes) {
    auto* staging = static_cast<byte*>(scratch.storage);
    auto* offsets = reinterpret_cast<int*>(staging + scratch.staging_bytes);
    submit_staged_encode(input, insize, output, outsize, staging, offsets, errorbound, threshold, stream);
  } else {
    submit_legacy_encode(input, insize, output, outsize, static_cast<int*>(scratch.storage),
                         legacy_blocks, errorbound, threshold, stream);
  }
}


struct GPUTimer
{
  hipEvent_t beg, end;
  GPUTimer() {hipEventCreate(&beg); hipEventCreate(&end);}
  ~GPUTimer() {hipEventDestroy(beg); hipEventDestroy(end);}
  void start() {hipEventRecord(beg, 0);}
  double stop() {hipEventRecord(end, 0); hipEventSynchronize(end); float ms; hipEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckHip(const int line)
{
  hipError_t e;
  hipDeviceSynchronize();
  if (hipSuccess != (e = hipGetLastError())) {
    fprintf(stderr, "HIP error %d on line %d: %s\n\n", e, line, hipGetErrorString(e));
    throw std::runtime_error("LC error");
  }
}


int main(int argc, char* argv []) try
{
  printf("PFPL GPU Single-Precision ABS Compressor\n");
  printf("Copyright 2025 Texas State University\n\n");

  // read input from file
  if (argc < 4 || argc > 5) {printf("USAGE: %s input_file_name compressed_file_name error_bound [threshold]\n\n", argv[0]);  throw std::runtime_error("LC error");}

  int insize;
  const auto input = read_f32_input(argv[1], insize);
  printf("original size: %d bytes\n", insize);

  // get GPU info
  hipSetDevice(0);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no HIP capable device detected\n\n"); throw std::runtime_error("LC error");}
  const int chunks = 1 + (insize - 1) / CS;
  const int legacy_blocks = deviceProp.multiProcessorCount * (deviceProp.maxThreadsPerMultiProcessor / TPB);
  CheckHip(__LINE__);
  const std::size_t maxsize = 16 + 2 * static_cast<std::size_t>(chunks) + insize;

  // allocate GPU memory
  byte* dencoded;
  check_compression_hip(hipHostMalloc((void**)&dencoded, maxsize), "allocate host packet buffer");
  byte* d_input;
  check_compression_hip(hipMalloc((void**)&d_input, insize), "allocate compression input");
  check_compression_hip(hipMemcpy(d_input, input.get(), insize, hipMemcpyHostToDevice), "transfer compression input");
  byte* d_encoded;
  check_compression_hip(hipMalloc((void**)&d_encoded, maxsize), "allocate compression packet buffer");
  int* d_encsize;
  check_compression_hip(hipMalloc((void**)&d_encsize, sizeof(int)), "allocate packet size");
  const EncodeScratch scratch(insize);
  CheckHip(__LINE__);

  const float errorbound = atof(argv[3]);
  const float threshold = (argc == 5) ? atof(argv[4]) : std::numeric_limits<float>::infinity();
  if (threshold < std::numeric_limits<float>::min()) {printf("ERROR: threshold must be a positive, normal, floating-point value\n");  throw std::runtime_error("LC error");}
  GPUTimer dtimer;
  for (int i = 0; i < NUM_RUNS; i++) {
    hipDeviceSynchronize();
    dtimer.start();

    submit_encode(d_input, insize, d_encoded, d_encsize, scratch, legacy_blocks,
                  errorbound, threshold);

    hipDeviceSynchronize();
    double runtime = dtimer.stop();

    if (i < NUM_RUNS - 1) {
      hipMemset(d_encsize, 0, sizeof(int));
      hipMemset(d_encoded, 0, maxsize);
    }
    CheckHip(__LINE__);
    printf("lc comp ecltime, %12.9f\n", runtime);
  }

  // get encoded GPU result
  int dencsize = 0;
  check_compression_hip(hipMemcpy(&dencsize, d_encsize, sizeof(int), hipMemcpyDeviceToHost),
                        "read packet size");
  if (dencsize <= 0 || static_cast<std::size_t>(dencsize) > maxsize) {
    fprintf(stderr, "ERROR: encoded packet size is outside the supported range\n\n");
    throw std::runtime_error("LC error");
  }
  check_compression_hip(hipMemcpy(dencoded, d_encoded, dencsize, hipMemcpyDeviceToHost),
                        "read encoded packet");
  printf("encoded size: %d bytes\n", dencsize);
  printf("compression ratio: %.2f\n", 1.0 * insize / dencsize);
  CheckHip(__LINE__);

  // write to file
  write_f32_output(argv[2], dencoded, dencsize);

  // clean up GPU memory
  hipFree(d_input);
  hipFree(d_encoded);
  hipFree(d_encsize);
  CheckHip(__LINE__);

  // clean up
  hipHostFree(dencoded);
  return 0;
}
catch (const std::exception& error)
{
  fprintf(stderr, "%s\n", error.what());
  return 1;
}
