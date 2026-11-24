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

#pragma once

#include <string>
#include <cmath>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cassert>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include "macros.h"
#include "sum_reduction.h"
#include "chunk_offsets.h"
#include "../components/d_BIT_4.h"
#define PFPL_F32_ABS_RZE_ENCODE
#include "../components/d_RZE_1.h"
#undef PFPL_F32_ABS_RZE_ENCODE
#include "f32_encode_common.h"


struct AbsoluteWordQuantizer
{
  float errorbound;
  float threshold;
  float eb2;
  float inv_eb2;

  __device__ AbsoluteWordQuantizer(float bound, float cutoff)
      : errorbound(bound), threshold(cutoff), eb2(2 * bound), inv_eb2(0.5f / bound)
  {
  }

  __device__ __forceinline__ unsigned int operator()(unsigned int original_bits) const
  {
    const int maxbin = 1 << 22;  // Leave one mantissa bit for sign.
    const float original = __uint_as_float(original_bits);
    const float scaled = original * inv_eb2;
    const int bin = static_cast<int>(roundf(scaled));
    const float reconstructed = bin * eb2;

    if ((bin >= maxbin) || (bin <= -maxbin) || (fabsf(original) >= threshold) ||
        (fabsf(original - reconstructed) > errorbound) || (original != original)) {
      return original_bits;
    }
    const unsigned int sign = 0u - static_cast<unsigned int>(bin < 0);
    return (static_cast<unsigned int>(bin) << 1) ^ sign;
  }
};


template <typename WordQuantizer>
static __device__ inline void quantize_chunk_in_place(int csize, byte* slab,
                                                     const WordQuantizer& quantize)
{
  const int words = csize / sizeof(float);
  auto* values = reinterpret_cast<unsigned int*>(slab);
  for (int i = threadIdx.x; i < words; i += TPB) {
    values[i] = quantize(values[i]);
  }
}


// Full chunks already have the pair alignment used by load_raw_chunk.
template <typename WordQuantizer>
static __device__ inline void load_quantized_full_chunk(const byte* source, byte* slab,
                                                         const WordQuantizer& quantize)
{
  using Pair = unsigned long long;
  constexpr int pairs_per_thread = CS / (sizeof(Pair) * TPB);
  static_assert(sizeof(Pair) == 8 && sizeof(unsigned int) == 4 && CS % (sizeof(Pair) * TPB) == 0);
  const auto* input_pairs = reinterpret_cast<const Pair*>(source);
  auto* output_pairs = reinterpret_cast<Pair*>(slab);
  const int tid = threadIdx.x;

  if ((reinterpret_cast<std::size_t>(source) & 15u) == 0) {
    constexpr int words_per_vector = 4;
    constexpr int vectors_per_thread = CS / (sizeof(uint4) * TPB);
    static_assert(sizeof(uint4) == words_per_vector * sizeof(unsigned int) && CS % (sizeof(uint4) * TPB) == 0);
    const auto* vectors = reinterpret_cast<const uint4*>(source);
    #pragma unroll
    for (int row = 0; row < vectors_per_thread; ++row) {
      const int i = tid + row * TPB;
      const uint4 raw = vectors[i];
      const unsigned int a = quantize(raw.x);
      const unsigned int b = quantize(raw.y);
      const unsigned int c = quantize(raw.z);
      const unsigned int d = quantize(raw.w);
      output_pairs[2 * i] = (static_cast<Pair>(b) << 32) | a;
      output_pairs[2 * i + 1] = (static_cast<Pair>(d) << 32) | c;
    }
    return;
  }

  #pragma unroll
  for (int row = 0; row < pairs_per_thread; ++row) {
    const int i = tid + row * TPB;
    const Pair original = input_pairs[i];
    const unsigned int low = quantize(static_cast<unsigned int>(original));
    const unsigned int high = quantize(static_cast<unsigned int>(original >> 32));
    output_pairs[i] = (static_cast<Pair>(high) << 32) | low;
  }
}


template <typename Quantizer>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode_legacy(const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int* const __restrict__ outsize, int* const __restrict__ fullcarry, int* const next_chunk_counter,
                     const Quantizer quantizer)
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
    quantizer(raw_size, slab);
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
      reinterpret_cast<float*>(&head_out[1])[0] = quantizer.stored_error_bound();
      reinterpret_cast<unsigned int*>(output)[3] = 0;
      const long long total = &data_out[fullcarry[chunk_id]] - output;
      *outsize = (total <= INT_MAX) ? static_cast<int>(total) : 0;
    }
  }
}


template <typename Quantizer>
static __global__ __launch_bounds__(TPB, 2)
void d_encode_staged(const byte* const __restrict__ input, const int insize,
                     const int first, byte* const __restrict__ staging,
                     unsigned short* const __restrict__ lengths,
                     const Quantizer quantizer)
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
    quantizer.load_full(&input[base], slab);
    __syncthreads();
    good = d_DIFFNB_BIT_ZE_4_full_chunk(csize, slab, scratch);
  } else {
    load_raw_chunk(&input[base], slab, raw_size);
    __syncthreads();
    quantizer(raw_size, slab);
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


template <typename Quantizer>
static __global__ __launch_bounds__(TPB)
void d_scan_staged_offsets(const unsigned short* const __restrict__ lengths,
                           const int first, const int count, const int chunks,
                           const int insize, const Quantizer quantizer,
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
      reinterpret_cast<float*>(output)[2] = quantizer.stored_error_bound();
      reinterpret_cast<unsigned int*>(output)[3] = 0;
    }
    const long long total = static_cast<long long>(directory_bytes) + payload_bytes;
    *outsize = (total <= INT_MAX) ? static_cast<int>(total) : 0;
  }
}


// A block sums the completed lengths preceding its slots independently.
// This adds bounded reads of the directory and needs no inter-block ordering.
template <typename Quantizer>
static __global__ __launch_bounds__(TPB)
void d_scan_pack_staged(const byte* const __restrict__ input, const int insize,
                        const int chunks, const byte* const __restrict__ staging,
                        const unsigned short* const __restrict__ lengths,
                        byte* const output, int* const __restrict__ outsize, const Quantizer quantizer)
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
      reinterpret_cast<float*>(output)[2] = quantizer.stored_error_bound();
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
template <typename Quantizer>
static void submit_staged_encode(const byte* input, int insize, byte* output, int* outsize,
                                 byte* staging, int* offsets, Quantizer quantizer,
                                 hipStream_t stream = nullptr)
{
  const int chunks = 1 + (insize - 1) / CS;
  auto* lengths = reinterpret_cast<unsigned short*>(output + 16);
  if (chunks <= STAGED_CHUNK_LIMIT) {
    const int blocks = (chunks + PACK_SLOTS_PER_BLOCK - 1) / PACK_SLOTS_PER_BLOCK;
    d_encode_staged<Quantizer><<<chunks, TPB, 0, stream>>>(input, insize, 0, staging, lengths, quantizer);
    d_scan_pack_staged<Quantizer><<<blocks, TPB, 0, stream>>>(input, insize, chunks, staging, lengths,
                                                            output, outsize, quantizer);
    return;
  }
  for (int first = 0; first < chunks; first += STAGED_CHUNK_LIMIT) {
    const int remaining = chunks - first;
    const int count = (remaining < STAGED_CHUNK_LIMIT) ? remaining : STAGED_CHUNK_LIMIT;
    d_encode_staged<Quantizer><<<count, TPB, 0, stream>>>(input, insize, first, staging, lengths, quantizer);
    d_scan_staged_offsets<Quantizer><<<1, TPB, 0, stream>>>(lengths, first, count, chunks, insize,
                                                          quantizer, offsets, output, outsize);
    d_pack_staged<<<count, TPB, 0, stream>>>(input, insize, first, chunks, staging, offsets, lengths,
                                          output, outsize);
  }
}


template <typename Quantizer>
static void submit_legacy_encode(const byte* input, int insize, byte* output, int* outsize,
                                 int* carry, int blocks, Quantizer quantizer,
                                 hipStream_t stream = nullptr)
{
  const int chunks = 1 + (insize - 1) / CS;
  check_compression_hip(hipMemsetAsync(carry, 0, static_cast<std::size_t>(chunks + 1) * sizeof(int), stream),
                        "initialize fallback compression scratch");
  d_encode_legacy<Quantizer><<<blocks, TPB, 0, stream>>>(input, insize, output, outsize, carry,
                                                       carry + chunks, quantizer);
}


template <typename Quantizer>
static void submit_encode(const byte* input, int insize, byte* output, int* outsize,
                          const EncodeScratch& scratch, int legacy_blocks,
                          Quantizer quantizer, hipStream_t stream = nullptr)
{
  if (scratch.staging_bytes) {
    auto* staging = static_cast<byte*>(scratch.storage);
    auto* offsets = reinterpret_cast<int*>(staging + scratch.staging_bytes);
    submit_staged_encode(input, insize, output, outsize, staging, offsets, quantizer, stream);
  } else {
    submit_legacy_encode(input, insize, output, outsize, static_cast<int*>(scratch.storage),
                         legacy_blocks, quantizer, stream);
  }
}
