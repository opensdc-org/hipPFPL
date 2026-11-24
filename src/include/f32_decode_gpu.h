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

#include <hip/hip_runtime.h>
#include "f32_decode_common.h"


static __device__ inline unsigned int restore_f32_absolute(unsigned int bits, float eb2)
{
  constexpr unsigned int max_quantized_code = 1u << 23;
  if (bits < max_quantized_code) {
    const int bin = static_cast<int>(bits >> 1) ^ -static_cast<int>(bits & 1u);
    return __float_as_uint(bin * eb2);
  }
  return bits;
}


template <bool FullChunk, typename RestoreWord>
static __device__ __forceinline__ void restore_diff_quant(const byte* differences,
    byte* destination, int bytes, int* scratch, const RestoreWord& restore_word)
{
  constexpr int words_per_thread = CS / sizeof(float) / TPB;
  static_assert(CS % (sizeof(float) * TPB) == 0);
  constexpr unsigned int diff_bias = 0xAAAAAAAAu;
  const auto* diff_words = reinterpret_cast<const unsigned int*>(differences);
  auto* output_words = reinterpret_cast<unsigned int*>(destination);
  const int word_count = bytes / sizeof(float);
  const int tid = threadIdx.x;
  const int begin = FullChunk ? tid * words_per_thread : tid * word_count / TPB;
  const int count = FullChunk ? words_per_thread : (tid + 1) * word_count / TPB - begin;

  // Each thread retains its differences across the scan. A partial chunk
  // can read and write the same range.
  unsigned int diffs[words_per_thread];
  unsigned int sum = 0;
  #pragma unroll
  for (int i = 0; i < words_per_thread; ++i) {
    const unsigned int diff = (i < count) ? (diff_words[begin + i] ^ diff_bias) - diff_bias : 0u;
    diffs[i] = diff;
    sum += diff;
  }
  if constexpr (FullChunk) {
    sum = restore_wave_inclusive(sum, scratch);
  } else {
    sum = block_prefix_sum(sum, scratch);
  }

  #pragma unroll
  for (int i = words_per_thread - 1; i >= 0; --i) {
    if (i < count) {
      output_words[begin + i] = restore_word(sum);
      sum -= diffs[i];
    }
  }
}


template <typename RestoreWord>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_decode(const byte* const __restrict__ input, byte* const __restrict__ output,
    int* const __restrict__ g_outsize, const int* const __restrict__ offsets, const int chunks)
{
  // RZE stores four levels of byte bitmaps after WS scan words.
  constexpr int rze_scratch_bytes = WS * sizeof(int) + 2048 + 256 + 32 + 4;
  __shared__ long long expanded[CS / sizeof(long long)];
  __shared__ int scratch[(rze_scratch_bytes + sizeof(int) - 1) / sizeof(int)];
  __shared__ int ticket[2];

  const auto* head_in = reinterpret_cast<const long long*>(input);
  const int outsize = static_cast<int>(head_in[0]);
  const RestoreWord restore_word(reinterpret_cast<const float*>(&head_in[1])[0]);
  const auto* size_in = reinterpret_cast<const unsigned short*>(&head_in[2]);
  const auto* data_in = reinterpret_cast<const byte*>(&size_in[chunks]);
  auto* expanded_bytes = reinterpret_cast<byte*>(expanded);
  auto* scratch_bytes = reinterpret_cast<byte*>(scratch);

  const int tid = threadIdx.x;
  if (tid == 0) ticket[0] = atomicAdd(&g_chunk_counter, 1);
  __syncthreads();

  for (int slot = 0; ; slot ^= 1) {
    const int chunk_id = ticket[slot];
    if (chunk_id >= chunks) break;
    const int base = chunk_id * CS;
    const byte* source = &data_in[offsets[chunk_id]];
    byte* destination = &output[base];
    int stored = static_cast<int>(size_in[chunk_id]);
    const int raw_size = min(CS, outsize - base);
    if (stored < raw_size) {
      d_iRZE<byte, true>(stored, source, expanded_bytes, scratch_bytes);
      __syncthreads();
      if (stored == CS) {
        inverse_bit_full_chunk(expanded_bytes);
        __syncthreads();  // Publish inverse-bit words in shared memory.
        restore_diff_quant<true>(expanded_bytes, destination, CS, scratch, restore_word);
      } else {
        d_iBIT_4(stored, expanded_bytes, destination, scratch_bytes);
        __syncthreads();  // Publish the temporary words in the output range.
        restore_diff_quant<false>(destination, destination, stored, scratch, restore_word);
      }
    } else {
      copy_raw_chunk(source, destination, raw_size);
    }
    // The other slot can be written while lanes still read this chunk's ticket.
    if (tid == 0) ticket[slot ^ 1] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // Publish the ticket and finish using this chunk's shared buffers.
  }

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}
