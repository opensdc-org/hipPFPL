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

#include <cassert>
#include <cstdint>
#include <hip/hip_runtime.h>
#include "macros.h"
#include "chunk_offsets.h"
#define PFPL_F32_ABS_RZE_DECODE
#include "../components/d_BIT_4.h"
#include "../components/d_RZE_1.h"
#undef PFPL_F32_ABS_RZE_DECODE


static __device__ int g_chunk_counter;


static __global__ __launch_bounds__(TPB)
void d_chunk_offsets(const byte* input, int chunks, int* offsets, int first_dynamic_chunk = 0)
{
  const auto* sizes = reinterpret_cast<const unsigned short*>(input + 16);
  __shared__ unsigned int scan_scratch[WS];
  __shared__ unsigned int tile_total;
  if (threadIdx.x == 0) g_chunk_counter = first_dynamic_chunk;
#if defined(PFPL_ABS_WAVE_OFFSETS) && defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && \
    defined(__gfx936__) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#if __has_builtin(__builtin_amdgcn_ds_swizzle)
  if (chunks <= 4 * TPB) {
    scan_chunk_offsets<false, true>(sizes, chunks, offsets, 0, scan_scratch, tile_total);
    return;
  }
#endif
#endif
  scan_chunk_offsets<false>(sizes, chunks, offsets, 0, scan_scratch, tile_total);
}


static __device__ inline void copy_raw_chunk(const byte* source, byte* destination, int bytes)
{
  const int tid = threadIdx.x;
  auto* output_words = reinterpret_cast<unsigned long long*>(destination);
  if ((reinterpret_cast<std::uintptr_t>(source) & 7u) == 0) {
    const auto* input_words = reinterpret_cast<const unsigned long long*>(source);
    for (int i = tid; i < bytes / 8; i += TPB) {
      output_words[i] = input_words[i];
    }
  } else {
    for (int i = tid; i < bytes / 8; i += TPB) {
      __builtin_memcpy(output_words + i, source + 8 * i, 8);
    }
  }

  const int tail = bytes % 8;
  if (tid < tail) {
    destination[bytes - tail + tid] = source[bytes - tail + tid];
  }
}


#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_perm)
template <int Distance, unsigned int Alternate>
static __device__ __forceinline__ unsigned int merge_bit_partner(unsigned int value, int lane)
{
  const unsigned int peer = bit_shuffle_xor<Distance>(value);
  const bool upper = (lane & Distance) != 0;
  const unsigned int from_peer = upper ? ~Alternate : Alternate;
  const unsigned int shift = upper ? 32 - Distance : Distance;
  const unsigned int rotated = __funnelshift_r(peer, peer, shift);
  return (value & ~from_peer) | (rotated & from_peer);
}


static __device__ __forceinline__ unsigned int transpose_inverse_word(unsigned int a, int sublane,
    unsigned int permute16, unsigned int permute8)
{
  unsigned int q = bit_shuffle_xor<16>(a);
  a = __builtin_amdgcn_perm(q, a, permute16);
  q = bit_shuffle_xor<8>(a);
  a = __builtin_amdgcn_perm(q, a, permute8);

  a = merge_bit_partner<4, 0x0F0F0F0Fu>(a, sublane);
  a = merge_bit_partner<2, 0x33333333u>(a, sublane);
  a = merge_bit_partner<1, 0x55555555u>(a, sublane);
  return a;
}
#endif
#endif


static __device__ inline void inverse_bit_full_chunk(byte* expanded)
{
  constexpr int width = 32;
  constexpr int words = CS / sizeof(unsigned int);
  constexpr int words_per_thread = words / TPB;
  static_assert(words % TPB == 0 && TPB % width == 0);

  auto* storage = reinterpret_cast<unsigned int*>(expanded);
  const int tid = threadIdx.x;
  const int lane = tid % width;
  unsigned int transposed[words_per_thread];
  #pragma unroll
  for (int row = 0; row < words_per_thread; ++row) {
    const int word = tid + row * TPB;
    const int source = word / width + lane * (words / width);
    const unsigned int packed = storage[ze_full_byte_bit_word_index(source)];
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && __has_builtin(__builtin_amdgcn_perm)
    const unsigned int p16 = (lane & 16) ? 0x05040100u : 0x03020706u;
    const unsigned int p8 = (lane & 8) ? 0x06020400u : 0x03070105u;
    transposed[row] = transpose_inverse_word(packed, lane, p16, p8);
#else
    transposed[row] = d_iBIT_4_word(packed, lane);
#endif
  }

  __syncthreads();  // All lanes have read the old bit planes.
  #pragma unroll
  for (int row = 0; row < words_per_thread; ++row) {
    storage[tid + row * TPB] = transposed[row];
  }
}


#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#if __has_builtin(__builtin_amdgcn_ds_swizzle)
template <int distance>
static __device__ inline void restore_unsigned_prefix_step(int lane, unsigned int& sum, unsigned int& offset)
{
  const unsigned int neighbor = __builtin_amdgcn_ds_swizzle(sum, 31 | (distance << 10));
  if (lane & distance) offset += neighbor;
  sum += neighbor;
}
#endif
#endif

// Each physical wave calculates the total of earlier waves after one
// publication. LDS holds only the eight wave totals, which are read-only
// until the decoder's barrier before reusing the chunk scratch.
static __device__ inline unsigned int restore_wave_inclusive(unsigned int count, int* scratch)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#if __has_builtin(__builtin_amdgcn_ds_swizzle)
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave_index = tid / 64;
  auto* totals = reinterpret_cast<unsigned int*>(scratch);
  unsigned int sum = count;
  unsigned int offset = 0;
  restore_unsigned_prefix_step<1>(lane, sum, offset);
  restore_unsigned_prefix_step<2>(lane, sum, offset);
  restore_unsigned_prefix_step<4>(lane, sum, offset);
  restore_unsigned_prefix_step<8>(lane, sum, offset);
  restore_unsigned_prefix_step<16>(lane, sum, offset);
  const unsigned int other_half = __shfl_xor_sync(~0ull, sum, 32, 64);
  if (tid & 32) offset += other_half;
  if ((tid & 63) == 0) totals[wave_index] = sum + other_half;
  __syncthreads();

  // Contributions occupy the first eight lanes of each logical half-wave.
  if (wave_index != 0) {
    unsigned int prior = lane < wave_index ? totals[lane] : 0u;
    prior += __builtin_amdgcn_ds_swizzle(prior, 31 | (1 << 10));
    prior += __builtin_amdgcn_ds_swizzle(prior, 31 | (2 << 10));
    prior += __builtin_amdgcn_ds_swizzle(prior, 31 | (4 << 10));
    offset += __builtin_amdgcn_ds_swizzle(prior, 0);
  }
  return offset + count;
#endif
#endif
  return block_prefix_sum(count, scratch);
}


static int decode_grid_for_packet(int nominal, int chunks)
{
  return nominal < chunks ? nominal : chunks;
}
