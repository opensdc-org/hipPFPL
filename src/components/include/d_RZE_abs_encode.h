/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2024, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
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

template <int Distance>
static __device__ __forceinline__ void upper_rze_prefix_step(int lane, int& total, int& exclusive)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_ds_swizzle)
  const int peer = __builtin_amdgcn_ds_swizzle(total, 31 | (Distance << 10));
#else
  const int peer = pfpl_shfl_xor_sync(~0u, total, Distance, 32);
#endif
#else
  const int peer = pfpl_shfl_xor_sync(~0u, total, Distance, 32);
#endif
  if (lane & Distance) exclusive += peer;
  total += peer;
}


static __device__ __forceinline__ unsigned int upper_rze_predecessor(unsigned int value, int lane)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && defined(__gfx936__) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#if __has_builtin(__builtin_amdgcn_mov_dpp) && __has_builtin(__builtin_amdgcn_ds_swizzle)
  const unsigned int inside = __builtin_amdgcn_mov_dpp(value, 0x111, 0xf, 0xf, false);
  const unsigned int boundary = __builtin_amdgcn_ds_swizzle(value, 31 | (31 << 10));
  return lane == 16 ? boundary : inside;
#endif
#endif
  return pfpl_shfl_up_sync(~0u, value, 1, 32);
}


// Finish both upper byte bitmap layers in a logical warp. Each lane owns
// eight input bytes and keeps their one-byte change mask in registers.
static __device__ inline bool encode_rze_upper_256(const byte* in, byte* out,
                                                   int& capacity, int* result, int raw_bytes)
{
  constexpr int width = 32;
  const int lane = threadIdx.x;
  if (lane < width) {
    const auto* words = reinterpret_cast<const unsigned int*>(in);
    const unsigned int lo = words[2 * lane];
    const unsigned int hi = words[2 * lane + 1];
    const unsigned int prior_word = upper_rze_predecessor(hi, lane);
    const unsigned int prior_byte = (lane == 0) ? 0u : (prior_word >> 24);
    const unsigned int low_changes = d_nonzero_byte_mask(lo ^ ((lo << 8) | prior_byte));
    const unsigned int high_changes = d_nonzero_byte_mask(hi ^ ((hi << 8) | (lo >> 24)));
    const unsigned int changes = low_changes | (high_changes << 4);
    const unsigned int prior_changes = upper_rze_predecessor(changes, lane);
    const unsigned int top = pfpl_ballot_sync(~0u, changes != ((lane == 0) ? 0u : prior_changes));

    const int count = __popc(changes);
    int next_start = count;
    int exclusive = 0;
    upper_rze_prefix_step<1>(lane, next_start, exclusive);
    upper_rze_prefix_step<2>(lane, next_start, exclusive);
    upper_rze_prefix_step<4>(lane, next_start, exclusive);
    upper_rze_prefix_step<8>(lane, next_start, exclusive);
    upper_rze_prefix_step<16>(lane, next_start, exclusive);
    const int top_start = next_start + __popc(top);
    const int length = top_start + sizeof(unsigned int);
    if (length < capacity) {
      int position = exclusive;
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        if ((low_changes >> i) & 1u) out[position++] = lo >> (8 * i);
      }
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        if ((high_changes >> i) & 1u) out[position++] = hi >> (8 * i);
      }
      if ((top >> lane) & 1u) {
        const int rank = __popc(top & ((1u << lane) - 1u));
        out[next_start + rank] = changes;
      }
      if (lane < 4) out[top_start + lane] = top >> (8 * lane);
    }
    if (lane == 0) {
      if (length < capacity) {
        out[length] = raw_bytes;
        out[length + 1] = raw_bytes >> 8;
      }
      *result = (length < capacity) ? length : 0;
    }
  }
  __syncthreads();
  const int length = *result;
  if (length == 0) return false;
  capacity = length;
  return true;
}


// Finish a full byte chunk after the first RE payload and mask were emitted.
static __device__ inline bool d_RZE_finish_full_byte(int& csize, byte out [CS],
                                                    byte temp [CS], int wpos)
{
  const int tid = threadIdx.x;
  const int avail = CS - 2;
  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (wpos >= avail) return false;
  if (wpos == 0) {
    if (tid == 0) {
      out[0] = csize;
      out[1] = csize >> 8;
    }
    csize = 2;
    __syncthreads();
    return true;
  }
  __syncthreads();

  int count = avail - wpos;
  if (!encode_rze_upper_256(bitmap + 2048, out + wpos, count, &temp_w[WS], csize)) return false;
  // The upper pass publishes its payload and the full-chunk footer together.
  csize = wpos + count + 2;
  return true;
}
