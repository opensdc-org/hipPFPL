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

// Construct both byte-permute selectors without a lane-dependent table load.
static __device__ __forceinline__ unsigned int ze_byte_permute_selectors(unsigned int marks)
{
  // Put one presence bit in each nibble of a 32-bit word.
  unsigned int present = (marks | (marks << 12)) & 0x000f000fu;
  present = (present | (present << 6)) & 0x03030303u;
  present = (present | (present << 3)) & 0x11111111u;

  // Multiplication accumulates the earlier presence bits in each group of
  // four nibbles. Keeping the halves separate restarts the rank at byte four.
  const unsigned int low_ranks = (present * 0x1110u) & 0xffffu;
  const unsigned int high_ranks = (present & 0xffff0000u) * 0x1110u;
  // Indices 4 through 7 all refer to the zero operand for absent bytes.
  return low_ranks | high_ranks | ((present ^ 0x11111111u) << 2);
}


// Swizzle full inverse-BIT planes across LDS banks without moving words
// within their aligned 16-byte group. Each plane occupies 128 uint32 words.
static __device__ __forceinline__ unsigned int ze_full_byte_bit_word_index(unsigned int word)
{
  return word ^ (((word >> 7) & 7u) << 2);
}


static __device__ __forceinline__ unsigned long long ze_expand_packed_word(
    unsigned long long packed, unsigned int bm)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_perm)
  {
    // One presence bit per nibble. Summing earlier nibbles gives the
    // absolute packed byte index (0..7) for every present output byte.
    unsigned int present = (bm | (bm << 12)) & 0x000f000fu;
    present = (present | (present << 6)) & 0x03030303u;
    present = (present | (present << 3)) & 0x11111111u;
    const unsigned int ranks = present * 0x11111110u;
    const unsigned int valid = present * 15u;
    // Selector 12 writes zero instead of reading either packed operand.
    const unsigned int selectors = (ranks & valid) | (0xccccccccu & ~valid);
    // v_perm uses one byte per selector. Interleave even and odd nibbles
    // into the low and high four-byte control words.
    const unsigned int even = selectors & 0x0f0f0f0fu;
    const unsigned int odd = (selectors >> 4) & 0x0f0f0f0fu;
    const unsigned int low_control = __builtin_amdgcn_perm(odd, even, 0x05010400u);
    const unsigned int high_control = __builtin_amdgcn_perm(odd, even, 0x07030602u);
    const unsigned int packed_lo = static_cast<unsigned int>(packed);
    const unsigned int packed_hi = static_cast<unsigned int>(packed >> 32);
    const unsigned int lo = __builtin_amdgcn_perm(packed_hi, packed_lo, low_control);
    const unsigned int hi = __builtin_amdgcn_perm(packed_hi, packed_lo, high_control);
    return static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
  }
#endif
#endif
  const unsigned int lower = bm & 15u;
  const unsigned int selectors = ze_byte_permute_selectors(bm);
  const auto lo = __byte_perm(static_cast<unsigned int>(packed), 0u, selectors & 0xffffu);
  const auto hi = __byte_perm(static_cast<unsigned int>(packed >> (8 * __popc(lower))), 0u,
                              selectors >> 16);
  return static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32);
}


// Compute packed addresses by rank and expand four consecutive 8-byte output words per thread.
static __device__ inline void d_ZEdecode_full_byte_permute(const byte* datain, int payload_bytes,
#if defined(PFPL_ABS_REGISTER_ZE)
    const unsigned int marks, byte* out, int* temp_w)
#else
    const byte* bitmap, byte* out, int* temp_w)
#endif
{
  constexpr int words_per_thread = 4;
  const int tid = threadIdx.x;
#if !defined(PFPL_ABS_REGISTER_ZE)
  const auto* bitmap_words = reinterpret_cast<const unsigned int*>(bitmap);
  const unsigned int marks = bitmap_words[tid];
#endif
  const unsigned int count = __popc(marks);
  const unsigned int end = byte_ze_exclusive_offset(count, temp_w) + count;
  if (payload_bytes < 0 || end > static_cast<unsigned int>(payload_bytes) ||
      (tid == TPB - 1 && end != static_cast<unsigned int>(payload_bytes))) {
    __trap();
    return;
  }
  const int base = static_cast<int>(end - count);
  auto* output_words = reinterpret_cast<unsigned long long*>(out);
  #pragma unroll
  for (int next = 0; next < words_per_thread; ++next) {
    const unsigned int bm = (marks >> (8 * next)) & 255u;
    unsigned long long expanded = 0;
    if (bm != 0) {
      const unsigned int previous = marks & ((1u << (8 * next)) - 1u);
      const int pos = base + static_cast<int>(__popc(previous));
      unsigned long long packed = 0;
      if (payload_bytes >= 8) {
        // An end group can reuse the last complete window. Validation of
        // this lane's span guarantees a shift in [0, 56] and enough bytes.
        const int start = min(pos, payload_bytes - 8);
        __builtin_memcpy(&packed, datain + start, sizeof(packed));
        packed >>= 8 * (pos - start);
      } else {
        const int valid = static_cast<int>(__popc(bm));
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          if (i < valid) packed |= static_cast<unsigned long long>(datain[pos + i]) << (8 * i);
        }
      }
      expanded = ze_expand_packed_word(packed, bm);
    }
    const unsigned int word = tid * words_per_thread + next;
    output_words[ze_full_byte_bit_word_index(2 * word) / 2] = expanded;
  }
}
