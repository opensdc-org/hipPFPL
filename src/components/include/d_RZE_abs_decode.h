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

#if defined(PFPL_ABS_REGISTER_ZE)
// Recover one byte from the published 32-byte repetition mask and ranks.
static __device__ __forceinline__ unsigned int reconstruct_upper_bitmap_byte(
    int index, int start, const byte* in, const byte* masks, const int* offsets)
{
  const int byte_index = index / 8;
  const int bit = index % 8;
  const unsigned int mask = masks[byte_index];
  const int through = offsets[byte_index] + __popc(mask & ((1u << bit) - 1u)) +
                      static_cast<int>((mask >> bit) & 1u);
  return through == 0 ? 0 : in[start + through - 1];
}


// Recover the 32-byte level. Partial chunks also materialize the next level.
#else
// Restore the first two byte-RZE bitmap levels from a warp-local snapshot.
#endif
// Scratch scan words contain the exclusive offsets of the 32 local masks;
// the obsolete top level holds the common cursor for their payload.
#if defined(PFPL_ABS_REGISTER_ZE)
static __device__ inline int d_iRZE_upper_byte(int cursor, const byte* in, byte* bitmap, int* temp_w,
    bool materialize_layer256)
#else
static __device__ inline int d_iRZE_upper_byte(int cursor, const byte* in, byte* bitmap, int* temp_w)
#endif
{
  constexpr int width = 32;
  constexpr int layer256_offset = 2048;
  constexpr int layer32_offset = layer256_offset + 256;
  constexpr int top_offset = layer32_offset + 32;
  auto* const layer256_start = reinterpret_cast<int*>(bitmap + top_offset);
  const int tid = threadIdx.x;

  if (tid < width) {
    unsigned int top_bits = 0;
    if (tid == 0) {
      // The last four encoded bytes need not be naturally aligned.
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        top_bits |= static_cast<unsigned int>(in[cursor - 4 + i]) << (8 * i);
      }
    }
    top_bits = pfpl_shfl_sync(~0u, top_bits, 0, width);
    const int layer32_start = cursor - 4 - __popc(top_bits);
    const int through = __popc(top_bits & ((1u << tid) - 1u)) +
                        static_cast<int>((top_bits >> tid) & 1u);
    const unsigned int mask = (through == 0) ? 0 : in[layer32_start + through - 1];
    bitmap[layer32_offset + tid] = static_cast<byte>(mask);

#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__) && (__AMDGCN_WAVEFRONT_SIZE == 64) && __has_builtin(__builtin_amdgcn_ds_swizzle)
    const int changes = __popc(mask);
    int total = changes;
    int exclusive = 0;
    byte_ze_prefix_step<1>(tid, total, exclusive);
    byte_ze_prefix_step<2>(tid, total, exclusive);
    byte_ze_prefix_step<4>(tid, total, exclusive);
    byte_ze_prefix_step<8>(tid, total, exclusive);
    byte_ze_prefix_step<16>(tid, total, exclusive);
    temp_w[tid] = exclusive;
    if (tid == width - 1) *layer256_start = layer32_start - total;
#else
    const int changes = __popc(mask);
    int inclusive = changes;
    #pragma unroll
    for (int distance = 1; distance < width; distance <<= 1) {
      const int prior = pfpl_shfl_up_sync(~0u, inclusive, distance, width);
      if (tid >= distance) inclusive += prior;
    }
    temp_w[tid] = inclusive - changes;
    if (tid == width - 1) *layer256_start = layer32_start - inclusive;
#endif
  }
  __syncthreads();  // Publish the 32 masks, offsets and common cursor.

  const int start = *layer256_start;
#if defined(PFPL_ABS_REGISTER_ZE)
  if (materialize_layer256) {
    if (tid < 256) {
      bitmap[layer256_offset + tid] =
          reconstruct_upper_bitmap_byte(tid, start, in, bitmap + layer32_offset, temp_w);
    }
    __syncthreads();  // Publish the reconstructed 256-byte level to the partial decode.
#else
  if (tid < 256) {
    const int byte_index = tid / 8;
    const int bit = tid % 8;
    const unsigned int mask = bitmap[layer32_offset + byte_index];
    const int through = temp_w[byte_index] + __popc(mask & ((1u << bit) - 1u)) +
                        static_cast<int>((mask >> bit) & 1u);
    bitmap[layer256_offset + tid] = (through == 0) ? 0 : in[start + through - 1];
#endif
  }
#if !defined(PFPL_ABS_REGISTER_ZE)
  __syncthreads();  // Publish the reconstructed 256-byte level to the wide decode.
#endif
  return start;
}


// One rank group uses at most four consecutive change-stream bytes.
// When all four positions change, the preceding group's value is irrelevant.
static __device__ inline unsigned int reconstruct_ranked_bitmap_word(
    const byte* packed, int packed_total, int preceding, unsigned int marks)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_perm)
  if (packed_total >= 4) {
    unsigned int word;
    if (marks == 15u) {
      __builtin_memcpy(&word, packed + preceding, sizeof(word));
      return word;
    }

    const int front = preceding > 0 ? preceding - 1 : 0;
    const int start = min(front, packed_total - 4);
    __builtin_memcpy(&word, packed + start, sizeof(word));

    // Place one mark in the low bit of each byte, then accumulate the
    // inclusive local rank. A lane at rank zero needs an initial zero.
    unsigned int presence = (marks | (marks << 14)) & 0x00030003u;
    presence = (presence | (presence << 7)) & 0x01010101u;
    const unsigned int ranks = presence * 0x01010101u;
    if (preceding == 0) word <<= 8;
    const unsigned int selectors = ranks + static_cast<unsigned int>(front - start) * 0x01010101u;
    return __builtin_amdgcn_perm(0u, word, selectors);
  }
#endif
#endif

  unsigned int word = 0;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int rank = preceding + __popc(marks & ((2u << i) - 1u));
    const unsigned int value = rank == 0 ? 0u : packed[rank - 1];
    word |= value << (8 * i);
  }
  return word;
}


// Recover the first 2048-byte bitmap from the 256-byte change mask. A
// single scan supplies both the packed length and each lane's initial rank.
static __device__ inline int d_iRZE_ranked_bitmap_2048(
#if defined(PFPL_ABS_REGISTER_ZE)
    int packed_end, const byte* in, byte* bitmap, int* upper_offsets, unsigned int& ze_marks)
#else
    int packed_end, const byte* in, byte* bitmap, int* scratch)
#endif
{
  constexpr int bitmap_bytes = 2048;
#if defined(PFPL_ABS_REGISTER_ZE)
  constexpr int layer256_bytes = 256;
  static_assert((WS + 1) * sizeof(int) <= layer256_bytes);
#else
  constexpr int total_offset = bitmap_bytes + 256 + 32;
#endif
  const int tid = threadIdx.x;
#if defined(PFPL_ABS_REGISTER_ZE)
  const unsigned int upper_byte = reconstruct_upper_bitmap_byte(
      tid / 2, packed_end, in, bitmap + bitmap_bytes + layer256_bytes, upper_offsets);
  const unsigned int marks = (upper_byte >> (4 * (tid & 1))) & 15u;
#else
  const unsigned int marks = (bitmap[bitmap_bytes + tid / 2] >> (4 * (tid & 1))) & 15u;
#endif
  const int count = __popc(marks);
#if defined(PFPL_ABS_REGISTER_ZE)

  // The unmaterialized level has room for the scan and total. The preceding
  // offsets, masks and cursor remain read-only until every wave has used them.
  auto* const rank_scratch = reinterpret_cast<int*>(bitmap + bitmap_bytes);
  const int preceding = byte_ze_exclusive_offset(count, rank_scratch);
  auto* const total = rank_scratch + WS;
#else
  const int preceding = byte_ze_exclusive_offset(count, scratch);
  // The upper-level cursor is already in registers, so its LDS slot can
  // publish the packed byte count for this level.
  auto* const total = reinterpret_cast<int*>(bitmap + total_offset);
#endif
  if (tid == TPB - 1) *total = preceding + count;
  __syncthreads();

  const int packed_begin = packed_end - *total;
  if (packed_begin < 0) {
    __trap();
    return 0;
  }

  const unsigned int word = reconstruct_ranked_bitmap_word(in + packed_begin, *total, preceding, marks);
#if defined(PFPL_ABS_REGISTER_ZE)
  ze_marks = word;
#else
  reinterpret_cast<unsigned int*>(bitmap)[tid] = word;
#endif
  return packed_begin;
}


template <typename T, bool WarpUpper = false>
static __device__ inline void d_iRZE(int& csize, const byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  int rpos = csize;
  csize = (int)in[--rpos] << 8;  // second byte
  csize |= in[--rpos];  // bottom byte
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  assert(CS == 16384);
  assert(TPB >= 256);

  // copy leftover byte
  if constexpr (sizeof(T) > 1) {
    const int extra = csize % sizeof(T);
    if (tid < extra) out[csize - extra + tid] = in[rpos - extra + tid];
    rpos -= extra;
  }

  if (rpos == 0) {
    // all zeros
    T* const out_t = (T*)out;
    for (int i = tid; i < size; i += TPB) {
      out_t[i] = 0;
    }
  } else {
    int* const temp_w = (int*)temp;
    byte* const bitmap = (byte*)&temp_w[WS];

    // iteratively decompress bitmaps
    int base, range;
    if constexpr (WarpUpper) {
      static_assert(std::is_same<T, byte>::value && CS == 16384 && TPB == 512 && WS == 32);
#if defined(PFPL_ABS_REGISTER_ZE)
      const bool full_chunk = size == CS;
      rpos = d_iRZE_upper_byte(rpos, in, bitmap, temp_w, !full_chunk);
      if (full_chunk) {
        unsigned int ze_marks = 0;
        rpos = d_iRZE_ranked_bitmap_2048(rpos, in, bitmap, temp_w, ze_marks);
        // The same lane owns this mask in both stages. The earlier length
        // barrier completed all upper-offset reads before ZE reuses temp_w.
        d_ZEdecode_full_byte_permute(in, rpos, ze_marks, out, temp_w);
#else
      rpos = d_iRZE_upper_byte(rpos, in, bitmap, temp_w);
      if (size == CS) {
        rpos = d_iRZE_ranked_bitmap_2048(rpos, in, bitmap, temp_w);
        __syncthreads();
        d_ZEdecode_full_byte_permute(in, rpos, bitmap, out, temp_w);
#endif
        return;
      }
      base = 2048;
      range = 256;
    } else {
      if constexpr (sizeof(T) == 8) {
        base = (2048 + 256) / sizeof(T);
        range = 32 / sizeof(T);
        // read in last level of bitmap
        rpos -= range;
        if (tid < range) bitmap[base + tid] = in[rpos + tid];
      } else {
        base = (2048 + 256 + 32) / sizeof(T);
        range = 4 / sizeof(T);
        // read in last level of bitmap
        rpos -= range;
        if (tid < range) bitmap[base + tid] = in[rpos + tid];

        rpos -= __syncthreads_count((tid < range * 8) && ((in[rpos + tid / 8] >> (tid % 8)) & 1));
        base = (2048 + 256) / sizeof(T);
        range = 32 / sizeof(T);
        d_REdecode<byte, 32 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
      }
      __syncthreads();

      rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
      base = 2048 / sizeof(T);
      range = 256 / sizeof(T);
      d_REdecode<byte, 256 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
      __syncthreads();
    }

    if constexpr (sizeof(T) >= 4) {
      rpos -= __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
    }
    if constexpr (sizeof(T) == 2) {
      int sum = __syncthreads_count((tid < range * 8) && ((bitmap[base + tid / 8] >> (tid % 8)) & 1));
      sum += __syncthreads_count((tid + TPB < range * 8) && ((bitmap[base + (tid + TPB) / 8] >> (tid % 8)) & 1));
      rpos -= sum;
    }
    if constexpr (sizeof(T) == 1) {
      int sum = 0;
      for (int i = 0; i < TPB * 4; i += TPB) {
        sum += __syncthreads_count((tid + i < range * 8) && ((bitmap[base + (tid + i) / 8] >> (tid % 8)) & 1));
      }
      rpos -= sum;
    }
    base = 0 / sizeof(T);
    range = 2048 / sizeof(T);
    d_REdecode<byte, 2048 / sizeof(T)>(range, &in[rpos], &bitmap[base + range], &bitmap[base], temp_w);
    __syncthreads();

    // copy non-zero values based on bitmap
    if (size > 0) d_ZEdecode(size, (const T*)in, (T*)bitmap, (T*)out, temp_w);
  }
}
