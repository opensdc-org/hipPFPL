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


#ifndef GPU_RZE
#define GPU_RZE


#include "d_zero_elimination.h"
#include "d_repetition_elimination.h"


// One thread owns four consecutive bytes of the full first-level bitmap.
// Compare the bytes with their predecessors and pack the resulting nibble.
static __device__ inline bool encode_rze_bitmap_2048(const byte* in, byte* output,
                                                     int& capacity, byte* mask, int* scratch)
{
  if constexpr (TPB != 512 || WS != 32) {
    return false;
  } else {
    const int tid = threadIdx.x;
    const int start = tid * sizeof(unsigned int);
    const unsigned int word = reinterpret_cast<const unsigned int*>(in)[tid];
    const unsigned int previous = (tid == 0) ? 0 : in[start - 1];
    const unsigned int changed = d_nonzero_byte_mask(word ^ ((word << 8) | previous));
    const int count = __popc(changed);
    const int inclusive = block_prefix_sum(count, scratch);
    if (tid == TPB - 1) scratch[WS] = inclusive;
    __syncthreads();
    const int length = scratch[WS];
    if (length > capacity) return false;

    const unsigned int other = pfpl_shfl_xor_sync(~0, changed, 1, WS);
    if ((tid & 1) == 0) mask[tid / 2] = changed | (other << 4);
    int position = inclusive - count;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      if ((changed >> i) & 1u) output[position++] = word >> (8 * i);
    }
    capacity = length;
    return true;
  }
}


// Finish the common RZE stream after zero elimination has populated the payload
// and the first bitmap. wpos is measured in bytes.
template <typename T, bool PackedFirstBitmap = false>
static __device__ inline bool d_RZE_finish(int& csize, const byte in [CS],
                                         byte out [CS], byte temp [CS], int wpos)
{
  const int tid = threadIdx.x;
  const int extra = csize % sizeof(T);
  const int avail = CS - 2 - extra;
  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (wpos >= avail) return false;
  __syncthreads();

  // check if not all zeros
  if (wpos != 0) {
    // iteratively compress bitmaps
    int base = 0 / sizeof(T);
    int range = 2048 / sizeof(T);
    int cnt = avail - wpos;
    if constexpr (PackedFirstBitmap) {
      static_assert(sizeof(T) == 1 && TPB == 512 && WS == 32);
      if (!encode_rze_bitmap_2048(bitmap, out + wpos, cnt, bitmap + range, temp_w)) return false;
    } else {
      if (!d_REencode<byte, 2048 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    }
    wpos += cnt;
    __syncthreads();

    base = 2048 / sizeof(T);
    range = 256 / sizeof(T);
    cnt = avail - wpos;
    if (!d_REencode<byte, 256 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
    wpos += cnt;
    __syncthreads();

    base = (2048 + 256) / sizeof(T);
    range = 32 / sizeof(T);
    if constexpr (sizeof(T) < 8) {
      cnt = avail - wpos;
      if (!d_REencode<byte, 32 / sizeof(T), true>(&bitmap[base], range, &out[wpos], cnt, &bitmap[base + range], temp_w)) return false;
      wpos += cnt;

      base = (2048 + 256 + 32) / sizeof(T);
      range = 4 / sizeof(T);
    }

    // output last level of bitmap
    if (wpos >= avail - range) return false;
    if (tid < range) {  // 4 / sizeof(T)
      out[wpos + tid] = bitmap[base + tid];
    }
    wpos += range;
  }

  // copy leftover bytes
  if constexpr (sizeof(T) > 1) {
    if (tid < extra) out[wpos + tid] = in[csize - extra + tid];
  }

  // output old csize and update csize
  const int new_size = wpos + 2 + extra;
  if (tid == 0) {
    out[new_size - 2] = csize;  // bottom byte
    out[new_size - 1] = csize >> 8;  // second byte
  }
  csize = new_size;
  return true;
}


template <typename T, bool InPlace = false>
static __device__ inline bool d_RZE(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int tid = threadIdx.x;
  const int size = csize / sizeof(T);  // words in chunk (rounded down)
  const int bits = 8 * sizeof(T);
  assert(CS == 16384);

  // zero out end of bitmap
  int* const temp_w = (int*)temp;
  byte* const bitmap = (byte*)&temp_w[WS + 1];
  if (csize < CS) {
    for (int i = csize / bits + tid; i < CS / bits; i += TPB) {
      bitmap[i] = 0;
    }
    __syncthreads();
  }

  // copy non-zero values and generate bitmap
  int wpos = 0;
  if (size > 0) {
    if constexpr (InPlace) {
      static_assert(std::is_same<T, byte>::value && CS == 16384 && TPB == 512);
      d_ZEencode_byte_in_place(out, size, wpos, bitmap, temp_w);
    } else {
      d_ZEencode((T*)in, size, (T*)out, wpos, (T*)bitmap, temp_w);
    }
  }
  return d_RZE_finish<T>(csize, in, out, temp, wpos * sizeof(T));
}


// Restore the first two byte-RZE bitmap levels from a warp-local snapshot.
// Scratch scan words contain the exclusive offsets of the 32 local masks;
// the obsolete top level holds the common cursor for their payload.
static __device__ inline int d_iRZE_upper_byte(int cursor, const byte* in, byte* bitmap, int* temp_w)
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

    const int changes = __popc(mask);
    int inclusive = changes;
    #pragma unroll
    for (int distance = 1; distance < width; distance <<= 1) {
      const int prior = pfpl_shfl_up_sync(~0u, inclusive, distance, width);
      if (tid >= distance) inclusive += prior;
    }
    temp_w[tid] = inclusive - changes;
    if (tid == width - 1) *layer256_start = layer32_start - inclusive;
  }
  __syncthreads();  // Publish the 32 masks, offsets and common cursor.

  const int start = *layer256_start;
  if (tid < 256) {
    const int byte_index = tid / 8;
    const int bit = tid % 8;
    const unsigned int mask = bitmap[layer32_offset + byte_index];
    const int through = temp_w[byte_index] + __popc(mask & ((1u << bit) - 1u)) +
                        static_cast<int>((mask >> bit) & 1u);
    bitmap[layer256_offset + tid] = (through == 0) ? 0 : in[start + through - 1];
  }
  __syncthreads();  // Publish the reconstructed 256-byte level to the wide decode.
  return start;
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
      static_assert(std::is_same<T, byte>::value && WS == 32);
      rpos = d_iRZE_upper_byte(rpos, in, bitmap, temp_w);
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


#endif
