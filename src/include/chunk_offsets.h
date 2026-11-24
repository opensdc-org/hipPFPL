#pragma once

#include "prefix_sum.h"

// Scan u16 packet lengths into byte offsets. Lengths and offsets start on
// four-entry boundaries; a partial group uses scalar loads and stores.
// The input packet limit guarantees that all offsets fit in an int.
#if defined(PFPL_ABS_WAVE_OFFSETS)
static __device__ inline unsigned int restore_wave_inclusive(unsigned int count, int* scratch);

template <bool return_total, bool WavePrefix = false>
#else
template <bool return_total>
#endif
static __device__ __forceinline__ int scan_chunk_offsets(const unsigned short* lengths, int count,
    int* offsets, int payload_bytes, unsigned int* scan_scratch, unsigned int& tile_total)
{
  constexpr int lengths_per_thread = 4;
  constexpr int tile_size = TPB * lengths_per_thread;
  const auto* length_groups = reinterpret_cast<const unsigned long long*>(lengths);
  auto* offset_groups = reinterpret_cast<int4*>(offsets);
  const int tid = threadIdx.x;
#if defined(PFPL_ABS_WAVE_OFFSETS)
  static_assert(!WavePrefix || !return_total);
  if constexpr (!WavePrefix) {
#endif
    if (tid < WS) scan_scratch[tid] = 0;
    __syncthreads();
#if defined(PFPL_ABS_WAVE_OFFSETS)
  }
#endif
  for (int tile = 0; tile < count; tile += tile_size) {
    const int local = tile + tid * lengths_per_thread;
    const int remaining = count - local;
    const int group = local / lengths_per_thread;
    unsigned long long packed_lengths = 0;
    if (remaining >= lengths_per_thread) {
      packed_lengths = length_groups[group];
    } else {
      #pragma unroll
      for (int i = 0; i < lengths_per_thread; ++i) {
        if (i < remaining) {
          packed_lengths |= static_cast<unsigned long long>(lengths[local + i]) << (16 * i);
        }
      }
    }

    const unsigned int s0 = packed_lengths & 0xffffu;
    const unsigned int s1 = (packed_lengths >> 16) & 0xffffu;
    const unsigned int s2 = (packed_lengths >> 32) & 0xffffu;
    const unsigned int s3 = packed_lengths >> 48;
    const unsigned int before2 = s0 + s1;
    const unsigned int before3 = before2 + s2;
    const unsigned int tile_group_length = before3 + s3;
#if defined(PFPL_ABS_WAVE_OFFSETS)
    const unsigned int inclusive = WavePrefix
        ? restore_wave_inclusive(tile_group_length, reinterpret_cast<int*>(scan_scratch))
        : block_prefix_sum(tile_group_length, scan_scratch);
#else
    const unsigned int inclusive = block_prefix_sum(tile_group_length, scan_scratch);
#endif
    const int offset = payload_bytes + static_cast<int>(inclusive - tile_group_length);

    if (remaining >= lengths_per_thread) {
      offset_groups[group] = make_int4(offset, offset + static_cast<int>(s0),
                                      offset + static_cast<int>(before2),
                                      offset + static_cast<int>(before3));
    } else {
      if (remaining > 0) offsets[local] = offset;
      if (remaining > 1) offsets[local + 1] = offset + static_cast<int>(s0);
      if (remaining > 2) offsets[local + 2] = offset + static_cast<int>(before2);
    }

#if defined(PFPL_ABS_WAVE_OFFSETS)
    // The selected path fits one tile and needs only offsets, never its total.
    if constexpr (WavePrefix) return payload_bytes;
#endif
    if (return_total || tile + tile_size < count) {
      if (tid == TPB - 1) tile_total = inclusive;
      __syncthreads();
      payload_bytes += static_cast<int>(tile_total);
    }
  }

  return payload_bytes;
}
