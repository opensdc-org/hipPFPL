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

// Shared FP32 encoder helpers. Include this after the encoder's RZE components
// and the translation unit's chunk, block, and logical wave sizes.


// Copy len bytes to global memory. The source must be word-aligned.
static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}


static inline __device__ void propagate_carry(const int value, const int chunkID, volatile int* const __restrict__ fullcarry, int* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? value : -value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const int cidm1ml = chunkID - 1 - lane;
      int val = -1;
      pfpl_syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = fullcarry[cidm1ml];
        }
      } while ((pfpl_any_sync(~0, val == 0)) || (pfpl_all_sync(~0, val <= 0)));
      const int mask = pfpl_ballot_sync(~0, val > 0);
      const int pos = __ffs(mask) - 1;
      int partc = (lane < pos) ? -val : 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      partc = __reduce_add_sync(~0, partc);
#else
      partc += pfpl_shfl_xor_sync(~0, partc, 1);
      partc += pfpl_shfl_xor_sync(~0, partc, 2);
      partc += pfpl_shfl_xor_sync(~0, partc, 4);
      partc += pfpl_shfl_xor_sync(~0, partc, 8);
      partc += pfpl_shfl_xor_sync(~0, partc, 16);
#endif
      if (lane == pos) {
        const int fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}


static __device__ inline void load_raw_chunk(const byte* source, byte* slab, int bytes)
{
  const int tid = threadIdx.x;
  const auto* source_words = reinterpret_cast<const long long*>(source);
  auto* slab_words = reinterpret_cast<long long*>(slab);
  for (int i = tid; i < bytes / 8; i += TPB) {
    slab_words[i] = source_words[i];
  }
  const int extra = bytes % 8;
  if (tid < extra) slab[bytes - extra + tid] = source[bytes - extra + tid];
}


static __device__ inline void d_DIFFNB_BIT_4_in_place(int csize, byte* slab)
{
  constexpr int width = 32;
  constexpr int words_per_thread = CS / sizeof(unsigned int) / TPB;
  constexpr unsigned int bias = 0xAAAAAAAAu;
  static_assert(CS % (sizeof(unsigned int) * TPB) == 0 && TPB % width == 0);

  auto* words = reinterpret_cast<unsigned int*>(slab);
  const int count = csize / sizeof(unsigned int);
  const int transposed = count - count % width;
  const int tid = threadIdx.x;
  const int lane = tid % width;
  // Spread the full chunk's bit-plane stores across LDS banks.
  const int bank_mask = (csize == CS) ? lane : 0;
  unsigned int staged[words_per_thread];

  #pragma unroll
  for (int row = 0; row < words_per_thread; ++row) {
    const int i = tid + row * TPB;
    if (i < count) {
      const unsigned int previous = (i == 0) ? 0u : words[i - 1];
      const unsigned int diff = ((words[i] - previous) + bias) ^ bias;
      // In a partial chunk the remaining complete words are differenced
      // without a bit transpose. Only whole logical warps call the shuffle.
      staged[row] = (i < transposed) ? d_BIT_4_word(diff, lane) : diff;
    }
  }
  __syncthreads();  // Permuted writes can overlap any lane's old input.
  #pragma unroll
  for (int row = 0; row < words_per_thread; ++row) {
    const int i = tid + row * TPB;
    if (i < count) {
      const int out = (i < transposed) ? i / width + lane * (transposed / width) : i;
      words[out ^ bank_mask] = staged[row];
    }
  }
}


// The byte-RZE bitmap follows WS+1 scan words in shared scratch.
static constexpr int rze_scratch_bytes = (WS + 1) * sizeof(int) + 2048 + 256 + 32 + 4;
static_assert(rze_scratch_bytes % sizeof(int) == 0);


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


static __device__ __forceinline__ unsigned int transpose_forward_word(unsigned int a, int sublane,
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


// Gather the high bit of each nonzero byte into one four-bit bitmap.
static __device__ __forceinline__ unsigned int encode_nonzero_byte_mask(unsigned int word)
{
  constexpr unsigned int low = 0x7f7f7f7fu;
  constexpr unsigned int high = 0x80808080u;
  const unsigned int nonzero = (((word & low) + low) | word) & high;
  return (nonzero * 0x00204081u) >> 28;
}


// Fuse the full-chunk bit transpose with byte zero elimination. Each logical
// warp handles a consecutive span of input. Redistribute its 32-byte segments
// through the original slab so neighboring writer lanes own adjacent segments.
static __device__ inline bool d_DIFFNB_BIT_ZE_4_full_chunk(int& csize, byte* slab,
                                                          int* scratch)
{
  static_assert(CS == 16384 && TPB == 512 && WS == 32);
  constexpr int word_bytes = sizeof(unsigned int);
  constexpr int rows_per_warp = CS / word_bytes / TPB;
  constexpr int warps = TPB / WS;
  constexpr unsigned int bias = 0xAAAAAAAAu;
  const int tid = threadIdx.x;
  const int lane = tid % WS;
  const int warp = tid / WS;
  const int group = lane * warps + warp;
  // Store one word per group in each row. Swizzle the group index for
  // both the producer's warp order and the consumer's bit-plane order.
  const int producer = group ^ (group / warps);
  const int consumer = tid ^ (tid / warps);
  auto* words = reinterpret_cast<unsigned int*>(slab);
  unsigned int transformed[rows_per_warp];
  unsigned int bits = 0;
  const unsigned int permute16 = (lane & 16) ? 0x05040100u : 0x03020706u;
  const unsigned int permute8 = (lane & 8) ? 0x06020400u : 0x03070105u;

  #pragma unroll
  for (int row = 0; row < rows_per_warp; ++row) {
    const int i = (warp * rows_per_warp + row) * WS + lane;
    const unsigned int previous = (i == 0) ? 0u : words[i - 1];
    const unsigned int diff = ((words[i] - previous) + bias) ^ bias;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_perm)
    const unsigned int word = transpose_forward_word(diff, lane, permute16, permute8);
#else
    const unsigned int word = d_BIT_4_word(diff, lane);
#endif
#else
    const unsigned int word = d_BIT_4_word(diff, lane);
#endif
    transformed[row] = word;
    bits |= encode_nonzero_byte_mask(word) << (4 * row);
  }

  auto* bitmap = reinterpret_cast<unsigned int*>(scratch + WS + 1);
  bitmap[group] = bits;
  if (tid < WS) scratch[tid] = 0;
  __syncthreads();  // The bitmap and all reads of the quantized input are complete.

  if (bits != 0) {
    #pragma unroll
    for (int row = 0; row < rows_per_warp; ++row) {
      words[row * TPB + producer] = transformed[row];
    }
  }

  // In scan order, each lane owns 32 ZE bytes and four first-layer RE bytes.
  constexpr int ze_bits = 15;
  constexpr int ze_mask = (1 << ze_bits) - 1;
  constexpr int bitmap_bytes = CS / 8;
  constexpr int avail = CS - 2;
  static_assert(CS <= ze_mask && (static_cast<long long>(bitmap_bytes) << ze_bits) + CS <= INT_MAX);

  auto* bitmap_bytes_ptr = reinterpret_cast<byte*>(bitmap);
  const unsigned int bitmap_word = bitmap[tid];
  const unsigned int previous = (tid == 0) ? 0u : bitmap_bytes_ptr[tid * word_bytes - 1];
  const unsigned int changed = encode_nonzero_byte_mask(bitmap_word ^ ((bitmap_word << 8) | previous));
  const int pair = __popc(bitmap_word) | (__popc(changed) << ze_bits);
  const int prefix = byte_ze_exclusive_offset(pair, scratch);
  if (tid == TPB - 1) scratch[WS] = prefix + pair;
  __syncthreads();  // Publish redistributed words and the packed total.

  const int total = scratch[WS];
  const int ze_length = total & ze_mask;
  const int wpos = ze_length + (total >> ze_bits);
  if (wpos >= avail) return false;

  if (ze_length != 0) {
    unsigned int scatter_words[rows_per_warp];
    if (bitmap_word != 0) {
      #pragma unroll
      for (int row = 0; row < rows_per_warp; ++row) {
        scatter_words[row] = words[row * TPB + consumer];
      }
    }
#if !defined(PFPL_ABS_EARLY_RZE_UPPER)
    __syncthreads();  // Finish all exchange reads before any payload overwrites the slab.
#endif
    const unsigned int other = pfpl_shfl_xor_sync(~0, changed, 1, WS);
    if ((tid & 1) == 0) bitmap_bytes_ptr[bitmap_bytes + tid / 2] = changed | (other << 4);
#if defined(PFPL_ABS_EARLY_RZE_UPPER)
    __syncthreads();  // Publish the upper bitmap together with the last exchange reads.
#endif
    int re_pos = ze_length + (prefix >> ze_bits);
    #pragma unroll
    for (int i = 0; i < word_bytes; ++i) {
      if ((changed >> i) & 1u) slab[re_pos++] = bitmap_word >> (8 * i);
    }

    if (bitmap_word != 0) {
      int pos = prefix & ze_mask;
      #pragma unroll
      for (int i = 0; i < rows_per_warp * word_bytes; ++i) {
        if ((bitmap_word >> i) & 1u) {
          slab[pos++] = scatter_words[i / word_bytes] >> ((i % word_bytes) * 8);
        }
      }
    }
#if defined(PFPL_ABS_EARLY_RZE_UPPER)
    // The upper warp can write past wpos while other waves finish the lower payload.
    // Its final barrier publishes all payload and the footer before staging.
    int upper_bytes = avail - wpos;
    if (!encode_rze_upper_256(bitmap_bytes_ptr + bitmap_bytes, slab + wpos,
                              upper_bytes, &scratch[WS], csize)) return false;
    csize = wpos + upper_bytes + 2;
    return true;
#endif
  }
  return d_RZE_finish_full_byte(csize, slab, reinterpret_cast<byte*>(scratch), wpos);
}


// Bound staging to 64 MiB regardless of the input size. Every window is
// completed in stream order before its slots and offsets can be reused.
static constexpr int STAGED_CHUNK_LIMIT = 4096;


// Source slots start at 16-byte boundaries. Align the destination with exact
// byte copies, then write complete vectors inside this chunk's packet range.
template <int Lanes>
static __device__ inline void copy_staged_packet(byte* __restrict__ destination,
                                                 const byte* __restrict__ source, int length)
{
  constexpr int word_bytes = sizeof(unsigned int);
  constexpr int vector_bytes = sizeof(uint4);
  static_assert(word_bytes == 4 && vector_bytes == 16);
  const int tid = threadIdx.x % Lanes;
  if (length < 128) {
    for (int i = tid; i < length; i += Lanes) destination[i] = source[i];
    return;
  }

  const auto address = reinterpret_cast<std::size_t>(destination);
  const int head = static_cast<int>(-address & (vector_bytes - 1));
  const int source_byte_offset = head & (word_bytes - 1);
  // For a shifted vector, the final aligned source word can extend beyond
  // the 16 useful bytes. Leave that vector to the tail if the word would
  // cross the exact source length, including an unpadded raw final chunk.
  const int source_guard = (word_bytes - source_byte_offset) & (word_bytes - 1);
  const int vectors = (length - head - source_guard) / vector_bytes;
  auto* const output_vectors = reinterpret_cast<uint4*>(destination + head);

  if (head == 0) {
    const auto* const input_vectors = reinterpret_cast<const uint4*>(source);
    for (int i = tid; i < vectors; i += Lanes) output_vectors[i] = input_vectors[i];
  } else {
    const auto* const input_words = reinterpret_cast<const unsigned int*>(source) + head / word_bytes;
    const int shift = 8 * source_byte_offset;
    for (int i = tid; i < vectors; i += Lanes) {
      const auto* const words = input_words + i * (vector_bytes / word_bytes);
      const unsigned int first = words[0];
      const unsigned int second = words[1];
      const unsigned int third = words[2];
      const unsigned int fourth = words[3];
      if (shift == 0) {
        output_vectors[i] = make_uint4(first, second, third, fourth);
      } else {
        output_vectors[i] = make_uint4(__funnelshift_r(first, second, shift),
                                       __funnelshift_r(second, third, shift),
                                       __funnelshift_r(third, fourth, shift),
                                       __funnelshift_r(fourth, words[4], shift));
      }
    }
  }

  if (tid < head) destination[tid] = source[tid];
  const int tail = head + vectors * vector_bytes + tid;
  if (tail < length) destination[tail] = source[tail];
}


static constexpr int PACK_THREADS_PER_SLOT = 64;
static constexpr int PACK_SLOTS_PER_BLOCK = TPB / PACK_THREADS_PER_SLOT;


static __global__ __launch_bounds__(TPB)
void d_pack_staged(const byte* const __restrict__ input, const int insize,
                   const int first, const int chunks, const byte* const __restrict__ staging,
                   const int* const __restrict__ offsets, const unsigned short* const __restrict__ lengths,
                   byte* const output, const int* const __restrict__ outsize)
{
  if (*outsize <= 0) return;
  const int slot = blockIdx.x;
  const int chunk_id = first + slot;
  const int base = chunk_id * CS;
  const int raw_size = min(CS, insize - base);
  const int length = lengths[chunk_id];
  const bool raw = (length == raw_size);
  const byte* source = raw ? &input[base] : &staging[slot * CS];
  auto* destination = &output[16 + 2 * chunks + offsets[slot]];

  copy_staged_packet<TPB>(destination, source, length);
}


// The later block reduction synchronizes these first-warp writes.
static __device__ __forceinline__ void write_local_pack_offsets(
    int tid, int block_start, int chunks, const unsigned short* lengths, int* local_offsets)
{
  if (tid < WS) {
    const int length = (tid < PACK_SLOTS_PER_BLOCK && block_start + tid < chunks)
                     ? lengths[block_start + tid] : 0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_ds_swizzle)
    static_assert(PACK_SLOTS_PER_BLOCK == 8);
    int total = length;
    int offset = 0;
    const int first = __builtin_amdgcn_ds_swizzle(total, 31 | (1 << 10));
    if (tid & 1) offset += first;
    total += first;
    const int second = __builtin_amdgcn_ds_swizzle(total, 31 | (2 << 10));
    if (tid & 2) offset += second;
    total += second;
    const int third = __builtin_amdgcn_ds_swizzle(total, 31 | (4 << 10));
    if (tid & 4) offset += third;
    total += third;
    if (tid < PACK_SLOTS_PER_BLOCK) local_offsets[tid] = offset;
    if (tid == PACK_SLOTS_PER_BLOCK - 1) local_offsets[PACK_SLOTS_PER_BLOCK] = total;
    return;
#endif
#endif
    int inclusive = length;
    #pragma unroll
    for (int distance = 1; distance < PACK_SLOTS_PER_BLOCK; distance <<= 1) {
      const int previous = pfpl_shfl_up_sync(~0u, inclusive, distance, WS);
      if (tid >= distance) inclusive += previous;
    }
    if (tid < PACK_SLOTS_PER_BLOCK) local_offsets[tid] = inclusive - length;
    if (tid == PACK_SLOTS_PER_BLOCK - 1) local_offsets[PACK_SLOTS_PER_BLOCK] = inclusive;
  }
}


static __device__ __forceinline__ unsigned int fused_pack_block_sum(unsigned int sum,
                                                                        unsigned int* wave_totals)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_ds_swizzle)
  constexpr int logical_waves = TPB / WS;
  static_assert(WS == 32 && logical_waves == 16);
  const int lane = threadIdx.x % WS;
  sum += bit_shuffle_xor<1>(sum);
  sum += bit_shuffle_xor<2>(sum);
  sum += bit_shuffle_xor<4>(sum);
  sum += bit_shuffle_xor<8>(sum);
  sum += bit_shuffle_xor<16>(sum);
  if (lane == 0) wave_totals[threadIdx.x / WS] = sum;
  __syncthreads();  // Offsets and logical wave totals were published together.
  sum = lane < logical_waves ? wave_totals[lane] : 0u;
  sum += bit_shuffle_xor<1>(sum);
  sum += bit_shuffle_xor<2>(sum);
  sum += bit_shuffle_xor<4>(sum);
  sum += bit_shuffle_xor<8>(sum);
  return sum + bit_shuffle_xor<16>(sum);
#endif
#endif
  return block_sum_reduction(sum, wave_totals);
}


static void check_compression_hip(hipError_t status, const char* operation)
{
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
  }
}


struct EncodeScratch
{
  explicit EncodeScratch(int insize)
  {
    const int chunks = 1 + (insize - 1) / CS;
    const int cap = (chunks < STAGED_CHUNK_LIMIT) ? chunks : STAGED_CHUNK_LIMIT;
    // Align the offsets following the staging area for grouped int4 stores.
    constexpr std::size_t offset_alignment = alignof(int4);
    const std::size_t aligned_input = (static_cast<std::size_t>(insize) + offset_alignment - 1)
                                   & ~(offset_alignment - 1);
    const std::size_t window_bytes = static_cast<std::size_t>(cap) * CS;
    staging_bytes = (aligned_input < window_bytes) ? aligned_input : window_bytes;

    // One allocation owns both the aligned packet slots and the int offsets.
    void* allocated = nullptr;
    const auto status = hipMalloc(&allocated, staging_bytes + static_cast<std::size_t>(cap) * sizeof(int));
    if (status == hipSuccess) {
      storage = allocated;
      return;
    }
    if (status != hipErrorOutOfMemory) check_compression_hip(status, "allocate staged compression scratch");
    // Only the expected out-of-memory result selects the small fallback.
    static_cast<void>(hipGetLastError());
    allocated = nullptr;
    staging_bytes = 0;
    check_compression_hip(hipMalloc(&allocated, static_cast<std::size_t>(chunks + 1) * sizeof(int)),
                          "allocate fallback compression scratch");
    storage = allocated;
  }

  EncodeScratch(const EncodeScratch&) = delete;
  EncodeScratch& operator=(const EncodeScratch&) = delete;
  ~EncodeScratch() { static_cast<void>(hipFree(storage)); }

  void* storage = nullptr;
  std::size_t staging_bytes = 0;
};
