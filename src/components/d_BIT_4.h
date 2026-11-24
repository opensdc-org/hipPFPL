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


template <int mask>
static __device__ __forceinline__ unsigned int bit_shuffle_xor(unsigned int value)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_ds_swizzle)
  // Bit mode routes lane XOR independently within each 32-lane half-wave.
  return __builtin_amdgcn_ds_swizzle(value, 31 | (mask << 10));
#endif
#endif
  return pfpl_shfl_xor(value, mask);
}


static __device__ inline unsigned int d_BIT_4_word(unsigned int a, const int sublane)
{
  unsigned int q = bit_shuffle_xor<16>(a);
  a = ((sublane & 16) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

  q = bit_shuffle_xor<8>(a);
  a = ((sublane & 8) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

  q = bit_shuffle_xor<4>(a);
  unsigned int mask = 0x0F0F0F0F;
  if ((sublane & 4) == 0) {
    a = (a & ~mask) | ((q >> 4) & mask);
  } else {
    a = ((q << 4) & ~mask) | (a & mask);
  }

  q = bit_shuffle_xor<2>(a);
  mask = 0x33333333;
  if ((sublane & 2) == 0) {
    a = (a & ~mask) | ((q >> 2) & mask);
  } else {
    a = ((q << 2) & ~mask) | (a & mask);
  }

  q = bit_shuffle_xor<1>(a);
  mask = 0x55555555;
  if ((sublane & 1) == 0) {
    a = (a & ~mask) | ((q >> 1) & mask);
  } else {
    a = ((q << 1) & ~mask) | (a & mask);
  }

  return a;
}


static __device__ inline bool d_BIT_4(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int SWS = 32;  // sub-warp size
  int* const in_w = (int*)in;
  int* const out_w = (int*)out;
  const int tid = threadIdx.x;
  const int sublane = tid % SWS;
  const int extra = csize % (32 * 32 / 8);
  const int size = (csize - extra) / 4;
  assert(WS % SWS == 0);

  // works for warpsize of 64 on AMD because shfl does not include sync

  for (int i = tid; i < size; i += TPB) {
    out_w[i / 32 + sublane * (size / 32)] = d_BIT_4_word(in_w[i], sublane);
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
  return true;
}


static __device__ inline unsigned int d_iBIT_4_word(unsigned int a, int sublane)
{
#if defined(PFPL_F32_ABS_RZE_DECODE)
  // Preserve the measured ABS decoder's GPU instruction sequence.
  unsigned int q = bit_shuffle_xor<16>(a);
  a = ((sublane & 16) == 0) ? __byte_perm(a, q, (3 << 12) | (2 << 8) | (7 << 4) | 6) : __byte_perm(a, q, (5 << 12) | (4 << 8) | (1 << 4) | 0);

  q = bit_shuffle_xor<8>(a);
  a = ((sublane & 8) == 0) ? __byte_perm(a, q, (3 << 12) | (7 << 8) | (1 << 4) | 5) : __byte_perm(a, q, (6 << 12) | (2 << 8) | (4 << 4) | 0);

  q = bit_shuffle_xor<4>(a);
  unsigned int mask = 0x0F0F0F0F;
  if ((sublane & 4) == 0) {
    a = (a & ~mask) | ((q >> 4) & mask);
  } else {
    a = (a & mask) | ((q << 4) & ~mask);
  }

  q = bit_shuffle_xor<2>(a);
  mask = 0x33333333;
  if ((sublane & 2) == 0) {
    a = (a & ~mask) | ((q >> 2) & mask);
  } else {
    a = (a & mask) | ((q << 2) & ~mask);
  }

  q = bit_shuffle_xor<1>(a);
  mask = 0x55555555;
  if ((sublane & 1) == 0) {
    a = (a & ~mask) | ((q >> 1) & mask);
  } else {
    a = (a & mask) | ((q << 1) & ~mask);
  }
  return a;
#else
  return d_BIT_4_word(a, sublane);
#endif
}


static __device__ inline void d_iBIT_4(int& csize, byte in [CS], byte out [CS], byte temp [CS])
{
  const int SWS = 32;  // sub-warp size
  int* const in_w = (int*)in;
  int* const out_w = (int*)out;
  const int tid = threadIdx.x;
  const int sublane = tid % SWS;
  const int extra = csize % (32 * 32 / 8);
  const int size = (csize - extra) / 4;
  assert(WS % SWS == 0);

  for (int i = tid; i < size; i += TPB) {
    out_w[i] = d_iBIT_4_word(in_w[i / 32 + sublane * (size / 32)], sublane);
  }

  // copy leftover bytes
  if (tid < extra) out[csize - extra + tid] = in[csize - extra + tid];
}
