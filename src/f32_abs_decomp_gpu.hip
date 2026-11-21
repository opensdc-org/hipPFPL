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


#define NDEBUG

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#define WS 32


#include <cmath>
#include <cstdint>
#include <string>
#include <cassert>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include <sys/time.h>
#include "include/macros.h"
#include "include/sum_reduction.h"
#include "include/prefix_sum.h"
#include "components/d_BIT_4.h"
#include "components/d_RZE_1.h"


const int NUM_RUNS = 9;


static __device__ int g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0;
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


static __device__ inline void restore_diff_quant(byte* destination, int bytes,
                                                   int* scratch, float eb2)
{
  constexpr int words_per_thread = CS / sizeof(float) / TPB;
  static_assert(CS % (sizeof(float) * TPB) == 0);
  constexpr unsigned int diff_bias = 0xAAAAAAAAu;
  constexpr unsigned int max_quantized_code = 1u << 23;
  auto* words = reinterpret_cast<unsigned int*>(destination);
  const int word_count = bytes / sizeof(float);
  const int tid = threadIdx.x;
  const int begin = tid * word_count / TPB;
  const int count = (tid + 1) * word_count / TPB - begin;

  // Each thread owns a contiguous range. Retain its differences across the
  // block scan so that expanded quantized words can be restored in place.
  unsigned int diffs[words_per_thread];
  unsigned int sum = 0;
  #pragma unroll
  for (int i = 0; i < words_per_thread; ++i) {
    const unsigned int diff = (i < count) ? (words[begin + i] ^ diff_bias) - diff_bias : 0u;
    diffs[i] = diff;
    sum += diff;
  }
  sum = block_prefix_sum(sum, scratch);

  #pragma unroll
  for (int i = words_per_thread - 1; i >= 0; --i) {
    if (i < count) {
      unsigned int bits = sum;
      if (bits < max_quantized_code) {
        const int bin = static_cast<int>(bits >> 1) ^ -static_cast<int>(bits & 1u);
        bits = __float_as_uint(bin * eb2);
      }
      words[begin + i] = bits;
      sum -= diffs[i];
    }
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_decode(const byte* const __restrict__ input, byte* const __restrict__ output, int* const __restrict__ g_outsize)
{
  // RZE stores four levels of byte bitmaps after WS scan words.
  constexpr int rze_scratch_bytes = WS * sizeof(int) + 2048 + 256 + 32 + 4;
  __shared__ long long expanded[CS / sizeof(long long)];
  __shared__ int scratch[(rze_scratch_bytes + sizeof(int) - 1) / sizeof(int)];
  __shared__ int carry[WS];
  __shared__ int next_chunk;

  const auto* head_in = reinterpret_cast<const long long*>(input);
  const int outsize = static_cast<int>(head_in[0]);
  const float eb2 = 2 * reinterpret_cast<const float*>(&head_in[1])[0];
  const int chunks = (outsize + CS - 1) / CS;
  const auto* size_in = reinterpret_cast<const unsigned short*>(&head_in[2]);
  const auto* data_in = reinterpret_cast<const byte*>(&size_in[chunks]);
  auto* expanded_bytes = reinterpret_cast<byte*>(expanded);
  auto* scratch_bytes = reinterpret_cast<byte*>(scratch);

  const int tid = threadIdx.x;
  int prev_chunk = 0;
  int prev_offset = 0;
  while (true) {
    if (tid == 0) next_chunk = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();

    const int chunk_id = next_chunk;
    const int base = chunk_id * CS;
    if (base >= outsize) break;

    int sum = 0;
    for (int i = prev_chunk + tid; i < chunk_id; i += TPB) {
      sum += static_cast<int>(size_in[i]);
    }
    const int offset = prev_offset + block_sum_reduction(sum, carry);
    prev_chunk = chunk_id;
    prev_offset = offset;

    int stored = static_cast<int>(size_in[chunk_id]);
    const int raw_size = min(CS, outsize - base);
    const byte* source = &data_in[offset];
    byte* destination = &output[base];
    if (stored < raw_size) {
      d_iRZE_1(stored, source, expanded_bytes, scratch_bytes);
      __syncthreads();
      d_iBIT_4(stored, expanded_bytes, destination, scratch_bytes);
      __syncthreads();  // Publish inverse-bit words to the threads owning their contiguous ranges.
      restore_diff_quant(destination, stored, scratch, eb2);
    } else {
      copy_raw_chunk(source, destination, raw_size);
    }
    __syncthreads();  // A subsequent chunk reuses this block's shared buffers.
  }

  if ((blockIdx.x == 0) && (tid == 0)) {
    *g_outsize = outsize;
  }
}


struct GPUTimer
{
  hipEvent_t beg, end;
  GPUTimer() {hipEventCreate(&beg); hipEventCreate(&end);}
  ~GPUTimer() {hipEventDestroy(beg); hipEventDestroy(end);}
  void start() {hipEventRecord(beg, 0);}
  double stop() {hipEventRecord(end, 0); hipEventSynchronize(end); float ms; hipEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckHip(const int line)
{
  hipError_t e;
  hipDeviceSynchronize();
  if (hipSuccess != (e = hipGetLastError())) {
    fprintf(stderr, "HIP error %d on line %d: %s\n\n", e, line, hipGetErrorString(e));
    throw std::runtime_error("LC error");
  }
}


int main(int argc, char* argv [])
{
  printf("PFPL GPU Single-Precision ABS Decompressor\n");
  printf("Copyright 2025 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s compressed_file_name decompressed_file_name\n\n", argv[0]);  throw std::runtime_error("LC error");}

  // read input file
  FILE* const fin = fopen(argv[1], "rb");
  int pre_size = 0;
  const int pre_val = fread(&pre_size, sizeof(pre_size), 1, fin); assert(pre_val == sizeof(pre_size));
  fseek(fin, 0, SEEK_END);
  const int hencsize = ftell(fin);  assert(hencsize > 0);
  byte* const hencoded = new byte [pre_size];
  fseek(fin, 0, SEEK_SET);
  const int insize = fread(hencoded, 1, hencsize, fin);  assert(insize == hencsize);
  fclose(fin);
  printf("encoded size: %d bytes\n", insize);

  // get GPU info
  hipSetDevice(0);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no HIP capable device detected\n\n"); throw std::runtime_error("LC error");}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  CheckHip(__LINE__);

  // allocate GPU memory
  byte* ddecoded;
  hipHostMalloc((void **)&ddecoded, pre_size);
  byte* d_encoded;
  hipMalloc((void **)&d_encoded, insize);
  hipMemcpy(d_encoded, hencoded, insize, hipMemcpyHostToDevice);
  byte* d_decoded;
  hipMalloc((void **)&d_decoded, pre_size);
  int* d_decsize;
  hipMalloc((void **)&d_decsize, sizeof(int));
  CheckHip(__LINE__);

  // time GPU decoding
  GPUTimer dtimer;
  int ddecsize = 0;
  for (int i = 0; i < NUM_RUNS; i++) {
    hipDeviceSynchronize();
    dtimer.start();

    d_reset<<<1, 1>>>();
    d_decode<<<blocks, TPB>>>(d_encoded, d_decoded, d_decsize);

    hipDeviceSynchronize();
    double runtime = dtimer.stop();

    if (i < NUM_RUNS - 1) {
      hipMemset(d_decsize, 0, sizeof(int));
      hipMemset(d_decoded, 0, pre_size);
    }
    CheckHip(__LINE__);
    printf("lc decomp ecltime, %12.9f\n", runtime);
  }

  // get decoded GPU result
  hipMemcpy(&ddecsize, d_decsize, sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(ddecoded, d_decoded, ddecsize, hipMemcpyDeviceToHost);
  printf("decoded size: %d bytes\n", ddecsize);
  printf("compression ratio: %.2f\n", 1.0 * ddecsize / insize);
  CheckHip(__LINE__);

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(ddecoded, 1, ddecsize, fout);
  fclose(fout);

  // clean up GPU memory
  hipFree(d_encoded);
  hipFree(d_decoded);
  hipFree(d_decsize);
  CheckHip(__LINE__);

  // clean up
  hipHostFree(ddecoded);
  return 0;
}
