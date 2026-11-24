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
#include <string>
#include <cassert>
#include <stdexcept>
#include <hip/hip_runtime.h>
#define PFPL_ABS_REGISTER_ZE
#define PFPL_ABS_WAVE_OFFSETS
#include "include/f32_decode_common.h"
#undef PFPL_ABS_WAVE_OFFSETS
#undef PFPL_ABS_REGISTER_ZE
#include "include/f32_file_io.h"


const int NUM_RUNS = 9;


static __device__ __forceinline__ void restore_diff_quant(const byte* differences,
    byte* destination, int bytes, int* scratch, float eb2)
{
  constexpr int words_per_thread = CS / sizeof(float) / TPB;
  static_assert(CS % (sizeof(float) * TPB) == 0);
  constexpr unsigned int diff_bias = 0xAAAAAAAAu;
  constexpr unsigned int max_quantized_code = 1u << 23;
  const auto* diff_words = reinterpret_cast<const unsigned int*>(differences);
  auto* output_words = reinterpret_cast<unsigned int*>(destination);
  const int word_count = bytes / sizeof(float);
  const int tid = threadIdx.x;
  const int begin = tid * word_count / TPB;
  const int count = (tid + 1) * word_count / TPB - begin;

  // Each thread owns a contiguous range and retains its differences across
  // the block scan. Input and output can alias for a partial final chunk.
  unsigned int diffs[words_per_thread];
  unsigned int sum = 0;
  #pragma unroll
  for (int i = 0; i < words_per_thread; ++i) {
    const unsigned int diff = (i < count) ? (diff_words[begin + i] ^ diff_bias) - diff_bias : 0u;
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
      output_words[begin + i] = bits;
      sum -= diffs[i];
    }
  }
}

static __device__ __forceinline__ void restore_diff_quant_full(const byte* differences,
    byte* destination, int* scratch, float eb2)
{
  constexpr int words_per_thread = CS / sizeof(float) / TPB;
  static_assert(CS % (sizeof(float) * TPB) == 0);
  constexpr unsigned int diff_bias = 0xAAAAAAAAu;
  constexpr unsigned int max_quantized_code = 1u << 23;
  const auto* diff_words = reinterpret_cast<const unsigned int*>(differences);
  auto* output_words = reinterpret_cast<unsigned int*>(destination);
  const int tid = threadIdx.x;
  const int begin = tid * words_per_thread;

  // Each thread owns a contiguous range and retains its differences across
  // the block scan. Input and output can alias for a partial final chunk.
  unsigned int diffs[words_per_thread];
  unsigned int sum = 0;
  #pragma unroll
  for (int i = 0; i < words_per_thread; ++i) {
    const unsigned int diff = (diff_words[begin + i] ^ diff_bias) - diff_bias;
    diffs[i] = diff;
    sum += diff;
  }
  sum = restore_wave_inclusive(sum, scratch);

  #pragma unroll
  for (int i = words_per_thread - 1; i >= 0; --i) {
    unsigned int bits = sum;
    if (bits < max_quantized_code) {
      const int bin = static_cast<int>(bits >> 1) ^ -static_cast<int>(bits & 1u);
      bits = __float_as_uint(bin * eb2);
    }
    output_words[begin + i] = bits;
    sum -= diffs[i];
  }
}


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
  const float eb2 = 2 * reinterpret_cast<const float*>(&head_in[1])[0];
  const auto* size_in = reinterpret_cast<const unsigned short*>(&head_in[2]);
  const auto* data_in = reinterpret_cast<const byte*>(&size_in[chunks]);
  auto* expanded_bytes = reinterpret_cast<byte*>(expanded);
  auto* scratch_bytes = reinterpret_cast<byte*>(scratch);

  const int tid = threadIdx.x;
  // Each CTA starts with a distinct chunk. Dynamic tickets start at gridDim.x.
  int chunk_id = static_cast<int>(blockIdx.x);
  for (int slot = 0; chunk_id < chunks; slot ^= 1) {
    const int base = chunk_id * CS;
    const int offset = offsets[chunk_id];

    int stored = static_cast<int>(size_in[chunk_id]);
    const int raw_size = min(CS, outsize - base);
    const byte* source = &data_in[offset];
    byte* destination = &output[base];
    if (stored < raw_size) {
      d_iRZE<byte, true>(stored, source, expanded_bytes, scratch_bytes);
      __syncthreads();
      if (stored == CS) {
        inverse_bit_full_chunk(expanded_bytes);
        __syncthreads();  // Publish inverse-bit words in shared memory.
        restore_diff_quant_full(expanded_bytes, destination, scratch, eb2);
      } else {
        d_iBIT_4(stored, expanded_bytes, destination, scratch_bytes);
        __syncthreads();  // Publish the temporary words in the output range.
        restore_diff_quant(destination, destination, stored, scratch, eb2);
      }
    } else {
      copy_raw_chunk(source, destination, raw_size);
    }
    // Slower lanes can still read this ticket while the leader writes the other slot.
    if (tid == 0) ticket[slot] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // Publish the next ticket and finish using this chunk's LDS.
    chunk_id = ticket[slot];
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


int main(int argc, char* argv []) try
{
  printf("PFPL GPU Single-Precision ABS Decompressor\n");
  printf("Copyright 2025 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s compressed_file_name decompressed_file_name\n\n", argv[0]);  throw std::runtime_error("LC error");}

  int pre_size, chunks, insize;
  auto hencoded = read_f32_packet(argv[1], pre_size, chunks, insize);
  printf("encoded size: %d bytes\n", insize);

  // get GPU info
  hipSetDevice(0);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no HIP capable device detected\n\n"); throw std::runtime_error("LC error");}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = decode_grid_for_packet(SMs * (mTpSM / TPB), chunks);
  CheckHip(__LINE__);

  // allocate GPU memory
  byte* ddecoded;
  hipHostMalloc((void **)&ddecoded, pre_size);
  byte* d_encoded;
  hipMalloc((void **)&d_encoded, insize);
  hipMemcpy(d_encoded, hencoded.get(), insize, hipMemcpyHostToDevice);
  hencoded.reset();
  byte* d_decoded;
  hipMalloc((void **)&d_decoded, pre_size);
  int* d_decsize;
  hipMalloc((void **)&d_decsize, sizeof(int));
  int* d_offsets;
  const auto offset_alloc = hipMalloc((void **)&d_offsets, sizeof(int) * chunks);
  if (offset_alloc != hipSuccess) throw std::runtime_error(hipGetErrorString(offset_alloc));
  CheckHip(__LINE__);

  // time GPU decoding
  GPUTimer dtimer;
  int ddecsize = 0;
  for (int i = 0; i < NUM_RUNS; i++) {
    hipDeviceSynchronize();
    dtimer.start();

    d_chunk_offsets<<<1, TPB>>>(d_encoded, chunks, d_offsets, blocks);
    d_decode<<<blocks, TPB>>>(d_encoded, d_decoded, d_decsize, d_offsets, chunks);

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
  if (hipMemcpy(&ddecsize, d_decsize, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess || ddecsize != pre_size)
    throw std::runtime_error("Decoded size does not match encoded header");
  if (hipMemcpy(ddecoded, d_decoded, ddecsize, hipMemcpyDeviceToHost) != hipSuccess)
    throw std::runtime_error("Cannot read decoded data");
  printf("decoded size: %d bytes\n", ddecsize);
  printf("compression ratio: %.2f\n", 1.0 * ddecsize / insize);
  CheckHip(__LINE__);

  // write to file
  write_f32_output(argv[2], ddecoded, ddecsize);

  // clean up GPU memory
  hipFree(d_encoded);
  hipFree(d_decoded);
  hipFree(d_decsize);
  hipFree(d_offsets);
  CheckHip(__LINE__);

  // clean up
  hipHostFree(ddecoded);
  return 0;
}
catch (const std::exception& error)
{
  fprintf(stderr, "%s\n", error.what());
  return 1;
}
