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


#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include <sys/time.h>
#include "include/macros.h"
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
#include "components/d_BIT_4.h"
#include "components/d_RZE_1.h"


const int NUM_RUNS = 9;


// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
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


static __device__ int g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0;
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
#if defined(WS) && (WS == 64)
      const long long mask = pfpl_ballot_sync(~0, val > 0);
      const int pos = __ffsll(mask) - 1;
#else
      const int mask = pfpl_ballot_sync(~0, val > 0);
      const int pos = __ffs(mask) - 1;
#endif
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


static __device__ inline void d_QABS_4_in_place(int csize, byte* slab, float errorbound, float threshold)
{
  const int words = csize / sizeof(float);
  const int maxbin = 1 << 22;  // Leave one mantissa bit for sign.
  const float eb2 = 2 * errorbound;
  const float inv_eb2 = 0.5f / errorbound;
  auto* values = reinterpret_cast<unsigned int*>(slab);

  for (int i = threadIdx.x; i < words; i += TPB) {
    const unsigned int original_bits = values[i];
    const float original = __uint_as_float(original_bits);
    const float scaled = original * inv_eb2;
    const int bin = static_cast<int>(roundf(scaled));
    const float reconstructed = bin * eb2;

    if ((bin >= maxbin) || (bin <= -maxbin) || (fabsf(original) >= threshold) ||
        (fabsf(original - reconstructed) > errorbound) || (original != original)) {
      values[i] = original_bits;
    } else {
      const unsigned int sign = 0u - static_cast<unsigned int>(bin < 0);
      values[i] = (static_cast<unsigned int>(bin) << 1) ^ sign;
    }
  }
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
      words[out] = staged[row];
    }
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode(const byte* const __restrict__ input, const int insize, byte* const __restrict__ output, int* const __restrict__ outsize, int* const __restrict__ fullcarry, const float errorbound, const float threshold)
{
  // The byte-RZE bitmap follows WS+1 scan words in the shared scratch.
  constexpr int scratch_bytes = (WS + 1) * sizeof(int) + 2048 + 256 + 32 + 4;
  static_assert(scratch_bytes % sizeof(int) == 0);
  __shared__ long long chunk[CS / sizeof(long long)];
  __shared__ int scratch[scratch_bytes / sizeof(int)];
  __shared__ int next_chunk;

  auto* slab = reinterpret_cast<byte*>(chunk);
  auto* scratch_data = reinterpret_cast<byte*>(scratch);
  const int tid = threadIdx.x;
  const int chunks = (insize + CS - 1) / CS;
  auto* head_out = reinterpret_cast<long long*>(output);
  auto* size_out = reinterpret_cast<unsigned short*>(&head_out[2]);
  auto* data_out = reinterpret_cast<byte*>(&size_out[chunks]);

  while (true) {
    if (tid == 0) next_chunk = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();
    const int chunk_id = next_chunk;
    if (chunk_id >= chunks) break;
    const int base = chunk_id * CS;
    const int raw_size = min(CS, insize - base);
    const byte* source = &input[base];

    load_raw_chunk(source, slab, raw_size);
    __syncthreads();
    d_QABS_4_in_place(raw_size, slab, errorbound, threshold);
    __syncthreads();
    d_DIFFNB_BIT_4_in_place(raw_size, slab);
    __syncthreads();

    int csize = raw_size;
    const bool good = d_RZE_1_in_place(csize, slab, scratch_data);
    __syncthreads();

    if (!good || (csize >= raw_size)) csize = raw_size;
    propagate_carry(csize, chunk_id, fullcarry, scratch);

    if (tid == 0) size_out[chunk_id] = csize;
    if (csize == raw_size) load_raw_chunk(source, slab, raw_size);
    __syncthreads();

    const int offset = (chunk_id == 0) ? 0 : scratch[0];
    s2g(&data_out[offset], slab, csize);

    if (tid == 0 && chunk_id == chunks - 1) {
      head_out[0] = static_cast<long long>(insize);
      reinterpret_cast<float*>(&head_out[1])[0] = errorbound;
      *outsize = &data_out[fullcarry[chunk_id]] - output;
    }
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
  printf("PFPL GPU Single-Precision ABS Compressor\n");
  printf("Copyright 2025 Texas State University\n\n");

  // read input from file
  if (argc < 4 || argc > 5) {printf("USAGE: %s input_file_name compressed_file_name error_bound [threshold]\n\n", argv[0]);  throw std::runtime_error("LC error");}

  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const long long fsize = ftell(fin);
  if (fsize <= 0) {fprintf(stderr, "ERROR: input file too small\n\n"); throw std::runtime_error("LC error");}
  if (fsize >= 2147221529) {fprintf(stderr, "ERROR: input file too large\n\n"); throw std::runtime_error("LC error");}
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const int insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %d bytes\n", insize);

  // get GPU info
  hipSetDevice(0);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no HIP capable device detected\n\n"); throw std::runtime_error("LC error");}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  const int chunks = (insize + CS - 1) / CS;  // round up
  CheckHip(__LINE__);
  const int maxsize = 3 * sizeof(int) + chunks * sizeof(short) + chunks * CS;

  // allocate GPU memory
  byte* dencoded;
  hipHostMalloc((void **)&dencoded, maxsize);
  byte* d_input;
  hipMalloc((void **)&d_input, insize);
  hipMemcpy(d_input, input, insize, hipMemcpyHostToDevice);
  byte* d_encoded;
  hipMalloc((void **)&d_encoded, maxsize);
  int* d_encsize;
  hipMalloc((void **)&d_encsize, sizeof(int));
  int* d_fullcarry;
  hipMalloc((void**)&d_fullcarry, chunks * sizeof(int));
  CheckHip(__LINE__);

  const float errorbound = atof(argv[3]);
  const float threshold = (argc == 5) ? atof(argv[4]) : std::numeric_limits<float>::infinity();
  if (threshold < std::numeric_limits<float>::min()) {printf("ERROR: threshold must be a positive, normal, floating-point value\n");  throw std::runtime_error("LC error");}
  GPUTimer dtimer;
  for (int i = 0; i < NUM_RUNS; i++) {
    hipDeviceSynchronize();
    dtimer.start();

    d_reset<<<1, 1>>>();
    hipMemset(d_fullcarry, 0, chunks * sizeof(int));
    d_encode<<<blocks, TPB>>>(d_input, insize, d_encoded, d_encsize, d_fullcarry, errorbound, threshold);

    hipDeviceSynchronize();
    double runtime = dtimer.stop();

    if (i < NUM_RUNS - 1) {
      hipMemset(d_encsize, 0, sizeof(int));
      hipMemset(d_encoded, 0, maxsize);
    }
    CheckHip(__LINE__);
    printf("lc comp ecltime, %12.9f\n", runtime);
  }

  // get encoded GPU result
  int dencsize = 0;
  hipMemcpy(&dencsize, d_encsize, sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(dencoded, d_encoded, dencsize, hipMemcpyDeviceToHost);
  printf("encoded size: %d bytes\n", dencsize);
  printf("compression ratio: %.2f\n", 1.0 * insize / dencsize);
  CheckHip(__LINE__);

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(dencoded, 1, dencsize, fout);
  fclose(fout);

  // clean up GPU memory
  hipFree(d_input);
  hipFree(d_encoded);
  hipFree(d_encsize);
  CheckHip(__LINE__);

  // clean up
  delete [] input;
  hipHostFree(dencoded);
  return 0;
}
