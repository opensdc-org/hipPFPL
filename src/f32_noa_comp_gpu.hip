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


#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include "include/f32_staged_encode.h"
#include "include/f32_file_io.h"


const int NUM_RUNS = 9;


struct NoaQuantizer
{
  float errorbound;
  float threshold;
  const float* maximum;
  const float* minimum;

  __device__ float stored_error_bound() const
  {
    return (*maximum - *minimum) * errorbound;
  }

  __device__ float checked_error_bound() const
  {
    const float adj_eb = stored_error_bound();
    if (threshold <= adj_eb) {
      printf("NOA QUANT: ERROR: threshold must be larger than error_bound, NOA error bound was calculated to be %e\n", adj_eb);
      __trap();
    }
    return adj_eb;
  }

  __device__ void operator()(int csize, byte* slab) const
  {
    const float bound = checked_error_bound();
    if (bound >= std::numeric_limits<float>::min()) {
      quantize_chunk_in_place(csize, slab, AbsoluteWordQuantizer(bound, threshold));
    }
  }

  __device__ void load_full(const byte* source, byte* slab) const
  {
    const float bound = checked_error_bound();
    if (bound >= std::numeric_limits<float>::min()) {
      load_quantized_full_chunk(source, slab, AbsoluteWordQuantizer(bound, threshold));
    } else {
      load_raw_chunk(source, slab, CS);
    }
  }
};


static constexpr int NOA_RANGE_THREADS = 256;


static __device__ inline float2 reduce_noa_wave(float2 range, int distance)
{
  #pragma unroll
  for (; distance > 0; distance >>= 1) {
    range.x = fminf(range.x, pfpl_shfl_xor(range.x, distance));
    range.y = fmaxf(range.y, pfpl_shfl_xor(range.y, distance));
  }
  return range;
}


static __device__ inline float2 reduce_noa_block(float2 range, float2* wave_ranges)
{
  constexpr int waves = NOA_RANGE_THREADS / WS;
  const int lane = threadIdx.x % WS;
  const int wave = threadIdx.x / WS;
  range = reduce_noa_wave(range, WS / 2);
  if (lane == 0) wave_ranges[wave] = range;
  __syncthreads();
  if (wave == 0) range = reduce_noa_wave(wave_ranges[lane % waves], waves / 2);
  return range;
}


static __device__ inline float2 preserve_noa_zero_range(float2 range, const float* values)
{
  // Use the same zero for both extrema when every value compares equal to zero.
  if (range.x == 0.0f && range.y == 0.0f) range.x = range.y = values[0];
  return range;
}


static __global__ void d_noa_range_partials(const float* __restrict__ values, int count,
                                            float2* __restrict__ ranges)
{
  __shared__ float2 wave_ranges[NOA_RANGE_THREADS / WS];
  float2 range = {__builtin_inff(), -__builtin_inff()};
  const int tid = threadIdx.x;
  const int vectors = count / 4;
  const auto* input = reinterpret_cast<const float4*>(values);
  const int stride = gridDim.x * NOA_RANGE_THREADS;
  for (int index = blockIdx.x * NOA_RANGE_THREADS + tid; index < vectors; index += stride) {
    const float4 value = input[index];
    range.x = fminf(range.x, fminf(fminf(value.x, value.y), fminf(value.z, value.w)));
    range.y = fmaxf(range.y, fmaxf(fmaxf(value.x, value.y), fmaxf(value.z, value.w)));
  }
  if (blockIdx.x == 0 && tid < count % 4) {
    const float value = values[vectors * 4 + tid];
    range.x = fminf(range.x, value);
    range.y = fmaxf(range.y, value);
  }
  range = reduce_noa_block(range, wave_ranges);
  if (tid == 0) ranges[blockIdx.x] = gridDim.x == 1 ? preserve_noa_zero_range(range, values) : range;
}


static __global__ void d_noa_range_finish(const float* values, int count, float2* ranges)
{
  __shared__ float2 wave_ranges[NOA_RANGE_THREADS / WS];
  float2 range = {__builtin_inff(), -__builtin_inff()};
  for (int index = threadIdx.x; index < count; index += NOA_RANGE_THREADS) {
    const float2 partial = ranges[index];
    range.x = fminf(range.x, partial.x);
    range.y = fmaxf(range.y, partial.y);
  }
  range = reduce_noa_block(range, wave_ranges);
  if (threadIdx.x == 0) ranges[0] = preserve_noa_zero_range(range, values);
}


static int noa_range_blocks(int insize, int blocks)
{
  const int chunks = 1 + (insize - 1) / CS;
  return (blocks < chunks) ? blocks : chunks;
}


// Scratch holds one float2 per range block. All work uses the default stream.
static NoaQuantizer prepare_noa_quantizer(const byte* input, int insize, float errorbound,
                                          float threshold, float2* range_scratch, int blocks)
{
  const auto* values = reinterpret_cast<const float*>(input);
  const int count = insize / sizeof(float);
  if (!range_scratch) {
    const auto dev_ptr = thrust::device_pointer_cast(values);
    const auto min_max = thrust::minmax_element(thrust::device, dev_ptr, dev_ptr + count);
    return {errorbound, threshold, thrust::raw_pointer_cast(min_max.second),
            thrust::raw_pointer_cast(min_max.first)};
  }
  const int range_blocks = noa_range_blocks(insize, blocks);
  d_noa_range_partials<<<range_blocks, NOA_RANGE_THREADS>>>(values, count, range_scratch);
  check_compression_hip(hipGetLastError(), "launch NOA range partials");
  if (range_blocks > 1) {
    d_noa_range_finish<<<1, NOA_RANGE_THREADS>>>(values, range_blocks, range_scratch);
    check_compression_hip(hipGetLastError(), "launch NOA range finish");
  }
  return {errorbound, threshold, &range_scratch[0].y, &range_scratch[0].x};
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
  printf("PFPL GPU Single-Precision NOA Compressor\n");
  printf("Copyright 2025 Texas State University\n\n");

  // read input from file
  if (argc < 4 || argc > 5) {printf("USAGE: %s input_file_name compressed_file_name error_bound [threshold]\n\n", argv[0]);  throw std::runtime_error("LC error");}

  int insize;
  const auto input = read_f32_input(argv[1], insize);
  printf("original size: %d bytes\n", insize);

  // get GPU info
  hipSetDevice(0);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no HIP capable device detected\n\n"); throw std::runtime_error("LC error");}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  const int chunks = 1 + (insize - 1) / CS;
  CheckHip(__LINE__);
  const std::size_t maxsize = 16 + 2 * static_cast<std::size_t>(chunks) + insize;

  // allocate GPU memory
  byte* dencoded;
  check_compression_hip(hipHostMalloc((void**)&dencoded, maxsize), "allocate host packet buffer");
  byte* d_input;
  check_compression_hip(hipMalloc((void**)&d_input, insize), "allocate compression input");
  check_compression_hip(hipMemcpy(d_input, input.get(), insize, hipMemcpyHostToDevice),
                        "transfer compression input");
  byte* d_encoded;
  check_compression_hip(hipMalloc((void**)&d_encoded, maxsize), "allocate compression packet buffer");
  int* d_encsize;
  check_compression_hip(hipMalloc((void**)&d_encsize, sizeof(int)), "allocate packet size");
  const EncodeScratch scratch(insize);
  float2* range_scratch = nullptr;
  const auto range_status = hipMalloc((void**)&range_scratch,
                                     static_cast<std::size_t>(noa_range_blocks(insize, blocks)) * sizeof(float2));
  if (range_status == hipErrorOutOfMemory) {
    static_cast<void>(hipGetLastError());
    range_scratch = nullptr;
  } else {
    check_compression_hip(range_status, "allocate NOA range scratch");
  }
  CheckHip(__LINE__);

  const float errorbound = atof(argv[3]);
  const float threshold = (argc == 5) ? atof(argv[4]) : std::numeric_limits<float>::infinity();
  if (threshold < std::numeric_limits<float>::min()) {printf("ERROR: threshold must be a positive, normal, floating-point value\n");  throw std::runtime_error("LC error");}
  GPUTimer dtimer;
  for (int i = 0; i < NUM_RUNS; i++) {
    hipDeviceSynchronize();
    dtimer.start();

    const auto quantizer = prepare_noa_quantizer(d_input, insize, errorbound, threshold, range_scratch, blocks);
    submit_encode(d_input, insize, d_encoded, d_encsize, scratch, blocks, quantizer);

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
  check_compression_hip(hipMemcpy(&dencsize, d_encsize, sizeof(int), hipMemcpyDeviceToHost),
                        "read packet size");
  if (dencsize <= 0 || static_cast<std::size_t>(dencsize) > maxsize)
    throw std::runtime_error("Encoded packet size is outside the supported range");
  check_compression_hip(hipMemcpy(dencoded, d_encoded, dencsize, hipMemcpyDeviceToHost),
                        "read encoded packet");
  printf("encoded size: %d bytes\n", dencsize);
  printf("compression ratio: %.2f\n", 1.0 * insize / dencsize);
  CheckHip(__LINE__);

  // write to file
  write_f32_output(argv[2], dencoded, dencsize);

  // clean up GPU memory
  if (range_scratch) check_compression_hip(hipFree(range_scratch), "release NOA range scratch");
  hipFree(d_input);
  hipFree(d_encoded);
  hipFree(d_encsize);
  CheckHip(__LINE__);

  // clean up
  hipHostFree(dencoded);
  return 0;
}
catch (const std::exception& error)
{
  fprintf(stderr, "%s\n", error.what());
  return 1;
}
