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
#include "include/f32_staged_encode.h"
#include "include/f32_file_io.h"


const int NUM_RUNS = 9;


static __device__ inline float d_REL_log2approxf(const float orig_f)
{
  //assert(orig_f > 0);
  const int mantissabits = 23;
  const int orig_i = *((int*)&orig_f);
  const int expo = (orig_i >> mantissabits) & 0xff;
  //if ((expo == 0) || (expo == 0xff)) return orig_f;
  const int frac_i = (127 << mantissabits) | (orig_i & ~(~0 << mantissabits));
  const float frac_f = *((float*)&frac_i);
  const float log_f = frac_f + (expo - 128);  // - bias - 1
  return log_f;
}


static __device__ inline float d_REL_pow2approxf(const float log_f)
{
  const int mantissabits = 23;
  const float biased = log_f + 127;
  const int expo = biased;
  const float frac_f = biased - (expo - 1);
  const int frac_i = *((int*)&frac_f);
  const int exp_i = (expo << mantissabits) | (frac_i & ~(~0 << mantissabits));
  const float recon_f = *((float*)&exp_i);
  return recon_f;
}


struct RelWordQuantizer
{
  float errorbound;
  float threshold;
  float log2eb;
  float inv_log2eb;

  __device__ __forceinline__ unsigned int operator()(unsigned int orig_i) const
  {
    constexpr int mantissabits = 23;
    constexpr unsigned int signexpomask = 0xff800000u;
    constexpr int maxbin = (1 << (mantissabits - 2)) - 1;
    const unsigned int abs_orig_i = orig_i & 0x7fffffffu;
    const float abs_orig_f = __uint_as_float(abs_orig_i);
    unsigned int output = orig_i;
    const int expo = (orig_i >> mantissabits) & 0xff;
    if (expo == 0) {
      if (abs_orig_i == 0) {
        output = signexpomask | 1;
      }
    } else if (expo == 0xff) {
      if (((orig_i & signexpomask) == signexpomask) && ((orig_i & ~signexpomask) != 0)) {
        output = abs_orig_i;  // Preserve the established positive-NaN encoding.
      }
    } else {
      const float log_f = d_REL_log2approxf(abs_orig_f);
      const float scaled = log_f * inv_log2eb;
      const int bin = static_cast<int>(roundf(scaled));
      const float abs_recon_f = d_REL_pow2approxf(bin * log2eb);
      const float lower = abs_orig_f / (1 + errorbound);
      const float upper = abs_orig_f * (1 + errorbound);
      if (!((bin >= maxbin) || (bin <= -maxbin) || (abs_orig_f >= threshold) ||
            (abs_recon_f < lower) || (abs_recon_f > upper) || (abs_recon_f == 0) || !isfinite(abs_recon_f))) {
        const unsigned int sign = 0u - static_cast<unsigned int>(bin < 0);
        const unsigned int tcms = (static_cast<unsigned int>(bin) << 1) ^ sign;
        const unsigned int code = ((tcms + 1) << 1) | (orig_i >> 31);
        output = signexpomask | code;
      }
    }
    return (output ^ signexpomask) - 1;
  }
};


struct RelQuantizer
{
  float errorbound;
  float threshold;

  __device__ float stored_error_bound() const { return errorbound; }

  __device__ RelWordQuantizer prepare() const
  {
    const float log2eb = 2 * d_REL_log2approxf(1 + errorbound);
    return {errorbound, threshold, log2eb, 1 / log2eb};
  }

  __device__ void operator()(int csize, byte* slab) const
  {
    quantize_chunk_in_place(csize, slab, prepare());
  }

  __device__ void load_full(const byte* source, byte* slab) const
  {
    load_quantized_full_chunk(source, slab, prepare());
  }
};


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
  printf("PFPL GPU Single-Precision REL Compressor\n");
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
  CheckHip(__LINE__);

  const float errorbound = atof(argv[3]);
  const float threshold = (argc == 5) ? atof(argv[4]) : std::numeric_limits<float>::infinity();
  if (threshold < std::numeric_limits<float>::min()) {printf("ERROR: threshold must be a positive, normal, floating-point value\n");  throw std::runtime_error("LC error");}
  GPUTimer dtimer;
  for (int i = 0; i < NUM_RUNS; i++) {
    hipDeviceSynchronize();
    dtimer.start();

    submit_encode(d_input, insize, d_encoded, d_encsize, scratch, blocks,
                  RelQuantizer{errorbound, threshold});

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
