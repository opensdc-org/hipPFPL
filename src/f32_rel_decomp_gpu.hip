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


#include "include/f32_decode_gpu.h"
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


struct RelRestoreWord
{
  __device__ explicit RelRestoreWord(float errorbound)
      : log2eb(2 * d_REL_log2approxf(1 + errorbound)) {}

  __device__ unsigned int operator()(unsigned int bits) const
  {
    constexpr unsigned int signexpomask = 0xff800000u;
    const unsigned int val = (bits + 1u) ^ signexpomask;
    const unsigned int code = val & ~signexpomask;
    if ((val & signexpomask) != signexpomask || code == 0) return val;
    if (code == 1) return 0;

    const unsigned int dec = (code >> 1) - 1;
    const int bin = static_cast<int>(dec >> 1) ^ -static_cast<int>(dec & 1u);
    const float magnitude = d_REL_pow2approxf(bin * log2eb);
    return __float_as_uint((val & 1u) ? -magnitude : magnitude);
  }

  float log2eb;
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
  printf("PFPL GPU Single-Precision REL Decompressor\n");
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

    d_chunk_offsets<<<1, TPB>>>(d_encoded, chunks, d_offsets);
    d_decode<RelRestoreWord><<<blocks, TPB>>>(d_encoded, d_decoded, d_decsize, d_offsets, chunks);

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
