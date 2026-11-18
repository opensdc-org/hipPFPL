#ifndef PFPL_HIP_WARP32_H
#define PFPL_HIP_WARP32_H

#include <hip/hip_runtime.h>

#ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#error "PFPL requires HIP_ENABLE_WARP_SYNC_BUILTINS for logical warp32 operations"
#endif

// PFPL's format and work distribution retain CUDA's 32-lane logical warps.
// A physical wave64 contains two independent logical warps. HIP masks use
// physical lane positions, whereas CUDA ballot results use logical positions.
static __device__ inline unsigned int pfpl_warp32_offset()
{
  return (threadIdx.x % warpSize) & ~31u;
}

static __device__ inline unsigned long long pfpl_warp32_mask(unsigned int mask)
{
  return static_cast<unsigned long long>(mask) << pfpl_warp32_offset();
}

static __device__ inline unsigned int pfpl_ballot_sync(unsigned int mask, int predicate)
{
  return static_cast<unsigned int>(
      __ballot_sync(pfpl_warp32_mask(mask), predicate) >> pfpl_warp32_offset());
}

static __device__ inline int pfpl_all_sync(unsigned int mask, int predicate)
{
  return __all_sync(pfpl_warp32_mask(mask), predicate);
}

static __device__ inline int pfpl_any_sync(unsigned int mask, int predicate)
{
  return __any_sync(pfpl_warp32_mask(mask), predicate);
}

static __device__ inline void pfpl_syncwarp(unsigned int mask = 0xffffffffu)
{
  __syncwarp(pfpl_warp32_mask(mask));
}

template <typename T>
static __device__ inline T pfpl_shfl_sync(unsigned int mask, T value, int lane, int width = 32)
{
  return __shfl_sync(pfpl_warp32_mask(mask), value, lane, width);
}

template <typename T>
static __device__ inline T pfpl_shfl_up_sync(unsigned int mask, T value, unsigned int delta, int width = 32)
{
  return __shfl_up_sync(pfpl_warp32_mask(mask), value, delta, width);
}

template <typename T>
static __device__ inline T pfpl_shfl_xor_sync(unsigned int mask, T value, int lane_mask, int width = 32)
{
  return __shfl_xor_sync(pfpl_warp32_mask(mask), value, lane_mask, width);
}

template <typename T>
static __device__ inline T pfpl_shfl_up(T value, unsigned int delta, int width = 32)
{
  return pfpl_shfl_up_sync(0xffffffffu, value, delta, width);
}

template <typename T>
static __device__ inline T pfpl_shfl_xor(T value, int lane_mask, int width = 32)
{
  return pfpl_shfl_xor_sync(0xffffffffu, value, lane_mask, width);
}

#endif  // PFPL_HIP_WARP32_H
