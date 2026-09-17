#pragma once

#include <cuda_runtime.h>
#if !defined(__CUDACC_RTC__)
#include <cmath>
#endif

// Stable live-kernel ABI. A piece must export render and may export reset,
// simulate, composite, and audio. Host services pass manifest-declared resources
// through a compact device table; source using these macros remains compatible.
struct CudalabParams {
  int width;
  int height;
  float time;
  float delta;
  float mouse_x;
  float mouse_y;
  int frame;
  int quality;
  float mouse_dx = 0.0f;
  float mouse_dy = 0.0f;
  int mouse_down = 0;
};

struct CudalabResource {
  void* data;
  unsigned long long bytes;
  cudaTextureObject_t texture;
  cudaSurfaceObject_t surface;
  int width;
  int height;
};

template <typename T>
__device__ inline T* cudalab_buffer(const CudalabResource* resources, int resource_count, int slot) {
  return resources && slot >= 0 && slot < resource_count ? static_cast<T*>(resources[slot].data) : nullptr;
}

#define CUDALAB_RENDER                                                                                       \
  extern "C" __global__ void render(uchar4* pixels,                                                          \
                                    void* state,                                                             \
                                    unsigned long long state_bytes,                                          \
                                    CudalabParams params,                                                    \
                                    const CudalabResource* resources,                                        \
                                    int resource_count)
#define CUDALAB_RESET                                                                                        \
  extern "C" __global__ void reset(void* state,                                                              \
                                   unsigned long long state_bytes,                                           \
                                   CudalabParams params,                                                     \
                                   const CudalabResource* resources,                                         \
                                   int resource_count)
#define CUDALAB_SIMULATE                                                                                     \
  extern "C" __global__ void simulate(void* state,                                                           \
                                      unsigned long long state_bytes,                                        \
                                      CudalabParams params,                                                  \
                                      const CudalabResource* resources,                                      \
                                      int resource_count)
#define CUDALAB_COMPOSITE                                                                                    \
  extern "C" __global__ void composite(uchar4* pixels,                                                       \
                                       void* state,                                                          \
                                       unsigned long long state_bytes,                                       \
                                       CudalabParams params,                                                 \
                                       const CudalabResource* resources,                                     \
                                       int resource_count)
#define CUDALAB_AUDIO                                                                                        \
  extern "C" __global__ void audio(float2* samples,                                                          \
                                   void* state,                                                              \
                                   unsigned long long state_bytes,                                           \
                                   CudalabParams params,                                                     \
                                   unsigned long long sample_offset,                                         \
                                   int sample_count,                                                         \
                                   int sample_rate,                                                          \
                                   const CudalabResource* resources,                                         \
                                   int resource_count)
#define CUDALAB_KERNEL CUDALAB_RENDER

__device__ inline float cudalab_saturate(float x) {
  return fminf(1.0f, fmaxf(0.0f, x));
}
__device__ inline float3 cudalab_tonemap(float3 x) {
  return make_float3((x.x * (2.51f * x.x + .03f)) / (x.x * (2.43f * x.x + .59f) + .14f),
                     (x.y * (2.51f * x.y + .03f)) / (x.y * (2.43f * x.y + .59f) + .14f),
                     (x.z * (2.51f * x.z + .03f)) / (x.z * (2.43f * x.z + .59f) + .14f));
}
