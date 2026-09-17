#pragma once

#include <cuda_runtime.h>

// Stable live-kernel ABI. A piece must export render and may export reset,
// simulate, and composite. Stages execute in that order on one CUDA stream.
struct CudalabParams {
  int width;
  int height;
  float time;
  float delta;
  float mouse_x;
  float mouse_y;
  int frame;
  int quality;
};

#define CUDALAB_RENDER                                                                                       \
  extern "C" __global__ void render(                                                                         \
      uchar4* pixels, void* state, unsigned long long state_bytes, CudalabParams params)
#define CUDALAB_RESET                                                                                        \
  extern "C" __global__ void reset(void* state, unsigned long long state_bytes, CudalabParams params)
#define CUDALAB_SIMULATE                                                                                     \
  extern "C" __global__ void simulate(void* state, unsigned long long state_bytes, CudalabParams params)
#define CUDALAB_COMPOSITE                                                                                    \
  extern "C" __global__ void composite(                                                                      \
      uchar4* pixels, void* state, unsigned long long state_bytes, CudalabParams params)
#define CUDALAB_AUDIO                                                                                        \
  extern "C" __global__ void audio(float2* samples,                                                          \
                                   void* state,                                                              \
                                   unsigned long long state_bytes,                                           \
                                   CudalabParams params,                                                     \
                                   unsigned long long sample_offset,                                         \
                                   int sample_count,                                                         \
                                   int sample_rate)
#define CUDALAB_KERNEL CUDALAB_RENDER

__device__ inline float cudalab_saturate(float x) {
  return fminf(1.0f, fmaxf(0.0f, x));
}
__device__ inline float3 cudalab_tonemap(float3 x) {
  return make_float3((x.x * (2.51f * x.x + .03f)) / (x.x * (2.43f * x.x + .59f) + .14f),
                     (x.y * (2.51f * x.y + .03f)) / (x.y * (2.43f * x.y + .59f) + .14f),
                     (x.z * (2.51f * x.z + .03f)) / (x.z * (2.43f * x.z + .59f) + .14f));
}
