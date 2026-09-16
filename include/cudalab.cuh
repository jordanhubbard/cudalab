#pragma once

#include <cuda_runtime.h>

// Stable live-kernel ABI. Every visual demo exports this exact entry point.
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

#define CUDALAB_KERNEL extern "C" __global__ void render(uchar4* pixels, CudalabParams params)

__device__ inline float cudalab_saturate(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ inline float3 cudalab_tonemap(float3 x) {
  return make_float3(
    (x.x * (2.51f * x.x + .03f)) / (x.x * (2.43f * x.x + .59f) + .14f),
    (x.y * (2.51f * x.y + .03f)) / (x.y * (2.43f * x.y + .59f) + .14f),
    (x.z * (2.51f * x.z + .03f)) / (x.z * (2.43f * x.z + .59f) + .14f));
}
