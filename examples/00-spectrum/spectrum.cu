#include <cudalab.cuh>

CUDALAB_KERNEL {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;

  const float2 uv =
      make_float2((2.0f * x - params.width) / params.height, (2.0f * y - params.height) / params.height);
  const float2 pointer = make_float2((2.0f * params.mouse_x - 1.0f) * params.width / params.height,
                                     2.0f * params.mouse_y - 1.0f);
  const float distance = hypotf(uv.x - pointer.x, uv.y - pointer.y);
  const float wave = .5f + .5f * cosf(12.0f * distance - params.time * 4.0f);
  const float3 color = make_float3(.5f + .5f * cosf(6.28318f * (wave + uv.x + 0.00f)),
                                   .5f + .5f * cosf(6.28318f * (wave + uv.x + 0.33f)),
                                   .5f + .5f * cosf(6.28318f * (wave + uv.x + 0.67f)));
  pixels[y * params.width + x] = make_uchar4(static_cast<unsigned char>(255 * cudalab_saturate(color.x)),
                                             static_cast<unsigned char>(255 * cudalab_saturate(color.y)),
                                             static_cast<unsigned char>(255 * cudalab_saturate(color.z)),
                                             255);
}
