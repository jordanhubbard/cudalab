#include <cudalab.cuh>

CUDALAB_RENDER {
  __shared__ float tile[18][18];
  int lx = threadIdx.x, ly = threadIdx.y, x = blockIdx.x * 16 + lx, y = blockIdx.y * 16 + ly;
  float2 uv =
      make_float2((2.0f * x - params.width) / params.height, (2.0f * y - params.height) / params.height);
  float r = hypotf(uv.x, uv.y), a = atan2f(uv.y, uv.x);
  float value = sinf(r * 34 - params.time * 1.2f) + cosf(a * (6 + int(params.mouse_x * 10)) + r * 8);
  tile[ly + 1][lx + 1] = value;
  if (lx == 0)
    tile[ly + 1][0] = sinf((r + .02f) * 34 - params.time * 1.2f) + cosf(a * 8);
  if (lx == 15)
    tile[ly + 1][17] = sinf((r - .02f) * 34 - params.time * 1.2f) + cosf(a * 8);
  if (ly == 0)
    tile[0][lx + 1] = value;
  if (ly == 15)
    tile[17][lx + 1] = value;
  __syncthreads();
  if (x >= params.width || y >= params.height)
    return;
  float gx = tile[ly + 1][lx + 2] - tile[ly + 1][lx], gy = tile[ly + 2][lx + 1] - tile[ly][lx + 1];
  float edge = cudalab_saturate(hypotf(gx, gy) * 1.4f);
  float petals = .5f + .5f * cosf(a * 12 + r * 9 + value * .4f);
  float lead = powf(1 - edge, 12);
  float3 glass = make_float3(.08f + .85f * petals, .035f + .18f * edge, .15f + .8f * (1 - petals));
  float glint = powf(edge, 8) * 2;
  glass.x = glass.x * lead + glint;
  glass.y = glass.y * lead + glint * .75f;
  glass.z = glass.z * lead + glint * .45f;
  glass = cudalab_tonemap(glass);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(glass.x), .4545f),
                                             255 * powf(cudalab_saturate(glass.y), .4545f),
                                             255 * powf(cudalab_saturate(glass.z), .4545f),
                                             255);
}
