#include <cub/block/block_radix_sort.cuh>
#include <cudalab.cuh>

constexpr int N = 1024;

__device__ float memory_hash(float2 p) {
  float h = sinf(p.x * 127.1f + p.y * 311.7f) * 43758.5453f;
  return h - floorf(h);
}

CUDALAB_SIMULATE {
  using Sort = cub::BlockRadixSort<float, 256, 1>;
  __shared__ typename Sort::TempStorage storage;
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= N * N || state_bytes < N * N * sizeof(float))
    return;

  int x = i % N;
  int y = i / N;
  float2 uv = make_float2(x / (float)N, y / (float)N);
  float2 center = make_float2(.5f + .18f * sinf(params.time * .17f), .48f + .13f * cosf(params.time * .21f));
  float dx = uv.x - center.x;
  float dy = uv.y - center.y;
  float radius = sqrtf(dx * dx + dy * dy);
  float portrait = .52f + .23f * sinf(uv.x * 19.0f + sinf(uv.y * 9.0f) * 3.0f) +
                   .17f * cosf(uv.y * 31.0f - params.time * .7f) + .30f * expf(-radius * radius * 25.0f) +
                   .12f * sinf(atan2f(dy, dx) * 9.0f + radius * 55.0f - params.time);
  portrait += (memory_hash(make_float2(x, y)) - .5f) * .08f;

  float keys[1] = {portrait};
  Sort(storage).SortDescending(keys);
  float block_phase = blockIdx.x * .6180339887f + params.time * (.05f + params.mouse_y * .15f);
  block_phase -= floorf(block_phase);
  float corruption = .08f + .88f * params.mouse_x;
  float tearing = .5f + .5f * sinf(y * .053f + params.time * 2.1f);
  static_cast<float*>(state)[i] = block_phase < corruption * (.35f + .65f * tearing) ? keys[0] : portrait;
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height || !state)
    return;

  int sx = x * N / params.width;
  int sy = y * N / params.height;
  float strip = sinf(sy * .071f + params.time * 3.0f) + sinf(sy * .017f - params.time);
  int displacement = (int)(strip * (6.0f + 85.0f * params.mouse_x));
  int rx = (sx + displacement + N) % N;
  int bx = (sx - displacement / 2 + N) % N;
  const float* image = static_cast<const float*>(state);
  float a = image[sy * N + rx];
  float b = image[sy * N + sx];
  float d = image[sy * N + bx];

  float prism = .5f + .5f * sinf(b * 14.0f + sx * .011f - sy * .008f + params.time * .6f);
  float contour = powf(.5f + .5f * cosf(b * 38.0f), 9.0f);
  float block_edge = ((sx & 255) < 3 || (sy & 31) < 2) ? .65f : 0.0f;
  float3 c = make_float3(.04f + 1.45f * a * a + .45f * prism,
                         .015f + .38f * b + .85f * contour,
                         .08f + 1.15f * d * (1.0f - prism) + .35f * b);
  c.x += block_edge * (1.0f - prism);
  c.y += block_edge * .15f;
  c.z += block_edge * prism;

  float aperture = cudalab_saturate(1.25f - 1.15f *
                                                ((x - params.width * .5f) * (x - params.width * .5f) +
                                                 (y - params.height * .5f) * (y - params.height * .5f)) /
                                                (params.height * params.height));
  c.x *= aperture;
  c.y *= aperture;
  c.z *= aperture;
  c = cudalab_tonemap(c);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(c.x), .4545f),
                                             255 * powf(cudalab_saturate(c.y), .4545f),
                                             255 * powf(cudalab_saturate(c.z), .4545f),
                                             255);
}
