#include <cudalab.cuh>

constexpr int N = 768;
constexpr int CELLS = N * N;

__device__ float hash21(int x, int y) {
  unsigned n = (unsigned)x * 1973u + (unsigned)y * 9277u + 0x68bc21ebu;
  n = (n << 13u) ^ n;
  return (n * (n * n * 15731u + 789221u) + 1376312589u) * 2.3283064e-10f;
}
__device__ float2 sample(const float2* f, int x, int y) {
  x = (x + N) % N;
  y = (y + N) % N;
  return f[y * N + x];
}

CUDALAB_RESET {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= CELLS || state_bytes < sizeof(float2) * CELLS * 2ull)
    return;
  int x = i % N, y = i / N;
  float nx = (x - N * .5f) / N, ny = (y - N * .5f) / N;
  float petals = cosf(atan2f(ny, nx) * 7.0f + hypotf(nx, ny) * 36.0f);
  float seed = (hypotf(nx, ny) < .075f || (petals > .92f && hypotf(nx, ny) < .30f)) ? .92f : 0.0f;
  seed += hash21(x, y) > .9985f ? .7f : 0.0f;
  float2 value = make_float2(1.0f - seed * .48f, fminf(1.0f, seed));
  auto* fields = static_cast<float2*>(state);
  fields[i] = value;
  fields[CELLS + i] = value;
}

CUDALAB_SIMULATE {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= CELLS || state_bytes < sizeof(float2) * CELLS * 2ull)
    return;
  auto* fields = static_cast<float2*>(state);
  const float2* src = fields + ((params.frame & 1) ? CELLS : 0);
  float2* dst = fields + ((params.frame & 1) ? 0 : CELLS);
  int x = i % N, y = i / N;
  float2 c = sample(src, x, y);
  float2 lap = make_float2(-c.x, -c.y);
  float2 n = sample(src, x, y - 1), s = sample(src, x, y + 1), e = sample(src, x + 1, y),
         w = sample(src, x - 1, y);
  lap.x += .2f * (n.x + s.x + e.x + w.x);
  lap.y += .2f * (n.y + s.y + e.y + w.y);
  float2 a = sample(src, x - 1, y - 1), b = sample(src, x + 1, y - 1), d = sample(src, x - 1, y + 1),
         q = sample(src, x + 1, y + 1);
  lap.x += .05f * (a.x + b.x + d.x + q.x);
  lap.y += .05f * (a.y + b.y + d.y + q.y);
  float feed = .034f + .004f * sinf(params.time * .07f), kill = .0615f;
  float reaction = c.x * c.y * c.y;
  float u = c.x + (.95f * lap.x - reaction + feed * (1 - c.x));
  float v = c.y + (.48f * lap.y + reaction - (kill + feed) * c.y);
  if (params.mouse_x > 0 && params.mouse_y > 0) {
    float dx = x / N - params.mouse_x, dy = y / N - params.mouse_y;
    if (dx * dx + dy * dy < .0012f)
      v = fminf(1.0f, v + .13f);
  }
  dst[i] = make_float2(cudalab_saturate(u), cudalab_saturate(v));
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height || !state)
    return;
  auto* fields = static_cast<float2*>(state);
  const float2* current = fields + ((params.frame & 1) ? 0 : CELLS);
  int gx = x * N / params.width, gy = y * N / params.height;
  float2 c = current[gy * N + gx];
  float edge = fabsf(c.y - sample(current, gx + 1, gy).y) + fabsf(c.y - sample(current, gx, gy + 1).y);
  float bloom = cudalab_saturate(c.y * 1.8f - c.x * .45f);
  float3 ink = make_float3(
      .015f + .08f * bloom, .025f + .36f * bloom + .6f * edge, .055f + .72f * bloom + .2f * sinf(c.y * 18));
  float3 gold = make_float3(1.4f, .36f, .035f);
  ink.x += gold.x * edge;
  ink.y += gold.y * edge;
  ink.z += gold.z * edge;
  ink = cudalab_tonemap(ink);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(ink.x), .4545f),
                                             255 * powf(cudalab_saturate(ink.y), .4545f),
                                             255 * powf(cudalab_saturate(ink.z), .4545f),
                                             255);
}
