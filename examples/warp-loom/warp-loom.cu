#include <cudalab.cuh>

__device__ float loom_hash(float2 p) {
  float value = sinf(p.x * 127.1f + p.y * 311.7f) * 43758.5453f;
  return value - floorf(value);
}

__device__ float loom_line(float x, float width) {
  return expf(-x * x / width);
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;

  unsigned mask = __activemask();
  int linear_lane = threadIdx.y * blockDim.x + threadIdx.x;
  int lane = linear_lane & 31;
  float2 p =
      make_float2((2.0f * x - params.width) / params.height, (2.0f * y - params.height) / params.height);

  float fold = .12f * sinf(p.x * 3.2f + params.time * .35f) + .055f * sinf(p.x * 8.7f - params.time * .21f);
  float cloth_y = p.y + fold;
  float vignette = cudalab_saturate(1.15f - .30f * p.x * p.x - .45f * cloth_y * cloth_y);
  float edge = cudalab_saturate((1.02f - fabsf(cloth_y)) * 22.0f);

  float voice = sinf(p.x * (25.0f + params.mouse_x * 30.0f) + lane * .41f + params.time) +
                cosf(cloth_y * (31.0f + params.mouse_y * 26.0f) - lane * .19f);
  float opposite = __shfl_xor_sync(mask, voice, 16);
  float quartet = __shfl_xor_sync(mask, voice, 8);
  float neighbor = __shfl_xor_sync(mask, voice, 1);
  unsigned chorus = __ballot_sync(mask, voice + opposite + .35f * quartet > 0.0f);
  float agreement = __popc(chorus) * (1.0f / 32.0f);

  float warp = .5f + .5f * sinf((p.x + .13f * opposite) * 165.0f);
  float weft = .5f + .5f * sinf((cloth_y + .11f * quartet) * 145.0f);
  float crossing = powf(warp * weft, 5.0f) + .32f * powf((1.0f - warp) * (1.0f - weft), 7.0f);
  float thread_glint = powf(cudalab_saturate(.5f + .5f * (neighbor - opposite)), 9.0f);

  float r = sqrtf(p.x * p.x + cloth_y * cloth_y);
  float angle = atan2f(cloth_y, p.x);
  float petals = .5f + .5f * cosf(angle * 12.0f + 4.0f * sinf(r * 11.0f) + params.time * .28f);
  float medallion = loom_line(sinf(r * 22.0f + 1.7f * petals), .055f) * cudalab_saturate((.84f - r) * 5.0f);
  float diamonds = powf(fabsf(sinf((p.x + cloth_y) * 11.0f) * sinf((p.x - cloth_y) * 11.0f)), 11.0f);
  float glyph = medallion + diamonds * (.25f + .75f * agreement);

  float hue = .5f + .5f * sinf(3.0f * p.x - 2.2f * cloth_y + voice * .22f + params.time * .16f);
  float3 silk = make_float3(.025f + .18f * hue, .018f + .07f * agreement, .07f + .34f * (1.0f - hue));
  float3 gold = make_float3(1.6f, .55f, .07f);
  float3 cyan = make_float3(.06f, .75f, .82f);
  float3 c = silk;
  c.x += gold.x * glyph + cyan.x * crossing;
  c.y += gold.y * glyph + cyan.y * crossing;
  c.z += gold.z * glyph + cyan.z * crossing;

  float normal_light = .55f + .45f * cosf(p.x * 3.2f + params.time * .35f);
  c.x *= (.45f + normal_light) * vignette * edge;
  c.y *= (.45f + normal_light) * vignette * edge;
  c.z *= (.45f + normal_light) * vignette * edge;
  c.x += thread_glint * .8f + crossing * .25f;
  c.y += thread_glint * .5f + crossing * .12f;
  c.z += thread_glint * .9f + crossing * .30f;

  float fringe = fabsf(cloth_y) > .98f
                     ? loom_line(fabsf(((p.x + 2.0f) * 48.0f - floorf((p.x + 2.0f) * 48.0f)) - .5f), .022f) *
                           cudalab_saturate((1.08f - fabsf(cloth_y)) * 18.0f)
                     : 0.0f;
  c.x += fringe * 1.2f;
  c.y += fringe * .62f;
  c.z += fringe * .18f;

  c = cudalab_tonemap(c);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(c.x), .4545f),
                                             255 * powf(cudalab_saturate(c.y), .4545f),
                                             255 * powf(cudalab_saturate(c.z), .4545f),
                                             255);
}
