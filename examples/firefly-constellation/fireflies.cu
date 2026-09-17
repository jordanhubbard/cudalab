#include <cudalab.cuh>

constexpr int COUNT = 262144;
struct Particle {
  float4 p;
  float4 v;
};
__device__ unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}
__device__ float rnd(unsigned x) {
  return (mix(x) & 0x00ffffffu) / 16777216.0f;
}

CUDALAB_RESET {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= COUNT || state_bytes < sizeof(Particle) * COUNT)
    return;
  auto* ps = static_cast<Particle*>(state);
  float a = rnd(i * 3u) * 6.28318f, r = .08f + .82f * sqrtf(rnd(i * 3u + 1));
  ps[i].p = make_float4(r * cosf(a), r * sinf(a), rnd(i * 3u + 2) * 2 - 1, 1);
  ps[i].v = make_float4(-sinf(a) * .08f, cosf(a) * .08f, 0, 0);
}

CUDALAB_SIMULATE {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= COUNT || !state)
    return;
  auto* ps = static_cast<Particle*>(state);
  Particle p = ps[i];
  float3 pos = make_float3(p.p.x, p.p.y, p.p.z), vel = make_float3(p.v.x, p.v.y, p.v.z);
  unsigned mask = __activemask();
  float cx = __shfl_xor_sync(mask, pos.x, 16), cy = __shfl_xor_sync(mask, pos.y, 8),
        cz = __shfl_xor_sync(mask, pos.z, 4);
  float t = params.time * .23f;
  float3 curl = make_float3(sinf(pos.y * 4 + t) - cosf(pos.z * 3),
                            sinf(pos.z * 3 - t) - cosf(pos.x * 4),
                            sinf(pos.x * 3 + t) - cosf(pos.y * 3));
  float3 kin = make_float3(cx - pos.x, cy - pos.y, cz - pos.z);
  float inv = rsqrtf(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z + .05f);
  float px = (params.mouse_x - .5f) * 2, py = (params.mouse_y - .5f) * 2, dx = px - pos.x, dy = py - pos.y,
        att = .0015f / (dx * dx + dy * dy + .04f);
  vel.x = vel.x * .982f + curl.x * .0018f - pos.x * inv * .0012f + kin.x * .0008f + dx * att;
  vel.y = vel.y * .982f + curl.y * .0018f - pos.y * inv * .0012f + kin.y * .0008f + dy * att;
  vel.z = vel.z * .982f + curl.z * .0018f - pos.z * inv * .0012f;
  pos.x += vel.x;
  pos.y += vel.y;
  pos.z += vel.z;
  if (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z > 3.2f) {
    pos.x *= .55f;
    pos.y *= .55f;
    pos.z *= .55f;
  }
  ps[i].p = make_float4(pos.x, pos.y, pos.z, 1);
  ps[i].v = make_float4(vel.x, vel.y, vel.z, 0);
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;
  auto* density = reinterpret_cast<unsigned*>(static_cast<Particle*>(state) + COUNT);
  int i = y * params.width + x;
  unsigned d = density[i];
  density[i] = (unsigned)(d * .936f);
  float e = log2f(1 + d * .018f);
  float vign = 1 - .28f * (powf((x / (float)params.width - .5f) * 2, 2) +
                           powf((y / (float)params.height - .5f) * 2, 2));
  float3 c = make_float3(.006f + .035f * e, .009f + .24f * e, .025f + .58f * e + .12f * e * e);
  c.x += .48f * fmaxf(0, e - 1.1f);
  c.y += .18f * fmaxf(0, e - 1.1f);
  c.x *= vign;
  c.y *= vign;
  c.z *= vign;
  c = cudalab_tonemap(c);
  pixels[i] = make_uchar4(255 * powf(cudalab_saturate(c.x), .4545f),
                          255 * powf(cudalab_saturate(c.y), .4545f),
                          255 * powf(cudalab_saturate(c.z), .4545f),
                          255);
}

CUDALAB_COMPOSITE {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= COUNT || !state)
    return;
  auto* ps = static_cast<Particle*>(state);
  auto* density = reinterpret_cast<unsigned*>(ps + COUNT);
  float3 p = make_float3(ps[i].p.x, ps[i].p.y, ps[i].p.z);
  float depth = 2.35f + p.z;
  int x = (int)((p.x / depth * .72f + .5f) * params.width),
      y = (int)((p.y / depth * .72f + .5f) * params.height);
  if (x > 1 && x < params.width - 2 && y > 1 && y < params.height - 2) {
    unsigned power = 5 + (mix(i + params.frame) & 15);
    atomicAdd(&density[y * params.width + x], power);
    if ((i & 7) == 0) {
      atomicAdd(&density[y * params.width + x + 1], power >> 2);
      atomicAdd(&density[(y + 1) * params.width + x], power >> 2);
    }
  }
}
