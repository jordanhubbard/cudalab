#include <cudalab.cuh>

__device__ float3 add(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 sub(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 mul(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float dot3(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 norm(float3 a) {
  return mul(a, rsqrtf(dot3(a, a) + 1e-9f));
}
__device__ float3 cross3(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ float scene(float3 p, float& trap) {
  float3 z = p;
  float dr = 1, r = 0;
  trap = 10;
#pragma unroll
  for (int i = 0; i < 12; i++) {
    r = sqrtf(dot3(z, z));
    trap = fminf(trap, fabsf(z.x * z.y * z.z));
    if (r > 2.3f)
      break;
    float theta = acosf(z.z / (r + 1e-8f)), phi = atan2f(z.y, z.x), zr = powf(r, 8.0f);
    dr = powf(r, 7.0f) * 8.0f * dr + 1.0f;
    theta *= 8;
    phi *= 8;
    z = add(mul(make_float3(sinf(theta) * cosf(phi), sinf(phi) * sinf(theta), cosf(theta)), zr), p);
  }
  return .5f * logf(r) * r / dr;
}
__device__ float3 normal(float3 p) {
  const float e = .0015f;
  float t;
  const float d = scene(p, t);
  return norm(make_float3(scene(add(p, make_float3(e, 0, 0)), t) - d,
                          scene(add(p, make_float3(0, e, 0)), t) - d,
                          scene(add(p, make_float3(0, 0, e)), t) - d));
}
__device__ float softshadow(float3 ro, float3 rd) {
  float shade = 1, t = .03f;
  for (int i = 0; i < 40; i++) {
    float trap;
    float h = scene(add(ro, mul(rd, t)), trap);
    shade = fminf(shade, 18 * h / t);
    t += fminf(.12f, fmaxf(.01f, h));
    if (h < .0005f || t > 6)
      break;
  }
  return cudalab_saturate(shade);
}

CUDALAB_KERNEL {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;
  float2 q =
      make_float2((2.0f * x - params.width) / params.height, (params.height - 2.0f * y) / params.height);
  float angle = params.time * .13f + (params.mouse_x - .5f) * 2.4f;
  float elevation = .18f + (params.mouse_y - .5f) * .9f;
  float3 ro = make_float3(3.2f * cosf(angle), 1.0f + elevation, 3.2f * sinf(angle));
  float3 target = make_float3(0, 0, 0), fw = norm(sub(target, ro));
  float3 rt = norm(cross3(fw, make_float3(0, 1, 0))), up = cross3(rt, fw);
  float3 rd = norm(add(fw, add(mul(rt, q.x * .72f), mul(up, q.y * .72f))));
  float travel = 0, trap = 0, lastTrap = 0;
  bool hit = false;
  int iterations = 72 + params.quality * 24;
  for (int i = 0; i < iterations; i++) {
    float distance = scene(add(ro, mul(rd, travel)), trap);
    lastTrap = trap;
    if (distance < .0008f) {
      hit = true;
      break;
    }
    travel += distance * .72f;
    if (travel > 8)
      break;
  }
  float3 color = make_float3(.008f, .012f, .035f);
  if (hit) {
    float3 p = add(ro, mul(rd, travel)), n = normal(p), light = norm(make_float3(-.5f, .8f, -.35f));
    float diffuse = fmaxf(0, dot3(n, light)) * softshadow(add(p, mul(n, .005f)), light);
    float rim = powf(1 - fmaxf(0, dot3(n, mul(rd, -1))), 3);
    float bands = .5f + .5f * cosf(18 * logf(lastTrap + .008f) + params.time * .3f);
    float3 base = make_float3(.10f + .55f * bands, .18f + .12f * bands, .35f + .55f * (1 - bands));
    color = add(mul(base, .13f + .95f * diffuse), mul(make_float3(.2f, .65f, 1.4f), rim * .8f));
  } else {
    float stars =
        powf(cudalab_saturate((sinf(rd.x * 831 + rd.y * 421 + rd.z * 313) * 43758.5f -
                               floorf(sinf(rd.x * 831 + rd.y * 421 + rd.z * 313) * 43758.5f) - .994f) /
                              .006f),
             9);
    color = add(color, mul(make_float3(.7f, .82f, 1), stars));
  }
  float fog = 1 - expf(-.018f * travel * travel);
  color = add(mul(color, 1 - fog), mul(make_float3(.015f, .025f, .07f), fog));
  color = cudalab_tonemap(color);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(color.x), .4545f),
                                             255 * powf(cudalab_saturate(color.y), .4545f),
                                             255 * powf(cudalab_saturate(color.z), .4545f),
                                             255);
}
