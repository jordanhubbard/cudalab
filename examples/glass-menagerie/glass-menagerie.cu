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
__device__ float3 had(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
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
__device__ float3 reflect3(float3 d, float3 n) {
  return sub(d, mul(n, 2 * dot3(d, n)));
}
__device__ unsigned rng(unsigned& s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}
__device__ float random(unsigned& s) {
  return (rng(s) & 0xffffff) * 5.9604645e-8f;
}
__device__ float3 random_sphere(unsigned& s) {
  float z = random(s) * 2 - 1, a = random(s) * 6.283185f, r = sqrtf(fmaxf(0.0f, 1 - z * z));
  return make_float3(r * cosf(a), r * sinf(a), z);
}

struct Hit {
  float t;
  float3 p, n, color;
  int material;
};
__device__ bool sphere(float3 ro, float3 rd, float3 c, float r, float3 color, int material, Hit& hit) {
  float3 oc = sub(ro, c);
  float b = dot3(oc, rd), q = b * b - dot3(oc, oc) + r * r;
  if (q < 0)
    return false;
  float t = -b - sqrtf(q);
  if (t < .002f)
    t = -b + sqrtf(q);
  if (t < .002f || t >= hit.t)
    return false;
  hit.t = t;
  hit.p = add(ro, mul(rd, t));
  hit.n = norm(sub(hit.p, c));
  hit.color = color;
  hit.material = material;
  return true;
}
__device__ bool scene(float3 ro, float3 rd, Hit& h) {
  h.t = 1e20f;
  bool yes = false;
  yes |= sphere(ro, rd, make_float3(-.72f, .02f, .12f), .72f, make_float3(.92f, .97f, 1), 2, h);
  yes |= sphere(ro, rd, make_float3(.68f, -.14f, -.18f), .56f, make_float3(1, .22f, .045f), 1, h);
  yes |= sphere(ro, rd, make_float3(.05f, .82f, .04f), .42f, make_float3(.82f, .45f, 1), 0, h);
  yes |= sphere(ro, rd, make_float3(-.08f, -.48f, .7f), .28f, make_float3(.1f, .65f, 1), 1, h);
  yes |= sphere(ro, rd, make_float3(.08f, -.55f, -.78f), .23f, make_float3(1, .72f, .22f), 0, h);
  if (fabsf(rd.y) > 1e-5f) {
    float t = (-.78f - ro.y) / rd.y;
    if (t > .002f && t < h.t) {
      h.t = t;
      h.p = add(ro, mul(rd, t));
      h.n = make_float3(0, 1, 0);
      h.color = make_float3(.15f, .17f, .22f);
      h.material = 1;
      yes = true;
    }
  }
  return yes;
}
__device__ float3 sky(float3 d) {
  float sun = powf(fmaxf(0, dot3(d, norm(make_float3(-.3f, .85f, -.4f)))), 500);
  float h = .5f + .5f * d.y;
  return add(make_float3(.018f + .08f * h, .025f + .11f * h, .055f + .22f * h),
             mul(make_float3(8, 5.5f, 3), sun));
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;
  unsigned seed = (x + y * params.width) * 9781u + (unsigned)params.frame * 6271u + 17u;
  float3 total = make_float3(0, 0, 0);
  int samples = 1 + params.quality;
  for (int sample = 0; sample < samples; sample++) {
    float2 uv = make_float2((2 * (x + random(seed)) - params.width) / params.height,
                            (params.height - 2 * (y + random(seed))) / params.height);
    float yaw = (params.mouse_x - .5f) * 1.5f, pitch = (params.mouse_y - .5f) * .55f;
    float3 ro = make_float3(3.7f * sinf(yaw), .25f + pitch, 3.7f * cosf(yaw));
    float3 fw = norm(mul(ro, -1)), rt = norm(cross3(fw, make_float3(0, 1, 0))), up = cross3(rt, fw),
           rd = norm(add(fw, add(mul(rt, uv.x * .68f), mul(up, uv.y * .68f))));
    float3 throughput = make_float3(1, 1, 1), radiance = make_float3(0, 0, 0);
    for (int bounce = 0; bounce < 7; bounce++) {
      Hit hit;
      if (!scene(ro, rd, hit)) {
        radiance = add(radiance, had(throughput, sky(rd)));
        break;
      }
      float facing = dot3(rd, hit.n);
      if (facing > 0)
        hit.n = mul(hit.n, -1);
      if (hit.material == 2) {
        float eta = facing < 0 ? 1 / 1.46f : 1.46f, cosi = fabsf(dot3(rd, hit.n)),
              k = 1 - eta * eta * (1 - cosi * cosi), fres = .035f + .965f * powf(1 - cosi, 5);
        if (k < 0 || random(seed) < fres)
          rd = reflect3(rd, hit.n);
        else
          rd = norm(add(mul(rd, eta), mul(hit.n, eta * cosi - sqrtf(k))));
        throughput = had(throughput, hit.color);
      } else if (hit.material == 1) {
        rd = norm(add(reflect3(rd, hit.n), mul(random_sphere(seed), .055f)));
        throughput = had(throughput, hit.color);
      } else {
        rd = norm(add(hit.n, random_sphere(seed)));
        throughput = had(throughput, hit.color);
      }
      ro = add(hit.p, mul(rd, .003f));
      if (bounce > 3) {
        float survive = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (random(seed) > survive)
          break;
        throughput = mul(throughput, 1 / fmaxf(.1f, survive));
      }
    }
    total = add(total, radiance);
  }
  total = mul(total, 1.0f / samples);
  total = cudalab_tonemap(total);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(total.x), .4545f),
                                             255 * powf(cudalab_saturate(total.y), .4545f),
                                             255 * powf(cudalab_saturate(total.z), .4545f),
                                             255);
}
