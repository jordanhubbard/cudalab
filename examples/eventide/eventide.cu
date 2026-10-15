#include <cudalab.cuh>

// Each lane owns one pixel (or stereo sample); no history is needed when seeking.
__device__ float3 add(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 mul(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 norm(float3 a) {
  return mul(a, rsqrtf(fmaxf(dot(a, a), 1e-8f)));
}
__device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float smooth(float a, float b, float x) {
  float u = cudalab_saturate((x - a) / (b - a));
  return u * u * (3 - 2 * u);
}
__device__ float square(float x) {
  return x * x;
}
__device__ float hash(float n) {
  float h = sinf(n * 127.1f) * 43758.5453f;
  return h - floorf(h);
}
__device__ float3 camera(float3 ro, float3 target, float2 uv, float lens, float roll) {
  float3 f = norm(add(target, mul(ro, -1))), r = norm(cross(f, make_float3(0, 1, 0)));
  float3 u = cross(r, f);
  float2 q = make_float2(uv.x * cosf(roll) - uv.y * sinf(roll), uv.x * sinf(roll) + uv.y * cosf(roll));
  return norm(add(mul(f, lens), add(mul(r, q.x), mul(u, q.y))));
}
__device__ uchar4 finish(float3 c, float2 uv) {
  // Linear-light exposure, gentle vignette, ACES curve, then display gamma.
  c = mul(c, (1 - .22f * smooth(.25f, 2.2f, dot(make_float3(uv.x, uv.y, 0), make_float3(uv.x, uv.y, 0)))));
  c = cudalab_tonemap(c);
  return make_uchar4(255 * powf(cudalab_saturate(c.x), .4545f),
                     255 * powf(cudalab_saturate(c.y), .4545f),
                     255 * powf(cudalab_saturate(c.z), .4545f),
                     255);
}

// Analytic sphere intersection and a finite-thickness, front-to-back ring volume.
// Ring samples are clipped against the planet so both near and far arcs occlude.
__device__ float sphere(float3 ro, float3 rd, float radius) {
  float b = dot(ro, rd), h = b * b - dot(ro, ro) + radius * radius;
  if (h < 0)
    return 1e5f;
  float d = -b - sqrtf(h);
  return d > 0 ? d : 1e5f;
}
__device__ float3 stars(float3 rd) {
  float2 uv = make_float2(atan2f(rd.z, rd.x) * 180, asinf(rd.y) * 180);
  float ix = floorf(uv.x), iy = floorf(uv.y), seed = hash(ix + iy * 317);
  float dx = uv.x - ix - .15f - .7f * hash(seed * 83), dy = uv.y - iy - .15f - .7f * hash(seed * 197);
  float star = expf(-(dx * dx + dy * dy) * 180) * powf(seed, 48);
  float haze = powf(fmaxf(0, 1 - fabsf(rd.y + .3f * rd.x)), 16) * .016f;
  return add(make_float3(.001f + haze * .35f, .003f + haze * .5f, .009f + haze),
             mul(make_float3(.7f + .3f * seed, .78f, .95f), star));
}
CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;
  float2 uv = make_float2((2.f * x + 1 - params.width) / params.height,
                          (params.height - 2.f * y - 1) / params.height);
  float t = fmodf(params.time, 48.f), phase = t * 6.2831853f / 48;
  float angle = .27f * sinf(phase) + (params.mouse_x - .5f) * .5f;
  float distance = 6.2f + .5f * cosf(phase);
  float3 ro = make_float3(
      distance * sinf(angle), 1.1f + .7f * sinf(phase) + (params.mouse_y - .5f), distance * cosf(angle));
  float3 rd = camera(ro, make_float3(.38f, 0, 0), uv, 2.0f, .18f + .08f * sinf(phase));
  float3 sun = norm(make_float3(-.22f + .42f * sinf(phase), -.10f, -1));
  float hit = sphere(ro, rd, 1.45f);
  float3 c = stars(rd);
  // A distant sun travels behind the limb. Its halo is angular, not screen-fixed.
  float alignment = fmaxf(0, dot(rd, sun));
  c = add(c, mul(make_float3(1, .67f, .32f), powf(alignment, 850) * 7 + powf(alignment, 32) * .08f));
  if (hit < 1e4f) {
    float3 p = add(ro, mul(rd, hit)), n = norm(p);
    float bands = .5f + .5f * sinf(p.y * 18 + 2 * sinf(p.x * 3 + p.z * 4 + phase));
    float lambert = fmaxf(0, dot(n, sun));
    float3 albedo = add(make_float3(.055f, .11f, .16f), mul(make_float3(.12f, .16f, .19f), bands));
    c = mul(albedo, .035f + lambert * .85f);
    float rim = powf(1 - fmaxf(0, -dot(n, rd)), 4);
    c = add(c, mul(make_float3(.08f, .42f, .85f), rim * (.18f + lambert * 1.5f)));
  }
  // Atmosphere outside the silhouette, with the brightest arc on the sunward side.
  float closest = fmaxf(0, -dot(ro, rd));
  float3 tangent = add(ro, mul(rd, closest));
  float limb = sqrtf(dot(tangent, tangent));
  if (hit > 1e4f && closest > 0) {
    float glow = expf(-fmaxf(0, limb - 1.45f) * 35);
    c = add(c, mul(make_float3(.12f, .48f, 1), glow * (.22f + .78f * fmaxf(0, dot(norm(tangent), sun)))));
  }
  float3 ringnormal = norm(make_float3(.12f, 1, .18f));
  float denom = dot(rd, ringnormal);
  // Intersect the slab, even near edge-on; bound the interval with a radius-3.5 sphere.
  float b = dot(ro, rd), disc = b * b - dot(ro, ro) + 3.5f * 3.5f;
  if (disc > 0 && fabsf(denom) > 1e-5f) {
    float a = (-.055f - dot(ro, ringnormal)) / denom, z = (.055f - dot(ro, ringnormal)) / denom;
    float enter = fmaxf(0, fmaxf(fminf(a, z), -b - sqrtf(disc)));
    float leave = fminf(hit, fminf(fmaxf(a, z), -b + sqrtf(disc)));
    if (leave > enter) {
      int q = params.quality < 0 ? 0 : (params.quality > 3 ? 3 : params.quality);
      int count = 24 + q * 12;
      float step = (leave - enter) / count, trans = 1;
      float3 ring = make_float3(0, 0, 0);
      for (int j = 0; j < count; j++) {
        float3 p = add(ro, mul(rd, enter + (j + .5f) * step));
        float r = sqrtf(dot(p, p));
        float edge = smooth(1.85f, 1.95f, r) * (1 - smooth(3.2f, 3.4f, r));
        float bands = .4f + .6f * powf(.5f + .5f * sinf(r * 130 + 2 * sinf(r * 31)), 3);
        float gap = 1 - .94f * expf(-square((r - 2.65f) * 35));
        float density = edge * bands * gap * 19;
        float opacity = 1 - expf(-density * step);
        float along = dot(p, sun);
        float clearance = sqrtf(fmaxf(0, dot(p, p) - along * along));
        float shadow = along < 0 ? .06f + .94f * smooth(1.40f, 1.53f, clearance) : 1;
        float pulse = .8f + .2f * sinf(r * 12 - phase * 2);
        float3 tint = make_float3(.95f, .45f + .15f * sinf(r * 8), .19f);
        ring = add(ring, mul(tint, trans * opacity * shadow * pulse * (1 + params.mouse_down)));
        trans *= 1 - opacity;
      }
      c = add(ring, mul(c, trans));
    }
  }
  pixels[y * params.width + x] = finish(c, uv);
}

// 48 seconds, 96 BPM: four 12-second movements. All oscillator phases are
// event-local and enveloped at both ends, including the loop and chord changes.
// Absolute sample positions make arbitrary timeline seeks deterministic.
__device__ float hz(int midi) {
  return 440 * exp2f((midi - 69) / 12.0f);
}
CUDALAB_AUDIO {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sample_count)
    return;
  float t = fmodf((sample_offset + i) / (float)sample_rate, 48.0f);
  int roots[4] = {45, 41, 48, 43};
  int root = roots[(int)(t / 12)];
  float section = fmodf(t, 12.0f);
  float padenv = smooth(0, 1.5f, section) * (1 - smooth(10, 12, section));
  float left = 0, right = 0;
  int chord[4] = {0, 7, 14, 19};
  for (int v = 0; v < 4; v++) {
    float f = hz(root + chord[v]);
    float phase = 6.2831853f * f * section;
    left += .047f * padenv * (sinf(phase) + .22f * sinf(phase * 2.001f + v));
    right += .047f * padenv * (sinf(phase * 1.0012f) + .22f * sinf(phase * 2.002f + v));
  }
  float step = .625f, age = fmodf(t, step);
  int sequence[8] = {12, 19, 24, 27, 19, 15, 24, 22};
  int note_root = roots[(int)((t - age) / 12)];
  float frequency = hz(note_root + sequence[((int)(t / step)) & 7]);
  float envelope = (1 - expf(-age * 100)) * expf(-age * 7) * (1 - smooth(.55f, step, age));
  float bell = sinf(6.2831853f * frequency * age) + .24f * sinf(6.2831853f * frequency * 2.01f * age);
  float pan = .5f + .35f * sinf(floorf(t / step) * 2.4f);
  float movement = smooth(5, 15, t) * (1 - smooth(38, 48, t));
  left += bell * envelope * (.045f + .09f * movement) * (1 - pan);
  right += bell * envelope * (.045f + .09f * movement) * pan;
  float beat = fmodf(t, .625f);
  // Integrated exponential pitch sweep avoids discontinuous kick frequency.
  float kick = sinf(6.2831853f * (42 * beat + 2.2f * (1 - expf(-beat * 32)))) * (1 - expf(-beat * 200)) *
               expf(-beat * 12) * .045f * movement;
  float fade = smooth(0, .025f, t) * (1 - smooth(47.7f, 48, t));
  samples[i] = make_float2(tanhf((left + kick) * fade), tanhf((right + kick) * fade));
}
