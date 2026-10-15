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

// Repeated vault ribs, columns, and a tiled obsidian floor. The vault is a
// clipped circular SDF extruded across Z, so marching remains conservative.
__device__ float scene(float3 p) {
  float z = p.z - 6 * floorf(p.z / 6 + .5f);
  float column = hypotf(fabsf(p.x) - 3.6f, z) - .23f;
  column = fmaxf(column, fmaxf(-p.y, p.y - 4));
  float arch = fmaxf(fabsf(hypotf(p.x, p.y - 4) - 3.6f) - .16f, fabsf(z) - .20f);
  arch = fmaxf(arch, 4 - p.y);
  float base = fmaxf(fabsf(fabsf(p.x) - 3.6f) - .43f, fmaxf(fabsf(z) - .48f, fabsf(p.y - .22f) - .22f));
  return fminf(p.y, fminf(base, fminf(column, arch)));
}
__device__ float march(float3 ro, float3 rd, int steps, float far) {
  float d = .025f;
  for (int j = 0; j < steps && d < far; j++) {
    float h = scene(add(ro, mul(rd, d)));
    if (h < fmaxf(.0008f, .00015f * d))
      return d;
    d += fmaxf(.002f, h * .85f);
  }
  return far;
}
__device__ float3 normal(float3 p) {
  float e = .002f;
  return norm(make_float3(scene(add(p, make_float3(e, 0, 0))) - scene(add(p, make_float3(-e, 0, 0))),
                          scene(add(p, make_float3(0, e, 0))) - scene(add(p, make_float3(0, -e, 0))),
                          scene(add(p, make_float3(0, 0, e))) - scene(add(p, make_float3(0, 0, -e)))));
}
__device__ float3 sky(float3 rd, float t) {
  float3 sun = norm(make_float3(.04f, .19f, 1));
  float a = acosf(fminf(.999999f, dot(rd, sun)));
  float ring = expf(-fabsf(a - .15f) * 240);
  float corona = expf(-fabsf(a - .15f) * 25);
  float3 c = make_float3(.014f, .023f, .045f);
  c = add(c, mul(make_float3(1, .42f, .12f), ring * 3 + corona * .4f));
  return c;
}
__device__ float3 shade(float3 ro, float3 rd, float d, float t) {
  if (d >= 85)
    return sky(rd, t);
  float3 p = add(ro, mul(rd, d)), n = normal(p);
  float diffuse = fmaxf(0, dot(n, norm(make_float3(-.6f, 1, -.4f))));
  float occlusion = cudalab_saturate(scene(add(p, mul(n, .35f))) / .35f);
  float3 c = mul(make_float3(.20f, .25f, .32f), (.12f + .65f * diffuse) * (.35f + .65f * occlusion));
  float z = p.z - 6 * floorf(p.z / 6 + .5f);
  float flute = .88f + .12f * cosf(atan2f(p.x - copysignf(3.6f, p.x), z) * 20);
  c = mul(c, p.y > .45f && p.y < 4 ? flute : 1);
  float edge = expf(-fabsf(fabsf(z) - .205f) * 65);
  float height = smooth(2.8f, 4.2f, p.y);
  c = add(c,
          mul(make_float3(.16f, .65f, 1.3f),
              edge * height * (.6f + .4f * sinf((p.z - t) * .12f - t * (6.2831853f / 48)))));
  float fog = expf(-d * .027f);
  return add(mul(c, fog), mul(sky(rd, t), 1 - fog));
}
CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;
  float2 uv = make_float2((2.f * x + 1 - params.width) / params.height,
                          (params.height - 2.f * y - 1) / params.height);
  float t = fmodf(params.time, 48.f), cycle = 6.2831853f * t / 48;
  // Eight bays per loop: the repeating architecture matches at the cut.
  float3 ro = make_float3(.8f * sinf(cycle) + (params.mouse_x - .5f) * 1.3f,
                          1.5f + .6f * (1 - cosf(cycle)) + (params.mouse_y - .5f) * .7f,
                          t);
  float3 target = add(ro, make_float3(-ro.x * .45f, .65f + .8f * sinf(cycle), 8));
  float3 rd = camera(ro, target, uv, 1.65f, .045f * sinf(cycle));
  int q = params.quality < 0 ? 0 : (params.quality > 3 ? 3 : params.quality);
  float d = march(ro, rd, 112 + q * 32, 85);
  float3 c = shade(ro, rd, d, t);
  float3 p = add(ro, mul(rd, d));
  if (d < 85 && p.y < .035f) {
    // One bounded reflection ray; analytic tile seams retain the floor's scale.
    float3 reflected = make_float3(rd.x, -rd.y, rd.z);
    float3 start = make_float3(p.x, .045f, p.z);
    float r = march(start, reflected, 80 + q * 24, 85);
    float3 reflection = shade(start, reflected, r, t);
    float seam = fminf(fabsf(p.x - floorf(p.x + .5f)), fabsf(p.z - floorf(p.z + .5f)));
    float fresnel = .22f + .65f * powf(1 - fabsf(rd.y), 5);
    c = add(mul(c, .25f), mul(reflection, fresnel * (.45f + .55f * smooth(0, .025f, seam))));
  }
  // Low-density shafts integrated along the visible ray, occluded by geometry.
  float3 light = make_float3(0, 0, 0);
  int samples = 16 + q * 8;
  float stride = fminf(d, 45.f) / samples;
  for (int j = 0; j < samples; j++) {
    float distance = (j + .5f) * stride;
    float3 v = add(ro, mul(rd, distance));
    float z = v.z - 6 * floorf(v.z / 6 + .5f);
    // Multiplication is deliberate: fast-math powf of a negative base is undefined.
    float beam = expf(-square((v.x + 1.7f - .32f * v.y) * 5) - z * z * 2);
    float ribbon = expf(-square((v.x - 2.8f + .15f * v.y) * 8) - z * z * 4);
    light = add(
        light,
        mul(make_float3(beam * .65f + ribbon * .06f, beam * .32f + ribbon * .24f, beam * .1f + ribbon * .5f),
            stride * .12f * expf(-distance * .035f)));
  }
  c = add(c, mul(light, 1 + params.mouse_down * 2));
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
  int roots[4] = {38, 34, 41, 36};
  int root = roots[(int)(t / 12)];
  float section = fmodf(t, 12.0f);
  float padenv = smooth(0, 1.5f, section) * (1 - smooth(10, 12, section));
  float left = 0, right = 0;
  int chord[4] = {0, 7, 12, 15};
  for (int v = 0; v < 4; v++) {
    float f = hz(root + chord[v]);
    float phase = 6.2831853f * f * section;
    left += .047f * padenv * (sinf(phase) + .22f * sinf(phase * 2.001f + v));
    right += .047f * padenv * (sinf(phase * 1.0012f) + .22f * sinf(phase * 2.002f + v));
  }
  float step = .3125f, age = fmodf(t, step);
  int sequence[8] = {12, 19, 24, 27, 19, 15, 24, 22};
  int note_root = roots[(int)((t - age) / 12)];
  float frequency = hz(note_root + sequence[((int)(t / step)) & 7]);
  float envelope = (1 - expf(-age * 100)) * expf(-age * 12) * (1 - smooth(.27f, step, age));
  float bell = sinf(6.2831853f * frequency * age) + .24f * sinf(6.2831853f * frequency * 2.01f * age);
  float pan = .5f + .35f * sinf(floorf(t / step) * 2.4f);
  float movement = smooth(5, 15, t) * (1 - smooth(38, 48, t));
  left += bell * envelope * (.045f + .09f * movement) * (1 - pan);
  right += bell * envelope * (.045f + .09f * movement) * pan;
  float beat = fmodf(t, .625f);
  // Integrated exponential pitch sweep avoids discontinuous kick frequency.
  float kick = sinf(6.2831853f * (42 * beat + 2.2f * (1 - expf(-beat * 32)))) * (1 - expf(-beat * 200)) *
               expf(-beat * 12) * .16f * movement;
  float fade = smooth(0, .025f, t) * (1 - smooth(47.7f, 48, t));
  samples[i] = make_float2(tanhf((left + kick) * fade), tanhf((right + kick) * fade));
}
