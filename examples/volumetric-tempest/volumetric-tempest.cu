#include <cudalab.cuh>

__device__ float3 storm_add(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 storm_mul(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float storm_dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 storm_norm(float3 p) {
  return storm_mul(p, rsqrtf(fmaxf(storm_dot(p, p), 1e-8f)));
}

__device__ float storm_noise(float3 p) {
  float total = 0.0f;
  float amplitude = .55f;
  for (int octave = 0; octave < 5; ++octave) {
    float wave =
        sinf(p.x + sinf(p.z * 1.17f)) * sinf(p.y * 1.13f + cosf(p.x * .91f)) * sinf(p.z + sinf(p.y * 1.21f));
    total += amplitude * wave;
    p = make_float3(p.y * 1.73f + p.z * .31f, p.z * 1.61f - p.x * .27f, p.x * 1.79f + p.y * .23f);
    amplitude *= .53f;
  }
  return total;
}

__device__ float storm_density(float3 p, float time, float* lightning) {
  float radius = sqrtf(p.x * p.x + p.z * p.z);
  float angle = atan2f(p.z, p.x);
  float twist = angle + p.y * .72f - time * .13f;
  float3 q = make_float3(p.x * 1.15f + sinf(twist * 3.0f) * .42f,
                         p.y + sinf(radius * 2.1f - time * .17f) * .32f,
                         p.z * 1.15f + cosf(twist * 2.0f) * .42f);
  float turbulence = storm_noise(storm_add(q, make_float3(0.0f, time * .07f, time * -.04f)));
  float shell = cudalab_saturate(1.0f - fabsf(radius - (1.8f + .45f * sinf(p.y * .8f))) * .72f);
  float vertical = cudalab_saturate(1.0f - fabsf(p.y) * .18f);
  float density = cudalab_saturate((turbulence + .24f) * 1.8f) * shell * vertical;

  float bolt_x = .32f * sinf(p.y * 2.7f + time * 1.7f) + .10f * sinf(p.y * 11.0f);
  float bolt_z = .30f * cosf(p.y * 2.4f - time * 1.3f) + .08f * sinf(p.y * 13.0f + 1.0f);
  float bolt_distance = sqrtf((p.x - bolt_x) * (p.x - bolt_x) + (p.z - bolt_z) * (p.z - bolt_z));
  float pulse = powf(.5f + .5f * sinf(time * 3.7f + floorf(p.y * 2.0f)), 13.0f);
  *lightning = expf(-bolt_distance * 42.0f) * (.18f + 3.5f * pulse);
  return density;
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;

  float2 uv =
      make_float2((2.0f * x - params.width) / params.height, (params.height - 2.0f * y) / params.height);
  float orbit = (params.mouse_x - .5f) * 3.0f + params.time * .035f;
  float distance = 6.8f - params.mouse_y * 2.3f;
  float3 ro =
      make_float3(sinf(orbit) * distance, .4f + (params.mouse_y - .5f) * 2.0f, cosf(orbit) * distance);
  float3 forward = storm_norm(storm_mul(ro, -1.0f));
  float3 right = storm_norm(make_float3(forward.z, 0.0f, -forward.x));
  float3 up =
      make_float3(right.z * forward.y, right.x * forward.z - right.z * forward.x, -right.x * forward.y);
  float3 ray =
      storm_norm(storm_add(forward, storm_add(storm_mul(right, uv.x * .72f), storm_mul(up, uv.y * .72f))));

  int steps = 64 + params.quality * 24;
  float step_size = 10.0f / steps;
  float transmittance = 1.0f;
  float3 color = make_float3(.003f, .002f, .012f);
  float jitter = sinf(x * 12.9898f + y * 78.233f) * 43758.5453f;
  jitter -= floorf(jitter);
  float t = .7f + jitter * step_size;
  for (int i = 0; i < steps && transmittance > .012f; ++i, t += step_size) {
    float3 p = storm_add(ro, storm_mul(ray, t));
    float lightning = 0.0f;
    float density = storm_density(p, params.time, &lightning);
    float absorption = density * step_size * 1.45f;
    float glow = density * density;
    float height_hue = .5f + .5f * sinf(p.y * .42f + params.time * .08f);
    float3 emission = make_float3(.10f + .42f * height_hue + lightning * 2.3f,
                                  .025f + .11f * glow + lightning * 2.8f,
                                  .20f + .65f * (1.0f - height_hue) + lightning * 4.2f);
    color = storm_add(color, storm_mul(emission, transmittance * (absorption + lightning * step_size)));
    transmittance *= expf(-absorption);
  }

  float stars = powf(cudalab_saturate(sinf(x * 91.7f + sinf(y * 37.1f) * 19.0f)), 180.0f);
  color = storm_add(color,
                    make_float3(stars * transmittance, stars * transmittance, stars * transmittance * 1.4f));
  color = cudalab_tonemap(color);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(color.x), .4545f),
                                             255 * powf(cudalab_saturate(color.y), .4545f),
                                             255 * powf(cudalab_saturate(color.z), .4545f),
                                             255);
}
