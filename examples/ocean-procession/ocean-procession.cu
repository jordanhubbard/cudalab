#include <cudalab.cuh>

__device__ float3 ocean_add(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 ocean_mul(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float ocean_dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 ocean_norm(float3 v) {
  return ocean_mul(v, rsqrtf(fmaxf(ocean_dot(v, v), 1e-8f)));
}

__device__ float ocean_height(float2 p, float time, int bands) {
  float h = 0.0f;
  float amplitude = .26f;
  float frequency = .55f;
  float speed = .55f;
  for (int i = 0; i < bands; ++i) {
    float angle = i * 2.3999632f + .35f * sinf(i * 1.71f);
    float2 direction = make_float2(cosf(angle), sinf(angle));
    float phase = (p.x * direction.x + p.y * direction.y) * frequency + time * speed;
    float wave = sinf(phase) + .35f * sinf(phase * 2.03f + i);
    h += amplitude * wave;
    frequency *= 1.235f;
    amplitude *= .82f;
    speed *= 1.055f;
  }
  return h;
}

__device__ float3 ocean_sky(float3 ray, float time) {
  float horizon = powf(cudalab_saturate(1.0f - fabsf(ray.y)), 5.0f);
  float sun = powf(cudalab_saturate(ocean_dot(ray, ocean_norm(make_float3(-.35f, .58f, .72f)))), 450.0f);
  float cloud =
      .5f + .5f * sinf(ray.x * 17.0f / fmaxf(.12f, ray.y + .35f) + sinf(ray.z * 13.0f + time * .04f) * 2.0f);
  cloud = powf(cloud, 7.0f) * cudalab_saturate(ray.y * 4.0f + .8f);
  return make_float3(.018f + .20f * horizon + 5.0f * sun + .16f * cloud,
                     .035f + .32f * horizon + 3.0f * sun + .18f * cloud,
                     .09f + .58f * horizon + 1.4f * sun + .22f * cloud);
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;

  float2 screen =
      make_float2((2.0f * x - params.width) / params.height, (params.height - 2.0f * y) / params.height);
  float orbit = (params.mouse_x - .5f) * 1.8f + .22f * sinf(params.time * .07f);
  float distance = 8.0f + params.mouse_y * 10.0f;
  float3 ro = make_float3(sinf(orbit) * distance, 3.2f + params.mouse_y * 3.8f, cosf(orbit) * distance);
  float3 target = make_float3(0.0f, -.15f, 0.0f);
  float3 forward = ocean_norm(make_float3(target.x - ro.x, target.y - ro.y, target.z - ro.z));
  float3 right = ocean_norm(make_float3(forward.z, 0.0f, -forward.x));
  float3 up =
      make_float3(right.z * forward.y, right.x * forward.z - right.z * forward.x, -right.x * forward.y);
  float3 ray = ocean_norm(
      ocean_add(forward, ocean_add(ocean_mul(right, screen.x * .78f), ocean_mul(up, screen.y * .78f))));

  int bands = 14 + params.quality * 4;
  int iterations = 8 + params.quality * 3;
  float t = ray.y < -.001f ? -ro.y / ray.y : 1000.0f;
  for (int i = 0; i < iterations && t < 100.0f; ++i) {
    float3 point = ocean_add(ro, ocean_mul(ray, t));
    float h = ocean_height(make_float2(point.x, point.z), params.time, bands);
    t += (h - point.y) / ray.y * .68f;
  }

  float3 c = ocean_sky(ray, params.time);
  if (t < 100.0f && t > 0.0f) {
    float3 point = ocean_add(ro, ocean_mul(ray, t));
    float epsilon = .018f;
    float h = ocean_height(make_float2(point.x, point.z), params.time, bands);
    float hx = ocean_height(make_float2(point.x + epsilon, point.z), params.time, bands);
    float hz = ocean_height(make_float2(point.x, point.z + epsilon), params.time, bands);
    float3 normal = ocean_norm(make_float3(h - hx, epsilon, h - hz));
    float fresnel = .035f + .965f * powf(1.0f - cudalab_saturate(-ocean_dot(ray, normal)), 5.0f);
    float3 reflected = ocean_add(ray, ocean_mul(normal, -2.0f * ocean_dot(ray, normal)));
    float3 sky = ocean_sky(reflected, params.time);
    float depth = expf(-t * .035f);
    c = make_float3(.005f + .015f * depth + sky.x * fresnel,
                    .035f + .13f * depth + sky.y * fresnel,
                    .075f + .24f * depth + sky.z * fresnel);
    float crest = cudalab_saturate((fabsf(hx - h) + fabsf(hz - h)) * 19.0f - .32f);
    c = ocean_add(c, make_float3(crest * .55f, crest * .72f, crest * .76f));

    float wake = 0.0f;
    for (int ship = 0; ship < 7; ++ship) {
      float phase = params.time * (.16f + ship * .008f) + ship * 5.37f;
      float2 center = make_float2(-5.5f + fmodf(phase + 30.0f, 12.0f),
                                  (ship - 3) * 1.28f + .45f * sinf(phase * .37f + ship));
      float dx = point.x - center.x;
      float dz = point.z - center.y;
      float trail = cudalab_saturate(-dx / 4.5f) * expf(-dz * dz * 2.4f) *
                    cudalab_saturate((dx + 5.5f) * .7f) * cudalab_saturate(-dx * .8f);
      wake += trail * (.5f + .5f * sinf(dx * 16.0f + fabsf(dz) * 20.0f));
    }
    c = ocean_add(c, make_float3(wake * .4f, wake * .6f, wake * .62f));
  }

  float nearest_ship = 1000.0f;
  float ship_pattern = 0.0f;
  float ship_light = 0.0f;
  for (int ship = 0; ship < 7; ++ship) {
    float phase = params.time * (.16f + ship * .008f) + ship * 5.37f;
    float2 center = make_float2(-5.5f + fmodf(phase + 30.0f, 12.0f),
                                (ship - 3) * 1.28f + .45f * sinf(phase * .37f + ship));
    float base = ocean_height(center, params.time, bands);
    float sample_x = ocean_height(make_float2(center.x + .18f, center.y), params.time, bands);
    float sample_z = ocean_height(make_float2(center.x, center.y + .18f), params.time, bands);
    float pitch = (sample_x - base) / .18f;
    float roll = (sample_z - base) / .18f;
    float plane_y = base + .16f;
    float ship_t = (plane_y - ro.y) / ray.y;
    if (ship_t <= 0.0f || ship_t >= nearest_ship || ship_t >= t)
      continue;
    float3 deck = ocean_add(ro, ocean_mul(ray, ship_t));
    float lx = deck.x - center.x;
    float lz = deck.z - center.y;
    ship_t = (plane_y + pitch * lx + roll * lz - ro.y) / ray.y;
    deck = ocean_add(ro, ocean_mul(ray, ship_t));
    lx = deck.x - center.x;
    lz = deck.z - center.y;
    float hull = fabsf(lx) / .72f + lz * lz / .075f;
    if (hull < 1.0f) {
      nearest_ship = ship_t;
      ship_pattern = .5f + .5f * sinf(lx * 38.0f + ship * 2.0f);
      ship_light = cudalab_saturate(.65f + pitch * .8f - roll * .45f);
    }
  }
  if (nearest_ship < 1000.0f) {
    float3 hull_color =
        make_float3(.20f + .75f * ship_pattern, .035f + .28f * ship_pattern, .018f + .08f * ship_pattern);
    c = ocean_mul(hull_color, .55f + ship_light);
  }

  c = cudalab_tonemap(c);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(c.x), .4545f),
                                             255 * powf(cudalab_saturate(c.y), .4545f),
                                             255 * powf(cudalab_saturate(c.z), .4545f),
                                             255);
}
