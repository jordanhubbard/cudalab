#include <cudalab.cuh>

constexpr int SHIP_COUNT = 12;
struct OceanShip {
  float4 motion; // position x/z, velocity x/z
  float4 traits; // heading, individual seed, age, sail trim
};

__device__ float3 ocean_add(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 ocean_sub(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 ocean_mul(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}
__device__ float ocean_dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 ocean_cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float3 ocean_norm(float3 v) {
  return ocean_mul(v, rsqrtf(fmaxf(ocean_dot(v, v), 1e-8f)));
}

__device__ float ocean_hash(unsigned value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return (value & 0x00ffffffu) / 16777216.0f;
}

__device__ float2 ocean_wind(CudalabParams params) {
  float2 wind = make_float2((params.mouse_x - .5f) * 2.0f, (params.mouse_y - .5f) * 2.0f);
  float magnitude = hypotf(wind.x, wind.y);
  if (magnitude < .24f)
    return make_float2(0.0f, -1.0f);
  return make_float2(wind.x / magnitude, wind.y / magnitude);
}

// Height and exact first derivatives. Keeping the surface differentiable makes the
// intersection and reflected image continuous even when the high-frequency bands move.
__device__ float3 ocean_wave(float2 p, float time, int bands, float2 wind, int beaufort) {
  float height = 0.0f;
  float gradient_x = 0.0f;
  float gradient_z = 0.0f;
  float force = cudalab_saturate(beaufort / 9.0f);
  float amplitude = .065f + .19f * force;
  float frequency = .48f;
  float speed = .24f + .72f * force;
  float wind_angle = atan2f(wind.y, wind.x);
  for (int i = 0; i < bands; ++i) {
    float spread = 1.65f - force * .72f;
    float angle = wind_angle + spread * sinf(i * 2.3999632f) + .12f * sinf(i * 1.73f);
    float2 direction = make_float2(cosf(angle), sinf(angle));
    float phase = (p.x * direction.x + p.y * direction.y) * frequency + time * speed + i * .37f;
    float sine = sinf(phase);
    float derivative = amplitude * frequency * cosf(phase);
    height += amplitude * sine;
    gradient_x += derivative * direction.x;
    gradient_z += derivative * direction.y;
    frequency *= 1.22f;
    amplitude *= .75f;
    speed *= 1.035f;
  }
  return make_float3(height, gradient_x, gradient_z);
}

__device__ float3 ocean_sky(float3 ray, float time) {
  float horizon = powf(cudalab_saturate(1.0f - fabsf(ray.y)), 3.0f);
  float daylight = cudalab_saturate(ray.y * .5f + .65f);
  float3 sun_direction = ocean_norm(make_float3(.18f, .43f, -.88f));
  float sun = powf(cudalab_saturate(ocean_dot(ray, sun_direction)), 700.0f);
  float cloud_field = .48f + .24f * sinf(ray.x * 7.0f + ray.z * 2.0f + time * .025f) +
                      .18f * sinf(ray.z * 11.0f - ray.x * 3.0f - time * .018f);
  float cloud = cudalab_saturate((cloud_field - .48f) * 3.2f);
  cloud = cloud * cloud * (3.0f - 2.0f * cloud) * cudalab_saturate(ray.y * 3.0f + .7f);
  return make_float3(.025f + .24f * horizon + .10f * daylight + 7.0f * sun + .34f * cloud,
                     .055f + .38f * horizon + .18f * daylight + 4.5f * sun + .36f * cloud,
                     .13f + .68f * horizon + .38f * daylight + 2.0f * sun + .39f * cloud);
}

__device__ float ocean_ellipsoid(float3 origin, float3 ray, float3 scale) {
  float3 o = make_float3(origin.x / scale.x, origin.y / scale.y, origin.z / scale.z);
  float3 d = make_float3(ray.x / scale.x, ray.y / scale.y, ray.z / scale.z);
  float a = ocean_dot(d, d);
  float b = ocean_dot(o, d);
  float c = ocean_dot(o, o) - 1.0f;
  float discriminant = b * b - a * c;
  if (discriminant <= 0.0f)
    return 1e4f;
  float t = (-b - sqrtf(discriminant)) / a;
  return t > 0.0f ? t : 1e4f;
}

__device__ float ocean_box(float3 origin, float3 ray, float3 center, float3 extent) {
  float3 o = ocean_sub(origin, center);
  float3 inverse = make_float3(1.0f / (fabsf(ray.x) > 1e-5f ? ray.x : copysignf(1e-5f, ray.x)),
                               1.0f / (fabsf(ray.y) > 1e-5f ? ray.y : copysignf(1e-5f, ray.y)),
                               1.0f / (fabsf(ray.z) > 1e-5f ? ray.z : copysignf(1e-5f, ray.z)));
  float3 near_value = make_float3(
      (-extent.x - o.x) * inverse.x, (-extent.y - o.y) * inverse.y, (-extent.z - o.z) * inverse.z);
  float3 far_value =
      make_float3((extent.x - o.x) * inverse.x, (extent.y - o.y) * inverse.y, (extent.z - o.z) * inverse.z);
  float near_t = fmaxf(fmaxf(fminf(near_value.x, far_value.x), fminf(near_value.y, far_value.y)),
                       fminf(near_value.z, far_value.z));
  float far_t = fminf(fminf(fmaxf(near_value.x, far_value.x), fmaxf(near_value.y, far_value.y)),
                      fmaxf(near_value.z, far_value.z));
  return far_t >= fmaxf(near_t, 0.0f) ? fmaxf(near_t, 0.0f) : 1e4f;
}

__device__ float ocean_mast(float3 origin, float3 ray, float mast_x) {
  float ox = origin.x - mast_x;
  float a = ray.x * ray.x + ray.z * ray.z;
  float b = ox * ray.x + origin.z * ray.z;
  float c = ox * ox + origin.z * origin.z - .0016f;
  float discriminant = b * b - a * c;
  if (discriminant <= 0.0f || a < 1e-6f)
    return 1e4f;
  float t = (-b - sqrtf(discriminant)) / a;
  float height = origin.y + ray.y * t;
  return t > 0.0f && height > .20f && height < 1.40f ? t : 1e4f;
}

__device__ float ocean_sail(float3 origin, float3 ray, float mast_x) {
  if (fabsf(ray.z) < 1e-5f)
    return 1e4f;
  float t = -origin.z / ray.z;
  float3 p = ocean_add(origin, ocean_mul(ray, t));
  if (t <= 0.0f || p.y < .48f || p.y > 1.34f)
    return 1e4f;
  float taper = (1.34f - p.y) / .86f;
  bool main_sail = p.x > mast_x + .035f && p.x < mast_x + .66f * taper + .04f;
  bool fore_sail = p.x < mast_x - .04f && p.x > mast_x - .36f * taper - .04f && p.y < 1.16f;
  return main_sail || fore_sail ? t : 1e4f;
}

__device__ float ocean_segment(float2 p, float2 a, float2 b) {
  float2 pa = make_float2(p.x - a.x, p.y - a.y);
  float2 ba = make_float2(b.x - a.x, b.y - a.y);
  float denominator = ba.x * ba.x + ba.y * ba.y;
  float along = cudalab_saturate((pa.x * ba.x + pa.y * ba.y) / fmaxf(denominator, 1e-6f));
  return hypotf(pa.x - ba.x * along, pa.y - ba.y * along);
}

CUDALAB_RESET {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= SHIP_COUNT || state_bytes < sizeof(OceanShip) * SHIP_COUNT)
    return;
  OceanShip* ships = static_cast<OceanShip*>(state);
  float column = i % 4;
  float row = i / 4;
  float jitter_x = ocean_hash(i * 5u + 1u) - .5f;
  float jitter_z = ocean_hash(i * 5u + 2u) - .5f;
  float speed = .16f + ocean_hash(i * 5u + 3u) * .08f;
  float heading = -1.5707963f + jitter_x * .62f;
  ships[i].motion = make_float4((column - 1.5f) * 1.85f + jitter_x * .52f,
                                3.45f - row * 2.18f + jitter_z * .46f,
                                cosf(heading) * speed,
                                sinf(heading) * speed);
  ships[i].traits = make_float4(
      heading, ocean_hash(i * 5u + 4u), ocean_hash(i * 5u + 5u) * 24.0f, .7f + ocean_hash(i * 7u) * .3f);
}

CUDALAB_SIMULATE {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (!state || state_bytes < sizeof(OceanShip) * SHIP_COUNT)
    return;
  OceanShip* ships = static_cast<OceanShip*>(state);
  __shared__ OceanShip neighborhood[SHIP_COUNT];
  if (i < SHIP_COUNT)
    neighborhood[i] = ships[i];
  __syncthreads();
  if (i >= SHIP_COUNT)
    return;

  OceanShip ship = neighborhood[i];
  float2 position = make_float2(ship.motion.x, ship.motion.y);
  float2 velocity = make_float2(ship.motion.z, ship.motion.w);
  float2 cohesion = make_float2(0.0f, 0.0f);
  float2 alignment = make_float2(0.0f, 0.0f);
  float2 separation = make_float2(0.0f, 0.0f);
  int neighbors = 0;
  for (int other = 0; other < SHIP_COUNT; ++other) {
    if (other == i)
      continue;
    float2 offset =
        make_float2(neighborhood[other].motion.x - position.x, neighborhood[other].motion.y - position.y);
    float distance_squared = offset.x * offset.x + offset.y * offset.y;
    if (distance_squared < 10.0f) {
      cohesion.x += neighborhood[other].motion.x;
      cohesion.y += neighborhood[other].motion.y;
      alignment.x += neighborhood[other].motion.z;
      alignment.y += neighborhood[other].motion.w;
      ++neighbors;
    }
    if (distance_squared < 1.70f && distance_squared > 1e-5f) {
      separation.x -= offset.x / distance_squared;
      separation.y -= offset.y / distance_squared;
    }
  }

  float dt = fminf(params.delta, 1.0f / 20.0f);
  float2 wind = ocean_wind(params);
  float force = params.beaufort / 9.0f;
  float target_speed = .035f + force * .48f;
  float2 desired = make_float2(wind.x * target_speed, wind.y * target_speed);
  velocity.x += (desired.x - velocity.x) * dt * (.24f + force * .38f);
  velocity.y += (desired.y - velocity.y) * dt * (.24f + force * .38f);
  if (neighbors > 0) {
    float inverse = 1.0f / neighbors;
    cohesion = make_float2(cohesion.x * inverse - position.x, cohesion.y * inverse - position.y);
    alignment = make_float2(alignment.x * inverse - velocity.x, alignment.y * inverse - velocity.y);
    velocity.x += (cohesion.x * .018f + alignment.x * .10f + separation.x * .24f) * dt;
    velocity.y += (cohesion.y * .018f + alignment.y * .10f + separation.y * .24f) * dt;
  }
  float wander = sinf(params.time * (.31f + ship.traits.y * .17f) + ship.traits.y * 31.0f);
  velocity.x += -wind.y * wander * dt * (.018f + ship.traits.y * .018f);
  velocity.y += wind.x * wander * dt * (.018f + ship.traits.y * .018f);
  float speed = hypotf(velocity.x, velocity.y);
  float maximum = fmaxf(.075f, target_speed * 1.32f);
  if (speed > maximum) {
    velocity.x *= maximum / speed;
    velocity.y *= maximum / speed;
  }
  position.x += velocity.x * dt;
  position.y += velocity.y * dt;
  ship.traits.z += dt;
  float desired_heading = atan2f(velocity.y, velocity.x);
  float heading_delta = atan2f(sinf(desired_heading - ship.traits.x), cosf(desired_heading - ship.traits.x));
  ship.traits.x += heading_delta * dt * (.45f + force);

  // A vessel is recycled only after it has sailed beyond the visible world.
  if (fabsf(position.x) > 11.0f || fabsf(position.y) > 11.0f) {
    float2 side = make_float2(-wind.y, wind.x);
    float lane = (i - (SHIP_COUNT - 1) * .5f) * .54f + (ship.traits.y - .5f) * .7f;
    position = make_float2(-wind.x * 9.5f + side.x * lane, -wind.y * 9.5f + side.y * lane);
    velocity = make_float2(wind.x * target_speed * (.8f + ship.traits.y * .3f),
                           wind.y * target_speed * (.8f + ship.traits.y * .3f));
    ship.traits.z = 0.0f;
  }
  ships[i].motion = make_float4(position.x, position.y, velocity.x, velocity.y);
  ships[i].traits = ship.traits;
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;

  float2 screen =
      make_float2((2.0f * x - params.width) / params.height, (params.height - 2.0f * y) / params.height);
  float orbit = .12f * sinf(params.time * .055f);
  float distance = 10.2f;
  float3 ro = make_float3(sinf(orbit) * distance, 4.3f, cosf(orbit) * distance);
  float3 target = make_float3(0.0f, .05f, 0.0f);
  float3 forward = ocean_norm(ocean_sub(target, ro));
  float3 right = ocean_norm(make_float3(forward.z, 0.0f, -forward.x));
  float3 up = ocean_cross(forward, right);
  float3 ray = ocean_norm(
      ocean_add(forward, ocean_add(ocean_mul(right, screen.x * .72f), ocean_mul(up, screen.y * .72f))));

  int bands = 12 + params.quality * 3;
  int iterations = 7 + params.quality * 2;
  float2 wind = ocean_wind(params);
  const OceanShip* fleet = static_cast<const OceanShip*>(state);
  float water_t = ray.y < -.015f ? -ro.y / ray.y : 1e4f;
  for (int i = 0; i < iterations && water_t < 100.0f; ++i) {
    float3 point = ocean_add(ro, ocean_mul(ray, water_t));
    float3 wave = ocean_wave(make_float2(point.x, point.z), params.time, bands, wind, params.beaufort);
    float error = point.y - wave.x;
    float derivative = ray.y - wave.y * ray.x - wave.z * ray.z;
    float correction = error / copysignf(fmaxf(fabsf(derivative), .08f), derivative);
    water_t -= fminf(.65f, fmaxf(-.65f, correction));
  }

  float3 color = ocean_sky(ray, params.time);
  if (water_t > 0.0f && water_t < 100.0f) {
    float3 point = ocean_add(ro, ocean_mul(ray, water_t));
    float3 wave = ocean_wave(make_float2(point.x, point.z), params.time, bands, wind, params.beaufort);
    float3 normal = ocean_norm(make_float3(-wave.y, 1.0f, -wave.z));
    float facing = cudalab_saturate(-ocean_dot(ray, normal));
    float fresnel = .021f + .979f * powf(1.0f - facing, 5.0f);

    float3 reflected_ray = ocean_sub(ray, ocean_mul(normal, 2.0f * ocean_dot(ray, normal)));
    float3 reflection = ocean_sky(reflected_ray, params.time);
    float eta = .75f;
    float refract_k = fmaxf(0.0f, 1.0f - eta * eta * (1.0f - facing * facing));
    float3 refracted_ray = ocean_add(ocean_mul(ray, eta), ocean_mul(normal, eta * facing - sqrtf(refract_k)));
    float depth = 2.5f + 2.0f * (.5f + .5f * sinf(point.x * .13f + point.z * .17f));
    float3 absorption = make_float3(expf(-depth * .82f), expf(-depth * .24f), expf(-depth * .10f));
    float caustic_phase =
        ocean_wave(make_float2(point.x + refracted_ray.x * 2.2f, point.z + refracted_ray.z * 2.2f),
                   params.time + .35f,
                   bands,
                   wind,
                   params.beaufort)
            .x;
    float caustic = powf(.5f + .5f * cosf(caustic_phase * 18.0f), 10.0f);
    float3 refraction = make_float3(.008f + .035f * absorption.x + .035f * caustic,
                                    .055f + .22f * absorption.y + .065f * caustic,
                                    .11f + .42f * absorption.z + .050f * caustic);
    color = ocean_add(ocean_mul(refraction, 1.0f - fresnel), ocean_mul(reflection, fresnel));

    float3 sun_direction = ocean_norm(make_float3(.18f, .43f, -.88f));
    float glitter = powf(cudalab_saturate(ocean_dot(reflected_ray, sun_direction)), 220.0f);
    float slope = hypotf(wave.y, wave.z);
    float foam = cudalab_saturate((slope - .66f) * 1.8f);
    foam = foam * foam * (.70f + .30f * sinf(point.x * 5.0f + point.z * 4.0f + params.time));
    color = ocean_add(color,
                      make_float3(glitter * 5.0f + foam * .48f,
                                  glitter * 3.6f + foam * .62f,
                                  glitter * 1.8f + foam * .68f));

    // Diverging wakes are shaded in the water rather than pasted over the final frame.
    float wake = 0.0f;
    for (int ship = 0; ship < SHIP_COUNT && fleet; ++ship) {
      float2 center = make_float2(fleet[ship].motion.x, fleet[ship].motion.y);
      float2 heading = make_float2(cosf(fleet[ship].traits.x), sinf(fleet[ship].traits.x));
      float2 local = make_float2(point.x - center.x, point.z - center.y);
      float behind = -(local.x * heading.x + local.y * heading.y);
      float cross_track = local.x * -heading.y + local.y * heading.x;
      float spread = fabsf(cross_track) - behind * .19f;
      float arms = expf(-spread * spread * 22.0f) +
                   expf(-(fabsf(cross_track) + behind * .19f) * (fabsf(cross_track) + behind * .19f) * 22.0f);
      wake += cudalab_saturate(behind * .8f) * cudalab_saturate(1.0f - behind / 4.5f) * arms;
    }
    color = ocean_add(color, make_float3(wake * .32f, wake * .47f, wake * .52f));
  }

  float nearest_ship = water_t;
  int ship_material = -1;
  float ship_light = 1.0f;
  for (int ship = 0; ship < SHIP_COUNT && fleet; ++ship) {
    float2 center = make_float2(fleet[ship].motion.x, fleet[ship].motion.y);
    float3 wave = ocean_wave(center, params.time, bands, wind, params.beaufort);
    float3 boat_up = ocean_norm(make_float3(-wave.y, 1.0f, -wave.z));
    float heading = fleet[ship].traits.x;
    float forward_x = cosf(heading);
    float forward_z = sinf(heading);
    float directional_slope = wave.y * forward_x + wave.z * forward_z;
    float3 boat_forward = ocean_norm(make_float3(forward_x, directional_slope, forward_z));
    float3 boat_side = ocean_norm(ocean_cross(boat_up, boat_forward));
    boat_forward = ocean_norm(ocean_cross(boat_side, boat_up));
    float3 boat_center = make_float3(center.x, wave.x + .16f, center.y);
    float3 relative = ocean_sub(ro, boat_center);
    float3 local_origin = make_float3(
        ocean_dot(relative, boat_forward), ocean_dot(relative, boat_up), ocean_dot(relative, boat_side));
    float3 local_ray =
        make_float3(ocean_dot(ray, boat_forward), ocean_dot(ray, boat_up), ocean_dot(ray, boat_side));

    float hull_t = ocean_ellipsoid(local_origin, local_ray, make_float3(.78f, .22f, .31f));
    if (hull_t < nearest_ship) {
      nearest_ship = hull_t;
      ship_material = ship & 1;
      ship_light = .62f + .38f * cudalab_saturate(ocean_dot(boat_up, make_float3(-.2f, .8f, .4f)));
    }
    float deck_t =
        ocean_box(local_origin, local_ray, make_float3(-.05f, .19f, 0.0f), make_float3(.49f, .055f, .235f));
    if (deck_t < nearest_ship) {
      nearest_ship = deck_t;
      ship_material = 2;
      ship_light = .95f;
    }
    float cabin_t =
        ocean_box(local_origin, local_ray, make_float3(-.20f, .34f, 0.0f), make_float3(.17f, .13f, .16f));
    if (cabin_t < nearest_ship) {
      nearest_ship = cabin_t;
      ship_material = 3;
      ship_light = 1.05f;
    }
    float mast_t = ocean_mast(local_origin, local_ray, -.02f);
    if (mast_t < nearest_ship) {
      nearest_ship = mast_t;
      ship_material = 4;
      ship_light = 1.1f;
    }
    float sail_t = ocean_sail(local_origin, local_ray, -.02f);
    if (sail_t < nearest_ship) {
      nearest_ship = sail_t;
      ship_material = 5 + (ship & 1);
      ship_light = .92f + .18f * fabsf(local_ray.z);
    }
  }

  if (ship_material >= 0) {
    float3 materials[7] = {make_float3(.10f, .025f, .018f),
                           make_float3(.025f, .08f, .16f),
                           make_float3(.40f, .19f, .055f),
                           make_float3(.82f, .73f, .48f),
                           make_float3(.18f, .08f, .025f),
                           make_float3(.92f, .82f, .58f),
                           make_float3(.68f, .10f, .055f)};
    color = ocean_mul(materials[ship_material], ship_light);
  }

  // GPU-drawn windsock: the tail points downwind and the illuminated bars show Beaufort force.
  float2 hud = make_float2(x / (float)params.width, y / (float)params.height);
  float2 socket = make_float2(.90f, .095f);
  float2 pole_base = make_float2(.90f, .205f);
  float pole = ocean_segment(hud, socket, pole_base);
  float2 sock_direction = make_float2(wind.x * .75f - wind.y * .66f, wind.x * .20f + wind.y * .35f);
  float sock_direction_length = hypotf(sock_direction.x, sock_direction.y);
  sock_direction.x /= sock_direction_length;
  sock_direction.y /= sock_direction_length;
  float2 sock_tip = make_float2(socket.x + sock_direction.x * .078f, socket.y + sock_direction.y * .078f);
  float sock = ocean_segment(hud, socket, sock_tip);
  float sock_length = hypotf(sock_tip.x - socket.x, sock_tip.y - socket.y);
  float projection =
      ((hud.x - socket.x) * (sock_tip.x - socket.x) + (hud.y - socket.y) * (sock_tip.y - socket.y)) /
      fmaxf(sock_length * sock_length, 1e-6f);
  float taper = .012f * (1.0f - .68f * cudalab_saturate(projection));
  if (pole < .0022f)
    color = make_float3(.34f, .27f, .16f);
  if (sock < taper) {
    float stripe = fmodf(floorf(cudalab_saturate(projection) * 6.0f), 2.0f);
    color = stripe < .5f ? make_float3(1.7f, .12f, .035f) : make_float3(1.4f, 1.25f, .85f);
  }
  if (hypotf(hud.x - socket.x, hud.y - socket.y) < .009f)
    color = make_float3(1.4f, .65f, .08f);
  for (int mark = 0; mark < 9; ++mark) {
    float2 a = make_float2(.842f + mark * .013f, .225f);
    float2 b = make_float2(a.x, .225f - .004f - .0022f * mark);
    if (ocean_segment(hud, a, b) < .0032f)
      color = mark < params.beaufort ? make_float3(1.2f, .28f + mark * .035f, .035f)
                                     : make_float3(.10f, .13f, .16f);
  }

  color = cudalab_tonemap(color);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(color.x), .4545f),
                                             255 * powf(cudalab_saturate(color.y), .4545f),
                                             255 * powf(cudalab_saturate(color.z), .4545f),
                                             255);
}
