#include <cudalab.cuh>

struct V {
  float x, y, z;
};
__device__ V v(float x, float y, float z) {
  return {x, y, z};
}
__device__ V add(V a, V b) {
  return v(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ V sub(V a, V b) {
  return v(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ V mul(V a, float s) {
  return v(a.x * s, a.y * s, a.z * s);
}
__device__ float dot(V a, V b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float len(V a) {
  return sqrtf(dot(a, a));
}
__device__ V norm(V a) {
  return mul(a, rsqrtf(dot(a, a) + 1e-8f));
}
__device__ V cross(V a, V b) {
  return v(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float sat(float x) {
  return fminf(1.0f, fmaxf(0.0f, x));
}
__device__ float hash(V p) {
  float h = sinf(dot(p, v(127.1f, 311.7f, 74.7f))) * 43758.5453f;
  return h - floorf(h);
}

CUDALAB_KERNEL {
  const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;
  const float aspect = (float)params.width / params.height;
  const float sx = ((x + .5f) / params.width * 2 - 1) * aspect, sy = 1 - (y + .5f) / params.height * 2;
  const float yaw = params.time * .055f + (params.mouse_x - .5f) * 2.2f;
  const float pitch = .16f + (params.mouse_y - .5f) * 1.25f;
  const float cy = cosf(yaw), syaw = sinf(yaw), cp = cosf(pitch), sp = sinf(pitch);
  V ro = v(10.5f * cy * cp, 10.5f * sp, 10.5f * syaw * cp), fw = norm(mul(ro, -1));
  V rt = norm(cross(fw, v(0, 1, 0))), up = cross(rt, fw);
  V rd = norm(add(fw, add(mul(rt, sx * .68f), mul(up, sy * .68f))));
  V col = v(0, 0, 0), p = ro;
  float trans = 1, glow = 0, oldY = p.y;
  bool swallowed = false;
  const int steps = 100 + params.quality * 35;
  for (int i = 0; i < steps; i++) {
    const float r = len(p);
    if (r < .82f) {
      swallowed = true;
      break;
    }
    if (r > 24 && i > 8)
      break;
    const float ds = fminf(.19f, fmaxf(.028f, r * .038f));
    rd = norm(sub(rd, mul(p, 1.52f / (r * r * r + .08f) * ds)));
    V np = add(p, mul(rd, ds));
    if ((oldY * np.y <= 0 || fabsf(np.y) < .025f) && trans > .01f) {
      const float rr = hypotf(np.x, np.z);
      if (rr > 1.28f && rr < 7.4f) {
        const float phi = atan2f(np.z, np.x);
        const float bands = .62f + .38f * sinf(rr * 17 - phi * 6 + params.time * 2.2f);
        const float noise = hash(v(floorf(rr * 18), floorf(phi * 25), 0));
        const float inner = powf(1.28f / rr, 1.65f);
        const float edge = sat((rr - 1.28f) * 2) * sat((7.4f - rr) * .65f);
        const V tangent = norm(v(-np.z, 0, np.x));
        const float dop = fminf(1.85f, fmaxf(.28f, 1 + .72f * dot(tangent, mul(rd, -1))));
        const float e = edge * inner * (.55f + .35f * bands + .22f * noise) * dop * dop;
        V hot = v(1.35f, .26f, .035f + inner * .9f + .5f * sat(dop - 1));
        col = add(col, mul(hot, e * trans * 1.9f));
        trans *= fmaxf(.12f, 1 - e * .72f);
      }
    }
    glow += expf(-fabsf(r - 1.05f) * 7) * ds * .055f;
    oldY = np.y;
    p = np;
  }
  if (!swallowed) {
    const float cell = 190;
    V id = v(floorf(rd.x * cell), floorf(rd.y * cell), floorf(rd.z * cell));
    const float h = hash(id), star = powf(sat((h - .985f) / .015f), 7);
    const float neb = powf(sat(.5f + .5f * sinf(rd.x * 7 + rd.z * 5 + sinf(rd.y * 11))), 5);
    col =
        add(col,
            mul(add(mul(v(.006f, .009f, .025f), 1 + neb * 2), mul(v(.8f, .82f, 1.15f), star * 2.8f)), trans));
  }
  col = add(col, mul(v(.20f, .055f, .012f), glow * trans));
  float3 tone = cudalab_tonemap(make_float3(col.x, col.y, col.z));
  tone = make_float3(powf(sat(tone.x), .4545f), powf(sat(tone.y), .4545f), powf(sat(tone.z), .4545f));
  pixels[y * params.width + x] = make_uchar4(255 * tone.x, 255 * tone.y, 255 * tone.z, 255);
}
