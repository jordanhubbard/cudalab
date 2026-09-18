#include <cudalab.cuh>

constexpr float pi = 3.14159265359f;
constexpr int score_size = 4096;

__device__ float frac(float x) {
  return x - floorf(x);
}

__device__ float hash11(float x) {
  return frac(sinf(x * 127.1f) * 43758.5453f);
}

__device__ float2 rotate2(float2 p, float angle) {
  float c = cosf(angle), s = sinf(angle);
  return make_float2(c * p.x - s * p.y, s * p.x + c * p.y);
}

__device__ float sd_segment(float2 p, float2 a, float2 b) {
  float2 pa = make_float2(p.x - a.x, p.y - a.y);
  float2 ba = make_float2(b.x - a.x, b.y - a.y);
  float h = cudalab_saturate((pa.x * ba.x + pa.y * ba.y) / (ba.x * ba.x + ba.y * ba.y));
  return hypotf(pa.x - ba.x * h, pa.y - ba.y * h);
}

__device__ float3 palette(float x) {
  return make_float3(.48f + .48f * cosf(6.283f * (x + .00f)),
                     .46f + .46f * cosf(6.283f * (x + .32f)),
                     .52f + .46f * cosf(6.283f * (x + .66f)));
}

CUDALAB_RESET {
  float4* score = cudalab_buffer<float4>(resources, resource_count, 0);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (score && i < score_size)
    score[i] = make_float4(0, 0, 0, 0);
}

CUDALAB_SIMULATE {
  float4* score = cudalab_buffer<float4>(resources, resource_count, 0);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!score || i >= score_size)
    return;
  float u = i / (float)score_size;
  float movement = floorf(fmodf(params.time, 32.0f) / 8.0f);
  float local = fmodf(params.time, 8.0f) / 8.0f;
  float angle = u * pi * 2 * (3 + movement) + params.time * (.15f + movement * .04f);
  float radius = .16f + .68f * sqrtf(u);
  if (movement == 1)
    radius *= .55f + .45f * sinf(angle * 5 + local * pi);
  if (movement == 2)
    radius = .18f + floorf(u * 12) / 16.0f;
  if (movement == 3)
    radius *= 1.0f - local * .72f;
  float x = cosf(angle) * radius + (params.mouse_x - .5f) * .18f;
  float y = sinf(angle * (movement == 2 ? .5f : 1.0f)) * radius;
  float pulse = sinf(pi * frac(u * 19 + local * 2));
  float energy = .35f + .65f * pulse * pulse;
  score[i] = make_float4(x, y, energy, movement + hash11(i) * .2f);
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height)
    return;

  float2 uv =
      make_float2((2.0f * x - params.width) / params.height, (params.height - 2.0f * y) / params.height);
  float local_time = fmodf(params.time, 32.0f);
  int movement = min(3, (int)(local_time / 8.0f));
  float local = fmodf(local_time, 8.0f) / 8.0f;
  float beat = frac(params.time * (112.0f / 60.0f));
  float pulse = expf(-beat * 5.5f);

  int write_slot = 1 + (params.frame & 1);
  int read_slot = 1 + ((params.frame + 1) & 1);
  float4 memory = make_float4(0, 0, 0, 0);
  if (resources && resource_count > read_slot && resources[read_slot].texture) {
    float2 warped = rotate2(make_float2((x + .5f) / params.width - .5f, (y + .5f) / params.height - .5f),
                            .002f + .004f * sinf(params.time * .13f));
    warped.x *= .997f - .004f * pulse;
    warped.y *= .997f - .004f * pulse;
    memory = tex2D<float4>(resources[read_slot].texture, warped.x + .5f, warped.y + .5f);
  }

  float3 color = make_float3(memory.x * .965f, memory.y * .958f, memory.z * .972f);
  float vignette = expf(-.42f * (uv.x * uv.x + uv.y * uv.y));
  color.x += .003f * vignette;
  color.y += .005f * vignette;
  color.z += .012f * vignette;

  float4* score = cudalab_buffer<float4>(resources, resource_count, 0);
  if (score) {
#pragma unroll
    for (int j = 0; j < 28; ++j) {
      int index = (j * 149 + ((x + y * 7) & 127) * 31) & (score_size - 1);
      float4 note = score[index];
      float d = hypotf(uv.x - note.x, uv.y - note.y);
      float glow = expf(-d * (50.0f - 14.0f * pulse)) * note.z;
      float3 ink = palette(note.w * .19f + index * .0007f + local * .15f);
      color.x += ink.x * glow * .13f;
      color.y += ink.y * glow * .13f;
      color.z += ink.z * glow * .17f;
    }
  }

  if (movement == 0) {
    float ring = fabsf(hypotf(uv.x, uv.y) - (.24f + local * .48f));
    float light = expf(-ring * 95) * (1.0f - local * .45f);
    color.x += light * .25f;
    color.y += light * .52f;
    color.z += light * 1.1f;
  } else if (movement == 1) {
    float a = atan2f(uv.y, uv.x);
    float r = hypotf(uv.x, uv.y);
    float petals = fabsf(r - .36f - .13f * sinf(a * 9 + params.time * .7f));
    float light = expf(-petals * 80) * (1 + pulse);
    color.x += light * 1.0f;
    color.y += light * .22f;
    color.z += light * .55f;
  } else if (movement == 2) {
    float columns = powf(.5f + .5f * cosf((uv.x + sinf(uv.y * 4) * .025f) * 38), 18.0f);
    float vault = expf(-fabsf(hypotf(uv.x * 1.15f, uv.y + .15f) - .62f) * 85);
    color.x += columns * .15f + vault * .7f;
    color.y += columns * .28f + vault * .42f;
    color.z += columns * .72f + vault * .95f;
  } else {
    float collapse = expf(-hypotf(uv.x, uv.y) * (4 + local * 28));
    float horizon = expf(-fabsf(uv.y + .32f * sinf(uv.x * 3 + params.time)) * 70);
    color.x += collapse * 1.2f + horizon * .3f;
    color.y += collapse * .68f + horizon * .12f;
    color.z += collapse * .22f + horizon * .7f;
  }

  color.x *= vignette;
  color.y *= vignette;
  color.z *= vignette;
  float4 remembered = make_float4(fminf(color.x, 4.0f), fminf(color.y, 4.0f), fminf(color.z, 4.0f), 1);
  if (resources && resource_count > write_slot && resources[write_slot].surface) {
    int sx0 = x * resources[write_slot].width / params.width;
    int sx1 = max(sx0 + 1, (x + 1) * resources[write_slot].width / params.width);
    int sy0 = y * resources[write_slot].height / params.height;
    int sy1 = max(sy0 + 1, (y + 1) * resources[write_slot].height / params.height);
    for (int sy = sy0; sy < min(sy1, resources[write_slot].height); ++sy) {
      for (int sx = sx0; sx < min(sx1, resources[write_slot].width); ++sx) {
        surf2Dwrite(remembered, resources[write_slot].surface, sx * (int)sizeof(float4), sy);
      }
    }
  }
  color = cudalab_tonemap(color);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(color.x), .4545f),
                                             255 * powf(cudalab_saturate(color.y), .4545f),
                                             255 * powf(cudalab_saturate(color.z), .4545f),
                                             255);
}

CUDALAB_COMPOSITE {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= params.width || i >= params.width * params.height)
    return;
  float progress = fmodf(params.time, 32.0f) / 32.0f;
  int y = params.height - 3;
  uchar4& pixel = pixels[y * params.width + i];
  if (i < progress * params.width) {
    float3 c = palette(progress + i / (float)params.width);
    pixel = make_uchar4(120 + 135 * c.x, 120 + 135 * c.y, 120 + 135 * c.z, 255);
  } else {
    pixel = make_uchar4(pixel.x / 3, pixel.y / 3, pixel.z / 3, 255);
  }
}

CUDALAB_AUDIO {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= sample_count)
    return;
  float t = (sample_offset + i) / (float)sample_rate;
  float show = fmodf(t, 32.0f);
  int movement = min(3, (int)(show / 8.0f));
  float beat_time = t * (112.0f / 60.0f);
  float beat = frac(beat_time);
  int roots[4] = {38, 43, 31, 46};
  int root = roots[movement];
  float base = 440.0f * exp2f((root - 69) / 12.0f);
  float env = expf(-beat * (5.0f + movement));
  float chord =
      sinf(2 * pi * base * t) + .42f * sinf(2 * pi * base * 1.5f * t) + .24f * sinf(2 * pi * base * 2.0f * t);
  float shimmer = sinf(2 * pi * base * (4 + movement) * t + sinf(t * .3f) * 3) * .12f;
  float kick = sinf(2 * pi * (48 + 70 * expf(-beat * 16)) * t) * expf(-beat * 12) * .22f;
  float signal = tanhf(chord * env * .16f + shimmer + kick);
  float pan = .5f + .35f * sinf(t * .17f + movement * 1.7f);
  samples[i] = make_float2(signal * sqrtf(1 - pan), signal * sqrtf(pan));
}
