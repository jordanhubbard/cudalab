// Standalone validation for the two stateless 48-second cinematic artworks.
// Compile once per entry with DEMO_SOURCE; see docs/cinematic-demos.md.
// This exercises actual device kernels, independently of SDL/OpenGL/NVRTC.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include DEMO_SOURCE
#define CHECK(x)                                                                                             \
  do {                                                                                                       \
    auto e = (x);                                                                                            \
    if (e != cudaSuccess) {                                                                                  \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e));                                                \
      exit(1);                                                                                               \
    }                                                                                                        \
  } while (0)
int main(int argc, char** argv) {
  if (argc < 3 || argc > 7) {
    fprintf(stderr, "usage: probe OUTPUT.ppm TIME [MOUSE_X MOUSE_Y BUTTON QUALITY]\n");
    return 2;
  }
  CudalabParams p{960, 540, (float)atof(argv[2]), 1 / 60.f, .5, .5, 0, 2};
  if (argc > 3)
    p.mouse_x = atof(argv[3]);
  if (argc > 4)
    p.mouse_y = atof(argv[4]);
  if (argc > 5)
    p.mouse_down = atoi(argv[5]);
  if (argc > 6)
    p.quality = atoi(argv[6]);
  uchar4* out;
  CHECK(cudaMalloc(&out, p.width * p.height * 4));
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  std::vector<float> times;
  for (int n = 0; n < 31; n++) {
    CHECK(cudaEventRecord(a));
    render<<<dim3((p.width + 15) / 16, (p.height + 15) / 16), dim3(16, 16)>>>(out, nullptr, 0, p, nullptr, 0);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    if (n)
      times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  std::vector<uchar4> image(p.width * p.height);
  CHECK(cudaMemcpy(image.data(), out, image.size() * 4, cudaMemcpyDeviceToHost));
  // Catch blank/NaN-to-black regressions such as signed powf under fast math.
  double luminance = 0, squared = 0;
  for (auto c : image) {
    double v = (c.x + c.y + c.z) / 3.0;
    luminance += v;
    squared += v * v;
  }
  double mean = luminance / image.size();
  if (mean < 2 || squared / image.size() - mean * mean < 4) {
    fprintf(stderr, "blank or nearly uniform image\n");
    return 7;
  }
  FILE* f = fopen(argv[1], "wb");
  if (!f)
    return 3;
  fprintf(f, "P6\n%d %d\n255\n", p.width, p.height);
  for (auto c : image)
    fwrite(&c, 1, 3, f);
  fclose(f);
  printf("%s t=%.2f pointer=%.2f,%.2f down=%d q=%d ms min=%.3f median=%.3f p95=%.3f max=%.3f\n",
         argv[1],
         p.time,
         p.mouse_x,
         p.mouse_y,
         p.mouse_down,
         p.quality,
         times[0],
         times[15],
         times[28],
         times[29]);
  float2* samples;
  int count = 48000 * 48;
  CHECK(cudaMalloc(&samples, count * sizeof(float2)));
  audio<<<(count + 255) / 256, 256>>>(samples, nullptr, 0, p, 0, count, 48000, nullptr, 0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<float2> sound(count);
  CHECK(cudaMemcpy(sound.data(), samples, count * sizeof(float2), cudaMemcpyDeviceToHost));
  double energy = 0;
  float peak = 0, jump = 0;
  for (int i = 0; i < count; i++) {
    auto s = sound[i];
    if (!std::isfinite(s.x) || !std::isfinite(s.y))
      return 4;
    peak = fmaxf(peak, fmaxf(fabsf(s.x), fabsf(s.y)));
    energy += s.x * s.x + s.y * s.y;
    if (i)
      jump = fmaxf(jump, fmaxf(fabsf(s.x - sound[i - 1].x), fabsf(s.y - sound[i - 1].y)));
  }
  printf("audio peak=%.4f rms=%.4f max-adjacent-delta=%.4f loop-delta=%.4f\n",
         peak,
         sqrt(energy / (count * 2)),
         jump,
         fabsf(sound.back().x - sound.front().x));
  // A separately launched audio block at an arbitrary seek must match continuous output.
  unsigned long long offset = 576003;
  audio<<<4, 256>>>(samples, nullptr, 0, p, offset, 1021, 48000, nullptr, 0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  std::vector<float2> seek(1021);
  CHECK(cudaMemcpy(seek.data(), samples, seek.size() * sizeof(float2), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 1021; i++)
    if (seek[i].x != sound[offset + i].x || seek[i].y != sound[offset + i].y)
      return 5;
  CHECK(cudaFree(samples));
  CHECK(cudaFree(out));
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(b));
  return peak > 0.9f || energy < 1 ? 6 : 0;
}
