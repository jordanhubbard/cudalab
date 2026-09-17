#include <cudalab.cuh>
#include <mma.h>

using namespace nvcuda;
constexpr int N = 1024, TILES = N / 16;

CUDALAB_SIMULATE {
  int warp = threadIdx.x / 32, lane = threadIdx.x & 31, tile = blockIdx.x * 8 + warp;
  if (tile >= TILES * TILES || state_bytes < N * N * sizeof(float))
    return;
  __shared__ half sa[8][256];
  __shared__ half sb[8][256];
  int tx = tile % TILES, ty = tile / TILES;
  float time = params.time * .35f, fold = 2.0f + params.mouse_x * 10.0f;
  for (int i = lane; i < 256; i += 32) {
    int r = i / 16, c = i % 16;
    float x = (tx * 16 + r - N * .5f) / N, y = (ty * 16 + c - N * .5f) / N;
    sa[warp][i] = __float2half(sinf((x * fold + y * 3) * 12 + time + r * .07f));
    sb[warp][i] = __float2half(cosf((y * fold - x * 4) * 10 - time + c * .09f));
  }
  __syncwarp();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::load_matrix_sync(a, sa[warp], 16);
  wmma::load_matrix_sync(b, sb[warp], 16);
  wmma::fill_fragment(c, 0.0f);
  wmma::mma_sync(c, a, b, c);
  auto* out = static_cast<float*>(state);
  wmma::store_matrix_sync(out + ty * 16 * N + tx * 16, c, N, wmma::mem_row_major);
}

CUDALAB_RENDER {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= params.width || y >= params.height || !state)
    return;
  auto* field = static_cast<float*>(state);
  int gx = x * N / params.width, gy = y * N / params.height;
  float z = field[gy * N + gx] / 16.0f;
  float thread = .5f + .5f * sinf(z * 5.5f + atan2f(gy - N * .5f, gx - N * .5f) * 3);
  float shine = powf(fabsf(cosf(z * 2.2f)), 10);
  float3 c = make_float3(.025f + .75f * thread + .65f * shine,
                         .018f + .13f * thread + .35f * shine,
                         .055f + .62f * (1 - thread) + .9f * shine);
  c = cudalab_tonemap(c);
  pixels[y * params.width + x] = make_uchar4(255 * powf(cudalab_saturate(c.x), .4545f),
                                             255 * powf(cudalab_saturate(c.y), .4545f),
                                             255 * powf(cudalab_saturate(c.z), .4545f),
                                             255);
}
