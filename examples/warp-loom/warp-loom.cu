#include <cudalab.cuh>

CUDALAB_RENDER {
  int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=params.width||y>=params.height)return;
  unsigned mask=__activemask();int lane=threadIdx.x&31;float2 uv=make_float2((2.0f*x-params.width)/params.height,(2.0f*y-params.height)/params.height);
  float note=sinf(uv.x*(18+params.mouse_x*52)+params.time*1.4f)+cosf(uv.y*(22+params.mouse_y*44)-params.time);
  float opposite=__shfl_xor_sync(mask,note,16),quarter=__shfl_xor_sync(mask,note,8),neighbor=__shfl_xor_sync(mask,note,1);
  unsigned constellation=__ballot_sync(mask,note+opposite>0);float votes=__popc(constellation)/32.0f;
  float thread=.5f+.5f*sinf((note-opposite)*5+(quarter-neighbor)*3+lane*.11f);
  float crossing=powf(fabsf(sinf(uv.x*90+opposite*2)*cosf(uv.y*70+quarter*2)),7);
  float3 c=make_float3(.02f+.65f*thread+.85f*crossing,.025f+.28f*(1-thread)+.4f*votes,.07f+.7f*votes+.3f*crossing);
  c=cudalab_tonemap(c);pixels[y*params.width+x]=make_uchar4(255*powf(cudalab_saturate(c.x),.4545f),255*powf(cudalab_saturate(c.y),.4545f),255*powf(cudalab_saturate(c.z),.4545f),255);
}
