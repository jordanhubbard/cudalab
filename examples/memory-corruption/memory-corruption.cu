#include <cudalab.cuh>
#include <cub/block/block_radix_sort.cuh>

constexpr int N=1024;
CUDALAB_SIMULATE {
  using Sort=cub::BlockRadixSort<float,256,1>;
  __shared__ typename Sort::TempStorage storage;
  int i=blockIdx.x*256+threadIdx.x;if(i>=N*N||state_bytes<N*N*sizeof(float))return;int x=i%N,y=i/N;float u=x/(float)N,v=y/(float)N;
  float sun=expf(-((u-.72f)*(u-.72f)+(v-.32f)*(v-.32f))*420);float ridge=.44f+.08f*sinf(u*13)+.035f*sinf(u*43+1);float landscape=v>ridge?0.04f:.12f+.42f*(1-v)+sun;
  float key=landscape+.08f*sinf(u*80+params.time)+.025f*sinf(v*310);float keys[1]={key};
  Sort(storage).SortDescending(keys);float sorted=keys[0];
  float band=fmodf(blockIdx.x*.618034f+params.time*.09f,1.0f);float threshold=.18f+.7f*params.mouse_x;
  static_cast<float*>(state)[i]=band<threshold?sorted:key;
}
CUDALAB_RENDER {
  int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=params.width||y>=params.height||!state)return;float z=static_cast<float*>(state)[(y*N/params.height)*N+x*N/params.width];float sky=cudalab_saturate((z-.02f)*2.2f);float hot=powf(cudalab_saturate(z),4);
  float3 c=make_float3(.025f+.85f*hot+.3f*sky,.018f+.24f*sky+.18f*hot,.055f+.52f*sky+.08f*hot);float scan=.93f+.07f*sinf(y*3.14159f);c.x*=scan;c.y*=scan;c.z*=scan;c=cudalab_tonemap(c);pixels[y*params.width+x]=make_uchar4(255*powf(cudalab_saturate(c.x),.4545f),255*powf(cudalab_saturate(c.y),.4545f),255*powf(cudalab_saturate(c.z),.4545f),255);
}
