#include <cudalab.cuh>

constexpr int N=512,CELLS=N*N;
__device__ float4 fetch(const float4* f,int x,int y){x=(x+N)%N;y=(y+N)%N;return f[y*N+x];}
__device__ float4 bilerp(const float4* f,float x,float y){int x0=floorf(x),y0=floorf(y);float ax=x-x0,ay=y-y0;float4 a=fetch(f,x0,y0),b=fetch(f,x0+1,y0),c=fetch(f,x0,y0+1),d=fetch(f,x0+1,y0+1);return make_float4((a.x*(1-ax)+b.x*ax)*(1-ay)+(c.x*(1-ax)+d.x*ax)*ay,(a.y*(1-ax)+b.y*ax)*(1-ay)+(c.y*(1-ax)+d.y*ax)*ay,(a.z*(1-ax)+b.z*ax)*(1-ay)+(c.z*(1-ax)+d.z*ax)*ay,(a.w*(1-ax)+b.w*ax)*(1-ay)+(c.w*(1-ax)+d.w*ax)*ay);}

CUDALAB_RESET {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=CELLS||state_bytes<sizeof(float4)*CELLS*2ull)return;int x=i%N,y=i/N;float nx=(x-N*.5f)/N,ny=(y-N*.5f)/N;
  float ring=expf(-fabsf(hypotf(nx,ny)-.22f)*90);float4 q=make_float4(-ny*.9f,nx*.9f,ring*.65f,ring*.15f);auto* f=static_cast<float4*>(state);f[i]=q;f[CELLS+i]=q;
}
CUDALAB_SIMULATE {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=CELLS||!state)return;auto* f=static_cast<float4*>(state);const float4* src=f+((params.frame&1)?CELLS:0);float4* dst=f+((params.frame&1)?0:CELLS);int x=i%N,y=i/N;float4 c=src[i];
  float4 adv=bilerp(src,x-c.x*1.35f,y-c.y*1.35f);float4 l=fetch(src,x-1,y),r=fetch(src,x+1,y),u=fetch(src,x,y-1),d=fetch(src,x,y+1);float curl=(r.y-l.y-u.x+d.x)*.5f;
  float angle=sinf(x*.017f+params.time*.21f)+cosf(y*.013f-params.time*.17f);adv.x=adv.x*.994f+sinf(angle)*.004f-curl*.002f;adv.y=adv.y*.994f+cosf(angle)*.004f+curl*.002f;adv.z*=.997f;adv.w*=.994f;
  float dx=x/N-params.mouse_x,dy=y/N-params.mouse_y,brush=expf(-(dx*dx+dy*dy)*4800);adv.x+=-dy*brush*3;adv.y+=dx*brush*3;adv.z=fminf(1.5f,adv.z+brush*.16f);adv.w=fminf(1.0f,adv.w+brush*.055f);
  dst[i]=adv;
}
CUDALAB_RENDER {
  int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=params.width||y>=params.height||!state)return;auto* f=static_cast<float4*>(state);const float4* current=f+((params.frame&1)?0:CELLS);float4 q=current[(y*N/params.height)*N+x*N/params.width];float speed=hypotf(q.x,q.y),ink=q.z,gold=q.w;
  float3 c=make_float3(.012f+.12f*ink+.9f*gold,.015f+.28f*ink+.42f*gold,.025f+.46f*ink+.06f*gold);c.z+=speed*.08f;c=cudalab_tonemap(c);pixels[y*params.width+x]=make_uchar4(255*powf(cudalab_saturate(c.x),.4545f),255*powf(cudalab_saturate(c.y),.4545f),255*powf(cudalab_saturate(c.z),.4545f),255);
}
