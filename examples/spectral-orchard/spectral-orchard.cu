#include <cudalab.cuh>

__device__ float note_hz(int midi){return 440.0f*exp2f((midi-69)/12.0f);}
__device__ float tone(float phase,float brightness){
  return sinf(phase)+sinf(phase*2.01f)*.32f*brightness+sinf(phase*3.99f)*.13f*brightness+sinf(phase*7.02f)*.045f;
}

CUDALAB_AUDIO {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=sample_count)return;unsigned long long n=sample_offset+i;float t=n/(float)sample_rate;
  int progression[4]={45,41,48,43};int root=progression[((int)(t/4))&3];float local=fmodf(t,4.0f),mix=0,left=0,right=0;
  int intervals[6]={0,7,12,16,19,24};
  #pragma unroll
  for(int v=0;v<6;v++){
    float start=floorf(t*2-v*.37f)*.5f+v*.37f,age=t-start;if(age<0)age+=.5f;float env=expf(-age*(2.2f+v*.18f));
    float freq=note_hz(root+intervals[v]+(params.mouse_y>.66f?2:0));float phase=6.283185f*freq*t;
    float voice=tone(phase,.8f)*env*(.12f/(1+v*.18f));float pan=.5f+.42f*sinf(v*2.1f+local*.22f+params.mouse_x*2);
    left+=voice*sqrtf(1-pan);right+=voice*sqrtf(pan);mix+=voice;
  }
  float bass=tone(6.283185f*note_hz(root-12)*t,.2f)*(.10f+.04f*sinf(local*1.5708f));
  float breath=sinf(t*6.283185f*.125f)*.018f*sinf(t*6.283185f*note_hz(root+31));
  samples[i]=make_float2(tanhf(left+bass+breath),tanhf(right+bass-breath));
}

__device__ float sd_segment(float2 p,float2 a,float2 b){float2 pa=make_float2(p.x-a.x,p.y-a.y),ba=make_float2(b.x-a.x,b.y-a.y);float h=cudalab_saturate((pa.x*ba.x+pa.y*ba.y)/(ba.x*ba.x+ba.y*ba.y));return hypotf(pa.x-ba.x*h,pa.y-ba.y*h);}

CUDALAB_RENDER {
  int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=params.width||y>=params.height)return;
  float2 uv=make_float2((2.0f*x-params.width)/params.height,(params.height-2.0f*y)/params.height);float beat=fmodf(params.time*2,1.0f),pulse=expf(-beat*5);
  float3 color=make_float3(.006f,.009f+.018f*(uv.y+.7f),.028f+.05f*(uv.y+.7f));
  float aur=.5f+.5f*sinf(uv.x*5+sinf(uv.y*7+params.time*.2f)*2);color.y+=powf(aur,8)*.035f;color.z+=powf(aur,8)*.08f;
  for(int tree=0;tree<7;tree++){
    float rootx=(tree-3)*.34f;float sway=sinf(params.time*.45f+tree)*.035f*(.3f+params.mouse_x);
    float trunk=sd_segment(uv,make_float2(rootx,-.78f),make_float2(rootx+sway,.10f));float wood=expf(-trunk*150);color.x+=wood*.12f;color.y+=wood*.075f;color.z+=wood*.035f;
    for(int branch=0;branch<5;branch++){
      float h=-.38f+branch*.15f,side=(branch&1)?1:-1;float2 a=make_float2(rootx+sway*(h+.78f)/.88f,h),b=make_float2(a.x+side*(.16f+.025f*branch)+sway,a.y+.20f);
      float twig=expf(-sd_segment(uv,a,b)*190);color.x+=twig*.16f;color.y+=twig*.10f;color.z+=twig*.055f;
      float fruit=hypotf(uv.x-b.x,uv.y-b.y);float glow=expf(-fruit*(32-8*pulse));float hue=.5f+.5f*sinf(tree*3.1f+branch*2.4f+params.time*.3f);
      color.x+=glow*(.8f+.9f*pulse);color.y+=glow*(.12f+.35f*hue);color.z+=glow*(.25f+.75f*(1-hue));
    }
  }
  float ground=expf(-fabsf(uv.y+.79f)*90);color.x+=ground*.08f;color.y+=ground*.16f;color.z+=ground*.22f;color=cudalab_tonemap(color);
  pixels[y*params.width+x]=make_uchar4(255*powf(cudalab_saturate(color.x),.4545f),255*powf(cudalab_saturate(color.y),.4545f),255*powf(cudalab_saturate(color.z),.4545f),255);
}
