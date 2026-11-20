// What plain Metal does when a memoryless texture is bound at a fragment (or compute) texture slot.
// One case per process; every case graded at the commit (status, error, counts the stages wrote).
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment,
// one case a process.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o memoryless_bind \
//     memoryless_bind.m -framework Foundation -framework Metal
//
// Cases: fs-read, fs-cond, fs-unread, fs-read-empty, fs-read-rebind, fs-read-nodraw,
//  fs-read-earlier, fs-read-prior-after, fs-read-single, fs-unread-vs, fs-unread-resource,
//  fs-unread-resource-nodraw, cs-read; "-private" for a Private texture's answer beside each.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
static void P(const char *fmt, ...) { va_list ap; va_start(ap,fmt);
  fputs("RESULT ",stdout); vfprintf(stdout,fmt,ap); fputc('\n',stdout); fflush(stdout); va_end(ap); }
static id<MTLDevice> D; static id<MTLCommandQueue> Q; static id<MTLLibrary> L;
static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct VOut { float4 pos [[position]]; };\n"
"vertex VOut vfull(uint vid [[vertex_id]], device atomic_uint* rep [[buffer(1)]]){\n"
"  float2 p[3]={float2(-1.0,-1.0),float2(3.0,-1.0),float2(-1.0,3.0)};\n"
"  atomic_fetch_add_explicit(&rep[1],1u,memory_order_relaxed);\n"
"  VOut o; o.pos=float4(p[vid],0.5,1.0); return o; }\n"
"fragment float4 fread(VOut in [[stage_in]], texture2d<float> t [[texture(0)]], device atomic_uint* rep [[buffer(1)]]){\n"
"  atomic_fetch_add_explicit(&rep[2],1u,memory_order_relaxed);\n"
"  return t.read(uint2(in.pos.xy)) + float4(0.0,0.0,0.0,0.0); }\n"
"fragment float4 fcond(VOut in [[stage_in]], texture2d<float> t [[texture(0)]], device atomic_uint* rep [[buffer(1)]], constant uint& flag [[buffer(2)]]){\n"
"  atomic_fetch_add_explicit(&rep[2],1u,memory_order_relaxed);\n"
"  if(flag!=0) return t.read(uint2(in.pos.xy)); return float4(0.25,0.5,0.75,1.0); }\n"
"fragment float4 fplain(VOut in [[stage_in]], device atomic_uint* rep [[buffer(1)]]){\n"
"  atomic_fetch_add_explicit(&rep[2],1u,memory_order_relaxed); return float4(0.25,0.5,0.75,1.0); }\n"
"kernel void kread(texture2d<float> t [[texture(0)]], device atomic_uint* rep [[buffer(1)]], device float4* out [[buffer(2)]], uint2 g [[thread_position_in_grid]]){\n"
"  atomic_fetch_add_explicit(&rep[3],1u,memory_order_relaxed); if(g.x==0&&g.y==0) out[0]=t.read(g); }\n";
static const char *STATUS[]={"NotEnqueued","Enqueued","Committed","Scheduled","Completed","Error"};
static int finish(id<MTLCommandBuffer> cb, const char *tag, double secs){
  dispatch_semaphore_t s=dispatch_semaphore_create(0);
  [cb addCompletedHandler:^(id<MTLCommandBuffer> c){ (void)c; dispatch_semaphore_signal(s); }];
  [cb commit]; P("%s: commit returned", tag);
  long r=dispatch_semaphore_wait(s,dispatch_time(DISPATCH_TIME_NOW,(int64_t)(secs*NSEC_PER_SEC)));
  if(r!=0){ P("%s: TIMEOUT status=%s",tag,((unsigned)cb.status<6)?STATUS[cb.status]:"?"); return 1; }
  NSError *err=cb.error;
  P("%s: status=%s error=%s",tag,((unsigned)cb.status<6)?STATUS[cb.status]:"?",
    err?[[NSString stringWithFormat:@"domain=%@ code=%ld desc=%@",err.domain,(long)err.code,err.localizedDescription] UTF8String]:"none");
  return 0; }
static id<MTLFunction> fn(const char*n){ id<MTLFunction> f=[L newFunctionWithName:[NSString stringWithUTF8String:n]];
  if(!f){P("fatal=no-function %s",n);exit(3);} return f; }
int main(int argc,char**argv){ if(argc<2){fprintf(stderr,"usage: probe <case>\n");return 2;}
 const char *c=argv[1];
 @autoreleasepool {
 #define HAS(s) (strstr(c,s)!=NULL)
 P("case=%s MTL_DEBUG_LAYER=%s",c,getenv("MTL_DEBUG_LAYER")?:"unset");
 D=MTLCreateSystemDefaultDevice(); Q=[D newCommandQueue];
 P("device=%s",[[D name] UTF8String]);
 NSError *e=nil; L=[D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
 if(!L){P("fatal=library %s",[[e description] UTF8String]);return 3;}
 id<MTLBuffer> rep=[D newBufferWithLength:64 options:MTLResourceStorageModeShared]; memset(rep.contents,0,64);
 id<MTLBuffer> out=[D newBufferWithLength:64 options:MTLResourceStorageModeShared]; memset(out.contents,0xAB,64);
 // The bound texture: memoryless unless "-private".
 MTLTextureDescriptor *x=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
 x.storageMode = HAS("-private") ? MTLStorageModePrivate : MTLStorageModeMemoryless;
 x.usage = HAS("-rtonly") ? MTLTextureUsageRenderTarget : (MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead);
 id<MTLTexture> bound=[D newTextureWithDescriptor:x];
 P("bound: made=%d storage=%lu usage=%lu",bound?1:0,(unsigned long)bound.storageMode,(unsigned long)bound.usage);
 if(!bound) return 4;
 if(HAS("cs-")){
   id<MTLComputePipelineState> ps=[D newComputePipelineStateWithFunction:fn("kread") error:&e];
   id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
   [ce setComputePipelineState:ps]; [ce setBuffer:rep offset:0 atIndex:1]; [ce setBuffer:out offset:0 atIndex:2];
   id<MTLTexture> ts[1]={bound}; [ce setTextures:ts withRange:NSMakeRange(0,1)]; P("setTextures:withRange: returned");
   [ce dispatchThreads:MTLSizeMake(16,16,1) threadsPerThreadgroup:MTLSizeMake(8,8,1)];
   [ce endEncoding]; P("endEncoding returned"); finish(cb,"cs",10.0);
   unsigned *r=(unsigned*)rep.contents; float *o=(float*)out.contents;
   P("cs: threads=%u out0=%g,%g,%g,%g",r[3],o[0],o[1],o[2],o[3]);
 } else {
   MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
   td.storageMode=MTLStorageModeShared; td.usage=MTLTextureUsageRenderTarget;
   id<MTLTexture> target=[D newTextureWithDescriptor:td];
   MTLRenderPipelineDescriptor *pd=[[MTLRenderPipelineDescriptor alloc] init];
   pd.vertexFunction=fn("vfull"); pd.fragmentFunction=fn(HAS("-unread")?"fplain":(HAS("-cond")?"fcond":"fread"));
   pd.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
   id<MTLRenderPipelineState> ps=[D newRenderPipelineStateWithDescriptor:pd error:&e];
   if(!ps){P("fatal=pipeline %s",[[e description] UTF8String]);return 3;}
   MTLRenderPassDescriptor *rp=[MTLRenderPassDescriptor renderPassDescriptor];
   rp.colorAttachments[0].texture=target; rp.colorAttachments[0].loadAction=MTLLoadActionClear;
   rp.colorAttachments[0].clearColor=MTLClearColorMake(0.0,1.0,0.0,1.0); rp.colorAttachments[0].storeAction=MTLStoreActionStore;
   id<MTLCommandBuffer> cb=[Q commandBuffer];
   if(HAS("-prior")){ id<MTLBlitCommandEncoder> b0=[cb blitCommandEncoder]; [b0 fillBuffer:out range:NSMakeRange(0,4) value:0x11]; [b0 endEncoding]; P("prior blit encoded"); }
   id<MTLRenderCommandEncoder> re=[cb renderCommandEncoderWithDescriptor:rp];
   [re setRenderPipelineState:ps]; [re setVertexBuffer:rep offset:0 atIndex:1]; [re setFragmentBuffer:rep offset:0 atIndex:1]; if(HAS("-vs")){ id<MTLTexture> vt[1]={bound}; [re setVertexTextures:vt withRange:NSMakeRange(0,1)]; P("setVertexTextures returned"); } { uint32_t fl=0; [re setFragmentBytes:&fl length:4 atIndex:2]; }
   if(HAS("-resource")){ [re useResource:bound usage:MTLResourceUsageRead stages:MTLRenderStageFragment]; P("useResource returned"); }
   else if(HAS("-single")){ [re setFragmentTexture:bound atIndex:0]; P("setFragmentTexture:atIndex: returned"); }
   else { id<MTLTexture> ts[1]={bound}; [re setFragmentTextures:ts withRange:NSMakeRange(0,1)]; P("setFragmentTextures:withRange: returned"); }
   if(HAS("-rebind")){ [re setFragmentTexture:nil atIndex:0]; P("rebound nil"); }
   if(HAS("-earlier")){ id<MTLRenderPipelineState> ps2; MTLRenderPipelineDescriptor *p2=[pd copy]; p2.fragmentFunction=fn("fplain"); ps2=[D newRenderPipelineStateWithDescriptor:p2 error:&e]; [re setRenderPipelineState:ps2]; [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [re setRenderPipelineState:ps]; P("earlier plain draw encoded"); }
   if(!HAS("-nodraw")){ [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:(HAS("-empty")?0:3)]; P("draw returned"); }
   [re endEncoding]; P("endEncoding returned");
   if(HAS("-after")){ id<MTLBlitCommandEncoder> b1=[cb blitCommandEncoder]; [b1 fillBuffer:out range:NSMakeRange(4,4) value:0x22]; [b1 endEncoding]; P("after blit encoded"); }
   finish(cb,"fs",10.0);
   P("fs: out bytes0=%02x bytes4=%02x",((unsigned char*)out.contents)[0],((unsigned char*)out.contents)[4]);
   unsigned *r=(unsigned*)rep.contents; P("fs: vertices=%u fragments=%u",r[1],r[2]);
   unsigned char px[16*16*4]; [target getBytes:px bytesPerRow:64 fromRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0];
   int same=1; for(int i=4;i<16*16*4;i++) if(px[i]!=px[i%4]) same=0;
   P("fs: target texel0=%02x%02x%02x%02x uniform=%d",px[0],px[1],px[2],px[3],same);
 }
 // Canary: does the queue still work?
 id<MTLCommandBuffer> cb2=[Q commandBuffer]; id<MTLBlitCommandEncoder> be=[cb2 blitCommandEncoder];
 [be fillBuffer:out range:NSMakeRange(32,32) value:7]; [be endEncoding]; finish(cb2,"canary",10.0);
 P("canary: byte32=%u",((unsigned char*)out.contents)[32]);
 }
 return 0; }
