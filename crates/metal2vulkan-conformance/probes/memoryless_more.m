// What plain Metal does with a memoryless texture in the commands the other probes left out: a
// one-level mipmap generation, a CPU-access hint, a copy of no slices, and a tile dispatch whose
// tile function statically reads it (or does not). One case per process, graded at the commit.
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment,
// one case a process.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o memoryless_more \
//     memoryless_more.m -framework Foundation -framework Metal
//
// Cases: mips1, optcpu, slices0, sync, tile, tile-unused, tile-private, mips1-private.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
static void P(const char *fmt, ...) { va_list ap; va_start(ap,fmt);
  fputs("RESULT ",stdout); vfprintf(stdout,fmt,ap); fputc('\n',stdout); fflush(stdout); va_end(ap); }
static id<MTLDevice> D; static id<MTLCommandQueue> Q;
static const char *STATUS[]={"NotEnqueued","Enqueued","Committed","Scheduled","Completed","Error"};
static void finish(id<MTLCommandBuffer> cb, const char *tag){
  [cb commit]; [cb waitUntilCompleted]; NSError *err=cb.error;
  P("%s: status=%s code=%ld",tag,((unsigned)cb.status<6)?STATUS[cb.status]:"?",err?(long)err.code:0L); }
static const char *SRC =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct VOut { float4 pos [[position]]; };\n"
"vertex VOut vfull(uint vid [[vertex_id]]){ float2 p[3]={float2(-1.0,-1.0),float2(3.0,-1.0),float2(-1.0,3.0)};\n"
"  VOut o; o.pos=float4(p[vid],0.5,1.0); return o; }\n"
"fragment float4 fplain(VOut in [[stage_in]]){ return float4(0.25,0.5,0.75,1.0); }\n"
"kernel void tread(texture2d<float> t [[texture(0)]], device uint* out [[buffer(0)]], ushort2 l [[thread_position_in_threadgroup]]){\n"
"  if(l.x==0&&l.y==0) out[0]=uint(t.read(uint2(0)).x*255.0)+1u; }\n"
"kernel void tplain(device uint* out [[buffer(0)]], ushort2 l [[thread_position_in_threadgroup]]){ if(l.x==0&&l.y==0) out[0]=77u; }\n";
int main(int argc,char**argv){ if(argc<2) return 2; const char *c=argv[1];
 @autoreleasepool {
 #define HAS(s) (strstr(c,s)!=NULL)
 P("case=%s",c); D=MTLCreateSystemDefaultDevice(); Q=[D newCommandQueue];
 MTLStorageMode sm = HAS("-private") ? MTLStorageModePrivate : MTLStorageModeMemoryless;
 MTLTextureDescriptor *x=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
 x.storageMode=sm; x.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
 id<MTLTexture> m=[D newTextureWithDescriptor:x];
 x.storageMode=MTLStorageModePrivate; id<MTLTexture> o=[D newTextureWithDescriptor:x];
 id<MTLBuffer> marks=[D newBufferWithLength:64 options:MTLResourceStorageModeShared]; memset(marks.contents,0,64);
 P("made=%d levels=%lu",m?1:0,(unsigned long)m.mipmapLevelCount);
 id<MTLCommandBuffer> cb=[Q commandBuffer];
 if(HAS("tile")){
   NSError *e=nil; id<MTLLibrary> L=[D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
   if(!L){P("fatal=library");return 3;}
   MTLRenderPipelineDescriptor *pd=[[MTLRenderPipelineDescriptor alloc] init];
   pd.vertexFunction=[L newFunctionWithName:@"vfull"]; pd.fragmentFunction=[L newFunctionWithName:@"fplain"];
   pd.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm;
   id<MTLRenderPipelineState> ps=[D newRenderPipelineStateWithDescriptor:pd error:&e];
   MTLTileRenderPipelineDescriptor *td=[[MTLTileRenderPipelineDescriptor alloc] init];
   td.tileFunction=[L newFunctionWithName:(HAS("-unused")?@"tplain":@"tread")];
   td.colorAttachments[0].pixelFormat=MTLPixelFormatRGBA8Unorm; td.threadgroupSizeMatchesTileSize=YES;
   id<MTLRenderPipelineState> tp=[D newRenderPipelineStateWithTileDescriptor:td options:0 reflection:nil error:&e];
   if(!ps||!tp){P("fatal=pipeline %s",e?[[e description] UTF8String]:"");return 3;}
   x.storageMode=MTLStorageModeShared; x.usage=MTLTextureUsageRenderTarget; id<MTLTexture> target=[D newTextureWithDescriptor:x];
   MTLRenderPassDescriptor *rp=[MTLRenderPassDescriptor renderPassDescriptor];
   rp.colorAttachments[0].texture=target; rp.colorAttachments[0].loadAction=MTLLoadActionClear;
   rp.colorAttachments[0].storeAction=MTLStoreActionStore; rp.tileWidth=16; rp.tileHeight=16;
   id<MTLRenderCommandEncoder> re=[cb renderCommandEncoderWithDescriptor:rp];
   [re setRenderPipelineState:ps]; [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
   [re setRenderPipelineState:tp]; [re setTileBuffer:marks offset:0 atIndex:0]; [re setTileTexture:m atIndex:0];
   [re dispatchThreadsPerTile:MTLSizeMake(16,16,1)]; [re endEncoding];
 } else {
   id<MTLBlitCommandEncoder> be=[cb blitCommandEncoder];
   [be fillBuffer:marks range:NSMakeRange(4,4) value:0x22];
   if(HAS("mips1")) [be generateMipmapsForTexture:m];
   else if(HAS("optcpu")) [be optimizeContentsForCPUAccess:m];
   else if(HAS("slices0")) [be copyFromTexture:m sourceSlice:0 sourceLevel:0 toTexture:o destinationSlice:0 destinationLevel:0 sliceCount:0 levelCount:1];
   else if(HAS("sync")) [be synchronizeResource:m];
   [be fillBuffer:marks range:NSMakeRange(8,4) value:0x33];
   [be endEncoding];
 }
 finish(cb,"cb");
 unsigned *k=(unsigned*)marks.contents; P("marks out0=%u fill4=%08x fill8=%08x",k[0],k[1],k[2]);
 }
 return 0; }
