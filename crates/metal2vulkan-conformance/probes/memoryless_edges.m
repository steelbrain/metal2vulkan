// Edges of plain Metal's failure at a memoryless texture's use: zero-size copies, zero-threadgroup
// dispatches, a texture buffer's blits, and a residency call alone.
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment,
// one case a process.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o memoryless_edges \
//     memoryless_edges.m -framework Foundation -framework Metal
//
// Cases: blit-zero, blit-tb, blit-tb-shared, cs-zero, cs-tb, cs-tb-shared, cs-resource,
//  cs-resource-nodispatch, cs-read, cs-unused, cs-tg0, cs-barrier, cs-resource-shared.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
static void P(const char *fmt, ...) { va_list ap; va_start(ap,fmt);
  fputs("RESULT ",stdout); vfprintf(stdout,fmt,ap); fputc('\n',stdout); fflush(stdout); va_end(ap); }
static const char *SRC =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void kmark(device uint* m [[buffer(0)]], uint t [[thread_position_in_grid]]){ if(t==0) m[0]=0x77u; }\n"
"kernel void kread(texture2d<float> tx [[texture(0)]], device uint* m [[buffer(0)]], uint t [[thread_position_in_grid]]){ if(t==0) m[1]=as_type<uint>(tx.read(uint2(0,0)).x)|1u; }\n"
"kernel void ktb(texture_buffer<float> tx [[texture(0)]], device uint* m [[buffer(0)]], uint t [[thread_position_in_grid]]){ if(t==0) m[1]=as_type<uint>(tx.read(0u).x)|1u; }\n";
static const char *STATUS[]={"NotEnqueued","Enqueued","Committed","Scheduled","Completed","Error"};
int main(int argc,char**argv){ if(argc<2) return 2; const char *c=argv[1];
 @autoreleasepool {
 #define HAS(s) (strstr(c,s)!=NULL)
 P("case=%s",c);
 id<MTLDevice> D=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> Q=[D newCommandQueue];
 NSError *e=nil; id<MTLLibrary> L=[D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
 id<MTLBuffer> m=[D newBufferWithLength:256 options:MTLResourceStorageModeShared]; memset(m.contents,0,256);
 id<MTLBuffer> buf=[D newBufferWithLength:4096 options:MTLResourceStorageModeShared]; memset(buf.contents,0xC3,4096);
 MTLStorageMode sm = HAS("shared") ? MTLStorageModeShared : MTLStorageModeMemoryless;
 MTLTextureDescriptor *x=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
 x.storageMode=sm; x.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
 id<MTLTexture> t2=[D newTextureWithDescriptor:x];
 MTLTextureDescriptor *xb=[MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 resourceOptions:(sm<<MTLResourceStorageModeShift) usage:MTLTextureUsageShaderRead];
 id<MTLTexture> tb=[D newTextureWithDescriptor:xb];
 P("t2=%d tb=%d",t2?1:0,tb?1:0);
 id<MTLCommandBuffer> cb=[Q commandBuffer];
 if(HAS("blit-zero")){ id<MTLBlitCommandEncoder> be=[cb blitCommandEncoder]; [be fillBuffer:m range:NSMakeRange(0,4) value:0x11];
   [be copyFromTexture:t2 sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(0,0,0) toBuffer:buf destinationOffset:0 destinationBytesPerRow:64 destinationBytesPerImage:1024]; [be endEncoding]; }
 if(HAS("blit-tb")){ id<MTLBlitCommandEncoder> be=[cb blitCommandEncoder]; [be fillBuffer:m range:NSMakeRange(0,4) value:0x11];
   @try { [be copyFromTexture:tb sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(64,1,1) toBuffer:buf destinationOffset:0 destinationBytesPerRow:256 destinationBytesPerImage:256]; P("tb copy returned"); } @catch(NSException *ex){ P("tb copy raised %s",[[ex reason] UTF8String]); }
   [be endEncoding]; }
 if(HAS("cs-")){
   id<MTLComputePipelineState> ps=[D newComputePipelineStateWithFunction:[L newFunctionWithName:(HAS("tb")?@"ktb":((HAS("resource")||HAS("unused")||HAS("barrier"))?@"kmark":@"kread"))] error:&e];
   id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder]; [ce setComputePipelineState:ps]; [ce setBuffer:m offset:0 atIndex:0];
   if(HAS("resource")) [ce useResource:t2 usage:MTLResourceUsageRead];
   else if(HAS("barrier")) { [ce dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [ce memoryBarrierWithResources:&t2 count:1]; }
   else [ce setTexture:(HAS("tb")?tb:t2) atIndex:0];
   if(HAS("nodispatch")) {}
   else if(HAS("zero")) [ce dispatchThreadgroups:MTLSizeMake(0,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
   else if(HAS("tg0")) [ce dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(0,1,1)];
   else [ce dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
   [ce endEncoding]; }
 dispatch_semaphore_t s=dispatch_semaphore_create(0);
 [cb addCompletedHandler:^(id<MTLCommandBuffer> cc){ (void)cc; dispatch_semaphore_signal(s); }];
 [cb commit]; dispatch_semaphore_wait(s,dispatch_time(DISPATCH_TIME_NOW,(int64_t)(10*NSEC_PER_SEC)));
 uint32_t *w=(uint32_t*)m.contents;
 P("status=%s code=%ld marks=%x,%x buf0=%02x",((unsigned)cb.status<6)?STATUS[cb.status]:"?",(long)cb.error.code,w[0],w[1],((unsigned char*)buf.contents)[0]);
 }
 return 0; }
