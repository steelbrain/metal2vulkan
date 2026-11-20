// Which of a command buffer's encoders run when one uses a memoryless texture (plain Metal): a
// sequence of encoders, each marking a shared buffer, one of them failing; event signals between.
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment,
// one case a process.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o memoryless_split \
//     memoryless_split.m -framework Foundation -framework Metal
//
// Cases: a sequence of B (blit), C (compute), R (render pass), S (event signal), b and c (a blit
//  and a compute encoder that use the texture): CcC, BbB, CbC, BcB, BSbB, bSB, SbB, BRbB, BbRB,
//  CSc, cSC, BB.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>
static void P(const char *fmt, ...) { va_list ap; va_start(ap,fmt);
  fputs("RESULT ",stdout); vfprintf(stdout,fmt,ap); fputc('\n',stdout); fflush(stdout); va_end(ap); }
static const char *SRC =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void kmark(device uint* m [[buffer(0)]], constant uint& at [[buffer(1)]], uint t [[thread_position_in_grid]]){ if(t==0) m[at]=0x77u; }\n"
"kernel void kread(texture2d<float> tx [[texture(0)]], device uint* m [[buffer(0)]], constant uint& at [[buffer(1)]], uint t [[thread_position_in_grid]]){ if(t==0) m[at]=as_type<uint>(tx.read(uint2(0,0)).x)|1u; }\n";
static const char *STATUS[]={"NotEnqueued","Enqueued","Committed","Scheduled","Completed","Error"};
int main(int argc,char**argv){ if(argc<2) return 2; const char *seq=argv[1];
 @autoreleasepool {
 P("seq=%s",seq);
 id<MTLDevice> D=MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> Q=[D newCommandQueue];
 NSError *e=nil; id<MTLLibrary> L=[D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
 id<MTLComputePipelineState> mark=[D newComputePipelineStateWithFunction:[L newFunctionWithName:@"kmark"] error:&e];
 id<MTLComputePipelineState> rd=[D newComputePipelineStateWithFunction:[L newFunctionWithName:@"kread"] error:&e];
 id<MTLBuffer> m=[D newBufferWithLength:256 options:MTLResourceStorageModeShared]; memset(m.contents,0,256);
 id<MTLBuffer> buf=[D newBufferWithLength:4096 options:MTLResourceStorageModeShared];
 MTLTextureDescriptor *x=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
 x.storageMode=MTLStorageModeMemoryless; x.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
 id<MTLTexture> ml=[D newTextureWithDescriptor:x];
 id<MTLSharedEvent> ev=[D newSharedEvent]; ev.signaledValue=0;
 id<MTLCommandBuffer> cb=[Q commandBuffer];
 // seq letters: B blit mark, b blit failing (copy from memoryless), C compute mark, c compute failing (reads memoryless), R render pass (clear, no draw), S event signal (value increments)
 uint32_t slot=0; uint64_t sig=0;
 for(const char *p=seq; *p; p++){
   switch(*p){
     case 'B': case 'b': { id<MTLBlitCommandEncoder> be=[cb blitCommandEncoder];
       [be fillBuffer:m range:NSMakeRange(slot*4,4) value:0x11];
       if(*p=='b') [be copyFromTexture:ml sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(1,1,1) toBuffer:buf destinationOffset:0 destinationBytesPerRow:4 destinationBytesPerImage:4];
       [be endEncoding]; slot++; break; }
     case 'C': case 'c': { id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
       [ce setComputePipelineState:mark]; [ce setBuffer:m offset:0 atIndex:0]; uint32_t at=slot; [ce setBytes:&at length:4 atIndex:1];
       [ce dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
       if(*p=='c'){ [ce setComputePipelineState:rd]; [ce setTexture:ml atIndex:0]; uint32_t at2=63; [ce setBytes:&at2 length:4 atIndex:1]; [ce dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; }
       [ce endEncoding]; slot++; break; }
     case 'R': { MTLTextureDescriptor *td=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Uint width:1 height:1 mipmapped:NO];
       td.storageMode=MTLStorageModeShared; td.usage=MTLTextureUsageRenderTarget; id<MTLTexture> t=[D newTextureWithDescriptor:td];
       MTLRenderPassDescriptor *rp=[MTLRenderPassDescriptor renderPassDescriptor]; rp.colorAttachments[0].texture=t;
       rp.colorAttachments[0].loadAction=MTLLoadActionClear; rp.colorAttachments[0].clearColor=MTLClearColorMake(0x55,0,0,0); rp.colorAttachments[0].storeAction=MTLStoreActionStore;
       id<MTLRenderCommandEncoder> re=[cb renderCommandEncoderWithDescriptor:rp]; [re endEncoding];
       id<MTLBlitCommandEncoder> be=nil; (void)be;
       // The render target's texel is read after the commit.
       break; }
     case 'S': { sig++; [cb encodeSignalEvent:ev value:sig]; break; }
   }
 }
 dispatch_semaphore_t s=dispatch_semaphore_create(0);
 [cb addCompletedHandler:^(id<MTLCommandBuffer> c){ (void)c; dispatch_semaphore_signal(s); }];
 [cb commit];
 dispatch_semaphore_wait(s,dispatch_time(DISPATCH_TIME_NOW,(int64_t)(10*NSEC_PER_SEC)));
 P("status=%s code=%ld",((unsigned)cb.status<6)?STATUS[cb.status]:"?",(long)cb.error.code);
 uint32_t *w=(uint32_t*)m.contents; char line[512]; int n=0;
 for(uint32_t i=0;i<slot;i++) n+=snprintf(line+n,sizeof line-n,"%s%x",i?",":"",w[i]);
 P("marks=%s event=%llu (encoded up to %llu)",line,(unsigned long long)ev.signaledValue,(unsigned long long)sig);
 }
 return 0; }
