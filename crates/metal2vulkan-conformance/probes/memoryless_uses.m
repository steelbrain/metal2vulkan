// What plain Metal does with a memoryless texture's CPU access, views and blits (the API
// validation layer off unless the environment turns it on). One case per process; a blit case is
// graded at the commit, with a fill encoder before it, a fill in its own encoder before the
// command, and a fill encoder after it, so the run says which of them ran.
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment,
// one case a process.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o memoryless_uses \
//     memoryless_uses.m -framework Foundation -framework Metal
//
// Cases: cpu-get, cpu-put, cpu-putget, cpu-tb-get, cpu-tb-put, view-format, view-slices,
//  view-swizzle, view-same, view-levels-slices, view-tb-same, view-tb-format, blit-tobuf,
//  blit-frombuf, blit-texsrc, blit-texdst, blit-wholesrc, blit-mips, blit-optimize,
//  blit-tobuf-private, blit-mips-private.
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
  dispatch_semaphore_t s=dispatch_semaphore_create(0);
  [cb addCompletedHandler:^(id<MTLCommandBuffer> c){ (void)c; dispatch_semaphore_signal(s); }];
  [cb commit];
  long r=dispatch_semaphore_wait(s,dispatch_time(DISPATCH_TIME_NOW,(int64_t)(10*NSEC_PER_SEC)));
  if(r!=0){ P("%s: TIMEOUT",tag); return; }
  NSError *err=cb.error;
  P("%s: status=%s error=%s",tag,((unsigned)cb.status<6)?STATUS[cb.status]:"?",
    err?[[NSString stringWithFormat:@"code=%ld desc=%@",(long)err.code,err.localizedDescription] UTF8String]:"none"); }
static id<MTLTexture> tex(MTLStorageMode sm, NSUInteger levels, MTLTextureUsage usage){
  MTLTextureDescriptor *x=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:16 mipmapped:NO];
  x.mipmapLevelCount=levels; x.storageMode=sm; x.usage=usage;
  return [D newTextureWithDescriptor:x]; }
static id<MTLTexture> tbuf(MTLStorageMode sm){
  MTLTextureDescriptor *x=[MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:64 resourceOptions:(sm<<MTLResourceStorageModeShift) usage:MTLTextureUsageShaderRead];
  return [D newTextureWithDescriptor:x]; }
int main(int argc,char**argv){ if(argc<2){fprintf(stderr,"usage: uses <case>\n");return 2;}
 const char *c=argv[1];
 @autoreleasepool {
 #define IS(s) (strcmp(c,s)==0)
 #define HAS(s) (strstr(c,s)!=NULL)
 P("case=%s MTL_DEBUG_LAYER=%s",c,getenv("MTL_DEBUG_LAYER")?:"unset");
 D=MTLCreateSystemDefaultDevice(); Q=[D newCommandQueue];
 MTLStorageMode sm = HAS("private") ? MTLStorageModePrivate : (HAS("shared") ? MTLStorageModeShared : MTLStorageModeMemoryless);
 MTLTextureUsage rw = MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
 unsigned char px[16*16*4]; memset(px,0x5A,sizeof px);
 if(HAS("cpu-")){
   id<MTLTexture> t = HAS("tb") ? tbuf(sm) : tex(sm,1,rw);
   P("made=%d storage=%lu",t?1:0,(unsigned long)t.storageMode); if(!t) return 4;
   NSUInteger w=t.width, h=t.height;
   MTLRegion r=MTLRegionMake2D(0,0,w,h);
   if(HAS("get")){ memset(px,0x5A,sizeof px);
     @try{ [t getBytes:px bytesPerRow:w*4 fromRegion:r mipmapLevel:0]; P("getBytes returned byte0=%02x byte_last=%02x",px[0],px[w*h*4-1]); }
     @catch(NSException *e){ P("getBytes raised %s",[[e reason] UTF8String]); } }
   if(HAS("put")){
     @try{ [t replaceRegion:r mipmapLevel:0 withBytes:px bytesPerRow:w*4]; P("replaceRegion returned"); }
     @catch(NSException *e){ P("replaceRegion raised %s",[[e reason] UTF8String]); }
     if(HAS("putget")){ unsigned char back[16*16*4]; memset(back,0,sizeof back);
       @try{ [t getBytes:back bytesPerRow:w*4 fromRegion:r mipmapLevel:0]; P("read back byte0=%02x",back[0]); }
       @catch(NSException *e){ P("read back raised %s",[[e reason] UTF8String]); } } }
 } else if(HAS("view-")){
   id<MTLTexture> t = HAS("tb") ? tbuf(sm) : tex(sm,HAS("levels")?3:1,rw|MTLTextureUsagePixelFormatView);
   P("made=%d storage=%lu",t?1:0,(unsigned long)t.storageMode); if(!t) return 4;
   id<MTLTexture> v=nil;
   @try{
     if(HAS("format")) v=[t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
     else if(HAS("slices")) v=[t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm textureType:MTLTextureType2D levels:NSMakeRange(0,1) slices:NSMakeRange(0,1)];
     else if(HAS("swizzle")) v=[t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm textureType:MTLTextureType2D levels:NSMakeRange(0,1) slices:NSMakeRange(0,1) swizzle:MTLTextureSwizzleChannelsMake(MTLTextureSwizzleBlue,MTLTextureSwizzleGreen,MTLTextureSwizzleRed,MTLTextureSwizzleAlpha)];
     else if(HAS("same")) v=[t newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
     P("view=%d storage=%lu format=%lu parent=%d",v?1:0,(unsigned long)v.storageMode,(unsigned long)v.pixelFormat,(v.parentTexture==t)?1:0);
   } @catch(NSException *e){ P("view raised %s",[[e reason] UTF8String]); }
   if(v && HAS("-attach")){
     // The view as a render pass's attachment: the pass clears it and stores DontCare.
     MTLRenderPassDescriptor *rp=[MTLRenderPassDescriptor renderPassDescriptor];
     rp.colorAttachments[0].texture=v; rp.colorAttachments[0].loadAction=MTLLoadActionClear;
     rp.colorAttachments[0].storeAction=MTLStoreActionDontCare;
     id<MTLCommandBuffer> cb=[Q commandBuffer]; id<MTLRenderCommandEncoder> re=[cb renderCommandEncoderWithDescriptor:rp];
     [re endEncoding]; finish(cb,"attach");
   }
 } else if(HAS("blit-")){
   id<MTLBuffer> marks=[D newBufferWithLength:64 options:MTLResourceStorageModeShared]; memset(marks.contents,0,64);
   id<MTLBuffer> buf=[D newBufferWithLength:16*16*4*4 options:MTLResourceStorageModeShared]; memset(buf.contents,0xC3,buf.length);
   id<MTLTexture> m = tex(sm, HAS("mips")?5:1, rw);
   id<MTLTexture> o = tex(MTLStorageModeShared,1,rw);
   [o replaceRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0 withBytes:px bytesPerRow:64];
   P("made=%d storage=%lu",m?1:0,(unsigned long)m.storageMode); if(!m) return 4;
   id<MTLCommandBuffer> cb=[Q commandBuffer];
   id<MTLBlitCommandEncoder> b0=[cb blitCommandEncoder]; [b0 fillBuffer:marks range:NSMakeRange(0,4) value:0x11]; [b0 endEncoding];
   id<MTLBlitCommandEncoder> be=[cb blitCommandEncoder];
   [be fillBuffer:marks range:NSMakeRange(4,4) value:0x22];
   @try{
     if(HAS("tobuf")) [be copyFromTexture:m sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(16,16,1) toBuffer:buf destinationOffset:0 destinationBytesPerRow:64 destinationBytesPerImage:1024];
     else if(HAS("frombuf")) [be copyFromBuffer:buf sourceOffset:0 sourceBytesPerRow:64 sourceBytesPerImage:1024 sourceSize:MTLSizeMake(16,16,1) toTexture:m destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
     else if(HAS("texsrc")) [be copyFromTexture:m sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(16,16,1) toTexture:o destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
     else if(HAS("texdst")) [be copyFromTexture:o sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0,0,0) sourceSize:MTLSizeMake(16,16,1) toTexture:m destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0,0,0)];
     else if(HAS("wholesrc")) [be copyFromTexture:m toTexture:o];
     else if(HAS("mips")) [be generateMipmapsForTexture:m];
     else if(HAS("optimize")) [be optimizeContentsForGPUAccess:m];
     P("blit command returned");
   } @catch(NSException *e){ P("blit raised %s",[[e reason] UTF8String]); }
   [be fillBuffer:marks range:NSMakeRange(8,4) value:0x33];
   [be endEncoding];
   id<MTLBlitCommandEncoder> b2=[cb blitCommandEncoder]; [b2 fillBuffer:marks range:NSMakeRange(12,4) value:0x44]; [b2 endEncoding];
   finish(cb,"blit");
   unsigned char *k=(unsigned char*)marks.contents; unsigned char *bb=(unsigned char*)buf.contents;
   P("marks prior_encoder=%02x same_before=%02x same_after=%02x later_encoder=%02x buf0=%02x",k[0],k[4],k[8],k[12],bb[0]);
   unsigned char back[16*16*4]; [o getBytes:back bytesPerRow:64 fromRegion:MTLRegionMake2D(0,0,16,16) mipmapLevel:0];
   P("other texel0=%02x%02x%02x%02x",back[0],back[1],back[2],back[3]);
 }
 // Canary.
 id<MTLCommandBuffer> cb2=[Q commandBuffer]; id<MTLBlitCommandEncoder> bc=[cb2 blitCommandEncoder];
 id<MTLBuffer> cbuf=[D newBufferWithLength:16 options:MTLResourceStorageModeShared];
 [bc fillBuffer:cbuf range:NSMakeRange(0,16) value:7]; [bc endEncoding]; finish(cb2,"canary");
 }
 return 0; }
