// What plain Metal stores into a color attachment of the pass that the draw's render pipeline has
// no pixel format for, when the fragment function writes that color itself.
//
// Measured on an Apple M3 Max under macOS 27.0.1, `MTL_DEBUG_LAYER` out of the environment
// (Metal's validation stops such a draw), one case a process; the runtime lane
// `runtime_lane::extra_colors` holds the answers.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o written_color_the_pipeline_lacks \
//     written_color_the_pipeline_lacks.m -framework Foundation -framework Metal
//
// A case is `<format>-<fragment>-<load>[-ms4]`: color 1's format (`rgba8`, `rgba16f`, `r32u`,
// `rgba32i`), the fragment function (`f`, `u`, `i`: color 1 written as `float4(1,0,0,1)`,
// `uint4(7,8,9,10)`, `int4(-7,8,-9,10)`; `none`: color 0 alone), and color 1's load action
// (`clear` to green, `(5,0,0,0)` or `(-5,6,-7,8)`; `load` of bytes 0xAB). The pass is 4x4, color 0
// `RGBA8Unorm` cleared black, both stored; the pipeline has color 0 alone. One draw: the
// lower-left half triangle, or at `-ms4` (4 samples, both colors resolved) the fullscreen one.
// Each texel of color 1 (its resolve at `-ms4`) and of color 0 is printed as its bytes in hex.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>
static const char *SRC =
"#include <metal_stdlib>\nusing namespace metal;\n"
"struct V { float4 p [[position]]; };\n"
"vertex V vhalf(uint i [[vertex_id]]) { float2 q[3] = {float2(-1,-1), float2(1,-1), float2(-1,1)};"
"  V v; v.p = float4(q[i], 0.5, 1); return v; }\n"
"vertex V vfull(uint i [[vertex_id]]) { float2 q[3] = {float2(-1,-1), float2(3,-1), float2(-1,3)};"
"  V v; v.p = float4(q[i], 0.5, 1); return v; }\n"
"struct OF { float4 c0 [[color(0)]]; float4 c1 [[color(1)]]; };\n"
"fragment OF ff() { OF o; o.c0 = float4(1); o.c1 = float4(1,0,0,1); return o; }\n"
"struct OU { float4 c0 [[color(0)]]; uint4 c1 [[color(1)]]; };\n"
"fragment OU fu() { OU o; o.c0 = float4(1); o.c1 = uint4(7,8,9,10); return o; }\n"
"struct OI { float4 c0 [[color(0)]]; int4 c1 [[color(1)]]; };\n"
"fragment OI fi() { OI o; o.c0 = float4(1); o.c1 = int4(-7,8,-9,10); return o; }\n"
"fragment float4 fnone() { return float4(1); }\n";

static void dump(const char *what, id<MTLTexture> t, size_t bpp) {
  unsigned char b[4 * 4 * 16];
  [t getBytes:b bytesPerRow:4 * bpp fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
  for (int y = 0; y < 4; y++) {
    printf("RESULT %s row%d", what, y);
    for (int x = 0; x < 4; x++) {
      printf(" ");
      for (size_t k = 0; k < bpp; k++) printf("%02x", b[(y * 4 + x) * bpp + k]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
#define HAS(s) (strstr(c, s) != NULL)
  @autoreleasepool {
    printf("RESULT case=%s debug_layer=%s\n", c, getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> L = [D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
    if (!L) { printf("RESULT library error %s\n", e.localizedDescription.UTF8String); return 1; }
    MTLPixelFormat f1; size_t bpp; MTLClearColor clear;
    if (HAS("rgba8-")) { f1 = MTLPixelFormatRGBA8Unorm; bpp = 4; clear = MTLClearColorMake(0, 1, 0, 1); }
    else if (HAS("rgba16f-")) { f1 = MTLPixelFormatRGBA16Float; bpp = 8; clear = MTLClearColorMake(0, 1, 0, 1); }
    else if (HAS("r32u-")) { f1 = MTLPixelFormatR32Uint; bpp = 4; clear = MTLClearColorMake(5, 0, 0, 0); }
    else if (HAS("rgba32i-")) { f1 = MTLPixelFormatRGBA32Sint; bpp = 16; clear = MTLClearColorMake(-5, 6, -7, 8); }
    else return 2;
    NSString *frag = HAS("-none-") ? @"fnone" : HAS("-u-") ? @"fu" : HAS("-i-") ? @"fi" : @"ff";
    BOOL ms = HAS("-ms4");
    NSUInteger samples = ms ? 4 : 1;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [L newFunctionWithName:ms ? @"vfull" : @"vhalf"];
    pd.fragmentFunction = [L newFunctionWithName:frag];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    pd.rasterSampleCount = samples;
    id<MTLRenderPipelineState> ps = [D newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 1; }
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> c0 = [D newTextureWithDescriptor:td];
    td.pixelFormat = f1;
    id<MTLTexture> c1 = [D newTextureWithDescriptor:td];
    unsigned char ab[4 * 4 * 16];
    memset(ab, 0xAB, sizeof ab);
    [c1 replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:ab bytesPerRow:4 * bpp];
    id<MTLTexture> m0 = nil, m1 = nil;
    if (ms) {
      MTLTextureDescriptor *md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:4 height:4 mipmapped:NO];
      md.textureType = MTLTextureType2DMultisample;
      md.sampleCount = 4;
      md.usage = MTLTextureUsageRenderTarget;
      md.storageMode = MTLStorageModePrivate;
      m0 = [D newTextureWithDescriptor:md];
      md.pixelFormat = f1;
      m1 = [D newTextureWithDescriptor:md];
    }
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = ms ? m0 : c0;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    rp.colorAttachments[1].texture = ms ? m1 : c1;
    rp.colorAttachments[1].loadAction = HAS("-load") ? MTLLoadActionLoad : MTLLoadActionClear;
    rp.colorAttachments[1].clearColor = clear;
    if (ms) {
      rp.colorAttachments[0].resolveTexture = c0;
      rp.colorAttachments[1].resolveTexture = c1;
      rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
      rp.colorAttachments[1].storeAction = MTLStoreActionMultisampleResolve;
    } else {
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      rp.colorAttachments[1].storeAction = MTLStoreActionStore;
    }
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLRenderCommandEncoder> re = [cb renderCommandEncoderWithDescriptor:rp];
    [re setRenderPipelineState:ps];
    [re drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [re endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("RESULT status=%ld error=%s\n", (long)cb.status, cb.error ? cb.error.localizedDescription.UTF8String : "none");
    dump("color1", c1, bpp);
    dump("color0", c0, 4);
  }
  return 0;
}
