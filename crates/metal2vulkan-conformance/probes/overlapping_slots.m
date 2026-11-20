// What Metal answers for two texture arguments that function constants put on one slot: the
// owned fixture `engine/overlapping_slots.metal`'s source, whose write texture `dst` is at
// texture(1) and whose sampled texture `src` is at texture(n), n function constant 0. For n = 0
// and n = 1 it prints whether Metal makes the function and the pipeline, or its error; for a
// pipeline it makes, it runs 4 threads over a 4 by 1 `R32Float` `src` holding 1 2 3 4 with
// Metal's default sampler state and prints the 4 words of `dst`.
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o overlapping_slots overlapping_slots.m \
//     -framework Foundation -framework Metal
// Run: ./overlapping_slots, MTL_DEBUG_LAYER unset.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "constant uint n [[function_constant(0)]];\n"
    "\n"
    "kernel void overlapping_slots(texture2d<float, access::write> dst [[texture(1)]],\n"
    "                              texture2d<float> src [[texture(n)]],\n"
    "                              sampler s [[sampler(0)]],\n"
    "                              uint2 gid [[thread_position_in_grid]]) {\n"
    "    dst.write(src.sample(s, (float2(gid) + 0.5) / 4.0), gid);\n"
    "}\n";

int main(void) {
  @autoreleasepool {
    printf("RESULT debug_layer=%s\n", getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    printf("RESULT device=%s\n", d.name.UTF8String);
    NSError *e = nil;
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:SOURCE]
                                         options:[MTLCompileOptions new]
                                           error:&e];
    if (!lib) {
      printf("RESULT library error: %s\n", e.localizedDescription.UTF8String);
      return 1;
    }
    for (uint32_t n = 0; n < 2; n++) {
      MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
      [v setConstantValue:&n type:MTLDataTypeUInt atIndex:0];
      e = nil;
      id<MTLFunction> f = [lib newFunctionWithName:@"overlapping_slots" constantValues:v error:&e];
      if (!f) {
        printf("RESULT n=%u function error: %s\n", n, e.localizedDescription.UTF8String);
        continue;
      }
      e = nil;
      id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:f error:&e];
      if (!ps) {
        printf("RESULT n=%u pipeline error: %s\n", n, e.localizedDescription.UTF8String);
        continue;
      }
      MTLTextureDescriptor *td =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                             width:4
                                                            height:1
                                                         mipmapped:NO];
      td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
      td.storageMode = MTLStorageModeShared;
      id<MTLTexture> src = [d newTextureWithDescriptor:td];
      id<MTLTexture> dst = [d newTextureWithDescriptor:td];
      const float texels[4] = {1, 2, 3, 4};
      [src replaceRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0 withBytes:texels bytesPerRow:16];
      id<MTLSamplerState> s = [d newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
      id<MTLCommandQueue> q = [d newCommandQueue];
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
      [ce setComputePipelineState:ps];
      [ce setTexture:dst atIndex:1];
      [ce setTexture:src atIndex:n];
      [ce setSamplerState:s atIndex:0];
      [ce dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
      [ce endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      float out[4] = {0};
      [dst getBytes:out bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0];
      printf("RESULT n=%u status=%ld words %g %g %g %g\n", n, (long)cb.status, out[0], out[1],
             out[2], out[3]);
    }
  }
  return 0;
}
