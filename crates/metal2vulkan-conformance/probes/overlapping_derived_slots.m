// What Metal answers for two texture arguments that function constants put on one slot through
// derived constants, the shape of a framework convolution kernel's interface: a write texture
// `dst` at texture(0), present where the derived `dst_is_2d` holds, and a sampled texture `src` at
// texture(batch), present where the derived `src_is_2d` holds. `batch` is function constant 125,
// and both predicates are bits of function constant 126 (`kind`): `dst_is_2d` is
// `(kind & 3) == 0` and `src_is_2d` is `((kind >> 6) & 3) == 0`. Its static initializer
// computes them, and the arguments' locations and gates name the results.
//
// For each case it prints whether Metal makes the function and the pipeline, or its error; for a
// pipeline it makes, it runs 4 threads over a 4 by 1 `R32Float` `src` holding 1 2 3 4 with
// Metal's default sampler state and prints the 4 words of `dst`.
//
// - kind 0, batch 0: both present, both at texture 0.
// - kind 0, batch 1: both present, at textures 0 and 1.
// - kind 2, batch 0: `dst` gated off, `src` alone at texture 0 (nothing written).
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o overlapping_derived_slots overlapping_derived_slots.m \
//     -framework Foundation -framework Metal
// Run: ./overlapping_derived_slots, MTL_DEBUG_LAYER unset.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *SOURCE =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "constant uint batch [[function_constant(125)]];\n"
    "constant uint kind [[function_constant(126)]];\n"
    "constant bool dst_is_2d = (kind & 3) == 0;\n"
    "constant bool src_is_2d = ((kind >> 6) & 3) == 0;\n"
    "\n"
    "kernel void overlapping_derived_slots(\n"
    "    texture2d<float, access::write> dst [[texture(0), function_constant(dst_is_2d)]],\n"
    "    texture2d<float> src [[texture(batch), function_constant(src_is_2d)]],\n"
    "    sampler s [[sampler(0)]],\n"
    "    uint2 gid [[thread_position_in_grid]]) {\n"
    "    if (dst_is_2d && src_is_2d) {\n"
    "        dst.write(src.sample(s, (float2(gid) + 0.5) / 4.0), gid);\n"
    "    }\n"
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
    const uint32_t cases[3][2] = {{0, 0}, {0, 1}, {2, 0}};
    for (int c = 0; c < 3; c++) {
      uint32_t kind = cases[c][0], batch = cases[c][1];
      MTLFunctionConstantValues *v = [MTLFunctionConstantValues new];
      [v setConstantValue:&batch type:MTLDataTypeUInt atIndex:125];
      [v setConstantValue:&kind type:MTLDataTypeUInt atIndex:126];
      e = nil;
      id<MTLFunction> f = [lib newFunctionWithName:@"overlapping_derived_slots"
                                    constantValues:v
                                             error:&e];
      if (!f) {
        printf("RESULT kind=%u batch=%u function error: %s\n", kind, batch,
               e.localizedDescription.UTF8String);
        continue;
      }
      e = nil;
      id<MTLComputePipelineState> ps = [d newComputePipelineStateWithFunction:f error:&e];
      if (!ps) {
        printf("RESULT kind=%u batch=%u pipeline error: %s\n", kind, batch,
               e.localizedDescription.UTF8String);
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
      [ce setTexture:dst atIndex:0];
      [ce setTexture:src atIndex:batch];
      [ce setSamplerState:s atIndex:0];
      [ce dispatchThreads:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
      [ce endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      float out[4] = {0};
      [dst getBytes:out bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 4, 1) mipmapLevel:0];
      printf("RESULT kind=%u batch=%u status=%ld words %g %g %g %g\n", kind, batch,
             (long)cb.status, out[0], out[1], out[2], out[3]);
    }
  }
  return 0;
}
