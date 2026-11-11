// How many samples plain Metal's alpha to coverage covers, pixel by pixel, over a sweep of alphas
// at every sample count the device offers: the reference `runtime_lane::alpha_coverage` and the
// `device-alpha-coverage` note's bound are read against.
//
// The sweep is the lane's, drawn the lane's way, so the two print maps that compare line by line:
// every multiple of 1/512 in [0, 1] and the float either side of every multiple of 1/16 inside it,
// sorted, each alpha filling its own 4x4 block (by a scissor) of a 128x128 RGBA16Float target of n
// samples cleared to zero, a fullscreen triangle whose fragment returns (1, 1, 1, alpha) with
// alphaToCoverageEnabled, resolved where n is above 1. A pixel's red is k / n for the k samples
// it covered. It prints one line an alpha and count:
//
//   A2C-MAP <n> <the alpha's float bits, hex> <the block's 16 counts, row by row, base 36>
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1, samples 1, 2 and 4 (the counts the device offers
// of 1, 2, 4 and 8):
//
//   - 1 sample: a pixel is covered exactly where alpha > 0.5, at every pixel alike;
//   - 2 and 4 samples: the count depends on the pixel, each pixel's floor(alpha * n) or
//     ceil(alpha * n), and exactly alpha * n where that is an integer;
//   - every line equals MoltenVK 1.4.2's on the same chip (`runtime_lane::alpha_coverage`'s dump),
//     which hands `alphaToCoverageEnable` to `alphaToCoverageEnabled`.
//
// So Metal's coverage is not floor(alpha * n): the conformance family `alpha_coverage_pairs` reads
// one pixel at alphas whose alpha * n is an integer, and 0.5 at one sample, where the two agree.
//
// Run it WITHOUT MTL_DEBUG_LAYER, with the Mac's GPU to itself:
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o alpha_coverage_sweep alpha_coverage_sweep.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const NSUInteger kSide = 128;
static const NSUInteger kBlock = 4;

static NSString *kSource =
  @"#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "vertex float4 vmain(uint id [[vertex_id]]) {\n"
   "  float2 p = float2(float((id << 1) & 2), float(id & 2)) * 2.0 - 1.0;\n"
   "  return float4(p, 0, 1);\n"
   "}\n"
   "fragment float4 fmain(constant float4 &color [[buffer(0)]]) { return color; }\n";

static int ascending(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return x < y ? -1 : x > y;
}

static float bits_float(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static uint32_t float_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

// The lane's sweep, ascending, each once; returns its length.
static NSUInteger sweep(float *out) {
  NSUInteger n = 0;
  for (int i = 0; i <= 512; i++) out[n++] = (float)i / 512.0f;
  for (int j = 1; j < 64; j++) {
    float edge = (float)j / 64.0f;
    out[n++] = bits_float(float_bits(edge) - 1);
    out[n++] = bits_float(float_bits(edge) + 1);
  }
  qsort(out, n, sizeof(float), ascending);
  NSUInteger kept = 0;
  for (NSUInteger i = 0; i < n; i++) {
    if (kept == 0 || out[kept - 1] != out[i]) out[kept++] = out[i];
  }
  return kept;
}

int main(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!library) {
      fprintf(stderr, "library: %s\n", error.description.UTF8String);
      return 1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    float alphas[1024];
    NSUInteger count = sweep(alphas);
    fprintf(stderr, "%s: %lu alphas\n", device.name.UTF8String, (unsigned long)count);
    NSUInteger row = kSide / kBlock;
    for (NSUInteger n = 1; n <= 8; n *= 2) {
      if (![device supportsTextureSampleCount:n]) continue;
      MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
      d.vertexFunction = [library newFunctionWithName:@"vmain"];
      d.fragmentFunction = [library newFunctionWithName:@"fmain"];
      d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
      d.rasterSampleCount = n;
      d.alphaToCoverageEnabled = YES;
      id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:d
                                                                                error:&error];
      if (!state) {
        fprintf(stderr, "%lu samples: %s\n", (unsigned long)n, error.description.UTF8String);
        return 1;
      }
      MTLTextureDescriptor *t =
          [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                             width:kSide
                                                            height:kSide
                                                         mipmapped:NO];
      t.usage = MTLTextureUsageRenderTarget;
      t.storageMode = MTLStorageModeShared;
      id<MTLTexture> read = [device newTextureWithDescriptor:t];
      MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
      if (n > 1) {
        MTLTextureDescriptor *m =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                               width:kSide
                                                              height:kSide
                                                           mipmapped:NO];
        m.textureType = MTLTextureType2DMultisample;
        m.sampleCount = n;
        m.usage = MTLTextureUsageRenderTarget;
        m.storageMode = MTLStorageModePrivate;
        pass.colorAttachments[0].texture = [device newTextureWithDescriptor:m];
        pass.colorAttachments[0].resolveTexture = read;
        pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
      } else {
        pass.colorAttachments[0].texture = read;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      }
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
      [e setRenderPipelineState:state];
      for (NSUInteger i = 0; i < count; i++) {
        MTLScissorRect s = {i % row * kBlock, i / row * kBlock, kBlock, kBlock};
        [e setScissorRect:s];
        float color[4] = {1, 1, 1, alphas[i]};
        [e setFragmentBytes:color length:sizeof color atIndex:0];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      }
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) {
        fprintf(stderr, "%lu samples: %s\n", (unsigned long)n, cb.error.description.UTF8String);
        return 1;
      }
      static _Float16 texels[128 * 128 * 4];
      [read getBytes:texels
          bytesPerRow:kSide * 8
           fromRegion:MTLRegionMake2D(0, 0, kSide, kSide)
          mipmapLevel:0];
      for (NSUInteger i = 0; i < count; i++) {
        char counts[17] = {0};
        NSUInteger x0 = i % row * kBlock, y0 = i / row * kBlock, at = 0;
        for (NSUInteger dy = 0; dy < kBlock; dy++) {
          for (NSUInteger dx = 0; dx < kBlock; dx++) {
            double red = (double)texels[((y0 + dy) * kSide + x0 + dx) * 4];
            double k = red * (double)n;
            if (k != floor(k) || k < 0 || k > n) {
              fprintf(stderr, "%lu samples: a red of %g is no count\n", (unsigned long)n, red);
              return 1;
            }
            counts[at++] = "0123456789abcdefghijklmnopqrstuvwxyz"[(int)k];
          }
        }
        printf("%lu %08x %s\n", (unsigned long)n, float_bits(alphas[i]), counts);
      }
    }
  }
  return 0;
}
