// What plain Metal makes of a fragment function that declares an input whose read its function
// constant makes dead: a `[[color(0)]]` input in a raster order group, a `const device` buffer in
// one, a `[[color(0)]]` input of an output in one, and a `[[color(1)]]` input of a color the
// pipeline has no pixel format for. The reference of `metal2vulkan`'s `tests::engine_dead_reads`
// and `tests::engine_color_groups`, and of the runtime lane `runtime_lane::color_groups`, over the
// owned fixture `engine/dfk_dead_rog` (its source is repeated here, with `color_groups`'
// `cg_cover`).
//
// Build (no cargo target needed):
//   clang -fobjc-arc -O1 -fmodules -o dead_ordered_and_unattached_reads \
//     dead_ordered_and_unattached_reads.m -framework Foundation -framework Metal
//
// A case is `<function>-<dead|live>`, one a process: `color`, `buffer` (five `RGBA16Float` colors),
// `output` (color 0 alone), `lacks` (a pipeline of color 0 alone drawn in a pass of colors 0 and
// 1; run it with `MTL_DEBUG_LAYER` out of the environment, whose validation stops such a draw).
// The function is made with `newFunctionWithName:constantValues:error:`, constant 0 `false`
// (dead) or `true` (live). The pass is 8x8, color `i` cleared to
// `(0.0625 (i + 1), 0.5, 0.125 i, 1)`, every color stored; the buffer holds
// `(0.125, 0.25, 0.375, 0.5)`. One draw of `cg_cover`'s fullscreen triangle. Printed: whether the
// state is made (or Metal's error), and for each color the number of texels whose bits are the
// closed form's (`tint` at the pixel center, `tint`'s swizzles, the clear for a color nothing
// draws, zero for the color the pipeline lacks; a live read's value plus `tint`) and its first
// texel's halves.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (`rasterOrderGroupsSupported` YES), with
// `MTL_DEBUG_LAYER` unset and set alike (the layer reported nothing):
//
//   color, buffer, output   the state is made and the draw completes, dead and live; every texel of
//                           every color is the closed form's (64 of 64), the dead read's `tint` at
//                           color 0, the live one's read plus `tint` (color 0's first texel
//                           (0.03125, 0.46875, 0.96875, 0.25) dead; (0.0625, 0.71875, 0.96875,
//                           0.75) for a color read live, (0.15625, 0.71875, 1.34375, 0.75) for the
//                           buffer's).
//   lacks                   no state, dead and live: "Shaders reads from a color attachment whose
//                           pixel format is MTLPixelFormatInvalid".
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

static const char *SRC =
"#include <metal_stdlib>\nusing namespace metal;\n"
"constant bool dfr_reads [[function_constant(0)]];\n"
"struct CgPoint { float4 position [[position]]; float4 tint; };\n"
"vertex CgPoint cg_cover(uint id [[vertex_id]]) {\n"
"  float2 a = float2(float(id & 1u), float(id >> 1));\n"
"  CgPoint out; out.position = float4(4.0 * a - 1.0, 0.5 * a.x, 1.0);\n"
"  out.tint = float4(a, 1.0 - a.x, 0.25); return out; }\n"
"struct DfrFive { float4 c0 [[color(0)]]; float4 c1 [[color(1)]]; float4 c2 [[color(2)]];\n"
"  float4 c3 [[color(3)]]; float4 c4 [[color(4)]]; };\n"
"struct DfrOrdered { float4 c0 [[color(0), raster_order_group(0)]]; };\n"
"fragment DfrFive dfr_color(CgPoint in [[stage_in]],\n"
"    float4 first [[color(0), raster_order_group(0)]]) {\n"
"  float4 t = in.tint; float4 c0 = t; if (dfr_reads) { c0 = first * 0.5 + t; }\n"
"  return DfrFive { c0, t.yzwx, t.zwxy, t.wxyz, t }; }\n"
"fragment DfrFive dfr_buffer(CgPoint in [[stage_in]],\n"
"    const device float4 *b [[buffer(0), raster_order_group(0)]]) {\n"
"  float4 t = in.tint; float4 c0 = t; if (dfr_reads) { c0 = b[0] + t; }\n"
"  return DfrFive { c0, t.yzwx, t.zwxy, t.wxyz, t }; }\n"
"fragment DfrOrdered dfr_output(CgPoint in [[stage_in]], float4 first [[color(0)]]) {\n"
"  float4 t = in.tint; float4 c0 = t; if (dfr_reads) { c0 = first * 0.5 + t; }\n"
"  return DfrOrdered { c0 }; }\n"
"fragment float4 dfr_lacks(CgPoint in [[stage_in]], float4 extra [[color(1)]]) {\n"
"  float4 t = in.tint; float4 c0 = t; if (dfr_reads) { c0 = extra * 0.5 + t; }\n"
"  return c0; }\n";

enum { SIDE = 8 };

static void clear_of(int i, float out[4]) {
  out[0] = 0.0625f * (float)(i + 1);
  out[1] = 0.5f;
  out[2] = 0.125f * (float)i;
  out[3] = 1.0f;
}

static void tint(int x, int y, float t[4]) {
  float ax = (float)(2 * x + 1) / (2.0f * SIDE) * 0.5f;
  float ay = (float)(2 * (SIDE - 1 - y) + 1) / (2.0f * SIDE) * 0.5f;
  t[0] = ax; t[1] = ay; t[2] = 1.0f - ax; t[3] = 0.25f;
}

// The closed form of color `index` at (x, y): `fn` the function, `live` its read.
static void expected(const char *fn, int live, int index, int x, int y, float out[4]) {
  float t[4];
  tint(x, y, t);
  static const int order[4][4] = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}};
  if (!strcmp(fn, "lacks")) {
    if (index == 1) { out[0] = out[1] = out[2] = out[3] = 0.0f; return; }
  }
  if (index >= 1 && index <= 3) {
    for (int k = 0; k < 4; k++) out[k] = t[order[index][k]];
    return;
  }
  for (int k = 0; k < 4; k++) out[k] = t[k];
  if (index == 0 && live) {
    float r[4];
    if (!strcmp(fn, "buffer")) {
      r[0] = 0.125f; r[1] = 0.25f; r[2] = 0.375f; r[3] = 0.5f;
      for (int k = 0; k < 4; k++) out[k] = r[k] + t[k];
    } else {
      clear_of(!strcmp(fn, "lacks") ? 1 : 0, r);
      for (int k = 0; k < 4; k++) out[k] = r[k] * 0.5f + t[k];
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  const char *c = argv[1];
  char fn[16] = {0};
  const char *dash = strchr(c, '-');
  if (!dash || dash - c >= (long)sizeof fn) return 2;
  memcpy(fn, c, (size_t)(dash - c));
  int live = !strcmp(dash + 1, "live");
  @autoreleasepool {
    printf("RESULT case=%s debug_layer=%s\n", c,
           getenv("MTL_DEBUG_LAYER") ? getenv("MTL_DEBUG_LAYER") : "unset");
    id<MTLDevice> D = MTLCreateSystemDefaultDevice();
    printf("RESULT device=%s rog_supported=%d\n", D.name.UTF8String, D.rasterOrderGroupsSupported);
    id<MTLCommandQueue> Q = [D newCommandQueue];
    NSError *e = nil;
    id<MTLLibrary> L = [D newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:nil error:&e];
    if (!L) { printf("RESULT library error %s\n", e.localizedDescription.UTF8String); return 1; }
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    bool reads = live;
    [cv setConstantValue:&reads type:MTLDataTypeBool atIndex:0];
    NSString *name = [NSString stringWithFormat:@"dfr_%s", fn];
    id<MTLFunction> F = [L newFunctionWithName:name constantValues:cv error:&e];
    if (!F) { printf("RESULT function error %s\n", e.localizedDescription.UTF8String); return 1; }
    int five = !strcmp(fn, "color") || !strcmp(fn, "buffer");
    int pipeline_colors = five ? 5 : 1;
    int pass_colors = five ? 5 : !strcmp(fn, "lacks") ? 2 : 1;
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
    pd.vertexFunction = [L newFunctionWithName:@"cg_cover"];
    pd.fragmentFunction = F;
    for (int i = 0; i < pipeline_colors; i++) pd.colorAttachments[i].pixelFormat = MTLPixelFormatRGBA16Float;
    id<MTLRenderPipelineState> ps = [D newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!ps) { printf("RESULT pipeline error %s\n", e.localizedDescription.UTF8String); return 0; }
    printf("RESULT pipeline made\n");
    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                 width:SIDE height:SIDE mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> T[5];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    for (int i = 0; i < pass_colors; i++) {
      T[i] = [D newTextureWithDescriptor:td];
      float k[4];
      clear_of(i, k);
      rp.colorAttachments[i].texture = T[i];
      rp.colorAttachments[i].loadAction = MTLLoadActionClear;
      rp.colorAttachments[i].clearColor = MTLClearColorMake(k[0], k[1], k[2], k[3]);
      rp.colorAttachments[i].storeAction = MTLStoreActionStore;
    }
    float data[4] = {0.125f, 0.25f, 0.375f, 0.5f};
    id<MTLBuffer> B = [D newBufferWithBytes:data length:sizeof data options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [Q commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:ps];
    [enc setFragmentBuffer:B offset:0 atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    printf("RESULT command buffer status=%ld error=%s\n", (long)cb.status,
           cb.error ? cb.error.localizedDescription.UTF8String : "none");
    for (int i = 0; i < pass_colors; i++) {
      _Float16 px[SIDE * SIDE * 4];
      [T[i] getBytes:px bytesPerRow:SIDE * 8 fromRegion:MTLRegionMake2D(0, 0, SIDE, SIDE) mipmapLevel:0];
      int same = 0;
      for (int y = 0; y < SIDE; y++)
        for (int x = 0; x < SIDE; x++) {
          float want[4];
          expected(fn, live, i, x, y, want);
          int ok = 1;
          for (int k = 0; k < 4; k++) {
            _Float16 w = (_Float16)want[k];
            if (memcmp(&w, &px[(y * SIDE + x) * 4 + k], sizeof w)) ok = 0;
          }
          same += ok;
        }
      printf("RESULT color %d: %d of %d texels the closed form's; first (%g, %g, %g, %g)\n", i, same,
             SIDE * SIDE, (double)px[0], (double)px[1], (double)px[2], (double)px[3]);
    }
  }
  return 0;
}
