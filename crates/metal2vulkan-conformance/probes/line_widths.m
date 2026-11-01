// What plain Metal draws for `-[MTLRenderCommandEncoder setLineWidth:]` (SPI), the reference the
// runtime's line widths (`metal2vulkan-runtime`'s `line_width.rs`) and the lane test
// `runtime_lane::line_widths` are read against.
//
// One case a process, named on the command line (no name lists them). Every target is cleared
// to zero and drawn by a pass-through vertex function that places pixel coordinates (y down) of
// an S x S target, `(x / (S / 2) - 1, 1 - y / (S / 2))`, and a fragment function that returns 1
// (or 1/255, added, where a map counts fragments). A map prints the rows a draw touched, one
// character a pixel: `.` none, `#` a full pixel, digits the fragments or the samples covered.
//
//   lane       the lane test's draws on 16 x 16: a list of two lines and a three-vertex strip at
//              widths 0, 0.5, 1, 2, 3.5 and 8, side by side;
//   quantize   which width Metal draws: the rows a horizontal line at y = 32 (64 x 64) covers in
//              column 32, and whether a vertical line at x = 16.5 is the diamond rule's (rows 3 to
//              26) or a band's (4 to 27), over widths near multiples of 1/16, past 16, negative,
//              NaN and infinite;
//   state      the width each encoder starts at, and that it holds across draws and pipeline states;
//   ties       a band's edges: rows of a horizontal line on a pixel edge and through centres, ends
//              on pixel centres in both directions, 45 degrees both ways, a y-major line;
//   zero       width 0 and below 1/16: centres on the segment, its ends, a segment through none;
//   strips     fragments counted where a strip's segments overlap, at widths 1, 3.5 and 8;
//   fill       triangles under the fill mode Lines at widths 1 and 4, and Fill at 4, counted;
//   msaa       4 samples, resolved: a horizontal line at y = 16 at widths 0.5, 1 and 2.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (every case's output is
// `metal2vulkan-validation`'s `runtime_lane/line_widths_m3_max.txt`):
//
//   - Each encoder starts at width 1; a width holds across draws and pipeline states.
//   - Metal draws w' = floor(16 w) / 16 clamped to [0, 16]: negative widths and NaN as 0, 16 and
//     above, infinity among them, as 16.
//   - At w' = 1 the diamond-exit lines; at every other w', below 1 too, the parallelogram whose
//     minor-axis edges pass through the endpoints, w' long and centred, its other edges parallel
//     to the segment, the ends not extended, a pixel drawn where its centre is inside (top-left
//     ties), the width not rounded; at w' = 0 exactly the pixel centres on the segment.
//   - A strip draws one parallelogram per segment, no joins: an overlap takes two fragments (at 1 a
//     shared vertex is drawn once).
//   - Triangles in fill mode Lines draw their edges so; fill mode Fill ignores the width.
//   - At 4 samples every width, 1 included, is the parallelogram at the sample positions.
//
// Run it WITHOUT MTL_DEBUG_LAYER, each case on the Mac's GPU lock:
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o line_widths line_widths.m
//   maclock /usr/bin/env -u MTL_DEBUG_LAYER ./line_widths lane

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static NSString *kSource =
  @"#include <metal_stdlib>\n"
   "using namespace metal;\n"
   "struct Out { float4 position [[position]]; };\n"
   "vertex Out place(uint v [[vertex_id]], const device float2 *at [[buffer(0)]],\n"
   "                 constant float &half_side [[buffer(1)]]) {\n"
   "  Out o; float2 p = at[v];\n"
   "  o.position = float4(p.x / half_side - 1.0, 1.0 - p.y / half_side, 0.0, 1.0);\n"
   "  return o;\n"
   "}\n"
   "fragment float4 white() { return float4(1.0); }\n"
   "fragment float4 one_count() { return float4(1.0 / 255.0); }\n";

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLibrary;

typedef enum { kWhite, kCount } Shade;

static id<MTLRenderPipelineState> pipeline(Shade shade, NSUInteger samples) {
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  d.vertexFunction = [gLibrary newFunctionWithName:@"place"];
  d.fragmentFunction = [gLibrary newFunctionWithName:shade == kCount ? @"one_count" : @"white"];
  d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  d.rasterSampleCount = samples;
  if (shade == kCount) {
    MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
    c.blendingEnabled = YES;
    c.sourceRGBBlendFactor = c.sourceAlphaBlendFactor = MTLBlendFactorOne;
    c.destinationRGBBlendFactor = c.destinationAlphaBlendFactor = MTLBlendFactorOne;
  }
  NSError *error = nil;
  id<MTLRenderPipelineState> state = [gDevice newRenderPipelineStateWithDescriptor:d error:&error];
  if (!state) {
    fprintf(stderr, "pipeline: %s\n", error.localizedDescription.UTF8String);
    exit(1);
  }
  return state;
}

static void set_width(id<MTLRenderCommandEncoder> e, float width) {
  ((void (*)(id, SEL, float))objc_msgSend)(e, sel_registerName("setLineWidth:"), width);
}

// One draw of an encoder: its primitive type, its points, and the width set before it (`set` 0
// sets none).
typedef struct {
  MTLPrimitiveType type;
  const float *points;
  int count;
  int set;
  float width;
  MTLTriangleFillMode fill;
  int new_pipeline;
} Draw;

// Draws each encoder's draws on an S x S target of `samples` samples; `encoders[i]` draws are
// `draws` from `starts[i]` to `starts[i + 1]`. Returns the red bytes.
static uint8_t *render(int side, NSUInteger samples, Shade shade, const Draw *draws,
                       const int *starts, int encoders) {
  MTLTextureDescriptor *td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:side
                                                        height:side
                                                     mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> resolved = [gDevice newTextureWithDescriptor:td];
  id<MTLTexture> target = resolved;
  if (samples > 1) {
    td.textureType = MTLTextureType2DMultisample;
    td.sampleCount = samples;
    td.storageMode = MTLStorageModePrivate;
    target = [gDevice newTextureWithDescriptor:td];
  }
  id<MTLRenderPipelineState> a = pipeline(shade, samples);
  id<MTLRenderPipelineState> b = pipeline(shade, samples);
  float half_side = side / 2.0f;
  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
  for (int e = 0; e < encoders; e++) {
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = target;
    rp.colorAttachments[0].loadAction = e == 0 ? MTLLoadActionClear : MTLLoadActionLoad;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
    if (samples > 1) {
      rp.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
      rp.colorAttachments[0].resolveTexture = resolved;
    } else {
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:a];
    [enc setVertexBytes:&half_side length:sizeof half_side atIndex:1];
    for (int i = starts[e]; i < starts[e + 1]; i++) {
      const Draw *d = &draws[i];
      if (d->set) set_width(enc, d->width);
      if (d->new_pipeline) [enc setRenderPipelineState:(i & 1) ? b : a];
      [enc setTriangleFillMode:d->fill];
      id<MTLBuffer> points = [gDevice newBufferWithBytes:d->points
                                                  length:sizeof(float) * 2 * d->count
                                                 options:MTLResourceStorageModeShared];
      [enc setVertexBuffer:points offset:0 atIndex:0];
      [enc drawPrimitives:d->type vertexStart:0 vertexCount:d->count];
    }
    [enc endEncoding];
  }
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) {
    fprintf(stderr, "command buffer: %s\n", cb.error.localizedDescription.UTF8String);
    exit(1);
  }
  uint8_t *rgba = malloc((size_t)side * side * 4);
  [resolved getBytes:rgba
         bytesPerRow:side * 4
          fromRegion:MTLRegionMake2D(0, 0, side, side)
         mipmapLevel:0];
  uint8_t *red = malloc((size_t)side * side);
  for (int i = 0; i < side * side; i++) red[i] = rgba[4 * i];
  free(rgba);
  return red;
}

// One draw alone in one encoder.
static uint8_t *one(int side, NSUInteger samples, Shade shade, MTLPrimitiveType type,
                    const float *points, int count, float width) {
  Draw d = {type, points, count, 1, width, MTLTriangleFillModeFill, 0};
  int starts[2] = {0, 1};
  return render(side, samples, shade, &d, starts, 1);
}

static char cell(uint8_t v, Shade shade, NSUInteger samples) {
  if (v == 0) return '.';
  if (shade == kCount) return v < 10 ? (char)('0' + v) : '*';
  if (samples > 1) {
    // The samples covered, of `samples`: the resolve averages them.
    int k = (int)lround(v * (double)samples / 255.0);
    return (char)('0' + k);
  }
  return v == 255 ? '#' : '?';
}

static void print_map(const char *title, int side, const uint8_t *r, Shade shade,
                      NSUInteger samples) {
  int covered = 0;
  for (int i = 0; i < side * side; i++) covered += r[i] != 0;
  printf("MAP %s (%d pixels)\n", title, covered);
  for (int y = 0; y < side; y++) {
    int any = 0;
    for (int x = 0; x < side; x++) any |= r[y * side + x];
    if (!any) continue;
    printf("%3d ", y);
    for (int x = 0; x < side; x++) putchar(cell(r[y * side + x], shade, samples));
    putchar('\n');
  }
}

// The first and last row column `x` covers, or -1.
static void rows(int side, const uint8_t *r, int x, int *first, int *last) {
  *first = *last = -1;
  for (int y = 0; y < side; y++) {
    if (!r[y * side + x]) continue;
    if (*first < 0) *first = y;
    *last = y;
  }
}

static const float kWidths[] = {0, 0.5f, 1, 2, 3.5f, 8};

static void lane(void) {
  // The lane test's geometry, 16 x 16: two lines, and a strip of three vertices.
  static const float lines[] = {1, 3, 15, 3, 1, 8, 15, 15};
  static const float strip[] = {3.5f, 1, 3.5f, 9.5f, 13.5f, 14.5f};
  struct {
    const char *name;
    MTLPrimitiveType type;
    const float *points;
    int count;
  } draws[] = {
      {"lines", MTLPrimitiveTypeLine, lines, 4},
      {"strip", MTLPrimitiveTypeLineStrip, strip, 3},
  };
  for (int d = 0; d < 2; d++) {
    for (size_t w = 0; w < sizeof kWidths / sizeof kWidths[0]; w++) {
      uint8_t *r = one(16, 1, kWhite, draws[d].type, draws[d].points, draws[d].count, kWidths[w]);
      // Every row, so the lane test reads the whole target.
      printf("LANE %s %g\n", draws[d].name, kWidths[w]);
      for (int y = 0; y < 16; y++) {
        printf("  \"");
        for (int x = 0; x < 16; x++) putchar(cell(r[y * 16 + x], kWhite, 1));
        printf("\",\n");
      }
      free(r);
    }
  }
}

static void quantize(void) {
  static const float horizontal[] = {4, 32, 60, 32};
  static const float vertical[] = {16.5f, 4, 16.5f, 28};
  float widths[64];
  int n = 0;
  // Either side of 1 and of each multiple of 1/16 near 1 and 3.
  for (int k = 2; k <= 14; k++) {
    widths[n++] = 1.0f + ldexpf(1.0f, -k);
    widths[n++] = 1.0f - ldexpf(1.0f, -k);
  }
  float more[] = {3,       3.03125f, 3.0625f, 3.09375f, 2.9375f, 2.96875f, 15.9375f, 15.96875f,
                  16,      16.03125f, 16.0625f, 17,      100,     1e30f,    INFINITY, -0.0f,
                  -1,      -INFINITY, NAN,     0.03125f, 0.0625f, 0.09375f, 0.125f};
  for (size_t i = 0; i < sizeof more / sizeof more[0]; i++) widths[n++] = more[i];
  for (int i = 0; i < n; i++) {
    uint8_t *h = one(64, 1, kWhite, MTLPrimitiveTypeLine, horizontal, 2, widths[i]);
    uint8_t *v = one(32, 1, kWhite, MTLPrimitiveTypeLine, vertical, 2, widths[i]);
    int hf, hl, vf, vl;
    rows(64, h, 32, &hf, &hl);
    rows(32, v, 16, &vf, &vl);
    printf("QUANTIZE width %.9g (bits %08x): horizontal y=32 rows %d..%d, vertical x=16.5 rows "
           "%d..%d\n",
           widths[i], *(uint32_t *)&widths[i], hf, hl, vf, vl);
    free(h);
    free(v);
  }
}

static void state(void) {
  // Two horizontal lines, rows apart; each encoder draws them.
  static const float upper[] = {4, 8, 28, 8};
  static const float lower[] = {4, 20, 28, 20};
  // Encoder 0: width 4 set before the first draw, none before the second, which changes the
  // pipeline state. Encoder 1: no width set.
  Draw draws[] = {
      {MTLPrimitiveTypeLine, upper, 2, 1, 4, MTLTriangleFillModeFill, 0},
      {MTLPrimitiveTypeLine, lower, 2, 0, 0, MTLTriangleFillModeFill, 1},
  };
  int starts[2] = {0, 2};
  uint8_t *r = render(32, 1, kWhite, draws, starts, 1);
  print_map("width 4, then a draw of another pipeline state with no width set", 32, r, kWhite, 1);
  free(r);
  // A second encoder of the same command buffer, which sets nothing.
  static const float third[] = {4, 14, 28, 14};
  Draw two[] = {
      {MTLPrimitiveTypeLine, upper, 2, 1, 4, MTLTriangleFillModeFill, 0},
      {MTLPrimitiveTypeLine, third, 2, 0, 0, MTLTriangleFillModeFill, 0},
  };
  int starts2[3] = {0, 1, 2};
  r = render(32, 1, kWhite, two, starts2, 2);
  print_map("width 4 in encoder 0; encoder 1 sets none", 32, r, kWhite, 1);
  free(r);
}

static void ties(void) {
  struct {
    const char *name;
    int side;
    float points[4];
    float width;
  } cases[] = {
      {"horizontal y=16 width 3", 32, {4, 16, 28, 16}, 3},
      {"horizontal y=16 width 0.5", 32, {4, 16, 28, 16}, 0.5f},
      {"horizontal y=16.5 width 0.5", 32, {4, 16.5f, 28, 16.5f}, 0.5f},
      {"horizontal y=16 width 3.5", 32, {4, 16, 28, 16}, 3.5f},
      {"horizontal y=16.5 width 3.5", 32, {4, 16.5f, 28, 16.5f}, 3.5f},
      {"ends on centres x 4.5 to 27.5 width 2", 32, {4.5f, 10, 27.5f, 10}, 2},
      {"ends on centres x 27.5 to 4.5 width 2", 32, {27.5f, 10, 4.5f, 10}, 2},
      {"ends on centres y 4.5 to 27.5 width 2", 32, {10, 4.5f, 10, 27.5f}, 2},
      {"ends on centres y 27.5 to 4.5 width 2", 32, {10, 27.5f, 10, 4.5f}, 2},
      {"45 degrees down-right width 4", 24, {4, 4, 20, 20}, 4},
      {"45 degrees up-left width 4", 24, {20, 20, 4, 4}, 4},
      {"45 degrees up-right width 4", 24, {4, 20, 20, 4}, 4},
      {"x-major diagonal width 8", 32, {2, 6, 30, 20}, 8},
      {"y-major width 3.5", 32, {8, 4, 20, 28}, 3.5f},
  };
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    uint8_t *r = one(cases[i].side, 1, kWhite, MTLPrimitiveTypeLine, cases[i].points, 2,
                     cases[i].width);
    print_map(cases[i].name, cases[i].side, r, kWhite, 1);
    free(r);
  }
}

static void zero(void) {
  struct {
    const char *name;
    float points[4];
    float width;
  } cases[] = {
      {"horizontal y=16.5 x 4.5 to 27.5 width 0", {4.5f, 16.5f, 27.5f, 16.5f}, 0},
      {"horizontal y=16.5 x 27.5 to 4.5 width 0", {27.5f, 16.5f, 4.5f, 16.5f}, 0},
      {"vertical x=16.5 y 4.5 to 27.5 width 0", {16.5f, 4.5f, 16.5f, 27.5f}, 0},
      {"diagonal through centres width 0", {4.5f, 4.5f, 28.5f, 16.5f}, 0},
      {"diagonal through centres reversed width 0", {28.5f, 16.5f, 4.5f, 4.5f}, 0},
      {"45 degrees through centres width 0", {4.5f, 4.5f, 20.5f, 20.5f}, 0},
      {"diagonal through centres width -3", {4.5f, 4.5f, 28.5f, 16.5f}, -3},
      {"diagonal through centres width NaN", {4.5f, 4.5f, 28.5f, 16.5f}, NAN},
      {"diagonal through centres width 0.0625", {4.5f, 4.5f, 28.5f, 16.5f}, 0.0625f},
      {"horizontal y=16 width 0 (through no centre)", {4, 16, 28, 16}, 0},
      {"horizontal y=16.5 x 4 to 28 width 0", {4, 16.5f, 28, 16.5f}, 0},
  };
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    uint8_t *r = one(32, 1, kWhite, MTLPrimitiveTypeLine, cases[i].points, 2, cases[i].width);
    print_map(cases[i].name, 32, r, kWhite, 1);
    free(r);
  }
}

static void strips(void) {
  static const float corner[] = {4, 8.5f, 24.5f, 8.5f, 24.5f, 28};
  static const float peak[] = {4, 28, 16, 4, 28, 28};
  float widths[] = {1, 3.5f, 8, 0};
  for (size_t w = 0; w < sizeof widths / sizeof widths[0]; w++) {
    char title[96];
    uint8_t *r = one(32, 1, kCount, MTLPrimitiveTypeLineStrip, corner, 3, widths[w]);
    snprintf(title, sizeof title, "strip L width %g, fragments", widths[w]);
    print_map(title, 32, r, kCount, 1);
    free(r);
    r = one(32, 1, kCount, MTLPrimitiveTypeLineStrip, peak, 3, widths[w]);
    snprintf(title, sizeof title, "strip peak width %g, fragments", widths[w]);
    print_map(title, 32, r, kCount, 1);
    free(r);
  }
}

static void fill(void) {
  static const float triangle[] = {4, 4, 28, 4, 16, 28};
  struct {
    MTLTriangleFillMode fill;
    float width;
    const char *name;
  } cases[] = {
      {MTLTriangleFillModeLines, 1, "triangle fill Lines width 1, fragments"},
      {MTLTriangleFillModeLines, 4, "triangle fill Lines width 4, fragments"},
      {MTLTriangleFillModeLines, 0, "triangle fill Lines width 0, fragments"},
      {MTLTriangleFillModeFill, 4, "triangle fill Fill width 4, fragments"},
  };
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
    Draw d = {MTLPrimitiveTypeTriangle, triangle, 3, 1, cases[i].width, cases[i].fill, 0};
    int starts[2] = {0, 1};
    uint8_t *r = render(32, 1, kCount, &d, starts, 1);
    print_map(cases[i].name, 32, r, kCount, 1);
    free(r);
  }
}

static void msaa(void) {
  static const float horizontal[] = {4, 16, 28, 16};
  static const float diagonal[] = {4, 4, 28, 20};
  float widths[] = {0.5f, 1, 2};
  for (size_t w = 0; w < sizeof widths / sizeof widths[0]; w++) {
    char title[96];
    uint8_t *r = one(32, 4, kWhite, MTLPrimitiveTypeLine, horizontal, 2, widths[w]);
    snprintf(title, sizeof title, "4 samples horizontal y=16 width %g, samples", widths[w]);
    print_map(title, 32, r, kWhite, 4);
    free(r);
  }
  uint8_t *r = one(32, 4, kWhite, MTLPrimitiveTypeLine, diagonal, 2, 1);
  print_map("4 samples diagonal width 1, samples", 32, r, kWhite, 4);
  free(r);
}

int main(int argc, char **argv) {
  @autoreleasepool {
    struct {
      const char *name;
      void (*run)(void);
    } cases[] = {{"lane", lane},       {"quantize", quantize}, {"state", state},
                 {"ties", ties},       {"zero", zero},         {"strips", strips},
                 {"fill", fill},       {"msaa", msaa}};
    size_t n = sizeof cases / sizeof cases[0];
    if (argc < 2) {
      for (size_t i = 0; i < n; i++) printf("%s\n", cases[i].name);
      return 0;
    }
    gDevice = MTLCreateSystemDefaultDevice();
    NSError *error = nil;
    gLibrary = [gDevice newLibraryWithSource:kSource options:nil error:&error];
    if (!gLibrary) {
      fprintf(stderr, "library: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    gQueue = [gDevice newCommandQueue];
    printf("DEVICE %s\n", gDevice.name.UTF8String);
    for (size_t i = 0; i < n; i++) {
      if (strcmp(argv[1], cases[i].name) == 0) {
        cases[i].run();
        return 0;
      }
    }
    fprintf(stderr, "no case %s\n", argv[1]);
    return 2;
  }
}
