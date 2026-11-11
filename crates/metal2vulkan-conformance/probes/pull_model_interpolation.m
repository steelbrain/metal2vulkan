// What plain Metal returns for its pull-model interpolation (`interpolant<T, P>` fragment inputs
// and `interpolate_at_center`, `_centroid`, `_sample` and `_offset`), the reference the
// translator's lowering of them and the lane test `runtime_lane::pull_model_interpolation` are
// read against. It draws the owned fixture `pull_model_interpolants.metal` (its path the second
// argument), whose `pm_place` places each vertex at a clip-space position and hands it a value
// as every varying.
//
// One case a process, named on the command line:
//
//   lane      the lane test's draws, printed as `runtime_lane/pull_model_interpolation_m3_max.txt`
//             holds them: one triangle whose vertices have w 1, 2 and 4 over a 6 x 6 target, at 1
//             and 4 samples, through each function of the fixture and the samples and offsets
//             listed in `kLaneSamples` and `kLaneOffsets`;
//   origin    where each form reads in the pixel: a no-perspective varying equal to the pixel
//             coordinates (x right, y down from the target's upper-left corner) across a triangle
//             covering an 8 x 8 target, read at the center, the centroid, each sample, and a list
//             of offsets, past Metal's range and not finite included, at 1 and 4 samples;
//   grid      the location `interpolate_at_offset` reads for offsets swept in steps of 1/512 and
//             1/64 pixel (`pm_offset_sweep`): where it rounds an offset and to what;
//   samples   `interpolate_at_sample` at every index to 15 and past it, at 1, 2, 4 and 8 samples,
//             against the device's default sample positions;
//   centroid  the centroid each partly covered pixel of a triangle's edge reads at 4 samples,
//             with the samples the triangle covers there.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (the lane test's module docs and the translator's
// `lower::fragment::interpolate` docs state them too):
//   - `interpolate_at_offset` reads at the offset from the pixel's upper-left corner, (0.5, 0.5)
//     its center, as given: no range (1e9 extrapolates), no grid (every 1/512 step is a step;
//     0.999 reads at 0.999), a NaN or infinite offset NaN;
//   - `interpolate_at_sample(i)` reads sample `i mod n` at Vulkan's standard positions (2x: 0.75,
//     0.25; 4x: (0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)); at 1 sample every
//     index reads the center bit for bit; 8 samples are not supported;
//   - `interpolate_at_centroid` reads the center of a pixel the triangle covers whole, else its
//     lowest covered sample's position;
//   - `interpolate_at_offset(0.5, 0.5)` differs from `interpolate_at_center` in the last bits;
//   - only `pm_sample_id` runs once per sample at 4 samples.
//
// Run it WITHOUT MTL_DEBUG_LAYER, each case with the Mac's GPU to itself:
//
//   clang -fobjc-arc -framework Metal -framework Foundation -o pull_model_interpolation \
//     pull_model_interpolation.m
//   /usr/bin/env -u MTL_DEBUG_LAYER ./pull_model_interpolation lane \
//     ../../metal2vulkan-validation/fixtures/public/engine/pull_model_interpolants.metal

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLibrary;

// The value every attachment is cleared to: no interpolation of the draws' values reaches it.
static const float kClear = -1000.0f;

typedef struct {
  float v[4];
} F4;

// A draw's vertices: clip-space positions and the value each hands its varyings.
typedef struct {
  F4 at[3];
  F4 values[3];
} Triangle;

// The fragment function's arguments: buffer 0 and buffer 1's bytes.
typedef struct {
  const void *b0;
  size_t b0_len;
  const void *b1;
  size_t b1_len;
} Args;

static uint32_t bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

static id<MTLFunction> function(NSString *name) {
  id<MTLFunction> f = [gLibrary newFunctionWithName:name];
  if (!f) {
    fprintf(stderr, "no function %s\n", name.UTF8String);
    exit(1);
  }
  return f;
}

static id<MTLRenderPipelineState> pipeline(NSString *fragment, MTLPixelFormat format, int targets,
                                           NSUInteger samples) {
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  d.vertexFunction = function(@"pm_place");
  d.fragmentFunction = function(fragment);
  for (int i = 0; i < targets; i++) d.colorAttachments[i].pixelFormat = format;
  d.rasterSampleCount = samples;
  NSError *error = nil;
  id<MTLRenderPipelineState> state = [gDevice newRenderPipelineStateWithDescriptor:d error:&error];
  if (!state) {
    fprintf(stderr, "pipeline %s: %s\n", fragment.UTF8String, error.localizedDescription.UTF8String);
    exit(1);
  }
  return state;
}

// Draws `t` through `fragment` into `targets` attachments of `format`, `side` x `side` at
// `samples` samples, each cleared to `kClear`, and returns every sample of every attachment as
// floats: attachment-major, then row, column and sample.
static float *draw(NSString *fragment, MTLPixelFormat format, int targets, int side,
                   NSUInteger samples, const Triangle *t, Args args) {
  id<MTLRenderPipelineState> state = pipeline(fragment, format, targets, samples);
  MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                width:side
                                                                               height:side
                                                                            mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  td.storageMode = samples > 1 ? MTLStorageModePrivate : MTLStorageModeShared;
  td.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
  td.sampleCount = samples;
  id<MTLTexture> textures[8];
  MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
  for (int i = 0; i < targets; i++) {
    textures[i] = [gDevice newTextureWithDescriptor:td];
    rp.colorAttachments[i].texture = textures[i];
    rp.colorAttachments[i].loadAction = MTLLoadActionClear;
    rp.colorAttachments[i].clearColor = MTLClearColorMake(kClear, kClear, kClear, kClear);
    rp.colorAttachments[i].storeAction = MTLStoreActionStore;
  }
  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
  id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
  [enc setRenderPipelineState:state];
  [enc setVertexBytes:t->at length:sizeof t->at atIndex:0];
  [enc setVertexBytes:t->values length:sizeof t->values atIndex:1];
  if (args.b0) [enc setFragmentBytes:args.b0 length:args.b0_len atIndex:0];
  if (args.b1) [enc setFragmentBytes:args.b1 length:args.b1_len atIndex:1];
  [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
  [enc endEncoding];
  // A multisampled attachment's every sample out through the fixture's `pm_read_samples`; a
  // single-sample one is read from its shared storage.
  size_t per = (size_t)side * side * samples;
  id<MTLBuffer> out = [gDevice newBufferWithLength:per * 16 * targets
                                           options:MTLResourceStorageModeShared];
  if (samples > 1) {
    NSError *error = nil;
    id<MTLComputePipelineState> read =
        [gDevice newComputePipelineStateWithFunction:function(@"pm_read_samples") error:&error];
    if (!read) {
      fprintf(stderr, "pm_read_samples: %s\n", error.localizedDescription.UTF8String);
      exit(1);
    }
    for (int i = 0; i < targets; i++) {
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
      [ce setComputePipelineState:read];
      [ce setTexture:textures[i] atIndex:0];
      [ce setBuffer:out offset:per * 16 * i atIndex:0];
      [ce dispatchThreads:MTLSizeMake(side, side, samples)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [ce endEncoding];
    }
  }
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) {
    fprintf(stderr, "command buffer: %s\n", cb.error.localizedDescription.UTF8String);
    exit(1);
  }
  float *values = malloc(per * 16 * targets);
  if (samples > 1) {
    memcpy(values, out.contents, per * 16 * targets);
    return values;
  }
  int half = format == MTLPixelFormatRGBA16Float;
  for (int i = 0; i < targets; i++) {
    float *dst = values + 4 * per * i;
    if (!half) {
      [textures[i] getBytes:dst
                bytesPerRow:side * 16
                 fromRegion:MTLRegionMake2D(0, 0, side, side)
                mipmapLevel:0];
      continue;
    }
    __fp16 *h = malloc(per * 8);
    [textures[i] getBytes:h
              bytesPerRow:side * 8
               fromRegion:MTLRegionMake2D(0, 0, side, side)
              mipmapLevel:0];
    for (size_t k = 0; k < 4 * per; k++) dst[k] = (float)h[k];
    free(h);
  }
  return values;
}

// The float4 of attachment `a`, pixel (x, y), sample `s` of a `draw` result.
static const float *at(const float *v, int a, int side, NSUInteger samples, int x, int y, int s) {
  size_t per = (size_t)side * side * samples;
  return v + 4 * (per * a + ((size_t)y * side + x) * samples + s);
}

static int covered(const float *p) { return p[0] != kClear; }

// ---- lane ---------------------------------------------------------------------------------

#define LANE_SIDE 6

static const Triangle kLaneTriangle = {
    .at = {{{-0.9f, -0.85f, 0.0f, 1.0f}},
           {{0.95f * 2.0f, -0.6f * 2.0f, 0.0f, 2.0f}},
           {{-0.3f * 4.0f, 0.9f * 4.0f, 0.0f, 4.0f}}},
    .values = {{{1.0f, 0.0f, 0.0f, 3.0f}}, {{0.0f, 1.0f, 0.0f, -2.0f}}, {{0.0f, 0.0f, 1.0f, 0.5f}}},
};

// The samples `pm_sample` reads at 4 samples (at 1, sample 0 alone).
static const uint32_t kLaneSamples[] = {0, 1, 2, 3};
// The offsets `pm_offset` reads.
static const float kLaneOffsets[][2] = {
    {0.0f, 0.0f}, {0.5f, 0.5f}, {0.25f, 0.8125f}, {0.9375f, 0.0625f}, {0.40625f, 0.59375f},
};
// The sample and offset `pm_narrow` and `pm_half` read.
static const uint32_t kNarrowSample = 0;
static const float kNarrowOffset[2] = {0.25f, 0.75f};

// Prints a draw's covered pixels: `x y mask` and the attachments' words, the first covered
// sample's; and where a pixel's covered samples differ, each covered sample on its own line
// `x y s` and its words.
static void print_lane(const char *what, const float *v, int targets, NSUInteger samples) {
  printf("DRAW %s samples %lu\n", what, (unsigned long)samples);
  for (int y = 0; y < LANE_SIDE; y++) {
    for (int x = 0; x < LANE_SIDE; x++) {
      unsigned mask = 0;
      int first = -1, same = 1;
      for (NSUInteger s = 0; s < samples; s++) {
        const float *p = at(v, 0, LANE_SIDE, samples, x, y, (int)s);
        if (!covered(p)) continue;
        mask |= 1u << s;
        if (first < 0) {
          first = (int)s;
          continue;
        }
        for (int a = 0; a < targets; a++) {
          const float *q = at(v, a, LANE_SIDE, samples, x, y, first);
          const float *r = at(v, a, LANE_SIDE, samples, x, y, (int)s);
          if (memcmp(q, r, 16)) same = 0;
        }
      }
      if (!mask) continue;
      if (same) {
        printf("%d %d m%x", x, y, mask);
        for (int a = 0; a < targets; a++) {
          const float *p = at(v, a, LANE_SIDE, samples, x, y, first);
          for (int k = 0; k < 4; k++) printf(" %08x", bits(p[k]));
        }
        printf("\n");
        continue;
      }
      for (NSUInteger s = 0; s < samples; s++) {
        if (!(mask >> s & 1)) continue;
        printf("%d %d s%lu", x, y, (unsigned long)s);
        for (int a = 0; a < targets; a++) {
          const float *p = at(v, a, LANE_SIDE, samples, x, y, (int)s);
          for (int k = 0; k < 4; k++) printf(" %08x", bits(p[k]));
        }
        printf("\n");
      }
    }
  }
}

static void lane(void) {
  printf("# What plain Metal returned for crates/metal2vulkan-conformance/probes/"
         "pull_model_interpolation.m\n"
         "# (case lane) on %s, read by runtime_lane::pull_model_interpolation. Generated by the "
         "probe;\n# do not edit. Each covered pixel: `x y m<covered samples>` and its "
         "attachments' words,\n# or one line `x y s<sample>` per covered sample where they "
         "differ.\n",
         gDevice.name.UTF8String);
  const Triangle *t = &kLaneTriangle;
  const NSUInteger counts[2] = {1, 4};
  for (int c = 0; c < 2; c++) {
    NSUInteger n = counts[c];
    MTLPixelFormat f32 = MTLPixelFormatRGBA32Float;
    Args none = {0};
    float *v = draw(@"pm_center", f32, 2, LANE_SIDE, n, t, none);
    print_lane("center", v, 2, n);
    free(v);
    v = draw(@"pm_centroid", f32, 2, LANE_SIDE, n, t, none);
    print_lane("centroid", v, 2, n);
    free(v);
    for (NSUInteger i = 0; i < n; i++) {
      uint32_t s = kLaneSamples[i];
      Args a = {&s, 4, 0, 0};
      v = draw(@"pm_sample", f32, 2, LANE_SIDE, n, t, a);
      char what[64];
      snprintf(what, sizeof what, "sample %u", s);
      print_lane(what, v, 2, n);
      free(v);
    }
    v = draw(@"pm_sample_id", f32, 2, LANE_SIDE, n, t, none);
    print_lane("sample_id", v, 2, n);
    free(v);
    for (size_t i = 0; i < sizeof kLaneOffsets / sizeof kLaneOffsets[0]; i++) {
      Args a = {kLaneOffsets[i], 8, 0, 0};
      v = draw(@"pm_offset", f32, 2, LANE_SIDE, n, t, a);
      char what[64];
      snprintf(what, sizeof what, "offset %08x %08x", bits(kLaneOffsets[i][0]),
               bits(kLaneOffsets[i][1]));
      print_lane(what, v, 2, n);
      free(v);
    }
    Args narrow = {&kNarrowSample, 4, kNarrowOffset, 8};
    v = draw(@"pm_narrow", f32, 4, LANE_SIDE, n, t, narrow);
    print_lane("narrow", v, 4, n);
    free(v);
    v = draw(@"pm_half", MTLPixelFormatRGBA16Float, 4, LANE_SIDE, n, t, narrow);
    print_lane("half", v, 4, n);
    free(v);
  }
}

// ---- where each form reads -----------------------------------------------------------------

// A triangle covering a `side` x `side` target whose varyings are the pixel coordinates (x
// right, y down from the upper-left corner) at w = 1, so `n.xy` is where the read was made.
static Triangle pixel_triangle(int side) {
  Triangle t = {
      .at = {{{-1.0f, 1.0f, 0.0f, 1.0f}}, {{3.0f, 1.0f, 0.0f, 1.0f}}, {{-1.0f, -3.0f, 0.0f, 1.0f}}},
  };
  for (int i = 0; i < 3; i++) {
    float cx = t.at[i].v[0], cy = t.at[i].v[1];
    t.values[i] = (F4){{(cx + 1.0f) * 0.5f * side, (1.0f - cy) * 0.5f * side, 0.0f, 1.0f}};
  }
  return t;
}

// Where a read of pixel (x, y) of `v` was made, relative to the pixel's upper-left corner, and
// whether every covered pixel's read lies at the same place in its pixel (to 2^-12).
static void location(const float *v, int side, NSUInteger samples, int s, float *lx, float *ly,
                     int *uniform) {
  const float *p = at(v, 1, side, samples, 3, 4, s);
  *lx = p[0] - 3.0f;
  *ly = p[1] - 4.0f;
  *uniform = 1;
  for (int y = 0; y < side; y++)
    for (int x = 0; x < side; x++) {
      const float *q = at(v, 1, side, samples, x, y, s);
      if (!covered(q)) continue;
      if (fabsf((q[0] - x) - *lx) > 1.0f / 4096 || fabsf((q[1] - y) - *ly) > 1.0f / 4096)
        *uniform = 0;
    }
}

static void origin(void) {
  const int side = 8;
  Triangle t = pixel_triangle(side);
  const NSUInteger counts[2] = {1, 4};
  for (int c = 0; c < 2; c++) {
    NSUInteger n = counts[c];
    MTLPixelFormat f32 = MTLPixelFormatRGBA32Float;
    Args none = {0};
    float lx, ly;
    int u;
    float *v = draw(@"pm_center", f32, 2, side, n, &t, none);
    for (NSUInteger s = 0; s < n; s++) {
      location(v, side, n, (int)s, &lx, &ly, &u);
      printf("samples %lu center (sample %lu): %.6f %.6f%s\n", (unsigned long)n, (unsigned long)s,
             lx, ly, u ? "" : " (not the same in every pixel)");
    }
    free(v);
    v = draw(@"pm_centroid", f32, 2, side, n, &t, none);
    location(v, side, n, 0, &lx, &ly, &u);
    printf("samples %lu centroid (fully covered): %.6f %.6f%s\n", (unsigned long)n, lx, ly,
           u ? "" : " (not the same in every pixel)");
    free(v);
    v = draw(@"pm_sample_id", f32, 2, side, n, &t, none);
    for (NSUInteger s = 0; s < n; s++) {
      location(v, side, n, (int)s, &lx, &ly, &u);
      printf("samples %lu sample_id, sample %lu: %.6f %.6f%s\n", (unsigned long)n,
             (unsigned long)s, lx, ly, u ? "" : " (not the same in every pixel)");
    }
    free(v);
    const float offsets[][2] = {
        {0.0f, 0.0f},          {0.5f, 0.5f},       {0.25f, 0.75f},     {0.9375f, 0.9375f},
        {0.96875f, 0.96875f},  {0.999f, 0.999f},   {1.0f, 1.0f},       {1.5f, -0.5f},
        {-0.0625f, -0.0625f},  {-0.5f, -0.5f},     {-1.0f, 2.0f},      {7.0f, -7.0f},
        {-0.03125f, 0.03125f}, {0.46875f, 0.53125f}, {1e9f, -1e9f},    {INFINITY, -INFINITY},
        {NAN, 0.5f},           {0.5f, NAN},
    };
    for (size_t i = 0; i < sizeof offsets / sizeof offsets[0]; i++) {
      Args a = {offsets[i], 8, 0, 0};
      v = draw(@"pm_offset", f32, 2, side, n, &t, a);
      location(v, side, n, 0, &lx, &ly, &u);
      printf("samples %lu offset (%g, %g): %.6f %.6f%s\n", (unsigned long)n, offsets[i][0],
             offsets[i][1], lx, ly, u ? "" : " (not the same in every pixel)");
      free(v);
    }
  }
}

// One sweep of `pm_offset_sweep` along x (`axis` 0) or y (1): offsets `base + k step` at pixel
// k of a 64-pixel row or column; prints each run of offsets that read at the same location.
static void sweep(float base, float step, int axis) {
  const int side = 64;
  Triangle t = pixel_triangle(side);
  float sw[4] = {axis == 0 ? base : 0.25f, axis == 1 ? base : 0.25f, axis == 0 ? step : 0.0f,
                 axis == 1 ? step : 0.0f};
  Args a = {sw, 16, 0, 0};
  float *v = draw(@"pm_offset_sweep", MTLPixelFormatRGBA32Float, 2, side, 1, &t, a);
  printf("sweep %s from %g by %g:\n", axis ? "y" : "x", base, step);
  float run_from = 0, run_loc = NAN;
  for (int k = 0; k <= side; k++) {
    float loc = NAN, off = base + k * step;
    if (k < side) {
      int x = axis == 0 ? k : 5, y = axis == 1 ? k : 5;
      const float *p = at(v, 1, side, 1, x, y, 0);
      loc = axis == 0 ? p[0] - x : p[1] - y;
      loc = roundf(loc * 4096.0f) / 4096.0f;
    }
    if (k == side || loc != run_loc) {
      if (k > 0) printf("  [%.6f, %.6f] -> %.6f\n", run_from, base + (k - 1) * step, run_loc);
      run_from = off;
      run_loc = loc;
    }
  }
  free(v);
}

static void grid(void) {
  sweep(-2.0f, 1.0f / 16, 0);
  sweep(-0.125f, 1.0f / 512, 0);
  sweep(0.4375f, 1.0f / 512, 0);
  sweep(0.875f, 1.0f / 512, 0);
  sweep(-0.125f, 1.0f / 512, 1);
  sweep(0.875f, 1.0f / 512, 1);
}

static void samples_case(void) {
  const int side = 8;
  Triangle t = pixel_triangle(side);
  const NSUInteger counts[4] = {1, 2, 4, 8};
  for (int c = 0; c < 4; c++) {
    NSUInteger n = counts[c];
    if (![gDevice supportsTextureSampleCount:n]) {
      printf("samples %lu: not supported\n", (unsigned long)n);
      continue;
    }
    MTLSamplePosition pos[16];
    [gDevice getDefaultSamplePositions:pos count:n];
    printf("samples %lu default positions:", (unsigned long)n);
    for (NSUInteger s = 0; s < n; s++) printf(" (%.4f, %.4f)", pos[s].x, pos[s].y);
    printf("\n");
    const uint32_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 100, 0x7fffffffu,
                                0xffffffffu};
    for (size_t i = 0; i < sizeof indices / sizeof indices[0]; i++) {
      Args a = {&indices[i], 4, 0, 0};
      float *v = draw(@"pm_sample", MTLPixelFormatRGBA32Float, 2, side, n, &t, a);
      float lx, ly;
      int u;
      location(v, side, n, 0, &lx, &ly, &u);
      printf("samples %lu interpolate_at_sample(%u): %.6f %.6f%s\n", (unsigned long)n, indices[i],
             lx, ly, u ? "" : " (not the same in every pixel)");
      free(v);
    }
  }
}

static void centroid(void) {
  const int side = 8;
  // A triangle with a shallow edge crossing the target, so many pixels are partly covered.
  Triangle t = {
      .at = {{{-1.0f, 0.9f, 0.0f, 1.0f}}, {{1.0f, 0.13f, 0.0f, 1.0f}}, {{-1.0f, -1.0f, 0.0f, 1.0f}}},
  };
  for (int i = 0; i < 3; i++) {
    float cx = t.at[i].v[0], cy = t.at[i].v[1];
    t.values[i] = (F4){{(cx + 1.0f) * 0.5f * side, (1.0f - cy) * 0.5f * side, 0.0f, 1.0f}};
  }
  const NSUInteger counts[2] = {1, 4};
  for (int c = 0; c < 2; c++) {
    NSUInteger n = counts[c];
    Args none = {0};
    float *cen = draw(@"pm_centroid", MTLPixelFormatRGBA32Float, 2, side, n, &t, none);
    float *mid = draw(@"pm_center", MTLPixelFormatRGBA32Float, 2, side, n, &t, none);
    printf("samples %lu: pixel, covered samples, centroid location, center location\n",
           (unsigned long)n);
    for (int y = 0; y < side; y++)
      for (int x = 0; x < side; x++) {
        unsigned mask = 0;
        int first = -1;
        for (NSUInteger s = 0; s < n; s++)
          if (covered(at(cen, 0, side, n, x, y, (int)s))) {
            mask |= 1u << s;
            if (first < 0) first = (int)s;
          }
        if (!mask || mask == (1u << n) - 1) continue;
        const float *p = at(cen, 1, side, n, x, y, first);
        const float *q = at(mid, 1, side, n, x, y, first);
        printf("  (%d, %d) m%x: %.6f %.6f | %.6f %.6f\n", x, y, mask, p[0] - x, p[1] - y,
               q[0] - x, q[1] - y);
      }
    free(cen);
    free(mid);
  }
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc != 3) {
      fprintf(stderr, "usage: %s lane|origin|grid|samples|centroid <fixture.metal>\n", argv[0]);
      return 2;
    }
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue = [gDevice newCommandQueue];
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:@(argv[2])
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (!source) {
      fprintf(stderr, "%s: %s\n", argv[2], error.localizedDescription.UTF8String);
      return 1;
    }
    gLibrary = [gDevice newLibraryWithSource:source options:nil error:&error];
    if (!gLibrary) {
      fprintf(stderr, "library: %s\n", error.localizedDescription.UTF8String);
      return 1;
    }
    if (!gDevice.supportsPullModelInterpolation) {
      fprintf(stderr, "the device does not support pull-model interpolation\n");
      return 1;
    }
    const char *c = argv[1];
    if (!strcmp(c, "lane")) lane();
    else if (!strcmp(c, "origin")) origin();
    else if (!strcmp(c, "grid")) grid();
    else if (!strcmp(c, "samples")) samples_case();
    else if (!strcmp(c, "centroid")) centroid();
    else {
      fprintf(stderr, "unknown case %s\n", c);
      return 2;
    }
  }
  return 0;
}
