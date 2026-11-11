// What Metal's private logic operation (`-[MTLRenderPipelineDescriptor setLogicOperationEnabled:]`
// and `setLogicOperation:`, SPI) stores, against the rule `metal2vulkan`'s
// `pipeline/private_state.rs` states: the low four bits choose the operation in Direct3D's order,
// applied to the fragment's code and the stored code of each normalized and integer channel, the
// write mask kept after it. Each case draws into a 16 x 16 target whose texels hold random bytes
// and prints how many texels hold the rule's answer, and the first channels that do not.
//
//   formats   every operation into each normalized, packed and integer format below, one
//             fullscreen triangle of per-pixel colors: specials (NaN, infinities, signed zeros,
//             one half), values a ulp either side of 8- and 10-bit code ties, and random values;
//             integer outputs of random bits;
//   snorm     the codes -128 and -32768 stored and produced (RGBA8Snorm, RG16Snorm);
//   overlap   two fullscreen triangles of one draw, in primitive order;
//   masks     write masks of some channels;
//   msaa      4 samples: each sample's own stored code, a partial triangle's coverage;
//   tests     a discarding fragment, and the depth test over a Depth32Float_Stencil8 attachment;
//   repro     RGBA8Unorm over Depth32Float_Stencil8, XOR, (1, 1, 0, 0) over (64, 128, 191, 255).
//
// The rule's source code: a normalized channel's value clamped (a NaN 0) times its largest code,
// the exact product rounded to nearest; a packed 16-bit channel's value rounded to `half` first;
// an integer channel's low bits.
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation -o logic_operation \
//     logic_operation.m
//   /usr/bin/env -u MTL_DEBUG_LAYER ./logic_operation [all|formats|snorm|overlap|masks|msaa|tests|repro]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { W = 16, N = W * W };

static NSString *kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct V { float4 p [[position]]; uint prim [[flat]]; };\n"
     "vertex V vs(uint vid [[vertex_id]], const device float4 *pos [[buffer(0)]]) {\n"
     "  V o; o.p = pos[vid]; o.prim = vid / 3; return o;\n"
     "}\n"
     "static uint at(V in) { return in.prim * 256u + uint(in.p.y) * 16u + uint(in.p.x); }\n"
     "fragment float4 ff(V in [[stage_in]], const device float4 *c [[buffer(0)]]) {\n"
     "  return c[at(in)];\n"
     "}\n"
     "fragment uint4 fu(V in [[stage_in]], const device uint4 *c [[buffer(0)]]) {\n"
     "  return c[at(in)];\n"
     "}\n"
     "fragment int4 fi(V in [[stage_in]], const device int4 *c [[buffer(0)]]) {\n"
     "  return c[at(in)];\n"
     "}\n"
     "fragment float4 ff_discard(V in [[stage_in]], const device float4 *c [[buffer(0)]]) {\n"
     "  if (in.p.x < 8.0) discard_fragment();\n"
     "  return c[at(in)];\n"
     "}\n"
     "fragment float4 ff_sample(V in [[stage_in]], uint s [[sample_id]],\n"
     "                          const device float4 *c [[buffer(0)]]) {\n"
     "  return c[(uint(in.p.y) * 16u + uint(in.p.x)) * 4u + s];\n"
     "}\n"
     "fragment uint fu_cover() { return 1u; }\n"
     "kernel void read_ms(texture2d_ms<float, access::read> t [[texture(0)]],\n"
     "                    device float4 *o [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
     "  for (uint s = 0; s < 4; s++) o[(g.y * 16u + g.x) * 4u + s] = t.read(g, s);\n"
     "}\n"
     "kernel void read_ms_u(texture2d_ms<uint, access::read> t [[texture(0)]],\n"
     "                      device uint *o [[buffer(0)]], uint2 g [[thread_position_in_grid]]) {\n"
     "  for (uint s = 0; s < 4; s++) o[(g.y * 16u + g.x) * 4u + s] = t.read(g, s).x;\n"
     "}\n";

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLibrary;

static void set_private(id o, const char *name, unsigned long long v) {
  SEL s = sel_registerName(name);
  if (![o respondsToSelector:s]) {
    printf("NO SELECTOR %s\n", name);
    exit(2);
  }
  Method m = class_getInstanceMethod(object_getClass(o), s);
  char t[64];
  method_getArgumentType(m, 2, t, sizeof t);
  switch (t[0]) {
    case 'B': case 'c': case 'C': ((void (*)(id, SEL, BOOL))objc_msgSend)(o, s, v != 0); break;
    default: ((void (*)(id, SEL, unsigned long long))objc_msgSend)(o, s, v);
  }
}

typedef enum { UN, SN, UI, SI, HU } Kind;
typedef struct {
  int shift, bits;  // bits 0: the format lacks the channel
} Field;
typedef struct {
  const char *name;
  MTLPixelFormat pf;
  int bpp;
  Kind kind;
  Field f[4];  // red, green, blue, alpha
} Fmt;

static const Fmt kFormats[] = {
    {"RGBA8Unorm", MTLPixelFormatRGBA8Unorm, 4, UN, {{0, 8}, {8, 8}, {16, 8}, {24, 8}}},
    {"BGRA8Unorm", MTLPixelFormatBGRA8Unorm, 4, UN, {{16, 8}, {8, 8}, {0, 8}, {24, 8}}},
    {"R8Unorm", MTLPixelFormatR8Unorm, 1, UN, {{0, 8}, {0, 0}, {0, 0}, {0, 0}}},
    {"A8Unorm", MTLPixelFormatA8Unorm, 1, UN, {{0, 0}, {0, 0}, {0, 0}, {0, 8}}},
    {"R16Unorm", MTLPixelFormatR16Unorm, 2, UN, {{0, 16}, {0, 0}, {0, 0}, {0, 0}}},
    {"RGBA16Unorm", MTLPixelFormatRGBA16Unorm, 8, UN, {{0, 16}, {16, 16}, {32, 16}, {48, 16}}},
    {"RGB10A2Unorm", MTLPixelFormatRGB10A2Unorm, 4, UN, {{0, 10}, {10, 10}, {20, 10}, {30, 2}}},
    {"BGR10A2Unorm", MTLPixelFormatBGR10A2Unorm, 4, UN, {{20, 10}, {10, 10}, {0, 10}, {30, 2}}},
    {"RGBA8Snorm", MTLPixelFormatRGBA8Snorm, 4, SN, {{0, 8}, {8, 8}, {16, 8}, {24, 8}}},
    {"RG8Snorm", MTLPixelFormatRG8Snorm, 2, SN, {{0, 8}, {8, 8}, {0, 0}, {0, 0}}},
    {"RG16Snorm", MTLPixelFormatRG16Snorm, 4, SN, {{0, 16}, {16, 16}, {0, 0}, {0, 0}}},
    {"RGBA16Snorm", MTLPixelFormatRGBA16Snorm, 8, SN, {{0, 16}, {16, 16}, {32, 16}, {48, 16}}},
    {"B5G6R5Unorm", MTLPixelFormatB5G6R5Unorm, 2, HU, {{11, 5}, {5, 6}, {0, 5}, {0, 0}}},
    {"A1BGR5Unorm", MTLPixelFormatA1BGR5Unorm, 2, HU, {{11, 5}, {6, 5}, {1, 5}, {0, 1}}},
    {"ABGR4Unorm", MTLPixelFormatABGR4Unorm, 2, HU, {{12, 4}, {8, 4}, {4, 4}, {0, 4}}},
    {"BGR5A1Unorm", MTLPixelFormatBGR5A1Unorm, 2, HU, {{10, 5}, {5, 5}, {0, 5}, {15, 1}}},
    {"RGBA8Uint", MTLPixelFormatRGBA8Uint, 4, UI, {{0, 8}, {8, 8}, {16, 8}, {24, 8}}},
    {"RGBA8Sint", MTLPixelFormatRGBA8Sint, 4, SI, {{0, 8}, {8, 8}, {16, 8}, {24, 8}}},
    {"R16Uint", MTLPixelFormatR16Uint, 2, UI, {{0, 16}, {0, 0}, {0, 0}, {0, 0}}},
    {"RG16Sint", MTLPixelFormatRG16Sint, 4, SI, {{0, 16}, {16, 16}, {0, 0}, {0, 0}}},
    {"R32Uint", MTLPixelFormatR32Uint, 4, UI, {{0, 32}, {0, 0}, {0, 0}, {0, 0}}},
    {"R32Sint", MTLPixelFormatR32Sint, 4, SI, {{0, 32}, {0, 0}, {0, 0}, {0, 0}}},
    {"RGB10A2Uint", MTLPixelFormatRGB10A2Uint, 4, UI, {{0, 10}, {10, 10}, {20, 10}, {30, 2}}},
};

static const Fmt *fmt(const char *name) {
  for (size_t i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++)
    if (!strcmp(kFormats[i].name, name)) return &kFormats[i];
  printf("no format %s\n", name);
  exit(2);
}

static uint64_t gSeed = 0x2545F4914F6CDD1DULL;
static uint32_t rnd(void) {
  gSeed = gSeed * 6364136223846793005ULL + 1442695040888963407ULL;
  return (uint32_t)(gSeed >> 32);
}

// A float output lane for pixel `i`, lane `l`: specials, near-ties of 8- and 10-bit codes, and
// random values in [-1.25, 1.25].
static float value(int i, int l) {
  static const float specials[] = {0.0f, 1.0f, 0.5f, -0.0f, NAN, INFINITY, -INFINITY, -1.0f,
                                   -0.5f, 2.0f, 1e-8f, 0.25f, 0.75f, -2.0f, 1.0f / 255.0f,
                                   -1.0f / 127.0f};
  int k = (i * 4 + l) % 64;
  if (i < 16) return specials[(i + l * 5) % 16];
  if (i < 96) {
    int bits = (i & 1) ? 8 : 10;
    double max = (double)((1u << bits) - 1);
    double tie = ((double)(rnd() % (1u << bits)) + 0.5) / max;
    float t = (float)tie;
    switch (k % 3) {
      case 0: return nextafterf(t, 2.0f);
      case 1: return nextafterf(t, -2.0f);
      default: return t;
    }
  }
  return ((float)(rnd() % 1000001) / 1000000.0f) * 2.5f - 1.25f;
}

static uint32_t mask_of(int bits) { return bits >= 32 ? 0xFFFFFFFFu : (1u << bits) - 1; }

// The rule's source code of output lane `x` (float bits, or integer bits) in a channel.
static uint32_t src_code(Kind kind, int bits, uint32_t x) {
  float f;
  memcpy(&f, &x, 4);
  double v = f;
  switch (kind) {
    case UI:
    case SI:
      return x & mask_of(bits);
    case HU:
      v = (double)(_Float16)f;
      // fall through
    case UN: {
      if (isnan(v)) v = 0;
      v = v < 0 ? 0 : v > 1 ? 1 : v;
      return (uint32_t)rint(v * (double)mask_of(bits));
    }
    case SN: {
      if (isnan(v)) v = 0;
      v = v < -1 ? -1 : v > 1 ? 1 : v;
      int32_t c = (int32_t)rint(v * (double)((1u << (bits - 1)) - 1));
      return (uint32_t)c & mask_of(bits);
    }
  }
  return 0;
}

static uint32_t operate(int raw, uint32_t s, uint32_t d, int bits) {
  uint32_t r;
  switch (raw & 15) {
    case 0: r = 0; break;
    case 1: r = ~0u; break;
    case 2: r = s; break;
    case 3: r = ~s; break;
    case 4: r = d; break;
    case 5: r = ~d; break;
    case 6: r = s & d; break;
    case 7: r = ~(s & d); break;
    case 8: r = s | d; break;
    case 9: r = ~(s | d); break;
    case 10: r = s ^ d; break;
    case 11: r = ~(s ^ d); break;
    case 12: r = s & ~d; break;
    case 13: r = ~s & d; break;
    case 14: r = s | ~d; break;
    default: r = ~s | d; break;
  }
  return r & mask_of(bits);
}

static uint64_t word_at(const uint8_t *bytes, int bpp, int i) {
  uint64_t w = 0;
  memcpy(&w, bytes + (size_t)i * bpp, bpp);
  return w;
}

static uint32_t field(uint64_t w, Field f) { return (uint32_t)((w >> f.shift) & mask_of(f.bits)); }

// Metal's color write masks: red 8, green 4, blue 2, alpha 1.
static const int kMaskBit[4] = {8, 4, 2, 1};

// The texel the rule stores: each channel's operation of the source's code over the stored one,
// in primitive order over `prims` sources, the channels outside `wmask` kept.
static uint64_t expected(const Fmt *f, int raw, const uint32_t *src[], int prims, uint64_t stored,
                         int wmask) {
  uint64_t out = stored;
  for (int c = 0; c < 4; c++) {
    Field fd = f->f[c];
    if (!fd.bits || !(wmask & kMaskBit[c])) continue;
    uint32_t d = field(stored, fd);
    for (int p = 0; p < prims; p++) d = operate(raw, src_code(f->kind, fd.bits, src[p][c]), d, fd.bits);
    out &= ~((uint64_t)mask_of(fd.bits) << fd.shift);
    out |= (uint64_t)d << fd.shift;
  }
  return out;
}

static const char *kNames = "RGBA";
static int gCases, gCasesMatching;

// One case's check of `got` against the rule over `stored` (one texel a pixel); prints a summary
// line and the first mismatching channels.
static void check(const char *label, const Fmt *f, int raw, const uint32_t *srcs, int prims,
                  const uint8_t *stored, const uint8_t *got, int wmask, const uint8_t *keep) {
  int wrong = 0, shown = 0;
  for (int i = 0; i < N; i++) {
    const uint32_t *src[2] = {srcs + 4 * i, srcs + 4 * (N + i)};
    uint64_t s = word_at(stored, f->bpp, i), g = word_at(got, f->bpp, i);
    uint64_t want = keep && !keep[i] ? s : expected(f, raw, src, prims, s, wmask);
    if (g == want) continue;
    wrong++;
    for (int c = 0; c < 4 && shown < 6; c++) {
      Field fd = f->f[c];
      if (!fd.bits || field(g, fd) == field(want, fd)) continue;
      float fl;
      memcpy(&fl, &src[0][c], 4);
      printf("    texel %d %c: out %08x (%.9g) code %x, stored %x, got %x, rule %x\n", i,
             kNames[c], src[0][c], (double)fl, src_code(f->kind, fd.bits, src[0][c]),
             field(s, fd), field(g, fd), field(want, fd));
      shown++;
    }
  }
  gCases++;
  gCasesMatching += !wrong;
  printf("%s %s op %d mask %x: %d of %d texels hold the rule's answer\n", label, f->name, raw,
         wmask, N - wrong, N);
}

static id<MTLRenderPipelineState> pipeline(const char *ffn, MTLPixelFormat pf, int samples, int raw,
                                           int wmask, BOOL depth) {
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  d.vertexFunction = [gLibrary newFunctionWithName:@"vs"];
  d.fragmentFunction = [gLibrary newFunctionWithName:@(ffn)];
  d.colorAttachments[0].pixelFormat = pf;
  d.colorAttachments[0].writeMask = (MTLColorWriteMask)wmask;
  d.rasterSampleCount = samples;
  if (depth) {
    d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    d.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
  }
  set_private(d, "setOpenGLModeEnabled:", 1);
  if (raw >= 0) {
    set_private(d, "setLogicOperationEnabled:", 1);
    set_private(d, "setLogicOperation:", (unsigned long long)raw);
  }
  NSError *err = nil;
  id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:d error:&err];
  if (!ps) {
    printf("pipeline %s op %d: %s\n", ffn, raw, err.localizedDescription.UTF8String);
    exit(1);
  }
  return ps;
}

static id<MTLTexture> target(MTLPixelFormat pf, int samples) {
  MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                 width:W
                                                                                height:W
                                                                             mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  td.storageMode = samples > 1 ? MTLStorageModePrivate : MTLStorageModeShared;
  if (samples > 1) {
    td.textureType = MTLTextureType2DMultisample;
    td.sampleCount = samples;
  }
  return [gDevice newTextureWithDescriptor:td];
}

// Clip-space corners: one fullscreen triangle, twice over for two primitives.
static const float kFull[6][4] = {{-1, -1, 0.5f, 1}, {3, -1, 0.5f, 1}, {-1, 3, 0.5f, 1},
                                  {-1, -1, 0.5f, 1}, {3, -1, 0.5f, 1}, {-1, 3, 0.5f, 1}};
// A triangle with slanted edges over part of the target.
static const float kPart[3][4] = {{-0.93f, -0.81f, 0.5f, 1}, {0.87f, -0.47f, 0.5f, 1},
                                  {-0.31f, 0.91f, 0.5f, 1}};

static const char *fragment_of(Kind k) { return k == UI ? "fu" : k == SI ? "fi" : "ff"; }

// A draw of `prims` fullscreen primitives through `ps` into a 1-sample target of `f` holding
// `stored`, the per-pixel outputs `srcs`; the texels after it into `got`.
static void draw(const Fmt *f, id<MTLRenderPipelineState> ps, int prims, const uint32_t *srcs,
                 const uint8_t *stored, uint8_t *got, BOOL depth) {
  id<MTLTexture> t = target(f->pf, 1);
  [t replaceRegion:MTLRegionMake2D(0, 0, W, W) mipmapLevel:0 withBytes:stored bytesPerRow:W * f->bpp];
  id<MTLBuffer> pos = [gDevice newBufferWithBytes:kFull length:sizeof kFull options:0];
  id<MTLBuffer> col = [gDevice newBufferWithBytes:srcs length:2 * N * 16 options:0];
  MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = t;
  rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
  rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  id<MTLDepthStencilState> ds = nil;
  if (depth) {
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float_Stencil8
                                                           width:W
                                                          height:W
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> dt = [gDevice newTextureWithDescriptor:td];
    rp.depthAttachment.texture = dt;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = MTLStoreActionDontCare;
    rp.stencilAttachment.texture = dt;
    rp.stencilAttachment.loadAction = MTLLoadActionClear;
    rp.stencilAttachment.storeAction = MTLStoreActionDontCare;
    MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
    dsd.depthCompareFunction = MTLCompareFunctionLess;
    dsd.depthWriteEnabled = YES;
    ds = [gDevice newDepthStencilStateWithDescriptor:dsd];
  }
  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
  id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
  [e setRenderPipelineState:ps];
  if (ds) [e setDepthStencilState:ds];
  [e setVertexBuffer:pos offset:0 atIndex:0];
  [e setFragmentBuffer:col offset:0 atIndex:0];
  [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 * prims];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) {
    printf("command buffer: %s\n", cb.error.localizedDescription.UTF8String);
    exit(1);
  }
  [t getBytes:got bytesPerRow:W * f->bpp fromRegion:MTLRegionMake2D(0, 0, W, W) mipmapLevel:0];
}

// Random stored texels and outputs of `f`'s class, two primitives' worth.
static void fill(const Fmt *f, uint8_t *stored, uint32_t *srcs) {
  for (int i = 0; i < N * f->bpp; i++) stored[i] = (uint8_t)rnd();
  for (int i = 0; i < 2 * N; i++)
    for (int l = 0; l < 4; l++) {
      uint32_t x;
      if (f->kind == UI || f->kind == SI) {
        x = rnd();
        // Small values too, in and around the channel's range.
        if ((i + l) % 3 == 0) x = rnd() % 600 - 300;
      } else {
        float v = value(i % N, l);
        memcpy(&x, &v, 4);
      }
      srcs[4 * i + l] = x;
    }
}

static void one(const char *label, const Fmt *f, int raw, int prims, int wmask, BOOL depth) {
  uint8_t stored[N * 8], got[N * 8];
  uint32_t srcs[2 * N * 4];
  fill(f, stored, srcs);
  id<MTLRenderPipelineState> ps = pipeline(fragment_of(f->kind), f->pf, 1, raw, wmask, depth);
  draw(f, ps, prims, srcs, stored, got, depth);
  check(label, f, raw, srcs, prims, stored, got, wmask, NULL);
}

static void formats(void) {
  for (size_t i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++)
    for (int raw = 0; raw < 16; raw++) one("formats", &kFormats[i], raw, 1, 0xf, NO);
}

// The lowest signed code stored and produced: every stored byte -128 or one of its neighbours,
// and outputs whose codes reach it under the operations.
static void snorm(void) {
  const char *names[] = {"RGBA8Snorm", "RG16Snorm"};
  for (int n = 0; n < 2; n++) {
    const Fmt *f = fmt(names[n]);
    for (int raw = 0; raw < 16; raw++) {
      uint8_t stored[N * 8], got[N * 8];
      uint32_t srcs[2 * N * 4];
      fill(f, stored, srcs);
      for (int i = 0; i < N; i++)
        for (int c = 0; c < 4; c++) {
          Field fd = f->f[c];
          if (!fd.bits) continue;
          uint32_t lowest = 1u << (fd.bits - 1), codes[4] = {lowest, lowest + 1, lowest - 1, 1};
          uint64_t w = word_at(stored, f->bpp, i);
          w &= ~((uint64_t)mask_of(fd.bits) << fd.shift);
          w |= (uint64_t)codes[(i + c) % 4] << fd.shift;
          memcpy(stored + (size_t)i * f->bpp, &w, f->bpp);
          float outs[4] = {-1.0f, 1.0f / 127.0f, 1.0f, 0.0f};
          memcpy(&srcs[4 * i + c], &outs[(i / 4 + c) % 4], 4);
        }
      id<MTLRenderPipelineState> ps = pipeline("ff", f->pf, 1, raw, 0xf, NO);
      draw(f, ps, 1, srcs, stored, got, NO);
      check("snorm", f, raw, srcs, 1, stored, got, 0xf, NULL);
    }
  }
}

static void overlap(void) {
  const char *names[] = {"RGBA8Unorm", "RGBA8Uint", "RGBA8Snorm"};
  int ops[] = {10, 12, 13, 14, 15, 6, 8, 3};
  for (int n = 0; n < 3; n++)
    for (int o = 0; o < 8; o++) one("overlap", fmt(names[n]), ops[o], 2, 0xf, NO);
}

static void masks(void) {
  const char *names[] = {"RGBA8Unorm", "RGBA8Snorm", "RGBA8Uint", "RGB10A2Unorm", "BGRA8Unorm",
                         "B5G6R5Unorm"};
  int ms[] = {0x8, 0x5, 0xe, 0x0, 0x1, 0x6};
  int ops[] = {10, 13, 3};
  for (int n = 0; n < 6; n++)
    for (int m = 0; m < 6; m++)
      for (int o = 0; o < 3; o++) one("masks", fmt(names[n]), ops[o], 1, ms[m], NO);
}

static void tests(void) {
  const Fmt *f = fmt("RGBA8Unorm");
  uint8_t stored[N * 8], got[N * 8], keep[N];
  uint32_t srcs[2 * N * 4];
  fill(f, stored, srcs);
  id<MTLRenderPipelineState> ps = pipeline("ff_discard", f->pf, 1, 10, 0xf, NO);
  draw(f, ps, 1, srcs, stored, got, NO);
  for (int i = 0; i < N; i++) keep[i] = (i % W) >= 8;
  check("discard", f, 10, srcs, 1, stored, got, 0xf, keep);
  // The second primitive at z 0.5 fails the depth test the first one's passed (Less, written).
  fill(f, stored, srcs);
  ps = pipeline("ff", f->pf, 1, 10, 0xf, YES);
  draw(f, ps, 2, srcs, stored, got, YES);
  check("depth", f, 10, srcs, 1, stored, got, 0xf, NULL);
}

static void repro(void) {
  const Fmt *f = fmt("RGBA8Unorm");
  uint8_t stored[N * 4], got[N * 4];
  uint32_t srcs[2 * N * 4];
  float out[4] = {1, 1, 0, 0};
  for (int i = 0; i < N; i++) {
    uint8_t s[4] = {64, 128, 191, 255};
    memcpy(stored + 4 * i, s, 4);
    memcpy(srcs + 4 * i, out, 16);
  }
  id<MTLRenderPipelineState> ps = pipeline("ff", f->pf, 1, 10, 0xf, YES);
  draw(f, ps, 1, srcs, stored, got, YES);
  printf("repro texel 0: %d %d %d %d\n", got[0], got[1], got[2], got[3]);
  check("repro", f, 10, srcs, 1, stored, got, 0xf, NULL);
}

// 4 samples: the samples' stored codes from a per-sample draw, then the operation through a
// partial triangle, each covered sample against its own code; coverage from the same triangle's
// draw into an R8Uint target.
static void msaa(void) {
  const Fmt *f = fmt("RGBA8Unorm");
  int ops[] = {10, 12, 3, 13};
  for (int o = 0; o < 4; o++) {
    int raw = ops[o];
    id<MTLTexture> t = target(f->pf, 4), cover = target(MTLPixelFormatR8Uint, 4);
    float presets[N * 4][4];
    for (int i = 0; i < N * 4; i++)
      for (int l = 0; l < 4; l++) presets[i][l] = (float)(rnd() % 256) / 255.0f;
    uint32_t srcs[2 * N * 4];
    for (int i = 0; i < 2 * N * 4; i++) {
      float v = value(i / 4 % N, i % 4);
      memcpy(&srcs[i], &v, 4);
    }
    id<MTLBuffer> full = [gDevice newBufferWithBytes:kFull length:sizeof kFull options:0];
    id<MTLBuffer> part = [gDevice newBufferWithBytes:kPart length:sizeof kPart options:0];
    id<MTLBuffer> pre = [gDevice newBufferWithBytes:presets length:sizeof presets options:0];
    id<MTLBuffer> col = [gDevice newBufferWithBytes:srcs length:sizeof srcs options:0];
    id<MTLBuffer> before = [gDevice newBufferWithLength:N * 4 * 16 options:0];
    id<MTLBuffer> after = [gDevice newBufferWithLength:N * 4 * 16 options:0];
    id<MTLBuffer> covered = [gDevice newBufferWithLength:N * 4 * 4 options:0];
    NSError *err = nil;
    id<MTLComputePipelineState> rf = [gDevice
        newComputePipelineStateWithFunction:[gLibrary newFunctionWithName:@"read_ms"]
                                      error:&err];
    id<MTLComputePipelineState> ru = [gDevice
        newComputePipelineStateWithFunction:[gLibrary newFunctionWithName:@"read_ms_u"]
                                      error:&err];
    id<MTLRenderPipelineState> preset = pipeline("ff_sample", f->pf, 4, -1, 0xf, NO);
    id<MTLRenderPipelineState> op = pipeline("ff", f->pf, 4, raw, 0xf, NO);
    id<MTLRenderPipelineState> cov = pipeline("fu_cover", MTLPixelFormatR8Uint, 4, -1, 0xf, NO);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    void (^pass)(id<MTLTexture>, MTLLoadAction, id<MTLRenderPipelineState>, id<MTLBuffer>,
                 id<MTLBuffer>) = ^(id<MTLTexture> tex, MTLLoadAction load,
                                    id<MTLRenderPipelineState> ps, id<MTLBuffer> pos,
                                    id<MTLBuffer> c) {
      MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = tex;
      rp.colorAttachments[0].loadAction = load;
      rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
      [e setRenderPipelineState:ps];
      [e setVertexBuffer:pos offset:0 atIndex:0];
      if (c) [e setFragmentBuffer:c offset:0 atIndex:0];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [e endEncoding];
    };
    void (^read)(id<MTLTexture>, id<MTLComputePipelineState>, id<MTLBuffer>) =
        ^(id<MTLTexture> tex, id<MTLComputePipelineState> k, id<MTLBuffer> o) {
          id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder];
          [c setComputePipelineState:k];
          [c setTexture:tex atIndex:0];
          [c setBuffer:o offset:0 atIndex:0];
          [c dispatchThreads:MTLSizeMake(W, W, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
          [c endEncoding];
        };
    pass(t, MTLLoadActionClear, preset, full, pre);
    read(t, rf, before);
    pass(t, MTLLoadActionLoad, op, part, col);
    read(t, rf, after);
    pass(cover, MTLLoadActionClear, cov, part, nil);
    read(cover, ru, covered);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
      printf("msaa command buffer: %s\n", cb.error.localizedDescription.UTF8String);
      exit(1);
    }
    const float(*b)[4] = before.contents;
    const float(*a)[4] = after.contents;
    const uint32_t *cv = covered.contents;
    int wrong = 0, ncov = 0, shown = 0;
    for (int i = 0; i < N * 4; i++) {
      ncov += cv[i] != 0;
      for (int c = 0; c < 4; c++) {
        uint32_t d = (uint32_t)rint(b[i][c] * 255.0), g = (uint32_t)rint(a[i][c] * 255.0);
        uint32_t s = src_code(UN, 8, srcs[4 * (i / 4) + c]);
        uint32_t want = cv[i] ? operate(raw, s, d, 8) : d;
        if (g != want) {
          wrong++;
          if (shown++ < 6)
            printf("    pixel %d sample %d %c: covered %u code %x stored %x got %x rule %x\n",
                   i / 4, i % 4, kNames[c], cv[i], s, d, g, want);
        }
      }
    }
    gCases++;
    gCasesMatching += !wrong;
    printf("msaa RGBA8Unorm op %d: %d of %d sample channels hold the rule's answer (%d of %d "
           "samples covered)\n",
           raw, N * 16 - wrong, N * 16, ncov, N * 4);
  }
}

int main(int argc, char **argv) {
  @autoreleasepool {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char *which = argc > 1 ? argv[1] : "all";
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue = [gDevice newCommandQueue];
    NSError *err = nil;
    gLibrary = [gDevice newLibraryWithSource:kSource options:nil error:&err];
    if (!gLibrary) {
      printf("library: %s\n", err.localizedDescription.UTF8String);
      return 1;
    }
    NSOperatingSystemVersion v = [NSProcessInfo processInfo].operatingSystemVersion;
    printf("device %s, macOS %ld.%ld.%ld\n", gDevice.name.UTF8String, (long)v.majorVersion,
           (long)v.minorVersion, (long)v.patchVersion);
    int all = !strcmp(which, "all");
    if (all || !strcmp(which, "repro")) repro();
    if (all || !strcmp(which, "formats")) formats();
    if (all || !strcmp(which, "snorm")) snorm();
    if (all || !strcmp(which, "overlap")) overlap();
    if (all || !strcmp(which, "masks")) masks();
    if (all || !strcmp(which, "tests")) tests();
    if (all || !strcmp(which, "msaa")) msaa();
    printf("%d of %d cases hold the rule's answer in every texel\n", gCasesMatching, gCases);
  }
  return 0;
}
