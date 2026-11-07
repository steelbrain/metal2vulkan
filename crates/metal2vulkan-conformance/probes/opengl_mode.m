// What `-[MTLRenderPipelineDescriptor setOpenGLModeEnabled:]` (SPI) changes in what Metal draws,
// reflects and refuses: each draw made twice, the mode off and on, its records compared byte for
// byte. The reference `PrivateRenderState`'s docs (`metal2vulkan`'s `pipeline/descriptor.rs`) are
// read against.
//
// Every draw goes to a 16 x 16 target (`RGBA8Unorm` color cleared to zero, `Depth32Float` depth
// cleared to 1 under `Always`, written unless a case says not) by a vertex function that passes
// clip-space positions through, its vertex ID flat and a point size from a constant, and a
// fragment function that records, per pixel and sample, its position, its flat vertex ID, its
// facing, its sample ID and a count of fragments, and returns 1. A map prints one character a
// pixel: `.` none, else the count (summed over samples), the flat vertex ID, `F`/`B` the facing,
// or the covered samples of a resolved color.
//
//   api         whether the descriptor takes the selectors, their argument types and defaults;
//   reflection  the bindings of a pipeline whose stages read buffers, a texture and a sampler;
//   refusals    pipelines Metal refuses or makes: a fragment input no vertex output writes or of
//               another type, a float output to an integer attachment, a depth or stencil output
//               without its attachment, a vertex function returning void (rasterizing or not,
//               with a fragment function or none), a color output without a color attachment,
//               stage-in attributes without a vertex descriptor or missing from it;
//   draws       the draws below, each with the mode off and on:
//     depth       three bands at clip z 0.25, -0.5 and 1.5 (w 1): the depth stored, `[[position]]`;
//     face        a counter-clockwise and a clockwise triangle (y up): `[[front_facing]]`, under
//                 `twoSideEnabled` 0 and 1;
//     provoking   two triangles of a list, a strip of three: the flat vertex ID;
//     points      size 1 on pixel edges and centres, size 3 off them: coverage, `[[point_coord]]`,
//                 under `pointCoordLowerLeft` and `pointSizeOutputVS` 0 and 1;
//     lines       a list of four (x-major, on a pixel edge, through centres, vertical) and a
//                 strip of two: coverage and the flat vertex ID;
//     clip        a vertex function writing clip distance -1, under `clipDistanceEnableMask` 0xff
//                 and 0;
//     msaa        4 samples: each sample's position;
//     coverage    4 samples, resolved: `sampleCoverage` 0.5, inverted and not; `sampleMask` 0x3;
//     logic       `logicOperationEnabled` with XOR (10) over a cleared color;
//     dswrite     `depthStencilWriteDisabled` 0 and 1 under a depth-writing state;
//     made        what pipelines Metal makes only in the mode draw: a depth output without a
//                 depth attachment, a stencil output without a stencil attachment, and a vertex
//                 function returning void, rasterizing, without a fragment function and with one.
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (the same under MTL_DEBUG_LAYER=1, which reported
// nothing):
//
//   - The descriptor (`MTLRenderPipelineDescriptorInternal`) takes `setOpenGLModeEnabled:` (a
//     BOOL), default NO; a copy keeps it and `reset` clears it; setting it changes no other
//     private property. The argument types: `sampleMask`, `alphaTestFunction`, `logicOperation`
//     and `vertexAmplificationMode` unsigned 64-bit, `sampleCoverage` float,
//     `clipDistanceEnableMask` unsigned 8-bit, the two depth compare clamp masks and
//     `explicitVisibilityGroupID` unsigned 32-bit, the rest BOOL.
//   - Every draw is byte for byte the same with the mode on: the depth band at z 0.25 stores 0.25
//     and the bands at -0.5 and 1.5 are clipped; pixel (0, 0)'s `[[position]]` is (0.5, 0.5); a
//     triangle counter-clockwise with y up is back-facing and the other front-facing, under
//     `twoSideEnabled` 0 and 1; the flat input is each triangle's and each line's first vertex's;
//     points, lines and `[[point_coord]]` alike under `pointCoordLowerLeft` and
//     `pointSizeOutputVS` 0 and 1; a clip distance of -1 clips everything under
//     `clipDistanceEnableMask` 0xff and 0; the 4 sample positions of a pixel are (3/8, 1/8),
//     (7/8, 3/8), (1/8, 5/8), (5/8, 7/8), `[[position]]` its centre; `sampleCoverage` 0.5, inverted
//     or not, and `sampleMask` 0x3 keep 2 of 4 samples; XOR inverts the cleared color;
//     `depthStencilWriteDisabled` 1 leaves depth written.
//   - Reflection is the same with the mode on.
//   - Two refusals are lifted, each code 3 in domain `AGXMetalG15X_M1` with `userInfo` holding
//     only `NSLocalizedDescription` without the mode: "depthAttachmentPixelFormat is not valid and
//     shader writes to depth" (the state then draws its color, as without the depth output), and
//     "RasterizationEnabled is true but the vertex shader's return type is void", with or without
//     a fragment function (the state's draw runs the vertex function, its three stores landing,
//     and leaves the color as cleared). Every other answer stands: fragment inputs no vertex
//     output writes or of another type, a float output to an integer attachment (code 3 in the
//     same domain), stage-in attributes without a vertex descriptor ("Vertex function has input
//     attributes but no vertex descriptor was set.", `CompilerError` code 1) or missing from it
//     ("Vertex attribute b(1) is missing from the vertex descriptor", code 2); a stencil output
//     without a stencil attachment, a void vertex function not rasterizing and a color output
//     without a color attachment are made either way.
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation -o opengl_mode opengl_mode.m
//   /usr/bin/env -u MTL_DEBUG_LAYER ./opengl_mode all

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { W = 16, MAXS = 4 };

static NSString *kSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "struct V { float4 p [[position]]; uint vid [[flat]]; float ps [[point_size]]; };\n"
     "vertex V vs(uint vid [[vertex_id]], const device float4 *pos [[buffer(0)]],\n"
     "            constant float &psz [[buffer(1)]]) {\n"
     "  V o; o.p = pos[vid]; o.vid = vid; o.ps = psz; return o;\n"
     "}\n"
     "struct VC { float4 p [[position]]; uint vid [[flat]]; float ps [[point_size]];\n"
     "            float cd [[clip_distance]] [1]; };\n"
     "vertex VC vs_clip(uint vid [[vertex_id]], const device float4 *pos [[buffer(0)]],\n"
     "                  constant float &psz [[buffer(1)]]) {\n"
     "  VC o; o.p = pos[vid]; o.vid = vid; o.ps = psz; o.cd[0] = -1.0; return o;\n"
     "}\n"
     "static uint at(float4 p, uint s, uint S) {\n"
     "  return (min(uint(p.y), 15u) * 16u + min(uint(p.x), 15u)) * S + s;\n"
     "}\n"
     "fragment float4 fs(V in [[stage_in]], bool ff [[front_facing]], uint sid [[sample_id]],\n"
     "                   device float4 *rec [[buffer(0)]], device atomic_uint *cnt [[buffer(1)]],\n"
     "                   constant uint &S [[buffer(2)]], device uint4 *irec [[buffer(3)]]) {\n"
     "  uint i = at(in.p, sid, S);\n"
     "  rec[i] = in.p;\n"
     "  irec[i] = uint4(in.vid, ff ? 1u : 0u, sid, 1u);\n"
     "  atomic_fetch_add_explicit(&cnt[i], 1u, memory_order_relaxed);\n"
     "  return float4(1.0);\n"
     "}\n"
     "fragment float4 fs_point(V in [[stage_in]], float2 pc [[point_coord]],\n"
     "                         device float4 *rec [[buffer(0)]], device atomic_uint *cnt [[buffer(1)]],\n"
     "                         constant uint &S [[buffer(2)]], device uint4 *irec [[buffer(3)]]) {\n"
     "  uint i = at(in.p, 0u, S);\n"
     "  rec[i] = float4(in.p.xy, pc);\n"
     "  irec[i] = uint4(in.vid, 0u, 0u, 1u);\n"
     "  atomic_fetch_add_explicit(&cnt[i], 1u, memory_order_relaxed);\n"
     "  return float4(1.0);\n"
     "}\n"
     "fragment float4 fs_plain() { return float4(1.0); }\n"
     "struct VP { float4 p [[position]]; float2 px [[sample_perspective]]; };\n"
     "vertex VP vs_px(uint vid [[vertex_id]], const device float4 *pos [[buffer(0)]]) {\n"
     "  VP o; o.p = pos[vid];\n"
     "  o.px = float2(o.p.x / o.p.w + 1.0, 1.0 - o.p.y / o.p.w) * 8.0; return o;\n"
     "}\n"
     "fragment float4 fs_samples(VP in [[stage_in]], uint sid [[sample_id]],\n"
     "                           device float4 *rec [[buffer(0)]], device atomic_uint *cnt [[buffer(1)]],\n"
     "                           constant uint &S [[buffer(2)]], device uint4 *irec [[buffer(3)]]) {\n"
     "  uint i = at(in.p, sid, S);\n"
     "  rec[i] = float4(in.px, in.p.xy);\n"
     "  irec[i] = uint4(0u, 0u, sid, 1u);\n"
     "  atomic_fetch_add_explicit(&cnt[i], 1u, memory_order_relaxed);\n"
     "  return float4(1.0);\n"
     "}"
     // Refusal cases.
     "struct VE { float4 p [[position]]; float extra; };\n"
     "fragment float4 fs_extra(VE in [[stage_in]]) { return float4(in.extra); }\n"
     "struct VT { float4 p [[position]]; float2 vid; };\n"
     "fragment float4 fs_mistyped(VT in [[stage_in]]) { return float4(in.vid, 0.0, 1.0); }\n"
     "struct FD { float4 c [[color(0)]]; float d [[depth(any)]]; };\n"
     "fragment FD fs_depth() { FD o; o.c = float4(1.0); o.d = 0.5; return o; }\n"
     "vertex void vs_void(uint vid [[vertex_id]], device float *o [[buffer(0)]]) { o[vid] = 1.0; }\n"
     "struct FS { float4 c [[color(0)]]; uint s [[stencil]]; };\n"
     "fragment FS fs_stencil() { FS o; o.c = float4(1.0); o.s = 7u; return o; }\n"
     "struct A { float4 a [[attribute(0)]]; float4 b [[attribute(1)]]; };\n"
     "vertex V vs_attr(A in [[stage_in]]) { V o; o.p = in.a + in.b; o.vid = 0u; o.ps = 1.0; return o; }\n"
     // Reflection.
     "struct U { float4 a; uint b; };\n"
     "vertex V vs_refl(uint vid [[vertex_id]], const device float4 *pos [[buffer(0)]],\n"
     "                 constant float &psz [[buffer(1)]], constant U &u [[buffer(4)]]) {\n"
     "  V o; o.p = pos[vid] + u.a; o.vid = vid + u.b; o.ps = psz; return o;\n"
     "}\n"
     "fragment float4 fs_refl(V in [[stage_in]], texture2d<float> t [[texture(0)]],\n"
     "                        sampler s [[sampler(0)]], constant U &u [[buffer(2)]],\n"
     "                        device float *unused [[buffer(5)]]) {\n"
     "  return t.sample(s, in.p.xy) + u.a;\n"
     "}\n";

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLibrary;

// The argument type of `name`'s setter, or NULL where the object has no such selector.
static const char *arg_type(id o, const char *name) {
  SEL s = sel_registerName(name);
  if (![o respondsToSelector:s]) return NULL;
  Method m = class_getInstanceMethod(object_getClass(o), s);
  if (!m) return NULL;
  static char t[64];
  method_getArgumentType(m, 2, t, sizeof t);
  return t;
}

// Sets a private property by its setter, of whatever scalar type it takes.
static void set_private(id o, const char *name, double v) {
  SEL s = sel_registerName(name);
  const char *t = arg_type(o, name);
  if (!t) {
    printf("NO SELECTOR %s\n", name);
    exit(2);
  }
  switch (t[0]) {
    case 'f': ((void (*)(id, SEL, float))objc_msgSend)(o, s, (float)v); break;
    case 'd': ((void (*)(id, SEL, double))objc_msgSend)(o, s, v); break;
    case 'B': case 'c': case 'C': ((void (*)(id, SEL, BOOL))objc_msgSend)(o, s, v != 0); break;
    default: ((void (*)(id, SEL, unsigned long long))objc_msgSend)(o, s, (unsigned long long)v);
  }
}

typedef struct {
  const char *sel;
  double value;
} Set;

typedef struct {
  const char *name;
  const char *vfn, *ffn;
  int samples;
  MTLPrimitiveType prim;
  const float *pixels;  // x, y, z, w a vertex: x and y in pixels (y down), z and w clip-space
  int count;
  float point_size;
  Set sets[3];
  int nsets;
  int no_depth_write;
  float clear[4];
  int show;
  int no_depth_attachment;
} Draw;

enum { SHOW_FLAT = 1, SHOW_FACE = 2, SHOW_DEPTH = 4, SHOW_COLOR = 8, SHOW_SAMPLES = 16,
       SHOW_PC = 32, SHOW_Z = 64, SHOW_VERTEX = 128 };

typedef struct {
  char error[512];  // empty where the pipeline was made and ran
  float rec[W * W * MAXS][4];
  uint32_t irec[W * W * MAXS][4];
  uint32_t cnt[W * W * MAXS];
  uint8_t color[W * W * 4];
  float depth[W * W];
  int vertex_ones;  // of the first `count` floats of the position buffer, those equal to 1
} Out;

static MTLRenderPipelineDescriptor *descriptor(const char *vfn, const char *ffn, int samples,
                                               BOOL gl) {
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  d.vertexFunction = [gLibrary newFunctionWithName:@(vfn)];
  if (ffn) d.fragmentFunction = [gLibrary newFunctionWithName:@(ffn)];
  d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
  d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
  d.rasterSampleCount = samples;
  set_private(d, "setOpenGLModeEnabled:", gl);
  return d;
}

static id<MTLTexture> texture(MTLPixelFormat f, int samples, MTLStorageMode mode) {
  MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f
                                                                                 width:W
                                                                                height:W
                                                                             mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget;
  td.storageMode = mode;
  if (samples > 1) {
    td.textureType = MTLTextureType2DMultisample;
    td.sampleCount = samples;
  }
  return [gDevice newTextureWithDescriptor:td];
}

static void run(const Draw *dr, BOOL gl, Out *out) {
  memset(out, 0, sizeof *out);
  MTLRenderPipelineDescriptor *d = descriptor(dr->vfn, dr->ffn, dr->samples, gl);
  for (int i = 0; i < dr->nsets; i++) set_private(d, dr->sets[i].sel, dr->sets[i].value);
  if (dr->no_depth_attachment) d.depthAttachmentPixelFormat = MTLPixelFormatInvalid;
  NSError *err = nil;
  id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:d error:&err];
  if (!ps) {
    snprintf(out->error, sizeof out->error, "%s", err.localizedDescription.UTF8String);
    return;
  }
  int S = dr->samples;
  id<MTLTexture> color = texture(MTLPixelFormatRGBA8Unorm, S,
                                 S > 1 ? MTLStorageModePrivate : MTLStorageModeShared);
  id<MTLTexture> resolve = S > 1 ? texture(MTLPixelFormatRGBA8Unorm, 1, MTLStorageModeShared) : nil;
  id<MTLTexture> depth = texture(MTLPixelFormatDepth32Float, S, MTLStorageModePrivate);
  float clip[64][4];
  for (int i = 0; i < dr->count; i++) {
    const float *p = dr->pixels + 4 * i;
    clip[i][0] = p[0] / (W / 2.0f) - 1.0f;
    clip[i][1] = 1.0f - p[1] / (W / 2.0f);
    clip[i][2] = p[2];
    clip[i][3] = p[3];
    // Pixel positions are of w 1; scale x and y with w.
    clip[i][0] *= p[3];
    clip[i][1] *= p[3];
  }
  id<MTLBuffer> pos = [gDevice newBufferWithBytes:clip
                                           length:sizeof clip
                                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> rec = [gDevice newBufferWithLength:sizeof out->rec
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> irec = [gDevice newBufferWithLength:sizeof out->irec
                                            options:MTLResourceStorageModeShared];
  id<MTLBuffer> cnt = [gDevice newBufferWithLength:sizeof out->cnt
                                           options:MTLResourceStorageModeShared];
  id<MTLBuffer> dbuf = [gDevice newBufferWithLength:sizeof out->depth
                                            options:MTLResourceStorageModeShared];
  memset(rec.contents, 0, rec.length);
  memset(irec.contents, 0, irec.length);
  memset(cnt.contents, 0, cnt.length);
  memset(dbuf.contents, 0, dbuf.length);
  MTLDepthStencilDescriptor *dsd = [MTLDepthStencilDescriptor new];
  dsd.depthCompareFunction = MTLCompareFunctionAlways;
  dsd.depthWriteEnabled = !dr->no_depth_write;
  id<MTLDepthStencilState> ds = [gDevice newDepthStencilStateWithDescriptor:dsd];
  MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture = color;
  rp.colorAttachments[0].loadAction = MTLLoadActionClear;
  rp.colorAttachments[0].clearColor =
      MTLClearColorMake(dr->clear[0], dr->clear[1], dr->clear[2], dr->clear[3]);
  if (S > 1) {
    rp.colorAttachments[0].resolveTexture = resolve;
    rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
  } else {
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
  }
  if (!dr->no_depth_attachment) {
    rp.depthAttachment.texture = depth;
    rp.depthAttachment.loadAction = MTLLoadActionClear;
    rp.depthAttachment.clearDepth = 1.0;
    rp.depthAttachment.storeAction = S > 1 ? MTLStoreActionDontCare : MTLStoreActionStore;
  }
  id<MTLCommandBuffer> cb = [gQueue commandBuffer];
  id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:rp];
  [e setRenderPipelineState:ps];
  if (!dr->no_depth_attachment) [e setDepthStencilState:ds];
  [e setVertexBuffer:pos offset:0 atIndex:0];
  [e setVertexBytes:&dr->point_size length:sizeof(float) atIndex:1];
  uint32_t s32 = (uint32_t)S;
  [e setFragmentBuffer:rec offset:0 atIndex:0];
  [e setFragmentBuffer:cnt offset:0 atIndex:1];
  [e setFragmentBytes:&s32 length:sizeof s32 atIndex:2];
  [e setFragmentBuffer:irec offset:0 atIndex:3];
  [e drawPrimitives:dr->prim vertexStart:0 vertexCount:dr->count];
  [e endEncoding];
  if (S == 1 && !dr->no_depth_attachment) {
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:depth
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(W, W, 1)
                    toBuffer:dbuf
           destinationOffset:0
      destinationBytesPerRow:W * 4
    destinationBytesPerImage:W * W * 4];
    [b endEncoding];
  }
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) {
    snprintf(out->error, sizeof out->error, "command buffer: %s",
             cb.error.localizedDescription.UTF8String);
    return;
  }
  memcpy(out->rec, rec.contents, sizeof out->rec);
  memcpy(out->irec, irec.contents, sizeof out->irec);
  memcpy(out->cnt, cnt.contents, sizeof out->cnt);
  memcpy(out->depth, dbuf.contents, sizeof out->depth);
  for (int i = 0; i < dr->count; i++) out->vertex_ones += ((const float *)pos.contents)[i] == 1.0f;
  [(S > 1 ? resolve : color) getBytes:out->color
                          bytesPerRow:W * 4
                           fromRegion:MTLRegionMake2D(0, 0, W, W)
                          mipmapLevel:0];
}

static int same(const Out *a, const Out *b) {
  if (a->error[0] || b->error[0]) return !strcmp(a->error, b->error);
  return !memcmp(a->rec, b->rec, sizeof a->rec) && !memcmp(a->irec, b->irec, sizeof a->irec) &&
         !memcmp(a->cnt, b->cnt, sizeof a->cnt) && !memcmp(a->color, b->color, sizeof a->color) &&
         !memcmp(a->depth, b->depth, sizeof a->depth) && a->vertex_ones == b->vertex_ones;
}

static uint32_t covered(const Out *o, int px, int S) {
  uint32_t n = 0;
  for (int s = 0; s < S; s++) n += o->cnt[px * S + s];
  return n;
}

static void map(const char *title, const Out *o, int S, char (*cell)(const Out *, int, int)) {
  printf("    %s\n", title);
  for (int y = 0; y < W; y++) {
    char row[W + 1];
    for (int x = 0; x < W; x++) row[x] = cell(o, y * W + x, S);
    row[W] = 0;
    printf("      %s\n", row);
  }
}

static char count_cell(const Out *o, int px, int S) {
  uint32_t n = covered(o, px, S);
  return n == 0 ? '.' : n < 10 ? (char)('0' + n) : '+';
}

static char flat_cell(const Out *o, int px, int S) {
  if (!covered(o, px, S)) return '.';
  uint32_t v = o->irec[px * S][0];
  return v < 10 ? (char)('0' + v) : v < 36 ? (char)('a' + v - 10) : '?';
}

static char face_cell(const Out *o, int px, int S) {
  if (!covered(o, px, S)) return '.';
  return o->irec[px * S][1] ? 'F' : 'B';
}

static char color_cell(const Out *o, int px, int S) {
  (void)S;
  int r = o->color[px * 4];
  return (char)('0' + (r * 4 + 127) / 255);
}

static void show(const Draw *dr, const Out *o) {
  if (o->error[0]) {
    printf("    refused: %s\n", o->error);
    return;
  }
  int S = dr->samples;
  map("fragments", o, S, count_cell);
  if (dr->show & SHOW_FLAT) map("flat vertex ID", o, S, flat_cell);
  if (dr->show & SHOW_FACE) map("facing", o, S, face_cell);
  if (dr->show & SHOW_COLOR) {
    if (S > 1) {
      map("resolved red, in samples of 4", o, S, color_cell);
    } else {
      // The distinct colors.
      printf("    colors:");
      uint32_t seen[8];
      int n = 0;
      for (int i = 0; i < W * W; i++) {
        uint32_t c;
        memcpy(&c, o->color + 4 * i, 4);
        int k = 0;
        while (k < n && seen[k] != c) k++;
        if (k == n && n < 8) {
          seen[n++] = c;
          printf(" %02x%02x%02x%02x", o->color[4 * i], o->color[4 * i + 1], o->color[4 * i + 2],
                 o->color[4 * i + 3]);
        }
      }
      printf("\n");
    }
  }
  if (dr->show & SHOW_DEPTH) {
    printf("    stored depth by row:");
    for (int y = 0; y < W; y += 1) printf(" %g", o->depth[y * W + 8]);
    printf("\n");
  }
  if (dr->show & SHOW_Z) {
    printf("    [[position]] z, w by row:");
    for (int y = 0; y < W; y++) {
      int px = y * W + 8;
      if (covered(o, px, S))
        printf(" %d:(%g, %g)", y, o->rec[px * S][2], o->rec[px * S][3]);
    }
    printf("\n");
    printf("    [[position]] of pixel (0, 0) and (15, 15): ");
    for (int k = 0; k < 2; k++) {
      int px = k ? W * W - 1 : 0;
      if (covered(o, px, S))
        printf(" (%g, %g)", o->rec[px * S][0], o->rec[px * S][1]);
      else
        printf(" none");
    }
    printf("\n");
  }
  if (dr->show & SHOW_SAMPLES) {
    int px = 5 * W + 5;
    printf("    pixel (5, 5)'s samples (id: sample-interpolated pixel position, [[position]]):");
    for (int s = 0; s < S; s++)
      printf(" %u:(%g, %g; %g, %g)", o->irec[px * S + s][2], o->rec[px * S + s][0],
             o->rec[px * S + s][1], o->rec[px * S + s][2], o->rec[px * S + s][3]);
    printf("\n");
  }
  if (dr->show & SHOW_VERTEX)
    printf("    the vertex function's writes: %d of %d\n", o->vertex_ones, dr->count);
  if (dr->show & SHOW_PC) {
    printf("    [[point_coord]] by pixel:");
    for (int i = 0; i < W * W; i++)
      if (o->cnt[i * S])
        printf(" (%d,%d):(%g,%g)", i % W, i / W, o->rec[i * S][2], o->rec[i * S][3]);
    printf("\n");
  }
}

// Vertices in pixels (x, y down) with clip-space z and w.
#define P(x, y) x, y, 0.0f, 1.0f
#define PZ(x, y, z) x, y, z, 1.0f

static const float kBands[] = {
    PZ(0, 0, 0.25), PZ(16, 0, 0.25), PZ(0, 5, 0.25), PZ(16, 0, 0.25), PZ(16, 5, 0.25),
    PZ(0, 5, 0.25), PZ(0, 5, -0.5), PZ(16, 5, -0.5), PZ(0, 10, -0.5), PZ(16, 5, -0.5),
    PZ(16, 10, -0.5), PZ(0, 10, -0.5), PZ(0, 10, 1.5), PZ(16, 10, 1.5), PZ(0, 16, 1.5),
    PZ(16, 10, 1.5), PZ(16, 16, 1.5), PZ(0, 16, 1.5),
};
// Counter-clockwise with y up (left), then clockwise (right).
static const float kFaces[] = {
    P(0.5, 15), P(7, 15), P(4, 1), P(9, 15), P(12, 1), P(15.5, 15),
};
static const float kTwo[] = {
    P(0, 0), P(8, 0), P(0, 8), P(16, 8), P(16, 16), P(8, 16),
};
static const float kStrip[] = {
    P(0, 4), P(0, 12), P(5, 4), P(10, 12), P(16, 4),
};
static const float kLines[] = {
    P(0.5, 0.5), P(15.5, 6.5), P(2, 10), P(14, 10), P(3.5, 13.5), P(12.5, 13.5),
    P(14.5, 8), P(14.5, 15),
};
static const float kLineStrip[] = {
    P(1, 2), P(14, 7), P(3, 14),
};
static const float kPoints1[] = {
    P(4, 2), P(8.5, 2.5), P(12.25, 2.75), P(2.6, 6.4),
};
static const float kPoints3[] = {
    P(3.5, 10.5), P(8, 10), P(12.3, 9.7),
};
static const float kFull[] = {
    P(-16, -16), P(48, -16), P(-16, 48),
};
static const float kFullDepth[] = {
    PZ(-16, -16, 0.25), PZ(48, -16, 0.25), PZ(-16, 48, 0.25),
};

#define NOSET {{0, 0}}

static const Draw kDraws[] = {
    {"depth", "vs", "fs", 1, MTLPrimitiveTypeTriangle, kBands, 18, 1, NOSET, 0, 0, {0},
     SHOW_DEPTH | SHOW_Z},
    {"face", "vs", "fs", 1, MTLPrimitiveTypeTriangle, kFaces, 6, 1, NOSET, 0, 0, {0}, SHOW_FACE},
    {"face twoSideEnabled 1", "vs", "fs", 1, MTLPrimitiveTypeTriangle, kFaces, 6, 1,
     {{"setTwoSideEnabled:", 1}}, 1, 0, {0}, SHOW_FACE},
    {"provoking list", "vs", "fs", 1, MTLPrimitiveTypeTriangle, kTwo, 6, 1, NOSET, 0, 0, {0},
     SHOW_FLAT},
    {"provoking strip", "vs", "fs", 1, MTLPrimitiveTypeTriangleStrip, kStrip, 5, 1, NOSET, 0, 0,
     {0}, SHOW_FLAT},
    {"lines", "vs", "fs", 1, MTLPrimitiveTypeLine, kLines, 8, 1, NOSET, 0, 0, {0}, SHOW_FLAT},
    {"line strip", "vs", "fs", 1, MTLPrimitiveTypeLineStrip, kLineStrip, 3, 1, NOSET, 0, 0, {0},
     SHOW_FLAT},
    {"points size 1", "vs", "fs_point", 1, MTLPrimitiveTypePoint, kPoints1, 4, 1, NOSET, 0, 0,
     {0}, SHOW_PC},
    {"points size 3", "vs", "fs_point", 1, MTLPrimitiveTypePoint, kPoints3, 3, 3, NOSET, 0, 0,
     {0}, SHOW_PC},
    {"points size 3 pointCoordLowerLeft 1", "vs", "fs_point", 1, MTLPrimitiveTypePoint, kPoints3,
     3, 3, {{"setPointCoordLowerLeft:", 1}}, 1, 0, {0}, SHOW_PC},
    {"points size 3 pointSizeOutputVS 1", "vs", "fs_point", 1, MTLPrimitiveTypePoint, kPoints3, 3,
     3, {{"setPointSizeOutputVS:", 1}}, 1, 0, {0}, SHOW_PC},
    {"clip distance -1", "vs_clip", "fs", 1, MTLPrimitiveTypeTriangle, kFull, 3, 1, NOSET, 0, 0,
     {0}, 0},
    {"clip distance -1 clipDistanceEnableMask 0", "vs_clip", "fs", 1, MTLPrimitiveTypeTriangle,
     kFull, 3, 1, {{"setClipDistanceEnableMask:", 0}}, 1, 0, {0}, 0},
    {"msaa", "vs_px", "fs_samples", 4, MTLPrimitiveTypeTriangle, kFull, 3, 1, NOSET, 0, 0, {0},
     SHOW_SAMPLES},
    {"coverage 0.5", "vs", "fs_plain", 4, MTLPrimitiveTypeTriangle, kFull, 3, 1,
     {{"setSampleCoverage:", 0.5}}, 1, 0, {0}, SHOW_COLOR},
    {"coverage 0.5 inverted", "vs", "fs_plain", 4, MTLPrimitiveTypeTriangle, kFull, 3, 1,
     {{"setSampleCoverage:", 0.5}, {"setSampleCoverageInvert:", 1}}, 2, 0, {0}, SHOW_COLOR},
    {"sampleMask 0x3", "vs", "fs_plain", 4, MTLPrimitiveTypeTriangle, kFull, 3, 1,
     {{"setSampleMask:", 3}}, 1, 0, {0}, SHOW_COLOR},
    {"logic xor", "vs", "fs_plain", 1, MTLPrimitiveTypeTriangle, kFull, 3, 1,
     {{"setLogicOperationEnabled:", 1}, {"setLogicOperation:", 10}}, 2, 0,
     {15 / 255.0f, 240 / 255.0f, 85 / 255.0f, 170 / 255.0f}, SHOW_COLOR},
    {"dswrite 0", "vs", "fs", 1, MTLPrimitiveTypeTriangle, kFullDepth, 3, 1, NOSET, 0, 0, {0},
     SHOW_DEPTH},
    {"dswrite 1", "vs", "fs", 1, MTLPrimitiveTypeTriangle, kFullDepth, 3, 1,
     {{"setDepthStencilWriteDisabled:", 1}}, 1, 0, {0}, SHOW_DEPTH},
    // What the pipelines Metal makes only in the mode draw.
    {"a depth output without a depth attachment", "vs", "fs_depth", 1, MTLPrimitiveTypeTriangle,
     kFull, 3, 1, NOSET, 0, 0, {0}, SHOW_COLOR, 1},
    {"a stencil output without a stencil attachment", "vs", "fs_stencil", 1,
     MTLPrimitiveTypeTriangle, kFull, 3, 1, NOSET, 0, 0, {0}, SHOW_COLOR},
    {"a vertex function returning void, rasterizing, no fragment function", "vs_void", NULL, 1,
     MTLPrimitiveTypeTriangle, kFull, 3, 1, NOSET, 0, 0, {0}, SHOW_COLOR | SHOW_VERTEX},
    {"a vertex function returning void, rasterizing, a fragment function", "vs_void", "fs_plain",
     1, MTLPrimitiveTypeTriangle, kFull, 3, 1, NOSET, 0, 0, {0}, SHOW_COLOR | SHOW_VERTEX},
};

static void draws(void) {
  static Out off, on, again;
  for (size_t i = 0; i < sizeof kDraws / sizeof kDraws[0]; i++) {
    const Draw *dr = &kDraws[i];
    run(dr, NO, &off);
    run(dr, NO, &again);
    run(dr, YES, &on);
    printf("draw %s: %s%s\n", dr->name, same(&off, &on) ? "the same with the mode on" : "DIFFERS",
           same(&off, &again) ? "" : " (the mode off twice differs: not deterministic)");
    printf("  mode off\n");
    show(dr, &off);
    if (!same(&off, &on)) {
      printf("  mode on\n");
      show(dr, &on);
    }
  }
}

static const char *kPrivate[] = {
    "openGLModeEnabled", "sampleMask", "sampleCoverage", "sampleCoverageInvert",
    "alphaTestEnabled", "alphaTestFunction", "logicOperationEnabled", "logicOperation",
    "clipDistanceEnableMask", "pointSmoothEnabled", "pointCoordLowerLeft", "pointSizeOutputVS",
    "twoSideEnabled", "vertexDepthCompareClampMask", "fragmentDepthCompareClampMask",
    "depthStencilWriteDisabled", "needsCustomBorderColorSamplers", "vertexAmplificationMode",
    "explicitVisibilityGroupID",
};

// The value of property `name` of `o`, or nil where it has none.
static id value(id o, const char *name) {
  @try {
    return [o valueForKey:@(name)];
  } @catch (NSException *e) {
    return nil;
  }
}

static void api(void) {
  MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
  printf("descriptor class %s\n", object_getClassName(d));
  for (size_t i = 0; i < sizeof kPrivate / sizeof kPrivate[0]; i++) {
    const char *name = kPrivate[i];
    char setter[96];
    snprintf(setter, sizeof setter, "set%c%s:", name[0] - 32, name + 1);
    const char *t = arg_type(d, setter);
    id v = value(d, name);
    printf("  %s: getter %s, setter %s (%s), default %s\n", name,
           [d respondsToSelector:sel_registerName(name)] ? "yes" : "no", t ? "yes" : "no",
           t ? t : "-", v ? [v description].UTF8String : "-");
  }
  set_private(d, "setOpenGLModeEnabled:", 1);
  MTLRenderPipelineDescriptor *c = [d copy];
  printf("  a copy keeps openGLModeEnabled: %s\n",
         [value(c, "openGLModeEnabled") description].UTF8String);
  for (size_t i = 1; i < sizeof kPrivate / sizeof kPrivate[0]; i++) {
    id before = value([MTLRenderPipelineDescriptor new], kPrivate[i]);
    id after = value(d, kPrivate[i]);
    if (before != after && ![before isEqual:after])
      printf("  setting openGLModeEnabled changed %s: %s to %s\n", kPrivate[i],
             [before description].UTF8String, [after description].UTF8String);
  }
  [d reset];
  printf("  after reset: %s\n", [value(d, "openGLModeEnabled") description].UTF8String);
  for (NSString *line in [[c description] componentsSeparatedByString:@"\n"])
    if ([line rangeOfString:@"GL" options:NSCaseInsensitiveSearch].location != NSNotFound)
      printf("  description: %s\n", line.UTF8String);
}

static NSString *binding_text(NSArray<id<MTLBinding>> *bindings) {
  NSMutableArray *rows = [NSMutableArray array];
  for (id<MTLBinding> b in bindings) {
    NSMutableString *s = [NSMutableString
        stringWithFormat:@"%@ type %ld index %lu used %d argument %d access %ld", b.name,
                         (long)b.type, (unsigned long)b.index, b.used, b.argument, (long)b.access];
    if (b.type == MTLBindingTypeBuffer) {
      id<MTLBufferBinding> bb = (id<MTLBufferBinding>)b;
      [s appendFormat:@" size %lu align %lu dataType %lu", (unsigned long)bb.bufferDataSize,
                      (unsigned long)bb.bufferAlignment, (unsigned long)bb.bufferDataType];
    }
    [rows addObject:s];
  }
  return [rows componentsJoinedByString:@"\n      "];
}

static void reflection(void) {
  NSString *text[2];
  for (int gl = 0; gl < 2; gl++) {
    MTLRenderPipelineDescriptor *d = descriptor("vs_refl", "fs_refl", 1, gl);
    MTLRenderPipelineReflection *r = nil;
    NSError *err = nil;
    id<MTLRenderPipelineState> ps = [gDevice
        newRenderPipelineStateWithDescriptor:d
                                     options:MTLPipelineOptionBindingInfo |
                                             MTLPipelineOptionBufferTypeInfo
                                  reflection:&r
                                       error:&err];
    text[gl] = ps ? [NSString stringWithFormat:@"vertex\n      %@\n    fragment\n      %@",
                                               binding_text(r.vertexBindings),
                                               binding_text(r.fragmentBindings)]
                  : [@"refused: " stringByAppendingString:err.localizedDescription];
  }
  printf("reflection: %s\n", [text[0] isEqualToString:text[1]] ? "the same with the mode on"
                                                                : "DIFFERS");
  printf("  mode off\n    %s\n", text[0].UTF8String);
  if (![text[0] isEqualToString:text[1]]) printf("  mode on\n    %s\n", text[1].UTF8String);
}

typedef struct {
  const char *name;
  const char *vfn, *ffn;
  MTLPixelFormat color, depth;
  int rasterize;  // 2: rasterizing, with a vertex descriptor of attribute 0 alone
} Refusal;

static void refusals(void) {
  static const Refusal kCases[] = {
      {"a fragment input no vertex output writes", "vs", "fs_extra", MTLPixelFormatRGBA8Unorm,
       MTLPixelFormatDepth32Float, 1},
      {"a fragment input of another type", "vs", "fs_mistyped", MTLPixelFormatRGBA8Unorm,
       MTLPixelFormatDepth32Float, 1},
      {"a float output to an integer attachment", "vs", "fs_plain", MTLPixelFormatRGBA8Uint,
       MTLPixelFormatDepth32Float, 1},
      {"a depth output without a depth attachment", "vs", "fs_depth", MTLPixelFormatRGBA8Unorm,
       MTLPixelFormatInvalid, 1},
      {"a vertex function returning void, rasterizing", "vs_void", NULL, MTLPixelFormatRGBA8Unorm,
       MTLPixelFormatInvalid, 1},
      {"a vertex function returning void, not rasterizing", "vs_void", NULL,
       MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, 0},
      {"a color output without a color attachment", "vs", "fs_plain", MTLPixelFormatInvalid,
       MTLPixelFormatDepth32Float, 1},
      {"a stencil output without a stencil attachment", "vs", "fs_stencil",
       MTLPixelFormatRGBA8Unorm, MTLPixelFormatDepth32Float, 1},
      {"a vertex function returning void with a fragment function, rasterizing", "vs_void",
       "fs_plain", MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, 1},
      {"stage-in attributes without a vertex descriptor", "vs_attr", "fs_plain",
       MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, 1},
      {"a stage-in attribute its vertex descriptor lacks", "vs_attr", "fs_plain",
       MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, 2},
  };
  for (size_t i = 0; i < sizeof kCases / sizeof kCases[0]; i++) {
    const Refusal *c = &kCases[i];
    NSString *answer[2];
    for (int gl = 0; gl < 2; gl++) {
      MTLRenderPipelineDescriptor *d = descriptor(c->vfn, c->ffn, 1, gl);
      d.colorAttachments[0].pixelFormat = c->color;
      d.depthAttachmentPixelFormat = c->depth;
      d.rasterizationEnabled = c->rasterize != 0;
      if (c->rasterize == 2) {
        // Attribute 0 only, of the two the function reads.
        MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        vd.attributes[0].format = MTLVertexFormatFloat4;
        vd.attributes[0].bufferIndex = 0;
        vd.layouts[0].stride = 16;
        d.vertexDescriptor = vd;
      }
      NSError *err = nil;
      id<MTLRenderPipelineState> ps = [gDevice newRenderPipelineStateWithDescriptor:d error:&err];
      answer[gl] = ps ? @"made"
                      : [NSString stringWithFormat:@"refused: %@ (domain %@, code %ld, userInfo keys %@)",
                                                   err.localizedDescription, err.domain,
                                                   (long)err.code,
                                                   [err.userInfo.allKeys
                                                       componentsJoinedByString:@","]];
    }
    printf("refusal %s: %s\n  mode off: %s\n", c->name,
           [answer[0] isEqualToString:answer[1]] ? "the same with the mode on" : "DIFFERS",
           answer[0].UTF8String);
    if (![answer[0] isEqualToString:answer[1]]) printf("  mode on: %s\n", answer[1].UTF8String);
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
    if (all || !strcmp(which, "api")) api();
    if (all || !strcmp(which, "reflection")) reflection();
    if (all || !strcmp(which, "refusals")) refusals();
    if (all || !strcmp(which, "draws")) draws();
  }
  return 0;
}
