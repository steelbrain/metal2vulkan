// What plain Metal does with a blend into a padded 10-bit or a bfloat colour attachment, and with a
// second (dual-source) output to an extended-range one.
//
//   create <format> <state>   make the pipeline (blending on at colour 0) and print CREATED or
//                             REFUSED with Metal's words. One shape a process: a pipeline Metal's
//                             own validation asserts on ends the process.
//   bare <format> <state>     the same without a fragment function.
//   sweep <format>...         blend every state below into a 256 x 256 target of random raw texels,
//                             for a `half4` and a `float4` output, under fast and safe math, and
//                             write the inputs and the stored texels into out/ for the analysis.
//   tile <format>...          what a blend reads as its destination after a clear and after an
//                             earlier unblended draw of the same pass: the tile's `half`, or the
//                             attachment's codes.
//
// Formats: r10x6 r10x6srgb rg10x12 rg10x12srgb rgb10a2srgb (the padded forms), r16bf rg16bf
// rgba16bf (bfloat), bgra10xr bgr10xr (extended range, for the second output alone).
//
// Run it WITHOUT MTL_DEBUG_LAYER for plain Metal's answers; run `create` again under
// MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog for the validation layer's words.
//
//   clang -O2 -fobjc-arc -framework Metal -framework Foundation -o padded_and_bfloat_blends \
//       padded_and_bfloat_blends.m
//
// Nothing here can fault a GPU: single-sampled targets of the probe's own extent, one full-screen
// triangle a draw, no loop, and every fetch inside its buffer.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { kPadded, kBfloat, kXr };

typedef struct {
    const char *name;
    NSUInteger format;
    uint32_t bytes;
    int kind;
} Format;

static const Format kFormats[] = {
    {"r10x6", 576, 2, kPadded},      {"r10x6srgb", 577, 2, kPadded},
    {"rg10x12", 578, 4, kPadded},    {"rg10x12srgb", 579, 4, kPadded},
    {"rgb10a2srgb", 586, 4, kPadded}, {"r16bf", 26, 2, kBfloat},
    {"rg16bf", 66, 4, kBfloat},      {"rgba16bf", 116, 8, kBfloat},
    {"bgra10xr", 552, 8, kXr},       {"bgr10xr", 554, 4, kXr},
};

typedef struct {
    const char *name;
    MTLBlendFactor srgb, drgb;
    MTLBlendOperation oprgb;
    MTLBlendFactor sa, da;
    MTLBlendOperation opa;
    MTLColorWriteMask mask;
    BOOL dual;
} State;

#define F(x) MTLBlendFactor##x
#define O(x) MTLBlendOperation##x
#define M(x) MTLColorWriteMask##x
static const State kStates[] = {
    {"one-zero", F(One), F(Zero), O(Add), F(One), F(Zero), O(Add), M(All), NO},
    {"add", F(One), F(One), O(Add), F(One), F(One), O(Add), M(All), NO},
    {"premul", F(One), F(OneMinusSourceAlpha), O(Add), F(One), F(OneMinusSourceAlpha), O(Add),
     M(All), NO},
    {"straight", F(SourceAlpha), F(OneMinusSourceAlpha), O(Add), F(One), F(OneMinusSourceAlpha),
     O(Add), M(All), NO},
    {"subtract", F(One), F(One), O(Subtract), F(One), F(One), O(Subtract), M(All), NO},
    {"reverse", F(One), F(One), O(ReverseSubtract), F(One), F(One), O(ReverseSubtract), M(All),
     NO},
    {"min", F(One), F(One), O(Min), F(One), F(One), O(Min), M(All), NO},
    {"max", F(One), F(One), O(Max), F(One), F(One), O(Max), M(All), NO},
    {"modulate", F(DestinationColor), F(Zero), O(Add), F(DestinationAlpha), F(Zero), O(Add),
     M(All), NO},
    {"screen", F(One), F(OneMinusSourceColor), O(Add), F(One), F(OneMinusSourceAlpha), O(Add),
     M(All), NO},
    {"dest-alpha", F(DestinationAlpha), F(OneMinusDestinationAlpha), O(Add), F(OneMinusDestinationAlpha),
     F(One), O(Add), M(All), NO},
    {"saturated", F(SourceAlphaSaturated), F(One), O(Add), F(SourceAlphaSaturated), F(One), O(Add),
     M(All), NO},
    {"constant", F(BlendColor), F(OneMinusBlendColor), O(Add), F(BlendAlpha),
     F(OneMinusBlendAlpha), O(Add), M(All), NO},
    {"dual", F(Source1Color), F(OneMinusSource1Color), O(Add), F(Source1Alpha),
     F(OneMinusSource1Alpha), O(Add), M(All), YES},
    {"premul-red", F(One), F(OneMinusSourceAlpha), O(Add), F(One), F(OneMinusSourceAlpha), O(Add),
     M(Red), NO},
    {"premul-green-alpha", F(One), F(OneMinusSourceAlpha), O(Add), F(One), F(OneMinusSourceAlpha),
     O(Add), M(Green) | M(Alpha), NO},
    {"premul-rgb", F(One), F(OneMinusSourceAlpha), O(Add), F(One), F(OneMinusSourceAlpha), O(Add),
     M(Red) | M(Green) | M(Blue), NO},
    {"premul-none", F(One), F(OneMinusSourceAlpha), O(Add), F(One), F(OneMinusSourceAlpha), O(Add),
     M(None), NO},
};
#undef F
#undef O
#undef M

// The blend colour: past both ends of [0, 1] in two channels.
static const float kBlendColor[4] = {0.3f, -0.2f, 1.4f, 0.45f};

enum { W = 256, H = 256, N = W * H };

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct V { float4 p [[position]]; };\n"
    @"vertex V vs(uint id [[vertex_id]]) {\n"
    @"  float2 a = float2(float(id & 1u), float(id >> 1));\n"
    @"  V o; o.p = float4(4.0 * a - 1.0, 0.0, 1.0); return o;\n"
    @"}\n"
    @"static uint at(V in, uint w) { return uint(in.p.y) * w + uint(in.p.x); }\n"
    @"fragment half4 fh(V in [[stage_in]], const device half4 *c [[buffer(0)]],\n"
    @"                  constant uint &w [[buffer(1)]]) { return c[at(in, w)]; }\n"
    @"fragment float4 ff(V in [[stage_in]], const device float4 *c [[buffer(0)]],\n"
    @"                   constant uint &w [[buffer(1)]]) { return c[at(in, w)]; }\n"
    @"struct DH { half4 c [[color(0), index(0)]]; half4 s [[color(0), index(1)]]; };\n"
    @"struct DF { float4 c [[color(0), index(0)]]; float4 s [[color(0), index(1)]]; };\n"
    @"fragment DH dh(V in [[stage_in]], const device half4 *c [[buffer(0)]],\n"
    @"               constant uint &w [[buffer(1)]], const device half4 *s [[buffer(2)]]) {\n"
    @"  uint i = at(in, w); return DH { c[i], s[i] };\n"
    @"}\n"
    @"fragment DF df(V in [[stage_in]], const device float4 *c [[buffer(0)]],\n"
    @"               constant uint &w [[buffer(1)]], const device float4 *s [[buffer(2)]]) {\n"
    @"  uint i = at(in, w); return DF { c[i], s[i] };\n"
    @"}\n";

static uint64_t gState = 0x9e3779b97f4a7c15ull;
static uint64_t next(void) {
    gState ^= gState >> 12;
    gState ^= gState << 25;
    gState ^= gState >> 27;
    return gState * 2685821657736338717ull;
}
static uint32_t below(uint32_t n) { return (uint32_t)(next() % n); }

static const Format *formatNamed(const char *name) {
    for (size_t i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++) {
        if (strcmp(kFormats[i].name, name) == 0) return &kFormats[i];
    }
    return NULL;
}

static const State *stateNamed(const char *name) {
    for (size_t i = 0; i < sizeof kStates / sizeof kStates[0]; i++) {
        if (strcmp(kStates[i].name, name) == 0) return &kStates[i];
    }
    return NULL;
}

// One source channel: specials, exact codes and midpoints of a 10-bit unorm code, and values
// spread past both ends of [0, 1]. `asHalf` rounds it to a `half` (nearest even).
static float sourceValue(BOOL asHalf) {
    static const float kSpecials[] = {NAN, INFINITY, -INFINITY, -0.0f, 0.0f, 1.0f, 2.0f, -1.0f,
                                      65504.0f, -65504.0f, 5.9604645e-8f, 0.5f, 1.5f, -0.5f};
    uint32_t r = below(100);
    float v;
    if (r < 8) {
        v = kSpecials[below(sizeof kSpecials / sizeof kSpecials[0])];
    } else if (r < 20) {
        v = (float)below(1024) / 1023.0f;
    } else if (r < 30) {
        v = ((float)below(1023) + 0.5f) / 1023.0f;
    } else {
        v = -0.3f + 1.6f * (float)(next() >> 40) / (float)(1u << 24);
    }
    return asHalf ? (float)(_Float16)v : v;
}

// A random raw texel of `f`: random codes with random low bits; bfloat codes of random
// `float`s, specials among them.
static void destinationTexel(const Format *f, uint8_t *out) {
    switch (f->kind) {
        case kPadded:
        case kXr:
            for (uint32_t i = 0; i < f->bytes; i += 4) {
                uint32_t w = (uint32_t)next();
                memcpy(out + i, &w, f->bytes - i < 4 ? f->bytes - i : 4);
            }
            break;
        case kBfloat:
            for (uint32_t i = 0; i < f->bytes; i += 2) {
                float v = sourceValue(NO);
                uint32_t b;
                memcpy(&b, &v, 4);
                uint16_t code = (uint16_t)(b >> 16);
                if (below(10) == 0) code = (uint16_t)next();
                memcpy(out + i, &code, 2);
            }
            break;
    }
}

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLibrary[2];  // fast, safe

static void setup(void) {
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue = [gDevice newCommandQueue];
    for (int i = 0; i < 2; i++) {
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = i == 0 ? MTLMathModeFast : MTLMathModeSafe;
        NSError *error = nil;
        gLibrary[i] = [gDevice newLibraryWithSource:kSource options:options error:&error];
        if (!gLibrary[i]) {
            printf("LIBRARY %s\n", error.localizedDescription.UTF8String);
            exit(2);
        }
    }
}

static id<MTLRenderPipelineState> pipeline(const Format *f, const State *s, BOOL blended,
                                           BOOL halfOut, int math, NSString **why) {
    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [gLibrary[math] newFunctionWithName:@"vs"];
    NSString *name = s->dual ? (halfOut ? @"dh" : @"df") : (halfOut ? @"fh" : @"ff");
    rp.fragmentFunction = [gLibrary[math] newFunctionWithName:name];
    MTLRenderPipelineColorAttachmentDescriptor *c = rp.colorAttachments[0];
    c.pixelFormat = (MTLPixelFormat)f->format;
    c.blendingEnabled = blended;
    c.sourceRGBBlendFactor = s->srgb;
    c.destinationRGBBlendFactor = s->drgb;
    c.rgbBlendOperation = s->oprgb;
    c.sourceAlphaBlendFactor = s->sa;
    c.destinationAlphaBlendFactor = s->da;
    c.alphaBlendOperation = s->opa;
    c.writeMask = s->mask;
    NSError *error = nil;
    id<MTLRenderPipelineState> p = [gDevice newRenderPipelineStateWithDescriptor:rp error:&error];
    if (!p) *why = error.localizedDescription ?: @"(no description)";
    return p;
}

static BOOL writeFile(NSString *path, const void *bytes, size_t length) {
    NSData *d = [NSData dataWithBytes:bytes length:length];
    return [d writeToFile:path atomically:NO];
}

static id<MTLTexture> target(const Format *f) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = (MTLPixelFormat)f->format;
    td.width = W;
    td.height = H;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    return [gDevice newTextureWithDescriptor:td];
}

// A blit of `bytes` into the whole of `t` (a host copy is a silent no-op where Metal keeps a padded
// texture in its compressed layout, so every texel goes in and out by a blit).
static void upload(id<MTLCommandBuffer> cb, id<MTLTexture> t, const Format *f, id<MTLBuffer> bytes) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:bytes
               sourceOffset:0
          sourceBytesPerRow:W * f->bytes
        sourceBytesPerImage:(NSUInteger)N * f->bytes
                 sourceSize:MTLSizeMake(W, H, 1)
                  toTexture:t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
}

// A blit of the whole of `t` into `bytes`.
static void download(id<MTLCommandBuffer> cb, id<MTLTexture> t, const Format *f, id<MTLBuffer> bytes) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:t
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(W, H, 1)
                        toBuffer:bytes
               destinationOffset:0
          destinationBytesPerRow:W * f->bytes
        destinationBytesPerImage:(NSUInteger)N * f->bytes];
    [blit endEncoding];
}

// One draw of `values` (and `seconds`) under `p` over the texels `dst`; the stored texels into
// `out`. NO and the reason where the draw did not complete.
static BOOL draw(const Format *f, id<MTLRenderPipelineState> p, const State *s, id<MTLBuffer> values,
                 id<MTLBuffer> seconds, const uint8_t *dst, uint8_t *out) {
    id<MTLTexture> t = target(f);
    id<MTLBuffer> in = [gDevice newBufferWithBytes:dst length:(NSUInteger)N * f->bytes options:0];
    id<MTLBuffer> back = [gDevice newBufferWithLength:(NSUInteger)N * f->bytes options:0];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = t;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    upload(cb, t, f, in);
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
    uint32_t width = W;
    [enc setRenderPipelineState:p];
    [enc setBlendColorRed:kBlendColor[0]
                    green:kBlendColor[1]
                     blue:kBlendColor[2]
                    alpha:kBlendColor[3]];
    [enc setFragmentBuffer:values offset:0 atIndex:0];
    [enc setFragmentBytes:&width length:sizeof width atIndex:1];
    if (s->dual) [enc setFragmentBuffer:seconds offset:0 atIndex:2];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    download(cb, t, f, back);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("DRAW-FAILED %s %s %s\n", f->name, s->name,
               cb.error.localizedDescription.UTF8String);
        return NO;
    }
    memcpy(out, back.contents, (size_t)N * f->bytes);
    return YES;
}

// The control that the blits move the texels: `dst` up and straight back.
static BOOL roundTrip(const Format *f, const uint8_t *dst) {
    id<MTLTexture> t = target(f);
    id<MTLBuffer> in = [gDevice newBufferWithBytes:dst length:(NSUInteger)N * f->bytes options:0];
    id<MTLBuffer> back = [gDevice newBufferWithLength:(NSUInteger)N * f->bytes options:0];
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    upload(cb, t, f, in);
    download(cb, t, f, back);
    [cb commit];
    [cb waitUntilCompleted];
    return memcmp(back.contents, dst, (size_t)N * f->bytes) == 0;
}

// `create` without a fragment function: the blend is the descriptor's alone.
static int bare(const char *fname, const char *sname) {
    const Format *f = formatNamed(fname);
    const State *s = stateNamed(sname);
    if (!f || !s) return 2;
    setup();
    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [gLibrary[0] newFunctionWithName:@"vs"];
    MTLRenderPipelineColorAttachmentDescriptor *c = rp.colorAttachments[0];
    c.pixelFormat = (MTLPixelFormat)f->format;
    c.blendingEnabled = YES;
    c.sourceRGBBlendFactor = s->srgb;
    c.destinationRGBBlendFactor = s->drgb;
    c.rgbBlendOperation = s->oprgb;
    c.sourceAlphaBlendFactor = s->sa;
    c.destinationAlphaBlendFactor = s->da;
    c.alphaBlendOperation = s->opa;
    c.writeMask = s->mask;
    NSError *error = nil;
    id<MTLRenderPipelineState> p = [gDevice newRenderPipelineStateWithDescriptor:rp error:&error];
    printf("%s %s %s no-fragment\n", p ? "CREATED" : "REFUSED", f->name, s->name);
    if (!p) printf("  %s\n", error.localizedDescription.UTF8String);
    return 0;
}

static int create(const char *fname, const char *sname) {
    const Format *f = formatNamed(fname);
    const State *s = stateNamed(sname);
    if (!f || !s) return 2;
    setup();
    for (int halfOut = 1; halfOut >= 0; halfOut--) {
        NSString *why = nil;
        id<MTLRenderPipelineState> p = pipeline(f, s, YES, halfOut, 0, &why);
        printf("%s %s %s %s\n", p ? "CREATED" : "REFUSED", f->name, s->name,
               halfOut ? "half4" : "float4");
        if (!p) printf("  %s\n", why.UTF8String);
        fflush(stdout);
    }
    return 0;
}

static int sweep(int argc, char **argv) {
    setup();
    [[NSFileManager defaultManager] createDirectoryAtPath:@"out"
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    // The sources: `half4`s and `float4`s, and the second outputs, one draw's worth, the same for
    // every format and state.
    float *srcH = malloc(N * 16), *srcF = malloc(N * 16), *secH = malloc(N * 16),
          *secF = malloc(N * 16);
    _Float16 *bufH = malloc(N * 8), *bufSH = malloc(N * 8);
    for (int i = 0; i < N * 4; i++) {
        srcH[i] = sourceValue(YES);
        srcF[i] = sourceValue(NO);
        secH[i] = sourceValue(YES);
        secF[i] = sourceValue(NO);
        bufH[i] = (_Float16)srcH[i];
        bufSH[i] = (_Float16)secH[i];
    }
    writeFile(@"out/src-half.bin", bufH, N * 8);
    writeFile(@"out/src-float.bin", srcF, N * 16);
    writeFile(@"out/second-half.bin", bufSH, N * 8);
    writeFile(@"out/second-float.bin", secF, N * 16);
    id<MTLBuffer> vh = [gDevice newBufferWithBytes:bufH length:N * 8 options:0];
    id<MTLBuffer> vf = [gDevice newBufferWithBytes:srcF length:N * 16 options:0];
    id<MTLBuffer> sh = [gDevice newBufferWithBytes:bufSH length:N * 8 options:0];
    id<MTLBuffer> sf = [gDevice newBufferWithBytes:secF length:N * 16 options:0];
    for (int a = 0; a < argc; a++) {
        const Format *f = formatNamed(argv[a]);
        if (!f) {
            printf("UNKNOWN %s\n", argv[a]);
            continue;
        }
        uint8_t *dst = malloc((size_t)N * f->bytes), *out = malloc((size_t)N * f->bytes);
        for (int i = 0; i < N; i++) destinationTexel(f, dst + (size_t)i * f->bytes);
        writeFile([NSString stringWithFormat:@"out/%s-dst.bin", f->name], dst,
                  (size_t)N * f->bytes);
        printf("ROUND-TRIP %s %s\n", f->name, roundTrip(f, dst) ? "exact" : "DIFFERS");
        for (size_t k = 0; k < sizeof kStates / sizeof kStates[0]; k++) {
            const State *s = &kStates[k];
            // The extended-range forms answer the second output alone.
            if (f->kind == kXr && !s->dual) continue;
            for (int math = 0; math < 2; math++) {
                for (int halfOut = 1; halfOut >= 0; halfOut--) {
                    NSString *why = nil;
                    id<MTLRenderPipelineState> p = pipeline(f, s, YES, halfOut, math, &why);
                    const char *type = halfOut ? "half" : "float";
                    const char *mode = math == 0 ? "fast" : "safe";
                    if (!p) {
                        printf("REFUSED %s %s %s %s: %s\n", f->name, s->name, type, mode,
                               why.UTF8String);
                        continue;
                    }
                    memset(out, 0xA5, (size_t)N * f->bytes);
                    if (!draw(f, p, s, halfOut ? vh : vf, halfOut ? sh : sf, dst, out)) continue;
                    writeFile([NSString stringWithFormat:@"out/%s-%s-%s-%s.bin", f->name,
                                                         s->name, type, mode],
                              out, (size_t)N * f->bytes);
                    printf("DREW %s %s %s %s\n", f->name, s->name, type, mode);
                }
            }
        }
        fflush(stdout);
        free(dst);
        free(out);
    }
    return 0;
}

// The tile legs: a pass that clears to `kClear` (leg 0) or loads zero texels and draws `firsts`
// unblended (leg 1), then blends a source of (0, 0, 0, 0.5) over premultiplied source-over (one,
// one minus the source alpha): every stored channel is half the destination the blend read.
static const float kClear[4] = {0.33371f, 0.70011f, 0.10049f, 0.5f};

static int tile(int argc, char **argv) {
    setup();
    [[NSFileManager defaultManager] createDirectoryAtPath:@"out"
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:nil];
    float *firsts = malloc(N * 16), *halves = malloc(N * 16);
    for (int i = 0; i < N * 4; i++) {
        firsts[i] = -0.5f + 2.0f * (float)(next() >> 40) / (float)(1u << 24);
        halves[i] = (i % 4 == 3) ? 0.5f : 0.0f;
    }
    writeFile(@"out/tile-firsts.bin", firsts, N * 16);
    id<MTLBuffer> first = [gDevice newBufferWithBytes:firsts length:N * 16 options:0];
    id<MTLBuffer> half = [gDevice newBufferWithBytes:halves length:N * 16 options:0];
    const State *plain = stateNamed("one-zero");
    const State *over = stateNamed("premul");
    for (int a = 0; a < argc; a++) {
        const Format *f = formatNamed(argv[a]);
        if (!f) continue;
        NSString *why = nil;
        id<MTLRenderPipelineState> writer = pipeline(f, plain, NO, NO, 0, &why);
        id<MTLRenderPipelineState> blender = pipeline(f, over, YES, NO, 0, &why);
        if (!writer || !blender) {
            printf("REFUSED %s tile: %s\n", f->name, why.UTF8String);
            continue;
        }
        for (int leg = 0; leg < 2; leg++) {
            id<MTLTexture> t = target(f);
            id<MTLBuffer> zero = [gDevice newBufferWithLength:(NSUInteger)N * f->bytes options:0];
            memset(zero.contents, 0, (size_t)N * f->bytes);
            id<MTLBuffer> back = [gDevice newBufferWithLength:(NSUInteger)N * f->bytes options:0];
            MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = t;
            pass.colorAttachments[0].loadAction = leg == 0 ? MTLLoadActionClear : MTLLoadActionLoad;
            pass.colorAttachments[0].clearColor =
                MTLClearColorMake(kClear[0], kClear[1], kClear[2], kClear[3]);
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLCommandBuffer> cb = [gQueue commandBuffer];
            upload(cb, t, f, zero);
            id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
            uint32_t width = W;
            [enc setFragmentBytes:&width length:sizeof width atIndex:1];
            if (leg == 1) {
                [enc setRenderPipelineState:writer];
                [enc setFragmentBuffer:first offset:0 atIndex:0];
                [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            }
            [enc setRenderPipelineState:blender];
            [enc setFragmentBuffer:half offset:0 atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [enc endEncoding];
            download(cb, t, f, back);
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status != MTLCommandBufferStatusCompleted) {
                printf("DRAW-FAILED %s tile %d\n", f->name, leg);
                continue;
            }
            writeFile([NSString stringWithFormat:@"out/%s-tile-%s.bin", f->name,
                                                 leg == 0 ? "clear" : "draw"],
                      back.contents, (size_t)N * f->bytes);
            printf("TILE %s %s\n", f->name, leg == 0 ? "clear" : "draw");
        }
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc >= 4 && strcmp(argv[1], "create") == 0) return create(argv[2], argv[3]);
        if (argc >= 4 && strcmp(argv[1], "bare") == 0) return bare(argv[2], argv[3]);
        if (argc >= 3 && strcmp(argv[1], "sweep") == 0) return sweep(argc - 2, argv + 2);
        if (argc >= 3 && strcmp(argv[1], "tile") == 0) return tile(argc - 2, argv + 2);
        fprintf(stderr, "usage: create <format> <state> | sweep <format>... | tile <format>...\n");
        return 2;
    }
}
