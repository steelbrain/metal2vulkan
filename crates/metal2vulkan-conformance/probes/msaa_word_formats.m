// Multisample textures of the four formats a device may render only through the view of their
// word (`metal::format::WordAttachment`): `R8Unorm_sRGB` (11), `A1BGR5Unorm` (41), `ABGR4Unorm`
// (42) and `RGB9E5Float` (93), at 2 and 4 samples. What Metal makes, what each sample holds after a
// draw, what a `texture2d_ms` read returns, what the resolve writes, and what a blend stores.
//
// Build (no GPU needed):
//
//   clang -O2 -fobjc-arc -fmodules -framework Metal -framework Foundation \
//       -o msaa_word_formats msaa_word_formats.m
//
// Run WITHOUT MTL_DEBUG_LAYER, one shape per process:
//
//   ./msaa_word_formats make                      which usages Metal makes a texture of
//   ./msaa_word_formats run <format> <samples>    the legs below, every value printed
//
// <format> is the MTLPixelFormat number: 11, 41, 42 or 93. Every shape prints `DEVICE <name>`,
// `RELEASE <os>`, then its lines, then `DONE`; a shape whose process Metal ends prints no `DONE`.
//
// Every target is 16 x 16. A fill draws the fullscreen triangle at sample rate (the fragment
// function takes `[[sample_id]]`), each invocation writing its own sample's value from a buffer of
// every pixel's samples in order, `values[(y * 16 + x) * samples + sample]`. A pixel `p` holds
// one value in every sample where `p % 4 == 0`, two (samples 0 and 1, then the rest) where
// `p % 4 == 1`, and its own value in each sample otherwise; where `p % 8 == 7` the last sample is
// left to the pass's clear (the function discards it). Unorm and sRGB values are uniform over
// [-0.05, 1.05]; `RGB9E5Float` ones are `2^u`, `u` uniform over [-12, 9], times a uniform
// mantissa, a few of them zero, negative or past 65408. Samples are read two ways: through
// `texture2d_ms<float>::read` (`F` lines, the four `float`s' bits), and through an unsigned view of
// the texture's bits (`R8Uint`, `R16Uint`, `R32Uint`; `W` lines, the word). Every 1-sample resolve
// texture is `Shared` and read back with `getBytes` (its words).
//
//   * `FILL` -- a pass clears to a color no code holds exactly and fills, storing the samples and
//     resolving them (`StoreAndMultisampleResolve`): the samples' words and values (`FILL W`,
//     `FILL F`) and the resolve (`FILL R`), resolved from what the pass drew.
//   * `LOADED` -- a second pass loads those samples, draws nothing, and resolves (`LOADED R`): the
//     resolve of the stored words alone. Where it differs from `FILL R`, the resolve read more than
//     the stored codes (the pass's tile held finer values).
//   * `BLEND` -- a third pass loads the samples and draws per-sample source colors (`BLEND IN`)
//     under a source-over blend (`SourceAlpha`, `OneMinusSourceAlpha`, added, every channel; alpha
//     the same), storing and resolving: `BLEND W`, `BLEND F`, `BLEND R`.
//
// Each value line is `<leg> <kind> <pixel> <sample> <hex>...`; the inputs are `<leg> IN` lines of
// the four `float`s' bits.
//
// **What it answered** (an Apple M3 Max, macOS 27.0.1; the layer unset; every shape run in its own
// process, each `run` at 2 and 4 samples for each of the four formats).
//
//   * `make`: Metal makes all 48 textures, every usage above of each format at 2 and 4 samples,
//     the memoryless render target included, and `supportsTextureSampleCount` answers yes at both.
//   * `FILL W`: every sample's word is the word a single-sampled target of the format stores (the
//     packed fields' `half` code rule, the 8-bit sRGB encode by Metal's 12-bit table,
//     `RGB9E5Float`'s mantissas truncated), the clear's words in the discarded samples too.
//   * `FILL F`: a packed field reads its code over its largest code (the `float` quotient), an
//     8-bit sRGB code its linear value with green and blue 0 and alpha 1, `RGB9E5Float` its value
//     exactly with alpha 1.
//   * `FILL R` and `LOADED R`: `R8Unorm_sRGB` and `RGB9E5Float` resolve alike in both, from the
//     stored codes. The sRGB resolve is the rule the crate's sRGB resolves state, every pixel;
//     `RGB9E5Float`'s is each channel's mean of the samples' values in `float`, rounded to the
//     nearest `half` (ties to even), then encoded with its mantissas rounded half up (the exponent
//     raised where one reaches 512): 256 of 256 pixels at 2 and at 4 samples, the `BLEND R` legs
//     too. The packed formats' `FILL R` follows the mean of the pass's outputs as `half`s (253 to
//     256 of 256 pixels), finer than the codes, and their `LOADED R` no rule of the codes alone:
//     each field's mean of the codes, a tie rounded down (a one-bit field's up), is Metal's in 90%
//     to 99% of the channels a loaded pass resolves (every one-bit alpha) and 74% to 88% of those a
//     drawing pass resolves, and within one code in every channel; a pixel whose samples hold one
//     word resolves to it.
//   * `BLEND`: Metal makes the source-over blend of every format, `RGB9E5Float`'s included.
//
// So the word's format holds Metal's samples exactly, and the crate-owned resolve
// (`bind::ColorResolveFormat::word`) is Metal's for `RGB9E5Float` and an 8-bit sRGB format, and a
// near-equivalent for the packed formats (`Degradation::WordResolve`). The validation lane
// `runtime_lane::word_samples_msaa` replays the first 16 pixels' answers.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { kSide = 16, kPixels = kSide * kSide };

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;
static id<MTLLibrary> gLibrary = nil;

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct V { float4 position [[position]]; };\n"
    @"vertex V vs_full(uint vid [[vertex_id]]) {\n"
    @"  float2 p[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };\n"
    @"  V o; o.position = float4(p[vid], 0.0, 1.0); return o;\n"
    @"}\n"
    @"fragment float4 fill(V in [[stage_in]], uint s [[sample_id]],\n"
    @"                     constant float4 *values [[buffer(0)]],\n"
    @"                     constant uint *shape [[buffer(1)]]) {\n"
    @"  uint x = uint(floor(in.position.x)), y = uint(floor(in.position.y));\n"
    @"  uint p = y * shape[0] + x;\n"
    @"  if (shape[2] != 0 && p % 8 == 7 && s == shape[1] - 1) discard_fragment();\n"
    @"  return values[p * shape[1] + s];\n"
    @"}\n"
    @"kernel void read_f(texture2d_ms<float> t [[texture(0)]], device float4 *out [[buffer(0)]],\n"
    @"                   uint2 g [[thread_position_in_grid]]) {\n"
    @"  uint n = t.get_num_samples();\n"
    @"  if (g.x >= t.get_width() || g.y >= t.get_height()) return;\n"
    @"  for (uint s = 0; s < n; s++) out[(g.y * t.get_width() + g.x) * n + s] = t.read(g, s);\n"
    @"}\n"
    @"kernel void read_u(texture2d_ms<uint> t [[texture(0)]], device uint *out [[buffer(0)]],\n"
    @"                   uint2 g [[thread_position_in_grid]]) {\n"
    @"  uint n = t.get_num_samples();\n"
    @"  if (g.x >= t.get_width() || g.y >= t.get_height()) return;\n"
    @"  for (uint s = 0; s < n; s++) out[(g.y * t.get_width() + g.x) * n + s] = t.read(g, s).r;\n"
    @"}\n";

static void head(void) {
    printf("DEVICE %s\n", gDevice.name.UTF8String);
    printf("RELEASE %s\n", NSProcessInfo.processInfo.operatingSystemVersionString.UTF8String);
}

static uint32_t bitsOf(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    return b;
}

// A deterministic generator, so every run draws the same values.
static uint64_t gState = 0x9E3779B97F4A7C15ull;
static double uniform(void) {
    gState = gState * 6364136223846793005ull + 1442695040888963407ull;
    return (double)(gState >> 11) / (double)(1ull << 53);
}

static BOOL isShared(NSUInteger format) { return format == 93; }

// The unsigned format of the texel's size, whose view reads a sample's bits.
static MTLPixelFormat rawFormat(NSUInteger format) {
    switch (format) {
        case 11: return MTLPixelFormatR8Uint;
        case 41:
        case 42: return MTLPixelFormatR16Uint;
        default: return MTLPixelFormatR32Uint;
    }
}

static uint32_t texelBytes(NSUInteger format) {
    switch (format) {
        case 11: return 1;
        case 41:
        case 42: return 2;
        default: return 4;
    }
}

static float channelValue(NSUInteger format, int channel) {
    if (format == 93) {
        double u = uniform();
        if (u < 0.04) return 0.0f;
        if (u < 0.06) return -1.5f;
        if (u < 0.08) return 70000.0f;
        double e = -12.0 + 21.0 * uniform();
        return (float)(exp2(floor(e)) * (1.0 + uniform()));
    }
    (void)channel;
    return (float)(-0.05 + 1.1 * uniform());
}

// Every pixel's samples, as the module docs lay them out.
static void values(NSUInteger format, uint32_t n, float *out) {
    for (uint32_t p = 0; p < kPixels; p++) {
        float distinct[4][4];
        for (uint32_t s = 0; s < n; s++) {
            for (int c = 0; c < 4; c++) distinct[s][c] = channelValue(format, c);
        }
        for (uint32_t s = 0; s < n; s++) {
            uint32_t from = s;
            if (p % 4 == 0) from = 0;
            if (p % 4 == 1) from = s < n / 2 ? 0 : 1;
            for (int c = 0; c < 4; c++) out[(p * n + s) * 4 + c] = distinct[from][c];
        }
    }
}

static id<MTLTexture> target(NSUInteger format, uint32_t n, MTLTextureUsage usage) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = MTLTextureType2DMultisample;
    d.pixelFormat = (MTLPixelFormat)format;
    d.width = kSide;
    d.height = kSide;
    d.sampleCount = n;
    d.usage = usage;
    d.storageMode = MTLStorageModePrivate;
    return [gDevice newTextureWithDescriptor:d];
}

static id<MTLTexture> resolveTexture(NSUInteger format) {
    MTLTextureDescriptor *d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(MTLPixelFormat)format
                                                           width:kSide
                                                          height:kSide
                                                       mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
    // Prefilled, so a resolve that writes nothing is seen.
    NSMutableData *fill = [NSMutableData dataWithLength:kPixels * texelBytes(format)];
    memset(fill.mutableBytes, 0x5a, fill.length);
    [t replaceRegion:MTLRegionMake2D(0, 0, kSide, kSide)
         mipmapLevel:0
           withBytes:fill.bytes
         bytesPerRow:kSide * texelBytes(format)];
    return t;
}

static id<MTLRenderPipelineState> pipeline(NSUInteger format, uint32_t n, BOOL blend) {
    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [gLibrary newFunctionWithName:@"vs_full"];
    rp.fragmentFunction = [gLibrary newFunctionWithName:@"fill"];
    rp.colorAttachments[0].pixelFormat = (MTLPixelFormat)format;
    rp.rasterSampleCount = n;
    if (blend) {
        MTLRenderPipelineColorAttachmentDescriptor *a = rp.colorAttachments[0];
        a.blendingEnabled = YES;
        a.rgbBlendOperation = MTLBlendOperationAdd;
        a.alphaBlendOperation = MTLBlendOperationAdd;
        a.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        a.sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        a.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        a.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    }
    NSError *error = nil;
    id<MTLRenderPipelineState> s = [gDevice newRenderPipelineStateWithDescriptor:rp error:&error];
    if (!s) printf("PIPELINE %s nil: %s\n", blend ? "blend" : "fill",
                   error.localizedDescription.UTF8String);
    return s;
}

// One pass over `ms`: `load` or a clear to `clear`, a draw of `vals` through `state` where it is
// not nil (`skip` leaves the last sample of every eighth pixel), stored and resolved into
// `resolved`.
static BOOL pass(id<MTLTexture> ms, id<MTLTexture> resolved, BOOL load, const float clear[4],
                 id<MTLRenderPipelineState> state, const float *vals, uint32_t n, BOOL skip) {
    MTLRenderPassDescriptor *d = [MTLRenderPassDescriptor renderPassDescriptor];
    d.colorAttachments[0].texture = ms;
    d.colorAttachments[0].resolveTexture = resolved;
    d.colorAttachments[0].loadAction = load ? MTLLoadActionLoad : MTLLoadActionClear;
    d.colorAttachments[0].clearColor = MTLClearColorMake(clear[0], clear[1], clear[2], clear[3]);
    d.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:d];
    if (state) {
        uint32_t shape[3] = {kSide, n, skip ? 1u : 0u};
        [e setRenderPipelineState:state];
        id<MTLBuffer> b = [gDevice newBufferWithBytes:vals
                                                length:kPixels * n * 16
                                               options:MTLResourceStorageModeShared];
        [e setFragmentBuffer:b offset:0 atIndex:0];
        [e setFragmentBytes:shape length:sizeof shape atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        printf("PASS error: %s\n", cb.error.localizedDescription.UTF8String);
        return NO;
    }
    return YES;
}

// Every sample of `t` through `kernel` (`read_f` or `read_u`), into a buffer of `per` bytes a
// sample.
static NSData *readSamples(id<MTLTexture> t, NSString *kernel, uint32_t per) {
    NSError *error = nil;
    id<MTLComputePipelineState> s =
        [gDevice newComputePipelineStateWithFunction:[gLibrary newFunctionWithName:kernel]
                                               error:&error];
    if (!s) {
        printf("KERNEL %s nil: %s\n", kernel.UTF8String, error.localizedDescription.UTF8String);
        return nil;
    }
    NSUInteger length = (NSUInteger)kPixels * t.sampleCount * per;
    id<MTLBuffer> out = [gDevice newBufferWithLength:length options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:s];
    [e setTexture:t atIndex:0];
    [e setBuffer:out offset:0 atIndex:0];
    [e dispatchThreads:MTLSizeMake(kSide, kSide, 1)
        threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        printf("READ error: %s\n", cb.error.localizedDescription.UTF8String);
        return nil;
    }
    return [NSData dataWithBytes:out.contents length:length];
}

static void printInputs(const char *leg, const float *vals, uint32_t n) {
    for (uint32_t p = 0; p < kPixels; p++) {
        for (uint32_t s = 0; s < n; s++) {
            const float *v = &vals[(p * n + s) * 4];
            printf("%s IN %u %u %08x %08x %08x %08x\n", leg, p, s, bitsOf(v[0]), bitsOf(v[1]),
                   bitsOf(v[2]), bitsOf(v[3]));
        }
    }
}

static void printSamples(const char *leg, id<MTLTexture> ms, id<MTLTexture> raw, uint32_t n) {
    NSData *f = readSamples(ms, @"read_f", 16);
    NSData *w = raw ? readSamples(raw, @"read_u", 4) : nil;
    for (uint32_t p = 0; p < kPixels; p++) {
        for (uint32_t s = 0; s < n; s++) {
            if (f) {
                const uint32_t *b = (const uint32_t *)f.bytes + (p * n + s) * 4;
                printf("%s F %u %u %08x %08x %08x %08x\n", leg, p, s, b[0], b[1], b[2], b[3]);
            }
            if (w) {
                const uint32_t *b = (const uint32_t *)w.bytes + (p * n + s);
                printf("%s W %u %u %08x\n", leg, p, s, b[0]);
            }
        }
    }
}

static void printResolve(const char *leg, id<MTLTexture> t, NSUInteger format) {
    uint32_t per = texelBytes(format);
    NSMutableData *bytes = [NSMutableData dataWithLength:kPixels * per];
    [t getBytes:bytes.mutableBytes
        bytesPerRow:kSide * per
         fromRegion:MTLRegionMake2D(0, 0, kSide, kSide)
        mipmapLevel:0];
    const uint8_t *b = bytes.bytes;
    for (uint32_t p = 0; p < kPixels; p++) {
        uint32_t word = 0;
        memcpy(&word, b + p * per, per);
        printf("%s R %u 0 %08x\n", leg, p, word);
    }
}

static int run(NSUInteger format, uint32_t n) {
    printf("SHAPE run %lu %u\n", (unsigned long)format, n);
    MTLTextureUsage usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                            MTLTextureUsagePixelFormatView;
    id<MTLTexture> ms = target(format, n, usage);
    if (!ms) {
        printf("TARGET nil\n");
        return 1;
    }
    id<MTLTexture> raw = [ms newTextureViewWithPixelFormat:rawFormat(format)];
    printf("RAWVIEW %s\n", raw ? "made" : "nil");
    id<MTLRenderPipelineState> fill = pipeline(format, n, NO);
    id<MTLRenderPipelineState> blend = pipeline(format, n, YES);
    if (!fill) return 1;

    float *fillValues = calloc(kPixels * n * 4, sizeof(float));
    float *blendValues = calloc(kPixels * n * 4, sizeof(float));
    values(format, n, fillValues);
    values(format, n, blendValues);
    // The blend's source alphas: uniform over [0, 1], so every factor is a proper fraction.
    for (uint32_t i = 0; i < kPixels * n; i++) blendValues[i * 4 + 3] = (float)uniform();
    const float shared[4] = {0.1f, 3.3f, 70.7f, 1.0f};
    const float unorm[4] = {0.1f, 0.3f, 0.7f, 0.45f};
    const float *clear = isShared(format) ? shared : unorm;
    printf("CLEAR %08x %08x %08x %08x\n", bitsOf(clear[0]), bitsOf(clear[1]), bitsOf(clear[2]),
           bitsOf(clear[3]));

    printInputs("FILL", fillValues, n);
    id<MTLTexture> r1 = resolveTexture(format);
    if (!pass(ms, r1, NO, clear, fill, fillValues, n, YES)) return 1;
    printSamples("FILL", ms, raw, n);
    printResolve("FILL", r1, format);

    id<MTLTexture> r2 = resolveTexture(format);
    if (!pass(ms, r2, YES, clear, nil, NULL, n, NO)) return 1;
    printResolve("LOADED", r2, format);

    if (blend) {
        printInputs("BLEND", blendValues, n);
        id<MTLTexture> r3 = resolveTexture(format);
        if (!pass(ms, r3, YES, clear, blend, blendValues, n, NO)) return 1;
        printSamples("BLEND", ms, raw, n);
        printResolve("BLEND", r3, format);
    }
    free(fillValues);
    free(blendValues);
    return 0;
}

static int make(void) {
    printf("SHAPE make\n");
    const NSUInteger formats[] = {11, 41, 42, 93};
    const struct {
        const char *label;
        MTLTextureUsage usage;
        MTLStorageMode storage;
    } shapes[] = {
        {"RenderTarget", MTLTextureUsageRenderTarget, MTLStorageModePrivate},
        {"RenderTarget|ShaderRead", MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead,
         MTLStorageModePrivate},
        {"ShaderRead", MTLTextureUsageShaderRead, MTLStorageModePrivate},
        {"RenderTarget|ShaderRead|ShaderWrite",
         MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
         MTLStorageModePrivate},
        {"RenderTarget|PixelFormatView",
         MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView, MTLStorageModePrivate},
        {"RenderTarget(memoryless)", MTLTextureUsageRenderTarget, MTLStorageModeMemoryless},
    };
    for (uint32_t n = 2; n <= 4; n *= 2) {
        printf("SUPPORTS samples %u %s\n", n, [gDevice supportsTextureSampleCount:n] ? "yes" : "no");
        for (size_t f = 0; f < sizeof formats / sizeof formats[0]; f++) {
            for (size_t u = 0; u < sizeof shapes / sizeof shapes[0]; u++) {
                MTLTextureDescriptor *d = [MTLTextureDescriptor new];
                d.textureType = MTLTextureType2DMultisample;
                d.pixelFormat = (MTLPixelFormat)formats[f];
                d.width = kSide;
                d.height = kSide;
                d.sampleCount = n;
                d.usage = shapes[u].usage;
                d.storageMode = shapes[u].storage;
                id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
                printf("MAKE %lu %u %s %s\n", (unsigned long)formats[f], n, shapes[u].label,
                       t ? "made" : "nil");
            }
        }
    }
    return 0;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        setvbuf(stdout, NULL, _IOLBF, 0);
        gDevice = MTLCreateSystemDefaultDevice();
        if (!gDevice) {
            printf("NO DEVICE\n");
            return 1;
        }
        gQueue = [gDevice newCommandQueue];
        NSError *error = nil;
        gLibrary = [gDevice newLibraryWithSource:kSource options:nil error:&error];
        if (!gLibrary) {
            printf("LIBRARY nil: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        head();
        int status = 2;
        if (argc == 2 && strcmp(argv[1], "make") == 0) {
            status = make();
        } else if (argc == 4 && strcmp(argv[1], "run") == 0) {
            status = run((NSUInteger)strtoul(argv[2], NULL, 10), (uint32_t)atoi(argv[3]));
        } else {
            fprintf(stderr, "usage: %s make | run <format> <samples>\n", argv[0]);
            return 2;
        }
        if (status == 0) printf("DONE\n");
        return status;
    }
}
