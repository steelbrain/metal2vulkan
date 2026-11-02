// Metal's answers behind `supports32BitMSAA`: the 32-bit integer formats as multisample textures
// (`R32Uint`, `R32Sint`, `RG32Uint`, `RG32Sint`, `RGBA32Uint`, `RGBA32Sint`) and the resolve of
// the 32-bit float ones (`R32Float`, `RG32Float`, `RGBA32Float`). One shape per process, so a
// process Metal ends is attributable to the shape that ended it; every process prints the device
// it ran on and the release, so each reading carries its own attribution.
//
// Build (no GPU needed):
//
//   clang -fobjc-arc -fmodules -framework Metal -framework Foundation \
//       -o msaa_32bit_formats msaa_32bit_formats.m
//
// Run one shape per process, WITHOUT MTL_DEBUG_LAYER (the layer asserts) unless the shape is
// meant to read what the layer says:
//
//   ./msaa_32bit_formats query
//   ./msaa_32bit_formats make <format> <samples>            (2D multisample, RenderTarget|ShaderRead)
//   ./msaa_32bit_formats int-samples <format> <samples>     (an integer format)
//   ./msaa_32bit_formats float-resolve <format> <samples> <store|resolve>
//   ./msaa_32bit_formats int-resolve <format> <samples>
//
// <format> is one of R32Uint R32Sint RG32Uint RG32Sint RGBA32Uint RGBA32Sint R32Float RG32Float
// RGBA32Float. Every shape prints `DEVICE <name> / RELEASE <os>`, then `SHAPE ...`, then `READ
// <key> <value>` lines, then `DONE`. A shape whose process Metal ends prints no `DONE`, and its
// exit status (and the layer's message on stderr) is the reading.
//
// Every target is 4 x 4. A fill draws the fullscreen triangle at sample rate (the fragment
// function takes `[[sample_id]]`), and each invocation writes its own sample's value from a buffer
// of every pixel's samples in order, `values[(y * 4 + x) * samples + sample]`. The samples are
// read back through `texture2d_ms<T>::read` by a kernel of one thread per sample, bit for bit.
//
// - `int-samples`: the integer values are `word(i)` for the i-th 32-bit channel of the buffer
//   (four a sample, whatever the format's channels): eight edge values, then
//   `i * 0x9E3779B9 + 0x7F4A7C15` (wrapping). The pass clears to (7, 8, 9, 10) and stores.
// - `float-resolve`: two channels a sample (`float2`, the first `R32Float`'s; a four-channel
//   format's blue and alpha are the negated pair). Pixels 0 to 7 hold the special values of
//   `kSpecial`, pixels 8 to 15 `pseudo(i)`. The pass resolves into a 4 x 4 texture of the same
//   format (`StoreAndMultisampleResolve` with `store`, `MultisampleResolve` with `resolve`), read
//   back by a blit, and the samples (after `store`) through the kernel. Each resolved channel is
//   compared with the rule `runtime_lane::resolve_rounding` states: the samples summed in order in
//   `float` from sample 0, each input, partial sum and the result times `1 / n` a denormal flushed
//   to the zero of its sign, and a NaN anywhere the canonical `0x7fc00000`.
// - `int-resolve`: an integer attachment filled as `int-samples` fills it, resolved into a
//   texture of its format (`MultisampleResolve`): with the layer, what it says; without, the
//   resolved words.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;
static id<MTLLibrary> gLibrary = nil;

enum { kSide = 4, kPixels = kSide * kSide };

static void head(void) {
    printf("DEVICE %s\n", gDevice.name.UTF8String);
    printf("RELEASE macOS %s\n",
           [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
    fflush(stdout);
}

static void readv(const char *key, const char *fmt, ...) {
    printf("READ %s ", key);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

static int done(void) {
    printf("DONE\n");
    fflush(stdout);
    return 0;
}

// ---------------------------------------------------------------------------------------------
// The formats.

typedef enum { kUint, kSint, kFloat } Kind;

typedef struct {
    const char *name;
    MTLPixelFormat pf;
    Kind kind;
    unsigned channels;
} Format;

static const Format kFormats[] = {
    {"R32Uint", MTLPixelFormatR32Uint, kUint, 1},
    {"R32Sint", MTLPixelFormatR32Sint, kSint, 1},
    {"RG32Uint", MTLPixelFormatRG32Uint, kUint, 2},
    {"RG32Sint", MTLPixelFormatRG32Sint, kSint, 2},
    {"RGBA32Uint", MTLPixelFormatRGBA32Uint, kUint, 4},
    {"RGBA32Sint", MTLPixelFormatRGBA32Sint, kSint, 4},
    {"R32Float", MTLPixelFormatR32Float, kFloat, 1},
    {"RG32Float", MTLPixelFormatRG32Float, kFloat, 2},
    {"RGBA32Float", MTLPixelFormatRGBA32Float, kFloat, 4},
};

static const Format *format_named(const char *name) {
    for (size_t i = 0; i < sizeof(kFormats) / sizeof(kFormats[0]); i++) {
        if (strcmp(kFormats[i].name, name) == 0) {
            return &kFormats[i];
        }
    }
    return NULL;
}

// ---------------------------------------------------------------------------------------------
// The functions.

static const char *const kShaders =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct V { float4 position [[position]]; };\n"
"struct Shape { uint width; uint samples; };\n"
"vertex V fill_vertex(uint id [[vertex_id]]) {\n"
"    float2 a = float2(float(id & 1u), float(id >> 1));\n"
"    V v; v.position = float4(4.0 * a - 1.0, 0.0, 1.0); return v;\n"
"}\n"
"static uint at(V in, uint s, constant Shape &shape) {\n"
"    return (uint(in.position.y) * shape.width + uint(in.position.x)) * shape.samples + s;\n"
"}\n"
"fragment uint4 fill_uint(V in [[stage_in]], uint s [[sample_id]],\n"
"        const device uint4 *v [[buffer(0)]], constant Shape &shape [[buffer(1)]]) {\n"
"    return v[at(in, s, shape)];\n"
"}\n"
"fragment int4 fill_sint(V in [[stage_in]], uint s [[sample_id]],\n"
"        const device int4 *v [[buffer(0)]], constant Shape &shape [[buffer(1)]]) {\n"
"    return v[at(in, s, shape)];\n"
"}\n"
"fragment float4 fill_float(V in [[stage_in]], uint s [[sample_id]],\n"
"        const device float2 *v [[buffer(0)]], constant Shape &shape [[buffer(1)]]) {\n"
"    float2 p = v[at(in, s, shape)];\n"
"    return float4(p, -p);\n"
"}\n"
"kernel void read_uint(texture2d_ms<uint, access::read> t [[texture(0)]],\n"
"        device uint4 *out [[buffer(0)]], constant Shape &shape [[buffer(1)]],\n"
"        uint3 p [[threadgroup_position_in_grid]]) {\n"
"    out[(p.y * shape.width + p.x) * shape.samples + p.z] = t.read(p.xy, p.z);\n"
"}\n"
"kernel void read_sint(texture2d_ms<int, access::read> t [[texture(0)]],\n"
"        device uint4 *out [[buffer(0)]], constant Shape &shape [[buffer(1)]],\n"
"        uint3 p [[threadgroup_position_in_grid]]) {\n"
"    out[(p.y * shape.width + p.x) * shape.samples + p.z] = as_type<uint4>(t.read(p.xy, p.z));\n"
"}\n"
"kernel void read_float(texture2d_ms<float, access::read> t [[texture(0)]],\n"
"        device uint4 *out [[buffer(0)]], constant Shape &shape [[buffer(1)]],\n"
"        uint3 p [[threadgroup_position_in_grid]]) {\n"
"    out[(p.y * shape.width + p.x) * shape.samples + p.z] = as_type<uint4>(t.read(p.xy, p.z));\n"
"}\n";

static id<MTLFunction> function(const char *name) {
    if (gLibrary == nil) {
        NSError *err = nil;
        gLibrary = [gDevice newLibraryWithSource:[NSString stringWithUTF8String:kShaders]
                                         options:nil
                                           error:&err];
        if (gLibrary == nil) {
            fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String);
            exit(3);
        }
    }
    return [gLibrary newFunctionWithName:[NSString stringWithUTF8String:name]];
}

// ---------------------------------------------------------------------------------------------
// The values.

// The integer channel `i`'s word.
static uint32_t word(uint32_t i) {
    static const uint32_t edge[8] = {0u,          1u,          0x7fffffffu, 0x80000000u,
                                     0xffffffffu, 0x00ffffffu, 0x01000001u, 0xdeadbeefu};
    return i < 8 ? edge[i] : i * 0x9E3779B9u + 0x7F4A7C15u;
}

static float from_bits(uint32_t b) {
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static uint32_t bits_of(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}

// Pixels 0 to 7: each sample's (red, green), samples 0 to 3.
static const uint32_t kSpecial[8][4][2] = {
    // Exact sums.
    {{0x3f800000u, 0x3f000000u}, {0x40000000u, 0x3e800000u},
     {0x40400000u, 0x3e000000u}, {0x40a00000u, 0x3d800000u}},
    // 1, 1e8, 1, -1e8 and -1e8, 1, 1e8, 1: the order of the sum shows.
    {{0x3f800000u, 0xccbebc20u}, {0x4cbebc20u, 0x3f800000u},
     {0x3f800000u, 0x4cbebc20u}, {0xccbebc20u, 0x3f800000u}},
    // 0.1 to 0.4 and their negations: rounding of an inexact sum.
    {{0x3dcccccdu, 0xbdcccccdu}, {0x3e4ccccdu, 0xbe4ccccdu},
     {0x3e99999au, 0xbe99999au}, {0x3ecccccdu, 0xbecccccdu}},
    // A NaN with a payload in sample 0 (red), a negative one in sample 3 (green).
    {{0x7fc12345u, 0x3f800000u}, {0x3f800000u, 0x40000000u},
     {0x40000000u, 0x40400000u}, {0x40400000u, 0xffc54321u}},
    // Infinities: +inf with finite values, and +inf with -inf.
    {{0x7f800000u, 0x7f800000u}, {0x3f800000u, 0xff800000u},
     {0x3f800000u, 0x00000000u}, {0x3f800000u, 0x00000000u}},
    // A denormal input (1e-40), and two normals whose average is a denormal (2e-38 twice).
    {{0x000116c2u, 0x00d9c7ddu}, {0x00000000u, 0x00d9c7ddu},
     {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u}},
    // Zeros of both signs.
    {{0x80000000u, 0x00000000u}, {0x80000000u, 0x80000000u},
     {0x80000000u, 0x00000000u}, {0x80000000u, 0x80000000u}},
    // FLT_MAX twice (the sum overflows), and FLT_MAX against -FLT_MAX.
    {{0x7f7fffffu, 0x7f7fffffu}, {0x7f7fffffu, 0xff7fffffu},
     {0x00000000u, 0x7f7fffffu}, {0x00000000u, 0xff7fffffu}},
};

// A pseudo-random float for the channel `i` of pixels 8 to 15: a 24-bit mantissa centred on 0,
// scaled by 2^(e) for e in -4..11.
static float pseudo(uint32_t i) {
    uint32_t x = i * 0x9E3779B9u + 0x7F4A7C15u;
    int e = (int)(x & 15u) - 4;
    float scale = from_bits((uint32_t)(127 + e) << 23);
    return ((float)(x >> 8) / 16777216.0f - 0.5f) * scale;
}

// Pixel `p`'s sample `s`'s channel `c` (0 red, 1 green) of a float fill.
static float float_value(unsigned p, unsigned s, unsigned c) {
    if (p < 8) {
        return from_bits(kSpecial[p][s][c]);
    }
    return pseudo((p * 4 + s) * 2 + c);
}

// ---------------------------------------------------------------------------------------------
// The passes.

static void commit(id<MTLCommandBuffer> cb, const char *what) {
    [cb commit];
    [cb waitUntilCompleted];
    readv(what, "status %ld error %s", (long)cb.status,
          cb.error ? cb.error.localizedDescription.UTF8String : "(none)");
}

static MTLTextureDescriptor *target(const Format *f, NSUInteger samples) {
    MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
    d.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    d.pixelFormat = f->pf;
    d.width = kSide;
    d.height = kSide;
    d.sampleCount = samples;
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    return d;
}

static id<MTLBuffer> shape_buffer(unsigned samples) {
    uint32_t shape[2] = {kSide, samples};
    return [gDevice newBufferWithBytes:shape length:sizeof(shape) options:MTLResourceStorageModeShared];
}

// A pass over `msaa` that clears it to `clear`, draws the fill of `values`, and stores by
// `store` into `resolve`.
static void fill(const Format *f, id<MTLTexture> msaa, unsigned samples, id<MTLBuffer> values,
                 MTLClearColor clear, MTLStoreAction store, id<MTLTexture> resolve) {
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = function("fill_vertex");
    pd.fragmentFunction = function(f->kind == kUint ? "fill_uint"
                                   : f->kind == kSint ? "fill_sint"
                                                      : "fill_float");
    pd.colorAttachments[0].pixelFormat = f->pf;
    pd.rasterSampleCount = samples;
    NSError *err = nil;
    id<MTLRenderPipelineState> state = [gDevice newRenderPipelineStateWithDescriptor:pd error:&err];
    if (state == nil) {
        readv("pipeline", "nil %s", err.localizedDescription.UTF8String);
        exit(4);
    }
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = msaa;
    rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = clear;
    rp.colorAttachments[0].storeAction = store;
    rp.colorAttachments[0].resolveTexture = resolve;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    [enc setRenderPipelineState:state];
    [enc setFragmentBuffer:values offset:0 atIndex:0];
    [enc setFragmentBuffer:shape_buffer(samples) offset:0 atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    commit(cb, "fill");
}

// Every sample of `msaa` as four words, through `texture2d_ms<T>::read`.
static id<MTLBuffer> read_samples(const Format *f, id<MTLTexture> msaa, unsigned samples) {
    id<MTLFunction> kernel = function(f->kind == kUint ? "read_uint"
                                      : f->kind == kSint ? "read_sint"
                                                         : "read_float");
    NSError *err = nil;
    id<MTLComputePipelineState> state = [gDevice newComputePipelineStateWithFunction:kernel
                                                                                error:&err];
    NSUInteger length = kPixels * samples * 16;
    id<MTLBuffer> out = [gDevice newBufferWithLength:length options:MTLResourceStorageModeShared];
    memset(out.contents, 0xdd, length);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:state];
    [enc setTexture:msaa atIndex:0];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBuffer:shape_buffer(samples) offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(kSide, kSide, samples)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    commit(cb, "read-samples");
    return out;
}

// The texels of a single-sample `texture` of `bytes` each, through a blit.
static id<MTLBuffer> read_texels(id<MTLTexture> texture, NSUInteger bytes) {
    NSUInteger length = kPixels * bytes;
    id<MTLBuffer> out = [gDevice newBufferWithLength:length options:MTLResourceStorageModeShared];
    memset(out.contents, 0xdd, length);
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(kSide, kSide, 1)
                        toBuffer:out
               destinationOffset:0
          destinationBytesPerRow:kSide * bytes
        destinationBytesPerImage:length];
    [blit endEncoding];
    commit(cb, "read-texels");
    return out;
}

// ---------------------------------------------------------------------------------------------
// The shapes.

static int query(void) {
    readv("supports32BitMSAA", "%d", gDevice.supports32BitMSAA ? 1 : 0);
    readv("supports32BitFloatFiltering", "%d", gDevice.supports32BitFloatFiltering ? 1 : 0);
    for (NSUInteger n = 1; n <= 16; n *= 2) {
        readv("supportsTextureSampleCount", "%lu %d", (unsigned long)n,
              [gDevice supportsTextureSampleCount:n] ? 1 : 0);
    }
    return done();
}

static int make(const Format *f, unsigned samples) {
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:target(f, samples)];
    readv("made", "%s %u %s", f->name, samples, t ? "yes" : "nil");
    if (t) {
        readv("sampleCount", "%lu", (unsigned long)t.sampleCount);
    }
    return done();
}

// The integer fill's buffer: four words a sample.
static id<MTLBuffer> int_values(unsigned samples) {
    NSUInteger words = kPixels * samples * 4;
    id<MTLBuffer> b = [gDevice newBufferWithLength:words * 4 options:MTLResourceStorageModeShared];
    uint32_t *w = (uint32_t *)b.contents;
    for (uint32_t i = 0; i < words; i++) {
        w[i] = word(i);
    }
    return b;
}

static int int_samples(const Format *f, unsigned samples) {
    id<MTLTexture> msaa = [gDevice newTextureWithDescriptor:target(f, samples)];
    if (msaa == nil) {
        readv("made", "nil");
        return done();
    }
    fill(f, msaa, samples, int_values(samples), MTLClearColorMake(7, 8, 9, 10),
         MTLStoreActionStore, nil);
    id<MTLBuffer> out = read_samples(f, msaa, samples);
    const uint32_t *got = (const uint32_t *)out.contents;
    unsigned mismatches = 0;
    for (unsigned i = 0; i < kPixels * samples; i++) {
        for (unsigned c = 0; c < 4; c++) {
            // A format's missing channels read 0, alpha 1.
            uint32_t want = c < f->channels ? word(i * 4 + c) : (c == 3 ? 1u : 0u);
            if (got[i * 4 + c] != want) {
                if (mismatches < 8) {
                    readv("mismatch", "sample-index %u channel %u got %08x want %08x", i, c,
                          got[i * 4 + c], want);
                }
                mismatches++;
            }
        }
    }
    readv("samples-read", "%u channels of %u samples, %u differ from the values written",
          kPixels * samples * 4, kPixels * samples, mismatches);
    for (unsigned i = 0; i < 2 * samples; i++) {
        readv("sample", "%u %08x %08x %08x %08x", i, got[i * 4], got[i * 4 + 1], got[i * 4 + 2],
              got[i * 4 + 3]);
    }
    return done();
}

static float flush(float v) {
    return fpclassify(v) == FP_SUBNORMAL ? copysignf(0.0f, v) : v;
}

// The rule's resolve of `n` samples `v`.
static uint32_t rule(const float *v, unsigned n) {
    for (unsigned s = 0; s < n; s++) {
        if (isnan(v[s])) {
            return 0x7fc00000u;
        }
    }
    volatile float acc = flush(v[0]);
    for (unsigned s = 1; s < n; s++) {
        acc = flush(acc + flush(v[s]));
    }
    volatile float scale = 1.0f / (float)n;
    float r = flush(acc * scale);
    return isnan(r) ? 0x7fc00000u : bits_of(r);
}

static int float_resolve(const Format *f, unsigned samples, bool keep) {
    id<MTLTexture> msaa = [gDevice newTextureWithDescriptor:target(f, samples)];
    id<MTLTexture> single = [gDevice newTextureWithDescriptor:target(f, 1)];
    if (msaa == nil || single == nil) {
        readv("made", "msaa %s single %s", msaa ? "yes" : "nil", single ? "yes" : "nil");
        return done();
    }
    NSUInteger count = kPixels * samples;
    id<MTLBuffer> values = [gDevice newBufferWithLength:count * 8
                                                options:MTLResourceStorageModeShared];
    float *v = (float *)values.contents;
    for (unsigned p = 0; p < kPixels; p++) {
        for (unsigned s = 0; s < samples; s++) {
            for (unsigned c = 0; c < 2; c++) {
                v[(p * samples + s) * 2 + c] = float_value(p, s, c);
            }
        }
    }
    fill(f, msaa, samples, values, MTLClearColorMake(0.25, 0.5, 0.75, 1),
         keep ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionMultisampleResolve,
         single);
    id<MTLBuffer> texels = read_texels(single, 4 * f->channels);
    const uint32_t *got = (const uint32_t *)texels.contents;
    const uint32_t *stored = NULL;
    if (keep) {
        stored = (const uint32_t *)read_samples(f, msaa, samples).contents;
    }
    unsigned differ = 0;
    for (unsigned p = 0; p < kPixels; p++) {
        char line[512];
        size_t at = 0;
        for (unsigned c = 0; c < f->channels; c++) {
            // The fill's channel: red and green, then their negations.
            float in[4];
            for (unsigned s = 0; s < samples; s++) {
                float x = float_value(p, s, c & 1);
                in[s] = c < 2 ? x : -x;
            }
            uint32_t want = rule(in, samples);
            uint32_t have = got[p * f->channels + c];
            if (have != want) {
                differ++;
            }
            at += (size_t)snprintf(line + at, sizeof(line) - at, " %08x%s", have,
                                   have == want ? "" : "*");
            if (have != want) {
                at += (size_t)snprintf(line + at, sizeof(line) - at, "(rule %08x)", want);
            }
        }
        readv("resolved", "pixel %u%s", p, line);
        if (stored != NULL) {
            at = 0;
            for (unsigned s = 0; s < samples; s++) {
                at += (size_t)snprintf(line + at, sizeof(line) - at, " %08x/%08x",
                                       stored[(p * samples + s) * 4],
                                       stored[(p * samples + s) * 4 + 1]);
            }
            readv("stored", "pixel %u red/green%s", p, line);
        }
    }
    readv("rule", "%u of %u channels differ from the rule", differ, kPixels * f->channels);
    return done();
}

static int int_resolve(const Format *f, unsigned samples) {
    id<MTLTexture> msaa = [gDevice newTextureWithDescriptor:target(f, samples)];
    id<MTLTexture> single = [gDevice newTextureWithDescriptor:target(f, 1)];
    if (msaa == nil || single == nil) {
        readv("made", "msaa %s single %s", msaa ? "yes" : "nil", single ? "yes" : "nil");
        return done();
    }
    fill(f, msaa, samples, int_values(samples), MTLClearColorMake(7, 8, 9, 10),
         MTLStoreActionMultisampleResolve, single);
    const uint32_t *got = (const uint32_t *)read_texels(single, 4 * f->channels).contents;
    for (unsigned p = 0; p < 4; p++) {
        char line[256];
        size_t at = 0;
        for (unsigned c = 0; c < f->channels; c++) {
            at += (size_t)snprintf(line + at, sizeof(line) - at, " %08x", got[p * f->channels + c]);
        }
        readv("resolved", "pixel %u%s", p, line);
    }
    return done();
}

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: see the header\n");
        return 2;
    }
    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        if (gDevice == nil) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        gQueue = [gDevice newCommandQueue];
        head();
        printf("SHAPE");
        for (int i = 1; i < argc; i++) {
            printf(" %s", argv[i]);
        }
        printf("\n");
        fflush(stdout);
        const char *shape = argv[1];
        if (strcmp(shape, "query") == 0) {
            return query();
        }
        const Format *f = argc >= 3 ? format_named(argv[2]) : NULL;
        unsigned samples = argc >= 4 ? (unsigned)atoi(argv[3]) : 0;
        if (f == NULL || samples == 0) {
            fprintf(stderr, "a shape takes <format> <samples>\n");
            return 2;
        }
        if (strcmp(shape, "make") == 0) {
            return make(f, samples);
        }
        if (strcmp(shape, "int-samples") == 0 && f->kind != kFloat) {
            return int_samples(f, samples);
        }
        if (strcmp(shape, "float-resolve") == 0 && f->kind == kFloat && argc == 5) {
            return float_resolve(f, samples, strcmp(argv[4], "store") == 0);
        }
        if (strcmp(shape, "int-resolve") == 0 && f->kind != kFloat) {
            return int_resolve(f, samples);
        }
        fprintf(stderr, "no such shape\n");
        return 2;
    }
}
