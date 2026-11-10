// What a fragment function reads from a padded 10-bit colour attachment through Metal's
// programmable blending (`[[color(0)]]`), and what the attachment holds between the draws of one
// pass. Nothing here is drawn or read outside a texture's own bytes, so no step can fault the GPU.
//
// `padded_and_bfloat_blends.m` measured what a BLEND reads as its destination (the tile's `half`
// through the pass); a fetch is a separate question, and this probe reads the value itself:
//
//   * `read` -- the attachment is loaded with every 10-bit code in each channel (and alpha's four
//     codes for `RGB10A2Unorm_sRGB`), half of the texels with their padding bits set, and a
//     fragment function reads `[[color(0)]]` and stores it unchanged into a second
//     `RGBA32Float` attachment, for a `float4` colour input and for a `half4` one.
//   * `clear` -- the same read after a clear whose channels are not codes' values, and past both
//     ends of [0, 1]: the read is the clear's `half` if the tile holds a clear as one, a code's
//     value if the clear quantises into the attachment's codes.
//   * `draw` -- the same read after an earlier draw of the SAME pass stored values unblended (a
//     `float4` output, in and past [0, 1]).
//
// Attachment 0's write mask is none in every reading draw; its bytes are read back after each
// pass into out/ beside the reads, so a leg whose attachment changed is visible.
//
// `RGBA8Unorm` (70) runs every leg as the control that the legs work on a format every device
// renders directly.
//
//   clang -O2 -fobjc-arc -framework Metal -framework Foundation -o padded_fetch padded_fetch.m
//   ./padded_fetch r10x6 r10x6srgb rg10x12 rg10x12srgb rgb10a2srgb rgba8
//
// Run it WITHOUT MTL_DEBUG_LAYER for plain Metal's answers. Each leg writes
// out/<format>-<leg>-<input>.bin (the reads, `float4` a texel) and out/<format>-<leg>-codes.bin
// (attachment 0 after the pass), and out/<format>-codes.bin and out/firsts.bin, the inputs.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    NSUInteger format;
    uint32_t bytes;
} Format;

static const Format kFormats[] = {
    {"r10x6", 576, 2},       {"r10x6srgb", 577, 2},  {"rg10x12", 578, 4},
    {"rg10x12srgb", 579, 4}, {"rgb10a2srgb", 586, 4}, {"rgba8", 70, 4},
};

// The clears: channels that are no code's value, and past both ends of [0, 1].
static const float kClears[3][4] = {
    {0.33371f, 0.70011f, 0.10049f, 0.5f},
    {-0.25f, 1.25f, 0.6f, 0.2f},
    {1.0f, 0.0f, 0.999f, 0.001f},
};

enum { W = 1024, H = 4, N = W * H };

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct V { float4 p [[position]]; };\n"
    @"vertex V vs(uint id [[vertex_id]]) {\n"
    @"  float2 a = float2(float(id & 1u), float(id >> 1));\n"
    @"  V o; o.p = float4(4.0 * a - 1.0, 0.0, 1.0); return o;\n"
    @"}\n"
    @"static uint at(V in, uint w) { return uint(in.p.y) * w + uint(in.p.x); }\n"
    @"struct One { float4 c [[color(1)]]; };\n"
    @"fragment One rf(V in [[stage_in]], float4 prev [[color(0)]]) { return One { prev }; }\n"
    @"fragment One rh(V in [[stage_in]], half4 prev [[color(0)]]) { return One { float4(prev) }; }\n"
    @"fragment float4 wf(V in [[stage_in]], const device float4 *c [[buffer(0)]],\n"
    @"                   constant uint &w [[buffer(1)]]) { return c[at(in, w)]; }\n";

static uint64_t gState = 0x9e3779b97f4a7c15ull;
static uint64_t next(void) {
    gState ^= gState >> 12;
    gState ^= gState << 25;
    gState ^= gState >> 27;
    return gState * 2685821657736338717ull;
}

static id<MTLDevice> gDevice;
static id<MTLCommandQueue> gQueue;
static id<MTLLibrary> gLibrary;

static BOOL writeFile(NSString *path, const void *bytes, size_t length) {
    return [[NSData dataWithBytes:bytes length:length] writeToFile:path atomically:NO];
}

static id<MTLTexture> texture(MTLPixelFormat format) {
    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = format;
    td.width = W;
    td.height = H;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    // Blits move only some of a padded texture's texels where Metal keeps it in its optimized
    // (compressed) layout.
    td.allowGPUOptimizedContents = NO;
    return [gDevice newTextureWithDescriptor:td];
}

static void blitIn(id<MTLCommandBuffer> cb, id<MTLTexture> t, uint32_t bytes, id<MTLBuffer> b) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:b
               sourceOffset:0
          sourceBytesPerRow:W * bytes
        sourceBytesPerImage:(NSUInteger)N * bytes
                 sourceSize:MTLSizeMake(W, H, 1)
                  toTexture:t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
}

static void blitOut(id<MTLCommandBuffer> cb, id<MTLTexture> t, uint32_t bytes, id<MTLBuffer> b) {
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:t
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(W, H, 1)
                        toBuffer:b
               destinationOffset:0
          destinationBytesPerRow:W * bytes
        destinationBytesPerImage:(NSUInteger)N * bytes];
    [blit endEncoding];
}

// A pipeline over attachment 0 of `f` and attachment 1 `RGBA32Float`: the reader `name` (attachment
// 0's mask none) or the writer `wf` (attachment 1's mask none).
static id<MTLRenderPipelineState> pipeline(const Format *f, NSString *name) {
    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [gLibrary newFunctionWithName:@"vs"];
    rp.fragmentFunction = [gLibrary newFunctionWithName:name];
    BOOL writer = [name isEqualToString:@"wf"];
    rp.colorAttachments[0].pixelFormat = (MTLPixelFormat)f->format;
    rp.colorAttachments[0].writeMask = writer ? MTLColorWriteMaskAll : MTLColorWriteMaskNone;
    rp.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Float;
    rp.colorAttachments[1].writeMask = writer ? MTLColorWriteMaskNone : MTLColorWriteMaskAll;
    NSError *error = nil;
    id<MTLRenderPipelineState> p = [gDevice newRenderPipelineStateWithDescriptor:rp error:&error];
    if (!p) printf("REFUSED %s %s: %s\n", f->name, name.UTF8String, error.localizedDescription.UTF8String);
    return p;
}

// The codes of texel `i`: every code of each 10-bit field across a row, the fields disagreeing,
// alpha's two bits cycling; the padding bits set in rows 2 and 3.
static void codes(const Format *f, int i, uint8_t *out) {
    uint32_t c0 = (uint32_t)(i % 1024), c1 = (uint32_t)((i * 7 + 3) % 1024),
             c2 = (uint32_t)((i * 13 + 500) % 1024), a = (uint32_t)(i % 4);
    BOOL pad = i / W >= 2;
    uint16_t p6 = pad ? 0x2A : 0, h0 = (uint16_t)(c0 << 6 | p6), h1 = (uint16_t)(c1 << 6 | p6);
    switch (f->format) {
        case 576:
        case 577:
            memcpy(out, &h0, 2);
            break;
        case 578:
        case 579:
            memcpy(out, &h0, 2);
            memcpy(out + 2, &h1, 2);
            break;
        case 586: {
            uint32_t w = c0 | c1 << 10 | c2 << 20 | a << 30;
            memcpy(out, &w, 4);
            break;
        }
        default: {
            uint8_t b[4] = {(uint8_t)c0, (uint8_t)(c1 >> 2), (uint8_t)(c2 >> 2), (uint8_t)(i * 3)};
            memcpy(out, b, 4);
            break;
        }
    }
}

// One pass over attachment 0 of `f`: loaded with `start` (leg "read" and "draw") or cleared to
// `clear`, then (leg "draw") `wf` of `firsts`, then the reader `name`. The reads into `reads`,
// attachment 0 after the pass into `after`.
static BOOL pass(const Format *f, id<MTLBuffer> start, const float *clear,
                 id<MTLBuffer> firsts, NSString *name, float *reads, uint8_t *after) {
    id<MTLRenderPipelineState> reader = pipeline(f, name);
    id<MTLRenderPipelineState> writer = firsts ? pipeline(f, @"wf") : nil;
    if (!reader || (firsts && !writer)) return NO;
    id<MTLTexture> a = texture((MTLPixelFormat)f->format), b = texture(MTLPixelFormatRGBA32Float);
    id<MTLBuffer> backA = [gDevice newBufferWithLength:(NSUInteger)N * f->bytes options:0];
    id<MTLBuffer> backB = [gDevice newBufferWithLength:(NSUInteger)N * 16 options:0];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = a;
    rp.colorAttachments[0].loadAction = clear ? MTLLoadActionClear : MTLLoadActionLoad;
    if (clear) rp.colorAttachments[0].clearColor = MTLClearColorMake(clear[0], clear[1], clear[2], clear[3]);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[1].texture = b;
    rp.colorAttachments[1].loadAction = MTLLoadActionClear;
    rp.colorAttachments[1].clearColor = MTLClearColorMake(-7, -7, -7, -7);
    rp.colorAttachments[1].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    blitIn(cb, a, f->bytes, start);
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rp];
    uint32_t width = W;
    if (firsts) {
        [enc setRenderPipelineState:writer];
        [enc setFragmentBuffer:firsts offset:0 atIndex:0];
        [enc setFragmentBytes:&width length:sizeof width atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [enc setRenderPipelineState:reader];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    blitOut(cb, a, f->bytes, backA);
    blitOut(cb, b, 16, backB);
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("PASS-FAILED %s %s: %s\n", f->name, name.UTF8String,
               cb.error.localizedDescription.UTF8String);
        return NO;
    }
    memcpy(reads, backB.contents, (size_t)N * 16);
    memcpy(after, backA.contents, (size_t)N * f->bytes);
    return YES;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        gDevice = MTLCreateSystemDefaultDevice();
        gQueue = [gDevice newCommandQueue];
        NSError *error = nil;
        gLibrary = [gDevice newLibraryWithSource:kSource options:[MTLCompileOptions new] error:&error];
        if (!gLibrary) {
            printf("LIBRARY %s\n", error.localizedDescription.UTF8String);
            return 2;
        }
        [[NSFileManager defaultManager] createDirectoryAtPath:@"out"
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:nil];
        // The earlier draw's values: codes' values and midpoints, values spread past both ends of
        // [0, 1], and specials.
        static const float kSpecials[] = {-0.0f, 0.0f, 1.0f, 2.0f, -1.0f, 0.5f, 1.5f, -0.5f,
                                          65504.0f, -65504.0f, 5.9604645e-8f};
        float *firstValues = malloc(N * 16);
        for (int i = 0; i < N * 4; i++) {
            uint32_t r = (uint32_t)(next() % 100);
            float v;
            if (r < 6) {
                v = kSpecials[next() % (sizeof kSpecials / sizeof kSpecials[0])];
            } else if (r < 20) {
                v = (float)(next() % 1024) / 1023.0f;
            } else if (r < 30) {
                v = ((float)(next() % 1023) + 0.5f) / 1023.0f;
            } else {
                v = -0.5f + 2.0f * (float)(next() >> 40) / (float)(1u << 24);
            }
            firstValues[i] = v;
        }
        writeFile(@"out/firsts.bin", firstValues, N * 16);
        id<MTLBuffer> firsts = [gDevice newBufferWithBytes:firstValues length:N * 16 options:0];
        float *reads = malloc(N * 16);
        for (int k = 1; k < argc; k++) {
            const Format *f = NULL;
            for (size_t j = 0; j < sizeof kFormats / sizeof kFormats[0]; j++) {
                if (strcmp(kFormats[j].name, argv[k]) == 0) f = &kFormats[j];
            }
            if (!f) {
                printf("UNKNOWN %s\n", argv[k]);
                continue;
            }
            uint8_t *start = malloc((size_t)N * f->bytes), *after = malloc((size_t)N * f->bytes);
            for (int i = 0; i < N; i++) codes(f, i, start + (size_t)i * f->bytes);
            writeFile([NSString stringWithFormat:@"out/%s-codes.bin", f->name], start,
                      (size_t)N * f->bytes);
            id<MTLBuffer> startBuf = [gDevice newBufferWithBytes:start
                                                          length:(NSUInteger)N * f->bytes
                                                         options:0];
            for (int in = 0; in < 2; in++) {
                NSString *name = in == 0 ? @"rf" : @"rh";
                const char *input = in == 0 ? "float4" : "half4";
                struct {
                    const char *leg;
                    const float *clear;
                    id<MTLBuffer> firsts;
                } legs[] = {
                    {"read", NULL, nil},          {"clear0", kClears[0], nil},
                    {"clear1", kClears[1], nil},  {"clear2", kClears[2], nil},
                    {"draw", NULL, firsts},
                };
                for (size_t l = 0; l < sizeof legs / sizeof legs[0]; l++) {
                    if (!pass(f, startBuf, legs[l].clear, legs[l].firsts, name, reads, after)) continue;
                    writeFile([NSString stringWithFormat:@"out/%s-%s-%s.bin", f->name, legs[l].leg, input],
                              reads, (size_t)N * 16);
                    writeFile([NSString stringWithFormat:@"out/%s-%s-%s-codes.bin", f->name, legs[l].leg,
                                                         input],
                              after, (size_t)N * f->bytes);
                    printf("READ %s %s %s\n", f->name, legs[l].leg, input);
                }
            }
            fflush(stdout);
            free(start);
            free(after);
        }
        return 0;
    }
}
