// What a fragment function reads from a packed 16-bit colour attachment through Metal's
// programmable blending (`[[color(0)]]`), and what the attachment holds between the draws of one
// pass. Nothing here is drawn or read outside a texture's own bytes, so no step can fault the GPU.
//
// `bgra4_write_and_mipmap.m` settled that Metal TAKES such a fetch, and takes a source-over blend
// of such an attachment, by rotating sixteen words through a fetch and storing them back. A
// round trip cannot say what value the function read: `float(code/max)` and `half(code/max)`
// re-encode to the same code under the measured write rule (value to `half`, clamp, times the
// maximum code, round to nearest even), so the rotation is consistent with either. The
// translator's decode (`stage_io::word_color::decode`) rounds the quotient to `half`, which was
// measured only through the blend of four other formats. This probe reads the value itself.
//
//   * `READ` -- the attachment is loaded with sixteen known words and a fragment function reads
//     `[[color(0)]]` and stores it, unchanged, into a second `RGBA32Float` attachment. The bits
//     that come back are Metal's value for each code, which the probe tests against two rules:
//     the `f32` quotient `code/max`, and that quotient rounded to `half`, nearest even. Run for a
//     `float4` colour input and for a `half4` one, which are separate questions: a `float4` input
//     could carry a finer tile value than a `half4` one.
//   * `CLEAR` -- the same read after a clear whose channels are NOT representable in the
//     format's fields. If the tile holds a clear as a `half`, the read is the clear's `half`; if
//     the clear quantises into the attachment's codes, the read is a code's value. The two
//     answers differ at every channel of the clear colour chosen, so the leg separates them.
//   * `DRAW` -- the same read after an earlier draw of the SAME pass stored unrepresentable
//     values unblended. Same separation, for a write rather than a clear.
//   * `BLEND` -- a source-over blend over sixteen loaded destination words and sixteen source
//     colours (256 texels), read back as stored words.
//
// **The controls, and why they separate rather than decorate.**
//
//   * `ABGR4Unorm` (42) runs every leg beside `BGRA4Unorm` (44). Its four fields are the same four
//     4-bit unorm fields in another order, and its rules are the ones already recorded
//     (`metal::format::word`, `ChannelWrite::HalfUnorm { bits: 4 }`, `WordAttachment::Packed16`),
//     so the two formats must agree channel by channel once each is read through its OWN measured
//     layout (`ABGR4Unorm` red at bit 12, green 8, blue 4, alpha 0; `BGRA4Unorm` red at 8,
//     green 4, blue 0, alpha 12) and must disagree word for word, since the orders differ. A
//     reading that is really a constant, or that ignores the layout, fails one of those two.
//   * `B5G6R5Unorm` (40) runs them too, with 5-, 6- and 5-bit fields: its values must move with
//     the field width. A reading insensitive to width would make it agree with the 4-bit formats
//     where the record says it must not.
//   * `RGBA8Unorm` (70) runs the read legs as the control that the leg works at all on a format
//     every device renders directly, and that `float4` and `half4` inputs differ there exactly
//     where 8-bit codes are not `half`-representable.
//   * Every texel of every leg carries four DISAGREEING codes, and the sixteen texels of the read
//     legs cover all sixteen codes of each 4-bit channel, so the 4-bit answer is exhaustive and no
//     channel's answer can stand in for another's.
//   * Attachment 0's write mask is none in every reading draw, and its bytes are read back after
//     the pass: a leg whose attachment changed is reported and not relied on.
//
// **What each reading licenses.**
//
//   * `READ ... float4 EXPLAINS half-quotient` and the same for `half4`, with both 4-bit formats
//     agreeing and `B5G6R5Unorm` moving with its widths, licenses building the packed 16-bit word
//     fetch on `stage_io::word_color::decode` as it stands, and licenses the `BGRA4Unorm` entry in
//     `WordAttachment::layout` at the fields the controls confirm.
//   * `READ ... EXPLAINS f32-quotient` instead would mean the decode the blend uses is NOT a
//     fetch's value, and the fetch needs its own decode.
//   * `CLEAR`/`DRAW` answering `codes` means the attachment's words are the whole tile state, so a
//     word-view fetch is exact. Answering `tile-half` means Metal's tile carries more than the
//     attachment's codes within a pass, so a word-view fetch is a near-equivalent and must be
//     announced (`PrecisionNote::DeviceColorConversion`, which the word blend already notes for
//     this reason).
//
// **What it answered** (an Apple M3 Max, macOS 27.0.1, 2026-10-25; the layer unset; twelve fresh
// processes byte-identical, one case per leg per process).
//
//   * `READ`: every channel of `BGRA4Unorm`, `ABGR4Unorm` and `B5G6R5Unorm`, through a `float4`
//     and a `half4` color input, `EXPLAINS half-quotient` and fails `f32-quotient` at 14 or 15 of
//     the 16 texels. Each 4-bit channel saw all sixteen codes (`distinct-codes=16`), so the 4-bit
//     answer is exhaustive, and the two 4-bit formats agreed code for code through their own
//     layouts. The `RGBA8Unorm` control parts the rules: its `float4` input `EXPLAINS
//     f32-quotient` (code 7 of 255 read `3ce0e0e1`) and its `half4` input the half-quotient
//     (`3ce0e000`), so a probe that cannot see an `f32` quotient is ruled out.
//   * `CLEAR` and `DRAW`: `ANSWERS tile-half` 16/16 and `codes` 0/16 (one coincidence at red)
//     for every channel of all three formats, both input types. Metal's tile carries a pass's
//     clear and an earlier draw's unblended write as a `half`, not as the format's codes: a clear
//     of 0.1 reads `3dccc000` (`half` 0.0999756), where the stored code 1 of 15 would read
//     `3d888000`.
//   * `BLEND`: taken by Metal for all three formats. The half-arithmetic model (the destination's
//     code decoded to a `half`, the blend in `half`, the result stored by the measured write rule)
//     explains 256/256 channels but red, where it explains 255/256 for `BGRA4Unorm` and for both
//     controls alike -- the one-code regrouping `PrecisionNote::DeviceColorConversion` already
//     records. `BLEND rotation ABGR4-vs-BGRA4 channels-agree 256/256 words-equal 0/256`: the two
//     formats' blends are each other's through the measured field orders and share no word, so no
//     reading here is a constant and the orders are confirmed.
//
// So Metal's fetch of a packed 16-bit attachment reads the value of its codes, by the decode the
// blend already uses (`stage_io::word_color::decode`): the product's refusal of such a color input
// (`NotImplemented(Format)`) is its own gap, not Metal's.
//
// clang -O2 -fobjc-arc -framework Metal -framework Foundation -o packed16_word_fetch \
//     packed16_word_fetch.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { kABGR4 = 42, kB5G6R5 = 40, kBGRA4 = 44, kRGBA8 = 70, kRGBA32F = 125 };

typedef struct {
    int shift;  // -1 for a channel the format lacks
    int bits;
} Field;

typedef struct {
    const char *label;
    NSUInteger format;
    uint32_t texelBytes;
    Field fields[4];  // red, green, blue, alpha
} FormatDesc;

// The layouts the landed probe measured off Metal's own words, least significant field first.
static const FormatDesc kFormats[] = {
    {"ABGR4Unorm(control)", kABGR4, 2, {{12, 4}, {8, 4}, {4, 4}, {0, 4}}},
    {"BGRA4Unorm", kBGRA4, 2, {{8, 4}, {4, 4}, {0, 4}, {12, 4}}},
    {"B5G6R5Unorm(control)", kB5G6R5, 2, {{11, 5}, {5, 6}, {0, 5}, {-1, 0}}},
    {"RGBA8Unorm(control)", kRGBA8, 4, {{0, 8}, {8, 8}, {16, 8}, {24, 8}}},
};

static float halfOf(float x) { return (float)(_Float16)x; }

static uint32_t bitsOf(float x) {
    uint32_t b;
    memcpy(&b, &x, sizeof b);
    return b;
}

// ---------------------------------------------------------------- shaders

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct Out { float4 position [[position]]; };\n"
    @"vertex Out vs_quad(uint vid [[vertex_id]]) {\n"
    @"  float2 p[6] = { float2(-1,-1), float2(3,-1), float2(-1,3),\n"
    @"                  float2(3,-1), float2(3,3), float2(-1,3) };\n"
    @"  Out o; o.position = float4(p[vid], 0.0, 1.0); return o;\n"
    @"}\n"
    // The fetch, stored raw into a 32-bit float attachment: whatever Metal handed the function.
    @"struct Raw { float4 raw [[color(1)]]; };\n"
    @"fragment Raw read_f(Out in [[stage_in]], float4 prev [[color(0)]]) {\n"
    @"  Raw o; o.raw = prev; return o;\n"
    @"}\n"
    @"fragment Raw read_h(Out in [[stage_in]], half4 prev [[color(0)]]) {\n"
    @"  Raw o; o.raw = float4(prev); return o;\n"
    @"}\n"
    // One float4 per texel out of a buffer, indexed by the pass position, so the value a texel
    // takes is exactly the float the host put there.
    @"struct Col0 { float4 c [[color(0)]]; };\n"
    @"fragment Col0 write_f(Out in [[stage_in]], constant float4 *values [[buffer(0)]],\n"
    @"                      constant uint &width [[buffer(1)]]) {\n"
    @"  uint x = uint(floor(in.position.x)), y = uint(floor(in.position.y));\n"
    @"  Col0 o; o.c = values[y * width + x]; return o;\n"
    @"}\n"
    // The same without a second attachment, for the blend leg's one-attachment pass.
    @"fragment float4 src_f(Out in [[stage_in]], constant float4 *values [[buffer(0)]],\n"
    @"                      constant uint &width [[buffer(1)]]) {\n"
    @"  uint x = uint(floor(in.position.x)), y = uint(floor(in.position.y));\n"
    @"  return values[y * width + x];\n"
    @"}\n";

// ---------------------------------------------------------------- the read legs

// The sixteen words of a 4x4 read leg: texel `i` takes codes that disagree across the channels and,
// over the sixteen texels, cover every code of a 4-bit channel exactly once per channel.
static uint32_t readCode(const FormatDesc *f, int i, int c) {
    static const int kTurn[4] = {0, 5, 11, 3};
    uint32_t max = (1u << f->fields[c].bits) - 1;
    if (f->fields[c].bits == 4) {
        return (uint32_t)((i + kTurn[c]) & 15);
    }
    // A wider channel: a stride coprime with the code count, offset per channel, so the codes
    // disagree and the leg still sweeps the field.
    return (uint32_t)(((uint64_t)i * 7 + (uint64_t)kTurn[c] * 3) % (max + 1));
}

static uint32_t packWord(const FormatDesc *f, const uint32_t codes[4]) {
    uint32_t w = 0;
    for (int c = 0; c < 4; c++) {
        if (f->fields[c].shift < 0) continue;
        uint32_t mask = (1u << f->fields[c].bits) - 1;
        w |= (codes[c] & mask) << f->fields[c].shift;
    }
    return w;
}

// `attachment0 + RGBA32Float` pass; returns the sixteen `float4`s the fetch stored, or nil.
// `load` names the leg: 0 loads the words, 1 clears to `clear`, 2 loads zero words and runs an
// earlier draw of `written` first.
static NSData *runRead(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                       const FormatDesc *f, BOOL halfInput, int leg, const float clear[4],
                       const float *written, NSString **why) {
    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [library newFunctionWithName:@"vs_quad"];
    rp.fragmentFunction = [library newFunctionWithName:halfInput ? @"read_h" : @"read_f"];
    rp.colorAttachments[0].pixelFormat = (MTLPixelFormat)f->format;
    rp.colorAttachments[0].writeMask = MTLColorWriteMaskNone;
    rp.colorAttachments[1].pixelFormat = (MTLPixelFormat)kRGBA32F;
    NSError *error = nil;
    id<MTLRenderPipelineState> reader = [device newRenderPipelineStateWithDescriptor:rp
                                                                              error:&error];
    if (!reader) {
        *why = error.localizedDescription ?: @"(no description)";
        return nil;
    }
    id<MTLRenderPipelineState> writer = nil;
    if (leg == 2) {
        MTLRenderPipelineDescriptor *wp = [MTLRenderPipelineDescriptor new];
        wp.vertexFunction = [library newFunctionWithName:@"vs_quad"];
        wp.fragmentFunction = [library newFunctionWithName:@"write_f"];
        wp.colorAttachments[0].pixelFormat = (MTLPixelFormat)f->format;
        wp.colorAttachments[1].pixelFormat = (MTLPixelFormat)kRGBA32F;
        wp.colorAttachments[1].writeMask = MTLColorWriteMaskNone;
        writer = [device newRenderPipelineStateWithDescriptor:wp error:&error];
        if (!writer) {
            *why = error.localizedDescription ?: @"(no description)";
            return nil;
        }
    }

    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = (MTLPixelFormat)f->format;
    td.width = 4;
    td.height = 4;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    MTLTextureDescriptor *rd = [MTLTextureDescriptor new];
    rd.textureType = MTLTextureType2D;
    rd.pixelFormat = (MTLPixelFormat)kRGBA32F;
    rd.width = 4;
    rd.height = 4;
    rd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    rd.storageMode = MTLStorageModeShared;
    id<MTLTexture> raw = [device newTextureWithDescriptor:rd];
    if (!target || !raw) {
        *why = @"no texture";
        return nil;
    }

    NSMutableData *input = [NSMutableData dataWithLength:16 * (size_t)f->texelBytes];
    uint8_t *bytes = input.mutableBytes;
    for (int i = 0; i < 16; i++) {
        uint32_t codes[4] = {readCode(f, i, 0), readCode(f, i, 1), readCode(f, i, 2),
                             readCode(f, i, 3)};
        uint32_t w = leg == 2 ? 0 : packWord(f, codes);
        memcpy(bytes + (size_t)i * f->texelBytes, &w, f->texelBytes);
    }
    [target replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
              mipmapLevel:0
                withBytes:input.bytes
              bytesPerRow:4 * (size_t)f->texelBytes];

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = leg == 1 ? MTLLoadActionClear : MTLLoadActionLoad;
    if (leg == 1) {
        pass.colorAttachments[0].clearColor =
            MTLClearColorMake(clear[0], clear[1], clear[2], clear[3]);
    }
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[1].texture = raw;
    pass.colorAttachments[1].loadAction = MTLLoadActionDontCare;
    pass.colorAttachments[1].storeAction = MTLStoreActionStore;

    id<MTLBuffer> values = nil;
    if (leg == 2) {
        values = [device newBufferWithBytes:written
                                     length:16 * 4 * sizeof(float)
                                    options:MTLResourceStorageModeShared];
    }
    uint32_t width = 4;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
    if (leg == 2) {
        [enc setRenderPipelineState:writer];
        [enc setFragmentBuffer:values offset:0 atIndex:0];
        [enc setFragmentBytes:&width length:sizeof width atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    }
    [enc setRenderPipelineState:reader];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        *why = [NSString stringWithFormat:@"draw status %ld %@", (long)cb.status,
                                          cb.error.localizedDescription];
        return nil;
    }
    NSMutableData *out = [NSMutableData dataWithLength:16 * 4 * sizeof(float)];
    [raw getBytes:out.mutableBytes
        bytesPerRow:4 * 4 * sizeof(float)
         fromRegion:MTLRegionMake2D(0, 0, 4, 4)
        mipmapLevel:0];
    // The attachment the reading draw masked out: its words must be what the leg left there.
    NSMutableData *after = [NSMutableData dataWithLength:16 * (size_t)f->texelBytes];
    [target getBytes:after.mutableBytes
         bytesPerRow:4 * (size_t)f->texelBytes
          fromRegion:MTLRegionMake2D(0, 0, 4, 4)
         mipmapLevel:0];
    *why = nil;
    if (leg == 0 && memcmp(after.bytes, input.bytes, input.length) != 0) {
        *why = @"the masked attachment changed";
    }
    return out;
}

// The word the leg left in texel `i` (what the fetch should have read), for the clear and draw
// legs: Metal's measured write rule, value to `half`, clamp, times the maximum code, round to
// nearest even.
static uint32_t storedCode(float x, int bits) {
    double h = (double)halfOf(x);
    if (isnan(h)) h = 0.0;
    if (h < 0.0) h = 0.0;
    if (h > 1.0) h = 1.0;
    double m = (double)((1u << bits) - 1);
    double p = h * m;
    double r = nearbyint(p);  // the default rounding is nearest-even
    return (uint32_t)r;
}

static void reportRead(const FormatDesc *f, const char *leg, BOOL halfInput, NSData *got,
                       const uint32_t codes[16][4]) {
    const float *raw = got.bytes;
    const char *mode = halfInput ? "half4" : "float4";
    for (int c = 0; c < 4; c++) {
        if (f->fields[c].shift < 0) continue;
        uint32_t max = (1u << f->fields[c].bits) - 1;
        int okHalf = 0, okF32 = 0;
        int firstBadHalf = -1, firstBadF32 = -1;
        unsigned seen = 0;
        for (int i = 0; i < 16; i++) {
            uint32_t code = codes[i][c];
            if (f->fields[c].bits == 4) seen |= 1u << code;
            float quotient = (float)code / (float)max;
            float asHalf = halfOf(quotient);
            float v = raw[i * 4 + c];
            if (bitsOf(v) == bitsOf(asHalf)) okHalf++;
            else if (firstBadHalf < 0) firstBadHalf = i;
            if (bitsOf(v) == bitsOf(quotient)) okF32++;
            else if (firstBadF32 < 0) firstBadF32 = i;
        }
        int distinct = 0;
        for (int b = 0; b < 32; b++) distinct += (int)((seen >> b) & 1u);
        printf("%s %s %s channel %c bits=%d distinct-codes=%d\n", leg, f->label, mode, "rgba"[c],
               f->fields[c].bits, distinct);
        if (okHalf == 16) printf("%s %s %s channel %c EXPLAINS half-quotient\n", leg, f->label, mode, "rgba"[c]);
        if (okF32 == 16) printf("%s %s %s channel %c EXPLAINS f32-quotient\n", leg, f->label, mode, "rgba"[c]);
        if (okHalf != 16) {
            int i = firstBadHalf;
            printf("%s %s %s channel %c fails half-quotient at %d of 16 (first texel %d code=%u "
                   "got=%08x want=%08x)\n",
                   leg, f->label, mode, "rgba"[c], 16 - okHalf, i, codes[i][c],
                   bitsOf(raw[i * 4 + c]), bitsOf(halfOf((float)codes[i][c] / (float)max)));
        }
        if (okF32 != 16) {
            int i = firstBadF32;
            printf("%s %s %s channel %c fails f32-quotient at %d of 16 (first texel %d code=%u "
                   "got=%08x want=%08x)\n",
                   leg, f->label, mode, "rgba"[c], 16 - okF32, i, codes[i][c],
                   bitsOf(raw[i * 4 + c]), bitsOf((float)codes[i][c] / (float)max));
        }
        for (int i = 0; i < 16; i++) {
            printf("%s %s %s texel %2d channel %c code=%2u got=%08x (%.9g)\n", leg, f->label, mode,
                   i, "rgba"[c], codes[i][c], bitsOf(raw[i * 4 + c]), raw[i * 4 + c]);
        }
    }
}

// `READ`: the attachment holds the leg's own words.
static void legRead(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                    const FormatDesc *f) {
    uint32_t codes[16][4];
    for (int i = 0; i < 16; i++)
        for (int c = 0; c < 4; c++) codes[i][c] = readCode(f, i, c);
    for (int halfInput = 0; halfInput < 2; halfInput++) {
        NSString *why = nil;
        NSData *got = runRead(device, queue, library, f, halfInput, 0, NULL, NULL, &why);
        if (!got) {
            printf("READ %s %s UNAVAILABLE %s\n", f->label, halfInput ? "half4" : "float4",
                   why.UTF8String);
            continue;
        }
        if (why) printf("READ %s CONTROL-FAILED %s\n", f->label, why.UTF8String);
        reportRead(f, "READ", halfInput, got, codes);
    }
}

// `CLEAR` and `DRAW`: the attachment holds a clear's or an earlier draw's value, which Metal's
// tile may carry finer than the format's codes. Reported against both candidates.
static void legTile(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                    const FormatDesc *f, int leg) {
    // Four channels that disagree and that no field of 4, 5, 6 or 8 bits represents exactly.
    static const float kClear[4] = {0.1f, 0.23f, 0.37f, 0.56f};
    float written[16][4];
    for (int i = 0; i < 16; i++) {
        for (int c = 0; c < 4; c++) {
            written[i][c] = kClear[c] + (float)i * 0.0037f;
        }
    }
    const char *name = leg == 1 ? "CLEAR" : "DRAW";
    for (int halfInput = 0; halfInput < 2; halfInput++) {
        NSString *why = nil;
        NSData *got = runRead(device, queue, library, f, halfInput, leg, kClear,
                              leg == 2 ? &written[0][0] : NULL, &why);
        if (!got) {
            printf("%s %s %s UNAVAILABLE %s\n", name, f->label, halfInput ? "half4" : "float4",
                   why.UTF8String);
            continue;
        }
        const float *raw = got.bytes;
        const char *mode = halfInput ? "half4" : "float4";
        for (int c = 0; c < 4; c++) {
            if (f->fields[c].shift < 0) continue;
            int bits = f->fields[c].bits;
            uint32_t max = (1u << bits) - 1;
            int okCode = 0, okTile = 0;
            for (int i = 0; i < 16; i++) {
                float value = leg == 1 ? kClear[c] : written[i][c];
                uint32_t code = storedCode(value, bits);
                float fromCode = halfOf((float)code / (float)max);
                float tile = halfOf(value);
                float v = raw[i * 4 + c];
                if (bitsOf(v) == bitsOf(fromCode)) okCode++;
                if (bitsOf(v) == bitsOf(tile)) okTile++;
                if (i < 4) {
                    printf("%s %s %s texel %d channel %c value=%.9g code=%u got=%08x "
                           "code-value=%08x tile-half=%08x\n",
                           name, f->label, mode, i, "rgba"[c], value, code, bitsOf(v),
                           bitsOf(fromCode), bitsOf(tile));
                }
            }
            printf("%s %s %s channel %c ANSWERS %s (codes %d/16, tile-half %d/16)\n", name,
                   f->label, mode, "rgba"[c],
                   okCode == 16 ? "codes" : (okTile == 16 ? "tile-half" : "neither"), okCode,
                   okTile);
        }
    }
}

// ---------------------------------------------------------------- the blend leg

// A source-over blend (source alpha over one minus source alpha, added) over the sixteen
// destination words down x and sixteen source colours down y: 256 texels, read back as words.
static void legBlend(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                     const FormatDesc *f, uint16_t out[256]) {
    for (int i = 0; i < 256; i++) out[i] = 0xdead;
    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [library newFunctionWithName:@"vs_quad"];
    rp.fragmentFunction = [library newFunctionWithName:@"src_f"];
    rp.colorAttachments[0].pixelFormat = (MTLPixelFormat)f->format;
    rp.colorAttachments[0].blendingEnabled = YES;
    rp.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
    rp.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
    rp.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    rp.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    rp.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    rp.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    NSError *error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:rp
                                                                                error:&error];
    if (!pipeline) {
        printf("BLEND %s REFUSED-BY-METAL %s\n", f->label,
               error.localizedDescription.UTF8String ?: "(no description)");
        return;
    }
    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = (MTLPixelFormat)f->format;
    td.width = 16;
    td.height = 16;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    if (!target) {
        printf("BLEND %s TAKEN-BY-METAL but no texture\n", f->label);
        return;
    }
    NSMutableData *input = [NSMutableData dataWithLength:256 * (size_t)f->texelBytes];
    uint8_t *bytes = input.mutableBytes;
    float sources[256][4];
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            uint32_t codes[4] = {readCode(f, x, 0), readCode(f, x, 1), readCode(f, x, 2),
                                 readCode(f, x, 3)};
            uint32_t w = packWord(f, codes);
            memcpy(bytes + ((size_t)y * 16 + x) * f->texelBytes, &w, f->texelBytes);
            // Four disagreeing source channels, the alpha sweeping the sixteen rows.
            float *s = sources[y * 16 + x];
            s[0] = 0.13f + 0.051f * (float)y;
            s[1] = 0.71f - 0.042f * (float)y;
            s[2] = 0.37f + 0.0033f * (float)x;
            s[3] = (float)y / 15.0f;
        }
    }
    [target replaceRegion:MTLRegionMake2D(0, 0, 16, 16)
              mipmapLevel:0
                withBytes:input.bytes
              bytesPerRow:16 * (size_t)f->texelBytes];
    id<MTLBuffer> values = [device newBufferWithBytes:sources
                                              length:sizeof sources
                                             options:MTLResourceStorageModeShared];
    uint32_t width = 16;
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
    [enc setRenderPipelineState:pipeline];
    [enc setFragmentBuffer:values offset:0 atIndex:0];
    [enc setFragmentBytes:&width length:sizeof width atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("BLEND %s TAKEN-BY-METAL but draw status %ld %s\n", f->label, (long)cb.status,
               cb.error.localizedDescription.UTF8String);
        return;
    }
    NSMutableData *read = [NSMutableData dataWithLength:256 * (size_t)f->texelBytes];
    [target getBytes:read.mutableBytes
         bytesPerRow:16 * (size_t)f->texelBytes
          fromRegion:MTLRegionMake2D(0, 0, 16, 16)
         mipmapLevel:0];
    const uint8_t *got = read.bytes;
    printf("BLEND %s TAKEN-BY-METAL\n", f->label);
    // The half-arithmetic model the translator computes: the destination's code decoded to a
    // `half`, the blend in `half`, the result stored by the measured write rule.
    int explained[4] = {0, 0, 0, 0}, counted[4] = {0, 0, 0, 0};
    for (int i = 0; i < 256; i++) {
        uint32_t word = 0;
        for (uint32_t b = 0; b < f->texelBytes; b++) {
            word |= (uint32_t)got[(size_t)i * f->texelBytes + b] << (8 * b);
        }
        if (f->texelBytes == 2) out[i] = (uint16_t)word;
        const float *s = sources[i];
        _Float16 sa = (_Float16)s[3];
        for (int c = 0; c < 4; c++) {
            if (f->fields[c].shift < 0) continue;
            int bits = f->fields[c].bits;
            uint32_t max = (1u << bits) - 1;
            uint32_t dcode = readCode(f, i % 16, c);
            _Float16 d = (_Float16)halfOf((float)dcode / (float)max);
            _Float16 sf = c == 3 ? (_Float16)1.0f : sa;
            _Float16 df = (_Float16)1.0f - sa;
            _Float16 r = (_Float16)s[c] * sf + d * df;
            uint32_t want = storedCode((float)r, bits);
            uint32_t have = (word >> f->fields[c].shift) & max;
            counted[c]++;
            if (want == have) explained[c]++;
        }
    }
    for (int c = 0; c < 4; c++) {
        if (f->fields[c].shift < 0) continue;
        printf("BLEND %s channel %c half-model %d/%d\n", f->label, "rgba"[c], explained[c],
               counted[c]);
    }
}

// ---------------------------------------------------------------- main

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        printf("device %s\n", device.name.UTF8String);
        printf("layer MTL_DEBUG_LAYER=%s\n", getenv("MTL_DEBUG_LAYER") ?: "(unset)");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&error];
        if (!library) {
            printf("library failed: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        size_t n = sizeof(kFormats) / sizeof(*kFormats);
        for (size_t i = 0; i < n; i++) {
            legRead(device, queue, library, &kFormats[i]);
        }
        // The tile legs are the packed formats'; `RGBA8Unorm`'s code rule is another one
        // (the exact product, not the `half` route), so it answers neither candidate by
        // construction and is left out.
        for (size_t i = 0; i < 3; i++) {
            legTile(device, queue, library, &kFormats[i], 1);
            legTile(device, queue, library, &kFormats[i], 2);
        }
        // The two 4-bit formats over the same codes and sources: their stored words must be each
        // other's through the two measured layouts, and must differ word for word.
        uint16_t abgr[256], bgra[256];
        legBlend(device, queue, library, &kFormats[0], abgr);
        legBlend(device, queue, library, &kFormats[1], bgra);
        int rotated = 0, equal = 0;
        for (int i = 0; i < 256; i++) {
            uint32_t a[4], b[4];
            for (int c = 0; c < 4; c++) {
                a[c] = (abgr[i] >> kFormats[0].fields[c].shift) & 15;
                b[c] = (bgra[i] >> kFormats[1].fields[c].shift) & 15;
            }
            if (a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]) rotated++;
            if (abgr[i] == bgra[i]) equal++;
        }
        printf("BLEND rotation ABGR4-vs-BGRA4 channels-agree %d/256 words-equal %d/256\n", rotated,
               equal);
        legBlend(device, queue, library, &kFormats[2], (uint16_t[256]){0});
    }
    return 0;
}
