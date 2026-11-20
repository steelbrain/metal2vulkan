// `MTLPixelFormatBGRA4Unorm` (raw 44, a private format): the three Metal answers the runtime
// holds no measurement of, and the controls that make each reading trustworthy. Nothing here is
// drawn or read outside a texture's own bytes, so no step can fault the GPU.
//
// The format has an exact Vulkan image (`A4R4G4B4_UNORM_PACK16`, bit for bit), but three of its
// answers were never asked of Metal, and each one gates a piece of the runtime:
//
//   1. **The colour-attachment write conversion** (`metal::format::write_conversion`, today
//      `WriteConversion::Unmeasured`). The other four packed 16-bit unorm formats were measured to
//      round a `float` output to a `half` first and then as a unorm channel
//      (`ChannelWrite::HalfUnorm`), but this format was never drawn into, so a fragment function
//      writing it stores what the device computes rather than Metal's code. Six committed profiles
//      render the format directly (RADV x2, lavapipe x2, ANV x2), so the rule is reachable there.
//   2. **`generateMipmapsForTexture:`** (refused `NotImplemented(MipmapGeneration)` in
//      `mipmap_plan::not_implemented`, and absent from `mipmap_answers.txt`). Its four 4-bit fields
//      from bit 0 are `ABGR4Unorm`'s, in a channel order neither Metal's filter nor
//      `bind::mipmap_filter`'s rule reads -- but that is an argument about our kernel, not a
//      measurement of Metal, which is why the refusal stands until this runs.
//   3. **The word-view render fallback** (`metal::format::word`, which has no entry for it).
//      `A4R4G4B4_UNORM_PACK16` carries no `COLOR_ATTACHMENT` and no `BLIT_DST` on `moltenvk-m3` or
//      `nvidia-rtx5080-laptop`, while `R16_UINT` carries `COLOR_ATTACHMENT` on both, so without the fallback a render target of this format is refused on Metal's own
//      device. Taking it, though, turns `framebuffer_fetch::renders_through_words` true on
//      MoltenVK -- an `any` over `PixelFormat::ALL` -- which turns the `ProgrammableBlending`
//      family row into `RowState::Gap`, and `RowState::blocks()` means MoltenVK would then claim
//      no Apple family at all. Leg C settles whether that gap is ours or Metal's.
//
// **The legs.**
//
//   * `WRITE` -- one draw per (format, output type, blend) into a 256-wide target, one texel per
//     input value: every `half` in [0, 1] (15,361), every midpoint between two of them (15,360),
//     every code's value and every tie of a 1-, 4-, 5- and 6-bit channel with two `f32` steps
//     either side, the values outside [0, 1], and the non-finite ones -- about 33,800 texels,
//     from a `float` and from a `half` output, plain and blended with factors one and zero.
//     It reads each channel's field out of Metal's own words (the four single-channel texels),
//     never out of an assumption, then reports which of six candidate rules explains every texel:
//     unorm and half-unorm, each rounded to nearest-even, nearest-away and toward zero.
//   * `MIP` -- `generateMipmapsForTexture:` of the three shapes `mipmap_answers.txt` records
//     (`tie` 8x8, `odd` 7x5, `vol` 4x4x4), replayed from that file's own level-0 bytes, each run
//     twice in the process to show the answer is deterministic.
//   * `FETCH` -- whether Metal takes a fragment function that reads the attachment it writes
//     (`[[color(0)]]`, its programmable blending), over 16 words whose every nibble is
//     distinguishable, with the channels rotated so a read that happened cannot be mistaken for a
//     write that did not.
//   * `BLEND` -- whether Metal takes a blend of the attachment other than factors one and zero.
//
// **The controls, and what makes them differentiated rather than decorative.**
//
//   * `ABGR4Unorm` (42) runs every leg beside `BGRA4Unorm`: four 4-bit unorm channels, recorded as
//     `HalfUnorm{4}`, with a block in `mipmap_answers.txt`.
//   * `B5G6R5Unorm` (40) runs them too, with 5-, 6- and 5-bit channels, so a reading that did not
//     move with field width would show up as the two controls agreeing when the record says they
//     must not.
//   * `RGBA8Unorm` (70) runs the fetch leg, as the control that the leg works at all on a format
//     every device renders directly.
//   * For `MIP`, the file holds two columns: an M3 Max on macOS 26.7 and an M4 on 27.0.1. This Mac
//     is an M3 Max on 27.0.1, and under 27.0.1's rule every level is the filter's, so the control
//     must reproduce the SECOND column and must differ from the first at exactly the six places the
//     file itself records them as different (`ABGR4Unorm` tie L1, L2, L3 and vol L2;
//     `B5G6R5Unorm` tie L1 and L2) and agree with both everywhere else. A run that matched the
//     26.7 column, or differed anywhere the columns agree, means the harness is wrong and not the
//     record. This prediction was written down before the run.
//   * `BGRA4Unorm` is then replayed over `ABGR4Unorm`'s level-0 bytes, which is the shape the file
//     already uses for a channel-rotated twin ("ARGB8Unorm: RGBA8Unorm's inputs", whose recorded
//     answers are byte-identical to `RGBA8Unorm`'s). Its own generator is seeded per format and
//     cannot be reproduced, and its fields are the same four 4-bit fields, so the same bytes are
//     the right inputs.
//
// **What each reading licenses.**
//
//   * `WRITE ... EXPLAINS half-unorm/nearest-even` for all four channels of `BGRA4Unorm`, with both
//     controls reproducing their recorded rules, licenses `write_conversion` returning
//     `Rules([ChannelWrite::HalfUnorm { bits: 4 }; 4])`, and the `hot` words license
//     `WordAttachment::layout` as red at bit 8, green at 4, blue at 0 and alpha at 12. Any other
//     rule, or a field at another bit, is the measurement and the code follows it.
//   * `MIP` lines that reproduce the controls' 27.0.1 column license adding a `BGRA4Unorm` block to
//     `mipmap_answers.txt` and dropping the format from `mipmap_plan::not_implemented`; they also
//     license `bind::mipmap_filter` computing it (`layout.rs`: `P::ABGR4Unorm | P::BGRA4Unorm =>
//     (2, uniform(4, 4, widened(4, 8)))`, plus the `half` set, plus `unlike_an_m3`), which is what
//     makes the path work on MoltenVK at all, since no blit chain can run there without `BLIT_DST`.
//     Its macOS 26.7 passes cannot be measured on this Mac, so `unlike_an_m3` stays true below the
//     first level under `MipmapRule::AppleM3MaxPasses` -- an announcement, not a refusal.
//   * `FETCH BGRA4Unorm REFUSED-BY-METAL` means the runtime's refusal of a fetch of a word
//     attachment matches Metal, so the format must not count toward
//     `framebuffer_fetch::renders_through_words`, MoltenVK's `ProgrammableBlending` row stays
//     `Exact`, and the word-view fallback can be taken. `FETCH BGRA4Unorm TAKEN-BY-METAL` means the
//     gap is ours, and the fallback waits on a packed-16 word fetch in
//     `lower::fragment::fetch` rather than being bought at the price of every Apple family on
//     MoltenVK.
//   * `BLEND ... REFUSED-BY-METAL` would likewise mean the word view's "no blend but one and zero"
//     refusal matches Metal rather than being a gap.
//
// clang -O2 -fobjc-arc -framework Metal -framework Foundation \
//     -o bgra4_write_and_mipmap bgra4_write_and_mipmap.m
//
// **What it answered** (an Apple M3 Max, macOS 27.0.1, 2026-10-25; the layer unset).
//
//   * The controls reproduced their recorded field positions exactly (`ABGR4Unorm` red at bit 12,
//     green 8, blue 4, alpha 0; `B5G6R5Unorm` red 11/5 bits, green 5/6, blue 0/5), and
//     `BGRA4Unorm` measured red at bit 8, green 4, blue 0 and alpha 12 -- Metal's own words, not an
//     assumption. The `graded` texel moved with the order (`1248` against `8124`), so no reading
//     is a constant.
//   * Every channel of all three formats is explained by half-unorm rounding and by nothing that
//     rounds the exact product: `ABGR4Unorm`'s and `BGRA4Unorm`'s red of 0.0333404541 stores 0
//     where the product rule wants 1 (57 of 31,864 values part company). Nearest-even and
//     nearest-away agree on every value these formats reach, so this run does not separate them.
//   * All nine mipmap cases were twice-identical in the process. Every control level reproduced
//     the file's 27.0.1 column and differed from its 26.7 column at exactly the six places the
//     file records them as different -- the prediction above, met. `BGRA4Unorm`'s levels came out
//     `ABGR4Unorm`'s 27.0.1 answer byte for byte.
//   * `FETCH BGRA4Unorm TAKEN-BY-METAL`, rotating all 16 words exactly through the measured
//     layout, and `BLEND BGRA4Unorm source-over TAKEN-BY-METAL`. So the word view's refusals
//     would be our gap, and the fallback was NOT taken.
//
// One defect in that run, fixed here: the `RGBA8Unorm` fetch control read a 4-byte texel back with
// `bytesPerRow` 8, which raised two AGX assertions and zeroed its texel lines. Its pipeline
// acceptance stood; its bytes were not relied on. The stride is now the format's.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { kABGR4 = 42, kB5G6R5 = 40, kBGRA4 = 44 };

// ---------------------------------------------------------------- hex

static NSMutableData *parseHex(const char *text) {
    size_t n = strlen(text);
    NSMutableData *data = [NSMutableData dataWithLength:n / 2];
    uint8_t *bytes = data.mutableBytes;
    for (size_t i = 0; i + 1 < n; i += 2) {
        char pair[3] = {text[i], text[i + 1], 0};
        bytes[i / 2] = (uint8_t)strtoul(pair, NULL, 16);
    }
    return data;
}

static NSString *printHex(const uint8_t *bytes, size_t n) {
    NSMutableString *s = [NSMutableString stringWithCapacity:2 * n];
    for (size_t i = 0; i < n; i++) {
        [s appendFormat:@"%02x", bytes[i]];
    }
    return s;
}

// ---------------------------------------------------------------- part A: the mipmap filter

struct MipCase {
    const char *name;
    uint32_t width, height, depth;  // depth > 1: a 3D texture
    uint32_t levels;
    const char *input;  // level 0's bytes, from `mipmap_answers.txt`
};

// `ABGR4Unorm`'s three cases, level 0 and the recorded M3 Max / M4 answers.
static const struct MipCase kAbgr4Cases[] = {
    {"tie", 8, 8, 1, 4,
     "809780b80f081f09f1f0e1ff1e102e10802880971f090f08e1fff1f02e101e11e7edf7eb787b887380809"
     "1800f0ffe1ff7e9e7e38875787191808080fe1f0f0f01f102e1ce1e6f2e97e708f70878098802e111f12f2"
     "ede1e08f7a7e709880878f080ff91100f10fee501e502791e7c8fff91f08010fe110fe902e6117c5f78ee"},
    {"odd", 7, 5, 1, 3,
     "f0ff00f000ff0fff0ff0000000f0f0ffff0fffffffffff000f0fff00fff0ffffff00f00f0fff0000ff0ffff0"
     "0000ffff000ffff00f0f000fffff0ff0fff0ff0ff0fff0ff0ff0"},
    {"vol", 4, 4, 4, 3,
     "e61e7dc0ef1cf000f0b11d67f9290ff0cfe7b5fb5801fff000f0ff0f00f00fff12060089371bf000e9a4d862"
     "c7fb00f0439f105ef8370ffff0ffff0ff0f0fff00a8e00ee7aeef000b35c1f58cc480fffe8f7c7f598e10fff"
     "00f0f00f0f0fffff00f0fff000f00f0ff0fff000ff0f0000ff00ffffff000fff0000fff000f00f0f"},
};
static const char *const kAbgr4Recorded[3][3] = {
    {"80880f09f1f81e10f7e9887590808f0f01f18f1e58f70988f890108fe6017a7f", "8878886666886878",
     "7777"},
    {"c87cbfff4b185771440f394e", "898f", NULL},
    {"86759765979b89c575a86868b8955c99", "8897", NULL},
};
// `B5G6R5Unorm`'s, the second control: other field widths, so other answers.
static const struct MipCase kB5g6r5Cases[] = {
    {"tie", 8, 8, 1, 4,
     "007c00841f803f88e107c1ff3e005e000084007c3f881f80c1ffe1075e003e08cf77efe7f04b103c00042104"
     "1ff8feffefefcf871094f06b21040004feff1ff8010802085ef2dff6af79f07a10801180020821081ff3bef7"
     "b07f0f7a11801080e007ff0f20f820f0ca0fc917e1f3fcfbff0fe00720f021f8c117dc0fe0fbe2f3"},
    {"odd", 7, 5, 1, 3,
     "e007000000001ff81ff81ff8ffff00000000e0ffffff00f8ff07ff07e0ffffff0000ffffe00700000000"
     "00f800f81ff80000e0ffe0ff1f00ffff00f8ff07ffff0000ffff00f8"},
    {"vol", 4, 4, 4, 3,
     "907838a36d141f00b00a1c44f0e7ff0710f0edd3fbd5e007e00700f8e0ffe0ffa1acd0f5e6c31ff8aec03bf4"
     "fb0fe0ffb5f7087f09301ff81f001ff8e0ff00f8262860cfdeffe0ffb30b807beb871ff83f68437ee2f600f8"
     "e0ff000000f80000ff071ff8e007ff07e0071ff81ff8e0ff1ff800f8ff071ff81ff81ff8ff070000"},
};
static const char *const kB5g6r5Recorded[3][3] = {
    {"00843f80e1875e00efb7106420041ffc0208fff4907b1080f00720f8cc17e8fb", "108410644c83cd83",
     "ce7b"},
    {"a000fffd9f46a3f808424dbd", "139c", NULL},
    {"329b137d4fa3a8bccf73f1add0b9ec7a", "cf9b", NULL},
};

static uint32_t levelAxis(uint32_t axis, uint32_t level) {
    uint32_t a = axis >> level;
    return a ? a : 1;
}

// `generateMipmapsForTexture:` of one case, run once: every level's bytes, in Metal's layout
// (rows tightly packed, images tightly packed). nil where Metal or the probe could not run it.
static NSArray<NSString *> *mipmapOnce(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                       NSUInteger format, const struct MipCase *c,
                                       NSString **problem) {
    BOOL volume = c->depth > 1;
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = volume ? MTLTextureType3D : MTLTextureType2D;
    d.pixelFormat = (MTLPixelFormat)format;
    d.width = c->width;
    d.height = c->height;
    d.depth = c->depth;
    d.mipmapLevelCount = c->levels;
    d.usage = MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:d];
    if (!texture) {
        *problem = @"newTextureWithDescriptor returned nil";
        return nil;
    }
    if ((NSUInteger)texture.pixelFormat != format) {
        *problem = [NSString stringWithFormat:@"texture.pixelFormat is %lu, asked %lu",
                                              (unsigned long)texture.pixelFormat,
                                              (unsigned long)format];
        return nil;
    }
    NSMutableData *input = parseHex(c->input);
    size_t texels = (size_t)c->width * c->height * c->depth;
    if (input.length != 2 * texels) {
        *problem = [NSString stringWithFormat:@"input is %lu bytes, the level is %zu",
                                              (unsigned long)input.length, 2 * texels];
        return nil;
    }
    [texture replaceRegion:MTLRegionMake3D(0, 0, 0, c->width, c->height, c->depth)
               mipmapLevel:0
                     slice:0
                 withBytes:input.bytes
               bytesPerRow:2 * c->width
             bytesPerImage:2 * (size_t)c->width * c->height];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit generateMipmapsForTexture:texture];
    [blit endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        *problem = [NSString stringWithFormat:@"command buffer status %ld: %@",
                                              (long)cb.status, cb.error];
        return nil;
    }
    NSMutableArray<NSString *> *levels = [NSMutableArray array];
    for (uint32_t level = 1; level < c->levels; level++) {
        uint32_t w = levelAxis(c->width, level), h = levelAxis(c->height, level);
        uint32_t z = volume ? levelAxis(c->depth, level) : 1;
        size_t n = 2 * (size_t)w * h * z;
        NSMutableData *out = [NSMutableData dataWithLength:n];
        [texture getBytes:out.mutableBytes
              bytesPerRow:2 * w
            bytesPerImage:2 * (size_t)w * h
               fromRegion:MTLRegionMake3D(0, 0, 0, w, h, z)
              mipmapLevel:level
                    slice:0];
        [levels addObject:printHex(out.bytes, n)];
    }
    return levels;
}

static void runMipmaps(id<MTLDevice> device, id<MTLCommandQueue> queue, const char *label,
                       NSUInteger format, const struct MipCase cases[3],
                       const char *const recorded[3][3]) {
    for (int i = 0; i < 3; i++) {
        const struct MipCase *c = &cases[i];
        NSString *problem = nil;
        NSArray<NSString *> *first = mipmapOnce(device, queue, format, c, &problem);
        if (!first) {
            printf("MIP %s %s REFUSED %s\n", label, c->name, problem.UTF8String);
            continue;
        }
        NSArray<NSString *> *again = mipmapOnce(device, queue, format, c, &problem);
        BOOL same = again && [first isEqualToArray:again];
        printf("MIP %s %s %ux%ux%u twice-identical=%s\n", label, c->name, c->width, c->height,
               c->depth, same ? "yes" : "NO");
        for (NSUInteger level = 0; level < first.count; level++) {
            const char *want = recorded ? recorded[i][level] : NULL;
            const char *got = first[level].UTF8String;
            const char *verdict = want ? (strcmp(want, got) == 0 ? "MATCHES-RECORD" : "DIFFERS")
                                       : "unrecorded";
            printf("MIP %s %s L%lu %s %s\n", label, c->name, (unsigned long)level + 1, got,
                   verdict);
            if (want && strcmp(want, got) != 0) {
                printf("MIP %s %s L%lu recorded %s\n", label, c->name, (unsigned long)level + 1,
                       want);
            }
        }
    }
}

// ---------------------------------------------------------------- part B: the write conversion

static NSString *const kSource =
    @"#include <metal_stdlib>\n"
    @"using namespace metal;\n"
    @"struct Out { float4 position [[position]]; };\n"
    @"vertex Out wc_quad(uint vid [[vertex_id]]) {\n"
    @"  float2 p[6] = { float2(-1,-1), float2(3,-1), float2(-1,3),\n"
    @"                  float2(3,-1), float2(3,3), float2(-1,3) };\n"
    @"  Out o; o.position = float4(p[vid], 0.0, 1.0); return o;\n"
    @"}\n"
    // One float4 per texel out of a buffer, indexed by the pass position, so the written value of
    // a texel is exactly the float the host put there.
    @"fragment float4 wc_float(Out in [[stage_in]], constant float4 *values [[buffer(0)]],\n"
    @"                         constant uint &width [[buffer(1)]]) {\n"
    @"  uint x = uint(floor(in.position.x)), y = uint(floor(in.position.y));\n"
    @"  return values[y * width + x];\n"
    @"}\n"
    @"fragment half4 wc_half(Out in [[stage_in]], constant half4 *values [[buffer(0)]],\n"
    @"                       constant uint &width [[buffer(1)]]) {\n"
    @"  uint x = uint(floor(in.position.x)), y = uint(floor(in.position.y));\n"
    @"  return values[y * width + x];\n"
    @"}\n"
    // Metal's programmable blending: the function reads the attachment it writes. The channels
    // are rotated so a read that happened cannot be mistaken for a write that did not.
    @"fragment float4 fb_rotate(Out in [[stage_in]], float4 prev [[color(0)]]) {\n"
    @"  return float4(prev.g, prev.b, prev.a, prev.r);\n"
    @"}\n";

typedef struct {
    float v[4];
} Float4;

static float halfOf(float x) { return (float)(_Float16)x; }

static float ulpUp(float x, int steps) {
    for (int i = 0; i < steps; i++) x = nextafterf(x, INFINITY);
    return x;
}
static float ulpDown(float x, int steps) {
    for (int i = 0; i < steps; i++) x = nextafterf(x, -INFINITY);
    return x;
}

// The candidate rules a 4-, 5-, 6- or 1-bit unsigned normalized channel could follow.
enum Rule {
    kUnormNearestEven,
    kUnormNearestAway,
    kUnormTowardZero,
    kHalfUnormNearestEven,
    kHalfUnormNearestAway,
    kHalfUnormTowardZero,
    kRuleCount,
};
static const char *const kRuleNames[kRuleCount] = {
    "unorm/nearest-even",      "unorm/nearest-away",      "unorm/toward-zero",
    "half-unorm/nearest-even", "half-unorm/nearest-away", "half-unorm/toward-zero",
};

static uint32_t predict(enum Rule rule, float x, uint32_t bits) {
    double m = (double)((1u << bits) - 1);
    BOOL half = rule >= kHalfUnormNearestEven;
    if (half) x = halfOf(x);
    if (isnan(x)) return 0;
    double v = (double)x;
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    double p = v * m;
    double k;
    switch (rule) {
        case kUnormNearestEven:
        case kHalfUnormNearestEven:
            k = rint(p);  // the default rounding mode: nearest, ties to even
            break;
        case kUnormNearestAway:
        case kHalfUnormNearestAway:
            k = round(p);  // nearest, ties away from zero
            break;
        default:
            k = trunc(p);
            break;
    }
    return (uint32_t)k;
}

// One channel of a measured format: its width and its lowest bit in the little-endian word.
typedef struct {
    const char *name;
    uint32_t shift, bits;
} ChannelGuess;

static void runWrites(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                      const char *label, NSUInteger format, uint32_t texelBytes, BOOL alpha,
                      BOOL blended, BOOL halfOutput) {
    // The values, one per texel, written into every channel of the output at once.
    NSMutableData *scalars = [NSMutableData data];
    void (^add)(float) = ^(float x) { [scalars appendBytes:&x length:sizeof(float)]; };
    // Five texels that are not one scalar in four channels: they separate the channels.
    static const Float4 kHot[5] = {
        {{1, 0, 0, 0}}, {{0, 1, 0, 0}}, {{0, 0, 1, 0}}, {{0, 0, 0, 1}},
        {{1.0f / 15, 2.0f / 15, 4.0f / 15, 8.0f / 15}},
    };
    for (int i = 0; i < 5; i++) add(0);  // placeholders; the hot texels are written below
    // Every half in [0, 1] and every midpoint between two of them.
    for (uint16_t h = 0; h <= 0x3C00; h++) {
        _Float16 f;
        memcpy(&f, &h, sizeof(h));
        add((float)f);
    }
    for (uint16_t h = 0; h < 0x3C00; h++) {
        _Float16 a, b;
        uint16_t next = (uint16_t)(h + 1);
        memcpy(&a, &h, sizeof(h));
        memcpy(&b, &next, sizeof(next));
        add((float)(((double)(float)a + (double)(float)b) / 2.0));
    }
    // Every code's value and every tie of a 1-, 4-, 5- and 6-bit channel, with two float steps
    // either side of each.
    const uint32_t widths[4] = {1, 4, 5, 6};
    for (int w = 0; w < 4; w++) {
        double m = (double)((1u << widths[w]) - 1);
        for (uint32_t k = 0; k <= (uint32_t)m; k++) {
            float exact = (float)((double)k / m);
            add(exact);
            add(ulpDown(exact, 1));
            add(ulpUp(exact, 1));
            add(ulpDown(exact, 2));
            add(ulpUp(exact, 2));
            if (k < (uint32_t)m) {
                float tie = (float)(((double)k + 0.5) / m);
                add(tie);
                add(ulpDown(tie, 1));
                add(ulpUp(tie, 1));
                add(ulpDown(tie, 2));
                add(ulpUp(tie, 2));
            }
        }
    }
    // Outside [0, 1], and the non-finite values.
    static const float kOutside[] = {
        -INFINITY, -1e30f, -2.0f, -1.5f, -1.0f, -0.5f, -1.0f / 15, -0.0f,
        0.0f,      1e-30f, 1.5f,  2.0f,  15.0f, 1e30f, 65536.0f,   INFINITY,
    };
    for (size_t i = 0; i < sizeof(kOutside) / sizeof(*kOutside); i++) add(kOutside[i]);
    for (int i = 0; i < 4; i++) {
        float x;
        uint32_t nans[4] = {0x7fc00000, 0x7fffffff, 0xffc00000, 0x7f800001};
        memcpy(&x, &nans[i], sizeof(x));
        add(x);
    }
    add(ulpDown(1.0f, 1));
    add(ulpUp(1.0f, 1));
    add(ulpUp(0.0f, 1));

    const float *values = scalars.bytes;
    NSUInteger count = scalars.length / sizeof(float);
    NSUInteger width = 256, height = (count + width - 1) / width;
    NSUInteger padded = width * height;

    NSMutableData *buffer = [NSMutableData dataWithLength:padded * (halfOutput ? 8 : 16)];
    if (halfOutput) {
        _Float16 *out = buffer.mutableBytes;
        for (NSUInteger i = 0; i < padded; i++) {
            float x = i < count ? values[i] : 0.0f;
            for (int c = 0; c < 4; c++) out[4 * i + c] = (_Float16)(i < 5 ? kHot[i].v[c] : x);
        }
    } else {
        float *out = buffer.mutableBytes;
        for (NSUInteger i = 0; i < padded; i++) {
            float x = i < count ? values[i] : 0.0f;
            for (int c = 0; c < 4; c++) out[4 * i + c] = i < 5 ? kHot[i].v[c] : x;
        }
    }

    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [library newFunctionWithName:@"wc_quad"];
    rp.fragmentFunction = [library newFunctionWithName:halfOutput ? @"wc_half" : @"wc_float"];
    rp.colorAttachments[0].pixelFormat = (MTLPixelFormat)format;
    if (blended) {
        rp.colorAttachments[0].blendingEnabled = YES;
        rp.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        rp.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        rp.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne;
        rp.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
        rp.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorZero;
        rp.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
    }
    NSError *error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:rp
                                                                                error:&error];
    if (!pipeline) {
        printf("WRITE %s %s%s REFUSED pipeline: %s\n", label, halfOutput ? "half" : "float",
               blended ? "/blend-one-zero" : "", error.localizedDescription.UTF8String);
        return;
    }

    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = (MTLPixelFormat)format;
    td.width = width;
    td.height = height;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    if (!target) {
        printf("WRITE %s REFUSED target: newTextureWithDescriptor returned nil\n", label);
        return;
    }

    id<MTLBuffer> gpuValues = [device newBufferWithBytes:buffer.bytes
                                                 length:buffer.length
                                                options:MTLResourceStorageModeShared];
    uint32_t widthWord = (uint32_t)width;
    id<MTLBuffer> gpuWidth = [device newBufferWithBytes:&widthWord
                                                length:sizeof(widthWord)
                                               options:MTLResourceStorageModeShared];

    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    // A destination the blend must be measured to ignore.
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.6, 0.4, 0.2, 0.8);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
    [enc setRenderPipelineState:pipeline];
    [enc setFragmentBuffer:gpuValues offset:0 atIndex:0];
    [enc setFragmentBuffer:gpuWidth offset:0 atIndex:1];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("WRITE %s REFUSED draw: status %ld %s\n", label, (long)cb.status,
               cb.error.localizedDescription.UTF8String);
        return;
    }

    NSMutableData *read = [NSMutableData dataWithLength:padded * texelBytes];
    [target getBytes:read.mutableBytes
         bytesPerRow:texelBytes * width
          fromRegion:MTLRegionMake2D(0, 0, width, height)
         mipmapLevel:0];
    const uint8_t *bytes = read.bytes;
    uint64_t (^word)(NSUInteger) = ^uint64_t(NSUInteger i) {
        uint64_t w = 0;
        for (uint32_t b = 0; b < texelBytes; b++) w |= (uint64_t)bytes[i * texelBytes + b] << (8 * b);
        return w;
    };

    const char *leg = halfOutput ? "half" : "float";
    const char *mode = blended ? "/blend-one-zero" : "";
    // The channel order, read out of Metal's own words: the single-channel texels.
    printf("WRITE %s %s%s hot r=%04llx g=%04llx b=%04llx a=%04llx graded=%04llx clear=%04llx\n",
           label, leg, mode, (unsigned long long)word(0), (unsigned long long)word(1),
           (unsigned long long)word(2), (unsigned long long)word(3),
           (unsigned long long)word(4), (unsigned long long)word(padded - 1));

    // The channels: each nibble or field of the word, laid out from bit 0. The fields come out of
    // the hot texels, not out of an assumption: the bits a one-channel write set.
    const uint32_t hotWidth = alpha ? 4 : 3;
    uint32_t shift[4] = {0, 0, 0, 0}, bits[4] = {0, 0, 0, 0};
    for (uint32_t c = 0; c < hotWidth; c++) {
        uint64_t w = word(c) ^ (word(8 /* a zero texel: half 0 */) & 0);
        // The set bits of a channel-1 write are that channel's field.
        uint32_t lo = 64, hi = 0;
        for (uint32_t b = 0; b < 8 * texelBytes; b++) {
            if ((w >> b) & 1) {
                if (b < lo) lo = b;
                hi = b;
            }
        }
        shift[c] = lo == 64 ? 0 : lo;
        bits[c] = lo == 64 ? 0 : hi - lo + 1;
        printf("WRITE %s %s%s field %c shift=%u bits=%u\n", label, leg, mode, "rgba"[c], shift[c],
               bits[c]);
    }

    // The rule, per channel: how many of the measured texels each candidate explains.
    for (uint32_t c = 0; c < hotWidth; c++) {
        if (bits[c] == 0) continue;
        uint64_t mask = (1ull << bits[c]) - 1;
        NSUInteger seen[kRuleCount];
        NSUInteger firstBad[kRuleCount];
        for (int r = 0; r < kRuleCount; r++) {
            seen[r] = 0;
            firstBad[r] = NSUIntegerMax;
        }
        uint64_t codes = 0;
        for (NSUInteger i = 5; i < count; i++) {
            float x = halfOutput ? halfOf(values[i]) : values[i];
            uint32_t got = (uint32_t)((word(i) >> shift[c]) & mask);
            codes |= 1ull << (got & 63);
            for (int r = 0; r < kRuleCount; r++) {
                if (predict((enum Rule)r, x, bits[c]) == got) {
                    seen[r]++;
                } else if (firstBad[r] == NSUIntegerMax) {
                    firstBad[r] = i;
                }
            }
        }
        NSUInteger total = count - 5;
        int distinct = 0;
        for (int b = 0; b < 64; b++) distinct += (codes >> b) & 1;
        printf("WRITE %s %s%s channel %c bits=%u texels=%lu distinct-codes=%d\n", label, leg, mode,
               "rgba"[c], bits[c], (unsigned long)total, distinct);
        for (int r = 0; r < kRuleCount; r++) {
            if (seen[r] == total) {
                printf("WRITE %s %s%s channel %c EXPLAINS %s\n", label, leg, mode, "rgba"[c],
                       kRuleNames[r]);
            } else {
                NSUInteger i = firstBad[r];
                float x = values[i];
                printf("WRITE %s %s%s channel %c fails %s at %lu of %lu (first x=%.9g bits=%08x "
                       "got=%u want=%u)\n",
                       label, leg, mode, "rgba"[c], kRuleNames[r], (unsigned long)(total - seen[r]),
                       (unsigned long)total, x, *(const uint32_t *)&x,
                       (uint32_t)((word(i) >> shift[c]) & mask),
                       predict((enum Rule)r, halfOutput ? halfOf(x) : x, bits[c]));
            }
        }
    }
}

// ------------------------------------------- part C: the fetch and the blend of the attachment

// Whether Metal takes a fragment function that reads a color attachment of `format` (its
// programmable blending), and what it stores when it does: a 4x4 texture of known words, loaded,
// drawn over by `fb_rotate`. The runtime refuses such a fetch of an attachment it renders through
// the view of its word, and whether that refusal matches Metal or is a gap of ours decides a whole
// family row on MoltenVK, so it is Metal's answer that settles it.
static void runFetch(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLLibrary> library,
                     const char *label, NSUInteger format, uint32_t texelBytes) {
    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [library newFunctionWithName:@"wc_quad"];
    rp.fragmentFunction = [library newFunctionWithName:@"fb_rotate"];
    rp.colorAttachments[0].pixelFormat = (MTLPixelFormat)format;
    NSError *error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:rp
                                                                                error:&error];
    if (!pipeline) {
        printf("FETCH %s REFUSED-BY-METAL %s\n", label,
               error.localizedDescription.UTF8String ?: "(no description)");
        return;
    }
    MTLTextureDescriptor *td = [MTLTextureDescriptor new];
    td.textureType = MTLTextureType2D;
    td.pixelFormat = (MTLPixelFormat)format;
    td.width = 4;
    td.height = 4;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    if (!target) {
        printf("FETCH %s TAKEN-BY-METAL but no texture\n", label);
        return;
    }
    // Sixteen words that make every nibble distinguishable.
    // Sixteen test words, laid into a buffer this format's own texel wide so the host buffer is
    // exactly the size Metal reads and writes. A texel wider than the word keeps its high bytes 0.
    static const uint16_t kWords[16] = {0x0000, 0xffff, 0x1234, 0x4321, 0x000f, 0x00f0, 0x0f00,
                                        0xf000, 0x8421, 0x1248, 0xabcd, 0xdcba, 0x0f0f, 0xf0f0,
                                        0x5a5a, 0xa5a5};
    NSMutableData *input = [NSMutableData dataWithLength:16 * (size_t)texelBytes];
    uint8_t *bytes = input.mutableBytes;
    for (int i = 0; i < 16; i++) {
        memcpy(bytes + (size_t)i * texelBytes, &kWords[i], sizeof(kWords[i]));
    }
    [target replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
              mipmapLevel:0
                withBytes:input.bytes
              bytesPerRow:4 * (size_t)texelBytes];
    MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
    [enc setRenderPipelineState:pipeline];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        printf("FETCH %s TAKEN-BY-METAL but draw status %ld %s\n", label, (long)cb.status,
               cb.error.localizedDescription.UTF8String);
        return;
    }
    NSMutableData *read = [NSMutableData dataWithLength:16 * (size_t)texelBytes];
    [target getBytes:read.mutableBytes
         bytesPerRow:4 * (size_t)texelBytes
          fromRegion:MTLRegionMake2D(0, 0, 4, 4)
         mipmapLevel:0];
    const uint8_t *got = read.bytes;
    printf("FETCH %s TAKEN-BY-METAL\n", label);
    for (int i = 0; i < 16; i++) {
        uint64_t word = 0;
        for (uint32_t b = 0; b < texelBytes; b++) {
            word |= (uint64_t)got[(size_t)i * texelBytes + b] << (8 * b);
        }
        printf("FETCH %s texel %2d in=%04x out=%0*llx\n", label, i, kWords[i], 2 * texelBytes,
               (unsigned long long)word);
    }
}

// Whether Metal takes a blend of the attachment other than factors one and zero (the runtime
// builds none for an attachment rendered through a word view).
static void runBlend(id<MTLDevice> device, id<MTLLibrary> library, const char *label,
                     NSUInteger format) {
    MTLRenderPipelineDescriptor *rp = [MTLRenderPipelineDescriptor new];
    rp.vertexFunction = [library newFunctionWithName:@"wc_quad"];
    rp.fragmentFunction = [library newFunctionWithName:@"wc_float"];
    rp.colorAttachments[0].pixelFormat = (MTLPixelFormat)format;
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
    printf("BLEND %s source-over %s %s\n", label, pipeline ? "TAKEN-BY-METAL" : "REFUSED-BY-METAL",
           pipeline ? "" : (error.localizedDescription.UTF8String ?: ""));
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

        struct {
            const char *label;
            NSUInteger format;
            uint32_t texelBytes;
            BOOL alpha;
        } writes[] = {
            {"ABGR4Unorm(control)", kABGR4, 2, YES},
            {"B5G6R5Unorm(control)", kB5G6R5, 2, NO},
            {"BGRA4Unorm", kBGRA4, 2, YES},
        };
        for (size_t i = 0; i < sizeof(writes) / sizeof(*writes); i++) {
            for (int halfOut = 0; halfOut < 2; halfOut++) {
                for (int blend = 0; blend < 2; blend++) {
                    runWrites(device, queue, library, writes[i].label, writes[i].format,
                              writes[i].texelBytes, writes[i].alpha, blend, halfOut);
                }
            }
        }

        for (size_t i = 0; i < sizeof(writes) / sizeof(*writes); i++) {
            runFetch(device, queue, library, writes[i].label, writes[i].format, writes[i].texelBytes);
            runBlend(device, library, writes[i].label, writes[i].format);
        }
        // A format every device renders directly, as the control that the fetch leg works at all.
        runFetch(device, queue, library, "RGBA8Unorm(control)", 70, 4);
        runBlend(device, library, "RGBA8Unorm(control)", 70);

        runMipmaps(device, queue, "ABGR4Unorm(control)", kABGR4, kAbgr4Cases, kAbgr4Recorded);
        runMipmaps(device, queue, "B5G6R5Unorm(control)", kB5G6R5, kB5g6r5Cases, kB5g6r5Recorded);
        // `BGRA4Unorm` over `ABGR4Unorm`'s inputs: its four fields are the same four 4-bit
        // unsigned normalized fields in another order, so the probe's generator gives the same
        // level-0 bytes (`ARGB8Unorm` holds `RGBA8Unorm`'s inputs in the table for the same
        // reason).
        runMipmaps(device, queue, "BGRA4Unorm", kBGRA4, kAbgr4Cases, NULL);
    }
    return 0;
}
