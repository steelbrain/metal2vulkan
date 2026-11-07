// What plain Metal's blits do at a `bytesPerRow` below the region's row: where a copy into a
// texture reads each row, and which row's bytes stand where a copy out of one writes rows that
// overlap.
//
// For a 32 x 8 2D texture (64 x 8 for `R8Unorm`, so every row is 64 bytes or more) of each format
// below, `Private` and `Shared`, every `bytesPerRow` that is a multiple of the texel below the row
// from 16 bytes up in steps of 16, and the row's half plus one texel where that is off 16:
//
//   in   The texture filled from a buffer of bytes unique to their offset at `bytesPerRow`
//        (`copyFromBuffer:...toTexture:`, offset 0, one image), then copied out at its packed row.
//        Printed: whether Metal returned, and how many rows read the source's bytes whole from
//        `y x bytesPerRow` (`whole`), how many read its first `bytesPerRow` bytes from there alone
//        (`head`), and how many read neither.
//   out  The texture filled at its packed row with bytes unique to their offset, then copied out
//        at `bytesPerRow` into a buffer of 0xCC (`copyFromTexture:...toBuffer:`). Printed: for
//        each 16-byte chunk of the rows' span, the row whose bytes stand there (`0`-`7`; `?` for
//        none, `*` for bytes of more than one row), and the bytes past the span written.
//   half For each of the nine one-plane YCbCr formats and `R8Unorm`, `R16Uint` and `RGBA8Unorm`,
//        `Private`, widths of 8 to 100 texels and 2 to 12 rows: a copy out at half the row, as
//        `out`. Printed: whether the buffer is the rows written in order (`later`, each later row
//        over the one before), the odd rows written and then the even ones (`even`), or neither.
//
// The formats: 501 `YCBCR8_422_1P` (2 bytes a texel), 504 `YCBCR10_444_1P` (4), and `R8Unorm`,
// `R16Uint`, `RGBA8Unorm`, `R32Uint`, `RGBA16Float`, `RGBA32Float` for the rule of the copy itself.
//
// Build (no GPU needed to build):
//
//   xcrun clang -fobjc-arc -framework Metal -framework Foundation \
//       -o short_row_blits short_row_blits.m
//
// Run without MTL_DEBUG_LAYER (its validation stops a `bytesPerRow` below the row), `out` in more
// than one process to see whether its answer holds:
//
//   ./short_row_blits in
//   ./short_row_blits out
//   ./short_row_blits half
//
// WHAT IT MEASURED, Apple M3 Max, macOS 27.0.1 (26A434), plain Metal:
//
// - `in`: every copy returned and read every row whole from `y x bytesPerRow`, 166 of 166 (eight
//   formats, `Private` and `Shared`, every pitch asked, multiples of 16 among them).
// - `out`: every copy returned, nothing written past the rows' span; where rows overlap, the
//   bytes that stand follow the format and the pitch (at a 64-byte row and a 32-byte pitch,
//   `R8Unorm`'s later rows over the earlier, `R16Uint`'s and `YCBCR8_422_1P`'s even rows over the
//   odd ones), and 7 of 166 cases answered otherwise in another of three processes.
// - `half`: of 300 shapes of 12 formats at half the row, each in three processes, about a fifth
//   of a format's shapes left the later rows' bytes standing (a width of 48 texels), about a fifth
//   every even row's (a width of 16) and the rest neither order's; 15 shapes answered otherwise
//   in another process. No one order holds a copy out whose rows overlap.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;

struct format {
    MTLPixelFormat pf;
    const char *name;
    NSUInteger unit;
};

static const struct format kFormats[] = {
    {(MTLPixelFormat)501, "YCBCR8_422_1P", 2},
    {(MTLPixelFormat)504, "YCBCR10_444_1P", 4},
    {MTLPixelFormatR8Unorm, "R8Unorm", 1},
    {MTLPixelFormatR16Uint, "R16Uint", 2},
    {MTLPixelFormatRGBA8Unorm, "RGBA8Unorm", 4},
    {MTLPixelFormatR32Uint, "R32Uint", 4},
    {MTLPixelFormatRGBA16Float, "RGBA16Float", 8},
    {MTLPixelFormatRGBA32Float, "RGBA32Float", 16},
};

static const NSUInteger kHeight = 8;

/// A byte unique to `(tag, index)` for any index below 2^24.
static uint8_t patternByte(uint32_t tag, uint32_t index) {
    uint32_t x = index * 2654435761u ^ tag * 40503u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    return (uint8_t)(x ^ (x >> 24));
}

static void waitFor(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        printf("    command buffer error %s\n", cb.error.localizedDescription.UTF8String);
    }
}

static id<MTLBuffer> sharedBuffer(NSUInteger length, uint8_t fill) {
    id<MTLBuffer> b = [gDevice newBufferWithLength:length options:MTLResourceStorageModeShared];
    memset(b.contents, fill, length);
    return b;
}

static NSUInteger widthOf(const struct format *f) {
    return f->unit == 1 ? 64 : 32;
}

static id<MTLTexture> texture(const struct format *f, MTLStorageMode mode) {
    MTLTextureDescriptor *d =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f->pf
                                                           width:widthOf(f)
                                                          height:kHeight
                                                       mipmapped:NO];
    d.storageMode = mode;
    d.usage = MTLTextureUsageShaderRead;
    return [gDevice newTextureWithDescriptor:d];
}

static void copyIn(id<MTLTexture> t, id<MTLBuffer> src, NSUInteger bpr) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src
               sourceOffset:0
          sourceBytesPerRow:bpr
        sourceBytesPerImage:bpr * kHeight
                 sourceSize:MTLSizeMake(t.width, kHeight, 1)
                  toTexture:t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    waitFor(cb);
}

static void copyOut(id<MTLTexture> t, id<MTLBuffer> dst, NSUInteger bpr) {
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromTexture:t
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(t.width, kHeight, 1)
                        toBuffer:dst
               destinationOffset:0
          destinationBytesPerRow:bpr
        destinationBytesPerImage:bpr * kHeight];
    [blit endEncoding];
    waitFor(cb);
}

/// The `bytesPerRow`s asked of a row of `row` bytes of `unit`-byte texels.
static NSUInteger pitches(NSUInteger row, NSUInteger unit, NSUInteger *out) {
    NSUInteger n = 0;
    for (NSUInteger bpr = 16; bpr < row; bpr += 16) {
        if (bpr % unit == 0) {
            out[n++] = bpr;
        }
    }
    NSUInteger half = row / 2 + unit;
    if (half < row && half % 16 != 0) {
        out[n++] = half;
    }
    return n;
}

static void inCase(const struct format *f, MTLStorageMode mode, NSUInteger bpr) {
    NSUInteger row = widthOf(f) * f->unit;
    id<MTLTexture> t = texture(f, mode);
    NSUInteger length = bpr * (kHeight - 1) + row;
    id<MTLBuffer> src = sharedBuffer(length, 0);
    uint8_t *p = src.contents;
    for (NSUInteger i = 0; i < length; i++) {
        p[i] = patternByte(1, (uint32_t)i);
    }
    copyIn(t, src, bpr);
    id<MTLBuffer> out = sharedBuffer(row * kHeight, 0xCC);
    copyOut(t, out, row);
    const uint8_t *q = out.contents;
    NSUInteger whole = 0, head = 0, neither = 0;
    for (NSUInteger y = 0; y < kHeight; y++) {
        if (memcmp(q + y * row, p + y * bpr, row) == 0) {
            whole++;
        } else if (memcmp(q + y * row, p + y * bpr, bpr) == 0) {
            head++;
        } else {
            neither++;
        }
    }
    printf("in %s %s row=%lu bpr=%lu: returned; whole %lu head %lu neither %lu of %lu\n", f->name,
           mode == MTLStorageModeShared ? "shared" : "private", (unsigned long)row,
           (unsigned long)bpr, (unsigned long)whole, (unsigned long)head, (unsigned long)neither,
           (unsigned long)kHeight);
}

static void outCase(const struct format *f, MTLStorageMode mode, NSUInteger bpr) {
    NSUInteger row = widthOf(f) * f->unit;
    id<MTLTexture> t = texture(f, mode);
    id<MTLBuffer> src = sharedBuffer(row * kHeight, 0);
    uint8_t *p = src.contents;
    for (NSUInteger i = 0; i < row * kHeight; i++) {
        p[i] = patternByte(2, (uint32_t)i);
    }
    copyIn(t, src, row);
    NSUInteger span = bpr * (kHeight - 1) + row;
    id<MTLBuffer> dst = sharedBuffer(span + 64, 0xCC);
    copyOut(t, dst, bpr);
    const uint8_t *q = dst.contents;
    printf("out %s %s row=%lu bpr=%lu: returned; chunks ", f->name,
           mode == MTLStorageModeShared ? "shared" : "private", (unsigned long)row,
           (unsigned long)bpr);
    for (NSUInteger c = 0; c < span; c += 16) {
        NSUInteger n = span - c < 16 ? span - c : 16;
        int winner = -1, matches = 0;
        for (NSUInteger y = 0; y < kHeight; y++) {
            if (c < y * bpr || c + n > y * bpr + row) {
                continue;
            }
            if (memcmp(q + c, p + y * row + (c - y * bpr), n) == 0) {
                winner = (int)y;
                matches++;
            }
        }
        putchar(matches == 0 ? '?' : matches > 1 ? '*' : (char)('0' + winner));
    }
    NSUInteger past = 0;
    for (NSUInteger i = span; i < span + 64; i++) {
        past += q[i] != 0xCC;
    }
    printf(" past %lu\n", (unsigned long)past);
}

/// `rows` rows of `row` bytes from `p` written `bpr` apart into `out` (of 0xCC), in the order
/// `order` lists them.
static void writeRows(uint8_t *out, const uint8_t *p, NSUInteger row, NSUInteger bpr,
                      const NSUInteger *order, NSUInteger rows) {
    for (NSUInteger k = 0; k < rows; k++) {
        memcpy(out + order[k] * bpr, p + order[k] * row, row);
    }
}

static void halfCase(MTLPixelFormat pf, const char *name, NSUInteger unit, NSUInteger width,
                     NSUInteger height) {
    NSUInteger row = width * unit, bpr = row / 2;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pf
                                                                                 width:width
                                                                                height:height
                                                                             mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> t = [gDevice newTextureWithDescriptor:d];
    id<MTLBuffer> src = sharedBuffer(row * height, 0);
    uint8_t *p = src.contents;
    for (NSUInteger i = 0; i < row * height; i++) {
        p[i] = patternByte(3, (uint32_t)i);
    }
    id<MTLCommandBuffer> cb = [gQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
    [blit copyFromBuffer:src
               sourceOffset:0
          sourceBytesPerRow:row
        sourceBytesPerImage:row * height
                 sourceSize:MTLSizeMake(width, height, 1)
                  toTexture:t
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [blit endEncoding];
    waitFor(cb);
    NSUInteger span = bpr * (height - 1) + row;
    id<MTLBuffer> dst = sharedBuffer(span + 64, 0xCC);
    cb = [gQueue commandBuffer];
    blit = [cb blitCommandEncoder];
    [blit copyFromTexture:t
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(width, height, 1)
                        toBuffer:dst
               destinationOffset:0
          destinationBytesPerRow:bpr
        destinationBytesPerImage:bpr * height];
    [blit endEncoding];
    waitFor(cb);
    NSUInteger order[16], n = 0;
    uint8_t *later = malloc(span + 64), *even = malloc(span + 64);
    memset(later, 0xCC, span + 64);
    memset(even, 0xCC, span + 64);
    for (NSUInteger y = 0; y < height; y++) {
        order[y] = y;
    }
    writeRows(later, p, row, bpr, order, height);
    for (NSUInteger y = 1; y < height; y += 2) {
        order[n++] = y;
    }
    for (NSUInteger y = 0; y < height; y += 2) {
        order[n++] = y;
    }
    writeRows(even, p, row, bpr, order, height);
    const uint8_t *q = dst.contents;
    bool isLater = memcmp(q, later, span + 64) == 0, isEven = memcmp(q, even, span + 64) == 0;
    printf("half %s %lux%lu row=%lu bpr=%lu: %s\n", name, (unsigned long)width,
           (unsigned long)height, (unsigned long)row, (unsigned long)bpr,
           isLater && isEven ? "both" : isLater ? "later" : isEven ? "even" : "neither");
    free(later);
    free(even);
}

static void halfMode(void) {
    static const struct format kHalf[] = {
        {(MTLPixelFormat)501, "YCBCR8_422_1P", 2},       {(MTLPixelFormat)504, "YCBCR10_444_1P", 4},
        {(MTLPixelFormat)521, "YCBCR8_422_1P_sRGB", 2},  {(MTLPixelFormat)524, "YCBCR10_444_1P_sRGB", 4},
        {(MTLPixelFormat)560, "YCBCRA8_444_1P", 4},      {(MTLPixelFormat)562, "YCBCR8_422_1P_REV", 2},
        {(MTLPixelFormat)563, "YCBCR10_444_1P_PQ", 4},   {(MTLPixelFormat)590, "YCBCR10_444_1P_XR", 4},
        {(MTLPixelFormat)607, "YCBCR10_444_1P_HLG", 4},  {MTLPixelFormatR8Unorm, "R8Unorm", 1},
        {MTLPixelFormatR16Uint, "R16Uint", 2},           {MTLPixelFormatRGBA8Unorm, "RGBA8Unorm", 4},
    };
    static const NSUInteger widths[] = {8, 16, 24, 48, 100};
    static const NSUInteger heights[] = {2, 3, 5, 8, 12};
    for (size_t i = 0; i < sizeof kHalf / sizeof kHalf[0]; i++) {
        for (size_t w = 0; w < sizeof widths / sizeof widths[0]; w++) {
            for (size_t h = 0; h < sizeof heights / sizeof heights[0]; h++) {
                @autoreleasepool {
                    halfCase(kHalf[i].pf, kHalf[i].name, kHalf[i].unit, widths[w], heights[h]);
                }
            }
        }
    }
}

int main(int argc, const char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s in|out|half\n", argv[0]);
        return 2;
    }
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue = [gDevice newCommandQueue];
    if (strcmp(argv[1], "half") == 0) {
        halfMode();
        return 0;
    }
    bool in = strcmp(argv[1], "in") == 0;
    if (!in && strcmp(argv[1], "out") != 0) {
        return 2;
    }
    printf("DEVICE %s / RELEASE %s\n", gDevice.name.UTF8String,
           [NSProcessInfo processInfo].operatingSystemVersionString.UTF8String);
    for (size_t i = 0; i < sizeof kFormats / sizeof kFormats[0]; i++) {
        const struct format *f = &kFormats[i];
        NSUInteger row = widthOf(f) * f->unit;
        NSUInteger bprs[64];
        NSUInteger n = pitches(row, f->unit, bprs);
        for (int m = 0; m < 2; m++) {
            MTLStorageMode mode = m == 0 ? MTLStorageModePrivate : MTLStorageModeShared;
            for (NSUInteger k = 0; k < n; k++) {
                @autoreleasepool {
                    if (in) {
                        inCase(f, mode, bprs[k]);
                    } else {
                        outCase(f, mode, bprs[k]);
                    }
                }
            }
        }
    }
    return 0;
}
