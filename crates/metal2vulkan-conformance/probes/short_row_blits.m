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

int main(int argc, const char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s in|out\n", argv[0]);
        return 2;
    }
    gDevice = MTLCreateSystemDefaultDevice();
    gQueue = [gDevice newCommandQueue];
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
